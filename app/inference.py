"""ONNX-backed inference wrapper.

The service never imports torch: it loads a pre-exported ``artifacts/model.onnx``
plus ``artifacts/model_meta.json`` and runs it through onnxruntime.  That is what
keeps the Docker image slim (~300 MB instead of ~1.5 GB).

The wrapper is deliberately strict about artifact integrity: if the ONNX file's
sha256 does not match the digest recorded in the metadata, loading fails loudly
instead of silently serving a model nobody validated.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from app.preprocess import IMAGE_SIZE, InvalidImageError, batch

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.environ.get("TBCXR_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "model.onnx"
META_PATH = ARTIFACTS_DIR / "model_meta.json"

DEFAULT_THRESHOLD = 0.5
LABELS = {0: "normal", 1: "tuberculosis"}


class ModelLoadError(RuntimeError):
    """Raised when the model artifacts are missing or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class TBClassifier:
    """Thin, synchronous wrapper around the exported ONNX classifier."""

    def __init__(self, model_path: Path | None = None, meta_path: Path | None = None) -> None:
        self.model_path = Path(model_path or MODEL_PATH)
        self.meta_path = Path(meta_path or META_PATH)

        if not self.model_path.exists():
            raise ModelLoadError(f"model artifact not found: {self.model_path}")
        if not self.meta_path.exists():
            raise ModelLoadError(f"model metadata not found: {self.meta_path}")

        self.meta: dict[str, Any] = json.loads(self.meta_path.read_text(encoding="utf-8"))

        expected = self.meta.get("onnx_sha256")
        if expected:
            actual = _sha256(self.model_path)
            if actual != expected:
                raise ModelLoadError(
                    "artifact integrity check failed: model.onnx sha256 " f"{actual} != metadata {expected}"
                )

        threads = int(os.environ.get("TBCXR_INTRA_OP_THREADS", "0")) or min(4, os.cpu_count() or 1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.preprocessing: str = self.meta.get("preprocessing", "v2")
        self.threshold: float = float(self.meta.get("threshold", DEFAULT_THRESHOLD))
        self.output_is_probability: bool = bool(self.meta.get("output_is_probability", True))

    # -- public API -------------------------------------------------------- #
    def predict_tensor(self, tensor: np.ndarray) -> dict[str, Any]:
        """Run one already-preprocessed tensor through the graph."""
        started = time.perf_counter()
        probabilities = self._forward(batch([tensor]))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._shape_result(probabilities[0], latency_ms)

    def predict_batch(self, tensors: list[np.ndarray]) -> list[dict[str, Any]]:
        """Run a batch of preprocessed tensors (single graph call)."""
        if not tensors:
            raise InvalidImageError("predict_batch needs at least one tensor")
        started = time.perf_counter()
        probabilities = self._forward(batch(tensors))
        latency_ms = (time.perf_counter() - started) * 1000.0
        per_item = latency_ms / len(tensors)
        return [self._shape_result(p, per_item) for p in probabilities]

    # -- internals --------------------------------------------------------- #
    def _forward(self, tensor_batch: np.ndarray) -> np.ndarray:
        raw = self.session.run([self.output_name], {self.input_name: tensor_batch})[0]
        flat = np.asarray(raw, dtype=np.float64).reshape(len(tensor_batch))
        # The graph is exported *with* a sigmoid, but an older artifact that emits
        # logits must still serve correctly.
        return np.clip(flat, 0.0, 1.0) if self.output_is_probability else _sigmoid(flat)

    def _shape_result(self, probability: float, latency_ms: float) -> dict[str, Any]:
        probability = float(probability)
        label = 1 if probability >= self.threshold else 0
        return {
            "label": label,
            "label_name": LABELS[label],
            "probability": round(probability, 6),
            "threshold": self.threshold,
            "latency_ms": round(latency_ms, 3),
        }

    def info(self) -> dict[str, Any]:
        """Metadata for ``GET /model-info`` -- what version is live, and why."""
        payload = {
            "model_name": self.meta.get("model_name"),
            "preprocessing": self.preprocessing,
            "image_size": IMAGE_SIZE,
            "threshold": self.threshold,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "onnx_sha256": self.meta.get("onnx_sha256"),
            "onnx_bytes": self.model_path.stat().st_size,
        }
        for key in ("mlflow_run_id", "git_commit", "data_version", "data_dvc_hash", "trained_at"):
            if key in self.meta:
                payload[key] = self.meta[key]
        if "metrics" in self.meta:
            payload["metrics"] = self.meta["metrics"]
        return payload
