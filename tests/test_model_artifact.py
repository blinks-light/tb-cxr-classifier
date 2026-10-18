"""Artifact-integrity and quality-gate tests.

Two jobs:

1. **Integrity** -- the ONNX graph in ``artifacts/`` is exactly the one that was
   validated, and it loads without torch.
2. **Quality gate** -- the committed metrics must still clear a floor, so a
   silently degraded model cannot reach ``main``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

QUALITY_FLOOR_AUROC = 0.80
QUALITY_FLOOR_ACCURACY = 0.75


@pytest.fixture(scope="module")
def meta(artifacts_dir) -> dict:
    path = artifacts_dir / "model_meta.json"
    assert (
        path.exists()
    ), "artifacts/model_meta.json missing -- run `python -m ml.train --config ml/configs/improved.yaml --promote`"
    return json.loads(path.read_text(encoding="utf-8"))


def test_onnx_file_exists(artifacts_dir) -> None:
    model = artifacts_dir / "model.onnx"
    assert model.exists()
    assert model.stat().st_size < 40 * 1024 * 1024, "slim-image promise: the graph should stay well under 40 MB"


def test_onnx_digest_matches_metadata(artifacts_dir, meta) -> None:
    digest = hashlib.sha256((artifacts_dir / "model.onnx").read_bytes()).hexdigest()
    assert digest == meta["onnx_sha256"]


def test_metadata_is_complete(meta) -> None:
    for key in (
        "model_name",
        "preprocessing",
        "image_size",
        "threshold",
        "output_is_probability",
        "data_version",
        "git_commit",
        "mlflow_run_id",
        "trained_at",
        "metrics",
    ):
        assert key in meta, f"{key} missing from model_meta.json"
    assert meta["preprocessing"] in {"v1", "v2"}
    assert 0.0 < meta["threshold"] < 1.0
    assert meta["output_is_probability"] is True


def test_graph_loads_without_torch(artifacts_dir) -> None:
    """Dependency guard: the serving stack must not need torch."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(artifacts_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    assert inputs[0].shape == ["batch", 3, 224, 224]
    assert outputs[0].shape in (["batch", 1], ["batch"])


def test_serving_import_graph_is_torch_free() -> None:
    """`import app.main` must not drag torch into the process.

    Checked in a subprocess so the result is unaffected by whatever else the test
    session imported: this is the property that keeps the Docker image slim.
    """
    import subprocess
    import sys

    from app.inference import REPO_ROOT

    code = "import sys, app.main; print(int('torch' in sys.modules))"
    completed = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "0", f"app.main imported torch: {completed.stderr[-400:]}"


def test_graph_outputs_probabilities(artifacts_dir) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(artifacts_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((4, 3, 224, 224)).astype(np.float32)
    probs = np.asarray(session.run(None, {"input": batch})[0]).reshape(-1)
    assert probs.shape == (4,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.isfinite(probs).all()


def test_graph_is_deterministic(artifacts_dir) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(artifacts_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    batch = np.random.default_rng(3).standard_normal((2, 3, 224, 224)).astype(np.float32)
    first = session.run(None, {"input": batch})[0]
    second = session.run(None, {"input": batch})[0]
    assert np.array_equal(first, second)


def test_graph_handles_any_batch_size(artifacts_dir) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(artifacts_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    for batch_size in (1, 3, 8):
        batch = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
        probs = np.asarray(session.run(None, {"input": batch})[0]).reshape(-1)
        assert probs.shape == (batch_size,)


# --------------------------------------------------------------------------- #
# quality gate
# --------------------------------------------------------------------------- #
def test_quality_floor_auroc(meta) -> None:
    auroc = meta["metrics"]["test_auroc"]
    assert auroc >= QUALITY_FLOOR_AUROC, f"test AUROC {auroc:.3f} below the {QUALITY_FLOOR_AUROC} floor"


def test_quality_floor_accuracy(meta) -> None:
    accuracy = meta["metrics"]["test_accuracy"]
    assert accuracy >= QUALITY_FLOOR_ACCURACY, f"test accuracy {accuracy:.3f} below the {QUALITY_FLOOR_ACCURACY} floor"


def test_clinically_sensible_sensitivity(meta) -> None:
    """At the tuned operating point, missing TB cases must stay rare."""
    sensitivity = meta["metrics"]["test_sensitivity"]
    assert sensitivity >= 0.80, f"sensitivity {sensitivity:.3f} is too low for a screening use case"


def test_onnx_matches_torch_export(meta) -> None:
    delta = meta.get("onnx", {}).get("max_abs_delta_vs_torch")
    if delta is None:  # older meta files did not record it
        pytest.skip("max_abs_delta_vs_torch not recorded")
    assert delta < 1e-4, f"ONNX export drifted from torch by {delta}"


# --------------------------------------------------------------------------- #
# registry document consistency
# --------------------------------------------------------------------------- #
def test_registry_document_is_not_stale(meta) -> None:
    from app.inference import REPO_ROOT

    doc = REPO_ROOT / "ml" / "registry" / "current_model.md"
    assert doc.exists(), "ml/registry/current_model.md is required by the assignment"
    text = doc.read_text(encoding="utf-8")
    assert meta["model_name"] in text, "current_model.md does not mention the live model name"
    assert meta["mlflow_run_id"] in text, "current_model.md does not mention the live MLflow run id"
