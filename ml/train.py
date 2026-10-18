"""Train the TB chest X-ray classifier and log everything to MLflow.

Example
-------
    python -m ml.train --config ml/configs/baseline.yaml --promote
    python -m ml.train --config ml/configs/improved.yaml

The script is the MLOps backbone of the project.  For every run it records

* the git commit (``git rev-parse HEAD``),
* the DVC data version + the actual ``data.dvc`` hash that was trained on,
* every hyper-parameter,
* validation/test metrics (AUROC, accuracy, sensitivity, specificity, F1),
* artifacts: ``model.onnx``, ``model_meta.json``, ROC curve, confusion matrix,
  ``metrics.json`` and a copy of the resolved config.

``--promote`` additionally copies the exported ONNX graph into ``artifacts/``,
which is the directory the FastAPI service and the Docker image consume.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python ml/train.py ...` as well as `-m`
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from app.preprocess import IMAGE_SIZE, preprocess  # noqa: E402

EXPERIMENT_NAME = "TB_CXR_Shenzhen"
PROB_COLUMN = "prob_tuberculosis"


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - not fatal, just record "unknown"
        return "unknown"


def sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_dvc_hash() -> str:
    """The ``md5`` recorded in data.dvc for the promoted dataset, if any."""
    for candidate in (REPO_ROOT / "data.dvc", REPO_ROOT / "data" / "processed.dvc"):
        if candidate.exists():
            import re

            match = re.search(r"md5:\s*([0-9a-f]{32})", candidate.read_text(encoding="utf-8"))
            if match:
                return match.group(1)
    return "untracked"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config.setdefault("seed", 42)
    config.setdefault("image_size", IMAGE_SIZE)
    config.setdefault("num_workers", 0)
    config["_config_path"] = str(path)
    return config


def flatten(prefix: str, obj: object, out: dict[str, object]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = ",".join(str(v) for v in obj)
    elif obj is not None:
        out[prefix] = obj


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class CXRDataset(Dataset):
    """PNG on disk -> normalised (3,224,224) float32 tensor.

    Augmentation happens on the PIL image *before* ``app.preprocess.preprocess``
    so that the exact same transform code path is shared with serving.
    """

    def __init__(self, frame: pd.DataFrame, image_dir: Path, preprocessing: str, augment: bool, seed: int = 42):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.preprocessing = preprocessing
        self.augment = augment
        self.seed = seed
        self._aug = None
        if augment:
            from torchvision import transforms

            self._aug = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.92, 1.08)),
                ]
            )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.image_dir / row["filename"]) as image:
            image.load()
            rgb = image.convert("L")
            if self._aug is not None:
                # Per-worker torch RNG is seeded by DataLoader (base_seed + worker_id)
                # on every epoch, so augmentation is random across epochs but still
                # reproducible for a given --seed.
                rgb = self._aug(rgb)
            tensor = preprocess(rgb, self.preprocessing)
        return torch.from_numpy(tensor), int(row["label"])


def build_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    labels = data_dir / "labels.csv"
    if not labels.exists():
        raise FileNotFoundError(f"{labels} not found -- run `python ml/data_pipeline.py build-v2` and `promote` first")
    frame = pd.read_csv(labels)
    missing = {"filename", "label", "split"} - set(frame.columns)
    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")
    frames = {split: frame[frame["split"] == split].reset_index(drop=True) for split in ("train", "val", "test")}
    for split, part in frames.items():
        if part.empty:
            raise ValueError(f"split {split!r} is empty")
    return frames


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(*, dropout: float, freeze_backbone: bool):
    from torchvision import models

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 1)
    model.classifier[2] = nn.Dropout(p=dropout)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    return model


class ProbabilityModel(nn.Module):
    """Wraps the backbone so the exported graph outputs a probability."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(x))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def tune_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, *, target_sensitivity: float
) -> tuple[float, dict[str, float]]:
    """Pick the most specific threshold that still reaches ``target_sensitivity``.

    Clinically the expensive error is a missed case (false negative), so the
    operating point is chosen on *validation* data by sensitivity first and
    specificity second.
    """
    best: tuple[float, dict[str, float]] | None = None
    for threshold in np.round(np.arange(0.02, 0.99, 0.01), 2):
        metrics = binary_metrics(y_true, y_prob, float(threshold))
        if metrics["sensitivity"] < target_sensitivity:
            continue
        if best is None or metrics["specificity"] > best[1]["specificity"]:
            best = (float(threshold), metrics)
    if best is None:
        # Sensitivity target unreachable: fall back to the threshold with the
        # highest Youden index instead of silently returning 0.5.
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        index = int(np.argmax(tpr - fpr))
        threshold = float(thresholds[index]) if index < len(thresholds) else 0.5
        return threshold, binary_metrics(y_true, y_prob, threshold)
    return best


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        logits = model(images.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probs)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def save_plots(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=130)
    ax.plot(fpr, tpr, lw=2, color="#388bfd", label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#888")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC - test split")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = out_dir / "roc_curve.png"
    fig.savefig(roc_path)
    plt.close(fig)
    paths.append(roc_path)

    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.8, 3.4), dpi=130)
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="#111", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], ["pred normal", "pred TB"])
    ax.set_yticks([0, 1], ["true normal", "true TB"])
    ax.set_title(f"confusion matrix @ {threshold:.2f}")
    fig.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)
    paths.append(cm_path)
    return paths


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    run_dir: Path
    onnx_path: Path
    meta_path: Path
    metrics: dict[str, dict[str, float]]
    threshold: float


def _worker_init(worker_id: int) -> None:
    """Seed python/numpy randomness coherently with torch's per-worker seeding.

    Must be a module-level function: on Windows (spawn start method) the
    DataLoader pickles it into each worker process.
    """
    base = torch.initial_seed() % (2**31)
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)


def train(config: dict, args: argparse.Namespace) -> RunResult:
    seed = int(config["seed"])
    seed_everything(seed)

    # Keep torchvision's ImageNet weight cache inside the repo (git-ignored) so the
    # run does not depend on a writable %USERPROFILE%\.cache.
    os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / "runs" / ".torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mpl"))

    data_dir = (
        (REPO_ROOT / config["data_dir"]).resolve()
        if not Path(config["data_dir"]).is_absolute()
        else Path(config["data_dir"])
    )
    frames = build_frames(data_dir)
    preprocessing = config.get("preprocessing", "v2")
    manifest = {}
    if (data_dir / "manifest.json").exists():
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    data_version = manifest.get("version") or data_dir.name

    datasets = {
        "train": CXRDataset(
            frames["train"], data_dir / "images", preprocessing, augment=bool(config.get("augment", False)), seed=seed
        ),
        "val": CXRDataset(frames["val"], data_dir / "images", preprocessing, augment=False, seed=seed),
        "test": CXRDataset(frames["test"], data_dir / "images", preprocessing, augment=False, seed=seed),
    }
    batch_size = int(config.get("batch_size", 16))

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(config.get("num_workers", 0)),
            drop_last=False,
            worker_init_fn=_worker_init,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
    }

    device = torch.device("cpu")
    freeze = int(config.get("freeze_backbone_epochs", 0))
    model = build_model(
        dropout=float(config.get("dropout", 0.2)),
        freeze_backbone=freeze >= int(config.get("epochs", 1)),
    ).to(device)

    # Class weighting: TB is the positive class and the clinically costly miss.
    if config.get("class_weight") == "balanced":
        train_labels = frames["train"]["label"].to_numpy()
        n_pos = max(int(train_labels.sum()), 1)
        n_neg = max(int((train_labels == 0).sum()), 1)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    epochs = int(config.get("epochs", 8))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = None
    if config.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    run_dir = REPO_ROOT / "runs" / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_auroc = -1.0
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    unfroze = False

    for epoch in range(1, epochs + 1):
        if epoch == freeze + 1 and freeze > 0 and not unfroze:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=lr * 0.2, weight_decay=weight_decay
            )
            if config.get("scheduler") == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - freeze, 1))
            unfroze = True

        model.train()
        running = 0.0
        seen = 0
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(targets)
            seen += len(targets)
        if scheduler is not None:
            scheduler.step()

        y_val, p_val = evaluate(model, loaders["val"], device)
        val_metrics = binary_metrics(y_val, p_val, 0.5)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"[{config['name']}] epoch {epoch:02d}/{epochs} loss={row['train_loss']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} val_acc={val_metrics['accuracy']:.4f}",
            flush=True,
        )
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # Operating point chosen on validation, then frozen for the test split.
    y_val, p_val = evaluate(model, loaders["val"], device)
    target_sens = float(config.get("target_sensitivity", 0.90))
    if config.get("threshold_strategy") == "tuned":
        threshold, val_at_threshold = tune_threshold(y_val, p_val, target_sensitivity=target_sens)
    else:
        threshold = float(config.get("fixed_threshold", 0.5))
        val_at_threshold = binary_metrics(y_val, p_val, threshold)
    val_default = binary_metrics(y_val, p_val, 0.5)

    y_test, p_test = evaluate(model, loaders["test"], device)
    test_default = binary_metrics(y_test, p_test, 0.5)
    test_at_threshold = binary_metrics(y_test, p_test, threshold)

    metrics = {
        "val": {
            **{f"at05_{k}": v for k, v in val_default.items()},
            **{f"at_op_{k}": v for k, v in val_at_threshold.items()},
        },
        "test": {
            **{f"at05_{k}": v for k, v in test_default.items()},
            **{f"at_op_{k}": v for k, v in test_at_threshold.items()},
        },
        "threshold": {"operating_point": threshold, "target_sensitivity": target_sens},
    }

    # ---- export ONNX ------------------------------------------------------
    onnx_path = run_dir / "model.onnx"
    export_wrapper = ProbabilityModel(model).eval()
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    export_kwargs: dict[str, object] = {
        "input_names": ["input"],
        "output_names": ["prob"],
        "dynamic_axes": {"input": {0: "batch"}, "prob": {0: "batch"}},
        "opset_version": 17,
        "do_constant_folding": True,
    }
    # `dynamo=` only exists on newer torch releases; keep the legacy (torchscript)
    # exporter everywhere so the export is byte-stable across versions.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(export_wrapper, (dummy,), str(onnx_path), **export_kwargs)

    # Baseline sanity: the exported graph must agree with torch on the test set.
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = torch.stack([datasets["test"][i][0] for i in range(min(8, len(datasets["test"])))]).numpy()
    ort_pred = session.run(["prob"], {"input": sample})[0].reshape(-1)
    with torch.no_grad():
        torch_pred = export_wrapper(torch.from_numpy(sample)).reshape(-1).numpy()
    max_delta = float(np.max(np.abs(ort_pred - torch_pred)))

    metrics["onnx"] = {"max_abs_delta_vs_torch": max_delta, "opset": 17}

    meta = {
        "model_name": config["name"],
        "architecture": "mobilenet_v3_small",
        "preprocessing": preprocessing,
        "image_size": IMAGE_SIZE,
        "threshold": threshold,
        "threshold_strategy": config.get("threshold_strategy", "fixed"),
        "target_sensitivity": target_sens,
        "output_is_probability": True,
        "onnx": {"max_abs_delta_vs_torch": max_delta, "opset": 17},
        "onnx_sha256": sha256_of(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "data_version": data_version,
        "data_dvc_hash": data_dvc_hash(),
        "data_counts": manifest.get("counts", {}),
        "git_commit": git_commit(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "host": socket.gethostname(),
            "device": str(device),
        },
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "metrics": {
            "test_accuracy": test_at_threshold["accuracy"],
            "test_auroc": test_at_threshold["auroc"],
            "test_sensitivity": test_at_threshold["sensitivity"],
            "test_specificity": test_at_threshold["specificity"],
            "test_f1": test_at_threshold["f1"],
            "val_auroc": val_at_threshold["auroc"],
        },
        "training_history": history,
    }

    # ---- artifact bundle --------------------------------------------------
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifacts_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    for plot in save_plots(y_test, p_test, threshold, artifacts_dir):
        print(f"[{config['name']}] wrote {plot.relative_to(REPO_ROOT)}")

    # per-sample test predictions: the audit trail for the README tables
    pd.DataFrame(
        {
            "filename": frames["test"]["filename"].to_numpy(),
            "label": y_test,
            "prob_tuberculosis": p_test,
            "predicted": (p_test >= threshold).astype(int),
        }
    ).to_csv(artifacts_dir / "test_predictions.csv", index=False)

    meta_path = run_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- MLflow -----------------------------------------------------------
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=config["name"]) as run:
        params: dict[str, object] = {}
        flatten("config", {k: v for k, v in config.items() if not k.startswith("_")}, params)
        params.update(
            {
                "data_version": data_version,
                "data_dvc_hash": meta["data_dvc_hash"],
                "git_commit": meta["git_commit"],
                "n_train": len(frames["train"]),
                "n_val": len(frames["val"]),
                "n_test": len(frames["test"]),
                "pos_weight": float(pos_weight.item()) if pos_weight is not None else 1.0,
                "dataset_sha256": sha256_of(data_dir / "labels.csv"),
            }
        )
        for key, value in params.items():
            mlflow.log_param(key.replace(" ", "_"), value)

        mlflow.set_tags(
            {"data_version": data_version, "stage": "baseline" if "baseline" in config["name"] else "improved"}
        )

        for epoch_row in history:
            mlflow.log_metrics(
                {key: float(value) for key, value in epoch_row.items() if key != "epoch"}, step=int(epoch_row["epoch"])
            )
        for split in ("val", "test"):
            for key, value in metrics[split].items():
                mlflow.log_metric(f"{split}_{key}", float(value))
        mlflow.log_metric("threshold_operating_point", threshold)
        mlflow.log_metric("onnx_max_abs_delta_vs_torch", max_delta)

        mlflow.log_artifacts(str(artifacts_dir), artifact_path="evaluation")
        mlflow.log_artifact(str(onnx_path), artifact_path="model")
        mlflow.log_artifact(str(meta_path), artifact_path="model")

        meta["mlflow_run_id"] = run.info.run_id
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{config['name']}] mlflow run {run.info.run_id} ({tracking_uri})")

    # ---- promote ----------------------------------------------------------
    if args.promote:
        artifacts_repo = REPO_ROOT / "artifacts"
        artifacts_repo.mkdir(exist_ok=True)
        import shutil

        shutil.copy2(onnx_path, artifacts_repo / "model.onnx")
        shutil.copy2(meta_path, artifacts_repo / "model_meta.json")
        print(f"[{config['name']}] promoted -> artifacts/model.onnx (+ model_meta.json)")

    # ---- change tracking --------------------------------------------------
    if os.environ.get("MLFLOW_MODEL_REGISTRY", "0") == "1":
        print("[train] MLFLOW_MODEL_REGISTRY=1 set; registry bookkeeping lives in ml/registry/current_model.md")

    return RunResult(run_dir=run_dir, onnx_path=onnx_path, meta_path=meta_path, metrics=metrics, threshold=threshold)


def build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "ml" / "configs" / "baseline.yaml")
    parser.add_argument(
        "--promote", action="store_true", help="copy the ONNX graph into artifacts/ (used by the service)"
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a config entry, e.g. --set epochs=2"
    )
    return parser.parse_args(argv)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        value: object = raw
        for caster in (int, float):
            try:
                value = caster(raw)
                break
            except ValueError:
                continue
        else:
            if raw.lower() in {"true", "false"}:
                value = raw.lower() == "true"
        config[key.strip()] = value
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    started = time.time()
    result = train(config, args)
    duration = time.time() - started
    print(
        f"\n[{config['name']}] done in {duration:.0f}s | test AUROC={result.metrics['test']['at_op_auroc']:.4f} "
        f"acc={result.metrics['test']['at_op_accuracy']:.4f} "
        f"sens={result.metrics['test']['at_op_sensitivity']:.3f} spec={result.metrics['test']['at_op_specificity']:.3f} "
        f"@ threshold {result.threshold:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
