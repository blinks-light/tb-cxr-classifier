# Current production model

> This file is the human-readable model registry entry. `tests/test_model_artifact.py::test_registry_document_is_not_stale`
> fails CI if the name or run id here stops matching `artifacts/model_meta.json`.

| | |
|---|---|
| **Model name** | `improved_v2_finetune` |
| **MLflow run id** | `a7963fd50d9b4310bb321b8f97de89f5` |
| **MLflow experiment** | `TB_CXR_Shenzhen` |
| **Artifact** | `artifacts/model.onnx` (6,089,390 B) |
| **Artifact sha256** | `7585ef3229c86f65…` (full digest in `artifacts/model_meta.json`) |
| **Architecture** | MobileNetV3-small (ImageNet weights, fine-tuned) |
| **Task** | binary: `P(tuberculosis)` from a frontal chest radiograph |
| **Input** | 224×224 greyscale → 3 channels → ImageNet normalisation |
| **Preprocessing vintage** | `v2` (resize → CLAHE 8×8 / 256 bins / clip 4.0 → normalise) |
| **Decision threshold** | **0.21** — chosen on validation for sensitivity ≥ 0.90 |
| **Data version** | `v2`, DVC hash `b2a03c53e7ebf76eff5746ecfde077d4.dir`, data commit `be783cb` |
| **Trained** | 2026-10-18, CPU (16 threads), 607 s, 16 epochs (best epoch 5) |

## Held-out performance (dataset v2 test split, n = 100: 51 TB / 49 normal)

Evidence committed next to this card (`ml/registry/evidence/`):

* `roc_curve.png`, `confusion_matrix.png` — the plots MLflow logged for this run
* `metrics_improved.json`, `metrics_baseline.json` — the full metric dicts
* `test_predictions_improved.csv` — per-image probabilities, so any metric above
  can be recomputed (this is how the §3 table in `README_experiments.md` was built)
* `mlflow_runs.json` — both runs' params/metrics/tags/artifact lists, exported so
  the experiment records survive a fresh clone (the local `mlflow.db` is ignored)

| Metric | Value at the shipped threshold (0.21) | Value at 0.50 (for reference) |
|---|---|---|
| AUROC | **0.9580** | 0.9580 |
| Accuracy | **0.8900** | 0.8800 |
| Sensitivity | **0.9412** | 0.8039 |
| Specificity | **0.8367** | 0.9592 |
| Precision | 0.8571 | 0.9535 |
| F1 | **0.8972** | 0.8723 |
| Confusion matrix | tp 48 / fp 8 / tn 41 / fn 3 | tp 41 / fp 2 / tn 47 / fn 10 |

## Why this model is the registered one

It dominates the baseline (`baseline_v1_linear_probe`,
run `82a12d411313487b9aefba649d538574`) at the screening operating point:
sensitivity 0.941 vs 0.200, specificity 0.837 vs 1.000 at threshold 0.5, and
**0.959 vs 0.780 specificity at equal 0.90 sensitivity** (threshold-independent
comparison). Full reasoning: [`README_experiments.md`](README_experiments.md).

## Promotion history

| Date | Event |
|---|---|
| 2026-10-18 | `baseline_v1_linear_probe` logged (dataset v1) — rejected, threshold 0.5 misses 80 % of cases |
| 2026-10-18 | `improved_v2_finetune` logged (dataset v2), copied into `artifacts/` with `--promote` |
| 2026-10-18 | promoted to the service; `GET /model-info` reports this run id |

## Rollback

The previous artifact is still in MLflow (`runs:/<run_id>/model/model.onnx`).
To roll back:

```powershell
mlflow artifacts download --artifact-uri runs:/82a12d411313487b9aefba649d538574/model/model.onnx --dst-path artifacts/
# then rewrite artifacts/model_meta.json's onnx_sha256 to match, or re-run:
python -m ml.train --config ml/configs/baseline.yaml --promote
```

## Intended use / restrictions

Screening research only. **Not a medical device.** Do not use for triage,
diagnosis or any clinical decision. Trained and validated on a single Chinese
hospital cohort; no external validation.

## Known failure modes to watch

1. Lateral or paediatric films (never seen in training) — reject OOD inputs upstream.
2. Very low-contrast or heavily compressed JPEG exports.
3. Anything that is not a frontal chest radiograph: the model will still return a
   confident probability (which is why the API validates the file, not the anatomy).
