# TB Chest X-ray Classifier — end-to-end ML / DevOps / MLOps project

Binary tuberculosis screening on the **NLM Shenzhen Hospital chest X-ray set**
(662 frontal radiographs, 336 TB-positive / 326 normal), packaged the way a real
model service is packaged: versioned data (DVC → Azure), tracked experiments
(MLflow), a slim ONNX inference container, a tested FastAPI service, and a
`feature → dev → staging → main` delivery pipeline on GitHub Actions.

> ⚠️ **Research/teaching artifact.** Not a medical device. Never use it to make a
> clinical decision. See [Limitations](#limitations).

---

## 1. What is in the box

| Layer | Deliverable | Where |
|---|---|---|
| **App** | FastAPI service, `POST /predict` + `/predict/batch`, tiny upload UI, 5 endpoints | `app/` |
| **Model** | MobileNetV3-small fine-tuned on the Shenzhen CXR set, exported to ONNX (~10 MB) | `artifacts/`, `ml/train.py` |
| **Tests** | 70+ tests across 4 groups: API contract, prediction boundary cases, preprocessing, artifact integrity + quality gate | `tests/` |
| **DevOps** | Multi-stage slim Dockerfile, `.dockerignore`, CI (lint + test + docker smoke), CD (staging / production) | `Dockerfile`, `.github/workflows/` |
| **MLOps** | 2 DVC data versions on Azure Blob, 2 MLflow experiments with commit/data provenance | `data.dvc`, `ml/`, `mlflow.db` |
| **Docs** | data lineage, experiment rationale, deployment runbook, model registry card | `ml/README_data.md`, `ml/README_experiments.md`, `ml/PROVENANCE.md`, `ml/registry/current_model.md`, `DEPLOYMENT.md` |

---

## 2. Quick start (Windows, this machine)

```powershell
cd E:\dsh\academic\task\progect_ai

# 0) activate the training environment
.\.venv\Scripts\Activate.ps1        # torch / mlflow / sklearn live here

# 1) get the data (3.5 GB from NLM, then build a versioned dataset)
python ml/data_pipeline.py download-raw --write-manifest
python ml/data_pipeline.py build-v2
python ml/data_pipeline.py promote --version v2
dvc pull                            # or: dvc add data/processed && dvc push

# 2) reproduce the two experiments (optional — the artifacts are committed)
python -m ml.train --config ml/configs/baseline.yaml
python -m ml.train --config ml/configs/improved.yaml --promote

# 3) run the service
uvicorn app.main:app --host 127.0.0.1 --port 8000      # http://127.0.0.1:8000
mlflow ui --backend-store-uri sqlite:///E:/dsh/academic/task/progect_ai/mlflow.db

# 4) verify everything the CI verifies
pytest -q
black --check . ; ruff check . ; flake8 app ml tests
docker build -t tb-cxr:local .
```

The committed `artifacts/model.onnx` means steps 1–2 are **not** required to run
the service or the test suite.

---

## 3. Repository layout

```
app/
  main.py              FastAPI app: /, /health, /model-info, /predict, /predict/batch
  inference.py         ONNX session wrapper, artifact integrity check, thresholding
  preprocess.py        THE image transform (shared by training and serving) + numpy CLAHE
  schemas.py           pydantic response models
  static/index.html    single-page upload UI
artifacts/
  model.onnx           exported graph consumed by the service and the Docker image
  model_meta.json      provenance: run id, git sha, data version, threshold, metrics
ml/
  data_pipeline.py     download-raw / build-v1 / build-v2 / promote / audit
  train.py             training + evaluation + ONNX export + MLflow logging
  configs/baseline.yaml
  configs/improved.yaml
  registry/current_model.md
  registry/evidence/   ROC curve, confusion matrix, metrics, per-image predictions, mlflow dump
  README_data.md       data lineage: source, v1→v2 changes, commit mapping
  README_experiments.md experiment rationale: which run ships and why
  PROVENANCE.md        checksums + reproduction commands
tests/
  conftest.py          fixtures (synthetic images, committed real radiographs)
  fixtures/            normal_sample.png, tb_sample.png  (real, labelled)
  test_api.py          routing, schemas, ops endpoints
  test_predict.py      prediction boundary cases (26 cases)
  test_preprocess.py   tensor contract, CLAHE invariants, train/serve parity
  test_model_artifact.py  digest, graph I/O, quality gate, registry freshness
data.dvc               DVC pointer for data/processed (v2, current)
requirements.txt       runtime deps (baked into the image)
requirements-dev.txt   training/lint/test deps (never in the image)
Dockerfile             multi-stage slim build
DEPLOYMENT.md          branch → environment → image mapping, runbook
```

---

## 4. Model

| | |
|---|---|
| Backbone | `mobilenet_v3_small` (ImageNet weights, fine-tuned) |
| Input | 224×224 grayscale → replicated to 3 channels → ImageNet normalisation |
| Preprocessing | `v2`: resize → **CLAHE** (8×8 tiles, numpy implementation) → normalise |
| Output | `P(tuberculosis)`, single sigmoid, exported as a probability |
| Operating point | threshold **0.21**, tuned on the validation split for **sensitivity ≥ 0.90** |
| Serving | ONNX Runtime CPU, no torch in the image |
| Shipped run | MLflow `a7963fd50d9b4310bb321b8f97de89f5` (`improved_v2_finetune`) |

**Held-out results (dataset v2 test split, n = 100: 51 TB / 49 normal)**

| Metric | Improved (shipped) | Baseline (frozen backbone, thr 0.5) |
|---|---|---|
| AUROC | **0.9580** | 0.9348 |
| Accuracy | **0.8900** | 0.6000 |
| Sensitivity | **0.9412** | 0.2000 |
| Specificity | 0.8367 | 1.0000 |
| F1 | **0.8972** | 0.3333 |
| Specificity at equal 0.90 sensitivity | **0.959** | 0.780 |

The baseline is the honest "it looks fine until you check sensitivity" run: AUROC
0.93 but it misses 80 % of TB cases at the default threshold. Full analysis in
[ml/README_experiments.md](ml/README_experiments.md).

### Why sensitivity first

For screening, a false negative (a missed TB case) is far more expensive than a
false positive (an extra follow-up). The service therefore ships a *tuned*
threshold rather than 0.5, and reports both the probability and the threshold so
a caller can re-threshold without redeploying.

### Why ONNX in the image

The runtime dependency set is `fastapi, uvicorn, python-multipart, onnxruntime,
numpy, pillow, pydantic`. Training pulls in torch (~200 MB wheel) and
scikit-learn; the image does not. That is the difference between a 522 MiB image
(128 MB compressed) and a ~1.5 GB one — enforced in CI (`docker` job fails above
700 MB).

---

## 5. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | upload UI (demo) |
| `GET` | `/health` | liveness + `model_loaded` |
| `GET` | `/model-info` | live model name, run id, git sha, data version, threshold, metrics |
| `POST` | `/predict` | one image → label, probability, threshold, latency |
| `POST` | `/predict/batch` | 1–8 images in a single forward pass |

```bash
curl -F "file=@tests/fixtures/tb_sample.png;type=image/png" http://127.0.0.1:8000/predict
```

```json
{
  "filename": "tb_sample.png",
  "label": 1,
  "label_name": "tuberculosis",
  "probability": 0.91,
  "threshold": 0.37,
  "latency_ms": 24.5,
  "model_version": "improved_v2_finetune"
}
```

### Error contract

| Status | When |
|---|---|
| 400 | empty file |
| 413 | payload > 8 MB, or image > 40 Mpx (decompression-bomb guard), or > 8 files |
| 415 | content type is not an image type |
| 422 | payload cannot be decoded (text, truncated PNG, …) |
| 503 | ONNX artifact missing or digest mismatch |

`tests/test_predict.py` pins every row of that table.

---

## 6. DevOps: branch strategy and triggers

```
feature/xxx ──PR──► dev ──PR──► staging ──PR──► main
                     │           │              │
                     │           │              └── CD production : build + push prod image
                     │           └───────────────── CD staging    : build + push staging image
                     └───────────────────────────── CI             : lint + test + docker smoke
```

| Event | Workflow | What runs |
|---|---|---|
| PR → `dev` / `staging` / `main` | `ci.yml` | black + ruff + flake8, pytest, `docker build`, container smoke test (`/health`, `/predict` on a real radiograph, garbage-upload rejection), single aggregate `CI status` check |
| push to `dev` / `feature/**` | `ci.yml` | same gate, no deployment |
| push/merge to `staging` | `cd-staging.yml` | verify secrets → build `.env` from `STAGING_*` secrets → validate artifact digest → build & push `ghcr.io/.../tb-cxr:staging-<sha>` → upload deployment descriptor |
| push/merge to `main` | `cd-production.yml` | same with `PRODUCTION_*` secrets + **quality gate** (AUROC ≥ 0.80, accuracy ≥ 0.75, sensitivity ≥ 0.80) → push `prod-<sha>`, `prod`, `latest` |

Both CD workflows only **build and publish** an image — actually running it on a
host is out of scope, as the assignment specifies. `DEPLOYMENT.md` documents how
to run the published image.

### Publishing this repo to GitHub (one manual step)

The repository is built and verified locally with the full branch history
(`main`, `staging`, `dev`, `feature/*` + tags `v1.0.0`, `v1.0.1`). Pushing it needs
*your* GitHub credentials, so it is the one thing left to do by hand:

```powershell
# 1. create an empty repo (no README, no .gitignore) named tb-cxr-classifier, then:
git remote add origin https://github.com/blinks-light/tb-cxr-classifier.git
git push -u origin main
git push origin staging dev
git push origin --tags

# 2. repository secrets (Settings -> Secrets and variables -> Actions):
#    STAGING_LOG_LEVEL, STAGING_INTRA_OP_THREADS,
#    PRODUCTION_LOG_LEVEL, PRODUCTION_INTRA_OP_THREADS

# 3. branch protection + the `production` environment: see below
```

GitHub needs the proxy for this machine: `git config http.proxy http://127.0.0.1:7892`
(already set in this working copy).

### Branch protection (must be configured in the GitHub UI)

`Settings → Branches → Add branch protection rule`, for `dev`, `staging`, `main`:

1. **Require a pull request before merging** (≥1 approval for `staging`, ≥1 for `main`).
2. **Require status checks to pass before merging** → select **`CI status`**.
3. **Require branches to be up to date before merging**.
4. **Do not allow bypassing the above settings**; disallow force pushes.
5. For `main`: `Environments → production → Required reviewers` (a second gate).

`.github/workflows/ci.yml` intentionally exposes one aggregate job named
`CI status` so the required-check list stays stable when jobs are added.

### Secrets

| Secret | Used by |
|---|---|
| `STAGING_LOG_LEVEL`, `STAGING_INTRA_OP_THREADS` | `cd-staging.yml` |
| `PRODUCTION_LOG_LEVEL`, `PRODUCTION_INTRA_OP_THREADS` | `cd-production.yml` |

`GITHUB_TOKEN` covers the GHCR push. No secret is ever written into a file that
is committed: the generated `.env` is created in the runner workspace only.

---

## 7. MLOps: data and experiments

* **Data (DVC → Azure Blob, container `dvc` in account `wut2026`).** Two versions,
  each pinned to a git commit: [ml/README_data.md](ml/README_data.md).
* **Experiments (MLflow, SQLite backend `mlflow.db`).** Baseline vs improved, each
  recording git commit, DVC hash, hyper-parameters, metrics and the ONNX artifact:
  [ml/README_experiments.md](ml/README_experiments.md).
* **Registry.** The promoted model and the reason it was promoted:
  [ml/registry/current_model.md](ml/registry/current_model.md).
* **Lineage / reproduction.** [ml/PROVENANCE.md](ml/PROVENANCE.md).

`ml/train.py` writes the following into every MLflow run: `git_commit`,
`data_version`, `data_dvc_hash`, `dataset_sha256`, all hyper-parameters, per-epoch
validation metrics, final val/test metrics, and artifacts
(`model.onnx`, `model_meta.json`, `roc_curve.png`, `confusion_matrix.png`,
`test_predictions.csv`, resolved `config.yaml`).

---

## 8. Limitations

* 662 images from a **single hospital** (Shenzhen, China) — no external validation
  set, so the reported numbers say nothing about generalisation to other scanners,
  populations or TB presentations (e.g. paediatric, HIV-coinfected).
* Patient identifiers are not in the public release, so split hygiene is
  *image-level*: near-duplicate removal reduces, but cannot eliminate, the risk of
  the same patient appearing in two splits.
* Labels are per-image (`_0` normal / `_1` TB) from the dataset release; no
  radiologist re-reading was performed.
* The model has no localisation output — it cannot tell a clinician *where* it
  looked, which limits its trustworthiness for real screening.

---

## 9. License / data source

Dataset: **NLM ChinaCXRSet (Shenzhen Hospital)**, publicly available from
<https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/CXR_png/index.html>.
Follow the terms on the NLM page. Code in this repository: MIT.
