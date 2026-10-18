# Provenance — how to trace a served prediction back to the data

Everything in the chain below is machine-checkable. Each row can be verified with
the command in the last section.

---

## 1. The chain

```
NLM Shenzhen CXR set (662 PNG, 3.51 GiB)
        │  ml/data_pipeline.py download-raw --write-manifest
        ▼
data/raw_manifest.json          sha256 per file, 662/662 present
        │  build-v2  → promote  →  dvc add / dvc push
        ▼
data.dvc                        md5 b2a03c53e7ebf76eff5746ecfde077d4.dir  (commit be783cb)
        │  git checkout be783cb && dvc checkout
        ▼
data/processed/                 labels.csv sha256 8697e1278cb0abec6e82c7e1110f27db398c4f9cd4ae70d121e2ebc4f1cb0a28
        │  python -m ml.train --config ml/configs/improved.yaml --promote
        ▼
MLflow run a7963fd50d9b4310bb321b8f97de89f5   (experiment TB_CXR_Shenzhen, sqlite mlflow.db)
        │  same run
        ▼
artifacts/model.onnx            sha256 7585ef3229c86f65876c728ef613c2c723266c804b540c25bec89a3377a0d990
artifacts/model_meta.json       6,089,390 B model, threshold 0.21, preprocessing v2
        │  docker build
        ▼
ghcr.io/<owner>/<repo>/tb-cxr:prod-<sha>   (built by .github/workflows/cd-production.yml)
        │  GET /model-info
        ▼
{"model_name": "improved_v2_finetune", "mlflow_run_id": "a7963f…", "git_commit": "…",
 "data_version": "v2", "data_dvc_hash": "b2a03c53…", "threshold": 0.21}
```

A running container therefore reports its own data version, experiment run and
git commit — no spreadsheet required.

## 2. Pointers

| Object | Identifier |
|---|---|
| Raw dataset | 662 files, 3,768,676,455 B, per-file SHA-256 in `data/raw_manifest.json` |
| Raw readme (NLM) | `sha256 7bff523f767aa29826bfe2c2f1f52bbf79f7b1b66bba857bd8725ef31c8d6dbe` |
| Dataset v1 | dvc `0fc29fbfacc122686e9566260246cee0.dir` · commit `9be3a10` · labels `2c44fdb7…` |
| Dataset v2 | dvc `b2a03c53e7ebf76eff5746ecfde077d4.dir` · commit `be783cb` · labels `8697e127…` |
| Baseline run | mlflow `82a12d411313487b9aefba649d538574` · ONNX sha `6bd5e3a2…` |
| Shipped run | mlflow `a7963fd50d9b4310bb321b8f97de89f5` · ONNX sha `7585ef32…` |
| DVC remote | `azure://dvc/tb-cxr`, storage account `wut2026`, SAS in `.dvc/config.local` (never committed) |

## 3. Software versions

Training environment (`requirements-dev.txt`), Python 3.12.10 on Windows 11 x64, CPU only:

| | |
|---|---|
| torch / torchvision | 2.4.0+cpu / 0.19.0 |
| mlflow | 3.17.1 (SQLite backend `mlflow.db`) |
| scikit-learn / scikit-image | 1.9.1 / 0.26.0 |
| numpy / pillow | 1.26.4 / 11.0.0 (pinned to match the runtime image) |
| onnx / onnxruntime | 1.23.2 / 1.20.1 |
| DVC | 3.67.1 (with `[azure]` extras) |

Serving environment (`requirements.txt`, the Docker image): fastapi 0.115.6,
uvicorn 0.34.0, python-multipart 0.0.20, onnxruntime 1.20.1, numpy 1.26.4,
pillow 11.0.0, pydantic 2.10.4. No torch, no scikit-learn.

`model_meta.json → environment` also records the exact `torch` build and the host
that produced the artifact, and `trained_at` is the UTC timestamp of the run
(`2026-10-18T08:13:01+00:00`).

## 4. Verify it yourself

```powershell
# 1. is the committed dataset the one the model was trained on?
git checkout be783cb ; dvc checkout
Select-String -Path data.dvc -Pattern 'md5'          # b2a03c53…7d4.dir
Get-FileHash data/processed/labels.csv -Algorithm SHA256   # 8697e127…0a28

# 2. does the committed artifact match its metadata?
python -c "import hashlib,json,pathlib; m=json.loads(pathlib.Path('artifacts/model_meta.json').read_text()); print(hashlib.sha256(pathlib.Path('artifacts/model.onnx').read_bytes()).hexdigest()==m['onnx_sha256'])"
# -> True

# 3. is the live service serving that artifact?
curl -s localhost:8000/model-info | python -m json.tool | Select-String 'onnx_sha256|mlflow_run_id|git_commit'

# 4. does the registry card agree?  (also enforced by the test suite)
pytest tests/test_model_artifact.py -q

# 5. what did the run actually log?
mlflow runs describe --run-id a7963fd50d9b4310bb321b8f97de89f5
# or: mlflow ui --backend-store-uri sqlite:///E:/dsh/academic/task/progect_ai/mlflow.db
```

## 5. What is *not* tracked, and why

| Not tracked | Reason |
|---|---|
| `data/raw/` (3.5 GiB) | re-downloadable; per-file digests committed instead |
| `data/versions/` (local builds) | intermediate; the version of record is `data/processed` via `data.dvc` |
| `runs/` (per-run ONNX + plots + predictions) | regenerable; MLflow keeps the same artifacts as a run |
| `mlruns/`, `mlflow.db` | local MLflow store; the run ids above identify the records |
| `.dvc/config.local` (Azure SAS) | credential — stored on the machine, excluded by git and Docker |
| `_wheels/` | local scaffolding used while building the project |

The **promoted** artifact is the exception: `artifacts/model.onnx` and
`artifacts/model_meta.json` are committed on purpose, so that a fresh clone, CI and
a `docker build` all work without DVC credentials or a training run.
