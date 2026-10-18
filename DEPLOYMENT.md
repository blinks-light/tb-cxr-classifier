# Deployment runbook

This document describes **what gets deployed, when, and how to operate it**.
The course requirement is to *build* staging and production images, not to keep
a live host running — the "run it" section below is therefore optional but real.

---

## 1. Branch → environment mapping

| Branch | Trigger | Workflow | Image published |
|---|---|---|---|
| `feature/*` | push | `ci.yml` | — (build only, gated) |
| `dev` | push / merge | `ci.yml` | — (build only, gated) |
| `staging` | push / merge | `cd-staging.yml` | `ghcr.io/<owner>/<repo>/tb-cxr:staging`, `:staging-<sha>` |
| `main` | push / merge | `cd-production.yml` | `ghcr.io/<owner>/<repo>/tb-cxr:prod`, `:prod-<sha>`, `:latest` |

Promotion path: `feature/*` → PR → `dev` → PR → `staging` → PR → `main`.
A PR into any of those three branches cannot be merged while the **`CI status`**
check is red (configure branch protection — see §2).

---

## 2. Branch protection (GitHub UI, not code)

For `dev`, `staging`, `main`:

1. Settings → Branches → Add branch protection rule (pattern = branch name).
2. ✅ Require a pull request before merging — 1 approval.
3. ✅ Require status checks to pass → add **`CI status`**.
4. ✅ Require branches to be up to date before merging.
5. 🚫 Disallow force pushes, allow no bypass.

For the production gate:

6. Settings → Environments → `production` → Required reviewers = 1 (the person
   who approves the merge cannot also self-approve the deployment if you add two).
7. Settings → Environments → `staging` → no reviewers (fast lane).

---

## 3. Secrets and configuration

CD never bakes secrets into the image; it materialises a `.env` in the *runner*
workspace and uses it as build/runtime configuration.

| Secret / variable | Scope | Meaning |
|---|---|---|
| `STAGING_LOG_LEVEL` | repo secret | `TBCXR_LOG_LEVEL` for staging |
| `STAGING_INTRA_OP_THREADS` | repo secret | onnxruntime thread budget for staging |
| `PRODUCTION_LOG_LEVEL` | repo secret | `TBCXR_LOG_LEVEL` for production |
| `PRODUCTION_INTRA_OP_THREADS` | repo secret | onnxruntime thread budget for production |
| `GITHUB_TOKEN` | automatic | GHCR push (`packages: write`) |

Runtime environment variables understood by the app:

| Variable | Default | Purpose |
|---|---|---|
| `TBCXR_LOG_LEVEL` | `INFO` | stderr log level |
| `TBCXR_INTRA_OP_THREADS` | `min(4, cpu_count)` | onnxruntime threads |
| `TBCXR_ARTIFACTS_DIR` | `/app/artifacts` | where `model.onnx` + `model_meta.json` live |

`.env.example` is the local template; `.env` is git-ignored.

---

## 4. What the image contains

```
/opt/venv          fastapi, uvicorn, python-multipart, onnxruntime, numpy, pillow, pydantic
/app/app           the FastAPI application
/app/artifacts     model.onnx (≈10 MB) + model_meta.json (provenance)
```

No torch, no scikit-learn, no data, no notebooks, no git history
(see `.dockerignore`). CI fails the build if the image exceeds **700 MB**; the
built image is **522 MiB** (128 MB compressed; `docker image inspect --format '{{.Size}}'`
reports 547,609,012 bytes).

Startup fails fast (container unhealthy) when:

* `artifacts/model.onnx` is missing,
* its sha256 differs from `model_meta.json` (artifact tampering / half-copied file).

---

## 5. Run it

```bash
# local build
docker build -t tb-cxr:local .

docker run --rm -p 8000:8000 --env-file .env tb-cxr:local

# published image
docker run --rm -p 8000:8000 \
  -e TBCXR_LOG_LEVEL=INFO -e TBCXR_INTRA_OP_THREADS=2 \
  ghcr.io/<owner>/<repo>/tb-cxr:prod-<sha>
```

Smoke test:

```bash
curl -fsS localhost:8000/health
curl -fsS localhost:8000/model-info | python -m json.tool
curl -fsS -F "file=@tests/fixtures/tb_sample.png;type=image/png" localhost:8000/predict
```

---

## 6. Verifying a deployment

| Check | Command | Expected |
|---|---|---|
| liveness | `curl /health` | `{"status":"ok","model_loaded":true,...}` |
| model identity | `curl /model-info` | `git_commit` = the deployed SHA, `data_version` = `v2` |
| inference | `curl -F file=@tests/fixtures/tb_sample.png /predict` | `label == 1` |
| rejection path | `curl -F file=@README.md;type=image/png /predict` | HTTP 422 |
| digest | `docker run --rm tb-cxr:prod-<sha> python -c "from app.inference import TBClassifier as C; print(C().meta['onnx_sha256'])"` | equals `artifacts/model_meta.json` |

`/model-info` carries the git commit and the MLflow run id, so a running
container can always be traced back to an experiment.

---

## 7. Rollback

Images are immutable and sha-tagged:

```bash
# list what has been shipped
docker pull ghcr.io/<owner>/<repo>/tb-cxr:prod-$(git rev-parse --short HEAD~1)

# or re-deploy the previous known-good SHA
docker run -d --name tb-cxr -p 8000:8000 ghcr.io/<owner>/<repo>/tb-cxr:prod-<previous-sha>
```

Rolling back the *model* is a separate concern: `git revert` the commit that
changed `artifacts/model.onnx` (registry card: `ml/registry/current_model.md`),
or point the service at an older artifact directory via `TBCXR_ARTIFACTS_DIR`.

---

## 8. Local pre-flight before opening a PR

```powershell
pytest -q
black --check . ; ruff check . ; flake8 app ml tests
docker build -t tb-cxr:local . ; docker run --rm -d -p 8001:8000 --name t tb-cxr:local
curl -fsS localhost:8001/health
```

If `pytest` is green and `docker build` succeeds locally, CI will be green too —
the CI workflow runs exactly these commands.
