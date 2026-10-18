# TB CXR inference service -- multi-stage build for a slim runtime image.
#
#   builder : installs the (small) runtime dependency set into /install
#   runtime : python:3.12-slim + /usr/local/libs + app/ + artifacts/  (~330 MB)
#
# torch / scikit-learn / mlflow are deliberately absent: training happens
# elsewhere, the image only *serves* the exported ONNX graph.

# ---------- stage 1: dependencies ---------- #
FROM python:3.12-slim AS builder

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --index-url "${PIP_INDEX_URL}" --upgrade pip \
 && /opt/venv/bin/pip install --index-url "${PIP_INDEX_URL}" -r requirements.txt

# ---------- stage 2: runtime ---------- #
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="tb-cxr-classifier" \
      org.opencontainers.image.description="Tuberculosis screening on chest X-rays (MobileNetV3-small, ONNX)" \
      org.opencontainers.image.source="https://github.com/blinks-light/tb-cxr-classifier" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    TBCXR_INTRA_OP_THREADS=2 \
    TBCXR_LOG_LEVEL=INFO

RUN groupadd --gid 1001 app \
 && useradd --uid 1001 --gid app --create-home app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app app ./app
COPY --chown=app:app artifacts ./artifacts

USER app
EXPOSE 8000

# The model is validated at build time, so a container that answers /health with
# "model_loaded": true is known to have an intact graph.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import json,urllib.request,sys; \
d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); \
sys.exit(0 if d.get('model_loaded') else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]
