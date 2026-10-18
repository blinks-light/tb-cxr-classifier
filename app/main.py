"""FastAPI application for tuberculosis detection on chest X-rays.

Endpoints
---------
``GET  /``             tiny upload UI (handy for a live demo)
``GET  /health``       liveness + whether the model is loaded
``GET  /model-info``   which model/version/threshold is live and why
``POST /predict``      one image -> P(tuberculosis)
``POST /predict/batch``up to 8 images in a single forward pass

Everything the service needs is in ``artifacts/``; the image never installs
torch.  See ``DEPLOYMENT.md`` for how staging/production get this artifact.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path as _Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.inference import TBClassifier, ModelLoadError
from app.preprocess import MAX_PIXELS, InvalidImageError, preprocess_bytes
from app.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(level=os.environ.get("TBCXR_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tbcxr.api")

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB, comfortably above a 6 MB Shenzhen PNG
MAX_BATCH_FILES = 8
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/tiff",
    "image/webp",
    "application/octet-stream",
}

_STARTED_AT = time.time()
INDEX_HTML = _Path(__file__).with_name("static") / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX model once, at startup."""
    try:
        app.state.classifier = TBClassifier()
        logger.info("model loaded: %s (preprocessing=%s)", app.state.classifier.meta.get("model_name"), app.state.classifier.preprocessing)
    except ModelLoadError as exc:
        # The API still boots so /health can report the failure in a readable way.
        app.state.classifier = None
        app.state.model_error = str(exc)
        logger.error("model failed to load: %s", exc)
    yield
    app.state.classifier = None


app = FastAPI(
    title="TB Chest X-ray Classifier",
    description="Binary tuberculosis detection on the NLM Shenzhen Hospital CXR set (MobileNetV3-small, ONNX).",
    version="1.0.0",
    lifespan=lifespan,
)


def get_classifier(request: Request) -> TBClassifier:
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        detail = getattr(request.app.state, "model_error", "model not loaded")
        raise HTTPException(status_code=503, detail=f"model unavailable: {detail}")
    return classifier


async def _read_upload(upload: UploadFile) -> bytes:
    """Read an upload with a hard byte cap (never trusts Content-Length)."""
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"unsupported content type: {content_type}")

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1 << 16)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file too large: exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _to_http_error(exc: InvalidImageError) -> HTTPException:
    message = str(exc)
    if "too large" in message or f"{MAX_PIXELS}" in message:
        return HTTPException(status_code=413, detail=message)
    if "empty payload" in message:
        return HTTPException(status_code=400, detail=message)
    return HTTPException(status_code=422, detail=message)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> str:
    if INDEX_HTML.exists():
        return INDEX_HTML.read_text(encoding="utf-8")
    return "<h1>TB CXR classifier</h1><p>See <a href='/docs'>/docs</a>.</p>"


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=getattr(request.app.state, "classifier", None) is not None,
        uptime_s=round(time.time() - _STARTED_AT, 3),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info(request: Request) -> ModelInfoResponse:
    return ModelInfoResponse(**get_classifier(request).info())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["inference"],
)
async def predict(request: Request, file: Annotated[UploadFile, File(description="chest X-ray image")]) -> JSONResponse:
    classifier = get_classifier(request)
    blob = await _read_upload(file)
    try:
        tensor = preprocess_bytes(blob, classifier.preprocessing)
    except InvalidImageError as exc:
        raise _to_http_error(exc) from exc

    result = classifier.predict_tensor(tensor)
    payload = PredictionResponse(
        filename=file.filename or "upload",
        model_version=classifier.meta.get("model_name"),
        **result,
    )
    logger.info("predict %s -> %s (p=%.4f)", payload.filename, payload.label_name, payload.probability)
    return JSONResponse(content=payload.model_dump())


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["inference"],
)
async def predict_batch(request: Request, files: Annotated[list[UploadFile], File(description="1-8 chest X-ray images")]) -> JSONResponse:
    classifier = get_classifier(request)
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=413, detail=f"too many files: max {MAX_BATCH_FILES} per request")

    blobs = [await _read_upload(item) for item in files]
    tensors, predictions_meta = [], []
    for item, blob in zip(files, blobs, strict=True):
        try:
            tensors.append(preprocess_bytes(blob, classifier.preprocessing))
        except InvalidImageError as exc:
            raise _to_http_error(exc) from exc
        predictions_meta.append(item.filename or "upload")

    results = classifier.predict_batch(tensors)
    payload = BatchPredictionResponse(
        predictions=[
            PredictionResponse(filename=name, model_version=classifier.meta.get("model_name"), **res)
            for name, res in zip(predictions_meta, results, strict=True)
        ],
        count=len(results),
        model_version=classifier.meta.get("model_name"),
    )
    return JSONResponse(content=payload.model_dump())
