"""Pydantic response models for the TB CXR service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    uptime_s: float = Field(..., ge=0, description="seconds since the process started")


class PredictionResponse(BaseModel):
    filename: str
    label: Literal[0, 1] = Field(..., description="1 = tuberculosis, 0 = normal")
    label_name: Literal["tuberculosis", "normal"]
    probability: float = Field(..., ge=0.0, le=1.0, description="P(tuberculosis)")
    threshold: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., ge=0)
    model_version: str | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int = Field(..., ge=0)
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str | None = None
    preprocessing: str
    image_size: int
    threshold: float
    input_name: str
    output_name: str
    onnx_sha256: str | None = None
    onnx_bytes: int = Field(..., ge=0)
    mlflow_run_id: str | None = None
    git_commit: str | None = None
    data_version: str | None = None
    data_dvc_hash: str | None = None
    trained_at: str | None = None
    metrics: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    detail: str
