"""Service-level tests: routing, schemas, operational endpoints."""

from __future__ import annotations

import json

import pytest


def test_health_reports_model_loaded(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True, "artifacts/model.onnx must be present (and committed) for the service to run"
    assert body["uptime_s"] >= 0


def test_model_info_is_consistent_with_disk(client, artifacts_dir) -> None:
    import hashlib

    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()

    assert body["preprocessing"] in {"v1", "v2"}
    assert 0.0 < body["threshold"] < 1.0
    assert body["image_size"] == 224
    assert body["onnx_bytes"] > 0

    digest = hashlib.sha256((artifacts_dir / "model.onnx").read_bytes()).hexdigest()
    assert body["onnx_sha256"] == digest


def test_model_info_exposes_provenance(client) -> None:
    body = client.get("/model-info").json()
    for key in ("model_name", "mlflow_run_id", "git_commit", "data_version"):
        assert body.get(key), f"{key} missing from /model-info -- provenance must be visible"
    metrics = body.get("metrics") or {}
    assert "test_auroc" in metrics


def test_openapi_schema(client) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]
    assert "/health" in schema["paths"]


def test_docs_render(client) -> None:
    assert client.get("/docs").status_code == 200


def test_landing_page(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "tuberculosis" in response.text.lower()


def test_unknown_route_is_404(client) -> None:
    assert client.get("/does-not-exist").status_code == 404


def test_model_info_is_json_serialisable(client) -> None:
    # Guards against numpy scalars leaking into the response model.
    json.dumps(client.get("/model-info").json())


@pytest.mark.parametrize("endpoint", ["/health", "/model-info"])
def test_ops_endpoints_are_fast(client, endpoint: str) -> None:
    import time

    started = time.perf_counter()
    client.get(endpoint)
    assert time.perf_counter() - started < 1.0
