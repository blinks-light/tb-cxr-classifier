"""Inference endpoint tests, with an emphasis on boundary cases.

A chest X-ray service is fed whatever the PACS/drop-box happens to contain, so
the contract has to survive empty files, text files, 1x1 thumbnails, 16-bit
DICOM-adjacent exports, palette PNGs and absurd uploads.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image


def _png(**kwargs) -> bytes:
    from tests.conftest import make_image_bytes

    return make_image_bytes(**kwargs)


def _post(client, blob: bytes, *, filename: str = "fx.png", content_type: str = "image/png"):
    return client.post("/predict", files={"file": (filename, blob, content_type)})


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_predict_valid_image(client, image_bytes) -> None:
    response = _post(client, image_bytes())
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in (0, 1)
    assert body["label_name"] in {"normal", "tuberculosis"}
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 < body["threshold"] < 1.0
    assert body["latency_ms"] >= 0
    assert body["filename"] == "fx.png"
    assert body["model_version"]


def test_label_matches_threshold(client, image_bytes) -> None:
    body = _post(client, image_bytes(seed=5)).json()
    expected = 1 if body["probability"] >= body["threshold"] else 0
    assert body["label"] == expected


def test_decision_is_deterministic(client, image_bytes) -> None:
    blob = image_bytes(seed=9)
    first, second = _post(client, blob).json(), _post(client, blob).json()
    assert first["probability"] == pytest.approx(second["probability"], abs=1e-6)


@pytest.mark.parametrize(("name", "expected_label"), [("normal", 0), ("tuberculosis", 1)])
def test_known_radiographs(client, fixture_images, name: str, expected_label: int) -> None:
    """End-to-end label check on two real, committed chest X-rays."""
    blob = fixture_images[name].read_bytes()
    body = _post(client, blob, filename=f"{name}.png").json()
    assert body["label"] == expected_label, (
        f"model disagrees with the committed {name} sample: p={body['probability']:.4f} "
        f"threshold={body['threshold']}"
    )


# --------------------------------------------------------------------------- #
# odd but valid inputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": (1, 1)},
        {"size": (4, 2000)},
        {"size": (2000, 4)},
        {"mode": "RGB", "size": (300, 200)},
        {"mode": "RGBA"},
        {"mode": "P"},
        {"mode": "16"},
        {"mode": "CMYK", "fmt": "JPEG"},
    ],
)
def test_unusual_but_valid_images(client, kwargs) -> None:
    response = _post(client, _png(**kwargs))
    assert response.status_code == 200, response.text
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_jpeg_upload(client) -> None:
    response = _post(client, _png(mode="RGB", fmt="JPEG"), filename="fx.jpg", content_type="image/jpeg")
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# rejected inputs
# --------------------------------------------------------------------------- #
def test_empty_file_is_400(client) -> None:
    response = _post(client, b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_text_file_is_422(client) -> None:
    response = _post(client, b"this is not an image", content_type="image/png")
    assert response.status_code == 422


def test_truncated_png_is_422(client, image_bytes) -> None:
    response = _post(client, image_bytes()[:150])
    assert response.status_code == 422


def test_wrong_content_type_is_415(client, image_bytes) -> None:
    response = _post(client, image_bytes(), content_type="text/plain")
    assert response.status_code == 415


def test_oversized_payload_is_413(client) -> None:
    # 3500x3000 noise PNG is ~10 MB, above the 8 MB cap but below the pixel cap.
    blob = _png(size=(3500, 3000))
    assert len(blob) > 8 * 1024 * 1024
    response = _post(client, blob)
    assert response.status_code == 413


def test_decompression_bomb_is_413(client) -> None:
    # Compresses to a few kB but declares 48 Mpx, above MAX_PIXELS.
    buffer = io.BytesIO()
    Image.new("L", (8000, 6000)).save(buffer, format="PNG")
    response = _post(client, buffer.getvalue())
    assert response.status_code == 413


def test_missing_file_field_is_422(client) -> None:
    assert client.post("/predict").status_code == 422


# --------------------------------------------------------------------------- #
# batch endpoint
# --------------------------------------------------------------------------- #
def test_batch_of_two(client, image_bytes) -> None:
    files = [
        ("files", ("a.png", image_bytes(seed=1), "image/png")),
        ("files", ("b.png", image_bytes(seed=2), "image/png")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [p["filename"] for p in body["predictions"]] == ["a.png", "b.png"]
    assert all(0.0 <= p["probability"] <= 1.0 for p in body["predictions"])


def test_batch_single_matches_single_endpoint(client, image_bytes) -> None:
    """Batching must not change the numbers."""
    blob = image_bytes(seed=4)
    single = _post(client, blob).json()
    batched = client.post("/predict/batch", files=[("files", ("a.png", blob, "image/png"))]).json()
    assert batched["predictions"][0]["probability"] == pytest.approx(single["probability"], abs=1e-5)


def test_batch_too_many_files_is_413(client, image_bytes) -> None:
    files = [("files", (f"{i}.png", image_bytes(seed=i), "image/png")) for i in range(9)]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 413


def test_batch_with_one_bad_file_is_422(client, image_bytes) -> None:
    files = [
        ("files", ("ok.png", image_bytes(), "image/png")),
        ("files", ("bad.png", b"garbage", "image/png")),
    ]
    assert client.post("/predict/batch", files=files).status_code == 422


def test_batch_empty_list_is_422(client) -> None:
    assert client.post("/predict/batch", files=[]).status_code == 422
