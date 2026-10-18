"""Shared pytest fixtures.

The API tests are deliberately independent of ``data/`` (which is DVC-tracked and
therefore absent in CI): every synthetic image is generated in-memory.  The two
real chest X-rays under ``tests/fixtures`` are committed for the end-to-end
label checks.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def client():
    """FastAPI test client with startup/lifespan executed."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def classifier():
    from app.inference import TBClassifier

    return TBClassifier()


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    return REPO_ROOT / "artifacts"


def pil_image(mode: str = "L", size: tuple[int, int] = (224, 224), seed: int = 0) -> Image.Image:
    """Build a deterministic PIL image in an awkward mode."""
    import numpy as np

    rng = np.random.default_rng(seed)
    width, height = size
    if mode == "L":
        return Image.fromarray(rng.integers(0, 256, size=(height, width), dtype=np.uint8), mode="L")
    if mode == "RGB":
        return Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8), mode="RGB")
    if mode == "RGBA":
        return Image.fromarray(rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8), mode="RGBA")
    if mode == "P":
        # palette indices only; the palette itself does not matter for the test
        return Image.fromarray(rng.integers(0, 16, size=(height, width), dtype=np.uint8), mode="P")
    if mode in {"16", "I;16"}:
        return Image.fromarray(rng.integers(0, 65535, size=(height, width), dtype=np.uint16))
    if mode == "CMYK":
        return Image.new("CMYK", size, (10, 20, 30, 40))
    if mode == "1":
        return Image.fromarray((rng.integers(0, 2, size=(height, width)) * 255).astype(np.uint8)).convert("1")
    raise ValueError(f"unsupported mode {mode}")


def make_image_bytes(
    *,
    size: tuple[int, int] = (224, 224),
    mode: str = "L",
    seed: int = 0,
    fmt: str = "PNG",
) -> bytes:
    """Deterministic synthetic image; content does not matter for contract tests."""
    image = pil_image(mode, size, seed)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def image_bytes():
    return make_image_bytes


@pytest.fixture(scope="session")
def fixture_images() -> dict[str, Path]:
    paths = {"normal": FIXTURES / "normal_sample.png", "tuberculosis": FIXTURES / "tb_sample.png"}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        pytest.skip(f"committed sample images missing: {missing}")
    return paths
