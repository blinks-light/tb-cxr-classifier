"""Preprocessing contract tests.

These are the cheapest tests in the suite and they protect the single most
fragile assumption in the project: **training and serving must apply the exact
same pixels-to-tensor transform**.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app import preprocess as pre


def _gray(seed: int = 0, size: tuple[int, int] = (224, 224)) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    return Image.fromarray(array, mode="L")


# --------------------------------------------------------------------------- #
# tensor contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("version", sorted(pre.SUPPORTED_PREPROCESSING))
def test_output_shape_dtype(version: str) -> None:
    tensor = pre.preprocess(_gray(), version)
    assert tensor.shape == (3, pre.IMAGE_SIZE, pre.IMAGE_SIZE)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


@pytest.mark.parametrize("version", sorted(pre.SUPPORTED_PREPROCESSING))
def test_normalisation_range(version: str) -> None:
    """After ImageNet normalisation values must sit in a sane band, not [0,1]."""
    tensor = pre.preprocess(_gray(), version)
    assert -3.5 < tensor.min() < tensor.max() < 3.5


@pytest.mark.parametrize("version", sorted(pre.SUPPORTED_PREPROCESSING))
def test_deterministic(version: str) -> None:
    image = _gray(seed=7)
    assert np.array_equal(pre.preprocess(image, version), pre.preprocess(image, version))


@pytest.mark.parametrize("version", sorted(pre.SUPPORTED_PREPROCESSING))
def test_three_channels_identical(version: str) -> None:
    """A grayscale input must be replicated, not interpolated.

    The three channels are *not* numerically equal after normalisation (ImageNet
    has a different mean/std per channel), so the check undoes the normalisation
    first and then compares.
    """
    tensor = pre.preprocess(_gray(), version)
    mean, std = pre._MEAN, pre._STD
    raw = tensor * std[:, None, None] + mean[:, None, None]
    # float32 round-trip through (x-mean)/std loses the last bit, hence the tolerance
    assert np.allclose(raw[0], raw[1], rtol=0, atol=1e-5)
    assert np.allclose(raw[1], raw[2], rtol=0, atol=1e-5)


def test_versions_actually_differ() -> None:
    image = _gray(seed=3)
    assert not np.allclose(pre.preprocess(image, "v1"), pre.preprocess(image, "v2"))


def test_constant_image_is_stable() -> None:
    """A blank film must not produce NaN/inf through the CLAHE path."""
    flat = Image.new("L", (224, 224), color=90)
    for version in ("v1", "v2"):
        tensor = pre.preprocess(flat, version)
        assert np.isfinite(tensor).all()
        # per channel: the three channels differ by a constant (ImageNet mean/std),
        # so the *within-channel* spread is what must vanish.
        assert max(float(tensor[c].std()) for c in range(3)) < 1e-4


@pytest.mark.parametrize("size", [(1, 1), (4, 300), (300, 4), (511, 389)])
def test_odd_geometries(size: tuple[int, int]) -> None:
    """1x1, extreme aspect ratios and non-square inputs must all survive."""
    tensor = pre.preprocess(_gray(size=size), "v2")
    assert tensor.shape == (3, 224, 224)
    assert np.isfinite(tensor).all()


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P", "16", "CMYK", "1"])
def test_odd_pil_modes(mode: str) -> None:
    from tests.conftest import pil_image

    tensor = pre.preprocess(pil_image(mode, (64, 64), seed=2), "v2")
    assert tensor.shape == (3, 224, 224)
    assert np.isfinite(tensor).all()


def test_unknown_version_rejected() -> None:
    with pytest.raises(pre.InvalidImageError, match="unknown preprocessing version"):
        pre.preprocess(_gray(), "v9")


def test_empty_payload_rejected() -> None:
    with pytest.raises(pre.InvalidImageError, match="empty payload"):
        pre.preprocess_bytes(b"", "v2")


def test_garbage_payload_rejected() -> None:
    with pytest.raises(pre.InvalidImageError, match="cannot decode image"):
        pre.preprocess_bytes(b"definitely not a png", "v2")


def test_truncated_png_rejected() -> None:
    import io

    buffer = io.BytesIO()
    _gray().save(buffer, format="PNG")
    truncated = buffer.getvalue()[:200]
    with pytest.raises(pre.InvalidImageError):
        pre.preprocess_bytes(truncated, "v2")


def test_oversized_image_rejected() -> None:
    huge = Image.new("L", (7000, 7000))  # 49 Mpx > MAX_PIXELS
    with pytest.raises(pre.InvalidImageError, match="too large"):
        pre.preprocess(huge, "v2")


# --------------------------------------------------------------------------- #
# CLAHE
# --------------------------------------------------------------------------- #
def test_clahe_preserves_shape_and_range() -> None:
    array = np.random.default_rng(0).random((224, 224)).astype(np.float32)
    out = pre.clahe(array)
    assert out.shape == array.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_clahe_increases_local_contrast() -> None:
    """CLAHE must amplify structure in a low-contrast, film-like image."""
    rng = np.random.default_rng(0)
    base = np.tile(np.linspace(0.45, 0.55, 224, dtype=np.float32), (224, 1))
    low_contrast = np.clip(base + rng.normal(0, 0.02, (224, 224)).astype(np.float32), 0.0, 1.0)
    out = pre.clahe(low_contrast)
    ratio = float(out.std()) / max(float(low_contrast.std()), 1e-9)
    assert ratio > 1.4, f"CLAHE barely changed the image (std ratio {ratio:.2f})"
    # ... and it must not do so by crushing the dynamic range.
    saturated = float(((out <= 0.001) | (out >= 0.999)).mean())
    assert saturated < 0.02, f"{saturated:.1%} of pixels saturated"


def test_clahe_constant_image() -> None:
    flat = np.full((224, 224), 0.5, dtype=np.float32)
    out = pre.clahe(flat)
    assert np.isfinite(out).all()
    assert float(out.std()) < 1e-6


def test_clahe_tiny_and_degenerate_inputs() -> None:
    for shape in [(1, 1), (1, 40), (40, 1), (8, 8), (3, 200), (200, 3)]:
        out = pre.clahe(np.zeros(shape, dtype=np.float32))
        assert out.shape == shape
        assert np.isfinite(out).all()


def test_clahe_rejects_bad_input() -> None:
    with pytest.raises(pre.InvalidImageError):
        pre.clahe(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(pre.InvalidImageError):
        pre.clahe(np.zeros((0, 0), dtype=np.float32))


def test_clahe_is_approximately_monotonic() -> None:
    """Within a tile, a brighter pixel must not become darker than a dimmer one."""
    rng = np.random.default_rng(11)
    smooth = rng.random((224, 224)).astype(np.float32)
    smooth = np.clip(smooth * 0.3 + 0.35, 0, 1)
    out = pre.clahe(smooth)
    # Compare only on non-degenerate pairs to avoid ties on flat regions.
    order_in = np.argsort(smooth, axis=None)[:2000]
    assert np.corrcoef(smooth.ravel()[order_in], out.ravel()[order_in])[0, 1] > 0.9


# --------------------------------------------------------------------------- #
# train/serve parity
# --------------------------------------------------------------------------- #
def test_training_uses_the_serving_transform() -> None:
    """`ml.train` must import *this* function, not a re-implementation."""
    import ml.train

    assert ml.train.preprocess is pre.preprocess
    assert ml.train.IMAGE_SIZE == pre.IMAGE_SIZE


def test_batch_builder() -> None:
    tensors = [pre.preprocess(_gray(seed=i), "v2") for i in range(3)]
    stacked = pre.batch(tensors)
    assert stacked.shape == (3, 3, 224, 224)
    assert stacked.dtype == np.float32
    with pytest.raises(pre.InvalidImageError):
        pre.batch([])
