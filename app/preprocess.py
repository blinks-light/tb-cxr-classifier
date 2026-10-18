"""Image pre-processing for the TB chest X-ray classifier.

This module is the **single source of truth** for the image transform.  It is
imported by the training script (``ml/train.py``) *and* by the inference
service (``app/inference.py``), which is what keeps training and serving from
drifting apart.  A test (``tests/test_preprocess.py::test_train_serve_parity``)
pins that contract.

Two preprocessing "vintages" exist, mirroring the two MLflow experiments:

``v1`` (baseline)
    greyscale -> resize 224x224 -> scale to [0, 1] -> 3 channels -> ImageNet
    normalisation.

``v2`` (improved)
    the same, but with a self-contained CLAHE (contrast-limited adaptive
    histogram equalisation) step before scaling.  CLAHE is implemented here in
    ~60 lines of numpy so that the serving image does *not* need
    scikit-image/OpenCV (a big win for the slim Docker image).  Because both
    training and serving call this exact function, the difference from
    ``skimage``'s implementation is irrelevant to model quality.

Both vintages return ``float32`` arrays of shape ``(3, 224, 224)`` that are
already normalised -- the ONNX graph does no normalisation of its own.
"""

from __future__ import annotations

import io
from typing import Final

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE: Final[int] = 224
CHANNELS: Final[int] = 3
SUPPORTED_PREPROCESSING: Final[frozenset[str]] = frozenset({"v1", "v2"})

# ImageNet statistics: the backbone is ImageNet-pretrained, so the input
# distribution must match what it was trained on.
_MEAN: Final[np.ndarray] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD: Final[np.ndarray] = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Reject absurdly large uploads by pixel count as well as by byte size; this
# stops decompression-bomb style inputs before PIL allocates the full buffer.
MAX_PIXELS: Final[int] = 40_000_000


class InvalidImageError(ValueError):
    """Raised when the payload cannot be decoded into a usable image."""


# --------------------------------------------------------------------------- #
# CLAHE (self-contained, numpy only)
# --------------------------------------------------------------------------- #
def _clahe_map(tile: np.ndarray, nbins: int, clip_limit: float) -> np.ndarray:
    """Return a normalised CDF (the equalisation lookup table) for one tile."""
    hist, _ = np.histogram(tile, bins=nbins, range=(0.0, 1.0))
    # Contrast limiting: cap every bin, then redistribute the excess evenly.
    # ``clip_limit`` is expressed as a multiple of the flat-histogram height
    # (OpenCV semantics), so 2.0 means "no bin may exceed twice uniform".
    max_count = max(1, int(clip_limit * tile.size / nbins))
    excess = int(np.maximum(hist - max_count, 0).sum())
    hist = np.minimum(hist, max_count) + excess // nbins
    cdf = np.cumsum(hist).astype(np.float32)
    total = float(cdf[-1])
    if total <= 0.0:  # fully empty tile -> identity mapping
        return np.linspace(0.0, 1.0, nbins, dtype=np.float32)
    return cdf / total


def _tile_centres(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


def clahe(
    image: np.ndarray,
    *,
    tiles: tuple[int, int] = (8, 8),
    clip_limit: float = 4.0,
    nbins: int = 256,
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on a float image.

    Parameters
    ----------
    image
        2-D array of floats in ``[0, 1]``.
    tiles
        Number of tiles as ``(rows, cols)``.
    clip_limit
        Per-bin cap as a multiple of the flat-histogram height.
    nbins
        Histogram resolution.

    Notes
    -----
    The defaults were calibrated on this dataset (see ``ml/README_experiments.md``):
    with ``8x8`` tiles / 256 bins / ``clip_limit=4`` the mean *local* contrast
    (per-tile standard deviation) of the Shenzhen films grows by ~1.7x while only
    ~0.1% of pixels saturate, so the equalisation helps without clipping anatomy.
    A tighter clip (2.0, OpenCV's default) barely changes these films because
    28-px tiles produce sparse histograms that the clip flattens back to identity.

    Returns
    -------
    np.ndarray
        New ``float32`` array, same shape, values clipped to ``[0, 1]``.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise InvalidImageError(f"CLAHE expects a 2-D array, got shape {img.shape}")
    if img.size == 0:
        raise InvalidImageError("CLAHE received an empty array")

    img = np.clip(img, 0.0, 1.0)
    h, w = img.shape
    ty, tx = tiles
    # Degenerate geometries fall back to global equalisation on a single tile.
    ty = max(1, min(ty, h // 4 or 1))
    tx = max(1, min(tx, w // 4 or 1))

    ys = np.linspace(0, h, ty + 1).astype(int)
    xs = np.linspace(0, w, tx + 1).astype(int)

    maps = np.empty((ty, tx, nbins), dtype=np.float32)
    for i in range(ty):
        for j in range(tx):
            tile = img[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            maps[i, j] = _clahe_map(tile, nbins, clip_limit)

    bins = np.clip(np.rint(img * (nbins - 1)), 0, nbins - 1).astype(np.intp)

    if ty == 1 and tx == 1:
        return maps[0, 0][bins].astype(np.float32)

    # Bilinear interpolation of the per-tile mappings at tile centres -- this is
    # what removes the visible tile seams.
    cy, cx = _tile_centres(ys), _tile_centres(xs)
    yy, xx = np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32)

    def _axis(coord: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = centres.size
        if n == 1:
            zeros = np.zeros(coord.size, dtype=np.intp)
            return zeros, zeros, np.zeros(coord.size, dtype=np.float32)
        hi = np.clip(np.searchsorted(centres, coord), 1, n - 1)
        lo = hi - 1
        span = centres[hi] - centres[lo]
        span = np.where(span <= 0, 1.0, span)
        weight = np.clip((coord - centres[lo]) / span, 0.0, 1.0)
        return lo, hi, weight.astype(np.float32)

    y0, y1, wy = _axis(yy, cy)
    x0, x1, wx = _axis(xx, cx)

    # Broadcast the 1-D interpolation indices/weights into the image grid.
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
    WY, WX = np.meshgrid(wy, wx, indexing="ij")

    m00 = maps[Y0, X0, bins]
    m01 = maps[Y0, X1, bins]
    m10 = maps[Y1, X0, bins]
    m11 = maps[Y1, X1, bins]

    top = m00 * (1.0 - WX) + m01 * WX
    bottom = m10 * (1.0 - WX) + m11 * WX
    return np.clip(top * (1.0 - WY) + bottom * WY, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def _to_grayscale_224(image: Image.Image) -> np.ndarray:
    """Any PIL mode -> L -> 224x224 -> float32 in [0, 1]."""
    if image.width * image.height > MAX_PIXELS:
        raise InvalidImageError(f"image too large: {image.width}x{image.height} exceeds {MAX_PIXELS} pixels")
    gray = image.convert("L")
    if gray.size != (IMAGE_SIZE, IMAGE_SIZE):
        # LANCZOS on down-scale, BILINEAR-ish on up-scale: LANCZOS is fine for both.
        gray = gray.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.float32) / 255.0


def _to_three_channel_normalised(array: np.ndarray) -> np.ndarray:
    rgb = np.repeat(array[:, :, None], CHANNELS, axis=2)
    normalised = (rgb - _MEAN) / _STD
    # HWC (224,224,3) -> CHW (3,224,224), contiguous float32 for onnxruntime.
    return np.ascontiguousarray(normalised.transpose(2, 0, 1), dtype=np.float32)


def preprocess(image: Image.Image, version: str = "v2") -> np.ndarray:
    """Convert a PIL image into the model input tensor.

    Parameters
    ----------
    image
        Any PIL image (mode is normalised internally).
    version
        ``"v1"`` or ``"v2"`` -- which training vintage this model expects.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(3, 224, 224)``, already normalised.
    """
    if version not in SUPPORTED_PREPROCESSING:
        raise InvalidImageError(
            f"unknown preprocessing version {version!r}; expected one of {sorted(SUPPORTED_PREPROCESSING)}"
        )
    array = _to_grayscale_224(image)
    if version == "v2":
        array = clahe(array)
    return _to_three_channel_normalised(array)


def preprocess_bytes(blob: bytes, version: str = "v2") -> np.ndarray:
    """Decode raw upload bytes and preprocess them.

    Raises
    ------
    InvalidImageError
        For empty payloads, undecodable payloads and oversized images.
    """
    if not blob:
        raise InvalidImageError("empty payload")
    try:
        with Image.open(io.BytesIO(blob)) as image:
            image.load()  # force decode now, so truncated files fail here
            return preprocess(image, version)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"cannot decode image: {exc}") from exc


def batch(tensors: list[np.ndarray]) -> np.ndarray:
    """Stack per-image tensors into a single ``(N, 3, 224, 224)`` batch."""
    if not tensors:
        raise InvalidImageError("cannot build a batch from zero images")
    return np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
