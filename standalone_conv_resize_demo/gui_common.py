"""Small GUI utilities shared by the standalone demos."""

from __future__ import annotations

import numpy as np


def unit_image(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        if info.min < 0:
            array = (array.astype(np.float32)-info.min)/(info.max-info.min)
        else:
            array = array.astype(np.float32)/info.max
    else:
        array = array.astype(np.float32)
        low, high = float(np.nanmin(array)), float(np.nanmax(array))
        if low < 0.0 or high > 1.0:
            array = np.zeros_like(array) if high == low else (array-low)/(high-low)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    return np.ascontiguousarray(array)


def rgba_preview(value: np.ndarray, maximum_side: int = 640) -> tuple[np.ndarray, int, int]:
    """Return a nearest-sampled display texture; no smoothing is introduced."""

    image = np.asarray(value, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image[..., :3]
    h, w = image.shape[:2]
    step = max(1, int(np.ceil(max(h, w)/maximum_side)))
    shown = np.clip(image[::step, ::step], 0.0, 1.0)
    alpha = np.ones(shown.shape[:2]+(1,), dtype=np.float32)
    rgba = np.ascontiguousarray(np.concatenate((shown, alpha), axis=2))
    return rgba.ravel(), rgba.shape[1], rgba.shape[0]


def range_excess(value: np.ndarray, source: np.ndarray) -> float:
    axes = tuple(range(source.ndim-1)) if source.ndim == 3 else tuple(range(source.ndim))
    low = np.min(source, axis=axes)
    high = np.max(source, axis=axes)
    return max(
        float(np.max(low-value, initial=0.0)),
        float(np.max(value-high, initial=0.0)),
    )


def mse(a: np.ndarray, b: np.ndarray) -> float:
    residual = np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)
    return float(np.mean(residual*residual))
