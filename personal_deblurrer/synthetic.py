"""Synthetic exposure families grounded in explicit image formation."""

from __future__ import annotations

import numpy as np

from .decomposition import apply_reflect
from .kernels import TransportKernel, apply_circular


def degrade(
    clean: np.ndarray,
    kernel: TransportKernel,
    *,
    gaussian_sigma: float = 0.0,
    poisson_peak: float = 0.0,
    seed: int = 0,
    clip: bool = True,
    boundary: str = "circular",
) -> np.ndarray:
    """Blur in linear intensity, then apply optional shot/read noise."""
    latent = np.asarray(clean, dtype=np.float64)
    if boundary == "circular":
        blurred = apply_circular(latent, kernel)
    elif boundary == "reflect":
        blurred = apply_reflect(latent, kernel)
    else:
        raise ValueError("boundary must be 'circular' or 'reflect'")
    rng = np.random.default_rng(int(seed))
    if float(poisson_peak) > 0.0:
        peak = float(poisson_peak)
        blurred = rng.poisson(np.maximum(blurred, 0.0) * peak) / peak
    if float(gaussian_sigma) > 0.0:
        blurred = blurred + rng.normal(0.0, float(gaussian_sigma), blurred.shape)
    return np.clip(blurred, 0.0, 1.0) if clip else blurred


def random_camera_path(
    *,
    extent: float,
    samples: int = 257,
    seed: int = 0,
) -> np.ndarray:
    """Smooth two-axis camera path with zero translation gauge."""
    rng = np.random.default_rng(int(seed))
    count = max(int(samples), 9)
    acceleration = rng.normal(size=(count, 2))
    # Repeated integration creates a smooth, non-parametric exposure path.
    velocity = np.cumsum(acceleration, axis=0)
    velocity -= np.mean(velocity, axis=0, keepdims=True)
    path = np.cumsum(velocity, axis=0)
    path -= np.mean(path, axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(path, axis=1)))
    if scale > 0.0:
        path *= float(extent) / scale
    return path
