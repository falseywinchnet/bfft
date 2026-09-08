"""Sparse glyph-like geometry from the real promoted speech trace."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class RealGaborConfig:
    wavelengths: tuple[float, ...] = (3.0, 6.0, 12.0)
    orientations: tuple[float, ...] = (
        0.0,
        np.pi / 4.0,
        np.pi / 2.0,
    )
    sigma_ratio: float = 0.55
    bright_percentile: float = 88.0
    erosion_iterations: int = 0
    point_count: int = 64

    def __post_init__(self) -> None:
        if len(self.wavelengths) != 3 or len(self.orientations) != 3:
            raise ValueError("the retained bank is exactly three scales by three orientations")
        if any(value <= 0.0 for value in self.wavelengths):
            raise ValueError("Gabor wavelengths must be positive")
        if self.sigma_ratio <= 0.0:
            raise ValueError("Gabor sigma ratio must be positive")
        if not 0.0 <= self.bright_percentile < 100.0:
            raise ValueError("invalid bright-point percentile")
        if self.erosion_iterations < 0:
            raise ValueError("invalid erosion geometry")
        if self.point_count < 1:
            raise ValueError("invalid sparse-point geometry")


@dataclass(frozen=True)
class SparseRidgeGlyph:
    """A fixed-budget set of native-coordinate ridge witnesses."""

    points: np.ndarray
    strengths: np.ndarray
    scales: np.ndarray
    orientations: np.ndarray
    response_shape: tuple[int, int]
    candidate_threshold: float

    @property
    def count(self) -> int:
        return int(self.points.shape[0])

    def mask(self) -> np.ndarray:
        output = np.zeros(self.response_shape, dtype=bool)
        if self.count:
            coordinates = np.rint(self.points).astype(int)
            output[coordinates[:, 0], coordinates[:, 1]] = True
        return output


def _real_gabor_kernel(
    wavelength: float,
    orientation: float,
    sigma_ratio: float,
) -> np.ndarray:
    sigma = sigma_ratio * wavelength
    radius = max(3, int(round(2.5 * sigma)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    along = xx * np.cos(orientation) + yy * np.sin(orientation)
    across = -xx * np.sin(orientation) + yy * np.cos(orientation)
    envelope = np.exp(
        -(along * along + across * across) / (2.0 * sigma * sigma)
    )
    # ``orientation`` names the ridge tangent.  The real carrier oscillates
    # across the ridge, not along it.
    kernel = envelope * np.cos(2.0 * np.pi * across / wavelength)
    kernel -= np.sum(kernel) / kernel.size
    kernel /= max(float(np.linalg.norm(kernel)), 1e-30)
    return np.asarray(kernel, dtype=np.float64)


@lru_cache(maxsize=8)
def real_gabor_kernels(
    wavelengths: tuple[float, ...] = (3.0, 6.0, 12.0),
    orientations: tuple[float, ...] = (
        0.0,
        np.pi / 4.0,
        np.pi / 2.0,
    ),
    sigma_ratio: float = 0.55,
) -> tuple[np.ndarray, ...]:
    """Return nine zero-DC, unit-energy, entirely real Gabor kernels."""

    return tuple(
        _real_gabor_kernel(wavelength, orientation, sigma_ratio)
        for wavelength in wavelengths
        for orientation in orientations
    )


def real_gabor_responses(
    trace_field: np.ndarray,
    config: RealGaborConfig = RealGaborConfig(),
) -> np.ndarray:
    """Convolve one native real trace with the declared nine real filters."""

    source = np.asarray(trace_field, dtype=np.float64)
    if source.ndim != 2 or not source.size:
        raise ValueError("real Gabor analysis requires a nonempty 2-D trace")
    kernels = real_gabor_kernels(
        config.wavelengths, config.orientations, config.sigma_ratio
    )
    return np.stack([
        fftconvolve(source, kernel, mode="same") for kernel in kernels
    ])


def bright_gabor_support(
    responses: np.ndarray,
    config: RealGaborConfig = RealGaborConfig(),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fuse the nine positive real responses, then threshold bright pixels."""

    values = np.asarray(responses, dtype=np.float64)
    expected = len(config.wavelengths) * len(config.orientations)
    if values.ndim != 3 or values.shape[0] != expected:
        raise ValueError("responses must contain the declared 3x3 filter bank")
    fused = np.max(np.maximum(values, 0.0), axis=0)
    observed = fused[fused > 0.0]
    threshold = (
        float(np.percentile(observed, config.bright_percentile))
        if observed.size else float("inf")
    )
    return fused, fused >= threshold, threshold


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Return the standard erosion/opening skeleton of one binary support."""

    current = np.asarray(mask, dtype=bool)
    structure = ndi.generate_binary_structure(2, 1)
    skeleton = np.zeros(current.shape, dtype=bool)
    while current.any():
        opened = ndi.binary_opening(current, structure=structure)
        skeleton |= current & ~opened
        current = ndi.binary_erosion(current, structure=structure)
    return skeleton


def eroded_gabor_ridges(
    responses: np.ndarray,
    config: RealGaborConfig = RealGaborConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Erode the fused bright Gabor support to its one-pixel skeleton."""

    fused, support, threshold = bright_gabor_support(responses, config)
    structure = ndi.generate_binary_structure(2, 1)
    core = support
    if config.erosion_iterations:
        core = ndi.binary_erosion(
            core,
            structure=structure,
            iterations=config.erosion_iterations,
        )
    skeleton = morphological_skeleton(core)
    ridge_strength = np.where(skeleton, fused, 0.0)
    return support, skeleton, ridge_strength, threshold


def sparse_ridge_glyph(
    responses: np.ndarray,
    config: RealGaborConfig = RealGaborConfig(),
    *,
    frame_slice: slice | None = None,
) -> SparseRidgeGlyph:
    """Convert bright oriented responses into a fixed-budget native point set."""

    _support, skeleton, score, threshold = eroded_gabor_ridges(responses, config)
    winner = np.argmax(np.maximum(responses, 0.0), axis=0)
    channel_scales = np.repeat(
        np.asarray(config.wavelengths, dtype=np.float64),
        len(config.orientations),
    )
    channel_orientations = np.tile(
        np.asarray(config.orientations, dtype=np.float64),
        len(config.wavelengths),
    )
    frame0 = 0 if frame_slice is None or frame_slice.start is None else int(frame_slice.start)
    frame1 = score.shape[1] if frame_slice is None or frame_slice.stop is None else int(frame_slice.stop)
    local = score[:, frame0:frame1]
    candidate_local = np.argwhere(local > 0.0)
    if not candidate_local.size:
        return SparseRidgeGlyph(
            points=np.empty((0, 2), dtype=np.float64),
            strengths=np.empty(0, dtype=np.float64),
            scales=np.empty(0, dtype=np.float64),
            orientations=np.empty(0, dtype=np.float64),
            response_shape=(score.shape[0], max(frame1 - frame0, 0)),
            candidate_threshold=threshold,
        )
    native = candidate_local.copy()
    native[:, 1] += frame0
    candidate_strengths = score[native[:, 0], native[:, 1]]
    selected = np.argsort(candidate_strengths)[::-1][: config.point_count]
    selected_native = native[selected]
    local_points = selected_native.astype(np.float64)
    local_points[:, 1] -= frame0
    return SparseRidgeGlyph(
        points=local_points,
        strengths=candidate_strengths[selected],
        scales=channel_scales[winner[
            selected_native[:, 0], selected_native[:, 1]
        ]],
        orientations=channel_orientations[winner[
            selected_native[:, 0], selected_native[:, 1]
        ]],
        response_shape=(score.shape[0], max(frame1 - frame0, 0)),
        candidate_threshold=threshold,
    )
