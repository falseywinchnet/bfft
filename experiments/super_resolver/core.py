"""Measured Eikonal interpolation core for the super-resolution GUI.

This first scaffold makes no generative claim.  A retained high-resolution
image is reduced to one observation, the observation alone supplies the
Eikonal support, and the retained image is used only after reconstruction for
MSE, SSIM, and fine-band error measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from PIL import Image

from port_needed.eikonal_lanczos import eikonal_lanczos_resize
from port_needed.fast_image_ops import gaussian_filter, sobel


DECIMATION_MODES = (
    "Point decimation",
    "Box prefilter",
    "Lanczos prefilter",
    "Eikonal prefilter",
)
SUPPORT_MODES = ("Local tensor", "Current V3 structural owners")


@dataclass(frozen=True)
class SuperResolutionConfig:
    scale: int = 2
    decimation: str = "Point decimation"
    support: str = "Local tensor"
    anisotropy: float = 0.75
    tensor_sigma: float = 1.0
    clamp_range: bool = True
    maximum_side: int = 256

    def validated(self) -> "SuperResolutionConfig":
        if self.scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        if self.decimation not in DECIMATION_MODES:
            raise ValueError(f"unknown decimation mode {self.decimation!r}")
        if self.support not in SUPPORT_MODES:
            raise ValueError(f"unknown support mode {self.support!r}")
        if self.maximum_side < 32:
            raise ValueError("maximum_side must be at least 32")
        if self.tensor_sigma < 0.0:
            raise ValueError("tensor_sigma must be non-negative")
        if self.anisotropy < 0.0:
            raise ValueError("anisotropy must be non-negative")
        return self


@dataclass(frozen=True)
class PreparedObservation:
    reference: np.ndarray
    observed: np.ndarray
    scale: int
    decimation: str
    forward_anisotropy: float
    forward_tensor_sigma: float
    forward_clamp_range: bool


@dataclass(frozen=True)
class SuperResolutionResult:
    prepared: PreparedObservation
    baseline: np.ndarray
    eikonal: np.ndarray
    labels: np.ndarray
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray]
    metrics: Mapping[str, Mapping[str, float]]
    views: Mapping[str, np.ndarray]
    support_description: str


def as_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError("image must have shape HxW or HxWx3")
    value = value[..., :3]
    if float(np.max(value, initial=0.0)) > 1.5:
        value = value / 255.0
    return np.ascontiguousarray(np.clip(value, 0.0, 1.0))


def _pillow_resize(image: np.ndarray, shape: tuple[int, int], resample: int) -> np.ndarray:
    value = as_rgb(image)
    height, width = map(int, shape)
    output = np.empty((height, width, 3), dtype=np.float64)
    for channel in range(3):
        plane = Image.fromarray(value[..., channel].astype(np.float32), mode="F")
        output[..., channel] = np.asarray(
            plane.resize((width, height), resample=resample), dtype=np.float64
        )
    return np.clip(output, 0.0, 1.0)


def fit_to_side(image: np.ndarray, maximum_side: int) -> np.ndarray:
    value = as_rgb(image)
    height, width = value.shape[:2]
    scale = min(1.0, int(maximum_side) / max(height, width))
    shape = (max(1, round(height * scale)), max(1, round(width * scale)))
    if shape == (height, width):
        return value.copy()
    return _pillow_resize(value, shape, Image.Resampling.LANCZOS)


def _crop_for_scale(image: np.ndarray, scale: int) -> np.ndarray:
    height, width = image.shape[:2]
    cropped_height = height - height % scale
    cropped_width = width - width % scale
    if cropped_height < scale or cropped_width < scale:
        raise ValueError("image is smaller than the requested scale factor")
    return np.ascontiguousarray(image[:cropped_height, :cropped_width])


def decimate(
    image: np.ndarray,
    scale: int,
    mode: str,
    *,
    anisotropy: float = 0.75,
    tensor_sigma: float = 1.0,
    clamp_range: bool = True,
) -> np.ndarray:
    """Apply a selected forward reduction to the compatible source lattice.

    Eikonal prefiltering is deliberately a richer forward model: its tensor is
    measured on the source lattice before reduction.  The resulting pixels
    therefore contain directionally integrated source information rather than
    being information-equivalent to literal point samples.
    """
    value = _crop_for_scale(as_rgb(image), int(scale))
    height, width = value.shape[:2]
    output_shape = (height // scale, width // scale)
    if mode == "Point decimation":
        return np.ascontiguousarray(value[::scale, ::scale])
    if mode == "Box prefilter":
        return value.reshape(
            output_shape[0], scale, output_shape[1], scale, 3
        ).mean(axis=(1, 3))
    if mode == "Lanczos prefilter":
        return _pillow_resize(value, output_shape, Image.Resampling.LANCZOS)
    if mode == "Eikonal prefilter":
        labels, tensor = local_tensor_support(value, tensor_sigma)
        return np.clip(
            eikonal_lanczos_resize(
                value,
                output_shape,
                labels,
                tensor,
                anisotropy=anisotropy,
                clamp_range=clamp_range,
            ),
            0.0,
            1.0,
        )
    raise ValueError(f"unknown decimation mode {mode!r}")


def prepare_observation(
    source: np.ndarray, config: SuperResolutionConfig
) -> PreparedObservation:
    config.validated()
    reference = _crop_for_scale(
        fit_to_side(source, config.maximum_side), config.scale
    )
    observed = decimate(
        reference,
        config.scale,
        config.decimation,
        anisotropy=config.anisotropy,
        tensor_sigma=config.tensor_sigma,
        clamp_range=config.clamp_range,
    )
    return PreparedObservation(
        reference,
        observed,
        config.scale,
        config.decimation,
        config.anisotropy,
        config.tensor_sigma,
        config.clamp_range,
    )


def _smooth_plane(value: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(np.ascontiguousarray(value, dtype=np.float64), sigma)


def local_tensor_support(
    observed: np.ndarray, sigma: float
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build a structure tensor in the observation lattice.

    A single owner intentionally leaves topology unconstrained.  The optional
    V3 mode supplies structural owners when that stronger prior is requested.
    """
    rgb = as_rgb(observed)
    grey = rgb @ np.array((0.2126, 0.7152, 0.0722))
    gx, gy = sobel(grey)
    smoothing = max(float(sigma), 0.0)
    xx = _smooth_plane(gx * gx, smoothing)
    xy = _smooth_plane(gx * gy, smoothing)
    yy = _smooth_plane(gy * gy, smoothing)
    labels = np.zeros(grey.shape, dtype=np.int32)
    return labels, tuple(np.ascontiguousarray(x) for x in (xx, xy, yy))


def v3_support(
    observed: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build structural owners and their measured tensor with Segmenting V3."""
    from experiments.segmenting_v3 import SegmentingV3Config, build_segmenting_v3

    pixels = int(np.prod(observed.shape[:2]))
    result = build_segmenting_v3(
        as_rgb(observed),
        SegmentingV3Config(
            structural_topology="canonical_v2",
            structural_flow_sweeps=1,
            structural_characteristic_passes=1,
            texture_model="parent_ridges",
            texture_cleanup=False,
            texture_graph_phase=False,
            texture_dirichlet_envelope=False,
            joint_leaf_collapse=False,
            texture_safety_cells=max(32768, pixels),
            threads=4,
        ),
    )
    geometry = result.get("texture_geometry") or result["cartoon_geometry"]
    labels = np.ascontiguousarray(result["labels"], dtype=np.int32)
    tensor = tuple(
        np.ascontiguousarray(geometry[name], dtype=np.float64)
        for name in ("boundary_xx", "boundary_xy", "boundary_yy")
    )
    return labels, tensor


def _ssim_rgb(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Mean local RGB SSIM with a Gaussian observation window."""
    x = as_rgb(reference)
    y = as_rgb(estimate)
    if x.shape != y.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(3):
        a = x[..., channel]
        b = y[..., channel]
        mean_a = _smooth_plane(a, 1.5)
        mean_b = _smooth_plane(b, 1.5)
        variance_a = np.maximum(_smooth_plane(a * a, 1.5) - mean_a * mean_a, 0.0)
        variance_b = np.maximum(_smooth_plane(b * b, 1.5) - mean_b * mean_b, 0.0)
        covariance = _smooth_plane(a * b, 1.5) - mean_a * mean_b
        numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)
        denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (
            variance_a + variance_b + c2
        )
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-15))))
    return float(np.mean(scores))


def _high_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    value = as_rgb(image)
    low = np.stack(
        [_smooth_plane(value[..., channel], sigma) for channel in range(3)], axis=2
    )
    return value - low


def _method_metrics(
    reference: np.ndarray, estimate: np.ndarray, fine_reference: np.ndarray
) -> dict[str, float]:
    error = as_rgb(estimate) - as_rgb(reference)
    fine_error = _high_pass(estimate, 1.0) - fine_reference
    return {
        "mse": float(np.mean(error * error)),
        "ssim": _ssim_rgb(reference, estimate),
        "fine_mse": float(np.mean(fine_error * fine_error)),
    }


def _heatmap(field: np.ndarray, scale: float | None = None) -> np.ndarray:
    value = np.asarray(field, dtype=np.float64)
    if value.ndim == 3:
        value = np.mean(np.abs(value), axis=2)
    if scale is None:
        positive = np.abs(value[np.isfinite(value)])
        scale = float(np.quantile(positive, 0.98)) if positive.size else 1.0
    z = np.clip(np.abs(value) / max(float(scale), 1e-12), 0.0, 1.0)
    return np.stack((z, np.sqrt(z) * 0.35, 1.0 - z), axis=2)


def _difference_map(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=np.float64)
    finite = np.abs(value[np.isfinite(value)])
    scale = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    z = np.clip(value / max(scale, 1e-12), -1.0, 1.0)
    neutral = 1.0 - np.abs(z)
    # Teal means the Eikonal result has less fine-band error; magenta means more.
    return np.stack(
        (neutral + np.maximum(-z, 0.0), neutral + np.maximum(z, 0.0), neutral + 0.8 * np.maximum(z, 0.0)),
        axis=2,
    )


def run_eikonal_upscale(
    prepared: PreparedObservation, config: SuperResolutionConfig
) -> SuperResolutionResult:
    config.validated()
    if prepared.scale != config.scale:
        raise ValueError("prepared observation and configuration scales differ")
    reference = as_rgb(prepared.reference)
    observed = as_rgb(prepared.observed)
    baseline = _pillow_resize(observed, reference.shape[:2], Image.Resampling.LANCZOS)
    if config.support == "Current V3 structural owners":
        labels, tensor = v3_support(observed)
        description = "Segmenting V3 owners + measured Eikonal tensor"
    else:
        labels, tensor = local_tensor_support(observed, config.tensor_sigma)
        description = "unpartitioned local Eikonal tensor"
    eikonal = np.clip(
        eikonal_lanczos_resize(
            observed,
            reference.shape[:2],
            labels,
            tensor,
            anisotropy=config.anisotropy,
            clamp_range=config.clamp_range,
        ),
        0.0,
        1.0,
    )
    fine_reference = _high_pass(reference, 1.0)
    fine_baseline_error = np.mean(
        np.abs(fine_reference - _high_pass(baseline, 1.0)), axis=2
    )
    fine_eikonal_error = np.mean(
        np.abs(fine_reference - _high_pass(eikonal, 1.0)), axis=2
    )
    metrics = {
        "Lanczos": _method_metrics(reference, baseline, fine_reference),
        "Eikonal": _method_metrics(reference, eikonal, fine_reference),
    }
    metrics["difference"] = {
        key: metrics["Lanczos"][key] - metrics["Eikonal"][key]
        for key in ("mse", "fine_mse")
    }
    metrics["difference"]["ssim"] = (
        metrics["Eikonal"]["ssim"] - metrics["Lanczos"]["ssim"]
    )
    energy = tensor[0] + tensor[2]
    views = {
        "Reference HR": reference,
        "Observed LR (nearest)": _pillow_resize(
            observed, reference.shape[:2], Image.Resampling.NEAREST
        ),
        "Lanczos baseline": baseline,
        "Eikonal upscale": eikonal,
        "Lanczos absolute error": _heatmap(reference - baseline),
        "Eikonal absolute error": _heatmap(reference - eikonal),
        "Fine error difference": _difference_map(
            fine_baseline_error - fine_eikonal_error
        ),
        "Eikonal support energy": _heatmap(
            _pillow_resize(energy, reference.shape[:2], Image.Resampling.BILINEAR)
        ),
    }
    return SuperResolutionResult(
        prepared, baseline, eikonal, labels, tensor, metrics, views, description
    )


def focus_crop(
    image: np.ndarray, center_x: float, center_y: float, side: int
) -> np.ndarray:
    """Extract a square focus view, keeping the requested size near edges."""
    value = as_rgb(image)
    height, width = value.shape[:2]
    crop_side = min(max(int(side), 8), height, width)
    cx = int(round(np.clip(center_x, 0.0, 1.0) * (width - 1)))
    cy = int(round(np.clip(center_y, 0.0, 1.0) * (height - 1)))
    left = min(max(cx - crop_side // 2, 0), width - crop_side)
    top = min(max(cy - crop_side // 2, 0), height - crop_side)
    return np.ascontiguousarray(value[top : top + crop_side, left : left + crop_side])
