"""Deterministic Haar wavelet-leader evidence on V3 regions.

This adapts the non-iterative observable from León-López et al., *Bayesian
Multifractal Image Segmentation* (arXiv:2501.08694), while deliberately not
adopting its Potts labels, class count, learned granularity parameters, patch
k-means initialization, or Gibbs sampler.

For every scalar field, orthonormal Haar details are computed at every dyadic
scale.  A leader is the supremum of normalized detail magnitude over the 3x3
spatial neighborhood and all finer scales.  Centered log-leader means and RMS
values are retained at every scale for every V3 region.  Nothing is classified
or thresholded.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def _next_power_of_two(value: int) -> int:
    return 1 << max(int(value - 1).bit_length(), 1)


def _pad_square(
    field: np.ndarray,
    analysis_side: int | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    value = np.asarray(field, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("wavelet-leader field must be one finite image")
    shape = value.shape
    requested = max(shape) if analysis_side is None else int(analysis_side)
    if requested < max(shape):
        raise ValueError("analysis side cannot be smaller than the field")
    side = _next_power_of_two(requested)
    pad = ((0, side - shape[0]), (0, side - shape[1]))
    # Reflection does not invent a constant exterior or a preferred value.
    padded = np.pad(value, pad, mode="reflect") if any(
        amount for pair in pad for amount in pair) else value.copy()
    return padded, shape


def haar_wavelet_leaders(
    field: np.ndarray,
    analysis_side: int | None = None,
) -> list[np.ndarray]:
    """Return every dyadic leader field, lifted to the original pixel grid."""
    approximation, original_shape = _pad_square(field, analysis_side)
    details: list[np.ndarray] = []
    scale = 1
    while min(approximation.shape) >= 2:
        first = approximation[0::2, 0::2]
        second = approximation[0::2, 1::2]
        third = approximation[1::2, 0::2]
        fourth = approximation[1::2, 1::2]
        low = 0.5 * (first + second + third + fourth)
        horizontal = 0.5 * (first - second + third - fourth)
        vertical = 0.5 * (first + second - third - fourth)
        diagonal = 0.5 * (first - second - third + fourth)
        magnitude = np.maximum.reduce((
            np.abs(horizontal), np.abs(vertical), np.abs(diagonal)))
        details.append((2.0 ** -scale) * magnitude)
        approximation = low
        scale += 1

    lifted = []
    for scale_index, current in enumerate(details):
        leader = np.zeros_like(current)
        for finer_index in range(scale_index + 1):
            pooled = details[finer_index]
            for _ in range(scale_index - finer_index):
                pooled = np.maximum.reduce((
                    pooled[0::2, 0::2], pooled[0::2, 1::2],
                    pooled[1::2, 0::2], pooled[1::2, 1::2],
                ))
            leader = np.maximum(
                leader,
                maximum_filter(pooled, size=3, mode="nearest"),
            )
        factor = 1 << (scale_index + 1)
        full = np.repeat(np.repeat(leader, factor, axis=0), factor, axis=1)
        lifted.append(full[:original_shape[0], :original_shape[1]])
    return lifted


def centered_log_leaders(
    field: np.ndarray,
    analysis_side: int | None = None,
) -> list[np.ndarray]:
    leaders = haar_wavelet_leaders(field, analysis_side)
    maximum = max((float(np.max(value, initial=0.0)) for value in leaders),
                  default=0.0)
    floor = np.finfo(np.float64).eps * max(maximum, 1.0)
    result = []
    for leader in leaders:
        value = np.log(np.maximum(leader, floor))
        result.append(value - float(np.mean(value)))
    return result


def region_wavelet_leader_features(
    labels: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    analysis_side: int | None = None,
    representation: str = "raw_chart",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Retain mean and RMS centered log leaders for every region and scale."""
    if representation not in ("raw_chart", "scale_law"):
        raise ValueError("leader representation must be raw_chart or scale_law")
    region = np.asarray(labels, dtype=np.int32)
    count = int(np.max(region, initial=-1)) + 1
    flat = region.ravel()
    area = np.bincount(flat, minlength=count).astype(np.float64)
    columns = []
    names = []
    expected_scales = None
    for field_name, field in fields.items():
        if np.asarray(field).shape != region.shape:
            raise ValueError(f"field {field_name!r} does not match labels")
        leaders = centered_log_leaders(field, analysis_side)
        if expected_scales is None:
            expected_scales = len(leaders)
        elif len(leaders) != expected_scales:
            raise ValueError("leader fields disagree on dyadic scale count")
        field_mean = []
        field_variance = []
        for index, leader in enumerate(leaders, 1):
            value = leader.ravel()
            mean = np.bincount(
                flat, weights=value, minlength=count) / np.maximum(area, 1.0)
            second = np.bincount(
                flat, weights=value * value, minlength=count
            ) / np.maximum(area, 1.0)
            variance = np.maximum(second - mean * mean, 0.0)
            field_mean.append(mean)
            field_variance.append(variance)
            if representation == "raw_chart":
                columns.extend((mean, np.sqrt(np.maximum(second, 0.0))))
                names.extend((
                    f"{field_name}_log_leader_mean_j{index}",
                    f"{field_name}_log_leader_rms_j{index}",
                ))
        if representation == "scale_law":
            side = max(region.shape) if analysis_side is None else analysis_side
            log_fraction = (
                np.arange(1, len(leaders) + 1, dtype=np.float64)
                - np.log2(float(_next_power_of_two(int(side))))
            )
            centered_scale = log_fraction - float(np.mean(log_fraction))
            denominator = float(np.sum(centered_scale * centered_scale))
            for statistic_name, chart in (
                ("mean", np.asarray(field_mean).T),
                ("variance", np.asarray(field_variance).T),
            ):
                slope = (chart @ centered_scale) / max(denominator, 1e-30)
                intercept = np.mean(chart, axis=1) - slope * float(
                    np.mean(log_fraction))
                fitted = (
                    intercept[:, None] + slope[:, None] * log_fraction[None, :])
                residual_rms = np.sqrt(np.mean(
                    (chart - fitted) ** 2, axis=1))
                columns.extend((intercept, slope, residual_rms))
                names.extend((
                    f"{field_name}_{statistic_name}_scale_intercept",
                    f"{field_name}_{statistic_name}_scale_slope",
                    f"{field_name}_{statistic_name}_scale_residual_rms",
                ))
    return np.ascontiguousarray(np.column_stack(columns)), tuple(names)


def summarize_wavelet_leaders(features: np.ndarray, names: tuple[str, ...]) -> dict:
    value = np.asarray(features, dtype=np.float64)
    return {
        "regions": int(value.shape[0]),
        "leader_coordinates": int(value.shape[1]),
        "fields": sorted({name.split("_")[0] for name in names}),
        "representation": (
            "raw_chart" if any("_j" in name for name in names)
            else "scale_law"),
    }
