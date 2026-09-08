"""Blind radio-nuisance transport for positive step-3 reference rasters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class RadioRasterProfile:
    """A measured silence witness and its scale relative to active speech."""

    noise_field: np.ndarray
    noise_to_active_ratio: float
    quantile: float = 0.995

    def __post_init__(self) -> None:
        field = np.asarray(self.noise_field)
        if (
            field.ndim != 2
            or not field.size
            or not np.all(np.isfinite(field))
            or np.any(field < 0.0)
            or not 0.0 < self.noise_to_active_ratio < 1.0
            or not 0.5 < self.quantile < 1.0
        ):
            raise ValueError("radio raster profile is invalid")


def _runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or not values.size:
        raise ValueError("run extraction requires a nonempty boolean vector")
    padded = np.r_[False, values, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops))


def estimate_radio_raster_profile(
    trace: np.ndarray,
    active: np.ndarray,
    *,
    rows: int = 128,
    quantile: float = 0.995,
) -> RadioRasterProfile:
    """Use the longest declared silence run as a correlated nuisance witness."""

    field = np.asarray(trace, dtype=np.float64)
    activity = np.asarray(active, dtype=bool)
    if (
        field.ndim != 2
        or field.shape[1] != activity.size
        or rows < 1
        or rows > field.shape[0]
        or not np.all(np.isfinite(field))
        or not 0.5 < quantile < 1.0
    ):
        raise ValueError("radio profile inputs are incompatible")
    source = np.maximum(field[:rows], 0.0)
    silent_runs = _runs(~activity)
    if not silent_runs or not np.any(activity):
        raise ValueError("radio profile needs both active and silent frames")
    start, stop = max(silent_runs, key=lambda run: (run[1] - run[0], -run[0]))
    noise = source[:, start:stop]
    noise_gauge = float(np.quantile(source[:, ~activity], quantile))
    active_gauge = float(np.quantile(source[:, activity], quantile))
    ratio = noise_gauge / max(active_gauge, 1e-30)
    return RadioRasterProfile(
        noise_field=noise,
        noise_to_active_ratio=float(np.clip(ratio, 1e-6, 1.0 - 1e-6)),
        quantile=quantile,
    )


def matched_noise_patch(
    profile: RadioRasterProfile,
    frame_count: int,
    phase: float = 0.0,
) -> np.ndarray:
    """Resize a circularly shifted silence witness to one reference duration."""

    if frame_count < 2 or not np.isfinite(phase):
        raise ValueError("matched noise patch needs a duration and finite phase")
    source = np.asarray(profile.noise_field, dtype=np.float64)
    shift = int(round((phase % 1.0) * source.shape[1]))
    shifted = np.roll(source, shift, axis=1)
    return ndi.zoom(
        shifted,
        (1.0, frame_count / source.shape[1]),
        order=1,
        mode="wrap",
        prefilter=False,
    )[:, :frame_count]


def transport_reference_raster(
    reference: np.ndarray,
    profile: RadioRasterProfile,
    *,
    noise_scale: float = 1.0,
    phase: float = 0.0,
    row_blur_sigma: float = 0.0,
    time_blur_sigma: float = 0.0,
    amplitude_gamma: float = 1.0,
) -> np.ndarray:
    """Apply measured radio floor, modest blur, and positive AGC compression.

    All operations occur before morphology, ridge extraction, and occupation
    lifting.  ``noise_scale=1`` means the measured inactive/active q995 ratio;
    it is not an arbitrary equality of clean signal and silence amplitude.
    """

    source = np.asarray(reference, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[0] != profile.noise_field.shape[0]
        or source.shape[1] < 2
        or not np.all(np.isfinite(source))
        or min(noise_scale, row_blur_sigma, time_blur_sigma) < 0.0
        or amplitude_gamma <= 0.0
        or not np.all(
            np.isfinite(
                (noise_scale, phase, row_blur_sigma, time_blur_sigma, amplitude_gamma)
            )
        )
    ):
        raise ValueError("radio raster transport configuration is invalid")
    source = np.maximum(source, 0.0)
    gauge = max(float(np.quantile(source, profile.quantile)), 1e-30)
    normalized = source / gauge
    if row_blur_sigma or time_blur_sigma:
        normalized = ndi.gaussian_filter(
            normalized,
            sigma=(row_blur_sigma, time_blur_sigma),
            mode="nearest",
        )
    noise = matched_noise_patch(profile, source.shape[1], phase)
    noise_gauge = max(float(np.quantile(noise, profile.quantile)), 1e-30)
    mixed = normalized + (
        noise_scale * profile.noise_to_active_ratio * noise / noise_gauge
    )
    return gauge * np.power(np.maximum(mixed, 0.0), amplitude_gamma)

