"""Continuous posterior over center, blur inverse, and noise transport."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from denoiser.fmmt_certified import denoise_fmmt

from .multicapture_transport import MultiCaptureTransportResult
from .spatial_transport import (
    CompactGlobalExposureField,
    pullback_barycentric_values,
    pullback_compact_global_values,
)
from .uncertainty import estimate_noise_discrepancy


@dataclass(frozen=True)
class MultiCapturePosteriorSolution:
    image: np.ndarray
    uncertainty: np.ndarray
    center_image: np.ndarray
    atlas_image: np.ndarray
    inverse_image: np.ndarray
    denoised_image: np.ndarray
    center_mass: float
    atlas_mass: float
    denoise_mass: float
    diagnostics: dict[str, object]


def _centered_observations(
    result: MultiCaptureTransportResult,
) -> tuple[np.ndarray, ...]:
    centered = []
    for image, field in zip(result.radiometric_images, result.fields):
        if isinstance(field, CompactGlobalExposureField):
            value, _ = pullback_compact_global_values(image, field)
        else:
            value, _ = pullback_barycentric_values(
                image, field.barycentric_field())
        centered.append(value)
    return tuple(centered)


def _denoise_channels(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        return denoise_fmmt(value)[0]
    channel_count = value.shape[2]
    # Each channel owns a complete immutable FMMT measure. Bounded concurrency
    # changes only scheduling; channel order and each scalar estimator remain
    # identical to the serial representation oracle.
    with ThreadPoolExecutor(max_workers=min(channel_count, 3)) as executor:
        channels = tuple(executor.map(
            lambda channel: denoise_fmmt(value[..., channel])[0],
            range(channel_count),
        ))
    return np.stack(channels, axis=2)


def _innovation_noise_authority(
    centered: tuple[np.ndarray, ...],
    predictions: np.ndarray,
    read_sigma: float,
    *,
    window: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return where a denoising displacement is supported by innovations.

    A closure residual shared by the captures is evidence of unresolved scene
    transport, not disposable noise.  Random fine residual has much less
    coherent mean after centering and averaging.  The resulting authority is
    continuous and local; it does not assign either residual a class.
    """
    residual = np.stack(centered, axis=0) - predictions
    coherent = np.mean(residual, axis=0)
    if coherent.ndim == 3:
        coherent_energy = np.mean(coherent ** 2, axis=2)
    else:
        coherent_energy = coherent ** 2
    coherent_energy = ndimage.uniform_filter(
        coherent_energy, size=max(int(window), 1), mode="reflect")
    noise_energy = max(float(read_sigma), 1e-8) ** 2
    closure_authority = noise_energy / (
        noise_energy + coherent_energy + 1e-16)

    # Separate replicated fine structure from capture-specific fine
    # innovation.  For independent innovations, averaging N captures retains
    # 1/N of their energy; repeated scene structure retains essentially all of
    # it.  Rescaling that interval gives a continuous novelty measure.
    stack = np.stack(centered, axis=0)
    if stack.ndim == 4:
        smooth = ndimage.gaussian_filter(
            stack, sigma=(0.0, 1.0, 1.0, 0.0), mode="reflect")
        fine = stack - smooth
        total_fine = np.mean(fine ** 2, axis=(0, 3))
        shared_fine = np.mean(np.mean(fine, axis=0) ** 2, axis=2)
    else:
        smooth = ndimage.gaussian_filter(
            stack, sigma=(0.0, 1.0, 1.0), mode="reflect")
        fine = stack - smooth
        total_fine = np.mean(fine ** 2, axis=0)
        shared_fine = np.mean(fine, axis=0) ** 2
    total_fine = ndimage.uniform_filter(
        total_fine, size=max(int(window), 1), mode="reflect")
    shared_fine = ndimage.uniform_filter(
        shared_fine, size=max(int(window), 1), mode="reflect")
    replicated_fraction = np.clip(
        shared_fine / (total_fine + 1e-16), 0.0, 1.0)
    independent_floor = 1.0 / max(len(centered), 1)
    fine_novelty = np.clip(
        (1.0 - replicated_fraction)
        / max(1.0 - independent_floor, 1e-8),
        0.0,
        1.0,
    )
    authority = closure_authority * fine_novelty
    return np.clip(authority, 0.0, 1.0), replicated_fraction


def _atlas_authority_map(
    atlas_record: dict[str, object] | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if atlas_record is None:
        return np.ones(shape, dtype=np.float64)
    height, width = shape
    extent = int(atlas_record["patch_size"])
    spatial_sigma = max(0.55 * extent, 1.0)
    yy, xx = np.mgrid[:height, :width]
    numerator = np.zeros(shape, dtype=np.float64)
    mass = np.zeros(shape, dtype=np.float64)
    for chart in atlas_record["chart_records"]:
        center_x, center_y = chart["center_xy"]
        chart_mass = float(chart["positive_chart_mass"])
        window = chart_mass * np.exp(-0.5 * (
            ((xx - center_x) / spatial_sigma) ** 2
            + ((yy - center_y) / spatial_sigma) ** 2
        ))
        numerator += float(chart["local_deviation_authority"]) * window
        mass += window
    return np.clip(
        numerator / np.maximum(mass, np.finfo(float).tiny), 0.0, 1.0)


def solve_multicapture_transport_posterior(
    result: MultiCaptureTransportResult,
    *,
    closure_floor: float = 0.08,
    authority_floor: float = 0.10,
    noise_floor: float = 0.02,
) -> MultiCapturePosteriorSolution:
    """Retain center, inverse, and denoising as one continuous measure.

    ``closure_floor`` is a one-measure dimensionless resolution and therefore
    concentrates as ``1/sqrt(capture_count)``. ``authority_floor`` is the
    local-chart resolution; ``noise_floor`` is a radiance-domain fine-residual
    floor. None depends on a named blur/noise family, source identity, or
    reference image.
    """
    centered = _centered_observations(result)
    stack = np.stack(centered, axis=0)
    center = np.mean(stack, axis=0)
    predictions = np.asarray(
        result.predicted_transport_gauge_observations, dtype=np.float64)
    disagreement = float(np.sqrt(np.mean(
        (stack - center[None, ...]) ** 2)))
    closure = float(np.sqrt(np.mean((predictions - stack) ** 2)))
    closure_ratio = closure / max(disagreement, 1e-12)
    closure_gain = max(1.0 - closure_ratio, 0.0)
    closure_scale = max(
        float(closure_floor) / np.sqrt(len(centered)), 1e-8)
    closure_mass = 1.0 - np.exp(-(
        closure_gain / closure_scale) ** 2)

    atlas_record = result.diagnostics.get("spatial_mixing_atlas")
    authority_map = _atlas_authority_map(atlas_record, center.shape[:2])
    authority = float(np.mean(authority_map))
    authority_scale = max(float(authority_floor), 1e-8)
    authority_mass_map = 1.0 - np.exp(-(
        authority_map / authority_scale) ** 2)
    authority_mass = float(np.mean(authority_mass_map))

    noise_records = [estimate_noise_discrepancy(observation, prediction)
                     for observation, prediction in zip(centered, predictions)]
    read_sigma = float(np.median([
        item.read_sigma for item in noise_records]))
    structured_rms = float(np.median([
        item.structured_rms for item in noise_records]))
    noise_scale = max(float(noise_floor), 1e-8)
    inverse_noise_mass = noise_scale ** 2 / (
        noise_scale ** 2 + read_sigma ** 2)
    atlas_mass_map = np.clip(
        closure_mass * authority_mass_map * inverse_noise_mass, 0.0, 1.0)
    center_mass_map = 1.0 - atlas_mass_map
    atlas_mass = float(np.mean(atlas_mass_map))
    center_mass = float(np.mean(center_mass_map))
    expanded_atlas_mass = atlas_mass_map
    expanded_center_mass = center_mass_map
    if center.ndim == 3:
        expanded_atlas_mass = atlas_mass_map[..., None]
        expanded_center_mass = center_mass_map[..., None]
    inverse_posterior = (
        expanded_center_mass * center + expanded_atlas_mass * result.image)

    fmmt_proposal = _denoise_channels(inverse_posterior)
    innovation_authority, replicated_fine_fraction = (
        _innovation_noise_authority(
        centered, predictions, read_sigma)
    )
    expanded_authority = innovation_authority
    if inverse_posterior.ndim == 3:
        expanded_authority = innovation_authority[..., None]
    denoised = inverse_posterior + expanded_authority * (
        fmmt_proposal - inverse_posterior)
    noise_evidence = read_sigma ** 2 / (
        read_sigma ** 2 + noise_scale ** 2)
    noise_purity = read_sigma ** 2 / max(
        read_sigma ** 2 + structured_rms ** 2, 1e-16)
    denoise_mass = float(np.clip(
        noise_evidence * noise_purity ** 2, 0.0, 1.0))
    image = (
        (1.0 - denoise_mass) * inverse_posterior
        + denoise_mass * denoised)

    center_variance = np.mean(
        (stack - center[None, ...]) ** 2, axis=0)
    center_uncertainty = np.sqrt(np.maximum(center_variance, 0.0))
    between_inverse = (
        expanded_center_mass * expanded_atlas_mass
        * (result.image - center) ** 2)
    inverse_uncertainty = np.sqrt(
        expanded_center_mass * center_uncertainty ** 2
        + expanded_atlas_mass * result.uncertainty ** 2
        + between_inverse)
    between_noise = (
        denoise_mass * (1.0 - denoise_mass)
        * (denoised - inverse_posterior) ** 2)
    uncertainty = np.sqrt(inverse_uncertainty ** 2 + between_noise)
    return MultiCapturePosteriorSolution(
        image=image,
        uncertainty=uncertainty,
        center_image=center,
        atlas_image=result.image,
        inverse_image=inverse_posterior,
        denoised_image=denoised,
        center_mass=center_mass,
        atlas_mass=atlas_mass,
        denoise_mass=denoise_mass,
        diagnostics={
            "method": "center_inverse_noise_positive_measure_posterior",
            "selection_policy": (
                "all_center_inverse_and_noise_measures_retained_no_winner_"
                "branch"),
            "center_mass": center_mass,
            "atlas_mass": atlas_mass,
            "denoise_mass": denoise_mass,
            "closure_ratio": closure_ratio,
            "closure_gain": closure_gain,
            "closure_sampling_resolution": closure_scale,
            "closure_evidence_mass": closure_mass,
            "mean_local_deviation_authority": authority,
            "authority_evidence_mass": authority_mass,
            "atlas_mass_range": [
                float(np.min(atlas_mass_map)),
                float(np.max(atlas_mass_map)),
            ],
            "local_deviation_authority_range": [
                float(np.min(authority_map)),
                float(np.max(authority_map)),
            ],
            "estimated_read_sigma_median": read_sigma,
            "estimated_structured_residual_rms_median": structured_rms,
            "inverse_noise_mass": inverse_noise_mass,
            "noise_evidence_mass": noise_evidence,
            "noise_purity": noise_purity,
            "innovation_noise_authority_mean": float(np.mean(
                innovation_authority)),
            "innovation_noise_authority_minimum": float(np.min(
                innovation_authority)),
            "innovation_noise_authority_maximum": float(np.max(
                innovation_authority)),
            "replicated_fine_fraction_mean": float(np.mean(
                replicated_fine_fraction)),
            "replicated_fine_fraction_minimum": float(np.min(
                replicated_fine_fraction)),
            "replicated_fine_fraction_maximum": float(np.max(
                replicated_fine_fraction)),
            "fmmt_displacement_rms": float(np.sqrt(np.mean(
                (fmmt_proposal - inverse_posterior) ** 2))),
            "transported_noise_displacement_rms": float(np.sqrt(np.mean(
                (denoised - inverse_posterior) ** 2))),
            "between_inverse_uncertainty_rms": float(np.sqrt(np.mean(
                between_inverse))),
            "between_noise_uncertainty_rms": float(np.sqrt(np.mean(
                between_noise))),
            "total_uncertainty_rms": float(np.sqrt(np.mean(
                uncertainty ** 2))),
        },
    )
