"""Positive image-domain posterior over covariance and common-K4 gauges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .full_quartic_transport import (
    FullQuarticTransport,
    _covariance_square_root,
    directional_quartic_dictionary,
    estimate_full_quartic_transport,
)
from .multicapture_transport import _positive_sigma_measure
from .real_capture_evaluation import _fourier_amplification
from .spatial_consensus import (
    SpatialFieldConsensusSolution,
    solve_spatial_field_consensus,
)
from .spatial_transport import CompactGlobalExposureField


@dataclass(frozen=True)
class QuarticGaugePosteriorSolution:
    image: np.ndarray
    uncertainty: np.ndarray
    predicted_transport_gauge_observations: np.ndarray
    covariance_solution: SpatialFieldConsensusSolution
    quartic_solution: SpatialFieldConsensusSolution
    quartic_estimate: FullQuarticTransport
    covariance_posterior_mass: float
    quartic_posterior_mass: float
    diagnostics: dict[str, object]


def _global_field(
    name: str,
    shape: tuple[int, int],
    points: np.ndarray,
    weights: np.ndarray,
) -> CompactGlobalExposureField:
    return CompactGlobalExposureField(
        name,
        shape,
        points,
        weights,
    )


def _realize_dictionary_measure(
    mixture: np.ndarray,
    covariance: np.ndarray,
    dictionary: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, np.ndarray]:
    factor = _covariance_square_root(covariance)
    point_parts = []
    weight_parts = []
    for component_mass, (standard_points, standard_weights) in zip(
        mixture, dictionary
    ):
        if component_mass <= 1e-10:
            continue
        point_parts.append(standard_points @ factor.T)
        weight_parts.append(component_mass * standard_weights)
    points = np.concatenate(point_parts)
    weights = np.concatenate(weight_parts)
    weights /= np.sum(weights)
    return points, weights


def _closure_rms(
    solution: SpatialFieldConsensusSolution,
    observations: Sequence[np.ndarray],
    frame_weights: np.ndarray | None = None,
) -> float:
    residual = (
        solution.predicted_transport_gauge_observations
        - np.stack(observations, axis=0))
    if frame_weights is None:
        return float(np.sqrt(np.mean(residual * residual)))
    weights = np.asarray(frame_weights, dtype=np.float64)
    if weights.shape == (len(observations),):
        weights = np.broadcast_to(
            weights[:, None, None], residual.shape[:3])
    if weights.shape != residual.shape[:3]:
        raise ValueError("closure weights must be N or NxHxW")
    weights_for_image = (
        weights if residual.ndim == 3 else weights[..., None])
    return float(np.sqrt(
        np.sum(weights_for_image * residual * residual)
        / max(float(np.sum(np.broadcast_to(
            weights_for_image, residual.shape))), 1e-12)))


def solve_quartic_gauge_posterior(
    observations: Sequence[np.ndarray],
    covariances: np.ndarray,
    *,
    estimate: FullQuarticTransport | None = None,
    frame_weights: np.ndarray | None = None,
    passes: int = 32,
    descent_method: str = "optimal_positive_line",
    posterior_temperature: float = 0.05,
) -> QuarticGaugePosteriorSolution:
    """Transport both covariance and K4 gauges through the latent inverse.

    The K4 member uses the baseline-common-gauge branch of the full relative
    tensor posterior.  Neither member is selected.  Their posterior action is
    the dimensionless forward-closure gain plus the absolute redistribution
    of outer Fourier-circle mass, so both ringing and oversmoothing are priced.
    """
    images = tuple(np.asarray(item, dtype=np.float64) for item in observations)
    covariance = np.asarray(covariances, dtype=np.float64)
    if len(images) < 3 or covariance.shape != (len(images), 2, 2):
        raise ValueError("quartic gauge posterior needs N>=3 covariances")
    if any(item.shape != images[0].shape for item in images[1:]):
        raise ValueError("quartic gauge observations must share one raster")
    shape = images[0].shape[:2]
    quartic_estimate = (
        estimate if estimate is not None
        else estimate_full_quartic_transport(images, covariance))
    dictionary, _, _ = directional_quartic_dictionary(8)
    branch_weights = np.asarray(quartic_estimate.diagnostics[
        "gauge_branch_dictionary_weights"], dtype=np.float64)
    if (
        branch_weights.ndim != 3
        or branch_weights.shape[1:] != (
            len(images), len(dictionary))
        or branch_weights.shape[0] < 1
    ):
        raise ValueError("quartic estimate has incompatible gauge branches")

    covariance_fields = []
    quartic_fields = []
    for capture in range(len(images)):
        covariance_points, covariance_weights = _positive_sigma_measure(
            covariance[capture])
        quartic_points, quartic_weights = _realize_dictionary_measure(
            branch_weights[0, capture], covariance[capture], dictionary)
        covariance_fields.append(_global_field(
            f"covariance_gauge_{capture}",
            shape,
            covariance_points,
            covariance_weights,
        ))
        quartic_fields.append(_global_field(
            f"quartic_relative_baseline_gauge_{capture}",
            shape,
            quartic_points,
            quartic_weights,
        ))
    covariance_solution = solve_spatial_field_consensus(
        images,
        covariance_fields,
        frame_weights=frame_weights,
        passes=passes,
        descent_method=descent_method,
    )
    quartic_solution = solve_spatial_field_consensus(
        images,
        quartic_fields,
        frame_weights=frame_weights,
        passes=passes,
        descent_method=descent_method,
    )
    covariance_closure = _closure_rms(
        covariance_solution, images, frame_weights)
    quartic_closure = _closure_rms(
        quartic_solution, images, frame_weights)
    covariance_fourier = _fourier_amplification(
        covariance_solution.image, images)
    quartic_fourier = _fourier_amplification(quartic_solution.image, images)
    closure_ratio = quartic_closure / max(covariance_closure, 1e-12)
    outer_ratio = (
        float(quartic_fourier["outer_three_mean_ratio"])
        / max(float(covariance_fourier["outer_three_mean_ratio"]), 1e-12))
    transport_action = float(
        closure_ratio - 1.0 + abs(np.log(max(outer_ratio, 1e-12))))
    temperature = max(float(posterior_temperature), 1e-8)
    quartic_mass = float(1.0 / (1.0 + np.exp(np.clip(
        transport_action / temperature, -60.0, 60.0))))
    covariance_mass = 1.0 - quartic_mass
    image = (
        covariance_mass * covariance_solution.image
        + quartic_mass * quartic_solution.image)
    prediction = (
        covariance_mass
        * covariance_solution.predicted_transport_gauge_observations
        + quartic_mass
        * quartic_solution.predicted_transport_gauge_observations)
    between = (
        covariance_mass * quartic_mass
        * (quartic_solution.image - covariance_solution.image) ** 2)
    uncertainty = np.sqrt(
        covariance_mass * covariance_solution.uncertainty ** 2
        + quartic_mass * quartic_solution.uncertainty ** 2
        + between)
    posterior_entropy = float(-sum(
        mass * np.log(max(mass, np.finfo(float).tiny))
        for mass in (covariance_mass, quartic_mass)))
    return QuarticGaugePosteriorSolution(
        image=image,
        uncertainty=uncertainty,
        predicted_transport_gauge_observations=prediction,
        covariance_solution=covariance_solution,
        quartic_solution=quartic_solution,
        quartic_estimate=quartic_estimate,
        covariance_posterior_mass=covariance_mass,
        quartic_posterior_mass=quartic_mass,
        diagnostics={
            "method": (
                "positive_covariance_quartic_image_posterior_transport"),
            "selection_policy": (
                "both_gauges_reconstructed_and_retained_no_winner_branch"),
            "transport_action": transport_action,
            "transport_action_terms": {
                "closure_ratio_minus_one": closure_ratio - 1.0,
                "absolute_log_outer_fourier_ratio": abs(np.log(max(
                    outer_ratio, 1e-12))),
            },
            "posterior_temperature": temperature,
            "covariance_posterior_mass": covariance_mass,
            "quartic_posterior_mass": quartic_mass,
            "posterior_entropy": posterior_entropy,
            "posterior_effective_branch_count": float(
                1.0 / (covariance_mass ** 2 + quartic_mass ** 2)),
            "covariance_forward_closure_rms": covariance_closure,
            "quartic_forward_closure_rms": quartic_closure,
            "forward_closure_measure": (
                "supplied_positive_precision"
                if frame_weights is not None else "uniform_pixel_measure"),
            "covariance_outer_fourier_ratio": float(
                covariance_fourier["outer_three_mean_ratio"]),
            "quartic_outer_fourier_ratio": float(
                quartic_fourier["outer_three_mean_ratio"]),
            "between_gauge_uncertainty_rms": float(np.sqrt(np.mean(between))),
            "total_uncertainty_rms": float(np.sqrt(np.mean(
                uncertainty * uncertainty))),
            "covariance_generated_global_storage_bytes": (
                covariance_solution.diagnostics[
                    "generated_global_fft_storage_bytes"]),
            "quartic_generated_global_storage_bytes": (
                quartic_solution.diagnostics[
                    "generated_global_fft_storage_bytes"]),
            "quartic_materialized_bytes_avoided": (
                quartic_solution.diagnostics[
                    "generated_global_materialized_bytes_avoided"]),
        },
    )
