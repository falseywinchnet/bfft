"""Relative transport-chart closure for causal image denoising.

The latent image value is common to every projective tangent chart.  It
therefore vanishes from every pairwise chart difference, just as an unknown
scene vanishes from relative multi-capture blur closure.  This module first
collapses the terminal HJ branch law *within* each direction, then retains the
between-direction closure dispersion as uncertainty about transport itself.

No noise label, iteration count, threshold, bandwidth, or fitted constant is
introduced.  Numerical floors only protect exact zero divisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .causal_information_lineage_2d import (
    causal_information_phase_integrated_law_2d,
)
from .witnessed_characteristic_transport_2d import _validate


def _projective_chart_index(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic direction representatives and branch chart ids."""
    direction = np.asarray(tangent, dtype=np.float64)
    if direction.ndim != 2 or direction.shape[1] != 2:
        raise ValueError("tangent catalogue must have shape Kx2")
    if not np.all(np.isfinite(direction)):
        raise ValueError("tangent catalogue must be finite")
    norm = np.linalg.norm(direction, axis=1)
    if np.any(norm <= 0.0):
        raise ValueError("projective tangents must be nonzero")
    unit = direction / norm[:, None]
    # Canonicalize the projective sign so t and -t identify the same chart.
    sign = np.where(
        (unit[:, 0] < 0.0)
        | ((unit[:, 0] == 0.0) & (unit[:, 1] < 0.0)),
        -1.0,
        1.0,
    )
    unit *= sign[:, None]
    # Trigonometric construction differs only at roundoff.  Quantizing at an
    # epsilon-derived precision restores the exact projective quotient; this
    # is representation cleanup, not a denoising scale.
    precision = int(-np.floor(np.log10(np.sqrt(np.finfo(float).eps))))
    representatives, inverse = np.unique(
        np.round(unit, precision), axis=0, return_inverse=True)
    return representatives, inverse


def _chart_source_gram(
    branch_mass: np.ndarray,
    chart_mass: np.ndarray,
    chart_index: np.ndarray,
    source_identity: np.ndarray,
    source_coefficient: np.ndarray,
) -> np.ndarray:
    """Return the exact coefficient Gram matrix of target-free chart maps.

    Each chart estimate is an affine linear combination of observation
    sources.  Its coefficient Gram matrix is its transported noise covariance
    up to an unknown common variance scale.  Sparse source identities avoid a
    dense pixel-by-pixel operator matrix.
    """
    mass = np.asarray(branch_mass, dtype=np.float64)
    identity = np.asarray(source_identity, dtype=np.int64)
    coefficient = np.asarray(source_coefficient, dtype=np.float64)
    if identity.shape != coefficient.shape:
        raise ValueError("source identity and coefficient must align")
    if identity.shape[:-1] != mass.shape:
        raise ValueError("source graph must align with terminal branches")
    chart_count = chart_mass.shape[-1]
    pixels = int(np.prod(mass.shape[:-1]))
    branch_count = mass.shape[-1]
    source_slots = identity.shape[-1]
    flat_mass = mass.reshape(pixels, branch_count)
    flat_chart_mass = chart_mass.reshape(pixels, chart_count)
    flat_identity = identity.reshape(pixels, branch_count, source_slots)
    flat_coefficient = coefficient.reshape(pixels, branch_count, source_slots)
    gram = np.zeros((pixels, chart_count, chart_count), dtype=np.float64)
    tiny = np.finfo(float).tiny
    members = [np.flatnonzero(chart_index == chart)
               for chart in range(chart_count)]
    for pixel in range(pixels):
        sparse_vectors: list[tuple[np.ndarray, np.ndarray]] = []
        for chart, indices in enumerate(members):
            denominator = flat_chart_mass[pixel, chart]
            if denominator <= tiny:
                sparse_vectors.append((
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float64)))
                continue
            ids = flat_identity[pixel, indices].reshape(-1)
            values = (
                flat_coefficient[pixel, indices]
                * (flat_mass[pixel, indices] / denominator)[:, None]
            ).reshape(-1)
            unique, inverse = np.unique(ids, return_inverse=True)
            accumulated = np.zeros(unique.size, dtype=np.float64)
            np.add.at(accumulated, inverse, values)
            active = accumulated != 0.0
            sparse_vectors.append((unique[active], accumulated[active]))
        for first in range(chart_count):
            first_id, first_value = sparse_vectors[first]
            for second in range(first, chart_count):
                second_id, second_value = sparse_vectors[second]
                _common, first_position, second_position = np.intersect1d(
                    first_id, second_id, assume_unique=True,
                    return_indices=True)
                value = float(np.dot(
                    first_value[first_position],
                    second_value[second_position]))
                gram[pixel, first, second] = value
                gram[pixel, second, first] = value
    return gram.reshape(mass.shape[:-1] + (chart_count, chart_count))


def cross_chart_transport_closure_readout(
    observation: np.ndarray,
    law: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    """Read a relative-closure posterior from a transported branch law.

    Terminal branches sharing a projective tangent are one transport chart.
    Their HJ collision barycenter is the chart estimate.  Normalized chart
    ownership chooses no winner; the weighted pairwise discrepancy is kept as
    transport-map variance.  Observation innovation may move toward chart
    consensus only to the degree that it exceeds that map uncertainty.
    """
    image = _validate(observation)
    signal = np.asarray(law["signal"], dtype=np.float64)
    mass = np.asarray(law["hj_simplex_collision_mass"], dtype=np.float64)
    tangent = np.asarray(law["tangent"], dtype=np.float64)
    if signal.shape != mass.shape or signal.shape[:2] != image.shape:
        raise ValueError("terminal signal and mass must align with observation")
    if signal.shape[-1] != tangent.shape[0]:
        raise ValueError("each terminal branch must have a tangent identity")
    if np.any(mass < 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError("terminal branch mass must be finite and nonnegative")

    representatives, chart_index = _projective_chart_index(tangent)
    chart_count = representatives.shape[0]
    chart_mass = np.empty(image.shape + (chart_count,), dtype=np.float64)
    chart_signal = np.empty_like(chart_mass)
    tiny = np.finfo(float).tiny
    for chart in range(chart_count):
        members = chart_index == chart
        owned_mass = np.sum(mass[..., members], axis=-1)
        chart_mass[..., chart] = owned_mass
        chart_signal[..., chart] = np.divide(
            np.sum(mass[..., members] * signal[..., members], axis=-1),
            owned_mass,
            out=image.copy(),
            where=owned_mass > tiny,
        )

    total_mass = np.sum(chart_mass, axis=-1, keepdims=True)
    if np.any(total_mass <= 0.0):
        raise RuntimeError("terminal law lost every transport chart")
    ownership = chart_mass / total_mass
    consensus = np.sum(ownership * chart_signal, axis=-1)

    # This is exactly 1/2 sum_ab pi_a pi_b (z_a-z_b)^2.  The centered form is
    # evaluated in O(D), while the pairwise identity is tested separately.
    closure_variance = np.sum(
        ownership * (chart_signal - consensus[..., None]) ** 2, axis=-1)
    innovation = image - consensus
    innovation_energy = innovation * innovation
    scale = max(float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
    floor = np.finfo(float).eps * scale * scale
    consensus_authority = innovation_energy / (
        innovation_energy + closure_variance + floor)
    estimate = (
        (1.0 - consensus_authority) * image
        + consensus_authority * consensus)
    lower = float(np.min(image))
    upper = float(np.max(image))
    effective_chart_count = 1.0 / np.sum(ownership * ownership, axis=-1)
    source_coverage_estimate = estimate
    source_coverage_authority = consensus_authority
    source_noise_scale = np.zeros(image.shape, dtype=np.float64)
    source_common_variance = np.zeros(image.shape, dtype=np.float64)
    source_consensus_gain = np.zeros(image.shape, dtype=np.float64)
    source_coverage = np.zeros_like(chart_mass)
    coverage_ownership = ownership
    if "source_identity" in law and "source_coefficient" in law:
        source_gram = _chart_source_gram(
            mass,
            chart_mass,
            chart_index,
            np.asarray(law["source_identity"]),
            np.asarray(law["source_coefficient"]),
        )
        diagonal = np.diagonal(source_gram, axis1=-2, axis2=-1)
        source_coverage = np.divide(
            1.0,
            diagonal,
            out=np.zeros_like(diagonal),
            where=diagonal > tiny,
        )
        # Absolute coverage must not be folded back into normalized ownership.
        # Ownership defines the relative map posterior; source coverage only
        # calibrates how uncertain that posterior remains in its common mode.
        coverage_ownership = ownership
        coverage_consensus = consensus
        coverage_variance = closure_variance
        pair_energy = np.zeros(image.shape, dtype=np.float64)
        pair_gain = np.zeros(image.shape, dtype=np.float64)
        for first in range(chart_count):
            for second in range(first + 1, chart_count):
                pair_weight = (
                    ownership[..., first] * ownership[..., second])
                difference = (
                    chart_signal[..., first] - chart_signal[..., second])
                difference_gain = np.maximum(
                    source_gram[..., first, first]
                    + source_gram[..., second, second]
                    - 2.0 * source_gram[..., first, second],
                    0.0,
                )
                pair_energy += pair_weight * difference * difference
                pair_gain += pair_weight * difference_gain
        source_noise_scale = np.divide(
            pair_energy,
            pair_gain,
            out=np.zeros_like(pair_energy),
            where=pair_gain > floor,
        )
        source_consensus_gain = np.einsum(
            "...a,...ab,...b->...",
            ownership, source_gram, ownership)
        source_common_variance = source_noise_scale * source_consensus_gain
        coverage_innovation = image - coverage_consensus
        coverage_innovation_energy = coverage_innovation * coverage_innovation
        source_coverage_authority = coverage_innovation_energy / (
            coverage_innovation_energy
            + coverage_variance
            + source_common_variance
            + floor)
        source_coverage_estimate = (
            (1.0 - source_coverage_authority) * image
            + source_coverage_authority * coverage_consensus)
    else:
        coverage_consensus = consensus
        coverage_variance = closure_variance
    readouts = {
        "cross_chart_closure_barycenter": np.clip(estimate, lower, upper),
        "source_coverage_closure_barycenter": np.clip(
            source_coverage_estimate, lower, upper),
        "transport_chart_consensus": consensus,
        "transport_chart_variance": closure_variance,
        "transport_chart_authority": consensus_authority,
        "transport_chart_ownership": ownership,
        "transport_chart_signal": chart_signal,
        "effective_transport_chart_count": effective_chart_count,
        "source_coverage_consensus": coverage_consensus,
        "source_coverage_variance": coverage_variance,
        "source_coverage_authority": source_coverage_authority,
        "source_chart_coverage": source_coverage,
        "source_coverage_ownership": coverage_ownership,
        "source_noise_scale": source_noise_scale,
        "source_consensus_gain": source_consensus_gain,
        "source_common_variance": source_common_variance,
    }
    diagnostic: dict[str, float | int | str] = {
        "status": (
            "latent-cancelling relative closure across projective transport "
            "charts"
        ),
        "chart_count": int(chart_count),
        "mass_maximum_error": float(np.max(np.abs(total_mass[..., 0] - 1.0))),
        "mean_transport_chart_variance": float(np.mean(closure_variance)),
        "mean_transport_chart_authority": float(np.mean(consensus_authority)),
        "mean_effective_transport_chart_count": float(np.mean(
            effective_chart_count)),
        "mean_observation_displacement_rms": float(np.sqrt(np.mean(
            (estimate - image) ** 2))),
        "mean_source_coverage_authority": float(np.mean(
            source_coverage_authority)),
        "mean_source_noise_scale": float(np.mean(source_noise_scale)),
        "mean_source_consensus_gain": float(np.mean(source_consensus_gain)),
        "mean_source_common_variance": float(np.mean(
            source_common_variance)),
        "mean_source_coverage_displacement_rms": float(np.sqrt(np.mean(
            (source_coverage_estimate - image) ** 2))),
    }
    return readouts, diagnostic


def denoise_cross_chart_transport_closure_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
    quantile_count: int = 16,
    phase_count: int = 1,
    memory_ceiling_bytes: int | None = None,
    complete_residual_moment: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the causal law, then apply relative chart closure once."""
    image = _validate(observation)
    law, transport_diagnostic = causal_information_phase_integrated_law_2d(
        image,
        angular_count=angular_count,
        quantile_count=quantile_count,
        phase_count=phase_count,
        memory_ceiling_bytes=memory_ceiling_bytes,
        complete_residual_moment=complete_residual_moment,
    )
    readouts, closure_diagnostic = cross_chart_transport_closure_readout(
        image, law)
    return readouts["cross_chart_closure_barycenter"], {
        "status": "causal HJ transport followed by relative chart closure",
        "transport": transport_diagnostic,
        "closure": closure_diagnostic,
        "readouts": readouts,
    }
