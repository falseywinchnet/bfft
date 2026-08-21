"""First 1-D relation-transport simmer; deliberately not a promoted denoiser.

The observation is never spatially averaged.  Instead, every local chord
``(a, y_a), (b, y_b)`` carries an affine jet along its characteristic

    T_{a,b}(x) = y_a + (x-a) (y_b-y_a) / (b-a).

At a target ``x`` the push-forwards of bracketing and adjacent chords form an
empirical proposal measure ``mu_x``.  Random replacement atoms generate mostly
incoherent proposals; repeatedly witnessed structure generates concentrated
proposal mass.  The readout below is a continuous pair-interaction barycenter,
not a median, threshold, or named corruption branch.

There is no selected KDE bandwidth.  Kernel scale is itself marginalized over
the empirical distribution of pairwise proposal distances with the
scale-invariant ``ds/s`` weight.  ``scale_quadrature`` and the sampled chord
horizon are numerical representation controls.

What remains unresolved is important: a fixed chord horizon can damage a clean
oscillatory signal.  The next law must make relation scale a transported state
whose survival is paid for by cross-predictive evidence.  This module exists to
make that statement executable and falsifiable; it is not stapled into FMMT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RelationResolution:
    """Quadrature controls for the empirical relation measure."""

    maximum_lag: int | None = None
    scale_quadrature: int = 12


def _chord_pushforwards(y: np.ndarray, target: int, maximum_lag: int) -> np.ndarray:
    """Push local affine chords to one target without reading ``y[target]``."""
    n = y.size
    proposals: list[float] = []
    for lag in range(1, maximum_lag + 1):
        # A chord bracketing the target transports by interpolation.
        if target - lag >= 0 and target + lag < n:
            proposals.append(0.5 * (y[target - lag] + y[target + lag]))
        # Adjacent chords transport by one chord length of extrapolation.
        if target - 2 * lag >= 0:
            proposals.append(2.0 * y[target - lag] - y[target - 2 * lag])
        if target + 2 * lag < n:
            proposals.append(2.0 * y[target + lag] - y[target + 2 * lag])
    if not proposals:
        # Only reachable for a pathologically short vector or zero horizon;
        # validation prevents both, but retain a total internal operation.
        proposals.append(float(y[target]))
    return np.asarray(proposals, dtype=np.float64)


def _scale_marginal_interaction(
    proposals: np.ndarray,
    scale_quadrature: int,
    resolution_floor: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the interaction barycenter and its proposal-density witness."""
    distance = np.abs(proposals[:, None] - proposals[None, :])
    positive = distance[distance > resolution_floor]
    if positive.size == 0:
        density = np.ones(proposals.size, dtype=np.float64)
        return float(np.mean(proposals)), density, np.empty(0, dtype=np.float64)

    # Interior empirical quantiles are a quadrature of the pair-distance
    # measure, not fitted bandwidth candidates.  Increasing the count refines
    # the same integral.
    probabilities = (
        np.arange(1, scale_quadrature + 1, dtype=np.float64)
        / (scale_quadrature + 1.0)
    )
    scales = np.maximum(np.quantile(positive, probabilities), resolution_floor)
    haar_weight = 1.0 / scales
    haar_weight /= np.sum(haar_weight)
    kernel = np.sum(
        np.exp(-0.5 * (distance[..., None] / scales) ** 2)
        * haar_weight[None, None, :],
        axis=-1,
    )
    density = np.mean(kernel, axis=1)
    barycenter = float(np.dot(density, proposals) / np.sum(density))
    return barycenter, density, scales


def denoise_affine_relations(
    observation: np.ndarray,
    resolution: RelationResolution = RelationResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct a line from transported affine-relation witnesses.

    The output is J-invariant pointwise: the proposal measure at index ``i``
    never reads the observed value at ``i``.  That lets discrepancy against the
    unchanged observation become future cross-predictive evidence for relation
    survival without circularly rewarding a replacement atom.
    """
    y = np.asarray(observation, dtype=np.float64)
    if y.ndim != 1 or y.size < 8:
        raise ValueError("affine relation transport expects at least 8 samples")
    if not np.all(np.isfinite(y)):
        raise ValueError("affine relation transport requires finite samples")
    quadrature = int(resolution.scale_quadrature)
    if quadrature < 1:
        raise ValueError("scale_quadrature must be positive")
    maximum_lag = (
        max(1, int(round(math.sqrt(y.size))))
        if resolution.maximum_lag is None
        else int(resolution.maximum_lag)
    )
    if maximum_lag < 1:
        raise ValueError("maximum_lag must be positive")
    maximum_lag = min(maximum_lag, max(1, (y.size - 1) // 2))

    data_span = max(float(np.ptp(y)), 1.0)
    representation_floor = math.sqrt(np.finfo(float).eps) * data_span
    output = np.empty_like(y)
    proposal_counts = np.empty(y.size, dtype=np.int32)
    scale_centers = np.empty(y.size, dtype=np.float64)
    witness_entropy = np.empty(y.size, dtype=np.float64)
    observation_agreement = np.empty(y.size, dtype=np.float64)

    for target in range(y.size):
        proposals = _chord_pushforwards(y, target, maximum_lag)
        center, density, scales = _scale_marginal_interaction(
            proposals, quadrature, representation_floor)
        output[target] = center
        proposal_counts[target] = proposals.size
        scale_centers[target] = float(np.exp(np.mean(np.log(scales)))) if scales.size else 0.0
        probability = density / np.sum(density)
        witness_entropy[target] = -float(np.sum(
            probability * np.log(np.maximum(probability, np.finfo(float).tiny))))
        if scales.size:
            haar_weight = 1.0 / scales
            haar_weight /= np.sum(haar_weight)
            direct_kernel = np.sum(
                np.exp(-0.5 * ((y[target] - proposals[:, None]) / scales) ** 2)
                * haar_weight[None, :],
                axis=1,
            )
            observation_agreement[target] = float(
                np.mean(direct_kernel) / max(float(np.max(density)), representation_floor))
        else:
            observation_agreement[target] = 1.0

    diagnostics: dict[str, Any] = {
        "status": "foundational simmer; not promoted into FMMT",
        "state": "empirical affine-relation pushforward measure",
        "readout": "scale-marginal pair-interaction barycenter",
        "pointwise_j_invariant": True,
        "samples": int(y.size),
        "maximum_lag": int(maximum_lag),
        "proposal_count_min": int(np.min(proposal_counts)),
        "proposal_count_max": int(np.max(proposal_counts)),
        "empirical_scale_geometric_mean": float(np.mean(scale_centers)),
        "witness_entropy_mean": float(np.mean(witness_entropy)),
        "observation_relation_agreement_mean": float(np.mean(observation_agreement)),
        "numerical_resolution": asdict(resolution),
        "unresolved": [
            "relation-horizon survival is not yet transported",
            "only affine jets are represented",
            "boundary chord multiplicity is not continuum-normalized",
        ],
    }
    return np.clip(output, 0.0, 1.0), diagnostics

