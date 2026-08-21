"""Residual-reflection consistency probe for a transport denoiser.

This is a non-neural translation of Positive2Negative and trajectory
consistency.  A base estimator produces ``x`` and residual ``r=y-x``.  The
counterfactual observation ``y-=x-r`` preserves the candidate signal while
reversing the nuisance realization.  A credible residual should make the base
estimator recover the same signal from both observations.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .continual_fabada_eikonal_2d import (
    denoise_continual_residual_posterior_2d,
)
from .continual_eikonal_noise_transport_2d import _validate_image


Estimator = Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]


def reflection_consistency_authority(
    observation: np.ndarray,
    estimate: np.ndarray,
    reflected_estimate: np.ndarray,
) -> np.ndarray:
    """Return residual authority from signal-preserving reflection agreement."""
    y = np.asarray(observation, dtype=np.float64)
    x = np.asarray(estimate, dtype=np.float64)
    reflected = np.asarray(reflected_estimate, dtype=np.float64)
    if y.shape != x.shape or reflected.shape != x.shape:
        raise ValueError("reflection-consistency fields must be aligned")
    residual_energy = (y - x) ** 2
    inconsistency_energy = (reflected - x) ** 2
    magnitude = max(float(np.max(np.abs(y))), float(np.ptp(y)), 1.0)
    floor = np.finfo(float).eps * magnitude * magnitude
    authority = residual_energy / (
        residual_energy + inconsistency_energy + floor)
    return np.clip(authority, 0.0, 1.0)


def denoise_reflection_consistent_posterior_2d(
    observation: np.ndarray,
    estimator: Estimator = denoise_continual_residual_posterior_2d,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cross-check a residual posterior against its reflected nuisance law."""
    image = _validate_image(observation)
    estimate, forward = estimator(image)
    residual = image - estimate
    reflected_observation = estimate - residual
    reflected_estimate, reflected = estimator(reflected_observation)
    authority = reflection_consistency_authority(
        image, estimate, reflected_estimate)
    result = image - authority * residual
    result = np.clip(result, float(np.min(image)), float(np.max(image)))
    return result, {
        "status": "residual-reflection consistency posterior probe",
        "theory_status": (
            "falsified as positive authority; retained only as a veto diagnostic"),
        "mean_reflection_authority": float(np.mean(authority)),
        "minimum_reflection_authority": float(np.min(authority)),
        "maximum_reflection_authority": float(np.max(authority)),
        "mean_reflection_disagreement_squared": float(np.mean(
            (reflected_estimate - estimate) ** 2)),
        "mean_forward_residual_squared": float(np.mean(residual * residual)),
        "maximum_reflection_identity_error": float(np.max(np.abs(
            reflected_observation - (estimate - residual)))),
        "forward": forward,
        "reflected": reflected,
        "laws": {
            "counterfactual": "candidate signal fixed; inferred residual sign reversed",
            "authority": "residual energy divided by residual plus inconsistency energy",
            "readout": "pointwise posterior interpolation between observation and base estimate",
        },
    }
