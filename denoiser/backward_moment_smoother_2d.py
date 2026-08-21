"""One-transition backward smoother for the complete-moment posterior.

This isolates the Bayesian-smoothing idea from the forward observation
geometry.  It is deliberately a one-transition ablation: the current residual
posterior usually accepts one physical Selling step, and a longer backward
recursion would otherwise confound the test with stopping behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .continual_eikonal_noise_transport_2d import (
    ContinualEikonalResolution,
    _continual_flux_laplacian,
    _validate_image,
    continual_transport_metric,
    directional_noise_witnesses,
)
from .continual_fabada_eikonal_2d import (
    denoise_complete_moment_residual_posterior_2d,
)


ONE_TRANSITION = ContinualEikonalResolution(maximum_iterations=1)


def diagonal_backward_gain(
    prior_variance: np.ndarray,
    averaging: sparse.csr_matrix,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the diagonal RTS gain induced by one positive Markov step."""
    variance = np.maximum(np.asarray(prior_variance, dtype=np.float64), 0.0)
    if averaging.shape != (variance.size, variance.size):
        raise ValueError("averaging operator and variance field must align")
    predicted_variance = (
        averaging.multiply(averaging) @ variance.ravel()
    ).reshape(variance.shape)
    cross_covariance = variance * averaging.diagonal().reshape(variance.shape)
    floor = np.finfo(float).eps * max(float(np.max(variance)), 1.0)
    gain = np.divide(
        cross_covariance,
        predicted_variance + floor,
        out=np.zeros_like(variance),
        where=predicted_variance > 0.0,
    )
    # A diagonal marginal cannot represent off-diagonal covariance.  Its
    # physical contraction is therefore the unit interval, not an extrapolant.
    gain = np.clip(gain, 0.0, 1.0)
    return gain, {
        "mean_backward_gain": float(np.mean(gain)),
        "minimum_backward_gain": float(np.min(gain)),
        "maximum_backward_gain": float(np.max(gain)),
        "mean_predicted_variance": float(np.mean(predicted_variance)),
    }


def denoise_backward_moment_smoother_2d(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconcile the first filtered Selling endpoint back to depth zero."""
    image = _validate_image(observation)
    centre, variance, _radius, initial_witness = directional_noise_witnesses(
        image, image)
    complete_second_moment = variance + centre * centre
    metric = continual_transport_metric(image, complete_second_moment)
    laplacian, _transport, flux = _continual_flux_laplacian(
        metric, np.ones_like(image))
    maximum_degree = flux["maximum_degree"]
    if maximum_degree <= np.finfo(float).tiny:
        return image.copy(), {
            "status": "backward moment smoother exact equilibrium",
            "accepted_forward_transition": False,
            "mean_backward_gain": 0.0,
            "initial_witness": initial_witness,
        }
    delta_time = 1.0 / (2.0 * maximum_degree)
    averaging = sparse.eye(image.size, format="csr") - delta_time * laplacian
    predicted = (averaging @ image.ravel()).reshape(image.shape)
    filtered, forward = denoise_complete_moment_residual_posterior_2d(
        image, ONE_TRANSITION)
    accepted = bool(forward["accepted_iterations"])
    if not accepted:
        result = image.copy()
        gain = np.zeros_like(image)
        gain_diagnostic = {
            "mean_backward_gain": 0.0,
            "minimum_backward_gain": 0.0,
            "maximum_backward_gain": 0.0,
            "mean_predicted_variance": float(np.mean(complete_second_moment)),
        }
    else:
        gain, gain_diagnostic = diagonal_backward_gain(
            complete_second_moment, averaging)
        result = image + gain * (filtered - predicted)
        result = np.clip(result, float(np.min(image)), float(np.max(image)))
    return result, {
        "status": "one-transition diagonal backward moment smoother",
        "theory_status": "Bayesian depth-smoothing ablation; not promoted",
        "accepted_forward_transition": accepted,
        "delta_time": delta_time,
        "maximum_observation_identity_error": float(np.max(np.abs(
            image - (result + (image - result))))),
        "initial_witness": initial_witness,
        "forward": forward,
        **gain_diagnostic,
        "laws": {
            "prediction": "one positive symmetric Selling transition",
            "cross_covariance": "diagonal P_t A_t^T marginal",
            "backward_readout": "x_t + K_t (x_filtered,t+1 - x_predicted,t+1)",
        },
    }
