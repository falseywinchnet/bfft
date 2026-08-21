"""One-pass scalar readout after predictive source-identity transport."""

from __future__ import annotations

from typing import Any

import numpy as np

from .continuous_tangent_transport_2d import (
    continuous_tangent_joint_population_2d,
    continuous_tangent_signal_population_2d,
)
from .fused_transport_geometry import (
    predictive_lineage_prolongation_geometry,
)
from .witnessed_characteristic_transport_2d import (
    _lineage_covariance_authority,
    _source_influence_and_lineage,
    _validate,
    _weighted_median,
)


def denoise_post_lineage_prolongation_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the transported predictive section without iterative smoothing.

    Characteristic proposals are validated against an independent direction-
    lane law, then quotiented by exact source identity.  The scalar source
    section is transported once through that lineage.  No named noise model,
    spatial band, support threshold, continuation count, or smoothing time is
    introduced.
    """
    image = _validate(observation)
    signal, signal_diagnostic = continuous_tangent_signal_population_2d(
        image, angular_count=angular_count)
    _influence, lineage = _source_influence_and_lineage(
        signal["mass"],
        signal["source_identity"],
        signal["source_coefficient"],
    )
    source_section = np.sum(
        signal["mass"] * signal["prediction"], axis=-1)
    lineage_field = lineage.reshape(image.shape + (image.size,))
    prolongation = predictive_lineage_prolongation_geometry(
        lineage_field, source_section)
    estimate = np.asarray(
        prolongation["transported_section"], dtype=np.float64)
    residual = image - estimate
    lower = float(np.min(image))
    upper = float(np.max(image))
    return np.clip(estimate, lower, upper), {
        "status": "one-pass post-lineage predictive section",
        "theory_status": (
            "source identity transported before scalar prolongation; "
            "readout gate pending"
        ),
        "angular_count": int(angular_count),
        "signal": signal_diagnostic,
        "prolongation": prolongation,
        "maximum_target_self_lineage": float(np.max(np.abs(
            np.diag(lineage)))),
        "lineage_row_mass_maximum_error": float(np.max(np.abs(
            np.sum(lineage, axis=1) - 1.0))),
        "observation_graph_maximum_error": float(np.max(np.abs(
            estimate + residual - image))),
        "unresolved": [
            "angular quadrature is a numerical refinement coordinate",
            "predictive lineage remains dense quadratic research state",
            "one-pass section has not passed the diverse image gate",
        ],
    }


def denoise_post_lineage_residual_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the literal residual component closing a lineage covariance loop."""
    image = _validate(observation)
    population, population_diagnostic = continuous_tangent_joint_population_2d(
        image, angular_count=angular_count)
    base = _weighted_median(population["signal"], population["mass"])
    _influence, lineage = _source_influence_and_lineage(
        population["prior_mass"],
        population["source_identity"],
        population["source_coefficient"],
    )
    residual = image - base
    prediction = (lineage @ residual.ravel()).reshape(image.shape)
    authority, covariance = _lineage_covariance_authority(
        lineage, residual, prediction)
    update = authority.reshape(image.shape) * prediction
    update_energy = float(np.mean(update * update))
    projection = float(np.mean(residual * update))
    descent = (
        float(np.clip(projection / update_energy, 0.0, 1.0))
        if update_energy > 0.0 else 0.0
    )
    lower = float(np.min(image))
    upper = float(np.max(image))
    estimate = np.clip(base + descent * update, lower, upper)
    returned = estimate - base
    remainder = image - estimate
    return estimate, {
        "status": "one-pass lineage-closed literal residual",
        "theory_status": (
            "fixed predictive base plus target-excluded residual covariance; "
            "quality gate pending"
        ),
        "angular_count": int(angular_count),
        "joint_population": population_diagnostic,
        "global_descent_coefficient": descent,
        "returned_residual_energy": float(np.mean(returned * returned)),
        "remaining_residual_energy": float(np.mean(remainder * remainder)),
        "maximum_target_self_lineage": float(np.max(np.abs(
            np.diag(lineage)))),
        "lineage_row_mass_maximum_error": float(np.max(np.abs(
            np.sum(lineage, axis=1) - 1.0))),
        "observation_graph_maximum_error": float(np.max(np.abs(
            estimate + remainder - image))),
        **covariance,
    }


def post_lineage_residual_forms_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 16,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Expose matched residual-loop readouts for a foundational ablation."""
    image = _validate(observation)
    population, population_diagnostic = continuous_tangent_joint_population_2d(
        image, angular_count=angular_count)
    base = _weighted_median(population["signal"], population["mass"])
    maximum_branch = np.take_along_axis(
        population["signal"],
        np.argmax(population["mass"], axis=-1)[..., None],
        axis=-1,
    )[..., 0]
    _influence, lineage = _source_influence_and_lineage(
        population["prior_mass"],
        population["source_identity"],
        population["source_coefficient"],
    )
    residual = image - base
    transported = (lineage @ residual.ravel()).reshape(image.shape)
    energy = float(np.mean(transported * transported))
    projection = float(np.mean(residual * transported))
    descent = (
        float(np.clip(projection / energy, 0.0, 1.0))
        if energy > 0.0 else 0.0
    )
    authority, covariance = _lineage_covariance_authority(
        lineage, residual, transported)
    lower = float(np.min(image))
    upper = float(np.max(image))
    return {
        "predictive_base": np.clip(base, lower, upper),
        "maximum_posterior_branch": np.clip(
            maximum_branch, lower, upper),
        "transported_residual": np.clip(
            base + descent * transported, lower, upper),
        "covariance_residual": np.clip(
            base + descent * authority.reshape(image.shape) * transported,
            lower,
            upper,
        ),
    }, {
        "joint_population": population_diagnostic,
        "global_descent_coefficient": descent,
        "transported_residual_energy": energy,
        "residual_projection": projection,
        "maximum_target_self_lineage": float(np.max(np.abs(
            np.diag(lineage)))),
        **covariance,
    }
