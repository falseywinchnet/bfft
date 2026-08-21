"""Conservative posterior/residual smoothing exchange in two dimensions.

This is a research dynamics, not a promoted denoiser.  Its state is the exact
decomposition

    y = p + r,

where neither ``p`` nor ``r`` is declared to be truth or noise.  One cycle has
three conservative acts:

1. smooth ``p`` and transfer its rejected signed field to ``r``;
2. smooth ``r`` and transfer its phase-supported signed field to ``p``;
3. smooth the recomposed observation in the evolved geometry and transfer
   only the joint correction witnessed by both residual phase and a
   target-excluded posterior-curvature cavity.

Every transfer preserves ``p+r`` pointwise.  Smoothing is the positive
screened resolvent of the Selling Laplacian of the inverse evolving
structure/uncertainty metric.  Reciprocal phase susceptibility is not an
after-the-fact gate: it is the local null authority of that Laplacian.  Thus a
phase-supported residual is transported toward the posterior, whereas
unsupported oscillation leaves a high-pass refusal behind.  A Z-style cavity
intersection prevents the joint smoothing proposal from certifying itself.

The returned trajectory is intentionally complete.  A numerical ceiling only
prevents an unresolved orbit from running forever; it is not a selected
denoising depth.  The experiment does not yet possess a truth-consistent
terminal law.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .causal_scale_transport_2d import (
    _screened_transport,
    causal_scale_transport_observation_2d,
)
from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
)
from .residual_erosion_transport_2d import _cavity_residual_relation
from .witnessed_characteristic_transport_2d import _validate


def _phase_action_authority(
    field: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the scale-action conditional reciprocal-phase authority."""
    _estimate, _residual, diagnostic = (
        causal_scale_transport_observation_2d(field))
    components = np.asarray(
        diagnostic["components_coarse_to_fine"], dtype=np.float64)
    susceptibility = np.asarray(
        diagnostic["phase_susceptibility_coarse_to_fine"],
        dtype=np.float64,
    )
    component_action = components * components
    total_action = np.sum(component_action, axis=0)
    weighted_action = np.sum(susceptibility * component_action, axis=0)
    authority = np.divide(
        weighted_action,
        total_action,
        out=np.zeros_like(total_action),
        where=total_action > np.finfo(float).tiny,
    )
    authority = np.clip(authority, 0.0, 1.0)
    global_action = float(np.sum(total_action))
    return authority, {
        "mean_authority": float(np.mean(authority)),
        "action_weighted_authority": (
            float(np.sum(weighted_action)) / global_action
            if global_action > np.finfo(float).tiny else 0.0
        ),
        "maximum_authority": float(np.max(authority)),
        "minimum_authority": float(np.min(authority)),
    }


def _phase_screened_smooth(
    field: np.ndarray,
    posterior: np.ndarray,
    residual: np.ndarray,
    phase_authority: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply one operator-normalized screened Selling smoothing action."""
    metric = continual_transport_metric(posterior, residual * residual)
    # Phase is a null direction of smoothing, not an empirical blend applied
    # after smoothing.  This keeps the operator positive and continuous.
    laplacian, _markov, stencil = _continual_flux_laplacian(
        metric, 1.0 - np.asarray(phase_authority, dtype=np.float64))
    maximum_degree = float(stencil["maximum_degree"])
    if maximum_degree <= 0.0:
        smoothed = np.asarray(field, dtype=np.float64).copy()
    else:
        smoothed = _screened_transport(
            laplacian,
            1.0 / maximum_degree,
            np.asarray(field, dtype=np.float64)[None, ...],
        )[0]
    rejected = np.asarray(field, dtype=np.float64) - smoothed
    return smoothed, {
        "maximum_degree": maximum_degree,
        "metric_condition_p90": float(metric["metric_condition_p90"]),
        "selling_reconstruction_error": float(
            stencil["selling_maximum_reconstruction_error"]),
        "laplacian_row_sum_error": float(
            stencil["laplacian_row_sum_error"]),
        "input_action": float(np.mean(field * field)),
        "smoothed_action": float(np.mean(smoothed * smoothed)),
        "rejected_action": float(np.mean(rejected * rejected)),
        "input_smoothed_cross_action": float(np.mean(smoothed * rejected)),
    }


def _conservation_error(
    posterior: np.ndarray,
    residual: np.ndarray,
    observation: np.ndarray,
) -> float:
    return float(np.max(np.abs(posterior + residual - observation)))


def conservative_exchange_cycle_2d(
    observation: np.ndarray,
    posterior: np.ndarray,
    residual: np.ndarray,
    *,
    observation_phase_authority: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one exact posterior -> residual -> joint exchange cycle."""
    y = _validate(observation)
    p = np.asarray(posterior, dtype=np.float64).copy()
    r = np.asarray(residual, dtype=np.float64).copy()
    if p.shape != y.shape or r.shape != y.shape:
        raise ValueError("posterior, residual, and observation must align")
    initial_error = _conservation_error(p, r, y)
    scale = max(float(np.max(np.abs(y))), 1.0)
    if initial_error > 64.0 * np.finfo(float).eps * scale:
        raise ValueError("posterior and residual must exactly recompose observation")

    posterior_phase, posterior_phase_diagnostic = (
        _phase_action_authority(p))
    smoothed_posterior, posterior_smoothing = _phase_screened_smooth(
        p, p, r, posterior_phase)
    posterior_shed = p - smoothed_posterior
    p = smoothed_posterior
    r = r + posterior_shed
    posterior_transfer_error = _conservation_error(p, r, y)

    residual_phase, residual_phase_diagnostic = _phase_action_authority(r)
    smoothed_residual, residual_smoothing = _phase_screened_smooth(
        r, p, r, residual_phase)
    # Positive smoothing supplies a possible return, but passing through the
    # smoother cannot certify amplitude.  Only its reciprocal-phase action is
    # transferred; the complement remains explicit residual ignorance.
    residual_donation = residual_phase * smoothed_residual
    residual_refusal = r - residual_donation
    p = p + residual_donation
    r = residual_refusal
    residual_transfer_error = _conservation_error(p, r, y)

    if observation_phase_authority is None:
        observation_phase, observation_phase_diagnostic = (
            _phase_action_authority(y))
    else:
        observation_phase = np.clip(
            np.asarray(observation_phase_authority, dtype=np.float64),
            0.0,
            1.0,
        )
        if observation_phase.shape != y.shape:
            raise ValueError("observation phase authority must align")
        observation_phase_diagnostic = {
            "status": "caller-supplied observation phase authority",
            "mean_authority": float(np.mean(observation_phase)),
        }
    joint_candidate, joint_smoothing = _phase_screened_smooth(
        p + r, p, r, observation_phase)
    joint_correction = joint_candidate - p
    closure_phase, closure_phase_diagnostic = _phase_action_authority(r)
    closure_metric = continual_transport_metric(p, r * r)
    closure_laplacian, _closure_markov, closure_stencil = (
        _continual_flux_laplacian(closure_metric, 1.0 - closure_phase))
    _raw_closure, closure_relation = _cavity_residual_relation(
        p,
        joint_correction,
        closure_laplacian,
        float(closure_stencil["maximum_degree"]),
    )
    # Phase and target-excluded curvature are independent necessary witnesses.
    # Their Hellinger product is a continuous intersection.  Passing either
    # witness alone never certifies the smoothed joint correction.
    closure_authority = np.sqrt(
        closure_phase * np.asarray(closure_relation["explained_action"]))
    admitted_joint_correction = closure_authority * joint_correction
    p = p + admitted_joint_correction
    r = y - p
    joint_transfer_error = _conservation_error(p, r, y)

    return p, r, {
        "posterior_shed_action": float(np.mean(
            posterior_shed * posterior_shed)),
        "residual_donation_action": float(np.mean(
            residual_donation * residual_donation)),
        "smoothed_residual_action": float(np.mean(
            smoothed_residual * smoothed_residual)),
        "residual_refusal_action": float(np.mean(r * r)),
        "joint_candidate_action": float(np.mean(
            joint_correction * joint_correction)),
        "joint_reassignment_action": float(np.mean(
            admitted_joint_correction * admitted_joint_correction)),
        "initial_conservation_error": initial_error,
        "posterior_transfer_conservation_error": posterior_transfer_error,
        "residual_transfer_conservation_error": residual_transfer_error,
        "joint_transfer_conservation_error": joint_transfer_error,
        "posterior_phase": posterior_phase_diagnostic,
        "residual_phase": residual_phase_diagnostic,
        "closure_phase": closure_phase_diagnostic,
        "closure_mean_authority": float(np.mean(closure_authority)),
        "closure_action_weighted_authority": float(np.divide(
            np.sum(closure_authority * joint_correction * joint_correction),
            np.sum(joint_correction * joint_correction),
            out=np.asarray(0.0),
            where=np.sum(joint_correction * joint_correction)
            > np.finfo(float).tiny,
        )),
        "closure_mean_curvature_authority": float(
            closure_relation["mean_positive_relation_authority"]),
        "observation_phase": observation_phase_diagnostic,
        "posterior_smoothing": posterior_smoothing,
        "residual_smoothing": residual_smoothing,
        "joint_smoothing": joint_smoothing,
    }


def denoise_conservative_exchange_transport_2d(
    observation: np.ndarray,
    *,
    initial_posterior: np.ndarray | None = None,
    numerical_cycle_ceiling: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Trace the conservative exchange orbit until equilibrium or ceiling."""
    image = _validate(observation)
    ceiling = int(numerical_cycle_ceiling)
    if ceiling < 1:
        raise ValueError("numerical cycle ceiling must be positive")

    if initial_posterior is None:
        _estimate, _residual, base = (
            causal_scale_transport_observation_2d(image))
        posterior = np.asarray(
            base["readouts"]["phase_susceptibility"],
            dtype=np.float64,
        ).copy()
    else:
        posterior = _validate(initial_posterior).copy()
        if posterior.shape != image.shape:
            raise ValueError("initial posterior must align with observation")
        base = {"status": "caller-supplied initial posterior"}
    residual = image - posterior
    observation_phase, observation_phase_diagnostic = (
        _phase_action_authority(image))

    posterior_trajectory = [posterior.copy()]
    residual_trajectory = [residual.copy()]
    cycles: list[dict[str, Any]] = []
    equilibrium = False
    scale_action = max(float(np.mean(image * image)), 1.0)
    tolerance = (64.0 * np.finfo(float).eps) ** 2 * scale_action
    for cycle in range(ceiling):
        preceding = posterior.copy()
        posterior, residual, record = conservative_exchange_cycle_2d(
            image,
            posterior,
            residual,
            observation_phase_authority=observation_phase,
        )
        displacement_action = float(np.mean(
            (posterior - preceding) ** 2))
        record = {
            "cycle": int(cycle + 1),
            "posterior_displacement_action": displacement_action,
            "posterior_action": float(np.mean(posterior * posterior)),
            "residual_action": float(np.mean(residual * residual)),
            **record,
        }
        cycles.append(record)
        posterior_trajectory.append(posterior.copy())
        residual_trajectory.append(residual.copy())
        if displacement_action <= tolerance:
            equilibrium = True
            break

    maximum_conservation_error = float(max(
        _conservation_error(p, r, image)
        for p, r in zip(posterior_trajectory, residual_trajectory)
    ))
    return posterior, {
        "status": (
            "conservative exchange equilibrium"
            if equilibrium
            else "numerical cycle ceiling reached; orbit unresolved"
        ),
        "theory_status": (
            "exact two-reservoir conservation and positive phase-screened "
            "Selling resolvents; truth-consistent terminal law unresolved"
        ),
        "equilibrium": bool(equilibrium),
        "numerical_cycle_ceiling_hit": bool(not equilibrium),
        "completed_cycles": int(len(cycles)),
        "maximum_conservation_error": maximum_conservation_error,
        "posterior_trajectory": np.stack(posterior_trajectory),
        "residual_trajectory": np.stack(residual_trajectory),
        "cycles": tuple(cycles),
        "base": base,
        "observation_phase": observation_phase_diagnostic,
        "laws": {
            "posterior_to_residual": "r <- r + (p - S_p p); p <- S_p p",
            "residual_to_posterior": (
                "d_r <- chi_phase S_r r; p <- p+d_r; r <- r-d_r"
            ),
            "joint_closure": (
                "c <- S_y(p+r)-p; a <- sqrt(chi_phase chi_cavity); "
                "p <- p+ac; r <- y-p"
            ),
            "conservation": "p+r=y pointwise after every substep",
            "screening": "S=(I+L/max(diag(L)))^-1",
            "phase_null_authority": (
                "L is the Selling flux under one minus action-conditional "
                "reciprocal phase susceptibility"
            ),
        },
    }


__all__ = [
    "conservative_exchange_cycle_2d",
    "denoise_conservative_exchange_transport_2d",
]
