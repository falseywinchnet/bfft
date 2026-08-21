"""Two-coordinate endpoint contractor on the lifted scale-moment state.

For the affine scale action ``a(s)=u_fine*(1-s)+u_coarse*s``, the transfer is

    t = u_fine * (m0-m1) + u_coarse * m1.

Only two coefficient fields are required. Joint posterior/residual normal
slabs are used as competing action witnesses rather than mistaken for a truth
set. Residual-normal violation supports moving an ancestry component into the
posterior; posterior-normal violation supports leaving it as noise. Their
nonnegative squared actions are pulled back to vertices, positively transported
through the eikonal graph, and normalized only after transport.

The rejected hard-contraction control collapsed toward the zero-action
stability component and is not retained in the runtime path. This is the first
point estimator built on the compact lift. It remains an
experiment: local normal slabs are target-excluded but not statistically
independent, and affine scale response is a second-moment closure rather than
the complete lineage-edge zonotope.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .causal_scale_transport_2d import _screened_transport
from .conservative_exchange_transport_2d import _phase_action_authority
from .joint_value_jet_zonotope_contractor_2d import (
    _covariance_normal_constraints,
    _crossfit_value_jet_constraints,
)
from .continual_eikonal_noise_transport_2d import _continual_flux_laplacian
from .lifted_scale_moment_transport_2d import (
    lifted_scale_moment_transport_state_2d,
)
from .witnessed_characteristic_transport_2d import _validate


def _phase_authority_from_scale_diagnostic(
    diagnostic: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """Reuse an existing causal-scale trace instead of recomputing it."""
    components = np.asarray(
        diagnostic["components_coarse_to_fine"], dtype=np.float64)
    susceptibility = np.asarray(
        diagnostic["phase_susceptibility_coarse_to_fine"], dtype=np.float64)
    action = components * components
    total = np.sum(action, axis=0)
    weighted = np.sum(susceptibility * action, axis=0)
    authority = np.divide(
        weighted,
        total,
        out=np.zeros_like(total),
        where=total > np.finfo(float).tiny,
    )
    authority = np.clip(authority, 0.0, 1.0)
    global_action = float(np.sum(total))
    return authority, {
        "mean_authority": float(np.mean(authority)),
        "action_weighted_authority": (
            float(np.sum(weighted)) / global_action
            if global_action > np.finfo(float).tiny else 0.0
        ),
        "maximum_authority": float(np.max(authority)),
        "minimum_authority": float(np.min(authority)),
    }


def _pulled_slab_gap(
    operator: sparse.csr_matrix,
    action: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Pull squared normal-violation action back to image vertices."""
    matrix = sparse.csr_matrix(operator, dtype=np.float64)
    coordinate = np.asarray(matrix @ np.asarray(action).reshape(-1)).ravel()
    gap = np.maximum(lower - coordinate, 0.0) + np.maximum(
        coordinate - upper, 0.0)
    square_adjoint = matrix.copy()
    square_adjoint.data *= square_adjoint.data
    pulled = np.asarray(square_adjoint.T @ (gap * gap)).ravel()
    mass = np.asarray(square_adjoint.T @ np.ones(gap.size)).ravel()
    local_action = np.divide(
        pulled,
        mass,
        out=np.zeros_like(pulled),
        where=mass > np.finfo(float).tiny,
    )
    return local_action, {
        "mean_gap": float(np.mean(gap)),
        "mean_squared_gap": float(np.mean(gap * gap)),
        "violated_constraint_fraction": float(np.mean(gap > 0.0)),
    }


def _transported_endpoint_evidence(
    posterior: np.ndarray,
    residual: np.ndarray,
    fine_basis: np.ndarray,
    coarse_basis: np.ndarray,
    graph_laplacian: sparse.csr_matrix,
    maximum_degree: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return fine/coarse support fractions from competing normal actions."""
    _q_r, _lo_r, _hi_r, rectangle_r = (
        _crossfit_value_jet_constraints(residual))
    _q_p, _lo_p, _hi_p, rectangle_p = (
        _crossfit_value_jet_constraints(posterior))
    h_r, lower_r, upper_r, _normal_r = _covariance_normal_constraints(
        residual, rectangle_r, additive_transfer=False)
    h_p, lower_p, upper_p, _normal_p = _covariance_normal_constraints(
        posterior, rectangle_p, additive_transfer=True)
    evidence = []
    diagnostic: dict[str, Any] = {}
    for name, basis in (("fine", fine_basis), ("coarse", coarse_basis)):
        # If removing this basis makes the residual leave its witnessed body,
        # the basis has support evidence.  If adding it makes the posterior
        # leave its witnessed body, it has noise evidence.
        support, support_diagnostic = _pulled_slab_gap(
            h_r, basis, lower_r, upper_r)
        noise, noise_diagnostic = _pulled_slab_gap(
            h_p, basis, lower_p, upper_p)
        evidence.extend((support.reshape(posterior.shape),
                         noise.reshape(posterior.shape)))
        diagnostic[name] = {
            "support": support_diagnostic,
            "noise": noise_diagnostic,
        }
    raw = np.stack(evidence)
    transported = (
        _screened_transport(
            graph_laplacian, 1.0 / maximum_degree, raw)
        if maximum_degree > 0.0 else raw
    )
    transported = np.maximum(transported, 0.0)
    fine_support, fine_noise, coarse_support, coarse_noise = transported
    magnitude = max(float(np.max(transported)), 1.0)
    floor = np.finfo(float).eps * magnitude

    def fraction(support: np.ndarray, noise: np.ndarray) -> np.ndarray:
        total = support + noise
        # Absence of contrary evidence retains the observation.  This is the
        # Back-to-Basics identity default, not a half-probability convention.
        return np.divide(
            support,
            total,
            out=np.ones_like(total),
            where=total > floor,
        )

    fine = fraction(fine_support, fine_noise)
    coarse = fraction(coarse_support, coarse_noise)
    diagnostic["mean_transported_fine_support_action"] = float(np.mean(
        fine_support))
    diagnostic["mean_transported_fine_noise_action"] = float(np.mean(
        fine_noise))
    diagnostic["mean_transported_coarse_support_action"] = float(np.mean(
        coarse_support))
    diagnostic["mean_transported_coarse_noise_action"] = float(np.mean(
        coarse_noise))
    diagnostic["identity_default_fraction_fine"] = float(np.mean(
        fine_support + fine_noise <= floor))
    diagnostic["identity_default_fraction_coarse"] = float(np.mean(
        coarse_support + coarse_noise <= floor))
    return fine, coarse, diagnostic


def denoise_lifted_endpoint_action_transport_2d(
    observation: np.ndarray,
    *,
    initial_posterior: np.ndarray | None = None,
    trace_refinement: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Denoise by transported competition of support and noise actions."""
    image = _validate(observation)
    lifted = lifted_scale_moment_transport_state_2d(
        image,
        initial_posterior=initial_posterior,
        trace_refinement=trace_refinement,
    )
    posterior = np.asarray(lifted["posterior_after_erosion"])
    residual = np.asarray(lifted["residual_after_erosion"])
    moments = np.asarray(lifted["vertex_lift"])
    signed_zeroth = moments[0] + moments[5]
    signed_first = moments[1] + moments[6]
    fine_basis = signed_zeroth - signed_first
    coarse_basis = signed_first
    metric = lifted["metric"]
    graph_laplacian, _markov, graph_diagnostic = _continual_flux_laplacian(
        metric, np.ones_like(image))
    maximum_degree = float(graph_diagnostic["maximum_degree"])
    fine_endpoint, coarse_endpoint, action_evidence = (
        _transported_endpoint_evidence(
            posterior,
            residual,
            fine_basis,
            coarse_basis,
            graph_laplacian,
            maximum_degree,
        ))
    if "components_coarse_to_fine" in lifted["base"]:
        observation_phase, observation_phase_diagnostic = (
            _phase_authority_from_scale_diagnostic(lifted["base"]))
    else:
        observation_phase, observation_phase_diagnostic = (
            _phase_action_authority(image))
    pushed_moments = np.asarray(lifted["pushed_vertex_lift"])
    absolute_mass = pushed_moments[2] + pushed_moments[7]
    absolute_first = pushed_moments[3] + pushed_moments[8]
    transported_scale_support = np.divide(
        absolute_first,
        absolute_mass,
        out=np.zeros_like(absolute_mass),
        where=absolute_mass > np.finfo(float).tiny,
    )
    transported_scale_support = np.clip(
        transported_scale_support, 0.0, 1.0)
    # Fuse the two bounded support/noise measures by their normalized Hadamard
    # intersection.  This is a logical agreement law, not an independence
    # claim: phase cannot override normal rejection, and normal support cannot
    # override phase incoherence.  Exact contradictory endpoints retain the
    # Back-to-Basics identity branch rather than inventing a half decision.
    def intersect_support(normal_support: np.ndarray) -> np.ndarray:
        context = float(observation_phase_diagnostic[
            "action_weighted_authority"])
        support = (
            observation_phase
            * context
            * transported_scale_support
            * normal_support
        )
        rejection = (
            (1.0 - observation_phase)
            * (1.0 - context)
            * (1.0 - transported_scale_support)
            * (1.0 - normal_support)
        )
        total = support + rejection
        floor = np.finfo(float).eps * max(float(np.max(total)), 1.0)
        return np.divide(
            support,
            total,
            out=np.ones_like(total),
            where=total > floor,
        )

    fine_endpoint = intersect_support(fine_endpoint)
    coarse_endpoint = intersect_support(coarse_endpoint)
    transfer = fine_endpoint * fine_basis + coarse_endpoint * coarse_basis
    estimate = posterior + transfer
    remaining = image - estimate
    return estimate, {
        "status": (
            "two-coordinate affine-scale estimator from positively "
            "transported support/noise normal-action competition"
        ),
        "posterior_before_endpoint_action": posterior,
        "residual_before_endpoint_action": residual,
        "transfer": transfer,
        "remaining_residual": remaining,
        "fine_endpoint": fine_endpoint,
        "coarse_endpoint": coarse_endpoint,
        "fine_basis": fine_basis,
        "coarse_basis": coarse_basis,
        "endpoint_action_evidence": action_evidence,
        "observation_phase_authority": observation_phase,
        "observation_phase": observation_phase_diagnostic,
        "transported_scale_support": transported_scale_support,
        "mean_transported_scale_support": float(np.mean(
            transported_scale_support)),
        "observation_recomposition_error": float(np.max(np.abs(
            estimate + remaining - image
        ))),
        "mean_fine_endpoint": float(np.mean(fine_endpoint)),
        "mean_coarse_endpoint": float(np.mean(coarse_endpoint)),
        "mean_absolute_transfer": float(np.mean(np.abs(transfer))),
        "lifted": lifted,
    }


__all__ = ["denoise_lifted_endpoint_action_transport_2d"]
