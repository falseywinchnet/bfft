"""Two-coordinate endpoint contractor on the lifted scale-moment state.

For the affine scale action ``a(s)=u_fine*(1-s)+u_coarse*s``, the transfer is

    t = u_fine * (m0-m1) + u_coarse * m1.

Only two coefficient fields are required. Joint posterior/residual normal
slabs are used as competing action witnesses rather than mistaken for a truth
set. Residual-normal violation supports moving an ancestry component into the
posterior; posterior-normal violation supports leaving it as noise. Their
nonnegative squared actions are pulled back to vertices. Their four raw action
fields are transported through the positive eikonal resolvent without first
being divided into scalar endpoint decisions. The first conservative
posterior/residual barycentre supplies a second metric; that metric carries the
first action measure while a second target-excluded normal observation is
made. The second observation enters the carried action only where the current
phase/scale explanation is rejected and where both its complete direction and
individual component overlap the carried measure. Between-cycle variance is
retained independently. A dimensionally matched stopping certificate compares
that radiance-fourth disagreement with squared unresolved endpoint variance
and contracts the complete second endpoint toward the first. Only then are the
fine and coarse endpoint fractions read out.

The rejected hard-contraction control collapsed toward the zero-action
stability component and is not retained in the runtime path. This remains an
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
from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
)
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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return four raw transported endpoint-action fields."""
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
    diagnostic["mean_transported_fine_support_action"] = float(np.mean(
        fine_support))
    diagnostic["mean_transported_fine_noise_action"] = float(np.mean(
        fine_noise))
    diagnostic["mean_transported_coarse_support_action"] = float(np.mean(
        coarse_support))
    diagnostic["mean_transported_coarse_noise_action"] = float(np.mean(
        coarse_noise))
    return transported, diagnostic


def _endpoint_fractions(
    raw_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Read fine/coarse support fractions without discarding raw actions."""
    action = np.asarray(raw_action, dtype=np.float64)
    if action.ndim != 3 or action.shape[0] != 4:
        raise ValueError("endpoint action must have four HxW fields")
    fine_support, fine_noise, coarse_support, coarse_noise = action
    magnitude = max(float(np.max(action)), 1.0)
    floor = np.finfo(float).eps * magnitude

    def fraction(support: np.ndarray, noise: np.ndarray) -> np.ndarray:
        total = support + noise
        # Absence of contrary evidence retains the observation. This is the
        # Back-to-Basics identity default, not a half-probability convention.
        return np.divide(
            support,
            total,
            out=np.ones_like(total),
            where=total > floor,
        )

    fine = fraction(fine_support, fine_noise)
    coarse = fraction(coarse_support, coarse_noise)
    return fine, coarse, {
        "identity_default_fraction_fine": float(np.mean(
            fine_support + fine_noise <= floor)),
        "identity_default_fraction_coarse": float(np.mean(
            coarse_support + coarse_noise <= floor)),
        "mean_fine_action_mass": float(np.mean(
            fine_support + fine_noise)),
        "mean_coarse_action_mass": float(np.mean(
            coarse_support + coarse_noise)),
    }


def _transported_scale_support(raw_moments: np.ndarray) -> np.ndarray:
    """Read continuous-scale support only after raw-moment transport."""
    moments = np.asarray(raw_moments, dtype=np.float64)
    if moments.ndim != 3 or moments.shape[0] != 10:
        raise ValueError("scale support expects ten transported raw moments")
    absolute_mass = moments[2] + moments[7]
    absolute_first = moments[3] + moments[8]
    support = np.divide(
        absolute_first,
        absolute_mass,
        out=np.zeros_like(absolute_mass),
        where=absolute_mass > np.finfo(float).tiny,
    )
    return np.clip(support, 0.0, 1.0)


def _hadamard_endpoint_intersection(
    normal_support: np.ndarray,
    phase_support: np.ndarray,
    phase_context: float,
    scale_support: np.ndarray,
) -> np.ndarray:
    """Intersect four bounded support/rejection measures symmetrically."""
    support = (
        phase_support
        * float(phase_context)
        * scale_support
        * normal_support
    )
    rejection = (
        (1.0 - phase_support)
        * (1.0 - float(phase_context))
        * (1.0 - scale_support)
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


def _causal_temporal_action_fusion(
    carried_action: np.ndarray,
    observed_action: np.ndarray,
    phase_support: np.ndarray,
    phase_context: float,
    scale_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Admit a second action only through witnessed rejection and overlap.

    Equal pooling mistakes two correlated temporal observations for exchangeable
    samples. Here the carried action is causal state. The new action receives
    authority only where the current phase/scale explanation is rejected, its
    four-channel direction agrees with the carried action, and the particular
    action component is present in both observations.
    """
    first = np.maximum(np.asarray(carried_action, dtype=np.float64), 0.0)
    second = np.maximum(np.asarray(observed_action, dtype=np.float64), 0.0)
    if first.shape != second.shape or first.ndim != 3 or first.shape[0] != 4:
        raise ValueError("temporal fusion expects aligned four-field actions")
    phase = np.clip(np.asarray(phase_support, dtype=np.float64), 0.0, 1.0)
    scale = np.clip(np.asarray(scale_support, dtype=np.float64), 0.0, 1.0)
    if phase.shape != first.shape[1:] or scale.shape != phase.shape:
        raise ValueError("temporal support fields must align with action")
    context = float(np.clip(phase_context, 0.0, 1.0))
    structural_support = phase * context * scale
    structural_rejection = (
        (1.0 - phase) * (1.0 - context) * (1.0 - scale))
    structural_total = structural_support + structural_rejection
    floor = np.finfo(float).eps * max(
        float(np.max(first * first + second * second)),
        float(np.max(structural_total)),
        1.0,
    )
    structure_authority = np.divide(
        structural_support,
        structural_total,
        out=np.ones_like(structural_total),
        where=structural_total > floor,
    )
    first_norm = np.sqrt(np.sum(first * first, axis=0))
    second_norm = np.sqrt(np.sum(second * second, axis=0))
    norm_product = first_norm * second_norm
    directional_overlap = np.divide(
        np.sum(first * second, axis=0),
        norm_product,
        out=np.zeros_like(norm_product),
        where=norm_product > floor,
    )
    square_sum = first * first + second * second
    component_overlap = np.divide(
        2.0 * first * second,
        square_sum,
        out=np.zeros_like(first),
        where=square_sum > floor,
    )
    second_authority = (
        (1.0 - structure_authority)[None, ...]
        * directional_overlap[None, ...]
        * component_overlap
    )
    fused = first + second_authority * second
    # This remains the exact variance of the equal normalized counting measure,
    # even though its mean has been rejected as the causal point estimator.
    variance = 0.25 * (first - second) ** 2
    return fused, variance, {
        "mean_structural_rejection_authority": float(np.mean(
            1.0 - structure_authority)),
        "mean_directional_overlap": float(np.mean(directional_overlap)),
        "mean_component_overlap": float(np.mean(component_overlap)),
        "mean_second_action_authority": float(np.mean(second_authority)),
        "maximum_second_action_authority": float(np.max(second_authority)),
    }


def _temporal_stopping_certificate(
    endpoint_action_variance: np.ndarray,
    between_cycle_action_variance: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return dimensionally matched authority for the second endpoint action.

    Endpoint uncertainty has radiance-squared units. Its square is compared
    with the four-action temporal disagreement, which has radiance-fourth
    units. Continuation is therefore a normalized contraction coordinate, not
    a tolerance or iteration counter.
    """
    endpoint_variance = np.maximum(
        np.asarray(endpoint_action_variance, dtype=np.float64), 0.0)
    temporal_variance = np.maximum(
        np.asarray(between_cycle_action_variance, dtype=np.float64), 0.0)
    if (endpoint_variance.ndim != 2 or temporal_variance.shape
            != (4,) + endpoint_variance.shape):
        raise ValueError("stopping certificate expects one endpoint and four actions")
    unresolved_fourth = endpoint_variance * endpoint_variance
    temporal_disagreement = np.mean(temporal_variance, axis=0)
    total = unresolved_fourth + temporal_disagreement
    floor = np.finfo(float).eps * max(float(np.max(total)), 1.0)
    continuation = np.divide(
        unresolved_fourth,
        total,
        out=np.ones_like(total),
        where=total > floor,
    )
    return continuation, {
        "mean_continuation_authority": float(np.mean(continuation)),
        "minimum_continuation_authority": float(np.min(continuation)),
        "maximum_continuation_authority": float(np.max(continuation)),
        "mean_unresolved_endpoint_fourth_moment": float(np.mean(
            unresolved_fourth)),
        "mean_temporal_action_disagreement": float(np.mean(
            temporal_disagreement)),
    }


def denoise_lifted_endpoint_action_transport_2d(
    observation: np.ndarray,
    *,
    initial_posterior: np.ndarray | None = None,
    trace_refinement: int = 0,
    return_temporal_ablation: bool = False,
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
    first_raw_action, first_action_diagnostic = (
        _transported_endpoint_evidence(
            posterior,
            residual,
            fine_basis,
            coarse_basis,
            graph_laplacian,
            maximum_degree,
        ))
    first_fine, first_coarse, first_fraction_diagnostic = (
        _endpoint_fractions(first_raw_action))
    provisional_transfer = (
        first_fine * fine_basis + first_coarse * coarse_basis)
    provisional_posterior = posterior + provisional_transfer
    provisional_residual = image - provisional_posterior
    # The barycentre defines the next observation geometry, but its unresolved
    # Bernoulli action variance remains in the metric. Fine/coarse variances
    # are separate mixture components; no independence cross term is asserted.
    endpoint_action_variance = (
        first_fine * (1.0 - first_fine) * fine_basis * fine_basis
        + first_coarse * (1.0 - first_coarse)
        * coarse_basis * coarse_basis
    )
    second_metric = continual_transport_metric(
        provisional_posterior,
        provisional_residual * provisional_residual
        + np.asarray(lifted["transport_uncertainty"])
        + endpoint_action_variance,
    )
    second_laplacian, _second_markov, second_graph_diagnostic = (
        _continual_flux_laplacian(second_metric, np.ones_like(image)))
    second_maximum_degree = float(
        second_graph_diagnostic["maximum_degree"])
    carried_first_action = (
        _screened_transport(
            second_laplacian,
            1.0 / second_maximum_degree,
            first_raw_action,
        )
        if second_maximum_degree > 0.0 else first_raw_action.copy()
    )
    first_action_mass = np.sum(first_raw_action, axis=(1, 2))
    carried_action_mass = np.sum(carried_first_action, axis=(1, 2))
    carried_action_mass_error = float(np.max(np.abs(
        carried_action_mass - first_action_mass
    )))
    second_raw_observation, second_action_diagnostic = (
        _transported_endpoint_evidence(
            provisional_posterior,
            provisional_residual,
            fine_basis,
            coarse_basis,
            second_laplacian,
            second_maximum_degree,
        ))
    if "components_coarse_to_fine" in lifted["base"]:
        observation_phase, observation_phase_diagnostic = (
            _phase_authority_from_scale_diagnostic(lifted["base"]))
    else:
        observation_phase, observation_phase_diagnostic = (
            _phase_action_authority(image))
    first_pushed_moments = np.asarray(lifted["pushed_vertex_lift"])
    first_scale_support = _transported_scale_support(first_pushed_moments)
    second_pushed_moments = (
        _screened_transport(
            second_laplacian,
            1.0 / second_maximum_degree,
            first_pushed_moments,
        )
        if second_maximum_degree > 0.0 else first_pushed_moments.copy()
    )
    transported_scale_support = _transported_scale_support(
        second_pushed_moments)
    context = float(observation_phase_diagnostic[
        "action_weighted_authority"])
    retained_raw_action, between_cycle_action_variance, temporal_diagnostic = (
        _causal_temporal_action_fusion(
            carried_first_action,
            second_raw_observation,
            observation_phase,
            context,
            transported_scale_support,
        ))
    normal_fine, normal_coarse, retained_fraction_diagnostic = (
        _endpoint_fractions(retained_raw_action))
    # Fuse the bounded support/noise measures by their normalized Hadamard
    # intersection.  This is a logical agreement law, not an independence
    # claim: phase cannot override normal rejection, and normal support cannot
    # override phase incoherence.  Exact contradictory endpoints retain the
    # Back-to-Basics identity branch rather than inventing a half decision.
    first_cycle_fine_endpoint = _hadamard_endpoint_intersection(
        first_fine, observation_phase, context, first_scale_support)
    first_cycle_coarse_endpoint = _hadamard_endpoint_intersection(
        first_coarse, observation_phase, context, first_scale_support)
    first_cycle_transfer = (
        first_cycle_fine_endpoint * fine_basis
        + first_cycle_coarse_endpoint * coarse_basis)
    first_cycle_estimate = posterior + first_cycle_transfer
    unstopped_fine_endpoint = _hadamard_endpoint_intersection(
        normal_fine,
        observation_phase,
        context,
        transported_scale_support,
    )
    unstopped_coarse_endpoint = _hadamard_endpoint_intersection(
        normal_coarse,
        observation_phase,
        context,
        transported_scale_support,
    )
    continuation, stopping_diagnostic = _temporal_stopping_certificate(
        endpoint_action_variance, between_cycle_action_variance)
    fine_endpoint = first_cycle_fine_endpoint + continuation * (
        unstopped_fine_endpoint - first_cycle_fine_endpoint)
    coarse_endpoint = first_cycle_coarse_endpoint + continuation * (
        unstopped_coarse_endpoint - first_cycle_coarse_endpoint)
    unstopped_transfer = (
        unstopped_fine_endpoint * fine_basis
        + unstopped_coarse_endpoint * coarse_basis)
    unstopped_estimate = posterior + unstopped_transfer
    transfer = fine_endpoint * fine_basis + coarse_endpoint * coarse_basis
    estimate = posterior + transfer
    remaining = image - estimate
    diagnostic: dict[str, Any] = {
        "status": (
            "two-coordinate affine-scale estimator from positively "
            "transported causal two-cycle action fusion with a dimensional "
            "stopping certificate"
        ),
        "posterior_before_endpoint_action": posterior,
        "residual_before_endpoint_action": residual,
        "transfer": transfer,
        "remaining_residual": remaining,
        "fine_endpoint": fine_endpoint,
        "coarse_endpoint": coarse_endpoint,
        "fine_basis": fine_basis,
        "coarse_basis": coarse_basis,
        "first_cycle_estimate": first_cycle_estimate,
        "first_cycle_fine_endpoint": first_cycle_fine_endpoint,
        "first_cycle_coarse_endpoint": first_cycle_coarse_endpoint,
        "provisional_action_barycentre": provisional_transfer,
        "provisional_posterior": provisional_posterior,
        "provisional_residual": provisional_residual,
        "provisional_recomposition_error": float(np.max(np.abs(
            provisional_posterior + provisional_residual - image
        ))),
        "endpoint_action_variance": endpoint_action_variance,
        "retained_raw_endpoint_action": retained_raw_action,
        "between_cycle_action_variance": between_cycle_action_variance,
        "endpoint_action_evidence": {
            "first_cycle": {
                **first_action_diagnostic,
                **first_fraction_diagnostic,
                "carried_action_mass_conservation_error": (
                    carried_action_mass_error),
            },
            "second_cycle": {
                **second_action_diagnostic,
            },
            "causal_temporal_fusion": {
                **retained_fraction_diagnostic,
                **temporal_diagnostic,
                **stopping_diagnostic,
                "mean_between_cycle_action_variance": float(np.mean(
                    between_cycle_action_variance)),
                "maximum_between_cycle_action_variance": float(np.max(
                    between_cycle_action_variance)),
            },
        },
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
        "retained_action_coordinate_count": 4,
        "retained_action_uncertainty_coordinate_count": 4,
        "two_cycle_persistent_dimension": 22,
        "mean_first_cycle_fine_endpoint": float(np.mean(
            first_cycle_fine_endpoint)),
        "mean_first_cycle_coarse_endpoint": float(np.mean(
            first_cycle_coarse_endpoint)),
        "first_cycle_observation_recomposition_error": float(np.max(np.abs(
            first_cycle_estimate + (image - first_cycle_estimate) - image
        ))),
        "second_graph": second_graph_diagnostic,
        "lifted": lifted,
    }
    if return_temporal_ablation:
        # Research instrumentation only: these fixed four-coordinate fields
        # are omitted from the default returned state to keep memory compact.
        diagnostic["carried_first_endpoint_action"] = carried_first_action
        diagnostic["second_observed_endpoint_action"] = second_raw_observation
        diagnostic["unstopped_second_cycle_estimate"] = unstopped_estimate
        diagnostic["temporal_continuation_authority"] = continuation
    return estimate, diagnostic


__all__ = ["denoise_lifted_endpoint_action_transport_2d"]
