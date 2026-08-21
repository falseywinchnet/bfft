"""Fixed-dimensional transport closure for continuous-scale ancestry.

The expanded local zonotope assigns a coefficient to every
``(scale lineage, Selling edge)`` pair.  That representation is useful as an
oracle but its pushed edge-response map is quadratic in image size.

This module retains each ancestry as five raw fields over a canonical
continuous scale coordinate ``s in [0,1]``:

``m0 = sum_g x_g``
    signed ancestry, hence exact scene recomposition;
``m1 = sum_g s_g x_g``
    signed first scale moment, enabling an affine continuous-scale action;
``a0 = sum_g |x_g|``
    total variation mass over scale;
``a1 = sum_g s_g |x_g|``
    first scale moment;
``a2 = sum_g s_g^2 |x_g|``
    second scale moment.

The inherited-residual and posterior-shed ancestries therefore occupy ten
channels independent of scale refinement.  Positive linear transport acts on
the raw moments directly.  Scale mean, scale variance, sign cancellation, and
transport uncertainty are read out only afterward, so nonlinear ratios are
never averaged as if they were sufficient statistics.

The lift does not yet select a denoised point.  It is a fixed-dimensional
state on which the joint posterior/residual normal contractor can act without
materializing lineage-edge coefficients or dense edge responses.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from .causal_scale_transport_2d import (
    _screened_transport,
    causal_scale_transport_observation_2d,
)
from .conservative_exchange_transport_2d import (
    _phase_action_authority,
    _phase_screened_smooth,
)
from .continuous_scale_zonotope_transport_2d import (
    _continuous_scale_lineage_generators,
)
from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
)
from .joint_value_jet_zonotope_contractor_2d import (
    _covariance_normal_constraints,
    _crossfit_value_jet_constraints,
)
from .witnessed_characteristic_transport_2d import _validate


_RAW_MOMENT_NAMES = (
    "signed_zeroth",
    "signed_first_scale",
    "absolute_zeroth",
    "absolute_first_scale",
    "absolute_second_scale",
)


def _canonical_scale_coordinate(
    labels: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Map lineage transport times continuously to ``[0,1]``.

    Heat increments use the midpoint after the intrinsic ``log1p`` chart of
    their two nonnegative transport times.  This remains finite at transport
    time zero.  The coarse endpoint is the terminal scale.  The chart makes
    multiplicative heat-time refinement affine without introducing a selected
    band or scale constant.
    """
    midpoint = np.zeros(len(labels), dtype=np.float64)
    heat = np.zeros(len(labels), dtype=bool)
    for ordinal, label in enumerate(labels):
        if label["kind"] != "heat_increment":
            continue
        coarse = max(float(label["transport_time_coarse"]), 0.0)
        fine = max(float(label["transport_time_fine"]), 0.0)
        midpoint[ordinal] = 0.5 * (np.log1p(coarse) + np.log1p(fine))
        heat[ordinal] = True
    terminal = float(np.max(midpoint)) if np.any(heat) else 0.0
    coordinate = np.zeros_like(midpoint)
    if terminal > 0.0:
        coordinate[heat] = midpoint[heat] / terminal
    coordinate[~heat] = 1.0
    return np.clip(coordinate, 0.0, 1.0)


def _raw_scale_moment_lift(
    lineage: np.ndarray,
    labels: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Collapse any number of lineage columns into five sufficient fields."""
    matrix = np.asarray(lineage, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise ValueError("lineage columns and scale labels must align")
    scale = _canonical_scale_coordinate(labels)
    magnitude = np.abs(matrix)
    lift = np.stack((
        np.sum(matrix, axis=1),
        matrix @ scale,
        np.sum(magnitude, axis=1),
        magnitude @ scale,
        magnitude @ (scale * scale),
    ))
    return lift, {
        "lineage_count": int(matrix.shape[1]),
        "scale_coordinate": scale,
        "signed_recomposition_error": float(np.max(np.abs(
            lift[0] - np.sum(matrix, axis=1)
        ))) if matrix.size else 0.0,
    }


def _moment_readouts(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Recover nonlinear scale and cancellation statistics from raw fields."""
    lift = np.asarray(raw, dtype=np.float64)
    if lift.ndim < 2 or lift.shape[0] != 5:
        raise ValueError("one ancestry lift must begin with five raw moments")
    signed, signed_first, absolute, first, second = lift
    tiny = np.finfo(float).tiny
    mean = np.divide(
        first, absolute, out=np.zeros_like(first), where=absolute > tiny)
    second_mean = np.divide(
        second, absolute, out=np.zeros_like(second), where=absolute > tiny)
    variance = np.maximum(second_mean - mean * mean, 0.0)
    coherence = np.divide(
        np.abs(signed), absolute,
        out=np.ones_like(absolute),
        where=absolute > tiny,
    )
    coherence = np.clip(coherence, 0.0, 1.0)
    # Both terms have squared-radiance units.  The first is unresolved sign
    # cancellation; the second is spread over the intrinsic scale fibre.
    uncertainty = np.maximum(
        absolute * absolute - signed * signed, 0.0
    ) + absolute * absolute * variance
    return {
        "signed_first_scale": signed_first,
        "scale_mean": mean,
        "scale_variance": variance,
        "sign_coherence": coherence,
        "transport_uncertainty": uncertainty,
    }


def affine_scale_action(
    raw: np.ndarray,
    theta_zeroth: np.ndarray | float,
    theta_first: np.ndarray | float,
) -> np.ndarray:
    """Apply ``a(s)=theta_zeroth + theta_first*s`` to signed ancestry.

    This readout is exact from the first two signed moments.  A nonnegative
    contraction on the whole scale fibre requires both endpoint values
    ``theta_zeroth`` and ``theta_zeroth + theta_first`` to lie in ``[0,1]``;
    the future joint contractor will evolve those two endpoint coordinates.
    """
    lift = np.asarray(raw, dtype=np.float64)
    if lift.ndim < 2 or lift.shape[0] != 5:
        raise ValueError("affine scale action expects five raw moments")
    return (
        np.asarray(theta_zeroth, dtype=np.float64) * lift[0]
        + np.asarray(theta_first, dtype=np.float64) * lift[1]
    )


def _joint_normal_full_action_audit(
    posterior: np.ndarray,
    residual: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the complete residual transfer in the compact normal charts."""
    _q_r, _lo_r, _hi_r, rectangle_r = (
        _crossfit_value_jet_constraints(residual))
    _q_p, _lo_p, _hi_p, rectangle_p = (
        _crossfit_value_jet_constraints(posterior))
    h_r, lower_r, upper_r, normal_r = _covariance_normal_constraints(
        residual, rectangle_r, additive_transfer=False)
    h_p, lower_p, upper_p, normal_p = _covariance_normal_constraints(
        posterior, rectangle_p, additive_transfer=True)
    action = residual.reshape(-1)
    coordinate_r = np.asarray(h_r @ action).ravel()
    coordinate_p = np.asarray(h_p @ action).ravel()
    feasible_r = (coordinate_r >= lower_r) & (coordinate_r <= upper_r)
    feasible_p = (coordinate_p >= lower_p) & (coordinate_p <= upper_p)
    return {
        "target_edge_count": int(coordinate_r.size),
        "constraint_scalar_count": int(
            coordinate_r.size + coordinate_p.size),
        "full_action_residual_compatibility": float(np.mean(feasible_r)),
        "full_action_posterior_compatibility": float(np.mean(feasible_p)),
        "full_action_joint_edge_compatibility": float(np.mean(
            feasible_r & feasible_p)),
        "full_action_constraint_violation_fraction": float(np.mean(
            np.concatenate((~feasible_r, ~feasible_p)))),
        "residual_normal": normal_r,
        "posterior_normal": normal_p,
    }


def lifted_scale_moment_transport_state_2d(
    observation: np.ndarray,
    *,
    initial_posterior: np.ndarray | None = None,
    trace_refinement: int = 0,
) -> dict[str, Any]:
    """Build and positively transport the fixed ten-channel ancestry lift."""
    image = _validate(observation)
    refinement = int(trace_refinement)
    if refinement < 0:
        raise ValueError("trace refinement must be nonnegative")
    if initial_posterior is None:
        _base, _base_residual, base = causal_scale_transport_observation_2d(
            image, trace_refinement=refinement)
        initial = np.asarray(
            base["readouts"]["phase_susceptibility"], dtype=np.float64)
    else:
        initial = _validate(initial_posterior).copy()
        if initial.shape != image.shape:
            raise ValueError("initial posterior must align with observation")
        base = {"status": "caller-supplied initial posterior"}

    inherited_residual = image - initial
    posterior_phase, posterior_phase_diagnostic = _phase_action_authority(initial)
    posterior, posterior_smoothing = _phase_screened_smooth(
        initial, initial, inherited_residual, posterior_phase)
    posterior_shed = initial - posterior
    residual = inherited_residual + posterior_shed

    inherited, inherited_labels, inherited_diagnostic = (
        _continuous_scale_lineage_generators(
            inherited_residual, "inherited_residual", refinement))
    shed, shed_labels, shed_diagnostic = (
        _continuous_scale_lineage_generators(
            posterior_shed, "posterior_shed", refinement))
    inherited_lift, inherited_lift_diagnostic = _raw_scale_moment_lift(
        inherited, inherited_labels)
    shed_lift, shed_lift_diagnostic = _raw_scale_moment_lift(
        shed, shed_labels)
    vertex_lift = np.concatenate((shed_lift, inherited_lift), axis=0)
    shape = image.shape
    vertex_lift_field = vertex_lift.reshape((10,) + shape)
    shed_readout = _moment_readouts(shed_lift.reshape((5,) + shape))
    inherited_readout = _moment_readouts(
        inherited_lift.reshape((5,) + shape))
    transport_uncertainty = (
        shed_readout["transport_uncertainty"]
        + inherited_readout["transport_uncertainty"]
    )

    metric = continual_transport_metric(
        posterior, residual * residual + transport_uncertainty)
    laplacian, _markov, stencil = _continual_flux_laplacian(
        metric, np.ones_like(image))
    maximum_degree = float(stencil["maximum_degree"])
    pushed_vertex_lift = (
        _screened_transport(
            laplacian, 1.0 / maximum_degree, vertex_lift_field)
        if maximum_degree > 0.0 else vertex_lift_field.copy()
    )
    pushed_shed_readout = _moment_readouts(pushed_vertex_lift[:5])
    pushed_inherited_readout = _moment_readouts(pushed_vertex_lift[5:])
    directly_pushed_signed = (
        _screened_transport(
            laplacian,
            1.0 / maximum_degree,
            np.stack((posterior_shed, inherited_residual)),
        )
        if maximum_degree > 0.0
        else np.stack((posterior_shed, inherited_residual))
    )
    pushed_signed_error = float(np.max(np.abs(
        pushed_vertex_lift[[0, 5]] - directly_pushed_signed
    )))
    recomposed_residual = (
        vertex_lift_field[0] + vertex_lift_field[5])
    normal_audit = _joint_normal_full_action_audit(posterior, residual)
    lineage_count = inherited.shape[1] + shed.shape[1]
    persistent_dimension = 14  # posterior + 10 moments + symmetric metric
    expanded_dense_push_scalar_count = int(
        image.size * int(stencil["undirected_edge_count"])
    )
    lifted_push_scalar_count = int(vertex_lift_field.size)
    return {
        "status": (
            "fixed-dimensional raw scale-moment lift with joint normal audit"
        ),
        "theory_status": (
            "exact zeroth ancestry and positive moment transport retained; "
            "closure beyond second scale moment and mixture action unresolved"
        ),
        "posterior_after_erosion": posterior,
        "initial_posterior": initial,
        "residual_after_erosion": residual,
        "posterior_shed": posterior_shed,
        "inherited_residual": inherited_residual,
        "raw_moment_names": _RAW_MOMENT_NAMES,
        "vertex_lift": vertex_lift_field,
        "pushed_vertex_lift": pushed_vertex_lift,
        "shed_readout": shed_readout,
        "inherited_readout": inherited_readout,
        "pushed_shed_readout": pushed_shed_readout,
        "pushed_inherited_readout": pushed_inherited_readout,
        "transport_uncertainty": transport_uncertainty,
        "metric": metric,
        "stencil": stencil,
        "joint_normal_audit": normal_audit,
        "persistent_dimension": persistent_dimension,
        "lineage_count_used_to_form_moments": int(lineage_count),
        "lifted_push_scalar_count": lifted_push_scalar_count,
        "expanded_dense_push_scalar_count": expanded_dense_push_scalar_count,
        "dense_push_to_lifted_push_scalar_ratio": (
            float(expanded_dense_push_scalar_count / lifted_push_scalar_count)
            if lifted_push_scalar_count else 0.0
        ),
        "observation_recomposition_error": float(np.max(np.abs(
            posterior + residual - image
        ))),
        "lifted_residual_recomposition_error": float(np.max(np.abs(
            recomposed_residual - residual
        ))),
        "pushed_signed_commutation_error": pushed_signed_error,
        "inherited_lineage": inherited_diagnostic,
        "shed_lineage": shed_diagnostic,
        "inherited_lift": inherited_lift_diagnostic,
        "shed_lift": shed_lift_diagnostic,
        "posterior_phase": posterior_phase_diagnostic,
        "posterior_smoothing": posterior_smoothing,
        "base": base,
    }


__all__ = [
    "affine_scale_action",
    "lifted_scale_moment_transport_state_2d",
]
