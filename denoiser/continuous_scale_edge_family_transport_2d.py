"""Factorized local edge families for continuous-scale lineage transport.

One coefficient for an entire heat increment is falsified: a single local
residual enclosure can suppress a globally valid scale lineage everywhere.
Here every exact scale increment is decomposed into its local Selling-edge
fluxes.  Each active ``(lineage, edge)`` pair has its own bounded coefficient,
while ancestry and scale labels remain explicit.

The potentially huge pushed generator is never materialized.  If ``D`` is the
oriented incidence matrix and ``F[e,g]`` is the flux of lineage ``g`` on edge
``e``, the local generators are ``D^T diag(F[:,g])``.  For a frozen positive
resolvent ``S``, compute the edge response ``R=S D^T`` once.  Every pushed
lineage generator is then ``F[e,g] R[:,e]``.  Centre and radius transport, and
evolved-graph flux re-expression, remain factorized over ``R`` and ``F``.

This is a research representation.  It performs one safe contraction and one
frozen push-forward; it does not select a point from the surviving set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

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
    _flux_pattern_coordinates,
)
from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
    directional_noise_witnesses,
)
from .residual_erosion_transport_2d import _off_diagonal_conductance
from .witnessed_characteristic_transport_2d import _validate


def _contract_sparse_generator_box(
    generator: sparse.csc_matrix,
    transfer_lower: np.ndarray,
    transfer_upper: np.ndarray,
    *,
    numerical_sweep_ceiling: int = 64,
    initial_lower: np.ndarray | None = None,
    initial_upper: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Jacobi interval contractor with work linear in sparse nonzeros."""
    matrix = sparse.csr_matrix(generator, dtype=np.float64)
    lower_bound = np.asarray(transfer_lower, dtype=np.float64).reshape(-1)
    upper_bound = np.asarray(transfer_upper, dtype=np.float64).reshape(-1)
    if (
        matrix.shape[0] != lower_bound.size
        or upper_bound.shape != lower_bound.shape
        or np.any(lower_bound > upper_bound)
    ):
        raise ValueError("linear transfer bounds must align and be ordered")
    ceiling = int(numerical_sweep_ceiling)
    if ceiling < 1:
        raise ValueError("numerical sweep ceiling must be positive")
    count = matrix.shape[1]
    lower = (
        np.zeros(count, dtype=np.float64)
        if initial_lower is None
        else np.asarray(initial_lower, dtype=np.float64).copy()
    )
    upper = (
        np.ones(count, dtype=np.float64)
        if initial_upper is None
        else np.asarray(initial_upper, dtype=np.float64).copy()
    )
    if (
        lower.shape != (count,)
        or upper.shape != (count,)
        or np.any(lower < 0.0)
        or np.any(upper > 1.0)
        or np.any(lower > upper)
    ):
        raise ValueError("initial coefficient box must be ordered inside [0,1]")
    tolerance = 64.0 * np.finfo(float).eps
    infeasible = False
    completed = 0
    last_contraction = 0.0
    for sweep in range(ceiling):
        completed = sweep + 1
        proposed_lower = lower.copy()
        proposed_upper = upper.copy()
        for row in range(matrix.shape[0]):
            start, stop = matrix.indptr[row], matrix.indptr[row + 1]
            indices = matrix.indices[start:stop]
            values = matrix.data[start:stop]
            active = values != 0.0
            indices = indices[active]
            values = values[active]
            if not indices.size:
                if (
                    lower_bound[row] > tolerance
                    or upper_bound[row] < -tolerance
                ):
                    infeasible = True
                    break
                continue
            value_lower = values * lower[indices]
            value_upper = values * upper[indices]
            contribution_lower = np.minimum(value_lower, value_upper)
            contribution_upper = np.maximum(value_lower, value_upper)
            total_lower = float(np.sum(contribution_lower))
            total_upper = float(np.sum(contribution_upper))
            allowed_low = (
                lower_bound[row]
                - (total_upper - contribution_upper)
            )
            allowed_high = (
                upper_bound[row]
                - (total_lower - contribution_lower)
            )
            candidate_lower = np.where(
                values > 0.0,
                allowed_low / values,
                allowed_high / values,
            )
            candidate_upper = np.where(
                values > 0.0,
                allowed_high / values,
                allowed_low / values,
            )
            np.maximum.at(proposed_lower, indices, candidate_lower)
            np.minimum.at(proposed_upper, indices, candidate_upper)
        if infeasible:
            break
        if np.any(proposed_lower > proposed_upper + tolerance):
            infeasible = True
            break
        proposed_lower = np.minimum(np.maximum(proposed_lower, lower), upper)
        proposed_upper = np.maximum(
            np.minimum(proposed_upper, upper), proposed_lower)
        last_contraction = float(max(
            np.max(proposed_lower - lower) if count else 0.0,
            np.max(upper - proposed_upper) if count else 0.0,
        ))
        lower = proposed_lower
        upper = proposed_upper
        if last_contraction <= tolerance:
            break

    positive = matrix.maximum(0.0)
    negative = matrix.minimum(0.0)
    row_minimum = np.asarray(
        positive @ lower + negative @ upper).ravel()
    row_maximum = np.asarray(
        positive @ upper + negative @ lower).ravel()
    disjoint = (
        (row_maximum < lower_bound - tolerance)
        | (row_minimum > upper_bound + tolerance)
    )
    infeasible = bool(infeasible or np.any(disjoint))
    width = np.maximum(upper - lower, 0.0)
    return lower, upper, {
        "feasible_outer_component": bool(not infeasible),
        "completed_sweeps": int(completed),
        "numerical_sweep_ceiling_hit": bool(
            completed == ceiling
            and not infeasible
            and last_contraction > tolerance
        ),
        "mean_coefficient_width": float(np.mean(width)) if count else 0.0,
        "contracted_coefficient_fraction": float(np.mean(
            width < 1.0 - tolerance)) if count else 0.0,
        "point_coefficient_fraction": float(np.mean(
            width <= tolerance)) if count else 0.0,
        "row_disjoint_fraction": float(np.mean(disjoint)),
    }


def _local_scale_edge_generators(
    laplacian: sparse.csr_matrix,
    lineage_field: np.ndarray,
) -> tuple[sparse.csc_matrix, dict[str, Any]]:
    """Expand lineage fields into independently bounded local edge fluxes."""
    fields = np.asarray(lineage_field, dtype=np.float64)
    if fields.ndim != 2 or fields.shape[0] != laplacian.shape[0]:
        raise ValueError("lineage fields and graph must align")
    pattern = _flux_pattern_coordinates(laplacian, fields)
    source = np.asarray(pattern["edge_source"], dtype=np.int64)
    target = np.asarray(pattern["edge_target"], dtype=np.int64)
    flux = np.asarray(pattern["edge_flux_pattern"], dtype=np.float64)
    component_mean = np.asarray(pattern["component_zero_modes"])
    component_labels = np.asarray(
        pattern["connected_component_labels"], dtype=np.int64)
    numerical = 64.0 * np.finfo(float).eps
    edge_scale = np.maximum(np.max(np.abs(flux), axis=0), 1.0)
    active_edge, active_lineage = np.nonzero(
        np.abs(flux) > numerical * edge_scale[None, :])
    active_flux = flux[active_edge, active_lineage]
    columns = np.arange(active_flux.size, dtype=np.int64)
    row_parts = [source[active_edge], target[active_edge]]
    column_parts = [columns, columns]
    value_parts = [active_flux, -active_flux]

    zero_component, zero_lineage = np.nonzero(
        np.abs(component_mean)
        > numerical * np.maximum(
            np.max(np.abs(component_mean), axis=0), 1.0
        )[None, :]
    )
    zero_start = active_flux.size
    zero_columns = zero_start + np.arange(zero_component.size, dtype=np.int64)
    for ordinal, (component, lineage) in enumerate(zip(
        zero_component, zero_lineage
    )):
        members = np.flatnonzero(component_labels == component)
        row_parts.append(members)
        column_parts.append(np.full(
            members.size, zero_columns[ordinal], dtype=np.int64))
        value_parts.append(np.full(
            members.size,
            component_mean[component, lineage],
            dtype=np.float64,
        ))
    generator = sparse.coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(fields.shape[0], active_flux.size + zero_component.size),
    ).tocsc()
    generator.eliminate_zeros()
    reconstructed = np.asarray(
        generator @ np.ones(generator.shape[1])).ravel()
    return generator, {
        "edge_source": source,
        "edge_target": target,
        "lineage_edge_flux": flux,
        "active_edge_index": active_edge,
        "active_edge_lineage": active_lineage,
        "active_edge_flux": active_flux,
        "zero_component": zero_component,
        "zero_lineage": zero_lineage,
        "component_zero_modes": component_mean,
        "connected_component_labels": component_labels,
        "edge_variable_count": int(active_flux.size),
        "zero_variable_count": int(zero_component.size),
        "variable_count": int(generator.shape[1]),
        "full_recomposition_maximum_error": float(np.max(np.abs(
            reconstructed - np.sum(fields, axis=1)))),
        "flux_pattern_reconstruction_error": pattern[
            "reconstruction_maximum_error"],
    }


def _factorized_push_enclosure(
    edge_response: np.ndarray,
    pushed_zero: np.ndarray,
    representation: dict[str, Any],
    coefficient_lower: np.ndarray,
    coefficient_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = 0.5 * (coefficient_lower + coefficient_upper)
    radius = 0.5 * (coefficient_upper - coefficient_lower)
    edge_count = representation["edge_variable_count"]
    active_edge = representation["active_edge_index"]
    active_flux = representation["active_edge_flux"]
    edge_center = np.zeros(edge_response.shape[1], dtype=np.float64)
    edge_radius = np.zeros(edge_response.shape[1], dtype=np.float64)
    np.add.at(
        edge_center,
        active_edge,
        active_flux * center[:edge_count],
    )
    np.add.at(
        edge_radius,
        active_edge,
        np.abs(active_flux) * radius[:edge_count],
    )
    pushed_center = edge_response @ edge_center
    pushed_radius = np.abs(edge_response) @ edge_radius
    if pushed_zero.shape[1]:
        pushed_center += pushed_zero @ center[edge_count:]
        pushed_radius += np.abs(pushed_zero) @ radius[edge_count:]
    return pushed_center, pushed_radius


def continuous_scale_edge_family_transport_state_2d(
    observation: np.ndarray,
    *,
    initial_posterior: np.ndarray | None = None,
    trace_refinement: int = 0,
) -> dict[str, Any]:
    """Contract and push one factorized local continuous-scale family."""
    image = _validate(observation)
    refinement = int(trace_refinement)
    if initial_posterior is None:
        _base, _residual, base = causal_scale_transport_observation_2d(image)
        initial = np.asarray(
            base["readouts"]["phase_susceptibility"], dtype=np.float64)
    else:
        initial = _validate(initial_posterior).copy()
        if initial.shape != image.shape:
            raise ValueError("initial posterior must align with observation")
        base = {"status": "caller-supplied initial posterior"}
    inherited_residual = image - initial
    posterior_phase, posterior_phase_diagnostic = (
        _phase_action_authority(initial))
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
    lineage = np.concatenate((shed, inherited), axis=1)
    lineage_labels = (*shed_labels, *inherited_labels)

    metric = continual_transport_metric(posterior, residual * residual)
    laplacian, _markov, stencil = _continual_flux_laplacian(
        metric, np.ones_like(image))
    generator, representation = _local_scale_edge_generators(
        laplacian, lineage)
    witness_centre, witness_variance, witness_radius, witness_diagnostic = (
        directional_noise_witnesses(image, posterior))
    raw_lower = witness_centre - witness_radius
    raw_upper = witness_centre + witness_radius
    safe_lower = np.minimum(raw_lower, residual)
    safe_upper = np.maximum(raw_upper, residual)
    coefficient_lower, coefficient_upper, contraction = (
        _contract_sparse_generator_box(
            generator,
            (residual - safe_upper).reshape(-1),
            (residual - safe_lower).reshape(-1),
        ))
    center_coefficient = 0.5 * (
        coefficient_lower + coefficient_upper)
    radius_coefficient = 0.5 * (
        coefficient_upper - coefficient_lower)
    center_transfer = np.asarray(generator @ center_coefficient).ravel()
    radius_transfer = np.asarray(abs(generator) @ radius_coefficient).ravel()
    center_posterior = posterior + center_transfer.reshape(image.shape)
    center_residual = image - center_posterior

    push_metric = continual_transport_metric(
        center_posterior, center_residual * center_residual)
    push_laplacian, _push_markov, push_stencil = _continual_flux_laplacian(
        push_metric, np.ones_like(image))
    push_degree = float(push_stencil["maximum_degree"])
    source = representation["edge_source"]
    target = representation["edge_target"]
    edge_count = source.size
    incidence = sparse.coo_matrix(
        (
            np.concatenate((np.ones(edge_count), -np.ones(edge_count))),
            (
                np.concatenate((source, target)),
                np.concatenate((np.arange(edge_count), np.arange(edge_count))),
            ),
        ),
        shape=(image.size, edge_count),
    ).tocsc()
    zero_count = representation["zero_variable_count"]
    zero_generator = (
        generator[:, representation["edge_variable_count"]:].toarray()
        if zero_count else np.empty((image.size, 0), dtype=np.float64)
    )
    push_fields = np.concatenate((
        posterior[None, ...],
        incidence.T.toarray().reshape((-1,) + image.shape),
        zero_generator.T.reshape((-1,) + image.shape),
    ), axis=0)
    pushed = (
        _screened_transport(
            push_laplacian,
            1.0 / push_degree,
            push_fields,
        )
        if push_degree > 0.0 else push_fields.copy()
    )
    pushed_posterior = pushed[0]
    edge_response = pushed[1:1 + edge_count].reshape((edge_count, -1)).T
    pushed_zero = (
        pushed[1 + edge_count:].reshape((zero_count, image.size)).T
        if zero_count
        else np.empty((image.size, 0), dtype=np.float64)
    )
    pushed_center, pushed_radius = _factorized_push_enclosure(
        edge_response,
        pushed_zero,
        representation,
        coefficient_lower,
        coefficient_upper,
    )
    pushed_center_posterior = (
        pushed_posterior + pushed_center.reshape(image.shape))
    pushed_center_residual = image - pushed_center_posterior
    evolved_metric = continual_transport_metric(
        pushed_center_posterior,
        pushed_center_residual * pushed_center_residual)
    evolved_laplacian, _evolved_markov, evolved_stencil = (
        _continual_flux_laplacian(evolved_metric, np.ones_like(image)))
    evolved_edge_response_flux = _flux_pattern_coordinates(
        evolved_laplacian, edge_response)
    evolved_zero_response_flux = _flux_pattern_coordinates(
        evolved_laplacian, pushed_zero)

    return {
        "status": (
            "factorized local scale-edge zonotope after one safe contraction "
            "and frozen positive push-forward"
        ),
        "theory_status": (
            "local scale ancestry and factorized evolved flux are retained; "
            "joint value/jet enclosure and nonlinear set metric unresolved"
        ),
        "posterior_after_erosion": posterior,
        "residual_after_erosion": residual,
        "generator": generator,
        "representation": representation,
        "lineage_labels": lineage_labels,
        "coefficient_lower": coefficient_lower,
        "coefficient_upper": coefficient_upper,
        "transfer_enclosure_lower": (
            center_transfer - radius_transfer).reshape(image.shape),
        "transfer_enclosure_upper": (
            center_transfer + radius_transfer).reshape(image.shape),
        "pushed_posterior_base": pushed_posterior,
        "pushed_transfer_enclosure_lower": (
            pushed_center - pushed_radius).reshape(image.shape),
        "pushed_transfer_enclosure_upper": (
            pushed_center + pushed_radius).reshape(image.shape),
        "branches": {
            "identity_lineage": {
                "posterior_base": posterior,
                "transfer_lower": (
                    center_transfer - radius_transfer).reshape(image.shape),
                "transfer_upper": (
                    center_transfer + radius_transfer).reshape(image.shape),
            },
            "positive_push_lineage": {
                "posterior_base": pushed_posterior,
                "transfer_lower": (
                    pushed_center - pushed_radius).reshape(image.shape),
                "transfer_upper": (
                    pushed_center + pushed_radius).reshape(image.shape),
            },
        },
        "edge_response": edge_response,
        "pushed_zero_response": pushed_zero,
        "evolved_edge_response_flux": evolved_edge_response_flux,
        "evolved_zero_response_flux": evolved_zero_response_flux,
        "contraction": contraction,
        "inherited_lineage": inherited_diagnostic,
        "shed_lineage": shed_diagnostic,
        "raw_witness_exclusion_fraction": float(np.mean(
            (residual < raw_lower) | (residual > raw_upper)
        )),
        "observation_recomposition_error": float(np.max(np.abs(
            posterior + residual - image))),
        "full_lineage_recomposition_error": representation[
            "full_recomposition_maximum_error"],
        "posterior_phase": posterior_phase_diagnostic,
        "posterior_smoothing": posterior_smoothing,
        "witness": witness_diagnostic,
        "initial_stencil": stencil,
        "push_stencil": push_stencil,
        "evolved_stencil": evolved_stencil,
        "base": base,
    }


__all__ = [
    "continuous_scale_edge_family_transport_state_2d",
]
