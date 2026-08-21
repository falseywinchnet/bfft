"""Continuous-scale lineage zonotope and its frozen positive push-forward.

The earlier edge-flux state collapsed every continuous-scale explanation into
one proposal field before constructing its zonotope.  This experiment retains
the exact heat-semigroup measure instead.  The original residual and the
posterior's newly shed field are decomposed independently into a coarse
endpoint and every coarse-to-fine increment.  Each increment is one labelled
generator coefficient in ``[0,1]``; their all-ones recomposition is exact.

After safe bounded-residual contraction, the complete zonotope is pushed
through one frozen positive Selling resolvent:

    S(p + G alpha) = S p + (S G) alpha.

No point readout is needed for this linear push-forward.  A centre is used
only to freeze the next nonlinear metric.  Every pushed generator is then
represented as one antisymmetric flux *pattern* on that evolved graph.  Its
coefficient and ancestry label remain intact, avoiding the quadratic
generator explosion that would result from making every new edge an
independent variable.

This is still a one-generation research state.  Trace-refinement covariance
and a joint value/jet enclosure remain unresolved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as sparse_linalg

from .causal_scale_transport_2d import (
    _screened_transport,
    causal_scale_transport_observation_2d,
)
from .conservative_exchange_transport_2d import (
    _phase_action_authority,
    _phase_screened_smooth,
)
from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
    directional_noise_witnesses,
)
from .residual_erosion_transport_2d import (
    _cavity_residual_relation,
    _off_diagonal_conductance,
)
from .witnessed_characteristic_transport_2d import _validate
from .zonotopic_edge_flux_2d import _contract_linear_zonotope_box


def _continuous_scale_lineage_generators(
    field: np.ndarray,
    ancestry: str,
    trace_refinement: int,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Return exact coarse endpoint and heat increments as labelled columns."""
    _estimate, _residual, diagnostic = causal_scale_transport_observation_2d(
        field, trace_refinement=trace_refinement)
    coarse = np.asarray(diagnostic["coarse_endpoint"], dtype=np.float64)
    components = np.asarray(
        diagnostic["components_coarse_to_fine"], dtype=np.float64)
    phase = np.asarray(
        diagnostic["phase_susceptibility_coarse_to_fine"],
        dtype=np.float64,
    )
    fields = np.concatenate((coarse[None, ...], components), axis=0)
    columns = fields.reshape((fields.shape[0], -1)).T
    labels: list[dict[str, Any]] = [{
        "ancestry": ancestry,
        "kind": "coarse_endpoint",
        "ordinal_coarse_to_fine": -1,
        "phase_action_authority": None,
    }]
    for ordinal, (generation, phase_field, component) in enumerate(zip(
        diagnostic["generations"], phase, components
    )):
        action = component * component
        total = float(np.sum(action))
        labels.append({
            "ancestry": ancestry,
            "kind": "heat_increment",
            "ordinal_coarse_to_fine": int(ordinal),
            "transport_time_coarse": generation["transport_time_coarse"],
            "transport_time_fine": generation["transport_time_fine"],
            "phase_action_authority": (
                float(np.sum(phase_field * action)) / total
                if total > np.finfo(float).tiny else 0.0
            ),
        })
    reconstructed = np.sum(fields, axis=0)
    return columns, tuple(labels), {
        "ancestry": ancestry,
        "generator_count": int(columns.shape[1]),
        "trace_refinement": int(trace_refinement),
        "exact_recomposition_maximum_error": float(np.max(np.abs(
            reconstructed - np.asarray(field, dtype=np.float64)))),
        "decomposition_maximum_error": diagnostic[
            "decomposition_maximum_error"],
        "transport_times": diagnostic["transport_times"],
    }


def _flux_pattern_coordinates(
    laplacian: sparse.csr_matrix,
    generator: np.ndarray,
) -> dict[str, Any]:
    """Represent each dense generator as one flux pattern plus zero modes."""
    matrix = np.asarray(generator, dtype=np.float64)
    if matrix.ndim != 2 or laplacian.shape[0] != matrix.shape[0]:
        raise ValueError("generator columns and evolved graph must align")
    conductance, _mass = _off_diagonal_conductance(laplacian)
    component_count, labels = csgraph.connected_components(
        conductance, directed=False, return_labels=True)
    if matrix.shape[1] == 0:
        upper = sparse.triu(conductance, k=1).tocoo()
        return {
            "edge_source": upper.row.astype(np.int64),
            "edge_target": upper.col.astype(np.int64),
            "edge_flux_pattern": np.empty(
                (upper.nnz, 0), dtype=np.float64),
            "connected_component_labels": labels,
            "component_zero_modes": np.empty(
                (component_count, 0), dtype=np.float64),
            "edge_count": int(upper.nnz),
            "generator_count": 0,
            "connected_component_count": int(component_count),
            "reconstruction_maximum_error": 0.0,
            "antisymmetric_flux_sum_error": 0.0,
        }
    potential = np.zeros_like(matrix)
    component_mean = np.zeros(
        (component_count, matrix.shape[1]), dtype=np.float64)
    for component in range(component_count):
        members = np.flatnonzero(labels == component)
        component_mean[component] = np.mean(matrix[members], axis=0)
        if members.size <= 1:
            continue
        rhs = matrix[members] - component_mean[component][None, :]
        local_laplacian = laplacian[members][:, members].tocsc()
        solved = sparse_linalg.spsolve(
            local_laplacian[1:, 1:], rhs[1:])
        if solved.ndim == 1:
            solved = solved[:, None]
        potential[members[1:]] = solved

    upper = sparse.triu(conductance, k=1).tocoo()
    source = upper.row.astype(np.int64)
    target = upper.col.astype(np.int64)
    flux_pattern = upper.data[:, None] * (
        potential[source] - potential[target])
    reconstructed = np.zeros_like(matrix)
    np.add.at(reconstructed, source, flux_pattern)
    np.add.at(reconstructed, target, -flux_pattern)
    reconstructed += component_mean[labels]
    edge_column_sum = np.sum(
        np.concatenate((flux_pattern, -flux_pattern), axis=0), axis=0)
    return {
        "edge_source": source,
        "edge_target": target,
        "edge_flux_pattern": flux_pattern,
        "connected_component_labels": labels,
        "component_zero_modes": component_mean,
        "edge_count": int(source.size),
        "generator_count": int(matrix.shape[1]),
        "connected_component_count": int(component_count),
        "reconstruction_maximum_error": float(np.max(np.abs(
            reconstructed - matrix))),
        "antisymmetric_flux_sum_error": (
            float(np.max(np.abs(edge_column_sum)))
            if edge_column_sum.size else 0.0
        ),
    }


def _transfer_enclosure(
    generator: np.ndarray,
    coefficient_lower: np.ndarray,
    coefficient_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_coefficient = 0.5 * (coefficient_lower + coefficient_upper)
    radius_coefficient = 0.5 * (coefficient_upper - coefficient_lower)
    center = generator @ center_coefficient
    radius = np.abs(generator) @ radius_coefficient
    return center, radius, center - radius, center + radius


def continuous_scale_zonotope_transport_state_2d(
    observation: np.ndarray,
    *,
    initial_posterior: np.ndarray | None = None,
    trace_refinement: int = 0,
    include_curvature_lineage: bool = True,
) -> dict[str, Any]:
    """Build, contract, push, and re-express one scale-lineage zonotope."""
    image = _validate(observation)
    refinement = int(trace_refinement)
    if refinement < 0:
        raise ValueError("trace refinement must be nonnegative")
    if initial_posterior is None:
        _base, _base_residual, base = (
            causal_scale_transport_observation_2d(
                image, trace_refinement=refinement))
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

    inherited_generator, inherited_labels, inherited_diagnostic = (
        _continuous_scale_lineage_generators(
            inherited_residual, "inherited_residual", refinement))
    shed_generator, shed_labels, shed_diagnostic = (
        _continuous_scale_lineage_generators(
            posterior_shed, "posterior_shed", refinement))
    generator_parts = [shed_generator, inherited_generator]
    lineage_labels = [*shed_labels, *inherited_labels]

    metric = continual_transport_metric(posterior, residual * residual)
    laplacian, _markov, stencil = _continual_flux_laplacian(
        metric, np.ones_like(image))
    maximum_degree = float(stencil["maximum_degree"])
    curvature_relation_diagnostic: dict[str, Any] | None = None
    if include_curvature_lineage:
        raw_curvature, curvature_relation = _cavity_residual_relation(
            posterior, residual, laplacian, maximum_degree)
        curvature = (
            _screened_transport(
                laplacian,
                1.0 / maximum_degree,
                raw_curvature[None, ...],
            )[0]
            if maximum_degree > 0.0 else raw_curvature
        )
        generator_parts.append(curvature.reshape((-1, 1)))
        lineage_labels.append({
            "ancestry": "posterior_curvature_cavity",
            "kind": "target_excluded_relation",
            "ordinal_coarse_to_fine": None,
            "phase_action_authority": None,
        })
        curvature_relation_diagnostic = {
            key: value
            for key, value in curvature_relation.items()
            if np.isscalar(value)
        }
    generator = np.concatenate(generator_parts, axis=1)
    full_recomposition = (posterior + np.sum(
        generator[:, :shed_generator.shape[1] + inherited_generator.shape[1]],
        axis=1,
    ).reshape(image.shape))

    witness_centre, witness_variance, witness_radius, witness_diagnostic = (
        directional_noise_witnesses(image, posterior))
    raw_lower = witness_centre - witness_radius
    raw_upper = witness_centre + witness_radius
    safe_lower = np.minimum(raw_lower, residual)
    safe_upper = np.maximum(raw_upper, residual)
    transfer_lower = (residual - safe_upper).reshape(-1)
    transfer_upper = (residual - safe_lower).reshape(-1)
    coefficient_lower, coefficient_upper, contraction = (
        _contract_linear_zonotope_box(
            sparse.csc_matrix(generator),
            transfer_lower,
            transfer_upper,
        ))
    center_transfer, radius_transfer, transfer_enclosure_lower, (
        transfer_enclosure_upper
    ) = _transfer_enclosure(
        generator, coefficient_lower, coefficient_upper)
    center_posterior = posterior + center_transfer.reshape(image.shape)
    center_residual = image - center_posterior

    push_metric = continual_transport_metric(
        center_posterior, center_residual * center_residual)
    push_laplacian, _push_markov, push_stencil = _continual_flux_laplacian(
        push_metric, np.ones_like(image))
    push_degree = float(push_stencil["maximum_degree"])
    fields = np.concatenate((
        posterior[None, ...],
        generator.T.reshape((-1,) + image.shape),
    ), axis=0)
    pushed_fields = (
        _screened_transport(
            push_laplacian,
            1.0 / push_degree,
            fields,
        )
        if push_degree > 0.0 else fields.copy()
    )
    pushed_posterior = pushed_fields[0]
    pushed_generator = pushed_fields[1:].reshape((generator.shape[1], -1)).T
    pushed_center_transfer, pushed_radius_transfer, (
        pushed_transfer_lower
    ), pushed_transfer_upper = _transfer_enclosure(
        pushed_generator, coefficient_lower, coefficient_upper)
    pushed_center_posterior = (
        pushed_posterior
        + pushed_center_transfer.reshape(image.shape)
    )
    pushed_center_residual = image - pushed_center_posterior

    evolved_metric = continual_transport_metric(
        pushed_center_posterior,
        pushed_center_residual * pushed_center_residual,
    )
    evolved_laplacian, _evolved_markov, evolved_stencil = (
        _continual_flux_laplacian(evolved_metric, np.ones_like(image)))
    pushed_flux = _flux_pattern_coordinates(
        evolved_laplacian, pushed_generator)

    return {
        "status": (
            "one frozen positive push-forward of an exact continuous-scale "
            "lineage zonotope"
        ),
        "theory_status": (
            "linear set push-forward and evolved-graph flux re-expression "
            "are exact; nonlinear metric enclosure and scale-density limit "
            "remain unresolved"
        ),
        "posterior_after_erosion": posterior,
        "residual_after_erosion": residual,
        "generator": generator,
        "lineage_labels": tuple(lineage_labels),
        "coefficient_lower": coefficient_lower,
        "coefficient_upper": coefficient_upper,
        "center_transfer": center_transfer.reshape(image.shape),
        "radius_transfer": radius_transfer.reshape(image.shape),
        "transfer_enclosure_lower": transfer_enclosure_lower.reshape(
            image.shape),
        "transfer_enclosure_upper": transfer_enclosure_upper.reshape(
            image.shape),
        "center_posterior_for_geometry_only": center_posterior,
        "pushed_posterior_base": pushed_posterior,
        "pushed_generator": pushed_generator,
        "pushed_center_posterior_for_geometry_only": pushed_center_posterior,
        "pushed_transfer_enclosure_lower": pushed_transfer_lower.reshape(
            image.shape),
        "pushed_transfer_enclosure_upper": pushed_transfer_upper.reshape(
            image.shape),
        "pushed_flux_patterns": pushed_flux,
        "contraction": contraction,
        "inherited_lineage": inherited_diagnostic,
        "shed_lineage": shed_diagnostic,
        "curvature_relation": curvature_relation_diagnostic,
        "raw_witness_exclusion_fraction": float(np.mean(
            (residual < raw_lower) | (residual > raw_upper)
        )),
        "observation_recomposition_error": float(np.max(np.abs(
            posterior + residual - image))),
        "full_lineage_recomposition_error": float(np.max(np.abs(
            full_recomposition - image))),
        "pushforward_center_linearity_error": float(np.max(np.abs(
            pushed_center_posterior
            - (
                pushed_posterior
                + (pushed_generator @ (
                    0.5 * (coefficient_lower + coefficient_upper)
                )).reshape(image.shape)
            )
        ))),
        "posterior_phase": posterior_phase_diagnostic,
        "posterior_smoothing": posterior_smoothing,
        "witness": witness_diagnostic,
        "initial_stencil": stencil,
        "push_stencil": push_stencil,
        "evolved_stencil": evolved_stencil,
        "base": base,
    }


__all__ = [
    "continuous_scale_zonotope_transport_state_2d",
]
