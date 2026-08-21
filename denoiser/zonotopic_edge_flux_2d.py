"""Set-valued residual return on conservative Selling-edge fluxes.

This experiment replaces a scalar residual-return authority by a finite
mixture of bounded flux hypotheses.  Exact posterior-shed ancestry,
reciprocal phase, and target-excluded posterior curvature remain distinct
components.  For any component, a
proposed signed correction ``d`` is represented exactly as

    d = A 1,

where every non-constant column of ``A`` is the divergence of one oriented
antisymmetric Selling-edge flux.  One constant generator per connected graph
component carries the Laplacian zero mode explicitly.  Unknown coefficients
``alpha in [0,1]`` therefore form a zonotope of conservative transfers.

A target-excluded bounded residual law contracts the coefficient box under

    lower <= r - A alpha <= upper.

Passing this contractor is not converted into probability.  The surviving
intervals remain intervals; an empty intersection may falsify a component.
The current implementation constructs one generation only.  Nonlinear
push-forward of the complete zonotope through evolving metric geometry is the
next unresolved operation.
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


def _selling_edge_flux_zonotope(
    laplacian: sparse.csr_matrix,
    proposal: np.ndarray,
) -> tuple[sparse.csc_matrix, dict[str, Any]]:
    """Represent a field by edge divergences plus explicit graph zero modes."""
    field = np.asarray(proposal, dtype=np.float64)
    if field.ndim != 2 or laplacian.shape != (field.size, field.size):
        raise ValueError("proposal and Selling Laplacian must align")
    conductance, _mass = _off_diagonal_conductance(laplacian)
    component_count, labels = csgraph.connected_components(
        conductance, directed=False, return_labels=True)
    flat = field.reshape(-1)
    numerical_zero = (
        64.0 * np.finfo(float).eps
        * max(float(np.max(np.abs(flat))), 1.0)
    )
    if float(np.max(np.abs(flat))) <= numerical_zero:
        return sparse.csc_matrix((flat.size, 0), dtype=np.float64), {
            "edge_source": np.empty(0, dtype=np.int64),
            "edge_target": np.empty(0, dtype=np.int64),
            "edge_flux": np.empty(0, dtype=np.float64),
            "edge_count": 0,
            "connected_component_count": int(component_count),
            "zero_mode_count": 0,
            "proposal_reconstruction_maximum_error": float(
                np.max(np.abs(flat))),
            "edge_antisymmetry_column_sum_error": 0.0,
            "proposal_mean": float(np.mean(flat)),
            "minimum_energy_potential": np.zeros_like(field),
        }
    potential = np.zeros_like(flat)
    component_mean = np.zeros(component_count, dtype=np.float64)
    for component in range(component_count):
        members = np.flatnonzero(labels == component)
        component_mean[component] = float(np.mean(flat[members]))
        if members.size <= 1:
            continue
        rhs = flat[members] - component_mean[component]
        local_laplacian = laplacian[members][:, members].tocsc()
        # Pin one gauge value.  The reduced SPD graph Laplacian then has a
        # unique solve without adding a physical regularization constant.
        potential[members[1:]] = sparse_linalg.spsolve(
            local_laplacian[1:, 1:], rhs[1:])

    upper = sparse.triu(conductance, k=1).tocoo()
    edge_source = upper.row.astype(np.int64)
    edge_target = upper.col.astype(np.int64)
    edge_flux = upper.data * (
        potential[edge_source] - potential[edge_target])
    edge_count = edge_flux.size
    column = np.arange(edge_count, dtype=np.int64)
    rows = [edge_source, edge_target]
    columns = [column, column]
    values = [edge_flux, -edge_flux]
    if component_count:
        zero_column = edge_count + np.arange(component_count, dtype=np.int64)
        rows.append(np.arange(flat.size, dtype=np.int64))
        columns.append(zero_column[labels])
        values.append(component_mean[labels])
    generator = sparse.coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(flat.size, edge_count + component_count),
    ).tocsc()
    reconstructed = np.asarray(generator @ np.ones(generator.shape[1])).ravel()
    column_sum = np.asarray(generator.sum(axis=0)).ravel()
    edge_column_sum = column_sum[:edge_count]
    return generator, {
        "edge_source": edge_source,
        "edge_target": edge_target,
        "edge_flux": edge_flux,
        "edge_count": int(edge_count),
        "connected_component_count": int(component_count),
        "zero_mode_count": int(component_count),
        "proposal_reconstruction_maximum_error": float(np.max(np.abs(
            reconstructed - flat))),
        "edge_antisymmetry_column_sum_error": (
            float(np.max(np.abs(edge_column_sum)))
            if edge_count else 0.0
        ),
        "proposal_mean": float(np.mean(flat)),
        "minimum_energy_potential": potential.reshape(field.shape),
    }


def _contract_linear_zonotope_box(
    generator: sparse.csc_matrix,
    transfer_lower: np.ndarray,
    transfer_upper: np.ndarray,
    *,
    numerical_sweep_ceiling: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Contract ``alpha in [0,1]`` under linear interval observations.

    This is an interval Gauss--Seidel contractor, not an optimizer or a
    probability update.  Each row contraction is a necessary consequence of
    the complete current box of every other generator.  Empty intersection is
    therefore a conclusive falsification of this outer representation.
    """
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
    variable_count = matrix.shape[1]
    coefficient_lower = np.zeros(variable_count, dtype=np.float64)
    coefficient_upper = np.ones(variable_count, dtype=np.float64)
    tolerance = 64.0 * np.finfo(float).eps
    infeasible = False
    completed = 0
    maximum_contraction = 0.0

    for sweep in range(ceiling):
        completed = sweep + 1
        sweep_contraction = 0.0
        for row in range(matrix.shape[0]):
            start, stop = matrix.indptr[row], matrix.indptr[row + 1]
            indices = matrix.indices[start:stop]
            values = matrix.data[start:stop]
            if not indices.size:
                if (
                    lower_bound[row] > tolerance
                    or upper_bound[row] < -tolerance
                ):
                    infeasible = True
                    break
                continue
            for local, variable in enumerate(indices):
                value = values[local]
                if value == 0.0:
                    continue
                contribution_lower = np.minimum(
                    values * coefficient_lower[indices],
                    values * coefficient_upper[indices],
                )
                contribution_upper = np.maximum(
                    values * coefficient_lower[indices],
                    values * coefficient_upper[indices],
                )
                other_lower = float(
                    np.sum(contribution_lower) - contribution_lower[local])
                other_upper = float(
                    np.sum(contribution_upper) - contribution_upper[local])
                allowed_a = lower_bound[row] - other_upper
                allowed_b = upper_bound[row] - other_lower
                if value > 0.0:
                    candidate_lower = allowed_a / value
                    candidate_upper = allowed_b / value
                else:
                    candidate_lower = allowed_b / value
                    candidate_upper = allowed_a / value
                old_lower = coefficient_lower[variable]
                old_upper = coefficient_upper[variable]
                new_lower = max(old_lower, candidate_lower)
                new_upper = min(old_upper, candidate_upper)
                if new_lower > new_upper + tolerance:
                    infeasible = True
                    break
                new_lower = min(max(new_lower, old_lower), old_upper)
                new_upper = max(min(new_upper, old_upper), new_lower)
                coefficient_lower[variable] = new_lower
                coefficient_upper[variable] = new_upper
                sweep_contraction = max(
                    sweep_contraction,
                    new_lower - old_lower,
                    old_upper - new_upper,
                )
            if infeasible:
                break
        maximum_contraction = max(maximum_contraction, sweep_contraction)
        if infeasible or sweep_contraction <= tolerance:
            break

    # Sparse ``minimum``/``maximum`` retain the signed coefficients; the
    # explicit interval products avoid relying on implicit interval magic.
    positive = matrix.maximum(0.0)
    negative = matrix.minimum(0.0)
    row_minimum = np.asarray(
        positive @ coefficient_lower + negative @ coefficient_upper).ravel()
    row_maximum = np.asarray(
        positive @ coefficient_upper + negative @ coefficient_lower).ravel()
    row_disjoint = (
        (row_maximum < lower_bound - tolerance)
        | (row_minimum > upper_bound + tolerance)
    )
    infeasible = bool(infeasible or np.any(row_disjoint))
    width = np.maximum(coefficient_upper - coefficient_lower, 0.0)
    return coefficient_lower, coefficient_upper, {
        "feasible_outer_component": bool(not infeasible),
        "completed_sweeps": int(completed),
        "numerical_sweep_ceiling_hit": bool(
            completed == ceiling and not infeasible
            and sweep_contraction > tolerance
        ),
        "mean_coefficient_width": float(np.mean(width)) if width.size else 0.0,
        "contracted_coefficient_fraction": float(np.mean(
            width < 1.0 - tolerance)) if width.size else 0.0,
        "point_coefficient_fraction": float(np.mean(
            width <= tolerance)) if width.size else 0.0,
        "maximum_contraction_in_one_sweep": float(maximum_contraction),
        "row_disjoint_fraction": float(np.mean(row_disjoint)),
        "transfer_enclosure_lower": row_minimum,
        "transfer_enclosure_upper": row_maximum,
    }


def _bounded_flux_component(
    name: str,
    laplacian: sparse.csr_matrix,
    proposal: np.ndarray | tuple[tuple[str, np.ndarray], ...],
    residual: np.ndarray,
    residual_lower: np.ndarray,
    residual_upper: np.ndarray,
) -> dict[str, Any]:
    if isinstance(proposal, tuple):
        proposal_parts = tuple(
            (part_name, np.asarray(part, dtype=np.float64))
            for part_name, part in proposal
        )
    else:
        proposal_parts = ((name, np.asarray(proposal, dtype=np.float64)),)
    represented_parts = tuple(
        (part_name, *_selling_edge_flux_zonotope(laplacian, part))
        for part_name, part in proposal_parts
    )
    generator = sparse.hstack(
        [part[1] for part in represented_parts], format="csc")
    complete_proposal = np.sum(
        np.stack([part for _part_name, part in proposal_parts]), axis=0)
    reconstructed = np.asarray(
        generator @ np.ones(generator.shape[1])).reshape(residual.shape)
    part_diagnostics = {
        part_name: diagnostic
        for part_name, _generator, diagnostic in represented_parts
    }
    representation = {
        "part_names": tuple(part_diagnostics),
        "part_diagnostics": part_diagnostics,
        "edge_count": int(sum(
            diagnostic["edge_count"]
            for diagnostic in part_diagnostics.values()
        )),
        "zero_mode_count": int(sum(
            diagnostic["zero_mode_count"]
            for diagnostic in part_diagnostics.values()
        )),
        "connected_component_count": int(max(
            (
                diagnostic["connected_component_count"]
                for diagnostic in part_diagnostics.values()
            ),
            default=0,
        )),
        "proposal_reconstruction_maximum_error": float(np.max(np.abs(
            reconstructed - complete_proposal))),
        "edge_antisymmetry_column_sum_error": float(max(
            (
                diagnostic["edge_antisymmetry_column_sum_error"]
                for diagnostic in part_diagnostics.values()
            ),
            default=0.0,
        )),
    }
    # residual_lower <= residual - A alpha <= residual_upper
    transfer_lower = residual - residual_upper
    transfer_upper = residual - residual_lower
    coefficient_lower, coefficient_upper, contraction = (
        _contract_linear_zonotope_box(
            generator, transfer_lower, transfer_upper))
    coefficient_center = 0.5 * (
        coefficient_lower + coefficient_upper)
    coefficient_radius = 0.5 * (
        coefficient_upper - coefficient_lower)
    midpoint_transfer = np.asarray(
        generator @ coefficient_center).reshape(residual.shape)
    transfer_radius = np.asarray(
        abs(generator) @ coefficient_radius).reshape(residual.shape)
    posterior_midpoint = midpoint_transfer
    return {
        "name": name,
        "generator": generator,
        "coefficient_lower": coefficient_lower,
        "coefficient_upper": coefficient_upper,
        "coefficient_center": coefficient_center,
        "coefficient_radius": coefficient_radius,
        "midpoint_transfer": midpoint_transfer,
        "transfer_radius": transfer_radius,
        "transfer_lower": midpoint_transfer - transfer_radius,
        "transfer_upper": midpoint_transfer + transfer_radius,
        "midpoint_transfer_action": float(np.mean(
            midpoint_transfer * midpoint_transfer)),
        "proposal_action": float(np.mean(
            complete_proposal * complete_proposal)),
        "full_proposal_in_contracted_box": bool(np.all(
            coefficient_upper >= 1.0 - 64.0 * np.finfo(float).eps)),
        "zero_transfer_in_contracted_box": bool(np.all(
            coefficient_lower <= 64.0 * np.finfo(float).eps)),
        "representation": representation,
        "contraction": contraction,
        "posterior_midpoint_increment": posterior_midpoint,
    }


def zonotopic_edge_flux_state_2d(
    observation: np.ndarray,
    *,
    initial_posterior: np.ndarray | None = None,
) -> dict[str, Any]:
    """Construct one phase/curvature mixture generation without collapse."""
    image = _validate(observation)
    if initial_posterior is None:
        _base, _base_residual, scale = (
            causal_scale_transport_observation_2d(image))
        posterior = np.asarray(
            scale["readouts"]["phase_susceptibility"],
            dtype=np.float64,
        ).copy()
    else:
        posterior = _validate(initial_posterior).copy()
        if posterior.shape != image.shape:
            raise ValueError("initial posterior must align with observation")
        scale = {"status": "caller-supplied initial posterior"}
    residual = image - posterior

    posterior_phase, posterior_phase_diagnostic = (
        _phase_action_authority(posterior))
    smoothed_posterior, posterior_smoothing = _phase_screened_smooth(
        posterior, posterior, residual, posterior_phase)
    posterior_shed = posterior - smoothed_posterior
    posterior = smoothed_posterior
    residual = residual + posterior_shed

    residual_phase, residual_phase_diagnostic = (
        _phase_action_authority(residual))
    metric = continual_transport_metric(posterior, residual * residual)
    laplacian, _markov, stencil = _continual_flux_laplacian(
        metric, np.ones_like(image))
    maximum_degree = float(stencil["maximum_degree"])
    smoothed_residual = (
        _screened_transport(
            laplacian,
            1.0 / maximum_degree,
            residual[None, ...],
        )[0]
        if maximum_degree > 0.0 else residual.copy()
    )
    phase_proposal = residual_phase * smoothed_residual
    raw_curvature, curvature_relation = _cavity_residual_relation(
        posterior, residual, laplacian, maximum_degree)
    curvature_proposal = (
        _screened_transport(
            laplacian,
            1.0 / maximum_degree,
            raw_curvature[None, ...],
        )[0]
        if maximum_degree > 0.0 else raw_curvature
    )

    witness_centre, witness_variance, witness_radius, witness_diagnostic = (
        directional_noise_witnesses(image, posterior))
    raw_witness_lower = witness_centre - witness_radius
    raw_witness_upper = witness_centre + witness_radius
    # The target-excluded witnesses are empirical alternatives, not a proved
    # noise support set.  Their safe enclosure must retain the exact residual
    # already in state.  Otherwise an under-covered four-member witness family
    # can falsely "falsify" the observation that generated it.
    residual_lower = np.minimum(raw_witness_lower, residual)
    residual_upper = np.maximum(raw_witness_upper, residual)
    components = (
        _bounded_flux_component(
            "posterior_shed_lineage",
            laplacian,
            posterior_shed,
            residual,
            residual_lower,
            residual_upper,
        ),
        _bounded_flux_component(
            "reciprocal_phase",
            laplacian,
            phase_proposal,
            residual,
            residual_lower,
            residual_upper,
        ),
        _bounded_flux_component(
            "posterior_curvature_cavity",
            laplacian,
            curvature_proposal,
            residual,
            residual_lower,
            residual_upper,
        ),
        _bounded_flux_component(
            "shed_lineage_plus_reciprocal_phase",
            laplacian,
            (
                ("posterior_shed_lineage", posterior_shed),
                ("reciprocal_phase", phase_proposal),
            ),
            residual,
            residual_lower,
            residual_upper,
        ),
        _bounded_flux_component(
            "shed_lineage_plus_curvature_cavity",
            laplacian,
            (
                ("posterior_shed_lineage", posterior_shed),
                ("posterior_curvature_cavity", curvature_proposal),
            ),
            residual,
            residual_lower,
            residual_upper,
        ),
    )
    midpoint_posteriors = {
        component["name"]: posterior + component[
            "posterior_midpoint_increment"]
        for component in components
    }
    midpoint_residuals = {
        name: image - member
        for name, member in midpoint_posteriors.items()
    }
    return {
        "status": (
            "one-generation bounded Selling-edge flux mixture; nonlinear "
            "set push-forward unresolved"
        ),
        "theory_status": (
            "shed lineage, phase, and curvature remain separate outer "
            "zonotopes; passing bounded residual intersection creates no "
            "probability"
        ),
        "posterior_after_erosion": posterior,
        "residual_after_erosion": residual,
        "posterior_shed": posterior_shed,
        "components": components,
        "midpoint_posteriors_for_audit_only": midpoint_posteriors,
        "midpoint_residuals_for_audit_only": midpoint_residuals,
        "residual_witness_centre": witness_centre,
        "residual_witness_variance": witness_variance,
        "residual_witness_radius": witness_radius,
        "residual_witness_lower": residual_lower,
        "residual_witness_upper": residual_upper,
        "raw_residual_witness_lower": raw_witness_lower,
        "raw_residual_witness_upper": raw_witness_upper,
        "raw_witness_exclusion_fraction": float(np.mean(
            (residual < raw_witness_lower) | (residual > raw_witness_upper)
        )),
        "observation_recomposition_error": float(np.max(np.abs(
            posterior + residual - image))),
        "midpoint_recomposition_errors": {
            name: float(np.max(np.abs(
                midpoint_posteriors[name]
                + midpoint_residuals[name]
                - image
            )))
            for name in midpoint_posteriors
        },
        "posterior_phase": posterior_phase_diagnostic,
        "residual_phase": residual_phase_diagnostic,
        "posterior_smoothing": posterior_smoothing,
        "curvature_relation": {
            key: value
            for key, value in curvature_relation.items()
            if np.isscalar(value)
        },
        "witness": witness_diagnostic,
        "stencil": stencil,
        "base": scale,
    }


__all__ = [
    "zonotopic_edge_flux_state_2d",
]
