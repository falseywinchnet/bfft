"""Selling-edge network-flow hypothesis for transport Chambolle.

The first transport-Chambolle experiment measured a good signed residual but
represented its correction with only two outgoing Cartesian flux coordinates.
This module tests the stronger inheritance shared by the V3 segmenter, the
geometry-conditioned codec, CONV*, Meyer decomposition, and High Vision:
transport the residual in the achieving coordinate system and retain its exact
inverse relation.

For a frozen observer generation, an SPD structure/noise metric is reduced by
Selling into a positive lattice graph.  Its oriented incidence ``B`` is the
complete correction coordinate.  The convex contraction is

    min_{|p_e| <= h_e} 1/2 ||r_N + B^T p||^2,

where the edge capacity ``h_e`` is the union of the two endpoint allocations.
Each point distributes its removable amplitude over incident edges in
proportion to Selling conductance, so its allocations sum exactly to that
amplitude.  ``B^T p`` has exactly zero total mass for every iterate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import sparse

try:
    from .continual_eikonal_noise_transport_2d import (
        _continual_flux_laplacian,
        _transport_phase_statistics,
        continual_transport_metric,
        phase_covector_noise_authority,
        phase_covector_sufficient_statistics,
    )
    from .continuous_source_transport import selling_decomposition
    from .dcnt import _validate_image, transport_uncertainty
    from .transport_chambolle import (
        TransportChambolleResolution,
        TransportFluxBody,
        transport_flux_body,
    )
except ImportError:  # pragma: no cover - direct research-script execution
    from continual_eikonal_noise_transport_2d import (
        _continual_flux_laplacian,
        _transport_phase_statistics,
        continual_transport_metric,
        phase_covector_noise_authority,
        phase_covector_sufficient_statistics,
    )
    from continuous_source_transport import selling_decomposition
    from dcnt import _validate_image, transport_uncertainty
    from transport_chambolle import (
        TransportChambolleResolution,
        TransportFluxBody,
        transport_flux_body,
    )


Array = np.ndarray
_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class SellingFluxGraph:
    """Undirected positive lattice graph and bounded edge-flow coordinate."""

    first: Array
    second: Array
    conductance: Array
    capacity: Array
    weighted_degree: Array
    incidence_degree: Array
    shape: tuple[int, int]
    selling_reconstruction_error: float

    @property
    def edge_count(self) -> int:
        return int(self.first.size)


def _selling_conductance_graph(
    metric_xx: Array,
    metric_xy: Array,
    metric_yy: Array,
) -> tuple[Array, Array, Array, dict[str, float]]:
    """Symmetrize the exact local Selling stencils into unique graph edges."""

    decomposition = selling_decomposition(metric_xx, metric_xy, metric_yy)
    vectors = np.asarray(decomposition["vectors"], dtype=np.int64)
    coefficient = np.asarray(decomposition["coefficient"], dtype=np.float64)
    height, width = coefficient.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    source = np.arange(height * width, dtype=np.int64).reshape(height, width)
    rows: list[Array] = []
    columns: list[Array] = []
    values: list[Array] = []
    for direction in range(3):
        for sign in (-1, 1):
            nx = xx + sign * vectors[..., direction, 0]
            ny = yy + sign * vectors[..., direction, 1]
            valid = (0 <= nx) & (nx < width) & (0 <= ny) & (ny < height)
            rows.append(source[valid])
            columns.append((ny[valid] * width + nx[valid]).astype(np.int64))
            values.append(coefficient[..., direction][valid])
    directed = sparse.coo_matrix(
        (np.concatenate(values),
         (np.concatenate(rows), np.concatenate(columns))),
        shape=(height * width, height * width),
    ).tocsr()
    directed.sum_duplicates()
    conductance = sparse.triu(
        0.5 * (directed + directed.T), k=1, format="coo")
    positive = conductance.data > 0.0
    first = conductance.row[positive].astype(np.int64)
    second = conductance.col[positive].astype(np.int64)
    weight = conductance.data[positive].astype(np.float64)
    if not weight.size:
        raise RuntimeError("Selling reduction emitted no positive edges")
    return first, second, weight, {
        "selling_reconstruction_error": float(
            decomposition["maximum_reconstruction_error"]),
        "selling_minimum_coefficient": float(
            decomposition["minimum_coefficient"]),
    }


def build_selling_flux_graph(
    body: TransportFluxBody,
    metric_fields: dict[str, Array] | None = None,
) -> SellingFluxGraph:
    """Build the residual's positive Selling graph and exact capacity split."""

    noise_variance = (
        body.predictive_scale * body.predictive_scale
        + body.removable_amplitude * body.removable_amplitude
    )
    metric = (
        continual_transport_metric(body.centre, noise_variance)
        if metric_fields is None else metric_fields
    )
    first, second, conductance, selling = _selling_conductance_graph(
        np.asarray(metric["metric_xx"]),
        np.asarray(metric["metric_xy"]),
        np.asarray(metric["metric_yy"]),
    )
    pixels = body.centre.size
    weighted_degree = np.bincount(
        np.concatenate((first, second)),
        weights=np.concatenate((conductance, conductance)),
        minlength=pixels,
    )
    incidence_degree = np.bincount(
        np.concatenate((first, second)), minlength=pixels).astype(np.int64)
    if np.any(weighted_degree <= 0.0) or np.any(incidence_degree <= 0):
        raise RuntimeError("Selling flux graph contains an isolated point")

    point_capacity = np.abs(body.removable_amplitude).ravel()
    first_share = point_capacity[first] * conductance / weighted_degree[first]
    second_share = point_capacity[second] * conductance / weighted_degree[second]
    # An edge is shared state. Permission from either endpoint must make the
    # shared coordinate available; taking the maximum is the interval union.
    edge_capacity = np.maximum(first_share, second_share)
    return SellingFluxGraph(
        first=np.ascontiguousarray(first),
        second=np.ascontiguousarray(second),
        conductance=np.ascontiguousarray(conductance),
        capacity=np.ascontiguousarray(edge_capacity),
        weighted_degree=np.ascontiguousarray(weighted_degree),
        incidence_degree=np.ascontiguousarray(incidence_degree),
        shape=body.centre.shape,
        selling_reconstruction_error=selling["selling_reconstruction_error"],
    )


def _paired_trace(field: Array, axis: int) -> Array:
    """Affine-annihilating two-sided jump trace on interior bonds."""

    value = np.moveaxis(np.asarray(field, dtype=np.float64), axis, -1)
    if value.shape[-1] < 4:
        raise ValueError("paired jump trace requires at least four samples")
    trace = np.zeros_like(value)
    trace[..., 1:-2] = (
        1.5 * (value[..., 2:-1] - value[..., 1:-2])
        - 0.5 * (value[..., 3:] - value[..., :-3])
    )
    return np.moveaxis(trace, -1, axis)


def _positive_tensor_part(
    xx: Array,
    xy: Array,
    yy: Array,
) -> tuple[Array, Array, Array]:
    tensor_trace = xx + yy
    gap = np.hypot(xx - yy, 2.0 * xy)
    high = np.maximum(0.5 * (tensor_trace + gap), 0.0)
    low = np.maximum(0.5 * (tensor_trace - gap), 0.0)
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return (
        high * cosine * cosine + low * sine * sine,
        (high - low) * cosine * sine,
        high * sine * sine + low * cosine * cosine,
    )


def _shift_chart_family(field: Array, dy: int, dx: int) -> Array:
    """Reflect a KxHxW chart family without filling target exclusions."""

    value = np.asarray(field, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("chart family must have shape KxHxW")
    padding_y = abs(int(dy))
    padding_x = abs(int(dx))
    padded = np.pad(
        value,
        ((0, 0), (padding_y, padding_y), (padding_x, padding_x)),
        mode="symmetric",
    )
    y0 = padding_y + int(dy)
    x0 = padding_x + int(dx)
    return padded[
        :, y0:y0 + value.shape[1], x0:x0 + value.shape[2]]


def cross_chart_phase_covector_statistics(
    observation: Array,
    chart_family: Array | None = None,
) -> tuple[Array, Array, dict[str, float]]:
    """Remove phase self-products between disjoint CONV* chart gatherers.

    Each parity chart is synthesized exclusively from one disjoint anchor
    sublattice.  At a target its own-anchor value remains NaN, so every
    one-sided jet below is target-excluded.  For the valid chart population,
    ordered distinct-lane moments are evaluated in closed form.  The resulting
    2x2 jet tensor is projected onto the PSD cone before its normalized phase
    numerator and denominator are read out.
    """

    image = _validate_image(observation)
    if chart_family is None:
        family = transport_uncertainty(image).family
    else:
        family = np.asarray(chart_family, dtype=np.float64)
        if family.shape != (4,) + image.shape:
            raise ValueError("chart family must have shape 4xHxW")
    numerators: list[Array] = []
    denominators: list[Array] = []
    valid_populations: list[Array] = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        minus = family - _shift_chart_family(family, -dy, -dx)
        plus = _shift_chart_family(family, dy, dx) - family
        valid = np.isfinite(minus) & np.isfinite(plus)
        population = np.sum(valid, axis=0)
        valid_populations.append(population)
        minus = np.where(valid, minus, 0.0)
        plus = np.where(valid, plus, 0.0)
        pair_count = np.maximum(population * (population - 1), 1)
        sum_minus = np.sum(minus, axis=0)
        sum_plus = np.sum(plus, axis=0)
        covariance_minus = (
            sum_minus * sum_minus - np.sum(minus * minus, axis=0)
        ) / pair_count
        covariance_plus = (
            sum_plus * sum_plus - np.sum(plus * plus, axis=0)
        ) / pair_count
        covariance_cross = (
            sum_minus * sum_plus - np.sum(minus * plus, axis=0)
        ) / pair_count
        xx, xy, yy = _positive_tensor_part(
            covariance_minus, covariance_cross, covariance_plus)
        enough = population >= 2
        numerators.append(np.where(enough, xy, 0.0))
        denominators.append(np.where(enough, 0.5 * (xx + yy), 0.0))
    numerator = np.stack(numerators)
    denominator = np.stack(denominators)
    populations = np.stack(valid_populations)
    return numerator, denominator, {
        "minimum_distinct_chart_population": float(np.min(populations)),
        "mean_distinct_chart_population": float(np.mean(populations)),
        "distinct_pair_coverage": float(np.mean(populations >= 2)),
        "phase_cauchy_violation": float(np.max(
            np.abs(numerator) - denominator)),
    }


def paired_trace_transport_metric(
    observation: Array,
    body: TransportFluxBody | None = None,
) -> dict[str, Array | float]:
    """Build an SPD metric from chart-persistent two-sided jump measure."""

    image = _validate_image(observation)
    flux_body = transport_flux_body(image) if body is None else body
    law = transport_uncertainty(image)
    completed = np.where(
        np.isfinite(law.family), law.family, law.centre[None, ...])
    trace_x = np.stack([_paired_trace(chart, 1) for chart in completed])
    trace_y = np.stack([_paired_trace(chart, 0) for chart in completed])
    mean_x = np.mean(trace_x, axis=0)
    mean_y = np.mean(trace_y, axis=0)
    dx = trace_x - mean_x
    dy = trace_y - mean_y
    variance_x = np.mean(dx * dx, axis=0)
    covariance_xy = np.mean(dx * dy, axis=0)
    variance_y = np.mean(dy * dy, axis=0)
    excess_xx, excess_xy, excess_yy = _positive_tensor_part(
        mean_x * mean_x - variance_x,
        mean_x * mean_y - covariance_xy,
        mean_y * mean_y - variance_y,
    )
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
    numerical = _EPS * magnitude * magnitude
    nuisance = (
        variance_x + variance_y + flux_body.removable_amplitude**2)
    denominator = nuisance + np.sqrt(np.maximum(
        excess_xx * excess_yy - excess_xy * excess_xy, 0.0)) + numerical
    metric_xx = 1.0 + excess_xx / denominator
    metric_xy = excess_xy / denominator
    metric_yy = 1.0 + excess_yy / denominator
    determinant = metric_xx * metric_yy - metric_xy * metric_xy
    if np.any(metric_xx <= 0.0) or np.any(determinant <= 0.0):
        raise RuntimeError("paired-trace metric left the SPD cone")
    return {
        "metric_xx": np.ascontiguousarray(metric_xx),
        "metric_xy": np.ascontiguousarray(metric_xy),
        "metric_yy": np.ascontiguousarray(metric_yy),
        "metric_determinant_minimum": float(np.min(determinant)),
        "mean_persistent_jump_action": float(np.mean(
            excess_xx + excess_yy)),
        "mean_trace_nuisance_action": float(np.mean(nuisance)),
    }


def _graph_gradient(field: Array, graph: SellingFluxGraph) -> Array:
    flat = np.asarray(field, dtype=np.float64).ravel()
    return flat[graph.second] - flat[graph.first]


def _graph_flux_readout(flux: Array, graph: SellingFluxGraph) -> Array:
    """Apply ``B^T``; the result has exactly zero total mass."""

    value = np.asarray(flux, dtype=np.float64)
    if value.shape != graph.capacity.shape:
        raise ValueError("edge flux does not match Selling graph")
    flat = (
        np.bincount(graph.second, weights=value, minlength=np.prod(graph.shape))
        - np.bincount(graph.first, weights=value, minlength=np.prod(graph.shape))
    )
    return flat.reshape(graph.shape)


def _graph_objective(
    removable: Array,
    flux: Array,
    graph: SellingFluxGraph,
) -> float:
    unexplained = removable + _graph_flux_readout(flux, graph)
    return float(0.5 * np.sum(unexplained * unexplained))


def _contract_selling_once(
    observation: Array,
    body: TransportFluxBody,
    resolution: TransportChambolleResolution,
    metric_fields: dict[str, Array] | None = None,
) -> tuple[Array, dict[str, Any]]:
    image = _validate_image(observation)
    graph = build_selling_flux_graph(body, metric_fields=metric_fields)
    capacity = graph.capacity
    if not np.any(capacity > 0.0):
        return image.copy(), {
            "status": "transport identity",
            "iterations": 0,
            "edge_count": graph.edge_count,
            "maximum_incidence_degree": int(np.max(graph.incidence_degree)),
            "contraction_objective_trace": [0.0],
            "unexplained_noise_action": 0.0,
            "mass_error": 0.0,
            "selling_reconstruction_error": graph.selling_reconstruction_error,
        }

    maximum = int(resolution.maximum_iterations)
    minimum = int(resolution.minimum_iterations)
    tolerance = float(resolution.relative_tolerance)
    maximum_degree = int(np.max(graph.incidence_degree))
    # ||B||^2 <= 2 d_max for an undirected incidence matrix.
    step = 1.0 / (2.0 * maximum_degree)
    flux = np.zeros_like(capacity)
    previous = flux.copy()
    removable = body.removable_amplitude
    readout = np.zeros_like(removable)
    previous_readout = readout.copy()
    clock = 1.0
    objective = float(0.5 * np.sum(removable * removable))
    trace = [objective]
    restarts = 0
    converged = False
    capacity_scale = max(float(np.sqrt(np.mean(capacity * capacity))), _EPS)
    for iteration in range(maximum):
        next_clock = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * clock * clock))
        momentum = (clock - 1.0) / next_clock
        extrapolated = flux + momentum * (flux - previous)
        # Incidence readout is linear.  Carry it with the FISTA state instead
        # of rebuilding it from every edge for both flux states each step.
        extrapolated_readout = (
            readout + momentum * (readout - previous_readout))
        unexplained = removable + extrapolated_readout
        gradient = _graph_gradient(unexplained, graph)
        candidate = np.clip(
            extrapolated - step * gradient, -capacity, capacity)
        candidate_readout = _graph_flux_readout(candidate, graph)
        candidate_unexplained = removable + candidate_readout
        candidate_objective = float(
            0.5 * np.sum(candidate_unexplained * candidate_unexplained))
        numerical = 128.0 * _EPS * max(abs(objective), image.size, 1.0)
        if candidate_objective > objective + numerical:
            restarts += 1
            next_clock = 1.0
            unexplained = removable + readout
            gradient = _graph_gradient(unexplained, graph)
            candidate = np.clip(flux - step * gradient, -capacity, capacity)
            candidate_readout = _graph_flux_readout(candidate, graph)
            candidate_unexplained = removable + candidate_readout
            candidate_objective = float(
                0.5 * np.sum(candidate_unexplained * candidate_unexplained))
        delta = float(np.sqrt(np.mean((candidate - flux) ** 2)))
        objective_change = abs(candidate_objective - objective)
        previous, flux = flux, candidate
        previous_readout, readout = readout, candidate_readout
        clock = next_clock
        objective = candidate_objective
        trace.append(objective)
        if (
            iteration + 1 >= minimum
            and delta <= tolerance * capacity_scale
            and objective_change <= tolerance * max(abs(objective), _EPS)
        ):
            converged = True
            break

    estimate = image + readout
    return estimate, {
        "status": "Selling flux equilibrium" if converged else "iteration ceiling",
        "iterations": int(iteration + 1),
        "restarts": int(restarts),
        "step": step,
        "edge_count": graph.edge_count,
        "maximum_incidence_degree": maximum_degree,
        "mean_edge_capacity": float(np.mean(capacity)),
        "maximum_capacity_violation": float(np.max(np.abs(flux) - capacity)),
        "contraction_objective_trace": trace,
        "unexplained_noise_action": objective,
        "mass_error": float(abs(np.sum(readout))),
        "selling_reconstruction_error": graph.selling_reconstruction_error,
    }


def denoise_selling_chambolle(
    observation: Array,
    resolution: TransportChambolleResolution = TransportChambolleResolution(),
    *,
    return_diagnostics: bool = True,
) -> tuple[Array, dict[str, Any]] | Array:
    """Run phase-stopped re-observation on the Selling edge-flow coordinate."""

    image = _validate_image(observation)
    current = image.copy()
    history: list[dict[str, Any]] = []
    phase_balanced = False
    last: dict[str, Any] = {}
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
    numerical = 256.0 * _EPS * magnitude * magnitude
    for cycle in range(int(resolution.maximum_observer_cycles)):
        body = transport_flux_body(current)
        candidate, last = _contract_selling_once(current, body, resolution)
        next_body = transport_flux_body(candidate)
        removable_action = float(np.mean(next_body.removable_amplitude**2))
        structure_action = float(np.mean(
            next_body.coherence * next_body.predictive_scale**2))
        history.append({
            "cycle": cycle + 1,
            "update_rms": float(np.sqrt(np.mean((candidate - current) ** 2))),
            "next_removable_action": removable_action,
            "next_supported_structure_action": structure_action,
            "inner": last,
        })
        current = candidate
        if removable_action <= structure_action + numerical:
            phase_balanced = True
            break

    residual = image - current
    diagnostic = {
        **last,
        "method": "Selling-edge transport Chambolle",
        "status": (
            "transport identity"
            if last.get("status") == "transport identity"
            else (
                "transport noise/structure phase balance"
                if phase_balanced else "observer-cycle numerical ceiling"
            )
        ),
        "observer_cycles": len(history),
        "observer_history": history,
        "residual": residual,
        "recomposition_error": float(np.max(np.abs(
            current + residual - image))),
        "coordinate": "positive Selling-edge incidence flow",
    }
    return (current, diagnostic) if return_diagnostics else current


def denoise_population_phase_chambolle(
    observation: Array,
    resolution: TransportChambolleResolution = TransportChambolleResolution(),
    *,
    return_diagnostics: bool = True,
) -> tuple[Array, dict[str, Any]] | Array:
    """Fuse conservative edge flow with exact detail by action population.

    The Selling contraction preserves the unexplained part of the removable
    residual because local divergence may not represent it safely.  CONV's
    exact correction coordinate instead removes the complete signed residual.
    Their phase is the dimensionless RMS amplitude quotient

        alpha = min(1, RMS(removable) / RMS(predictive scale)).

    This is the positive projective coordinate of the two measured actions;
    it introduces neither a noise class nor a fitted interpolation parameter.
    """

    image = _validate_image(observation)
    current = image.copy()
    history: list[dict[str, Any]] = []
    phase_balanced = False
    last: dict[str, Any] = {}
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
    numerical = 256.0 * _EPS * magnitude * magnitude
    for cycle in range(int(resolution.maximum_observer_cycles)):
        body = transport_flux_body(current)
        graph_candidate, last = _contract_selling_once(
            current, body, resolution)
        removable_rms = float(np.sqrt(np.mean(body.removable_amplitude**2)))
        predictive_rms = float(np.sqrt(np.mean(body.predictive_scale**2)))
        if removable_rms == 0.0:
            detail_phase = 0.0
        elif predictive_rms <= 64.0 * _EPS * magnitude:
            detail_phase = 1.0
        else:
            detail_phase = min(1.0, removable_rms / predictive_rms)
        exact_detail_candidate = current - body.removable_amplitude
        candidate = (
            (1.0 - detail_phase) * graph_candidate
            + detail_phase * exact_detail_candidate
        )
        next_body = transport_flux_body(candidate)
        removable_action = float(np.mean(next_body.removable_amplitude**2))
        structure_action = float(np.mean(
            next_body.coherence * next_body.predictive_scale**2))
        history.append({
            "cycle": cycle + 1,
            "detail_phase": detail_phase,
            "removable_rms": removable_rms,
            "predictive_rms": predictive_rms,
            "update_rms": float(np.sqrt(np.mean((candidate - current) ** 2))),
            "next_removable_action": removable_action,
            "next_supported_structure_action": structure_action,
            "inner": last,
        })
        current = candidate
        if removable_action <= structure_action + numerical:
            phase_balanced = True
            break

    residual = image - current
    diagnostic = {
        **last,
        "method": "population-phase Selling/CONV Chambolle",
        "status": (
            "transport identity"
            if last.get("status") == "transport identity"
            else (
                "transport noise/structure phase balance"
                if phase_balanced else "observer-cycle numerical ceiling"
            )
        ),
        "observer_cycles": len(history),
        "observer_history": history,
        "residual": residual,
        "recomposition_error": float(np.max(np.abs(
            current + residual - image))),
        "coordinate": (
            "projective population phase of Selling flow and exact CONV detail"
        ),
    }
    return (current, diagnostic) if return_diagnostics else current


def transported_noise_authority(
    body: TransportFluxBody,
    phase_statistics: tuple[Array, Array] | None = None,
) -> tuple[Array, dict[str, Array | float], dict[str, float | str]]:
    """Fuse transported phase and amplitude evidence without a noise class.

    The phase statistic asks whether the Meyer-removable residual lies on one
    integrable full-band covector.  Its numerator and denominator are moved
    separately by the positive Selling Markov operator; angles are never
    averaged.  The amplitude statistic is the local projective coordinate of
    removable and predictive action.  Their probabilistic union

        A = 1 - (1 - A_phase) (1 - A_amplitude)

    means either independent witness can authorize removal, while structure
    is protected only when both witnesses refuse it.
    """

    removable = np.asarray(body.removable_amplitude, dtype=np.float64)
    predictive = np.asarray(body.predictive_scale, dtype=np.float64)
    metric = continual_transport_metric(
        body.centre, predictive * predictive + removable * removable)
    _laplacian, statistic_transport, flux = _continual_flux_laplacian(
        metric, np.ones_like(removable))
    if phase_statistics is None:
        numerator, denominator = phase_covector_sufficient_statistics(removable)
        phase_source = "Meyer-removable residual"
    else:
        numerator = np.asarray(phase_statistics[0], dtype=np.float64)
        denominator = np.asarray(phase_statistics[1], dtype=np.float64)
        if numerator.shape != (4,) + removable.shape or denominator.shape != (
                4,) + removable.shape:
            raise ValueError("phase statistics must have shape 4xHxW")
        phase_source = "distinct target-excluded CONV charts"
    numerator, denominator = _transport_phase_statistics(
        statistic_transport, numerator, denominator)
    phase_authority, phase = phase_covector_noise_authority(
        numerator, denominator)
    magnitude = max(
        float(np.max(np.abs(body.centre))),
        float(np.max(np.abs(removable))),
        1.0,
    )
    numerical = 64.0 * _EPS * magnitude
    amplitude_authority = np.minimum(
        1.0, np.abs(removable) / (predictive + numerical))
    authority = 1.0 - (
        (1.0 - phase_authority) * (1.0 - amplitude_authority))
    authority = np.clip(authority, 0.0, 1.0)
    return np.ascontiguousarray(authority), metric, {
        "mean_phase_noise_authority": float(np.mean(phase_authority)),
        "mean_amplitude_noise_authority": float(np.mean(amplitude_authority)),
        "mean_union_noise_authority": float(np.mean(authority)),
        "phase_covector_defect": float(
            phase["mean_phase_covector_defect"]),
        "statistic_transport_row_sum_error": float(
            flux["transport_row_sum_error"]),
        "statistic_transport_column_sum_error": float(
            flux["transport_column_sum_error"]),
        "phase_source": phase_source,
    }


def denoise_phase_action_chambolle(
    observation: Array,
    resolution: TransportChambolleResolution = TransportChambolleResolution(),
    *,
    return_diagnostics: bool = True,
    phase_source: str = "removable residual",
) -> tuple[Array, dict[str, Any]] | Array:
    """Contract the phase-vetted residual in its positive Selling coordinate."""

    image = _validate_image(observation)
    if phase_source not in ("removable residual", "distinct charts"):
        raise ValueError("unknown phase-action source")
    current = image.copy()
    history: list[dict[str, Any]] = []
    phase_balanced = False
    last: dict[str, Any] = {}
    fixed_phase_statistics: tuple[Array, Array] | None = None
    fixed_chart_diagnostic: dict[str, float] | None = None
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
    numerical = 256.0 * _EPS * magnitude * magnitude
    for cycle in range(int(resolution.maximum_observer_cycles)):
        body = transport_flux_body(current)
        if phase_source == "distinct charts":
            if fixed_phase_statistics is None:
                numerator, denominator, fixed_chart_diagnostic = (
                    cross_chart_phase_covector_statistics(
                        current, body.chart_family))
                fixed_phase_statistics = (numerator, denominator)
            phase_statistics = fixed_phase_statistics
            chart_diagnostic = fixed_chart_diagnostic
        else:
            chart_diagnostic = None
            phase_statistics = None
        authority, metric, authority_diagnostic = transported_noise_authority(
            body, phase_statistics=phase_statistics)
        admitted_removal = body.removable_amplitude * authority
        admitted_body = replace(
            body, removable_amplitude=np.ascontiguousarray(admitted_removal))
        graph_candidate, last = _contract_selling_once(
            current, admitted_body, resolution, metric_fields=metric)
        removable_rms = float(np.sqrt(np.mean(admitted_removal**2)))
        predictive_rms = float(np.sqrt(np.mean(body.predictive_scale**2)))
        if removable_rms == 0.0:
            detail_phase = 0.0
        elif predictive_rms <= 64.0 * _EPS * magnitude:
            detail_phase = 1.0
        else:
            detail_phase = min(1.0, removable_rms / predictive_rms)
        exact_detail_candidate = current - admitted_removal
        candidate = (
            (1.0 - detail_phase) * graph_candidate
            + detail_phase * exact_detail_candidate)
        next_body = transport_flux_body(candidate)
        removable_action = float(np.mean(next_body.removable_amplitude**2))
        structure_action = float(np.mean(
            next_body.coherence * next_body.predictive_scale**2))
        history.append({
            "cycle": cycle + 1,
            "detail_phase": detail_phase,
            "removable_rms": removable_rms,
            "predictive_rms": predictive_rms,
            "update_rms": float(np.sqrt(np.mean((candidate - current) ** 2))),
            "next_removable_action": removable_action,
            "next_supported_structure_action": structure_action,
            "authority": authority_diagnostic,
            "distinct_chart_phase": chart_diagnostic,
            "inner": last,
        })
        current = candidate
        if removable_action <= structure_action + numerical:
            phase_balanced = True
            break

    residual = image - current
    diagnostic = {
        **last,
        "method": (
            "distinct-chart phase-action Chambolle"
            if phase_source == "distinct charts"
            else "transported phase-action Chambolle"),
        "status": (
            "transport identity"
            if last.get("status") == "transport identity"
            else (
                "transport noise/structure phase balance"
                if phase_balanced else "observer-cycle numerical ceiling")),
        "observer_cycles": len(history),
        "observer_history": history,
        "residual": residual,
        "recomposition_error": float(np.max(np.abs(
            current + residual - image))),
        "coordinate": (
            "positive Selling flow with transported phase/amplitude authority"
        ),
        "phase_source": phase_source,
        "phase_evidence_reused_across_cycles": (
            phase_source == "distinct charts"),
    }
    return (current, diagnostic) if return_diagnostics else current


def denoise_cross_chart_phase_chambolle(
    observation: Array,
    resolution: TransportChambolleResolution = TransportChambolleResolution(),
    *,
    return_diagnostics: bool = True,
) -> tuple[Array, dict[str, Any]] | Array:
    """Use distinct parity-chart cross moments as phase-action evidence."""

    return denoise_phase_action_chambolle(
        observation,
        resolution,
        return_diagnostics=return_diagnostics,
        phase_source="distinct charts",
    )


def denoise_paired_trace_chambolle(
    observation: Array,
    resolution: TransportChambolleResolution = TransportChambolleResolution(),
    *,
    return_diagnostics: bool = True,
) -> tuple[Array, dict[str, Any]] | Array:
    """Population-phase readout on an affine-annihilating jump metric."""

    image = _validate_image(observation)
    current = image.copy()
    history: list[dict[str, Any]] = []
    phase_balanced = False
    last: dict[str, Any] = {}
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
    numerical = 256.0 * _EPS * magnitude * magnitude
    for cycle in range(int(resolution.maximum_observer_cycles)):
        body = transport_flux_body(current)
        metric = paired_trace_transport_metric(current, body)
        graph_candidate, last = _contract_selling_once(
            current, body, resolution, metric_fields=metric)
        removable_rms = float(np.sqrt(np.mean(body.removable_amplitude**2)))
        predictive_rms = float(np.sqrt(np.mean(body.predictive_scale**2)))
        detail_phase = (
            0.0 if removable_rms == 0.0 else
            min(1.0, removable_rms / max(
                predictive_rms, 64.0 * _EPS * magnitude))
        )
        exact_detail = current - body.removable_amplitude
        candidate = (
            (1.0 - detail_phase) * graph_candidate
            + detail_phase * exact_detail)
        next_body = transport_flux_body(candidate)
        removable_action = float(np.mean(next_body.removable_amplitude**2))
        structure_action = float(np.mean(
            next_body.coherence * next_body.predictive_scale**2))
        history.append({
            "cycle": cycle + 1,
            "detail_phase": detail_phase,
            "update_rms": float(np.sqrt(np.mean((candidate - current) ** 2))),
            "next_removable_action": removable_action,
            "next_supported_structure_action": structure_action,
            "mean_persistent_jump_action": metric[
                "mean_persistent_jump_action"],
            "mean_trace_nuisance_action": metric[
                "mean_trace_nuisance_action"],
            "inner": last,
        })
        current = candidate
        if removable_action <= structure_action + numerical:
            phase_balanced = True
            break
    residual = image - current
    diagnostic = {
        **last,
        "method": "paired-trace population-phase Chambolle",
        "status": (
            "transport identity"
            if last.get("status") == "transport identity"
            else (
                "transport noise/structure phase balance"
                if phase_balanced else "observer-cycle numerical ceiling")),
        "observer_cycles": len(history),
        "observer_history": history,
        "residual": residual,
        "recomposition_error": float(np.max(np.abs(
            current + residual - image))),
        "coordinate": "chart-persistent paired BV jump trace",
    }
    return (current, diagnostic) if return_diagnostics else current


__all__ = [
    "SellingFluxGraph",
    "build_selling_flux_graph",
    "cross_chart_phase_covector_statistics",
    "denoise_cross_chart_phase_chambolle",
    "denoise_paired_trace_chambolle",
    "denoise_phase_action_chambolle",
    "denoise_population_phase_chambolle",
    "denoise_selling_chambolle",
    "paired_trace_transport_metric",
    "transported_noise_authority",
]
