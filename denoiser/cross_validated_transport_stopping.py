"""Observation-only stopping probe for cross-chart transport Chambolle.

The primary denoising trajectory is accompanied by four strictly disjoint
CONV* observer lanes.  Lane ``q`` is reconstructed from parity sublattice
``q`` alone and is scored only at the other three parities.  Consequently the
three predictions scored at a target never sampled that target.

Stopping minimizes a complete squared action made from

* the error of the three-observer barycentre against the unchanged source;
* primary displacement that the original noise authority did not authorize.

The displacement action is divided by three because it is paired with the
variance of a barycentre of three target-excluded observations.  Between two
observer generations both the primary estimate and observer barycentre are
affine, so the exact minimum is a scalar quadratic projection.  The returned
endpoint is therefore continuous between generations and has no fitted
threshold, noise label, or integer-cycle decision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .continual_eikonal_noise_transport_2d import (
        _continual_flux_laplacian,
        _transport_phase_statistics,
        continual_transport_metric,
        phase_covector_noise_authority,
    )
    from .dcnt import transport_uncertainty
    from .selling_chambolle import (
        _contract_selling_once,
        cross_chart_phase_covector_statistics,
        transported_noise_authority,
    )
    from .transport_chambolle import (
        TransportChambolleResolution,
        transport_flux_body,
        transport_flux_bodies,
    )
except ImportError:  # pragma: no cover - direct research-script execution
    from continual_eikonal_noise_transport_2d import (
        _continual_flux_laplacian,
        _transport_phase_statistics,
        continual_transport_metric,
        phase_covector_noise_authority,
    )
    from dcnt import transport_uncertainty
    from selling_chambolle import (
        _contract_selling_once,
        cross_chart_phase_covector_statistics,
        transported_noise_authority,
    )
    from transport_chambolle import (
        TransportChambolleResolution,
        transport_flux_body,
        transport_flux_bodies,
    )


Array = np.ndarray
_EPS = np.finfo(np.float64).eps


def _primary_trajectory(
    observation: Array,
    resolution: TransportChambolleResolution,
) -> tuple[list[Array], Array, Array, list[float]]:
    image = np.asarray(observation, dtype=np.float64)
    current = image.copy()
    states = [current.copy()]
    body = transport_flux_body(current)
    fixed_phase = cross_chart_phase_covector_statistics(
        current, body.chart_family)[:2]
    metric = continual_transport_metric(
        body.centre,
        body.predictive_scale**2 + body.removable_amplitude**2,
    )
    _laplacian, statistic_transport, _flux = _continual_flux_laplacian(
        metric, np.ones_like(image))
    numerator, denominator = _transport_phase_statistics(
        statistic_transport, fixed_phase[0], fixed_phase[1])
    original_phase_authority, _phase_diagnostic = (
        phase_covector_noise_authority(numerator, denominator))
    original_authority: Array | None = None
    phase_gaps: list[float] = []
    maximum_cycles = int(resolution.maximum_observer_cycles)
    for cycle in range(maximum_cycles):
        phase_gaps.append(float(
            np.mean(body.removable_amplitude**2)
            - np.mean(body.coherence * body.predictive_scale**2)
        ))
        authority, metric, _diagnostic = transported_noise_authority(
            body, fixed_phase)
        if original_authority is None:
            original_authority = authority.copy()
        admitted = body.removable_amplitude * authority
        admitted_body = replace(
            body, removable_amplitude=np.ascontiguousarray(admitted))
        graph_candidate, _inner = _contract_selling_once(
            current, admitted_body, resolution, metric_fields=metric)
        removable_rms = float(np.sqrt(np.mean(admitted * admitted)))
        predictive_rms = float(np.sqrt(np.mean(
            body.predictive_scale * body.predictive_scale)))
        magnitude = max(
            float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
        if removable_rms == 0.0:
            detail_phase = 0.0
        elif predictive_rms <= 64.0 * _EPS * magnitude:
            detail_phase = 1.0
        else:
            detail_phase = min(1.0, removable_rms / predictive_rms)
        current = (
            (1.0 - detail_phase) * graph_candidate
            + detail_phase * (current - admitted)
        )
        states.append(np.ascontiguousarray(current))
        if cycle + 1 < maximum_cycles:
            body = transport_flux_body(current)
    assert original_authority is not None
    terminal_body = transport_flux_body(current)
    phase_gaps.append(float(
        np.mean(terminal_body.removable_amplitude**2)
        - np.mean(
            terminal_body.coherence * terminal_body.predictive_scale**2)
    ))
    return states, original_authority, original_phase_authority, phase_gaps


def _trajectory_value(states: list[Array], coordinate: float) -> Array:
    value = float(np.clip(coordinate, 0.0, len(states) - 1))
    lower = min(int(np.floor(value)), len(states) - 1)
    upper = min(lower + 1, len(states) - 1)
    fraction = value - lower
    return np.ascontiguousarray(
        states[lower] + fraction * (states[upper] - states[lower]))


def _continuous_phase_balance(phase_gaps: list[float]) -> float:
    """Locate the first exact zero of the affine inter-generation gap."""

    gap = np.asarray(phase_gaps, dtype=np.float64)
    if gap[0] <= 0.0:
        return 0.0
    for cycle in range(1, gap.size):
        if gap[cycle] <= 0.0:
            denominator = gap[cycle - 1] - gap[cycle]
            fraction = float(np.clip(
                gap[cycle - 1] / max(denominator, _EPS), 0.0, 1.0))
            return float(cycle - 1 + fraction)
    return float(gap.size - 1)


def _observer_population_trajectory(
    lanes: Array,
    resolution: TransportChambolleResolution,
) -> list[Array]:
    """Evolve four independent lanes in one component-batched transport."""

    current = np.asarray(lanes, dtype=np.float64).copy()
    if current.ndim != 3:
        raise ValueError("observer population must have shape BxHxW")
    states = [current.copy()]
    bodies = transport_flux_bodies(current)
    fixed_phases = [
        cross_chart_phase_covector_statistics(
            current[index], body.chart_family)[:2]
        for index, body in enumerate(bodies)
    ]
    maximum_cycles = int(resolution.maximum_observer_cycles)
    for cycle in range(maximum_cycles):
        next_population = []
        for index, body in enumerate(bodies):
            authority, _metric, _diagnostic = transported_noise_authority(
                body, fixed_phases[index])
            next_population.append(
                current[index] - body.removable_amplitude * authority)
        current = np.ascontiguousarray(np.stack(next_population))
        states.append(np.ascontiguousarray(current))
        if cycle + 1 < maximum_cycles:
            bodies = transport_flux_bodies(current)
    return states


def _held_out_barycentre_trajectory(
    observation: Array,
    resolution: TransportChambolleResolution,
) -> tuple[list[Array], list[Array], Array, dict[str, float]]:
    image = np.asarray(observation, dtype=np.float64)
    law = transport_uncertainty(image)
    initial_lanes: list[Array] = []
    validity: list[Array] = []
    for chart in range(4):
        valid = np.isfinite(law.family[chart])
        # The missing sites are precisely this chart's source anchors.  Put
        # those samples back: the complete lane still depends on this one
        # parity sublattice and no other source samples.
        lane = np.where(valid, law.family[chart], image)
        initial_lanes.append(lane)
        validity.append(valid)
    populations = _observer_population_trajectory(
        np.stack(initial_lanes), resolution)
    valid_family = np.stack(validity)
    count = np.sum(valid_family, axis=0)
    if not np.all(count == 3):
        raise RuntimeError("each target must have three disjoint observers")
    barycentres: list[Array] = []
    for cycle in range(int(resolution.maximum_observer_cycles) + 1):
        family = populations[cycle]
        barycentres.append(np.sum(
            np.where(valid_family, family, 0.0), axis=0) / count)
    return barycentres, populations, valid_family, {
        "observer_count": float(np.min(count)),
        "maximum_observer_count": float(np.max(count)),
    }


def _quadratic_action(
    source: Array,
    estimate: Array,
    observer: Array,
    protection: Array,
    observer_count: float,
) -> float:
    observer_error = observer - source
    protected_displacement = (source - estimate) * protection
    return float(
        np.mean(observer_error * observer_error)
        + np.mean(protected_displacement * protected_displacement)
        / observer_count
    )


def _continuous_action_endpoint(
    source: Array,
    estimates: list[Array],
    observers: list[Array],
    protection: Array,
    observer_count: float,
) -> tuple[Array, dict[str, Any]]:
    best_action = np.inf
    best_coordinate = 0.0
    best_estimate = estimates[0]
    trace = [
        _quadratic_action(
            source, estimate, observer, protection, observer_count)
        for estimate, observer in zip(estimates, observers)
    ]
    for cycle in range(len(estimates) - 1):
        observer_error = observers[cycle] - source
        observer_step = observers[cycle + 1] - observers[cycle]
        protected_displacement = (
            (source - estimates[cycle]) * protection)
        protected_step = (
            (estimates[cycle] - estimates[cycle + 1]) * protection)
        linear = (
            np.mean(observer_error * observer_step)
            + np.mean(protected_displacement * protected_step)
            / observer_count
        )
        quadratic = (
            np.mean(observer_step * observer_step)
            + np.mean(protected_step * protected_step) / observer_count
        )
        fraction = float(np.clip(
            -linear / max(float(quadratic), _EPS), 0.0, 1.0))
        estimate = (
            estimates[cycle]
            + fraction * (estimates[cycle + 1] - estimates[cycle]))
        observer = (
            observers[cycle]
            + fraction * (observers[cycle + 1] - observers[cycle]))
        action = _quadratic_action(
            source, estimate, observer, protection, observer_count)
        if action < best_action:
            best_action = action
            best_coordinate = cycle + fraction
            best_estimate = estimate
    terminal_action = trace[-1]
    if terminal_action < best_action:
        best_action = terminal_action
        best_coordinate = float(len(estimates) - 1)
        best_estimate = estimates[-1]
    return np.ascontiguousarray(best_estimate), {
        "continuous_observer_coordinate": float(best_coordinate),
        "minimum_complete_action": float(best_action),
        "generation_action_trace": [float(value) for value in trace],
    }


def _continuous_population_action_endpoint(
    source: Array,
    estimates: list[Array],
    populations: list[Array],
    valid: Array,
    protection: Array,
    observer_count: float,
) -> tuple[Array, dict[str, Any]]:
    """Minimize complete lane action before its barycentric collapse."""

    source_family = source[None, ...]
    valid_count = float(np.sum(valid))

    def action(estimate: Array, population: Array) -> float:
        error = np.where(valid, population - source_family, 0.0)
        protected = (source - estimate) * protection
        return float(
            np.sum(error * error) / valid_count
            + np.mean(protected * protected) / observer_count)

    trace = [
        action(estimate, population)
        for estimate, population in zip(estimates, populations)
    ]
    best_action = np.inf
    best_coordinate = 0.0
    best_estimate = estimates[0]
    for cycle in range(len(estimates) - 1):
        error = np.where(
            valid, populations[cycle] - source_family, 0.0)
        observer_step = np.where(
            valid, populations[cycle + 1] - populations[cycle], 0.0)
        protected = (source - estimates[cycle]) * protection
        protected_step = (
            (estimates[cycle] - estimates[cycle + 1]) * protection)
        linear = (
            np.sum(error * observer_step) / valid_count
            + np.mean(protected * protected_step) / observer_count
        )
        quadratic = (
            np.sum(observer_step * observer_step) / valid_count
            + np.mean(protected_step * protected_step) / observer_count
        )
        fraction = float(np.clip(
            -linear / max(float(quadratic), _EPS), 0.0, 1.0))
        estimate = (
            estimates[cycle]
            + fraction * (estimates[cycle + 1] - estimates[cycle]))
        population = (
            populations[cycle]
            + fraction * (populations[cycle + 1] - populations[cycle]))
        candidate_action = action(estimate, population)
        if candidate_action < best_action:
            best_action = candidate_action
            best_coordinate = cycle + fraction
            best_estimate = estimate
    if trace[-1] < best_action:
        best_action = trace[-1]
        best_coordinate = float(len(estimates) - 1)
        best_estimate = estimates[-1]
    return np.ascontiguousarray(best_estimate), {
        "continuous_observer_coordinate": float(best_coordinate),
        "minimum_complete_action": float(best_action),
        "generation_action_trace": [float(value) for value in trace],
    }


def _causal_phase_time_contraction(
    estimates: list[Array],
    proposed_estimate: Array,
    proposed_coordinate: float,
    phase_coordinate: float,
) -> tuple[Array, float]:
    """Contract a proposed endpoint by its witnessed phase-time support."""

    phase_estimate = _trajectory_value(estimates, phase_coordinate)
    if proposed_coordinate <= phase_coordinate:
        continuation = 1.0
        estimate = proposed_estimate
    elif proposed_coordinate <= _EPS:
        continuation = 0.0
        estimate = phase_estimate
    else:
        continuation = float(np.clip(
            phase_coordinate / proposed_coordinate, 0.0, 1.0))
        estimate = phase_estimate + continuation * (
            proposed_estimate - phase_estimate)
    return np.ascontiguousarray(estimate), continuation


def denoise_cross_validated_transport_chambolle(
    observation: Array,
    resolution: TransportChambolleResolution = TransportChambolleResolution(),
    *,
    return_diagnostics: bool = True,
    protection_source: str = "union authority",
    observer_action_source: str = "complete population",
) -> tuple[Array, dict[str, Any]] | Array:
    """Resolve a continuous endpoint using only target-excluded evidence."""

    image = np.asarray(observation, dtype=np.float64)
    estimates, authority, phase_authority, phase_gaps = _primary_trajectory(
        image, resolution)
    observers, populations, valid, observer_diagnostic = (
        _held_out_barycentre_trajectory(image, resolution))
    if protection_source == "union authority":
        protection = 1.0 - authority
    elif protection_source == "phase authority":
        protection = 1.0 - phase_authority
    else:
        raise ValueError("unknown stopping protection source")
    if observer_action_source == "barycentre":
        proposed_estimate, endpoint = _continuous_action_endpoint(
            image,
            estimates,
            observers,
            protection,
            observer_diagnostic["observer_count"],
        )
    elif observer_action_source == "complete population":
        proposed_estimate, endpoint = _continuous_population_action_endpoint(
            image,
            estimates,
            populations,
            valid,
            protection,
            observer_diagnostic["observer_count"],
        )
    else:
        raise ValueError("unknown observer action source")
    phase_coordinate = _continuous_phase_balance(phase_gaps)
    proposed_coordinate = float(endpoint["continuous_observer_coordinate"])
    estimate, continuation = _causal_phase_time_contraction(
        estimates,
        proposed_estimate,
        proposed_coordinate,
        phase_coordinate,
    )
    diagnostic = {
        "method": "cross-validated continuous transport Chambolle",
        "status": "complete observer/protection action minimum",
        "protection_source": protection_source,
        "observer_action_source": observer_action_source,
        **observer_diagnostic,
        **endpoint,
        "proposed_action_coordinate": proposed_coordinate,
        "phase_balance_coordinate": phase_coordinate,
        "causal_temporal_continuation": continuation,
        "phase_gap_trace": phase_gaps,
        "residual": image - estimate,
        "recomposition_error": float(np.max(np.abs(
            estimate + (image - estimate) - image))),
    }
    return (estimate, diagnostic) if return_diagnostics else estimate


def run(
    size: int,
    output: Path,
    protection_source: str = "union authority",
    observer_action_source: str = "complete population",
) -> dict[str, Any]:
    from .probe_transport_chambolle import CASES
    from .run_2d_denoiser_battery import metrics, sources
    from .sample_series import corrupt

    selected = {
        key: value for key, value in sources(size).items()
        if key in ("cameraman", "geometric interfaces", "woven chirps")
    }
    rows: list[dict[str, Any]] = []
    for source_name, truth in selected.items():
        for case_name, kind, amount, density in CASES:
            observation = (
                truth.copy() if kind == "none" else corrupt(
                    truth, kind, amount=amount, density=density, seed=9100)
            )
            estimate, diagnostic = denoise_cross_validated_transport_chambolle(
                observation,
                protection_source=protection_source,
                observer_action_source=observer_action_source,
            )
            rows.append({
                "source": source_name,
                "case": case_name,
                **metrics(estimate, truth),
                "continuous_observer_coordinate": diagnostic[
                    "continuous_observer_coordinate"],
                "complete_action": diagnostic["minimum_complete_action"],
                "phase_balance_coordinate": diagnostic[
                    "phase_balance_coordinate"],
                "causal_temporal_continuation": diagnostic[
                    "causal_temporal_continuation"],
            })
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "mse", "ssim", "edge_retention", "variance_ratio",
            "continuous_observer_coordinate",
        )
    }
    report = {
        "experiment": "cross-validated continuous transport stopping",
        "size": int(size),
        "protection_source": protection_source,
        "observer_action_source": observer_action_source,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument(
        "--out", type=Path,
        default=Path("/tmp/cross_validated_transport_stopping.json"),
    )
    parser.add_argument(
        "--protection-source",
        choices=("union authority", "phase authority"),
        default="union authority",
    )
    parser.add_argument(
        "--observer-action-source",
        choices=("barycentre", "complete population"),
        default="complete population",
    )
    args = parser.parse_args()
    print(json.dumps(run(
        args.size,
        args.out,
        protection_source=args.protection_source,
        observer_action_source=args.observer_action_source,
    )["summary"], indent=2))


if __name__ == "__main__":
    main()
