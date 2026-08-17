#!/usr/bin/env python3
"""Exact GDL solve over the Bruun/DIP seventeen-square packet chart.

The pairwise polygon-area energy from ``spectral_pose_transport.py`` defines
a finite factor graph with one packet variable per persistent square identity.
For the current chart that graph has induced treewidth four.  Min-sum variable
elimination (the generalized distributive law) therefore finds the exact
global minimum over ``K**17`` configurations using tables no larger than
``K**5``.  There is no tensor truncation, Euclidean descent, candidate score,
or clearance term.

This is the zero-temperature endpoint of the positive configuration
transport: all packet directions remain represented until one exact backward
preimage pass emits the minimizing Euclidean chart.  SAT clearance is computed
only afterward as an independent feasibility audit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chip_transport import write_svg
from floorplan_banach import physical_area_energy
from geometry import SQUARE_COUNT, capacity_state
from reference_chart import REFERENCE_SIDE
from spectral_pose_transport import (
    SpectralPoseTransportConfig,
    scaled_reference_chart,
    spectral_physical_energies,
    spectral_pose_packets,
)


@dataclass(frozen=True)
class Factor:
    scope: tuple[int, ...]
    value: np.ndarray


@dataclass(frozen=True)
class EliminationRecord:
    variable: int
    remaining_scope: tuple[int, ...]
    argmin: np.ndarray


def interaction_graph(
    wall_energy: np.ndarray,
    pair_energy: np.ndarray,
    tolerance: float = 1.0e-15,
) -> tuple[dict[int, set[int]], list[Factor]]:
    """Create the physical-area factor graph, omitting identically zero pairs."""

    wall = np.asarray(wall_energy, dtype=np.float64)
    pair = np.asarray(pair_energy, dtype=np.float64)
    if wall.ndim != 2 or wall.shape[0] != SQUARE_COUNT:
        raise ValueError("wall energy must have shape (17, K)")
    packet_count = wall.shape[1]
    if pair.shape != (SQUARE_COUNT, SQUARE_COUNT, packet_count, packet_count):
        raise ValueError("pair energy has the wrong shape")
    adjacency = {variable: set() for variable in range(SQUARE_COUNT)}
    factors = [
        Factor((variable,), wall[variable].copy())
        for variable in range(SQUARE_COUNT)
    ]
    for first in range(SQUARE_COUNT):
        for second in range(first + 1, SQUARE_COUNT):
            value = pair[first, second]
            if float(np.max(value)) <= float(tolerance):
                continue
            adjacency[first].add(second)
            adjacency[second].add(first)
            factors.append(Factor((first, second), value.copy()))
    return adjacency, factors


def min_fill_order(
    adjacency: dict[int, set[int]],
) -> tuple[list[int], int, list[dict]]:
    """Deterministic minimum-fill elimination order and induced width."""

    graph = {variable: set(neighbor) for variable, neighbor in adjacency.items()}
    order = []
    trace = []
    induced_width = 0
    while graph:
        def score(variable: int) -> tuple[int, int, int]:
            neighbor = sorted(graph[variable])
            missing = sum(
                second not in graph[first]
                for position, first in enumerate(neighbor)
                for second in neighbor[:position]
            )
            return int(missing), len(neighbor), variable

        variable = min(graph, key=score)
        neighbor = sorted(graph[variable])
        missing = score(variable)[0]
        induced_width = max(induced_width, len(neighbor))
        trace.append(
            {
                "variable": variable,
                "neighbor_count": len(neighbor),
                "fill_edges": missing,
            }
        )
        for position, first in enumerate(neighbor):
            for second in neighbor[:position]:
                graph[first].add(second)
                graph[second].add(first)
        for other in neighbor:
            graph[other].discard(variable)
        del graph[variable]
        order.append(variable)
    return order, induced_width, trace


def _expand_factor(
    factor: Factor,
    union_scope: tuple[int, ...],
    packet_count: int,
) -> np.ndarray:
    shape = [1] * len(union_scope)
    for variable in factor.scope:
        shape[union_scope.index(variable)] = packet_count
    return factor.value.reshape(shape)


def exact_min_sum(
    factors: list[Factor],
    order: list[int],
    packet_count: int,
) -> tuple[float, np.ndarray, dict]:
    """Exact min-sum GDL with a single backward identity-preimage pass."""

    active = list(factors)
    records: list[EliminationRecord] = []
    maximum_table_entries = 0
    total_table_entries = 0
    for variable in order:
        bucket = [factor for factor in active if variable in factor.scope]
        active = [factor for factor in active if variable not in factor.scope]
        if not bucket:
            records.append(
                EliminationRecord(variable, (), np.asarray(0, dtype=np.int16))
            )
            continue
        union_scope = tuple(sorted({item for factor in bucket for item in factor.scope}))
        table_shape = (packet_count,) * len(union_scope)
        combined = np.zeros(table_shape, dtype=np.float64)
        for factor in bucket:
            combined += _expand_factor(factor, union_scope, packet_count)
        entries = int(combined.size)
        maximum_table_entries = max(maximum_table_entries, entries)
        total_table_entries += entries
        axis = union_scope.index(variable)
        reduced = np.min(combined, axis=axis)
        argmin = np.argmin(combined, axis=axis).astype(np.int16)
        remaining_scope = tuple(item for item in union_scope if item != variable)
        records.append(EliminationRecord(variable, remaining_scope, argmin))
        if remaining_scope:
            active.append(Factor(remaining_scope, reduced))
        else:
            active.append(Factor((), np.asarray(reduced)))

    constant = sum(
        float(np.asarray(factor.value)) for factor in active if not factor.scope
    )
    if any(factor.scope for factor in active):
        raise AssertionError("elimination left a nonconstant factor")

    assignment: dict[int, int] = {}
    for record in reversed(records):
        if record.remaining_scope:
            index = tuple(assignment[variable] for variable in record.remaining_scope)
            choice = int(record.argmin[index])
        else:
            choice = int(record.argmin)
        assignment[record.variable] = choice
    variable_count = max(order) + 1 if order else 0
    packets = np.asarray(
        [assignment[variable] for variable in range(variable_count)],
        dtype=np.int32,
    )
    return constant, packets, {
        "maximum_table_entries": maximum_table_entries,
        "total_table_entries": total_table_entries,
        "backward_records": len(records),
    }


def assignment_energy(
    assignment: np.ndarray,
    wall_energy: np.ndarray,
    pair_energy: np.ndarray,
) -> float:
    index = np.asarray(assignment, dtype=np.int64)
    total = float(np.sum(wall_energy[np.arange(SQUARE_COUNT), index]))
    for first in range(SQUARE_COUNT):
        for second in range(first + 1, SQUARE_COUNT):
            total += float(pair_energy[first, second, index[first], index[second]])
    return total


def solve_exact_packet_transport(
    config: SpectralPoseTransportConfig,
    *,
    base: np.ndarray | None = None,
) -> dict:
    source = (
        scaled_reference_chart(config.side)
        if base is None
        else np.asarray(base, dtype=np.float64)
    )
    packets = spectral_pose_packets(source, config)
    wall, pair = spectral_physical_energies(packets, config.side)
    adjacency, factors = interaction_graph(wall, pair)
    order, induced_width, elimination_trace = min_fill_order(adjacency)
    minimum, selected_packets, solve_record = exact_min_sum(
        factors, order, config.packet_count
    )
    direct_energy = assignment_energy(selected_packets, wall, pair)
    if abs(minimum - direct_energy) > 1.0e-10:
        raise AssertionError("GDL minimum and direct physical factor energy disagree")
    selected = packets[np.arange(SQUARE_COUNT), selected_packets]
    physical, _, overlap_area, wall_area = physical_area_energy(
        selected, config.side, 1.0
    )
    audit = capacity_state(selected, config.side)
    return {
        "method": "exact_bruun_dip_packet_gdl_transport",
        "config": asdict(config),
        "implicit_equidistant_configurations": str(
            config.packet_count ** SQUARE_COUNT
        ),
        "factor_graph_edges": int(sum(map(len, adjacency.values())) // 2),
        "induced_treewidth": induced_width,
        "elimination_order": order,
        "elimination_trace": elimination_trace,
        "exact_global_factor_energy": minimum,
        "direct_factor_energy": direct_energy,
        "physical_area_energy": float(physical),
        "physical_overlap_area": float(overlap_area),
        "physical_wall_area": float(wall_area),
        "selected_packets": selected_packets.tolist(),
        "terminal_sat_audit": {
            "minimum_clearance": float(audit.minimum_clearance),
            "overlap_residual": float(audit.overlap_residual),
            "worst_penetration": max(-float(audit.minimum_clearance), 0.0),
        },
        "solve": solve_record,
        "poses": selected.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=float, default=REFERENCE_SIDE)
    parser.add_argument("--packets", type=int, default=16)
    parser.add_argument("--dip-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--translation-radius", type=float, default=0.08)
    parser.add_argument("--phase-radius", type=float, default=0.08)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    config = SpectralPoseTransportConfig(
        side=args.side,
        packet_count=args.packets,
        dip_level=args.dip_level,
        seed=args.seed,
        translation_radius=args.translation_radius,
        phase_radius=args.phase_radius,
    )
    result = solve_exact_packet_transport(config)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.svg is not None:
        write_svg(args.svg, np.asarray(result["poses"]), config.side)


if __name__ == "__main__":
    main()
