#!/usr/bin/env python3
"""Rejected scalar-control ablation for the floor-plan contraction.

This was the first attempt to make the transport state seventeen-dimensional.
It is retained because it demonstrates a real contraction, but it is *not*
the intended BFFT solve: the contractive state consists of only seventeen
scalar weights attached to an already chosen Euclidean pose chart, and the
outer loop still performs ordinary Euclidean gradient descent.  It can
therefore stall in a pose-space local minimum.

A continuous floor-plan solve assigns
every point to its first-arriving square source.  It produces one local action
yield per square and one adjacency transport operator.  The only iterative
transport state is therefore ``z in R**17``:

    B(z) = e + gamma P z,       0 <= gamma < 1.

``P`` is row stochastic, so this inner ``B`` is a Banach contraction in ``l_infinity``
with constant ``gamma``.  After its unique fixed point is reached, each scalar
``z_i`` is restricted through the achieving floor-plan characteristic of
square ``i``.  All seventeen resulting pose directions descend into Euclidean
space simultaneously.  The floor plan is then rebuilt and the process repeats.

The ablation energy is an integral floor-plan energy: oriented first-arrival action,
soft physical overlap area, and physical mass outside the container.  Pairwise
SAT clearance is absent from evolution and is evaluated once at terminal
audit.  This remains a numerical experiment, not a proof of global optimality.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from chip_transport import write_svg
from geometry import SQUARE_COUNT, capacity_state, square_corners, wrap_square_phase


@dataclass(frozen=True)
class FloorplanConfig:
    side: float = 4.6
    sweeps: int = 800
    resolution: int = 72
    margin: float = 0.72
    gauge_temperature: float = 0.018
    absolute_smoothing: float = 2.0e-4
    wall_weight: float = 2.0
    action_weight: float = 0.080
    end_action_weight: float = 1.0e-6
    mass_weight: float = 0.050
    end_mass_weight: float = 1.0e-6
    anneal_fraction: float = 0.55
    discount: float = 0.72
    contraction_tolerance: float = 1.0e-12
    contraction_iterations: int = 160
    translation_trust: float = 0.030
    phase_trust: float = 0.020
    line_search_steps: int = 12
    seed: int = 0


@dataclass
class FloorplanState:
    local_cost: np.ndarray
    transport: np.ndarray
    restriction_gradient: np.ndarray
    owner: np.ndarray
    area: np.ndarray
    action_energy: np.ndarray
    packing_energy: np.ndarray
    overlap_energy: float
    wall_energy: float
    mass_energy: float
    total_energy: float


def _fractional(value: np.ndarray) -> np.ndarray:
    return value - np.floor(value)


def initial_floorplan(side: float, seed: int = 0) -> np.ndarray:
    """One owner-free low-discrepancy floor plan, never a row template."""

    index = np.arange(1, SQUARE_COUNT + 1, dtype=np.float64)
    offset = 0.5 + 0.17320508075688773 * int(seed)
    x_phase = _fractional(offset + index * ((math.sqrt(5.0) - 1.0) / 2.0))
    y_phase = _fractional(offset + index * (math.sqrt(2.0) - 1.0))
    inset = 0.56
    centers = inset + (side - 2.0 * inset) * np.column_stack((x_phase, y_phase))
    angle_phase = _fractional(index * (math.sqrt(3.0) - 1.0) + offset)
    theta = 0.24 * np.sin(2.0 * math.pi * angle_phase)
    return np.column_stack((centers, theta))


def _floor_grid(
    side: float,
    resolution: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    spacing = side / max(int(resolution), 8)
    coordinate = np.arange(
        -margin + 0.5 * spacing,
        side + margin,
        spacing,
        dtype=np.float64,
    )
    x, y = np.meshgrid(coordinate, coordinate)
    inside = (x >= 0.0) & (x <= side) & (y >= 0.0) & (y <= side)
    return x, y, inside, spacing * spacing


def _oriented_square_fields(
    poses: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: FloorplanConfig,
) -> tuple[np.ndarray, np.ndarray]:
    dx = x[None, :, :] - poses[:, 0, None, None]
    dy = y[None, :, :] - poses[:, 1, None, None]
    cosine = np.cos(poses[:, 2])[:, None, None]
    sine = np.sin(poses[:, 2])[:, None, None]
    u = cosine * dx + sine * dy
    v = -sine * dx + cosine * dy

    epsilon = max(float(config.absolute_smoothing), 1.0e-12)
    absolute_u = np.sqrt(u * u + epsilon * epsilon)
    absolute_v = np.sqrt(v * v + epsilon * epsilon)
    temperature = max(float(config.gauge_temperature), 1.0e-8)
    maximum = np.maximum(absolute_u, absolute_v)
    exp_u = np.exp(np.clip((absolute_u - maximum) / temperature, -60.0, 0.0))
    exp_v = np.exp(np.clip((absolute_v - maximum) / temperature, -60.0, 0.0))
    partition = exp_u + exp_v
    gauge = maximum + temperature * np.log(partition)
    probability_u = exp_u / partition
    probability_v = exp_v / partition
    gauge_u = probability_u * u / absolute_u
    gauge_v = probability_v * v / absolute_v

    gauge_gradient = np.empty((SQUARE_COUNT, 3) + x.shape, dtype=np.float64)
    gauge_gradient[:, 0] = -cosine * gauge_u + sine * gauge_v
    gauge_gradient[:, 1] = -sine * gauge_u - cosine * gauge_v
    gauge_gradient[:, 2] = gauge_u * v - gauge_v * u

    action = gauge * gauge
    return action, gauge_gradient


def _first_arrival_adjacency(owner: np.ndarray, cell_area: float) -> np.ndarray:
    transport = np.zeros((SQUARE_COUNT, SQUARE_COUNT), dtype=np.float64)
    for first, second in (
        (owner[:, :-1], owner[:, 1:]),
        (owner[:-1, :], owner[1:, :]),
    ):
        different = first != second
        if not np.any(different):
            continue
        left = first[different]
        right = second[different]
        np.add.at(transport, (left, right), math.sqrt(cell_area))
        np.add.at(transport, (right, left), math.sqrt(cell_area))
    row_sum = np.sum(transport, axis=1)
    isolated = row_sum <= 0.0
    transport[isolated, isolated] = 1.0
    row_sum = np.sum(transport, axis=1)
    return transport / row_sum[:, None]


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland--Hodgman intersection of two CCW convex polygons."""

    output = np.asarray(subject, dtype=np.float64)
    for edge in range(len(clip)):
        if len(output) == 0:
            break
        start = clip[edge]
        end = clip[(edge + 1) % len(clip)]
        edge_vector = end - start

        def signed(point: np.ndarray) -> float:
            relative = point - start
            return float(edge_vector[0] * relative[1] - edge_vector[1] * relative[0])

        source = output
        vertices = []
        previous = source[-1]
        previous_signed = signed(previous)
        for current in source:
            current_signed = signed(current)
            previous_inside = previous_signed >= -1.0e-14
            current_inside = current_signed >= -1.0e-14
            if current_inside != previous_inside:
                denominator = previous_signed - current_signed
                fraction = (
                    previous_signed / denominator
                    if abs(denominator) > 1.0e-30
                    else 0.5
                )
                vertices.append(previous + fraction * (current - previous))
            if current_inside:
                vertices.append(current)
            previous = current
            previous_signed = current_signed
        output = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    return output


def physical_area_energy(
    poses: np.ndarray,
    side: float,
    wall_weight: float,
) -> tuple[float, np.ndarray, float, float]:
    """Exact polygon overlap/outside energy and its 17 local attributions."""

    corners = [square_corners(pose) for pose in np.asarray(poses)]
    container = np.asarray(
        ((0.0, 0.0), (side, 0.0), (side, side), (0.0, side)),
        dtype=np.float64,
    )
    local = np.zeros(SQUARE_COUNT, dtype=np.float64)
    overlap = 0.0
    for first in range(SQUARE_COUNT):
        for second in range(first + 1, SQUARE_COUNT):
            area = _polygon_area(_clip_convex(corners[first], corners[second]))
            overlap += area
            local[first] += 0.5 * area
            local[second] += 0.5 * area
    outside = 0.0
    for square in range(SQUARE_COUNT):
        contained = _polygon_area(_clip_convex(corners[square], container))
        escaped = max(1.0 - contained, 0.0)
        outside += escaped
        local[square] += wall_weight * escaped
    wall = wall_weight * outside
    return overlap + wall, local, overlap, wall


def _physical_area_gradient(
    poses: np.ndarray,
    side: float,
    wall_weight: float,
) -> np.ndarray:
    gradient = np.zeros((SQUARE_COUNT, 3), dtype=np.float64)
    steps = (2.0e-4, 2.0e-4, 1.0e-4)
    for square in range(SQUARE_COUNT):
        for coordinate, step in enumerate(steps):
            positive = np.asarray(poses, dtype=np.float64).copy()
            negative = np.asarray(poses, dtype=np.float64).copy()
            positive[square, coordinate] += step
            negative[square, coordinate] -= step
            positive_energy = physical_area_energy(
                positive, side, wall_weight
            )[0]
            negative_energy = physical_area_energy(
                negative, side, wall_weight
            )[0]
            gradient[square, coordinate] = (
                positive_energy - negative_energy
            ) / (2.0 * step)
    return gradient


def floorplan_state(
    poses: np.ndarray,
    config: FloorplanConfig,
    *,
    with_gradient: bool = True,
) -> FloorplanState:
    """Measure one continuous first-arrival floor plan and its restrictions."""

    chart = np.asarray(poses, dtype=np.float64)
    if chart.shape != (SQUARE_COUNT, 3):
        raise ValueError("floor plan requires one (x,y,theta) row per square")
    x, y, inside, cell_area = _floor_grid(
        config.side, config.resolution, config.margin
    )
    action, gauge_gradient = _oriented_square_fields(
        chart, x, y, config
    )

    inside_action = np.where(inside[None, :, :], action, np.inf)
    owner_full = np.argmin(inside_action, axis=0)
    inside_rows = np.any(inside, axis=1)
    inside_columns = np.any(inside, axis=0)
    owner = owner_full[np.ix_(inside_rows, inside_columns)]
    owner_flat = owner_full[inside]
    area_count = np.bincount(owner_flat, minlength=SQUARE_COUNT)
    area = area_count.astype(np.float64) * cell_area

    action_energy = np.empty(SQUARE_COUNT, dtype=np.float64)
    action_gradient = np.zeros((SQUARE_COUNT, 3), dtype=np.float64)
    for square in range(SQUARE_COUNT):
        mask = inside & (owner_full == square)
        count = max(int(np.sum(mask)), 1)
        action_energy[square] = float(np.sum(action[square, mask])) / count
        action_gradient[square] = np.sum(
            2.0
            * np.sqrt(action[square, mask])[:, None]
            * gauge_gradient[square].reshape(3, -1)[:, mask.ravel()].T,
            axis=0,
        ) / count

    physical_energy, packing_energy, overlap_energy, wall_energy = (
        physical_area_energy(chart, config.side, config.wall_weight)
    )
    packing_gradient = (
        _physical_area_gradient(chart, config.side, config.wall_weight)
        if with_gradient
        else np.zeros((SQUARE_COUNT, 3), dtype=np.float64)
    )

    mean_area = config.side * config.side / SQUARE_COUNT
    mass_strain = area / mean_area - 1.0
    mass_energy = float(np.sum(mass_strain * mass_strain))
    local_cost = (
        packing_energy
        + config.action_weight * action_energy
        + config.mass_weight * mass_strain * mass_strain
    )
    restriction_gradient = (
        packing_gradient + config.action_weight * action_gradient
    )
    transport = _first_arrival_adjacency(owner, cell_area)
    total_energy = (
        physical_energy
        + config.action_weight * float(np.sum(action_energy))
        + config.mass_weight * mass_energy
    )
    return FloorplanState(
        local_cost=local_cost,
        transport=transport,
        restriction_gradient=restriction_gradient,
        owner=owner,
        area=area,
        action_energy=action_energy,
        packing_energy=packing_energy,
        overlap_energy=overlap_energy,
        wall_energy=wall_energy,
        mass_energy=mass_energy,
        total_energy=total_energy,
    )


def banach_energy_fixed_point(
    local_cost: np.ndarray,
    transport: np.ndarray,
    *,
    discount: float,
    tolerance: float,
    iterations: int,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Solve ``z=e+gamma Pz`` in exactly seventeen scalar dimensions."""

    cost = np.asarray(local_cost, dtype=np.float64)
    matrix = np.asarray(transport, dtype=np.float64)
    if cost.shape != (SQUARE_COUNT,) or matrix.shape != (
        SQUARE_COUNT, SQUARE_COUNT
    ):
        raise ValueError("Banach floor-plan state must be exactly 17-dimensional")
    gamma = float(discount)
    if not 0.0 <= gamma < 1.0:
        raise ValueError("discount must lie in [0, 1)")
    if np.any(matrix < -1.0e-15) or not np.allclose(
        np.sum(matrix, axis=1), 1.0, atol=1.0e-12
    ):
        raise ValueError("floor-plan transport must be row stochastic")

    value = (
        np.zeros(SQUARE_COUNT, dtype=np.float64)
        if initial is None
        else np.asarray(initial, dtype=np.float64).copy()
    )
    previous_step = math.inf
    measured_ratio = 0.0
    residual = math.inf
    used = 0
    for used in range(1, max(int(iterations), 1) + 1):
        next_value = banach_energy_map(cost, matrix, value, gamma)
        step = float(np.max(np.abs(next_value - value)))
        if (
            np.isfinite(previous_step)
            and previous_step > max(1000.0 * tolerance, 1.0e-9)
        ):
            measured_ratio = max(measured_ratio, step / previous_step)
        value = next_value
        residual = step
        if step <= tolerance:
            break
        previous_step = step
    return value, {
        "dimension": SQUARE_COUNT,
        "iterations": used,
        "residual": residual,
        "theoretical_contraction": gamma,
        "measured_contraction_upper": measured_ratio,
    }


def banach_energy_map(
    local_cost: np.ndarray,
    transport: np.ndarray,
    value: np.ndarray,
    discount: float,
) -> np.ndarray:
    """One fixed-floor-plan Bellman transport step."""

    return (
        np.asarray(local_cost, dtype=np.float64)
        + float(discount)
        * (np.asarray(transport, dtype=np.float64) @ np.asarray(value))
    )


def _restricted_euclidean_direction(
    fixed_point: np.ndarray,
    gradient: np.ndarray,
    config: FloorplanConfig,
) -> np.ndarray:
    physical_gradient = np.asarray(gradient, dtype=np.float64).copy()
    physical_gradient[:, 2] /= 0.65
    norm = np.linalg.norm(physical_gradient, axis=1)
    unit = np.zeros_like(physical_gradient)
    active = norm > 1.0e-15
    unit[active] = -physical_gradient[active] / norm[active, None]
    pressure = fixed_point / max(float(np.mean(fixed_point)), 1.0e-30)
    pressure = np.clip(pressure, 0.20, 5.0)
    direction = pressure[:, None] * unit
    translation = np.linalg.norm(direction[:, :2], axis=1)
    direction[:, :2] *= np.minimum(
        1.0,
        config.translation_trust / np.maximum(translation, 1.0e-30),
    )[:, None]
    direction[:, 2] = np.clip(
        direction[:, 2], -config.phase_trust, config.phase_trust
    )
    return direction


def solve_floorplan_banach(config: FloorplanConfig) -> dict[str, object]:
    poses = initial_floorplan(config.side, config.seed)
    value = np.zeros(SQUARE_COUNT, dtype=np.float64)
    trace: list[dict[str, float | int]] = []

    for sweep in range(1, config.sweeps + 1):
        fraction = (sweep - 1) / max(config.sweeps - 1, 1)
        anneal_progress = min(
            fraction / max(config.anneal_fraction, 1.0e-6), 1.0
        )
        smooth = (
            anneal_progress
            * anneal_progress
            * (3.0 - 2.0 * anneal_progress)
        )
        action_weight = config.action_weight * (
            config.end_action_weight / config.action_weight
        ) ** smooth
        mass_weight = config.mass_weight * (
            config.end_mass_weight / config.mass_weight
        ) ** smooth
        sweep_config = replace(
            config,
            action_weight=action_weight,
            mass_weight=mass_weight,
        )
        state = floorplan_state(poses, sweep_config)
        value, contraction = banach_energy_fixed_point(
            state.local_cost,
            state.transport,
            discount=config.discount,
            tolerance=config.contraction_tolerance,
            iterations=config.contraction_iterations,
            initial=value,
        )
        direction = _restricted_euclidean_direction(
            value, state.restriction_gradient, sweep_config
        )

        accepted = 0.0
        scale = 1.0
        for _ in range(config.line_search_steps):
            candidate = poses + scale * direction
            candidate[:, 2] = wrap_square_phase(candidate[:, 2])
            candidate_state = floorplan_state(
                candidate, sweep_config, with_gradient=False
            )
            if candidate_state.total_energy < state.total_energy - 1.0e-14:
                poses = candidate
                state = candidate_state
                accepted = scale
                break
            scale *= 0.5

        if sweep == 1 or sweep % 10 == 0 or sweep == config.sweeps:
            trace.append({
                "sweep": sweep,
                "energy": state.total_energy,
                "action_weight": action_weight,
                "mass_weight": mass_weight,
                "overlap_area_energy": state.overlap_energy,
                "outside_mass_energy": state.wall_energy,
                "first_arrival_action": float(np.sum(state.action_energy)),
                "floorplan_mass_energy": state.mass_energy,
                "accepted_action": accepted,
                "banach_dimension": int(contraction["dimension"]),
                "banach_iterations": int(contraction["iterations"]),
                "banach_residual": float(contraction["residual"]),
                "banach_contraction": float(
                    contraction["theoretical_contraction"]
                ),
                "measured_contraction_upper": float(
                    contraction["measured_contraction_upper"]
                ),
            })

    # Clearance is intentionally absent above.  It is only the independent
    # terminal geometry audit of the energy-minimizing preimage.
    terminal_config = replace(
        config,
        action_weight=config.end_action_weight,
        mass_weight=config.end_mass_weight,
    )
    terminal_state = floorplan_state(poses, terminal_config, with_gradient=False)
    audit = capacity_state(poses, config.side)
    return {
        "status": "floorplan_energy_minimum_not_global_proof",
        "method": "17d_banach_first_arrival_then_euclidean_restriction",
        "config": asdict(config),
        "transport_dimension": SQUARE_COUNT,
        "best_energy": terminal_state.total_energy,
        "terminal_audit": {
            "minimum_clearance": audit.minimum_clearance,
            "overlap_residual": audit.overlap_residual,
            "feasible": audit.minimum_clearance >= -1.0e-8,
        },
        "poses": poses.tolist(),
        "trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=float, default=4.6)
    parser.add_argument("--sweeps", type=int, default=800)
    parser.add_argument("--resolution", type=int, default=72)
    parser.add_argument("--discount", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    config = FloorplanConfig(
        side=args.side,
        sweeps=args.sweeps,
        resolution=args.resolution,
        discount=args.discount,
        seed=args.seed,
    )
    result = solve_floorplan_banach(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        write_svg(args.svg, np.asarray(result["poses"]), args.side)
    print(json.dumps({
        "side": args.side,
        "energy": result["best_energy"],
        **result["terminal_audit"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
