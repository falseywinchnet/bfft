#!/usr/bin/env python3
"""Apply relaxed-chip-design transport to seventeen rigid square cells."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from relaxed_chip_design.interval_connection import (  # noqa: E402
    decompose_interval_stalk,
    synthesize_interval_stalk,
)
from relaxed_chip_design.preimage import (  # noqa: E402
    requires_row_self_distillation,
)
from relaxed_chip_design.unrelaxation import (  # noqa: E402
    conjugate_support_quantiles,
)
from relaxed_chip_design.vector_diffusion import (  # noqa: E402
    diffuse_connection_orientations,
)

from geometry import (  # noqa: E402
    PAIR_I,
    PAIR_J,
    SQUARE_COUNT,
    capacity_loss_gradient,
    capacity_state,
    square_corners,
    wrap_square_phase,
)
from reference_chart import REFERENCE_SIDE, reference_chart  # noqa: E402


@dataclass
class ConditionerAudit:
    net_count: int
    paired_axis_count: int
    ordinary_redirected_fraction: float
    ordinary_graph_propagated_fraction: float


def contact_nets(state, activation: float = 0.08) -> tuple[list[np.ndarray], np.ndarray]:
    active = np.flatnonzero(state.pair_clearance < activation)
    nets = [np.asarray([PAIR_I[pair], PAIR_J[pair]], dtype=np.int64) for pair in active]
    return nets, active


def condition_capacity_direction(
    poses: np.ndarray,
    raw_descent: np.ndarray,
    state,
    *,
    activation: float = 0.08,
) -> tuple[np.ndarray, ConditionerAudit]:
    """Transport translation direction, retaining opposing boundary phase."""

    local = raw_descent[:, :2]
    radii = np.linalg.norm(local, axis=1)
    positive = radii[radii > 1.0e-15]
    scale = float(np.median(positive)) if len(positive) else 1.0
    confidence = np.clip(radii / max(2.0 * scale, 1.0e-15), 0.0, 1.0)
    nets, active_pairs = contact_nets(state, activation)
    endpoint_counts = np.full(len(nets), 2, dtype=np.int64)
    if nets:
        ordinary, ordinary_audit = diffuse_connection_orientations(
            local,
            np.minimum(radii, 0.08),
            confidence,
            nets,
            endpoint_counts,
            diffusion=1.0,
        )
    else:
        ordinary = local.copy()
        ordinary_audit = {
            "redirected_fraction": 0.0,
            "graph_propagated_fraction": 0.0,
        }

    odd_sum = np.zeros((SQUARE_COUNT, 2), dtype=np.float64)
    odd_weight = np.zeros(SQUARE_COUNT, dtype=np.float64)
    paired_axes = 0
    for pair in active_pairs:
        first, second = int(PAIR_I[pair]), int(PAIR_J[pair])
        pair_directions = np.vstack(
            (
                state.pair_center_i_gradient[pair],
                state.pair_center_j_gradient[pair],
            )
        )
        severity = max(activation - float(state.pair_clearance[pair]), 1.0e-12)
        pair_confidence = np.full(2, min(severity / activation, 1.0))
        even, odd, face_sign, audit = decompose_interval_stalk(
            pair_directions,
            pair_confidence,
            poses[[first, second], :2],
        )
        restricted = synthesize_interval_stalk(even, odd, face_sign, normalize=True)
        odd_sum[first] += severity * restricted[0]
        odd_sum[second] += severity * restricted[1]
        odd_weight[[first, second]] += severity
        paired_axes += int(audit["paired_axis_count"])
    has_odd = odd_weight > 0.0
    odd_sum[has_odd] /= odd_weight[has_odd, None]

    direct_norm = np.linalg.norm(local, axis=1)
    direct_unit = np.zeros_like(local)
    direct_unit[direct_norm > 1.0e-15] = (
        local[direct_norm > 1.0e-15] / direct_norm[direct_norm > 1.0e-15, None]
    )
    ordinary_norm = np.linalg.norm(ordinary, axis=1)
    ordinary_unit = np.zeros_like(ordinary)
    ordinary_unit[ordinary_norm > 1.0e-15] = (
        ordinary[ordinary_norm > 1.0e-15]
        / ordinary_norm[ordinary_norm > 1.0e-15, None]
    )
    mixed = 0.55 * direct_unit + 0.20 * ordinary_unit + 0.25 * odd_sum
    mixed_norm = np.linalg.norm(mixed, axis=1)
    valid = mixed_norm > 1.0e-15
    mixed[valid] /= mixed_norm[valid, None]
    conditioned = raw_descent.copy()
    conditioned[:, :2] = radii[:, None] * mixed
    return conditioned, ConditionerAudit(
        net_count=len(nets),
        paired_axis_count=paired_axes,
        ordinary_redirected_fraction=float(ordinary_audit["redirected_fraction"]),
        ordinary_graph_propagated_fraction=float(
            ordinary_audit["graph_propagated_fraction"]
        ),
    )


def initial_population(seed: int, side: float = 5.0) -> np.ndarray:
    """A relaxed multi-row chart with no forced five-cell row."""

    rng = np.random.default_rng(seed)
    # Four staggered row charts with counts 4,4,5,4.  Their vertical phase is
    # deliberately soft: unlike the 5x4 legal seed, no complete rigid row is
    # inherited as an integral ownership decision.
    counts = (4, 4, 5, 4)
    rows = []
    for row, count in enumerate(counts):
        y = (row + 0.72) * side / 4.45
        x = np.linspace(0.62, side - 0.62, count)
        if count == 4:
            x += (0.12 if row % 2 else -0.12)
        rows.extend((value, y) for value in x)
    centers = np.asarray(rows, dtype=np.float64)
    centers += rng.normal(0.0, 0.035, centers.shape)
    theta = rng.uniform(-0.22, 0.22, SQUARE_COUNT)
    return np.column_stack((centers, theta))


def transport_chart(poses: np.ndarray, old_side: float, new_side: float) -> np.ndarray:
    result = poses.copy()
    scale = (new_side - 1.0) / (old_side - 1.0)
    result[:, :2] = 0.5 * new_side + scale * (result[:, :2] - 0.5 * old_side)
    return result


@dataclass
class SoftSolve:
    poses: np.ndarray
    minimum_clearance: float
    continuation_steps: int
    audit: ConditionerAudit
    trace: list[dict[str, float | int]]


def legalize_capacity(
    poses: np.ndarray,
    side: float,
    *,
    iterations: int = 240,
    target: float = 1.0e-7,
) -> np.ndarray:
    """Capacity-only legal emission after the transported chart is fixed.

    Each pass solves the minimum-norm displacement of the currently violated
    boundary/contact incidence system.  Physical square phase is retained;
    no destination, row alternative, or packing objective is evaluated.
    """

    result = poses.copy()
    variables = 2 * SQUARE_COUNT
    for _ in range(iterations):
        state = capacity_state(result, side)
        if state.minimum_clearance >= 0.0:
            break
        rows = []
        desired = []
        for square in range(SQUARE_COUNT):
            boundary_gradients = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
            for face, gradient in enumerate(boundary_gradients):
                clearance = float(state.boundary_clearance[square, face])
                if clearance >= target:
                    continue
                row = np.zeros(variables, dtype=np.float64)
                row[2 * square:2 * square + 2] = gradient
                rows.append(row)
                desired.append(target - clearance)
        for pair, (first, second) in enumerate(zip(PAIR_I, PAIR_J)):
            clearance = float(state.pair_clearance[pair])
            if clearance >= target:
                continue
            row = np.zeros(variables, dtype=np.float64)
            row[2 * first:2 * first + 2] = state.pair_center_i_gradient[pair]
            row[2 * second:2 * second + 2] = state.pair_center_j_gradient[pair]
            rows.append(row)
            desired.append(target - clearance)
        if not rows:
            break
        incidence = np.asarray(rows)
        right = np.asarray(desired)
        regularization = 1.0e-5
        augmented = np.vstack((incidence, math.sqrt(regularization) * np.eye(variables)))
        augmented_right = np.concatenate((right, np.zeros(variables)))
        displacement = np.linalg.lstsq(augmented, augmented_right, rcond=None)[0]
        displacement = displacement.reshape(SQUARE_COUNT, 2)
        norms = np.linalg.norm(displacement, axis=1)
        displacement *= np.minimum(1.0, 0.035 / np.maximum(norms, 1.0e-30))[:, None]
        baseline = state.overlap_residual
        accepted = False
        scale = 1.0
        for _ in range(14):
            candidate = result.copy()
            candidate[:, :2] += scale * displacement
            candidate_state = capacity_state(candidate, side)
            if candidate_state.overlap_residual < baseline - 1.0e-15:
                result = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    return result


def legalize_pose_capacity(
    poses: np.ndarray,
    side: float,
    *,
    iterations: int = 720,
    dual_iterations: int = 240,
    target: float = 1.0e-8,
) -> np.ndarray:
    """Settle a discovered chart before freezing its physical square phase.

    At each nonlinear step, solve the minimum-norm linearized correction

        minimize ||d||^2 / 2  subject to  A d >= target - clearance

    by projected ascent in the nonnegative contact-stress dual.  Unlike the
    final fixed-phase legalizer, this discovery operation has all 51 pose
    coordinates available and can therefore cross an angle-contact cusp.
    """

    result = poses.copy()
    for _ in range(iterations):
        state = capacity_state(result, side)
        if state.minimum_clearance >= 0.0:
            break
        rows = []
        desired = []
        cosine = np.cos(result[:, 2])
        sine = np.sin(result[:, 2])
        half_width_prime = 0.5 * (
            -np.sign(cosine) * sine + np.sign(sine) * cosine
        )
        boundary_gradients = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        for square in range(SQUARE_COUNT):
            for face, gradient in enumerate(boundary_gradients):
                clearance = float(state.boundary_clearance[square, face])
                if clearance >= target:
                    continue
                row = np.zeros(3 * SQUARE_COUNT, dtype=np.float64)
                row[3 * square:3 * square + 2] = gradient
                row[3 * square + 2] = -half_width_prime[square]
                rows.append(row)
                desired.append(target - clearance)
        for pair, (first, second) in enumerate(zip(PAIR_I, PAIR_J)):
            clearance = float(state.pair_clearance[pair])
            if clearance >= target:
                continue
            row = np.zeros(3 * SQUARE_COUNT, dtype=np.float64)
            row[3 * first:3 * first + 2] = state.pair_center_i_gradient[pair]
            row[3 * first + 2] = state.pair_theta_i_gradient[pair]
            row[3 * second:3 * second + 2] = state.pair_center_j_gradient[pair]
            row[3 * second + 2] = state.pair_theta_j_gradient[pair]
            rows.append(row)
            desired.append(target - clearance)
        if not rows:
            break
        incidence = np.asarray(rows)
        right = np.asarray(desired)
        gram = incidence @ incidence.T
        lipschitz = float(np.linalg.eigvalsh(gram)[-1]) + 1.0e-12
        dual = np.zeros(len(right), dtype=np.float64)
        dual_step = 0.95 / lipschitz
        for _ in range(dual_iterations):
            dual = np.maximum(0.0, dual + dual_step * (right - gram @ dual))
        displacement = (incidence.T @ dual).reshape(SQUARE_COUNT, 3)
        translation_norm = np.linalg.norm(displacement[:, :2], axis=1)
        displacement[:, :2] *= np.minimum(
            1.0, 0.025 / np.maximum(translation_norm, 1.0e-30)
        )[:, None]
        displacement[:, 2] = np.clip(displacement[:, 2], -0.018, 0.018)

        baseline = state.overlap_residual
        accepted = False
        scale = 1.0
        for _ in range(16):
            candidate = result + scale * displacement
            candidate[:, 2] = wrap_square_phase(candidate[:, 2])
            candidate_state = capacity_state(candidate, side)
            if candidate_state.overlap_residual < baseline - 1.0e-15:
                result = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    return result


def polish_pose_capacity_slsqp(
    poses: np.ndarray,
    side: float,
    *,
    iterations: int = 1600,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, dict[str, object]]:
    """Find the nearest exact capacity preimage of a transported pose chart.

    This is deliberately a readout operation: the transported pose is fixed
    before the nonlinear feasibility solve, and the objective is only squared
    distance back to that pose.  SciPy is imported lazily so the core BFFT
    experiment and deterministic tests retain their NumPy-only dependency.
    """

    try:
        from scipy.optimize import Bounds, minimize
    except ImportError as error:  # pragma: no cover - optional Mini polisher
        raise RuntimeError("SciPy is required for the SLSQP capacity polisher") from error

    anchor = np.asarray(poses, dtype=np.float64).copy()
    scale = np.tile(np.asarray([1.0, 1.0, 0.65]), SQUARE_COUNT)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        delta = (flat - anchor.ravel()) * scale
        return 0.5 * float(np.dot(delta, delta)), delta * scale

    def constraints(flat: np.ndarray) -> np.ndarray:
        state = capacity_state(flat.reshape(SQUARE_COUNT, 3), side)
        return np.concatenate((state.boundary_clearance.ravel(), state.pair_clearance))

    def constraint_jacobian(flat: np.ndarray) -> np.ndarray:
        chart = flat.reshape(SQUARE_COUNT, 3)
        state = capacity_state(chart, side)
        jacobian = np.zeros((4 * SQUARE_COUNT + len(PAIR_I), 3 * SQUARE_COUNT))
        cosine = np.cos(chart[:, 2])
        sine = np.sin(chart[:, 2])
        half_width_prime = 0.5 * (
            -np.sign(cosine) * sine + np.sign(sine) * cosine
        )
        boundary_gradients = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        for square in range(SQUARE_COUNT):
            for face, gradient in enumerate(boundary_gradients):
                row = 4 * square + face
                jacobian[row, 3 * square:3 * square + 2] = gradient
                jacobian[row, 3 * square + 2] = -half_width_prime[square]
        offset = 4 * SQUARE_COUNT
        for pair, (first, second) in enumerate(zip(PAIR_I, PAIR_J)):
            row = offset + pair
            jacobian[row, 3 * first:3 * first + 2] = state.pair_center_i_gradient[pair]
            jacobian[row, 3 * first + 2] = state.pair_theta_i_gradient[pair]
            jacobian[row, 3 * second:3 * second + 2] = state.pair_center_j_gradient[pair]
            jacobian[row, 3 * second + 2] = state.pair_theta_j_gradient[pair]
        return jacobian

    lower = np.tile(np.asarray([-0.5, -0.5, -0.25 * math.pi]), SQUARE_COUNT)
    upper = np.tile(np.asarray([side + 0.5, side + 0.5, 0.25 * math.pi]), SQUARE_COUNT)
    result = minimize(
        objective,
        anchor.ravel(),
        method="SLSQP",
        jac=True,
        bounds=Bounds(lower, upper),
        constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
        options={"maxiter": iterations, "ftol": tolerance, "disp": False},
    )
    polished = np.asarray(result.x).reshape(SQUARE_COUNT, 3)
    polished[:, 2] = wrap_square_phase(polished[:, 2])
    state = capacity_state(polished, side)
    return polished, {
        "success": bool(result.success and state.minimum_clearance >= -1.0e-8),
        "solver_success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "minimum_clearance": state.minimum_clearance,
        "overlap_residual": state.overlap_residual,
        "anchor_distance": float(np.linalg.norm((polished - anchor).ravel() * scale)),
    }


def solve_soft_transport(
    initial: np.ndarray,
    side: float,
    *,
    iterations: int,
    seed: int,
) -> SoftSolve:
    """Evolve the relaxed chart; stop only on the capacity residual."""

    rng = np.random.default_rng(seed)
    poses = initial.copy()
    if not np.any(np.abs(poses[:, 2]) > 1.0e-12):
        poses[:, 2] = rng.normal(0.0, 0.018, SQUARE_COUNT)
    first_moment = np.zeros_like(poses)
    second_moment = np.zeros_like(poses)
    best = poses.copy()
    best_minimum = capacity_state(best, side).minimum_clearance
    trace: list[dict[str, float | int]] = []
    continuation = 0
    last_audit = ConditionerAudit(0, 0, 0.0, 0.0)

    for iteration in range(1, iterations + 1):
        fraction = (iteration - 1) / max(iterations - 1, 1)
        temperature = 0.028 * (0.0018 / 0.028) ** fraction
        loss, gradient, capacity = capacity_loss_gradient(
            poses, side, target_clearance=3.0e-3, temperature=temperature
        )
        descent, last_audit = condition_capacity_direction(poses, -gradient, capacity)
        first_moment = 0.9 * first_moment + 0.1 * descent
        second_moment = 0.999 * second_moment + 0.001 * descent * descent
        m_hat = first_moment / (1.0 - 0.9**iteration)
        v_hat = second_moment / (1.0 - 0.999**iteration)
        learning_rate = 0.0035 * (0.3 + 0.7 * (1.0 - fraction))
        step = learning_rate * m_hat / (np.sqrt(v_hat) + 1.0e-8)
        step[:, :2] = np.clip(step[:, :2], -0.012, 0.012)
        step[:, 2] = np.clip(step[:, 2], -0.008, 0.008)
        poses += step
        poses[:, 2] = wrap_square_phase(poses[:, 2])
        current = capacity_state(poses, side)
        if current.minimum_clearance > best_minimum:
            best = poses.copy()
            best_minimum = current.minimum_clearance
        if iteration % 100 == 0 or iteration == 1:
            continuation += 1
            trace.append(
                {
                    "iteration": iteration,
                    "loss": loss,
                    "minimum_clearance": current.minimum_clearance,
                    "overlap_residual": current.overlap_residual,
                    "net_count": last_audit.net_count,
                    "paired_axis_count": last_audit.paired_axis_count,
                }
            )
        if current.overlap_residual <= 1.0e-7 and current.minimum_clearance >= -1.0e-8:
            best = poses.copy()
            best_minimum = current.minimum_clearance
            break
        if iteration % 600 == 0 and best_minimum < -1.0e-3:
            # Rephase within the same square chart; positions and accumulated
            # transport state are retained.
            poses[:, 2] = wrap_square_phase(
                poses[:, 2] + rng.normal(0.0, 0.012, SQUARE_COUNT)
            )
    return SoftSolve(best, best_minimum, continuation, last_audit, trace)


def quantile_emit(source: np.ndarray, transported: np.ndarray) -> np.ndarray:
    """One bounded six-slot CDF inverse along each measured chart secant."""

    delta = transported[:, :2] - source[:, :2]
    radii = np.linalg.norm(delta, axis=1)
    unit = np.zeros_like(delta)
    valid = radii > 1.0e-15
    unit[valid] = delta[valid] / radii[valid, None]
    offsets = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    segment_count = SQUARE_COUNT * len(offsets)
    segment_x = np.empty(segment_count)
    segment_y = np.empty(segment_count)
    active = np.empty((SQUARE_COUNT, len(offsets)), dtype=np.int64)
    anchors = np.empty(SQUARE_COUNT, dtype=np.int64)
    reference = np.tile(np.asarray([0.06, 0.18, 0.52, 0.18, 0.05, 0.01]), (SQUARE_COUNT, 1))
    transported_support = np.tile(
        np.asarray([0.01, 0.04, 0.12, 0.28, 0.40, 0.15]), (SQUARE_COUNT, 1)
    )
    for square in range(SQUARE_COUNT):
        base = square * len(offsets)
        indices = base + np.arange(len(offsets))
        active[square] = indices
        anchors[square] = base + 2
        positions = source[square, :2] + offsets[:, None] * delta[square, :2]
        segment_x[indices] = positions[:, 0]
        segment_y[indices] = positions[:, 1]
    emitted = conjugate_support_quantiles(
        active_segments=active,
        anchor_segments=anchors,
        transported_support=transported_support,
        reference_support=reference,
        segment_y=segment_y,
        segment_x=segment_x,
        source_phase=np.full(SQUARE_COUNT, 0.5),
    )
    result = source.copy()
    result[:, :2] = np.column_stack(
        (segment_x[emitted.target_segments], segment_y[emitted.target_segments])
    )
    result[:, 2] = transported[:, 2]
    return result


@dataclass
class CompressionResult:
    poses: np.ndarray
    side: float
    minimum_clearance: float
    trace: list[dict[str, object]]


def compress(
    seed: int,
    *,
    initializer: str,
    stop_side: float,
    side_step: float,
    iterations: int,
) -> CompressionResult:
    if initializer == "reference":
        side = REFERENCE_SIDE
        source = reference_chart()
    elif initializer == "relaxed":
        side = 5.0
        initial_solve = solve_soft_transport(
            initial_population(seed, side), side, iterations=iterations, seed=seed
        )
        source = legalize_capacity(initial_solve.poses, side)
    else:
        raise ValueError(f"unknown initializer {initializer}")
    last_feasible = source.copy()
    last_side = side
    last_clearance = capacity_state(source, side).minimum_clearance
    trace: list[dict[str, object]] = []
    while side - side_step >= stop_side - 1.0e-12:
        trial_side = round(side - side_step, 10)
        initial = transport_chart(source, side, trial_side)
        soft = solve_soft_transport(
            initial, trial_side, iterations=iterations, seed=seed + len(trace) * 104729
        )
        emitted = quantile_emit(initial, soft.poses)
        emitted_capacity = capacity_state(emitted, trial_side)
        # The hard inverse is audited, not selected by its physical score.  A
        # multi-pass chart receives the prescribed one-time row/translation
        # self-distillation; otherwise the first transported chart is final.
        self_distilled = requires_row_self_distillation(soft.continuation_steps)
        transported_final = soft.poses if self_distilled else emitted
        final = legalize_capacity(transported_final, trial_side)
        final_capacity = capacity_state(final, trial_side)
        feasible = final_capacity.minimum_clearance >= -1.0e-8
        trace.append(
            {
                "side": trial_side,
                "soft_minimum_clearance": soft.minimum_clearance,
                "hard_minimum_clearance": emitted_capacity.minimum_clearance,
                "final_minimum_clearance": final_capacity.minimum_clearance,
                "soft_continuation_steps": soft.continuation_steps,
                "self_distilled": self_distilled,
                "net_count": soft.audit.net_count,
                "paired_axis_count": soft.audit.paired_axis_count,
                "feasible": feasible,
                "soft_trace": soft.trace,
            }
        )
        source = final
        side = trial_side
        if feasible:
            last_feasible = final.copy()
            last_side = side
            last_clearance = final_capacity.minimum_clearance
        elif final_capacity.minimum_clearance < -0.02:
            break
    return CompressionResult(last_feasible, last_side, last_clearance, trace)


def write_svg(path: Path, poses: np.ndarray, side: float) -> None:
    canvas = 900
    scale = canvas / side
    colors = ("2563eb", "db2777", "059669", "d97706", "7c3aed", "0891b2")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas}" height="{canvas}" viewBox="0 0 {canvas} {canvas}">',
        '<rect width="100%" height="100%" fill="#fafaf9"/>',
        f'<rect x="1" y="1" width="{canvas-2}" height="{canvas-2}" fill="none" stroke="#111827" stroke-width="3"/>',
    ]
    for index, pose in enumerate(poses):
        polygon = " ".join(
            f"{x*scale:.4f},{canvas-y*scale:.4f}" for x, y in square_corners(pose)
        )
        color = colors[index % len(colors)]
        lines.append(
            f'<polygon points="{polygon}" fill="#{color}" fill-opacity="0.34" stroke="#{color}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{pose[0]*scale:.3f}" y="{canvas-pose[1]*scale:.3f}" font-family="monospace" font-size="14" text-anchor="middle" dominant-baseline="central">{index+1}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initializer", choices=("reference", "relaxed"), default="reference")
    parser.add_argument("--stop-side", type=float, default=4.70)
    parser.add_argument("--side-step", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    result = compress(
        args.seed,
        initializer=args.initializer,
        stop_side=args.stop_side,
        side_step=args.side_step,
        iterations=args.iterations,
    )
    payload = {
        "status": "floating_point_construction_not_lower_bound_proof",
        "method": "relaxed_chip_design_support_sparse_transport",
        "seed": args.seed,
        "side": result.side,
        "minimum_clearance": result.minimum_clearance,
        "poses": result.poses.tolist(),
        "trace": result.trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        write_svg(args.svg, result.poses, result.side)
    print(json.dumps({"side": result.side, "minimum_clearance": result.minimum_clearance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
