#!/usr/bin/env python3
"""
N=8 BFFT cone-DIF whiffletree / Euler-Bernoulli beam study.

The script compares two passive mechanical interpretations of the same
normalized Bruun/DIF factorization:

1. Force-mode whiffletree
   The mass-potential gauge makes each cone stage A column-stochastic.
   A reciprocal linkage enforcing q_in = A.T q_out then transmits generalized
   forces as f_out = A f_in. Each row of A.T is a convex combination and is
   implemented by one or more rigid interpolation bars represented with
   Euler-Bernoulli beam elements. Negative signs remain rail swaps in the cone.

2. Displacement-mode whiffletree
   A forward diagonal kinematic gauge makes each physical stage row-stochastic,
   so every output displacement is a convex combination of prior displacements.
   This removes all internal gain levers. A special stage-EI ratio makes the
   differential output compliance isotropic, so equal output loading produces
   only a single scalar gain error.

The beam model is linear, small-deflection Euler-Bernoulli bending. Every bar
has independent pin rotations at its connection points. Optional rotational
springs model non-ideal flexure hinges. Tap-position Monte Carlo models geometry
error in the whiffletree ratios.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


TOL = 1e-12


def real_fourier_matrix(n: int) -> np.ndarray:
    """Orthonormal packed real Fourier matrix [DC, cos1, -sin1, ..., Nyquist]."""
    t = np.arange(n)
    rows: list[np.ndarray] = [np.ones(n) / np.sqrt(n)]
    for k in range(1, n // 2):
        rows.append(np.sqrt(2.0 / n) * np.cos(2.0 * np.pi * k * t / n))
        rows.append(-np.sqrt(2.0 / n) * np.sin(2.0 * np.pi * k * t / n))
    rows.append(((-1.0) ** t) / np.sqrt(n))
    return np.vstack(rows)


def cone_lift(matrix: np.ndarray) -> np.ndarray:
    pos = np.maximum(matrix, 0.0)
    neg = np.maximum(-matrix, 0.0)
    return np.block([[pos, neg], [neg, pos]])


def bruun_norm_cell(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, c, -s],
            [0.0, 1.0, s, c],
            [1.0, 0.0, -c, s],
            [0.0, -1.0, s, c],
        ]
    ) / np.sqrt(2.0)


def top_half_split(n: int) -> np.ndarray:
    h = n // 2
    out = np.zeros((n, n))
    for j in range(h):
        out[j, j] = 1.0 / np.sqrt(2.0)
        out[j, j + h] = 1.0 / np.sqrt(2.0)
        out[h + j, j] = 1.0 / np.sqrt(2.0)
        out[h + j, j + h] = -1.0 / np.sqrt(2.0)
    return out


def bruun8_factorization() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Three active normalized stages, final wire relabeling, and target F8."""
    target = real_fourier_matrix(8)
    stage0 = top_half_split(8)

    # Odd half-bin branch: [B0,B1,B2,B3] -> [cos1,-sin1,cos3,-sin3].
    pbin = np.eye(4)[[0, 2, 1, 3], :]
    odd = np.diag([1.0, -1.0, 1.0, -1.0]) @ bruun_norm_cell(np.pi / 4.0) @ pbin

    # Even branch: a 4-point real Fourier walk split across two stages.
    even0 = top_half_split(4)
    even1 = np.array(
        [
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0, 0.0],
        ]
    )

    stage1 = linalg.block_diag(even0, odd)
    stage2 = linalg.block_diag(even1, np.eye(4))

    # Natural stage output -> packed real Fourier order.
    permutation = np.eye(8)[[0, 4, 5, 1, 2, 6, 7, 3], :]
    product = permutation @ stage2 @ stage1 @ stage0
    if np.linalg.norm(product - target) > 2e-14:
        raise RuntimeError("N=8 factorization no longer matches the target")
    return [stage0, stage1, stage2], permutation, target


def backward_mass_gauge(
    stages: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """BFFT mass-potential gauge: abs(gauged stage) has unit column sums."""
    gauges: list[np.ndarray] = [np.empty(0)] * (len(stages) + 1)
    gauges[-1] = np.ones(stages[-1].shape[0])
    for s in range(len(stages) - 1, -1, -1):
        gauges[s] = np.abs(stages[s]).T @ gauges[s + 1]
    gauged = [
        (gauges[s + 1][:, None] * stage) / gauges[s][None, :]
        for s, stage in enumerate(stages)
    ]
    return gauges, gauged


def forward_kinematic_gauge(
    stages: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Choose physical displacement scales so every cone stage is row-stochastic.

    For q_s = D_s z_s and z_{s+1} = A_s z_s, choose
        d_{s+1,j} = 1 / sum_i A_s[j,i] / d_{s,i}.
    Then B_s = D_{s+1} A_s D_s^-1 has nonnegative rows summing to one.
    """
    cone = [cone_lift(stage) for stage in stages]
    scales = [np.ones(cone[0].shape[1])]
    row_stochastic: list[np.ndarray] = []
    for stage in cone:
        next_scale = 1.0 / (stage @ (1.0 / scales[-1]))
        physical = next_scale[:, None] * stage * (1.0 / scales[-1])[None, :]
        if np.max(np.abs(physical.sum(axis=1) - 1.0)) > 1e-13:
            raise RuntimeError("kinematic gauge failed row-stochasticity")
        row_stochastic.append(physical)
        scales.append(next_scale)
    return scales, row_stochastic


def best_scalar_error(matrix: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    gain = float(np.sum(matrix * target) / np.sum(matrix * matrix))
    error = float(np.linalg.norm(gain * matrix - target) / np.linalg.norm(target))
    return error, gain


@dataclass(frozen=True)
class BeamElement:
    v1: int
    r1: int
    v2: int
    r2: int
    length: float
    ei: float
    stage: int


class BeamNetwork:
    """Vertical-displacement Euler-Bernoulli beams with local pin rotations."""

    def __init__(self, hinge_eta: float = 0.0) -> None:
        self.n_vertical = 0
        self.n_rotation = 0
        self.elements: list[BeamElement] = []
        self.vertical_springs: list[tuple[int, float]] = []
        self.rotation_springs: list[tuple[int, float]] = []
        self.hinge_eta = float(hinge_eta)

    def new_vertical(self) -> int:
        index = self.n_vertical
        self.n_vertical += 1
        return index

    def new_rotation(self) -> int:
        index = self.n_rotation
        self.n_rotation += 1
        return index

    def add_interpolation_bar(
        self,
        vertical_nodes: Sequence[int],
        abscissae: Sequence[float],
        ei: float,
        stage: int,
    ) -> None:
        order = np.argsort(abscissae)
        x = np.asarray(abscissae, dtype=float)[order]
        v = [vertical_nodes[i] for i in order]
        if np.min(np.diff(x)) <= 1e-10:
            raise ValueError(f"degenerate beam abscissae: {x}")
        rotations = [self.new_rotation() for _ in x]
        total_length = float(x[-1] - x[0])
        for j in range(len(x) - 1):
            self.elements.append(
                BeamElement(
                    v[j], rotations[j], v[j + 1], rotations[j + 1],
                    float(x[j + 1] - x[j]), float(ei), stage,
                )
            )
        # Worst-case simple flexure model: each bar-to-shuttle rotation is
        # resisted relative to the shuttle's fixed orientation.
        if self.hinge_eta > 0.0:
            k_theta = self.hinge_eta * float(ei) / total_length
            self.rotation_springs.extend((r, k_theta) for r in rotations)

    def assemble(self) -> np.ndarray:
        n = self.n_vertical + self.n_rotation
        stiffness = np.zeros((n, n))
        for element in self.elements:
            length = element.length
            factor = element.ei / length**3
            local = factor * np.array(
                [
                    [12.0, 6.0 * length, -12.0, 6.0 * length],
                    [6.0 * length, 4.0 * length**2, -6.0 * length, 2.0 * length**2],
                    [-12.0, -6.0 * length, 12.0, -6.0 * length],
                    [6.0 * length, 2.0 * length**2, -6.0 * length, 4.0 * length**2],
                ]
            )
            indices = [
                element.v1,
                self.n_vertical + element.r1,
                element.v2,
                self.n_vertical + element.r2,
            ]
            stiffness[np.ix_(indices, indices)] += local
        for node, spring in self.vertical_springs:
            stiffness[node, node] += spring
        for rotation, spring in self.rotation_springs:
            stiffness[self.n_vertical + rotation, self.n_vertical + rotation] += spring
        return stiffness


def _choose_three_group(weights: np.ndarray) -> tuple[int, int, int, float, float]:
    """Select the binary tree with the least extreme tap positions."""
    best: tuple[float, int, int, int, float, float] | None = None
    for pair in ((0, 1), (0, 2), (1, 2)):
        third = ({0, 1, 2} - set(pair)).pop()
        i, j = pair
        pair_sum = weights[i] + weights[j]
        beta_pair = weights[j] / pair_sum
        beta_final = weights[third] / weights.sum()
        score = min(beta_pair, 1.0 - beta_pair, beta_final, 1.0 - beta_final)
        candidate = (score, i, j, third, beta_pair, beta_final)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3], best[4], best[5]


def add_convex_tree(
    network: BeamNetwork,
    input_nodes: Sequence[int],
    weights: np.ndarray,
    ei: float,
    stage: int,
) -> int:
    """Build a binary whiffletree for a convex combination of up to 4 inputs."""
    nodes = list(input_nodes)
    weights = np.asarray(weights, dtype=float)
    count = len(nodes)
    if count == 1:
        return nodes[0]
    if count == 2:
        output = network.new_vertical()
        beta = float(weights[1] / weights.sum())
        network.add_interpolation_bar([nodes[0], output, nodes[1]], [0.0, beta, 1.0], ei, stage)
        return output
    if count == 3:
        i, j, third, beta_pair, beta_final = _choose_three_group(weights)
        partial = network.new_vertical()
        network.add_interpolation_bar(
            [nodes[i], partial, nodes[j]], [0.0, beta_pair, 1.0], ei, stage
        )
        output = network.new_vertical()
        network.add_interpolation_bar(
            [partial, output, nodes[third]], [0.0, beta_final, 1.0], ei, stage
        )
        return output
    if count == 4:
        pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
        best = None
        for pair1, pair2 in pairings:
            sum1 = float(weights[list(pair1)].sum())
            sum2 = float(weights[list(pair2)].sum())
            fractions = [
                weights[pair1[0]] / sum1,
                weights[pair1[1]] / sum1,
                weights[pair2[0]] / sum2,
                weights[pair2[1]] / sum2,
                sum1 / (sum1 + sum2),
                sum2 / (sum1 + sum2),
            ]
            score = float(min(fractions))
            if best is None or score > best[0]:
                best = (score, pair1, pair2, sum1, sum2)
        assert best is not None
        _, pair1, pair2, sum1, sum2 = best
        partial1 = network.new_vertical()
        partial2 = network.new_vertical()
        network.add_interpolation_bar(
            [nodes[pair1[0]], partial1, nodes[pair1[1]]],
            [0.0, float(weights[pair1[1]] / sum1), 1.0], ei, stage,
        )
        network.add_interpolation_bar(
            [nodes[pair2[0]], partial2, nodes[pair2[1]]],
            [0.0, float(weights[pair2[1]] / sum2), 1.0], ei, stage,
        )
        output = network.new_vertical()
        network.add_interpolation_bar(
            [partial1, output, partial2], [0.0, sum2 / (sum1 + sum2), 1.0], ei, stage
        )
        return output
    raise NotImplementedError("N=8 construction should never require >4 inputs per tree")


def build_force_whiffletree(
    stages: Sequence[np.ndarray],
    stage_ei: Sequence[float],
    hinge_eta: float,
    output_springs: np.ndarray | None = None,
) -> tuple[BeamNetwork, list[list[int]], list[np.ndarray]]:
    """
    Build q_s = A_s.T q_{s+1}. By virtual work the forward force map is
    f_{s+1} = A_s f_s, where A_s is the mass-gauged cone stage.
    """
    gauges, gauged_signed = backward_mass_gauge(stages)
    cone_stages = [cone_lift(stage) for stage in gauged_signed]
    for stage in cone_stages:
        if np.max(np.abs(stage.sum(axis=0) - 1.0)) > 1e-13:
            raise RuntimeError("mass-gauged cone stage is not column-stochastic")

    network = BeamNetwork(hinge_eta)
    next_boundary = [network.new_vertical() for _ in range(cone_stages[-1].shape[0])]
    boundaries: list[list[int]] = [[] for _ in range(len(cone_stages) + 1)]
    boundaries[-1] = next_boundary

    for s in range(len(cone_stages) - 1, -1, -1):
        displacement_stage = cone_stages[s].T
        current_boundary: list[int] = []
        for row in displacement_stage:
            support = np.flatnonzero(row > TOL)
            current_boundary.append(
                add_convex_tree(
                    network,
                    [next_boundary[index] for index in support],
                    row[support],
                    float(stage_ei[s]),
                    s,
                )
            )
        boundaries[s] = current_boundary
        next_boundary = current_boundary

    if output_springs is not None:
        for node, spring in zip(boundaries[-1], output_springs):
            network.vertical_springs.append((node, float(spring)))
    return network, boundaries, gauges


def force_transfer(
    stages: Sequence[np.ndarray],
    permutation: np.ndarray,
    target: np.ndarray,
    stage_ei: Sequence[float],
    hinge_eta: float = 0.0,
    output_springs: np.ndarray | None = None,
    fixed_outputs: bool = False,
) -> tuple[np.ndarray, float, float, BeamNetwork]:
    network, boundaries, gauges = build_force_whiffletree(
        stages, stage_ei, hinge_eta, output_springs
    )
    stiffness = network.assemble()
    n = stiffness.shape[0]
    force = np.zeros((n, target.shape[1]))
    input_force_map = np.vstack([np.diag(gauges[0]), np.zeros((8, 8))])
    for row, node in enumerate(boundaries[0]):
        force[node] += input_force_map[row]

    output_nodes = boundaries[-1]
    if fixed_outputs:
        fixed = np.asarray(output_nodes, dtype=int)
        free = np.setdiff1d(np.arange(n), fixed)
        displacement = np.zeros((n, target.shape[1]))
        displacement[free] = linalg.solve(
            stiffness[np.ix_(free, free)], force[free], assume_a="sym"
        )
        reactions = stiffness[np.ix_(fixed, free)] @ displacement[free] - force[fixed]
        output_forces = -reactions
    else:
        if output_springs is None:
            raise ValueError("supply output_springs or use fixed_outputs=True")
        displacement = linalg.solve(stiffness, force, assume_a="sym")
        output_forces = np.vstack(
            [spring * displacement[node] for node, spring in zip(output_nodes, output_springs)]
        )

    projection = np.hstack([np.eye(8), -np.eye(8)])
    matrix = permutation @ projection @ output_forces
    shape_error, gain = best_scalar_error(matrix, target)
    return matrix, shape_error, gain, network


def build_displacement_whiffletree(
    stages: Sequence[np.ndarray],
    stage_ei: Sequence[float],
    hinge_eta: float,
    output_load: float,
) -> tuple[BeamNetwork, list[int], list[int], list[np.ndarray]]:
    scales, physical_stages = forward_kinematic_gauge(stages)
    network = BeamNetwork(hinge_eta)
    input_nodes = [network.new_vertical() for _ in range(physical_stages[0].shape[1])]
    previous = input_nodes
    for s, stage in enumerate(physical_stages):
        current: list[int] = []
        for row in stage:
            support = np.flatnonzero(row > TOL)
            current.append(
                add_convex_tree(
                    network,
                    [previous[index] for index in support],
                    row[support],
                    float(stage_ei[s]),
                    s,
                )
            )
        previous = current
    if output_load > 0.0:
        network.vertical_springs.extend((node, output_load) for node in previous)
    return network, input_nodes, previous, scales


def displacement_transfer(
    stages: Sequence[np.ndarray],
    permutation: np.ndarray,
    target: np.ndarray,
    stage_ei: Sequence[float],
    hinge_eta: float = 0.0,
    output_load: float = 0.0,
) -> tuple[np.ndarray, float, float, BeamNetwork]:
    network, inputs, outputs, scales = build_displacement_whiffletree(
        stages, stage_ei, hinge_eta, output_load
    )
    stiffness = network.assemble()
    known = np.asarray(inputs, dtype=int)
    free = np.setdiff1d(np.arange(stiffness.shape[0]), known)
    input_map = np.vstack([np.eye(8), np.zeros((8, 8))])
    displacement = np.zeros((stiffness.shape[0], 8))
    displacement[known] = input_map
    displacement[free] = linalg.solve(
        stiffness[np.ix_(free, free)],
        -stiffness[np.ix_(free, known)] @ input_map,
        assume_a="sym",
    )
    physical_output = np.vstack([displacement[node] for node in outputs])
    projection = np.hstack([np.eye(8), -np.eye(8)])
    output_map = permutation @ projection @ np.diag(1.0 / scales[-1])
    matrix = output_map @ physical_output
    shape_error, gain = best_scalar_error(matrix, target)
    return matrix, shape_error, gain, network


def perturb_tree_weights(weights: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb the actual binary-bar tap fractions by an absolute fraction of bar length."""
    weights = np.asarray(weights, dtype=float)
    count = len(weights)
    if count == 1:
        return np.ones(1)
    if count == 2:
        beta = np.clip(weights[1] / weights.sum() + rng.normal(0.0, sigma), 1e-9, 1.0 - 1e-9)
        return np.array([1.0 - beta, beta])
    if count == 3:
        i, j, third, beta_pair, beta_final = _choose_three_group(weights)
        beta_pair = np.clip(beta_pair + rng.normal(0.0, sigma), 1e-9, 1.0 - 1e-9)
        beta_final = np.clip(beta_final + rng.normal(0.0, sigma), 1e-9, 1.0 - 1e-9)
        result = np.zeros(3)
        result[i] = (1.0 - beta_final) * (1.0 - beta_pair)
        result[j] = (1.0 - beta_final) * beta_pair
        result[third] = beta_final
        return result
    if count == 4:
        pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
        best = None
        for pair1, pair2 in pairings:
            sum1 = float(weights[list(pair1)].sum())
            sum2 = float(weights[list(pair2)].sum())
            fractions = [
                weights[pair1[0]] / sum1,
                weights[pair1[1]] / sum1,
                weights[pair2[0]] / sum2,
                weights[pair2[1]] / sum2,
                sum1 / (sum1 + sum2),
                sum2 / (sum1 + sum2),
            ]
            score = float(min(fractions))
            if best is None or score > best[0]:
                best = (score, pair1, pair2, sum1, sum2)
        assert best is not None
        _, pair1, pair2, sum1, sum2 = best
        beta1 = np.clip(weights[pair1[1]] / sum1 + rng.normal(0.0, sigma), 1e-9, 1.0 - 1e-9)
        beta2 = np.clip(weights[pair2[1]] / sum2 + rng.normal(0.0, sigma), 1e-9, 1.0 - 1e-9)
        beta_final = np.clip(sum2 / (sum1 + sum2) + rng.normal(0.0, sigma), 1e-9, 1.0 - 1e-9)
        result = np.zeros(4)
        result[pair1[0]] = (1.0 - beta_final) * (1.0 - beta1)
        result[pair1[1]] = (1.0 - beta_final) * beta1
        result[pair2[0]] = beta_final * (1.0 - beta2)
        result[pair2[1]] = beta_final * beta2
        return result
    raise NotImplementedError


def perturbed_force_matrix(
    cone_stages: Sequence[np.ndarray],
    gauges: Sequence[np.ndarray],
    permutation: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    perturbed: list[np.ndarray] = []
    for stage in cone_stages:
        transposed = stage.T
        changed = np.zeros_like(transposed)
        for row_index, row in enumerate(transposed):
            support = np.flatnonzero(row > TOL)
            changed[row_index, support] = perturb_tree_weights(row[support], sigma, rng)
        perturbed.append(changed.T)
    product = np.eye(cone_stages[0].shape[1])
    for stage in perturbed:
        product = stage @ product
    projection = np.hstack([np.eye(8), -np.eye(8)])
    input_map = np.vstack([np.diag(gauges[0]), np.zeros((8, 8))])
    return permutation @ projection @ product @ input_map


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("build/mechanical-cone-fft/beam"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stages, permutation, target = bruun8_factorization()
    gauges, gauged_signed = backward_mass_gauge(stages)
    cone_stages = [cone_lift(stage) for stage in gauged_signed]

    # Exact force-mode test with deliberately extreme stiffness and sensor variation.
    rng = np.random.default_rng(20260812)
    random_errors: list[float] = []
    for _ in range(100):
        stage_ei = np.exp(rng.uniform(-4.0, 4.0, 3))
        output_springs = np.exp(rng.uniform(-6.0, 6.0, 16))
        matrix, _, _, _ = force_transfer(
            stages, permutation, target, stage_ei, 0.0, output_springs, False
        )
        random_errors.append(float(np.linalg.norm(matrix - target) / np.linalg.norm(target)))

    # Hinge sensitivity for the force-mode implementation.
    hinge_rows: list[dict[str, float]] = []
    for eta in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        matrix, shape_error, gain, network = force_transfer(
            stages, permutation, target, [1.0, 1.0, 1.0], eta, None, True
        )
        raw_error = float(np.linalg.norm(matrix - target) / np.linalg.norm(target))
        hinge_rows.append(
            {
                "hinge_eta": eta,
                "raw_relative_error": raw_error,
                "shape_error_after_global_gain": shape_error,
                "global_gain": gain,
            }
        )

    # Actual tap-fraction tolerance, not generic coefficient noise.
    geometry_rows: list[dict[str, float]] = []
    for sigma in [1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
        errors: list[float] = []
        for _ in range(args.trials):
            matrix = perturbed_force_matrix(
                cone_stages, gauges, permutation, sigma, rng
            )
            errors.append(best_scalar_error(matrix, target)[0])
        geometry_rows.append(
            {
                "tap_position_sigma_fraction_of_bar_length": sigma,
                "median_shape_error": float(np.median(errors)),
                "p95_shape_error": float(np.quantile(errors, 0.95)),
            }
        )

    # Displacement mode: equal EI versus analytically impedance-matched EI.
    sqrt2 = np.sqrt(2.0)
    matched_ei = np.array(
        [
            1.0,
            (-338.0 + 248.0 * sqrt2) / 7.0,
            (482.0 - 124.0 * sqrt2) / 161.0,
        ]
    )
    load_rows: list[dict[str, float]] = []
    for load in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        for design, ei in (("equal_EI", np.ones(3)), ("matched_EI", matched_ei)):
            matrix, shape_error, gain, _ = displacement_transfer(
                stages, permutation, target, ei, 0.0, load
            )
            load_rows.append(
                {
                    "design": design,
                    "dimensionless_output_load": load,
                    "raw_relative_error": float(
                        np.linalg.norm(matrix - target) / np.linalg.norm(target)
                    ),
                    "shape_error_after_global_gain": shape_error,
                    "global_gain": gain,
                }
            )

    # Network census.
    _, _, _, force_network = force_transfer(
        stages, permutation, target, [1.0, 1.0, 1.0], 0.0, np.ones(16), False
    )
    bars_per_stage = []
    for stage in range(3):
        element_count = sum(element.stage == stage for element in force_network.elements)
        bars_per_stage.append(element_count // 2)

    # Differential compliance check for the matched displacement network.
    matched_network, inputs, outputs, _ = build_displacement_whiffletree(
        stages, matched_ei, 0.0, 0.0
    )
    stiffness = matched_network.assemble()
    known = np.asarray(inputs, dtype=int)
    free = np.setdiff1d(np.arange(stiffness.shape[0]), known)
    free_map = {dof: index for index, dof in enumerate(free)}
    selector = np.zeros((16, len(free)))
    for row, node in enumerate(outputs):
        selector[row, free_map[node]] = 1.0
    compliance = selector @ linalg.solve(
        stiffness[np.ix_(free, free)], selector.T, assume_a="sym"
    )
    differential_compliance = compliance[:8, :8] - compliance[:8, 8:]
    scalar_compliance = float(np.trace(differential_compliance) / 8.0)
    compliance_defect = float(
        np.linalg.norm(differential_compliance - scalar_compliance * np.eye(8))
    )

    summary_rows = [
        {"metric": "force_mode_random_configuration_max_error", "value": max(random_errors)},
        {"metric": "force_mode_random_configuration_p95_error", "value": float(np.quantile(random_errors, 0.95))},
        {"metric": "force_mode_binary_bars", "value": float(sum(bars_per_stage))},
        {"metric": "force_mode_EB_elements", "value": float(len(force_network.elements))},
        {"metric": "force_mode_vertical_DOFs", "value": float(force_network.n_vertical)},
        {"metric": "force_mode_rotation_DOFs", "value": float(force_network.n_rotation)},
        {"metric": "stage0_binary_bars", "value": float(bars_per_stage[0])},
        {"metric": "stage1_binary_bars", "value": float(bars_per_stage[1])},
        {"metric": "stage2_binary_bars", "value": float(bars_per_stage[2])},
        {"metric": "matched_EI_stage1_over_stage0", "value": float(matched_ei[1])},
        {"metric": "matched_EI_stage2_over_stage0", "value": float(matched_ei[2])},
        {"metric": "matched_differential_compliance", "value": scalar_compliance},
        {"metric": "matched_compliance_isotropy_defect", "value": compliance_defect},
        {"metric": "N8_mass_gauge_min", "value": float(gauges[0].min())},
        {"metric": "N8_mass_gauge_max", "value": float(gauges[0].max())},
    ]

    write_csv(args.out / "hinge_sensitivity.csv", hinge_rows)
    write_csv(args.out / "geometry_tolerance.csv", geometry_rows)
    write_csv(args.out / "displacement_loading.csv", load_rows)
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Plots: one chart per figure, no style/color overrides.
    hinge_x = np.array([row["hinge_eta"] for row in hinge_rows[1:]])
    hinge_y = np.array([row["shape_error_after_global_gain"] for row in hinge_rows[1:]])
    plt.figure(figsize=(7.0, 4.5))
    plt.loglog(hinge_x, hinge_y, marker="o")
    plt.xlabel(r"Hinge stiffness ratio $\eta=k_\theta L/EI$")
    plt.ylabel("Relative transform shape error")
    plt.title("N=8 force-whiffletree: flexure-hinge sensitivity")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(args.out / "hinge_sensitivity.png", dpi=180)
    plt.close()

    geometry_x = np.array(
        [row["tap_position_sigma_fraction_of_bar_length"] for row in geometry_rows]
    )
    geometry_median = np.array([row["median_shape_error"] for row in geometry_rows])
    geometry_p95 = np.array([row["p95_shape_error"] for row in geometry_rows])
    plt.figure(figsize=(7.0, 4.5))
    plt.loglog(geometry_x, geometry_median, marker="o", label="median")
    plt.loglog(geometry_x, geometry_p95, marker="o", label="95th percentile")
    plt.xlabel("Tap-position standard deviation / bar length")
    plt.ylabel("Relative transform shape error")
    plt.title("N=8 force-whiffletree: geometric tolerance")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "geometry_tolerance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    for design in ("equal_EI", "matched_EI"):
        rows = [row for row in load_rows if row["design"] == design and row["dimensionless_output_load"] > 0.0]
        x = np.array([row["dimensionless_output_load"] for row in rows])
        y = np.array([row["shape_error_after_global_gain"] for row in rows])
        plt.loglog(x, np.maximum(y, 1e-16), marker="o", label=design)
    plt.xlabel(r"Output load ratio $k_L L^3/EI_0$")
    plt.ylabel("Relative transform shape error")
    plt.title("N=8 displacement-whiffletree: impedance matching")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "displacement_impedance_matching.png", dpi=180)
    plt.close()

    report = f"""# N=8 BFFT whiffletree / beam-element result

The strongest architecture is force-mode.  The mass-potential-gauged cone
stages are column-stochastic.  Building the reciprocal displacement constraint
`q_s = A_s.T q_(s+1)` from convex interpolation bars gives, by virtual work,
`f_(s+1) = A_s f_s`.  This is the original cone-DIF in force coordinates.

## Numerical findings

- The N=8 network uses {sum(bars_per_stage)} binary interpolation bars
  ({len(force_network.elements)} Euler-Bernoulli elements), split by stage as
  {bars_per_stage}.
- With ideal pins, the transform is exact to numerical precision even when
  all three stage rigidities and all sixteen output-sensor spring constants are
  varied independently over many orders of magnitude.  Across 100 random
  configurations, the worst relative matrix error was {max(random_errors):.3e}.
- Beam rigidity and output sensor compliance therefore change displacement,
  not the force split.  Geometry and parasitic hinge moments are the first-order
  error sources.
- At `eta = k_theta L/EI = 0.01`, hinge-induced shape error is
  {next(row['shape_error_after_global_gain'] for row in hinge_rows if row['hinge_eta'] == 1e-2):.3e}.
- A 0.1% one-sigma tap-position error gives median transform error
  {next(row['median_shape_error'] for row in geometry_rows if row['tap_position_sigma_fraction_of_bar_length'] == 1e-3):.3e}
  and 95th-percentile error
  {next(row['p95_shape_error'] for row in geometry_rows if row['tap_position_sigma_fraction_of_bar_length'] == 1e-3):.3e}.

## Displacement alternative

A forward kinematic gauge makes every cone row a convex combination, eliminating
internal gain levers.  Its differential output compliance becomes exactly
isotropic when

- `EI_1 / EI_0 = (-338 + 248 sqrt(2))/7 = {matched_ei[1]:.12f}`
- `EI_2 / EI_0 = (482 - 124 sqrt(2))/161 = {matched_ei[2]:.12f}`

The computed compliance-isotropy defect is {compliance_defect:.3e}.  Equal
output loading then causes only the scalar attenuation
`1 / (1 + k_L * {scalar_compliance:.12f})`, with no Fourier-shape error.

## Boundary of this result

This is a linear small-deflection beam model.  It does not yet include planar
layout, axial/geometric nonlinearity, stress concentration in real notch or
cross-spring pivots, collision, or fabrication-rule constraints.  Those belong
in the next 2D frame/continuum model.  The static architecture question is now
much narrower: build moment-free force-splitting joints with accurately placed
lever taps.
"""
    (args.out / "REPORT.md").write_text(report)

    print(f"force random max error: {max(random_errors):.3e}")
    print(f"binary bars: {sum(bars_per_stage)}; EB elements: {len(force_network.elements)}")
    print(f"matched EI ratios: {matched_ei.tolist()}")
    print(f"compliance isotropy defect: {compliance_defect:.3e}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
