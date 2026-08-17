#!/usr/bin/env python3
"""Causal frontier recovery for the matrix-valued Walsh Hessian spike.

The full algorithm in ``2608.02478v2`` materializes every matrix-valued
Walsh coefficient.  This experiment asks whether a first-arrival frontier
can locate the useful parity while visiting only a narrow collection of
partial parity cells.

For a matrix histogram A(y), split y=(x,z), and fix a d-bit Walsh prefix p.
Parseval on the remaining coordinates gives the exact descendant energy

    E(p) = sum_{theta extends p} ||Ahat(theta)||_F^2
         = 2^(ell-d) sum_z ||sum_x (-1)^(p.x) A(x,z)||_F^2.

The two child energies add exactly to the parent energy.  This makes E a
causal, conservative action on the parity tree.  For random samples, two
independent panels replace each squared norm by an inner product.  The
result is unbiased and omits the large diagonal/self-noise term.

The experiment deliberately reports target retention at *every* depth.
Final recovery after accidentally dropping the true branch is not counted.
It also reports a sample-touch work proxy, so a narrow output beam is not
mistaken for a sub-Walsh algorithm when scoring the beam is itself costly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from experiments.walsh_hessian_noise_audit import (
    ROOT,
    _binary_index,
    _gf2_inverse,
    _integer_chunk,
    _matrix_opnorms,
    _sym_outer,
    _symmetric_layout,
    fwht,
    generic_basis,
    random_gl2,
    shortest_vector_coefficients,
)


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_eikonal_recovery.json"


def _parity_table(size: int) -> np.ndarray:
    values = np.arange(size, dtype=np.uint64)
    values ^= values >> np.uint64(32)
    values ^= values >> np.uint64(16)
    values ^= values >> np.uint64(8)
    values ^= values >> np.uint64(4)
    values &= np.uint64(0xF)
    return ((np.uint64(0x6996) >> values) & np.uint64(1)).astype(np.int8)


def descendant_energy(
    histogram: np.ndarray,
    prefix: int,
    depth: int,
    parity: np.ndarray | None = None,
) -> float:
    """Exact total squared Walsh energy below one low-bit prefix."""
    histogram = np.asarray(histogram, dtype=np.float64)
    outputs = histogram.shape[0]
    if outputs < 1 or outputs & (outputs - 1):
        raise ValueError("histogram length must be a power of two")
    ell = outputs.bit_length() - 1
    if not 0 <= depth <= ell or not 0 <= prefix < (1 << depth):
        raise ValueError("prefix does not belong to the requested depth")
    if parity is None:
        parity = _parity_table(outputs)
    y = np.arange(outputs, dtype=np.int64)
    sign = 1.0 - 2.0 * parity[np.bitwise_and(y, prefix)]
    suffixes = 1 << (ell - depth)
    buckets = np.zeros((suffixes,) + histogram.shape[1:], dtype=np.float64)
    np.add.at(buckets, y >> depth, sign.reshape((-1,) + (1,) * (histogram.ndim - 1)) * histogram)
    return float(suffixes * np.sum(buckets * buckets))


def split_descendant_energy(
    labels: np.ndarray,
    values: np.ndarray,
    panel: np.ndarray,
    prefix: int,
    depth: int,
    ell: int,
    parity: np.ndarray | None = None,
) -> float:
    """Unbiased two-panel estimate of descendant Walsh energy."""
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    panel = np.asarray(panel, dtype=np.int8)
    if labels.shape != panel.shape or values.shape[0] != labels.size:
        raise ValueError("labels, values, and panel must describe the same samples")
    if not 0 <= depth <= ell or not 0 <= prefix < (1 << depth):
        raise ValueError("prefix does not belong to the requested depth")
    if parity is None:
        parity = _parity_table(1 << ell)
    sign = 1.0 - 2.0 * parity[np.bitwise_and(labels, prefix)]
    suffixes = 1 << (ell - depth)
    accumulators = []
    for which in (0, 1):
        mask = panel == which
        count = int(np.count_nonzero(mask))
        if count == 0:
            return -math.inf
        bucket = np.zeros((suffixes,) + values.shape[1:], dtype=np.float64)
        factor = sign[mask].reshape((-1,) + (1,) * (values.ndim - 1))
        np.add.at(bucket, labels[mask] >> depth, factor * values[mask] / count)
        accumulators.append(bucket)
    return float(suffixes * np.sum(accumulators[0] * accumulators[1]))


def _compress_unmasked(labels: np.ndarray, mask: int, ell: int) -> np.ndarray:
    """Pack the label bits outside mask into consecutive low bits."""
    labels = np.asarray(labels, dtype=np.int64)
    packed = np.zeros(labels.shape, dtype=np.int64)
    target_bit = 0
    for source_bit in range(ell):
        if not (mask >> source_bit) & 1:
            packed |= ((labels >> source_bit) & 1) << target_bit
            target_bit += 1
    return packed


def masked_descendant_energy(
    histogram: np.ndarray,
    prefix: int,
    mask: int,
    parity: np.ndarray | None = None,
) -> float:
    """Exact descendant energy for an arbitrary set of resolved bits."""
    histogram = np.asarray(histogram, dtype=np.float64)
    outputs = histogram.shape[0]
    ell = outputs.bit_length() - 1
    if outputs != 1 << ell or prefix & ~mask or mask >= 1 << ell:
        raise ValueError("invalid histogram, mask, or prefix")
    if parity is None:
        parity = _parity_table(outputs)
    y = np.arange(outputs, dtype=np.int64)
    sign = 1.0 - 2.0 * parity[np.bitwise_and(y, prefix)]
    suffixes = 1 << (ell - mask.bit_count())
    keys = _compress_unmasked(y, mask, ell)
    buckets = np.zeros((suffixes,) + histogram.shape[1:], dtype=np.float64)
    factor = sign.reshape((-1,) + (1,) * (histogram.ndim - 1))
    np.add.at(buckets, keys, factor * histogram)
    return float(suffixes * np.sum(buckets * buckets))


def masked_split_descendant_energy(
    labels: np.ndarray,
    values: np.ndarray,
    panel: np.ndarray,
    prefix: int,
    mask: int,
    ell: int,
    parity: np.ndarray | None = None,
) -> float:
    """Two-panel descendant-energy estimate for an arbitrary resolved mask."""
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    panel = np.asarray(panel, dtype=np.int8)
    if prefix & ~mask or mask >= 1 << ell:
        raise ValueError("prefix must be supported on the resolved mask")
    if parity is None:
        parity = _parity_table(1 << ell)
    sign = 1.0 - 2.0 * parity[np.bitwise_and(labels, prefix)]
    suffixes = 1 << (ell - mask.bit_count())
    keys = _compress_unmasked(labels, mask, ell)
    accumulators = []
    for which in (0, 1):
        selected = panel == which
        count = int(np.count_nonzero(selected))
        if count == 0:
            return -math.inf
        buckets = np.zeros((suffixes,) + values.shape[1:], dtype=np.float64)
        factor = sign[selected].reshape(
            (-1,) + (1,) * (values.ndim - 1)
        )
        np.add.at(
            buckets, keys[selected], factor * values[selected] / count
        )
        accumulators.append(buckets)
    return float(suffixes * np.sum(accumulators[0] * accumulators[1]))


def causal_frontier(
    score,
    ell: int,
    width: int,
    theta_star: int | None = None,
) -> dict:
    """Advance a conservative best-energy beam through the parity tree."""
    if ell < 1 or width < 1:
        raise ValueError("ell and width must be positive")
    frontier = [0]
    visited = 0
    target_retained = True
    trace = []
    for depth in range(1, ell + 1):
        bit = 1 << (depth - 1)
        candidates = [child for parent in frontier for child in (parent, parent | bit)]
        scored = [(float(score(child, depth)), child) for child in candidates]
        visited += len(scored)
        scored.sort(key=lambda item: (-item[0], item[1]))
        frontier = [item[1] for item in scored[:width]]
        if theta_star is not None:
            target_prefix = theta_star & ((1 << depth) - 1)
            target_at_depth = target_prefix in frontier
            target_retained = target_retained and target_at_depth
        else:
            target_prefix = None
            target_at_depth = None
        trace.append({
            "depth": depth,
            "candidate_count": len(scored),
            "frontier": frontier.copy(),
            "scores": [item[0] for item in scored[:width]],
            "target_prefix": target_prefix,
            "target_retained": target_at_depth,
        })
    return {
        "frontier": frontier,
        "visited_nodes": visited,
        "target_retained_all_depths": target_retained,
        "target_recovered": theta_star in frontier if theta_star is not None else None,
        "trace": trace,
    }


def adaptive_causal_frontier(
    score,
    ell: int,
    width: int,
    theta_star: int | None = None,
) -> dict:
    """Choose each unresolved parity direction by causal concentration."""
    if ell < 1 or width < 1:
        raise ValueError("ell and width must be positive")
    frontier = [0]
    mask = 0
    visited = 0
    target_retained = True
    trace = []
    for depth in range(1, ell + 1):
        proposals = []
        for bit_index in range(ell):
            bit = 1 << bit_index
            if mask & bit:
                continue
            candidate_mask = mask | bit
            candidates = [
                child for parent in frontier for child in (parent, parent | bit)
            ]
            scored = [
                (float(score(child, candidate_mask)), child)
                for child in candidates
            ]
            visited += len(scored)
            scored.sort(key=lambda item: (-item[0], item[1]))
            kept = scored[:width]
            concentration = float(np.sum([item[0] for item in kept]))
            proposals.append((concentration, -bit_index, bit, kept))
        proposals.sort(key=lambda item: (-item[0], -item[1]))
        _, _, chosen_bit, kept = proposals[0]
        mask |= chosen_bit
        frontier = [item[1] for item in kept]
        if theta_star is not None:
            target_prefix = theta_star & mask
            target_at_depth = target_prefix in frontier
            target_retained = target_retained and target_at_depth
        else:
            target_prefix = None
            target_at_depth = None
        trace.append({
            "depth": depth,
            "chosen_bit": chosen_bit.bit_length() - 1,
            "resolved_mask": mask,
            "frontier": frontier.copy(),
            "scores": [item[0] for item in kept],
            "target_prefix": target_prefix,
            "target_retained": target_at_depth,
        })
    return {
        "frontier": frontier,
        "visited_nodes": visited,
        "target_retained_all_depths": target_retained,
        "target_recovered": theta_star in frontier if theta_star is not None else None,
        "bit_order": [row["chosen_bit"] for row in trace],
        "trace": trace,
    }


def certified_causal_frontier(
    score,
    ell: int,
    threshold: float,
    theta_star: int | None = None,
) -> dict:
    """Keep every cell whose descendant-energy upper bound can reach tau."""
    if ell < 1 or threshold < 0.0:
        raise ValueError("ell must be positive and threshold nonnegative")
    frontier = [0]
    visited = 0
    target_retained = True
    trace = []
    for depth in range(1, ell + 1):
        bit = 1 << (depth - 1)
        candidates = [child for parent in frontier for child in (parent, parent | bit)]
        scored = [(float(score(child, depth)), child) for child in candidates]
        visited += len(scored)
        frontier = [child for value, child in scored if value >= threshold]
        if theta_star is not None:
            target_prefix = theta_star & ((1 << depth) - 1)
            target_at_depth = target_prefix in frontier
            target_retained = target_retained and target_at_depth
        else:
            target_prefix = None
            target_at_depth = None
        trace.append({
            "depth": depth,
            "candidate_count": len(scored),
            "frontier_width": len(frontier),
            "target_prefix": target_prefix,
            "target_retained": target_at_depth,
        })
        if not frontier:
            break
    return {
        "frontier": frontier,
        "visited_nodes": visited,
        "maximum_frontier_width": max(
            (row["frontier_width"] for row in trace), default=0
        ),
        "target_retained_all_depths": target_retained,
        "target_recovered": theta_star in frontier if theta_star is not None else None,
        "trace": trace,
    }


def _build_population(
    n: int,
    cutoff: int,
    seed: int,
    r: float,
    R: float,
    chi: float,
    chunk_size: int,
) -> dict:
    rng = np.random.default_rng(seed + 1009 * n)
    basis = generic_basis(n, rng)
    shortest, shortest_coeff = shortest_vector_coefficients(basis)
    h = min(max(int(math.floor(chi * n)), 0), n - 1)
    ell = n - h
    outputs = 1 << ell
    cosets = 1 << h
    transform = random_gl2(n, rng)
    inverse = _gf2_inverse(transform)
    row, column, sym_scale, trace_direction = _symmetric_layout(n)
    matrix_width = row.size
    xi_r = math.sqrt(4.0 * n * r * math.log(2.0) / (math.pi * shortest ** 2))
    xi_R = math.sqrt(4.0 * n * R * math.log(2.0) / (math.pi * shortest ** 2))
    inverse_basis = np.linalg.inv(basis)
    count = (2 * cutoff + 1) ** n
    source_mass = np.zeros(cosets)
    retained_mass = np.zeros(cosets)
    boundary_mass = np.zeros(cosets)
    histogram = np.zeros((cosets, outputs, matrix_width))
    iota = 0.5 * math.log2(R * R / (r * (2.0 * R - r)))
    retain_normalizer = (2.0 ** (iota * n)) * ((r / R) ** (0.5 * n))

    for start in range(0, count, chunk_size):
        coeff = _integer_chunk(start, min(start + chunk_size, count), n, cutoff)
        dual = coeff @ inverse_basis
        norm2 = np.einsum("ij,ij->i", dual, dual)
        rho_R = np.exp(-math.pi * norm2 / (xi_R * xi_R))
        rho_r = np.exp(-math.pi * norm2 / (xi_r * xi_r))
        weight = np.divide(rho_r, rho_R, out=np.zeros_like(rho_r), where=rho_R > 0)
        retain = np.minimum(1.0, weight / retain_normalizer)
        coordinates = (((coeff & 1).astype(np.uint8) @ inverse) & 1)
        j_index = _binary_index(coordinates[:, :h])
        y_index = _binary_index(coordinates[:, h:])
        sym = _sym_outer(dual, row, column, sym_scale)
        radial = sym @ trace_direction
        traceless = sym - radial[:, None] * trace_direction[None, :]
        np.add.at(source_mass, j_index, rho_R)
        np.add.at(retained_mass, j_index, rho_R * retain)
        np.add.at(histogram, (j_index, y_index), rho_r[:, None] * traceless)
        boundary = np.any(np.abs(coeff) == cutoff, axis=1)
        np.add.at(boundary_mass, j_index, rho_R * boundary)

    histogram /= source_mass[:, None, None]
    u_star = shortest_coeff & 1
    transformed_u = (transform @ u_star.astype(np.uint8)) & 1
    theta_star = int(_binary_index(transformed_u[h:][None, :])[0])
    return {
        "dimension": n,
        "cutoff": cutoff,
        "enumerated_dual_points": count,
        "h": h,
        "ell": ell,
        "outputs": outputs,
        "theta_star": theta_star,
        "histogram": histogram,
        "source_mass": source_mass,
        "retained_mass": retained_mass,
        "retain_normalizer": retain_normalizer,
        "boundary_fraction": boundary_mass / source_mass,
        "basis": basis,
        "inverse_basis": inverse_basis,
        "inverse_binary_transform": inverse,
        "row": row,
        "column": column,
        "sym_scale": sym_scale,
        "trace_direction": trace_direction,
        "xi_r": xi_r,
        "xi_R": xi_R,
    }


def _draw_samples(
    population: dict,
    coset: int,
    sample_count: int,
    rng: np.random.Generator,
    chunk_size: int,
    horvitz_thompson: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact finite-cube IID draws via a streaming inverse CDF."""
    n = population["dimension"]
    cutoff = population["cutoff"]
    mass_key = "retained_mass" if horvitz_thompson else "source_mass"
    total = float(population[mass_key][coset])
    thresholds = np.sort(rng.random(sample_count) * total)
    labels = np.empty(sample_count, dtype=np.int64)
    values = np.empty((sample_count, population["row"].size), dtype=np.float64)
    cursor = 0
    cumulative = 0.0
    count = population["enumerated_dual_points"]
    inverse_basis = population["inverse_basis"]
    inverse = population["inverse_binary_transform"]
    h = population["h"]
    xi_r = population["xi_r"]
    xi_R = population["xi_R"]
    retain_normalizer = population["retain_normalizer"]
    row = population["row"]
    column = population["column"]
    sym_scale = population["sym_scale"]
    trace_direction = population["trace_direction"]

    for start in range(0, count, chunk_size):
        if cursor == sample_count:
            break
        coeff = _integer_chunk(start, min(start + chunk_size, count), n, cutoff)
        dual = coeff @ inverse_basis
        norm2 = np.einsum("ij,ij->i", dual, dual)
        rho_R = np.exp(-math.pi * norm2 / (xi_R * xi_R))
        rho_r = np.exp(-math.pi * norm2 / (xi_r * xi_r))
        weight = np.divide(rho_r, rho_R, out=np.zeros_like(rho_r), where=rho_R > 0)
        retain = np.minimum(1.0, weight / retain_normalizer)
        coordinates = (((coeff & 1).astype(np.uint8) @ inverse) & 1)
        j_index = _binary_index(coordinates[:, :h])
        mask = j_index == coset
        if not np.any(mask):
            continue
        local_weights = rho_R[mask] * (retain[mask] if horvitz_thompson else 1.0)
        local_cdf = cumulative + np.cumsum(local_weights)
        stop = int(np.searchsorted(thresholds, local_cdf[-1], side="right"))
        if stop > cursor:
            positions = np.searchsorted(local_cdf, thresholds[cursor:stop], side="left")
            chosen_coeff = coeff[mask][positions]
            chosen_dual = dual[mask][positions]
            chosen_norm2 = norm2[mask][positions]
            chosen_retain = retain[mask][positions]
            chosen_coordinates = coordinates[mask][positions]
            labels[cursor:stop] = _binary_index(chosen_coordinates[:, h:])
            rho_ratio = np.exp(
                -math.pi * chosen_norm2 *
                (1.0 / (xi_r * xi_r) - 1.0 / (xi_R * xi_R))
            )
            sym = _sym_outer(chosen_dual, row, column, sym_scale)
            radial = sym @ trace_direction
            traceless = sym - radial[:, None] * trace_direction[None, :]
            if horvitz_thompson:
                rho_ratio = rho_ratio / np.maximum(chosen_retain, 1e-300)
            values[cursor:stop] = rho_ratio[:, None] * traceless
            cursor = stop
        cumulative = float(local_cdf[-1])

    if cursor != sample_count:
        # Floating endpoint roundoff can affect only thresholds in the final ulp.
        labels[cursor:] = labels[cursor - 1]
        values[cursor:] = values[cursor - 1]
    return labels, values


def audit_eikonal_recovery(
    n: int,
    *,
    widths: tuple[int, ...] = (1, 2, 4, 8, 16),
    sample_counts: tuple[int, ...] = (256, 1024),
    trials: int = 16,
    cutoff: int = 2,
    seed: int = 260802478,
    r: float = 0.2222355,
    R: float = 0.400613,
    chi: float = 0.3961331,
    chunk_size: int = 100_000,
) -> dict:
    started = time.perf_counter()
    population = _build_population(n, cutoff, seed, r, R, chi, chunk_size)
    ell = population["ell"]
    outputs = population["outputs"]
    theta_star = population["theta_star"]
    parity = _parity_table(outputs)
    exact_reports = []
    for coset in range(population["histogram"].shape[0]):
        histogram = population["histogram"][coset]
        transformed = fwht(histogram)
        leaf_energy = np.einsum("ij,ij->i", transformed, transformed)
        target_energy_rank = 1 + int(np.count_nonzero(
            leaf_energy > leaf_energy[theta_star] + 1e-14
        ))
        target_energy = float(leaf_energy[theta_star])
        beams = {}
        adaptive_beams = {}
        for width in widths:
            result = causal_frontier(
                lambda prefix, depth: descendant_energy(
                    histogram, prefix, depth, parity
                ),
                ell, min(width, outputs), theta_star,
            )
            beams[str(width)] = {
                "target_retained_all_depths": result[
                    "target_retained_all_depths"
                ],
                "target_recovered": result["target_recovered"],
                "visited_nodes": result["visited_nodes"],
                "first_missed_depth": next((
                    row["depth"] for row in result["trace"]
                    if not row["target_retained"]
                ), None),
            }
            adaptive = adaptive_causal_frontier(
                lambda prefix, mask: masked_descendant_energy(
                    histogram, prefix, mask, parity
                ),
                ell, min(width, outputs), theta_star,
            )
            adaptive_beams[str(width)] = {
                "target_retained_all_depths": adaptive[
                    "target_retained_all_depths"
                ],
                "target_recovered": adaptive["target_recovered"],
                "visited_nodes": adaptive["visited_nodes"],
                "bit_order": adaptive["bit_order"],
                "first_missed_depth": next((
                    row["depth"] for row in adaptive["trace"]
                    if not row["target_retained"]
                ), None),
            }
        certified = {}
        for threshold_ratio in (1.0, 0.5, 0.25):
            result = certified_causal_frontier(
                lambda prefix, depth: descendant_energy(
                    histogram, prefix, depth, parity
                ),
                ell, threshold_ratio * target_energy, theta_star,
            )
            certified[str(threshold_ratio)] = {
                "threshold_over_target_energy": threshold_ratio,
                "maximum_frontier_width": result["maximum_frontier_width"],
                "final_frontier_width": len(result["frontier"]),
                "visited_nodes": result["visited_nodes"],
                "target_retained_all_depths": result[
                    "target_retained_all_depths"
                ],
            }
        exact_reports.append({
            "coset": coset,
            "target_leaf_energy_rank": target_energy_rank,
            "target_leaf_energy": target_energy,
            "beams": beams,
            "adaptive_beams": adaptive_beams,
            "certified_frontiers": certified,
        })

    rng = np.random.default_rng(seed + 7919 * n)
    sampled_reports = []
    for sample_count in sample_counts:
        by_width = {str(width): [] for width in widths}
        adaptive_by_width = {str(width): [] for width in widths}
        full_recovery = []
        for trial in range(trials):
            coset = trial % population["histogram"].shape[0]
            labels, values = _draw_samples(
                population, coset, sample_count, rng, chunk_size,
                horvitz_thompson=True,
            )
            panel = rng.integers(0, 2, size=sample_count, dtype=np.int8)
            # Avoid the vanishingly unlikely empty-panel degeneracy.
            panel[0], panel[-1] = 0, 1
            sample_histogram = np.zeros((outputs, values.shape[1]))
            np.add.at(sample_histogram, labels, values / sample_count)
            matrices = fwht(sample_histogram)
            leaf_energy = np.einsum("ij,ij->i", matrices, matrices)
            full_recovery.append(int(np.argmax(leaf_energy)) == theta_star)
            for width in widths:
                result = causal_frontier(
                    lambda prefix, depth: split_descendant_energy(
                        labels, values, panel, prefix, depth, ell, parity
                    ),
                    ell, min(width, outputs), theta_star,
                )
                by_width[str(width)].append(result)
                adaptive = adaptive_causal_frontier(
                    lambda prefix, mask: masked_split_descendant_energy(
                        labels, values, panel, prefix, mask, ell, parity
                    ),
                    ell, min(width, outputs), theta_star,
                )
                adaptive_by_width[str(width)].append(adaptive)
        width_summaries = {}
        adaptive_width_summaries = {}
        for width in widths:
            rows = by_width[str(width)]
            visited = float(np.mean([row["visited_nodes"] for row in rows]))
            # Each node score touches every stored sample.  Full FWHT work is
            # represented as ell touches of every output matrix.
            width_summaries[str(width)] = {
                "retention_probability": float(np.mean([
                    row["target_retained_all_depths"] for row in rows
                ])),
                "recovery_probability": float(np.mean([
                    row["target_recovered"] for row in rows
                ])),
                "mean_visited_nodes": visited,
                "sample_touch_work": visited * sample_count,
                "sample_touch_over_full_fwht_butterflies":
                    visited * sample_count / max(outputs * ell, 1),
            }
            adaptive_rows = adaptive_by_width[str(width)]
            adaptive_visited = float(np.mean([
                row["visited_nodes"] for row in adaptive_rows
            ]))
            adaptive_width_summaries[str(width)] = {
                "retention_probability": float(np.mean([
                    row["target_retained_all_depths"] for row in adaptive_rows
                ])),
                "recovery_probability": float(np.mean([
                    row["target_recovered"] for row in adaptive_rows
                ])),
                "mean_visited_nodes": adaptive_visited,
                "sample_touch_work": adaptive_visited * sample_count,
                "sample_touch_over_full_fwht_butterflies":
                    adaptive_visited * sample_count / max(outputs * ell, 1),
            }
        sampled_reports.append({
            "sample_count": sample_count,
            "trials": trials,
            "full_frobenius_argmax_recovery_probability": float(
                np.mean(full_recovery)
            ),
            "widths": width_summaries,
            "adaptive_widths": adaptive_width_summaries,
        })

    return {
        "dimension": n,
        "cutoff": cutoff,
        "enumerated_dual_points": population["enumerated_dual_points"],
        "elapsed_seconds": time.perf_counter() - started,
        "h": population["h"],
        "ell": ell,
        "walsh_outputs": outputs,
        "theta_star": theta_star,
        "maximum_boundary_mass_fraction": float(np.max(
            population["boundary_fraction"]
        )),
        "exact_population": exact_reports,
        "sampled": sampled_reports,
        "sample_model": (
            "IID draws conditional on Algorithm 5 Horvitz-Thompson retention; "
            "each stored contribution is weighted by w/pi."
        ),
    }


def summarize(report: dict) -> dict:
    widths = report["exact_population"][0]["beams"].keys()
    exact = {
        width: float(np.mean([
            row["beams"][width]["target_retained_all_depths"]
            for row in report["exact_population"]
        ])) for width in widths
    }
    adaptive_exact = {
        width: float(np.mean([
            row["adaptive_beams"][width]["target_retained_all_depths"]
            for row in report["exact_population"]
        ])) for width in widths
    }
    certified = {
        ratio: {
            "maximum_width": int(max(
                row["certified_frontiers"][ratio]["maximum_frontier_width"]
                for row in report["exact_population"]
            )),
            "median_width": float(np.median([
                row["certified_frontiers"][ratio]["maximum_frontier_width"]
                for row in report["exact_population"]
            ])),
            "maximum_width_fraction": float(max(
                row["certified_frontiers"][ratio]["maximum_frontier_width"]
                for row in report["exact_population"]
            ) / report["walsh_outputs"]),
        } for ratio in ("1.0", "0.5", "0.25")
    }
    return {
        "dimension": report["dimension"],
        "ell": report["ell"],
        "walsh_outputs": report["walsh_outputs"],
        "maximum_boundary_mass_fraction": report[
            "maximum_boundary_mass_fraction"
        ],
        "exact_population_retention_by_width": exact,
        "adaptive_exact_population_retention_by_width": adaptive_exact,
        "oracle_threshold_certified_frontiers": certified,
        "sampled": [{
            "sample_count": row["sample_count"],
            "full_recovery_probability": row[
                "full_frobenius_argmax_recovery_probability"
            ],
            "widths": row["widths"],
            "adaptive_widths": row["adaptive_widths"],
        } for row in report["sampled"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(6, 7, 8))
    parser.add_argument("--widths", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--sample-counts", type=int, nargs="*", default=(256, 1024))
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reports = []
    summaries = []
    for n in args.dimensions:
        report = audit_eikonal_recovery(
            n, widths=tuple(args.widths), sample_counts=tuple(args.sample_counts),
            trials=args.trials, cutoff=args.cutoff, seed=args.seed,
            chunk_size=args.chunk_size,
        )
        summary = summarize(report)
        reports.append(report)
        summaries.append(summary)
        exact = summary["exact_population_retention_by_width"]
        print(
            f"n={n}: N={report['walsh_outputs']} "
            + " ".join(f"B{width}={probability:.2f}" for width, probability in exact.items()),
            flush=True,
        )
    payload = {
        "experiment": "eikonal_matrix_walsh_recovery",
        "guardrail": (
            "Finite-cube causal-frontier measurements are a recovery and work "
            "audit, not an asymptotic SVP theorem."
        ),
        "summaries": summaries,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
