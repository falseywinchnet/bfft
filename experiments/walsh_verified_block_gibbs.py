#!/usr/bin/env python3
"""Finite audit of the verified Gibbs parity law and block entrance.

The theorem in ``notes/WALSH_VERIFIED_GIBBS_SEED.md`` gives overwhelming
stationary mass to minimum verifier-energy cells.  This audit deliberately
uses the stronger oracle energy: the minimum enumerated lattice norm in each
parity class.  It compares dense lattice fixtures with a planted-word control
whose stationary mass is also overwhelming but whose half-block heat bath
has exponentially small entrance probability.

The experiment is evidence about entrance only; no finite spectral gap is
used in the asymptotic theorem.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import generic_basis
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_verified_block_gibbs.json"


def parity_index(coefficients: np.ndarray) -> int:
    value = 0
    for bit, coefficient in enumerate(np.asarray(coefficients, dtype=np.int64)):
        value |= (int(coefficient) & 1) << bit
    return value


def minimum_parity_energies(basis: np.ndarray, cutoff: int) -> np.ndarray:
    """Enumerate the minimum nonzero squared norm in every parity class."""
    n = basis.shape[0]
    energies = np.full(1 << n, np.inf, dtype=np.float64)
    values = range(-cutoff, cutoff + 1)
    for entry in itertools.product(values, repeat=n):
        coefficients = np.asarray(entry, dtype=np.int64)
        if not np.any(coefficients):
            continue
        vector = basis @ coefficients
        norm2 = float(vector @ vector)
        address = parity_index(coefficients)
        if norm2 < energies[address]:
            energies[address] = norm2
    finite = energies[np.isfinite(energies)]
    if finite.size != energies.size:
        raise RuntimeError("coefficient cutoff missed a parity class")
    return energies


def cold_weights(energies: np.ndarray) -> tuple[np.ndarray, float]:
    """Choose beta so nonminimum stationary mass is at most 2^(-2n)."""
    n = int(round(math.log2(len(energies))))
    minimum = float(np.min(energies))
    gaps = energies[energies > minimum + 1e-12] - minimum
    gap = float(np.min(gaps)) if gaps.size else 1.0
    beta = 3.0 * n * math.log(2.0) / gap
    log_weights = -beta * (energies - minimum)
    log_weights -= float(np.max(log_weights))
    # The theorem uses exact dyadic weights.  The finite spectral diagnostic
    # keeps exponentially irrelevant states positive so the reversible
    # similarity transform remains numerically defined.
    log_weights = np.maximum(log_weights, -700.0)
    weights = np.exp(log_weights)
    weights /= float(np.sum(weights))
    return weights, beta


def cyclic_blocks(n: int, block_size: int) -> list[tuple[int, ...]]:
    return [tuple((start + offset) % n for offset in range(block_size)) for start in range(n)]


def heat_bath_transition(
    weights: np.ndarray,
    blocks: list[tuple[int, ...]],
) -> np.ndarray:
    """Exact random-block heat-bath transition on a small Boolean cube."""
    states = len(weights)
    transition = np.zeros((states, states), dtype=np.float64)
    for state in range(states):
        for block in blocks:
            mask = sum(1 << bit for bit in block)
            outside = state & ~mask
            candidates = []
            for word in range(1 << len(block)):
                candidate = outside
                for local, bit in enumerate(block):
                    candidate |= ((word >> local) & 1) << bit
                candidates.append(candidate)
            conditional = weights[candidates]
            conditional = conditional / float(np.sum(conditional))
            transition[state, candidates] += conditional / len(blocks)
    return transition


def reversible_gap(transition: np.ndarray, stationary: np.ndarray) -> float:
    root = np.sqrt(stationary)
    similarity = root[:, None] * transition / root[None, :]
    similarity = 0.5 * (similarity + similarity.T)
    eigenvalues = np.linalg.eigvalsh(similarity)
    return float(max(0.0, 1.0 - eigenvalues[-2]))


def audit_energy(name: str, energies: np.ndarray) -> dict[str, object]:
    n = int(round(math.log2(len(energies))))
    target = np.flatnonzero(energies <= float(np.min(energies)) + 1e-12)
    weights, beta = cold_weights(energies)
    blocks = cyclic_blocks(n, (n + 1) // 2)
    transition = heat_bath_transition(weights, blocks)
    distribution = np.full(1 << n, 1.0 / (1 << n), dtype=np.float64)
    entrance = []
    target_mass = lambda law: float(np.sum(law[target]))
    entrance.append(target_mass(distribution))
    for _ in range(2 * n):
        distribution = distribution @ transition
        entrance.append(target_mass(distribution))
    return {
        "family": name,
        "dimension": n,
        "block_size": (n + 1) // 2,
        "minimum_cell_count": int(len(target)),
        "inverse_temperature": beta,
        "stationary_minimum_mass": target_mass(weights),
        "random_block_spectral_gap": reversible_gap(transition, weights),
        "minimum_mass_from_uniform_by_step": entrance,
    }


def build_report(min_dimension: int, max_dimension: int, cutoff: int, seed: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for n in range(min_dimension, max_dimension + 1):
        needle = np.ones(1 << n, dtype=np.float64)
        needle[(1 << n) - 1] = 0.0
        rows.append(audit_energy("planted_parity_control", needle))
        fixtures = [
            ("simplex_cancellation", simplex_cancellation_basis(n, 0.97)),
            ("generic", generic_basis(n, rng)),
        ]
        for name, basis in fixtures:
            rows.append(audit_energy(name, minimum_parity_energies(basis, cutoff)))
    return {
        "experiment": "walsh_verified_block_gibbs",
        "status": (
            "stationary mass is proved; finite block entrance is diagnostic and "
            "does not prove worst-case mixing"
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dimension", type=int, default=3)
    parser.add_argument("--max-dimension", type=int, default=7)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.min_dimension, args.max_dimension, args.cutoff, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
