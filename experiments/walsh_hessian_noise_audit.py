#!/usr/bin/env python3
"""Exact small-dimensional audit of the midpoint-Hessian Walsh noise floor.

This experiment implements the estimator in Eqs. (26)--(34) of
``2608.02478v2`` on a finite, explicitly enumerated dual-lattice cube.  It is
not an SVP implementation and it does not extrapolate a small-n measurement
into an asymptotic theorem.  Its purpose is narrower: determine which part of
the empirical Walsh floor is radial identity noise, which part is traceless
anisotropic noise, whether the latter is white across parity coefficients,
and how importance sampling and Horvitz--Thompson sparsification change that
covariance.

For a fixed affine coset j, let Y=V_P(X) and let Z(X) be one matrix-valued
sample contribution.  The experiment computes exactly on the truncated cube

    T_theta = E[Z(X) (-1)^(theta.Y)]
    K_delta = E[||Z(X)||_F^2 (-1)^(delta.Y)]

and therefore the cross-output covariance Gram matrix for one sample,

    G[theta,phi] = K[theta xor phi] - <T_theta,T_phi>_F.

The effective dimension of G and the off-origin energy of K distinguish a
white Walsh floor from a correlated/low-dimensional one.  Symmetric matrices
are stored in a Frobenius-isometric vectorization, so every reported norm and
inner product has its literal matrix meaning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_hessian_noise.json"


def fwht(value: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh--Hadamard transform along the first axis."""
    out = np.asarray(value, dtype=np.float64).copy()
    n = out.shape[0]
    if n < 1 or n & (n - 1):
        raise ValueError("the Walsh axis must have power-of-two length")
    width = 1
    while width < n:
        block = out.reshape(-1, 2 * width, *out.shape[1:])
        left = block[:, :width].copy()
        right = block[:, width:].copy()
        block[:, :width] = left + right
        block[:, width:] = left - right
        width *= 2
    return out


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.uint8) & 1
    n = value.shape[0]
    if value.shape != (n, n):
        raise ValueError("GF(2) inverse expects a square matrix")
    aug = np.concatenate((value.copy(), np.eye(n, dtype=np.uint8)), axis=1)
    for column in range(n):
        pivots = np.flatnonzero(aug[column:, column])
        if not pivots.size:
            raise ValueError("singular matrix over GF(2)")
        pivot = column + int(pivots[0])
        if pivot != column:
            aug[[column, pivot]] = aug[[pivot, column]]
        rows = np.flatnonzero(aug[:, column])
        rows = rows[rows != column]
        aug[rows] ^= aug[column]
    return aug[:, n:]


def random_gl2(n: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic-seed random-looking invertible binary map."""
    value = np.eye(n, dtype=np.uint8)
    for _ in range(8 * n):
        a, b = rng.choice(n, size=2, replace=False)
        if rng.random() < 0.25:
            value[[a, b]] = value[[b, a]]
        else:
            value[a] ^= value[b]
    _gf2_inverse(value)
    return value


def generic_basis(n: int, rng: np.random.Generator) -> np.ndarray:
    """A well-conditioned generic basis, avoiding the orthogonal control."""
    q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    singular = np.linspace(0.82, 1.18, n)
    rng.shuffle(singular)
    return q1 @ np.diag(singular) @ q2.T


def _integer_chunk(start: int, stop: int, n: int, cutoff: int) -> np.ndarray:
    base = 2 * cutoff + 1
    ids = np.arange(start, stop, dtype=np.int64)
    powers = base ** np.arange(n, dtype=np.int64)
    return ((ids[:, None] // powers[None, :]) % base - cutoff).astype(
        np.int16, copy=False
    )


def shortest_vector_coefficients(
    basis: np.ndarray,
    cutoff: int = 2,
    chunk_size: int = 200_000,
) -> tuple[float, np.ndarray]:
    """Exact shortest coefficient in the supplied finite coefficient cube."""
    n = basis.shape[0]
    count = (2 * cutoff + 1) ** n
    best_norm2 = math.inf
    best = None
    for start in range(0, count, chunk_size):
        coeff = _integer_chunk(start, min(start + chunk_size, count), n, cutoff)
        nonzero = np.any(coeff != 0, axis=1)
        coeff = coeff[nonzero]
        vectors = coeff @ basis.T
        norm2 = np.einsum("ij,ij->i", vectors, vectors)
        where = int(np.argmin(norm2))
        if norm2[where] < best_norm2:
            best_norm2 = float(norm2[where])
            best = coeff[where].astype(np.int64)
    if best is None:
        raise RuntimeError("shortest-vector search cube contained no vector")
    return math.sqrt(best_norm2), best


def _symmetric_layout(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row, column = np.triu_indices(n)
    scale = np.where(row == column, 1.0, math.sqrt(2.0))
    trace = np.zeros(row.size, dtype=np.float64)
    trace[row == column] = 1.0 / math.sqrt(n)
    return row, column, scale, trace


def _sym_outer(
    vectors: np.ndarray,
    row: np.ndarray,
    column: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return vectors[:, row] * vectors[:, column] * scale


def _sym_to_matrix(
    vector: np.ndarray,
    n: int,
    row: np.ndarray,
    column: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    result = np.zeros((n, n), dtype=np.float64)
    values = np.asarray(vector) / scale
    result[row, column] = values
    result[column, row] = values
    return result


def _binary_index(bits: np.ndarray) -> np.ndarray:
    if bits.shape[1] == 0:
        return np.zeros(bits.shape[0], dtype=np.int64)
    weights = (1 << np.arange(bits.shape[1], dtype=np.int64))
    return bits.astype(np.int64) @ weights


def _xor_gram(kernel: np.ndarray, means: np.ndarray) -> np.ndarray:
    n = kernel.size
    ids = np.arange(n, dtype=np.int64)
    gram = kernel[np.bitwise_xor(ids[:, None], ids[None, :])]
    gram -= means @ means.T
    return 0.5 * (gram + gram.T)


def _matrix_opnorms(
    means: np.ndarray,
    n: int,
    row: np.ndarray,
    column: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    result = np.empty(means.shape[0], dtype=np.float64)
    for index, vector in enumerate(means):
        matrix = _sym_to_matrix(vector, n, row, column, scale)
        result[index] = float(np.max(np.abs(np.linalg.eigvalsh(matrix))))
    return result


def _kernel_summary(kernel: np.ndarray, gram: np.ndarray) -> dict:
    scale = max(float(np.linalg.norm(gram)), 1e-300)
    diagonal = np.diag(np.diag(gram))
    eigenvalues = np.linalg.eigvalsh(gram)
    negative = np.minimum(eigenvalues, 0.0)
    positive = np.maximum(eigenvalues, 0.0)
    trace = float(np.sum(positive))
    square = float(np.sum(positive * positive))
    effective = trace * trace / max(square, 1e-300)
    kernel_energy = kernel * kernel
    order = np.argsort(kernel_energy)[::-1]
    cumulative = np.cumsum(kernel_energy[order])
    support90 = int(np.searchsorted(cumulative, 0.9 * cumulative[-1]) + 1)
    return {
        "covariance_effective_dimension": effective,
        "covariance_effective_fraction": effective / kernel.size,
        "covariance_off_diagonal_fraction": float(
            np.linalg.norm(gram - diagonal) / scale
        ),
        "covariance_negative_eigenvalue_fraction": float(
            -np.sum(negative) / max(trace, 1e-300)
        ),
        "kernel_off_origin_energy_fraction": float(
            np.sum(kernel_energy[1:]) / max(np.sum(kernel_energy), 1e-300)
        ),
        "kernel_support_90_percent": support90,
        "mean_coefficient_variance": float(np.mean(np.diag(gram))),
        "maximum_coefficient_variance": float(np.max(np.diag(gram))),
    }


def _variant_report(
    means: np.ndarray,
    second_by_y: np.ndarray,
    theta_star: int,
    signal_direction: np.ndarray,
    sample_exponent: float,
    n: int,
    row: np.ndarray,
    column: np.ndarray,
    scale: np.ndarray,
) -> dict:
    kernel = fwht(second_by_y)
    gram = _xor_gram(kernel, means)
    summary = _kernel_summary(kernel, gram)
    opnorm = _matrix_opnorms(means, n, row, column, scale)
    false = np.delete(opnorm, theta_star)
    signal = float(opnorm[theta_star])
    scalar = np.abs(means @ signal_direction)
    scalar_false = np.delete(scalar, theta_star)
    max_variance = max(summary["maximum_coefficient_variance"], 0.0)
    outputs = means.shape[0]
    max_noise_unit = math.sqrt(max_variance) * math.sqrt(
        2.0 * math.log(max(2 * outputs, 2))
    )
    m_exp = 2.0 ** (sample_exponent * n)
    m_paper = (n ** 6) * m_exp
    summary.update({
        "target_operator_norm": signal,
        "maximum_false_operator_norm": float(np.max(false)) if false.size else 0.0,
        "target_to_false_ratio": signal / max(float(np.max(false)), 1e-300)
        if false.size else math.inf,
        "target_direction_amplitude": float(scalar[theta_star]),
        "maximum_false_target_direction": float(np.max(scalar_false))
        if scalar_false.size else 0.0,
        "target_direction_to_false_ratio": float(scalar[theta_star]) /
        max(float(np.max(scalar_false)), 1e-300) if scalar_false.size else math.inf,
        "maximum_noise_proxy_one_sample": max_noise_unit,
        "target_to_noise_proxy_exponential_samples": signal /
        max(max_noise_unit / math.sqrt(m_exp), 1e-300),
        "target_to_noise_proxy_paper_polynomial_samples": signal /
        max(max_noise_unit / math.sqrt(m_paper), 1e-300),
    })
    return summary


def audit_dimension(
    n: int,
    *,
    cutoff: int = 3,
    seed: int = 260802478,
    basis_mode: str = "generic",
    r: float = 0.2222355,
    R: float = 0.400613,
    chi: float = 0.3961331,
    chunk_size: int = 100_000,
) -> dict:
    """Enumerate one generic lattice and return every affine-coset audit."""
    if n < 2:
        raise ValueError("dimension must be at least two")
    rng = np.random.default_rng(seed + 1009 * n)
    if basis_mode == "generic":
        basis = generic_basis(n, rng)
    elif basis_mode == "orthogonal":
        basis = np.eye(n, dtype=np.float64)
    else:
        raise ValueError(f"unknown basis mode {basis_mode!r}")
    condition = float(np.linalg.cond(basis))
    shortest, shortest_coeff = shortest_vector_coefficients(basis)
    xi_r = math.sqrt(4.0 * n * r * math.log(2.0) / (math.pi * shortest ** 2))
    xi_R = math.sqrt(4.0 * n * R * math.log(2.0) / (math.pi * shortest ** 2))
    iota = 0.5 * math.log2(R * R / (r * (2.0 * R - r)))
    sample_exponent = iota + 2.0 * r

    h = min(max(int(math.floor(chi * n)), 0), n - 1)
    ell = n - h
    outputs = 1 << ell
    cosets = 1 << h
    groups = 1 << n
    transform = random_gl2(n, rng)
    inverse = _gf2_inverse(transform)
    row, column, sym_scale, trace_direction = _symmetric_layout(n)
    width = row.size

    mass_R = np.zeros(groups)
    mass_r = np.zeros(groups)
    boundary_R = np.zeros(groups)
    first = np.zeros((groups, width))
    second_raw = np.zeros(groups)
    second_trace = np.zeros(groups)
    second_identity = np.zeros(groups)
    second_trace_ht = np.zeros(groups)
    second_trace_target = np.zeros(groups)

    inverse_basis = np.linalg.inv(basis)
    count = (2 * cutoff + 1) ** n
    pi_normalizer = (2.0 ** (iota * n)) * ((r / R) ** (0.5 * n))
    started = time.perf_counter()
    for start in range(0, count, chunk_size):
        coeff = _integer_chunk(start, min(start + chunk_size, count), n, cutoff)
        dual = coeff @ inverse_basis
        norm2 = np.einsum("ij,ij->i", dual, dual)
        rho_R = np.exp(-math.pi * norm2 / (xi_R * xi_R))
        rho_r = np.exp(-math.pi * norm2 / (xi_r * xi_r))
        weight = np.divide(rho_r, rho_R, out=np.zeros_like(rho_r), where=rho_R > 0)
        retain = np.minimum(1.0, weight / pi_normalizer)

        parity = (coeff & 1).astype(np.uint8)
        coordinates = (parity @ inverse) & 1
        j_index = _binary_index(coordinates[:, :h])
        y_index = _binary_index(coordinates[:, h:])
        group = j_index * outputs + y_index

        sym = _sym_outer(dual, row, column, sym_scale)
        radial = sym @ trace_direction
        identity = radial[:, None] * trace_direction[None, :]
        traceless = sym - identity
        raw_norm2 = np.einsum("ij,ij->i", sym, sym)
        identity_norm2 = radial * radial
        trace_norm2 = np.maximum(raw_norm2 - identity_norm2, 0.0)

        np.add.at(mass_R, group, rho_R)
        np.add.at(mass_r, group, rho_r)
        np.add.at(first, group, rho_r[:, None] * sym)
        np.add.at(second_raw, group, rho_R * weight * weight * raw_norm2)
        np.add.at(second_trace, group, rho_R * weight * weight * trace_norm2)
        np.add.at(second_identity, group,
                  rho_R * weight * weight * identity_norm2)
        np.add.at(second_trace_ht, group,
                  rho_R * weight * weight * trace_norm2 /
                  np.maximum(retain, 1e-300))
        np.add.at(second_trace_target, group, rho_r * trace_norm2)
        boundary = np.any(np.abs(coeff) == cutoff, axis=1)
        np.add.at(boundary_R, group, rho_R * boundary)

    u_star = shortest_coeff & 1
    transformed_u = (transform @ u_star.astype(np.uint8)) & 1
    alpha_star = transformed_u[:h]
    theta_star = int(_binary_index(transformed_u[h:][None, :])[0])

    physical_shortest = basis @ shortest_coeff
    unit = physical_shortest / np.linalg.norm(physical_shortest)
    spike_matrix = np.outer(unit, unit) - np.eye(n) / n
    signal_direction = _sym_outer(
        unit[None, :], row, column, sym_scale
    )[0] - trace_direction / math.sqrt(n)
    signal_direction /= np.linalg.norm(signal_direction)
    assert np.allclose(
        _sym_to_matrix(signal_direction, n, row, column, sym_scale),
        spike_matrix / np.linalg.norm(spike_matrix),
        atol=1e-12,
    )

    affine_reports = []
    for j in range(cosets):
        sl = slice(j * outputs, (j + 1) * outputs)
        normal_R = float(np.sum(mass_R[sl]))
        normal_r = float(np.sum(mass_r[sl]))
        if normal_R <= 0.0 or normal_r <= 0.0:
            continue
        raw_means = fwht(first[sl] / normal_R)
        target_means = fwht(first[sl] / normal_r)
        radial_coordinate = raw_means @ trace_direction
        identity_means = radial_coordinate[:, None] * trace_direction[None, :]
        trace_means = raw_means - identity_means
        target_radial = target_means @ trace_direction
        target_trace_means = target_means - (
            target_radial[:, None] * trace_direction[None, :]
        )
        j_bits = ((j >> np.arange(h)) & 1).astype(np.uint8)
        expected_sign = -1 if int(alpha_star @ j_bits) & 1 else 1

        variants = {
            "raw_importance": _variant_report(
                raw_means, second_raw[sl] / normal_R, theta_star,
                signal_direction, sample_exponent, n, row, column, sym_scale),
            "identity_importance": _variant_report(
                identity_means, second_identity[sl] / normal_R, theta_star,
                trace_direction, sample_exponent, n, row, column, sym_scale),
            "traceless_importance": _variant_report(
                trace_means, second_trace[sl] / normal_R, theta_star,
                signal_direction, sample_exponent, n, row, column, sym_scale),
            "traceless_horvitz_thompson": _variant_report(
                trace_means, second_trace_ht[sl] / normal_R, theta_star,
                signal_direction, sample_exponent, n, row, column, sym_scale),
            "traceless_target_direct": _variant_report(
                target_trace_means, second_trace_target[sl] / normal_r,
                theta_star, signal_direction, sample_exponent, n,
                row, column, sym_scale),
        }
        affine_reports.append({
            "j": j,
            "expected_target_sign": expected_sign,
            "source_mass": normal_R,
            "target_mass": normal_r,
            "mean_importance_weight": normal_r / normal_R,
            "truncation_boundary_mass_fraction": float(
                np.sum(boundary_R[sl]) / normal_R
            ),
            "variants": variants,
        })

    return {
        "dimension": n,
        "basis_mode": basis_mode,
        "cutoff": cutoff,
        "enumerated_dual_points": count,
        "elapsed_seconds": time.perf_counter() - started,
        "basis_condition": condition,
        "shortest_length_in_coefficient_cube": shortest,
        "shortest_coefficients": shortest_coeff.tolist(),
        "r": r,
        "R": R,
        "chi": chi,
        "iota": iota,
        "sample_exponent_iota_plus_2r": sample_exponent,
        "h": h,
        "ell": ell,
        "walsh_outputs": outputs,
        "affine_cosets": cosets,
        "theta_star": theta_star,
        "affine_reports": affine_reports,
    }


def _aggregate_dimension(report: dict) -> dict:
    result = {
        key: report[key] for key in (
            "dimension", "cutoff", "enumerated_dual_points",
            "elapsed_seconds", "basis_condition", "h", "ell",
            "walsh_outputs", "affine_cosets", "theta_star",
        )
    }
    boundary = [row["truncation_boundary_mass_fraction"]
                for row in report["affine_reports"]]
    result["maximum_boundary_mass_fraction"] = max(boundary)
    variants = report["affine_reports"][0]["variants"]
    result["variants"] = {}
    for name in variants:
        rows = [item["variants"][name] for item in report["affine_reports"]]
        result["variants"][name] = {
            key: float(np.median([row[key] for row in rows]))
            for key in (
                "covariance_effective_dimension",
                "covariance_effective_fraction",
                "covariance_off_diagonal_fraction",
                "kernel_off_origin_energy_fraction",
                "kernel_support_90_percent",
                "target_to_false_ratio",
                "target_direction_to_false_ratio",
                "target_to_noise_proxy_exponential_samples",
                "target_to_noise_proxy_paper_polynomial_samples",
            )
        }
    raw = result["variants"]["raw_importance"]
    trace = result["variants"]["traceless_importance"]
    ht = result["variants"]["traceless_horvitz_thompson"]
    direct = result["variants"]["traceless_target_direct"]
    result["comparisons"] = {
        "trace_effective_fraction_over_raw":
            trace["covariance_effective_fraction"] /
            max(raw["covariance_effective_fraction"], 1e-300),
        "ht_mean_variance_proxy_over_importance":
            ht["target_to_noise_proxy_exponential_samples"] and
            (trace["target_to_noise_proxy_exponential_samples"] /
             max(ht["target_to_noise_proxy_exponential_samples"], 1e-300)) ** 2,
        "importance_variance_proxy_over_direct":
            (direct["target_to_noise_proxy_exponential_samples"] /
             max(trace["target_to_noise_proxy_exponential_samples"], 1e-300)) ** 2,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(6, 7, 8))
    parser.add_argument("--cutoff", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument(
        "--basis-mode", choices=("generic", "orthogonal"), default="generic"
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reports = []
    summaries = []
    for n in args.dimensions:
        report = audit_dimension(
            n, cutoff=args.cutoff, seed=args.seed, basis_mode=args.basis_mode,
            chunk_size=args.chunk_size
        )
        summary = _aggregate_dimension(report)
        reports.append(report)
        summaries.append(summary)
        trace = summary["variants"]["traceless_importance"]
        ht = summary["variants"]["traceless_horvitz_thompson"]
        print(
            f"n={n}: N={summary['walsh_outputs']} "
            f"boundary={summary['maximum_boundary_mass_fraction']:.2e} "
            f"trace Neff/N={trace['covariance_effective_fraction']:.3f} "
            f"Koff={trace['kernel_off_origin_energy_fraction']:.3f} "
            f"HT Neff/N={ht['covariance_effective_fraction']:.3f}",
            flush=True,
        )

    payload = {
        "experiment": "midpoint_hessian_walsh_noise",
        "interpretation_guardrail": (
            "Finite truncated-cube measurements diagnose covariance structure; "
            "they are not asymptotic SVP complexity claims."
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
