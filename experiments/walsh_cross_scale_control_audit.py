#!/usr/bin/env python3
"""Shared-sample cross-scale control variates for midpoint-Hessian noise.

The importance estimators for several target widths r_k can be formed from
the same source sample X~D_{Lambda_j,xi_R}.  Their noise is strongly
correlated because every contribution is

    w_k(X) * traceless(XX^T),

with only the radial weight changing.  This experiment forms the exact
finite-cube cross-covariance, then finds the minimum integrated-variance
linear combination that preserves the paper's asymptotic shortest-vector
spike scale

    a_k proportional to (r_k/R)^(n/2) r_k^2 2^(-r_k n).

The resulting coefficients use only the known widths and a covariance that
can be estimated from the samples; they do not use the unknown shortest-vector
direction.  An oracle-direction result is also reported, explicitly marked as
diagnostic rather than implementable.
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
    _kernel_summary,
    _matrix_opnorms,
    _symmetric_layout,
    _sym_outer,
    _xor_gram,
    fwht,
    generic_basis,
    random_gl2,
    shortest_vector_coefficients,
)


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_cross_scale_control.json"


def _minimum_variance_coefficients(
    covariance: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """Minimize c^T C c subject to c.signal=1."""
    covariance = 0.5 * (covariance + covariance.T)
    scale = max(float(np.trace(covariance) / len(covariance)), 1e-300)
    inverse_signal = np.linalg.pinv(
        covariance + np.eye(len(covariance)) * scale * 1e-12,
        rcond=1e-12,
    ) @ signal
    denominator = float(signal @ inverse_signal)
    if denominator <= 0.0:
        raise RuntimeError("cross-scale covariance did not give a positive constraint")
    return inverse_signal / denominator


def audit_cross_scale(
    n: int,
    *,
    widths: tuple[float, ...] = (0.18, 0.19, 0.205, 0.2222355, 0.24),
    source_width: float = 0.400613,
    chi: float = 0.3961331,
    cutoff: int = 3,
    seed: int = 260802478,
    chunk_size: int = 100_000,
) -> dict:
    widths = tuple(float(value) for value in widths)
    if any(not 0.0 < value < source_width for value in widths):
        raise ValueError("target widths must lie strictly between zero and R")
    rng = np.random.default_rng(seed + 1009 * n)
    basis = generic_basis(n, rng)
    shortest, shortest_coeff = shortest_vector_coefficients(basis)
    h = min(max(int(math.floor(chi * n)), 0), n - 1)
    ell = n - h
    outputs = 1 << ell
    cosets = 1 << h
    groups = 1 << n
    transform = random_gl2(n, rng)
    inverse = _gf2_inverse(transform)
    row, column, sym_scale, trace_direction = _symmetric_layout(n)
    matrix_width = row.size
    targets = len(widths)

    xi_R = math.sqrt(
        4.0 * n * source_width * math.log(2.0) /
        (math.pi * shortest * shortest)
    )
    xi = np.sqrt(
        4.0 * n * np.asarray(widths) * math.log(2.0) /
        (math.pi * shortest * shortest)
    )
    mass_R = np.zeros(groups)
    boundary_R = np.zeros(groups)
    first = np.zeros((targets, groups, matrix_width))
    cross_second = np.zeros((targets, targets, groups))

    inverse_basis = np.linalg.inv(basis)
    count = (2 * cutoff + 1) ** n
    started = time.perf_counter()
    for start in range(0, count, chunk_size):
        coeff = _integer_chunk(start, min(start + chunk_size, count), n, cutoff)
        dual = coeff @ inverse_basis
        norm2 = np.einsum("ij,ij->i", dual, dual)
        rho_R = np.exp(-math.pi * norm2 / (xi_R * xi_R))
        rho = np.exp(-math.pi * norm2[:, None] / (xi[None, :] ** 2))
        weight = np.divide(
            rho, rho_R[:, None], out=np.zeros_like(rho),
            where=rho_R[:, None] > 0,
        )

        parity = (coeff & 1).astype(np.uint8)
        coordinates = (parity @ inverse) & 1
        j_index = _binary_index(coordinates[:, :h])
        y_index = _binary_index(coordinates[:, h:])
        group = j_index * outputs + y_index

        sym = _sym_outer(dual, row, column, sym_scale)
        radial = sym @ trace_direction
        traceless = sym - radial[:, None] * trace_direction[None, :]
        trace_norm2 = np.einsum("ij,ij->i", traceless, traceless)

        np.add.at(mass_R, group, rho_R)
        for k in range(targets):
            np.add.at(first[k], group, rho[:, k, None] * traceless)
            for m in range(k + 1):
                value = rho_R * weight[:, k] * weight[:, m] * trace_norm2
                np.add.at(cross_second[k, m], group, value)
                if m != k:
                    cross_second[m, k] = cross_second[k, m]
        boundary = np.any(np.abs(coeff) == cutoff, axis=1)
        np.add.at(boundary_R, group, rho_R * boundary)

    u_star = shortest_coeff & 1
    transformed_u = (transform @ u_star.astype(np.uint8)) & 1
    theta_star = int(_binary_index(transformed_u[h:][None, :])[0])
    physical_shortest = basis @ shortest_coeff
    unit = physical_shortest / np.linalg.norm(physical_shortest)
    signal_direction = _sym_outer(
        unit[None, :], row, column, sym_scale
    )[0] - trace_direction / math.sqrt(n)
    signal_direction /= np.linalg.norm(signal_direction)

    log_proxy = (
        0.5 * n * np.log(np.asarray(widths) / source_width)
        + 2.0 * np.log(np.asarray(widths))
        - np.asarray(widths) * n * math.log(2.0)
    )
    proxy = np.exp(log_proxy - np.max(log_proxy))
    baseline = int(np.argmin(np.abs(np.asarray(widths) - 0.2222355)))
    affine_reports = []
    for j in range(cosets):
        sl = slice(j * outputs, (j + 1) * outputs)
        normal = float(np.sum(mass_R[sl]))
        means = np.stack([fwht(first[k, sl] / normal) for k in range(targets)])
        grams = np.empty((targets, targets, outputs, outputs))
        integrated = np.empty((targets, targets))
        for k in range(targets):
            for m in range(targets):
                kernel = fwht(cross_second[k, m, sl] / normal)
                gram = kernel[
                    np.bitwise_xor(
                        np.arange(outputs)[:, None], np.arange(outputs)[None, :]
                    )
                ] - means[k] @ means[m].T
                gram = 0.5 * (gram + gram.T)
                grams[k, m] = gram
                integrated[k, m] = float(np.trace(gram) / outputs)

        coefficients = _minimum_variance_coefficients(integrated, proxy)
        baseline_coefficients = np.zeros(targets)
        baseline_coefficients[baseline] = 1.0 / proxy[baseline]
        oracle_signal = np.abs(means[:, theta_star] @ signal_direction)
        oracle_signal /= max(float(np.max(oracle_signal)), 1e-300)
        oracle_coefficients = _minimum_variance_coefficients(
            integrated, oracle_signal
        )

        def combination_report(c: np.ndarray) -> dict:
            combined_means = np.einsum("k,kyd->yd", c, means)
            combined_gram = np.einsum("k,m,kmab->ab", c, c, grams)
            # Recover the xor kernel before the mean outer-product subtraction.
            ids = np.arange(outputs)
            combined_kernel = (
                combined_gram[0, ids] +
                combined_means[0] @ combined_means.T
            )
            summary = _kernel_summary(combined_kernel, combined_gram)
            opnorm = _matrix_opnorms(
                combined_means, n, row, column, sym_scale
            )
            scalar = np.abs(combined_means @ signal_direction)
            false_op = np.delete(opnorm, theta_star)
            false_scalar = np.delete(scalar, theta_star)
            summary.update({
                "coefficients": c.tolist(),
                "coefficient_l1": float(np.sum(np.abs(c))),
                "integrated_variance": float(c @ integrated @ c),
                "target_operator_norm": float(opnorm[theta_star]),
                "target_to_false_operator_ratio": float(opnorm[theta_star]) /
                    max(float(np.max(false_op)), 1e-300),
                "target_direction_amplitude": float(scalar[theta_star]),
                "target_direction_to_false_ratio": float(scalar[theta_star]) /
                    max(float(np.max(false_scalar)), 1e-300),
            })
            return summary

        base_report = combination_report(baseline_coefficients)
        proxy_report = combination_report(coefficients)
        oracle_report = combination_report(oracle_coefficients)
        reduction = (
            base_report["integrated_variance"] /
            max(proxy_report["integrated_variance"], 1e-300)
        )
        oracle_reduction = (
            base_report["integrated_variance"] /
            max(oracle_report["integrated_variance"], 1e-300)
        )
        correlation = integrated / np.sqrt(
            np.maximum(np.diag(integrated)[:, None] *
                       np.diag(integrated)[None, :], 1e-300)
        )
        affine_reports.append({
            "j": j,
            "truncation_boundary_mass_fraction": float(
                np.sum(boundary_R[sl]) / normal
            ),
            "integrated_noise_correlation": correlation.tolist(),
            "baseline": base_report,
            "proxy_preserving_control": proxy_report,
            "oracle_direction_control_diagnostic_only": oracle_report,
            "variance_reduction": reduction,
            "finite_n_equivalent_exponent_gain": math.log2(reduction) / n,
            "oracle_variance_reduction": oracle_reduction,
            "oracle_finite_n_equivalent_exponent_gain":
                math.log2(oracle_reduction) / n,
        })

    return {
        "dimension": n,
        "cutoff": cutoff,
        "enumerated_dual_points": count,
        "elapsed_seconds": time.perf_counter() - started,
        "widths": list(widths),
        "source_width": source_width,
        "signal_scale_proxy": proxy.tolist(),
        "baseline_width_index": baseline,
        "h": h,
        "ell": ell,
        "walsh_outputs": outputs,
        "theta_star": theta_star,
        "affine_reports": affine_reports,
    }


def summarize(report: dict) -> dict:
    rows = report["affine_reports"]
    return {
        "dimension": report["dimension"],
        "walsh_outputs": report["walsh_outputs"],
        "maximum_boundary_mass_fraction": max(
            row["truncation_boundary_mass_fraction"] for row in rows
        ),
        "median_variance_reduction": float(np.median([
            row["variance_reduction"] for row in rows
        ])),
        "median_finite_n_equivalent_exponent_gain": float(np.median([
            row["finite_n_equivalent_exponent_gain"] for row in rows
        ])),
        "median_oracle_variance_reduction": float(np.median([
            row["oracle_variance_reduction"] for row in rows
        ])),
        "median_control_l1": float(np.median([
            row["proxy_preserving_control"]["coefficient_l1"] for row in rows
        ])),
        "median_target_direction_to_false_ratio": float(np.median([
            row["proxy_preserving_control"]["target_direction_to_false_ratio"]
            for row in rows
        ])),
        "median_effective_fraction": float(np.median([
            row["proxy_preserving_control"]["covariance_effective_fraction"]
            for row in rows
        ])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(6, 7, 8))
    parser.add_argument(
        "--widths", type=float, nargs="+",
        default=(0.18, 0.19, 0.205, 0.2222355, 0.24),
    )
    parser.add_argument("--source-width", type=float, default=0.400613)
    parser.add_argument("--cutoff", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reports = []
    summaries = []
    for n in args.dimensions:
        report = audit_cross_scale(
            n, widths=tuple(args.widths), source_width=args.source_width,
            cutoff=args.cutoff, seed=args.seed, chunk_size=args.chunk_size,
        )
        summary = summarize(report)
        reports.append(report)
        summaries.append(summary)
        print(
            f"n={n}: reduction={summary['median_variance_reduction']:.3f}x "
            f"finite-n gain={summary['median_finite_n_equivalent_exponent_gain']:.4f} "
            f"l1={summary['median_control_l1']:.2f}",
            flush=True,
        )
    payload = {
        "experiment": "midpoint_hessian_cross_scale_control",
        "guardrail": (
            "The exponent gain is log2 of a finite-n variance ratio divided "
            "by n; it is a diagnostic, not an asymptotic claim."
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
