#!/usr/bin/env python3
"""Continuous radial matched-filter bound for the Walsh Hessian spike.

This is a proof-target audit, not a lattice algorithm.  It removes the
endpoint importance family and asks for the best *rotationally equivariant*
traceless matrix contribution under the wide continuous Gaussian source.

For a source sample X=rho*omega and a unit target direction u, put

    H(X) = X X^T - ||X||^2 I/n,
    c_n(z) = E_omega[((u.omega)^2 - 1/n) cos(z u.omega)].

Among all scalar radial multipliers a(rho), Cauchy--Schwarz shows that the
best signal-to-Frobenius-noise ratio has exponential part

    E_rho[c_n(pi*d*rho)^2].

The minimizing multiplier is proportional to c_n(pi*d*rho)/rho^2.  We
evaluate the expectation by generalized Gauss--Laguerre quadrature in rho^2
and Gauss--Jacobi quadrature on the sphere.  Polynomial factors such as
(1-1/n)^-3 are reported separately and do not affect the exponent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_radial_matched_filter.json"


def angular_traceless_character(
    n: int,
    z: np.ndarray,
    *,
    order: int = 384,
) -> np.ndarray:
    """Return c_n(z) by normalized Gauss--Jacobi quadrature."""
    from scipy.special import roots_jacobi

    if n < 2:
        raise ValueError("dimension must be at least two")
    nodes, weights = roots_jacobi(order, (n - 3.0) / 2.0, (n - 3.0) / 2.0)
    weights = weights / np.sum(weights)
    centered_square = nodes * nodes - 1.0 / n
    result = np.empty_like(z, dtype=np.float64)
    for start in range(0, len(z), 64):
        stop = min(start + 64, len(z))
        result[start:stop] = (
            np.cos(z[start:stop, None] * nodes[None, :])
            * centered_square[None, :]
            * weights[None, :]
        ).sum(axis=1)
    return result


def matched_filter_report(
    n: int,
    source_width: float,
    *,
    angular_order: int = 384,
    radial_order: int = 160,
) -> dict[str, float | int]:
    """Evaluate the optimal continuous radial Hessian SNR at one dimension."""
    from scipy.special import gammaln, ive, roots_genlaguerre

    # If p_R(x) is proportional to exp(-pi ||x||^2/xi_R^2), then
    # U=||X||^2/xi_R^2 has Gamma(n/2, rate=pi).  With y=pi*U,
    # y is Gamma(n/2, rate=1), exactly the generalized-Laguerre measure.
    y, radial_weight = roots_genlaguerre(radial_order, n / 2.0 - 1.0)
    radial_weight = radial_weight / np.sum(radial_weight)
    # z=pi*d*||X|| and xi_R^2=4*n*R*ln(2)/(pi*d^2).
    z = np.sqrt(4.0 * n * source_width * math.log(2.0) * y)
    character = angular_traceless_character(
        n,
        z,
        order=angular_order,
    )
    exponential_snr2 = float(np.sum(radial_weight * character * character))
    # The target extreme eigenvalue is E[a(rho) rho^2 c_n], while
    # E||a(rho)H(X)||_F^2=(1-1/n)E[a(rho)^2 rho^4].
    full_snr2 = exponential_snr2 / (1.0 - 1.0 / n)
    nu = n / 2.0 - 1.0
    x = 2.0 * n * source_width * math.log(2.0)
    # Funk--Hecke gives
    # c_n(z)=-A_n z^-nu J_(nu+2)(z), where
    # A_n=2^(nu-1) Gamma(nu+1)(2nu+1)/(nu+1).  Weber's integral
    # then evaluates the radial second moment exactly.
    log_A = (
        (nu - 1.0) * math.log(2.0)
        + float(gammaln(nu + 1.0))
        + math.log(2.0 * nu + 1.0)
        - math.log(nu + 1.0)
    )
    scaled_bessel = float(ive(nu + 2.0, x))
    log_exact_snr2 = (
        2.0 * log_A
        - float(gammaln(nu + 1.0))
        - nu * math.log(2.0 * x)
        + math.log(scaled_bessel)
    )
    exact_snr2 = math.exp(log_exact_snr2)
    asymptotic = asymptotic_radial_exponent(source_width)
    importance_r = optimal_gaussian_target_width(source_width)
    importance = gaussian_importance_exponent(importance_r, source_width)
    return {
        "dimension": n,
        "source_width": source_width,
        "angular_order": angular_order,
        "radial_order": radial_order,
        "radial_character_second_moment": exponential_snr2,
        "closed_form_radial_character_second_moment": exact_snr2,
        "quadrature_to_closed_form_relative_error": abs(
            exponential_snr2 / exact_snr2 - 1.0
        ),
        "full_matched_snr_squared": full_snr2,
        "sample_exponent_exponential_part": -math.log2(exponential_snr2) / n,
        "closed_form_sample_exponent": -log_exact_snr2 / (n * math.log(2.0)),
        "sample_exponent_including_polynomial_factor": -math.log2(full_snr2) / n,
        "asymptotic_sample_exponent": asymptotic,
        "optimal_gaussian_target_width": importance_r,
        "optimal_gaussian_importance_exponent": importance,
        "asymptotic_minus_gaussian_optimum": asymptotic - importance,
        "maximum_quadrature_character": float(np.max(np.abs(character))),
    }


def gaussian_importance_exponent(r: float, R: float) -> float:
    return 2.0 * r + 0.5 * math.log2(R * R / (r * (2.0 * R - r)))


def optimal_gaussian_target_width(R: float) -> float:
    """Return the unique minimizer of iota(r,R)+2r on 0<r<R."""
    a = 2.0 * R * math.log(2.0)
    ratio = ((2.0 * a + 1.0) - math.sqrt(4.0 * a * a + 1.0)) / (2.0 * a)
    return R * ratio


def asymptotic_radial_exponent(R: float) -> float:
    """Large-n exponent of the exact radial matched-filter SNR."""
    c = 4.0 * R * math.log(2.0)
    root = math.sqrt(1.0 + c * c)
    eta = root + math.log(c / (1.0 + root))
    return (
        1.0 + math.log(c) - math.log(2.0) - eta + c
    ) / (2.0 * math.log(2.0))


def audit(
    dimensions: tuple[int, ...],
    source_widths: tuple[float, ...],
    *,
    angular_order: int,
    radial_order: int,
) -> dict[str, object]:
    rows = [
        matched_filter_report(
            n,
            width,
            angular_order=angular_order,
            radial_order=radial_order,
        )
        for width in source_widths
        for n in dimensions
    ]
    return {
        "experiment": "continuous_radial_matched_walsh_hessian",
        "interpretation": (
            "A limiting sample exponent below 1/2 would justify pursuing a "
            "Laguerre/Bessel lattice kernel.  An exponent above 1/2 rules out "
            "the unrestricted continuous radial matched filter at that source "
            "width, hence also every scalar radial Gaussian-mixture control."
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=(16, 24, 32, 48, 64, 96, 128, 160),
    )
    parser.add_argument(
        "--source-widths",
        type=float,
        nargs="+",
        default=(0.400613, 0.46294041585),
    )
    parser.add_argument("--angular-order", type=int, default=384)
    parser.add_argument("--radial-order", type=int, default=160)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(
        tuple(args.dimensions),
        tuple(args.source_widths),
        angular_order=args.angular_order,
        radial_order=args.radial_order,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
