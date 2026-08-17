#!/usr/bin/env python3
"""Transfer the improved spherical-code bound into Hhan's SVP exponent.

The lattice paper uses the cap-optimized Kabatianskii--Levenshtein exponent
twice: in the global shell constant beta and in the per-parity-coset counting
function K_2.  Chapter 2 of ``Ten Advances`` supplies a strictly smaller
cap-optimized exponent.  This script implements the simplest member of that
hierarchy (the moving one-row stabilizer representation), substitutes it in
g_2(R), and reoptimizes the classical importance-sampling parameters.

SciPy is used only for numerical optimization.  Run the full audit on the M4
Mini as documented by the repository AGENTS.md.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar


CLASSICAL_T0 = 0.23147
SQRT2 = math.sqrt(2.0)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "out" / "walsh_spherical_bound_transfer.json"
)


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0 if p in (0.0, 1.0) else math.inf
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def hsph(u: float) -> float:
    if u <= 0.0:
        return 0.0 if u == 0.0 else math.inf
    return (1.0 + u) * math.log2(1.0 + u) - u * math.log2(u)


def a0(s: float) -> float:
    if s <= 0.0:
        return 0.0
    return 0.5 * (1.0 / math.sqrt(1.0 - s * s) - 1.0)


def direct_kl(s: float) -> float:
    return hsph(a0(s))


def kl_closed(s: float) -> float:
    """Classical cap-optimized KL exponent at maximum inner product s."""
    if s <= 0.0:
        return 0.0
    y = math.sqrt(1.0 - s * s)
    p = (1.0 + y) / (2.0 * y)
    q = (1.0 - y) / (2.0 * y)
    first = p * math.log2(p)
    second = 0.0 if q == 0.0 else q * math.log2(q)
    return first - second


def cap_optimize(s: float, direct_bound) -> tuple[float, float]:
    if s <= 0.0:
        return 0.0, 0.0

    def objective(t: float) -> float:
        return direct_bound(t) + 0.5 * math.log2((1.0 - t) / (1.0 - s))

    result = minimize_scalar(
        objective,
        bounds=(0.0, s),
        method="bounded",
        options={"xatol": 2e-12},
    )
    candidates = [(objective(0.0), 0.0), (objective(s), s), (result.fun, result.x)]
    return min(candidates)


def gamma_row(a: float, b: float) -> float:
    if not (0.0 <= b < a):
        return 0.0
    return (a - b) * (1.0 + a + b) / (
        (1.0 + 2.0 * a) * math.sqrt(a * (1.0 + a))
    )


def row_root_a(s: float, b: float) -> float:
    """Smallest a>b satisfying 2 Gamma_row(a,b)=s."""
    if s <= 0.0:
        return b
    lo = max(b * (1.0 + 1e-14), b + 1e-15)
    hi = max(1.0, 2.0 * b + 1.0)
    while 2.0 * gamma_row(hi, b) < s:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("failed to bracket one-row spectral boundary")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 2.0 * gamma_row(mid, b) < s:
            lo = mid
        else:
            hi = mid
    return hi


@functools.lru_cache(maxsize=8192)
def direct_row_cached(s_key: float) -> tuple[float, float, float]:
    s = float(s_key)
    if s <= 0.0:
        return 0.0, 0.0, 0.0

    def objective(log1pb: float) -> float:
        b = math.expm1(log1pb)
        a = row_root_a(s, b)
        return hsph(a) - hsph(b)

    # b=0 is the classical boundary.  log(1+b)<=log(101) is ample for
    # fixed angles in this audit; the upper endpoint is checked explicitly.
    result = minimize_scalar(
        objective,
        bounds=(0.0, math.log(101.0)),
        method="bounded",
        options={"xatol": 2e-11},
    )
    candidates = [
        (objective(0.0), 0.0),
        (result.fun, result.x),
        (objective(math.log(101.0)), math.log(101.0)),
    ]
    value, log1pb = min(candidates)
    b = math.expm1(log1pb)
    return value, row_root_a(s, b), b


def direct_row(s: float) -> float:
    # Quantization makes nested g_2 optimization reproducible and its error
    # is far below the displayed exponent precision.
    return direct_row_cached(round(float(s), 12))[0]


@functools.lru_cache(maxsize=8192)
def cap_row_cached(s_key: float) -> tuple[float, float]:
    return cap_optimize(float(s_key), direct_row)


def cap_row(s: float) -> float:
    return cap_row_cached(round(float(s), 10))[0]


def make_row_interpolants(points: int = 401):
    """Precompute the expensive nested variational bound for optimization."""
    grid = np.linspace(0.0, 0.995, points)
    direct_values = np.array([direct_row(float(s)) for s in grid])
    cap_values = np.array([cap_row(float(s)) for s in grid])

    def interpolate(values, s: float) -> float:
        value = float(s)
        if value <= 0.0:
            return 0.0
        if value >= grid[-1]:
            return float(values[-1])
        return float(np.interp(value, grid, values))

    return (
        lambda s: interpolate(direct_values, s),
        lambda s: interpolate(cap_values, s),
        grid,
        direct_values,
        cap_values,
    )


def cap_kl(s: float) -> float:
    return cap_optimize(float(s), direct_kl)[0]


def k2(x: float, spherical_bound) -> float:
    if x <= SQRT2:
        return 0.0
    return spherical_bound(1.0 - 2.0 / (x * x))


def threshold_from_bound(spherical_bound) -> float:
    """Gaussian shell threshold induced by the bound at angle 1/2."""
    log2_beta = spherical_bound(0.5)
    return 2.0 ** (2.0 * log2_beta) / (4.0 * math.e * math.log(2.0))


def g(width: float, threshold: float) -> float:
    return 0.5 * math.log2(width / threshold)


def iota(r: float, R: float) -> float:
    return 0.5 * math.log2(R * R / (r * (2.0 * R - r)))


def g2(width: float, spherical_bound) -> tuple[float, float, float]:
    beta_log = spherical_bound(0.5)

    def exponent_at_x(x: float) -> float:
        count = min(beta_log + math.log2(x), 1.0 + k2(x, spherical_bound))
        return count - 0.5 * width * x * x

    # The tail is dominated by -R x^2/2.  x<=12 is vastly beyond the
    # relevant maximizer for all widths considered here.
    result = differential_evolution(
        lambda z: -exponent_at_x(float(z[0])),
        bounds=[(1.0, 12.0)],
        tol=2e-9,
        polish=True,
        seed=7,
        workers=1,
    )
    candidates = [(exponent_at_x(1.0), 1.0), (-result.fun, float(result.x[0]))]
    sup_value, x_star = max(candidates)
    return 1.0 - sup_value, x_star, beta_log


def final_exponent(params: tuple[float, float], spherical_bound) -> tuple[float, dict]:
    r, R = map(float, params)
    threshold = threshold_from_bound(spherical_bound)
    if not (0.18 <= r < 0.25 and threshold < R and r < R):
        return 10.0, {}
    s_aux = r * R / (2.0 * R - r)
    if s_aux <= threshold / 2.0:
        return 10.0, {}
    g2_R, x_star, beta_log = g2(R, spherical_bound)
    gr = g(r, threshold)
    chi_max = min(
        g2_R,
        0.5 + gr,
        1.0 + min(gr, 2.0 * gr) - 2.0 * r,
        0.5,
    )
    if chi_max <= 0.0:
        return 10.0, {}
    sample = 2.0 * r + iota(r, R)
    transform = 1.0 - chi_max
    exponent = max(0.5, sample, transform)
    return exponent, {
        "r": r,
        "R": R,
        "chi": chi_max,
        "sample_exponent": sample,
        "transform_exponent": transform,
        "g2_R": g2_R,
        "g_r": gr,
        "auxiliary_width": s_aux,
        "g2_shell_x": x_star,
        "log2_beta": beta_log,
        "gaussian_threshold": threshold,
    }


def optimize_final(spherical_bound, seed: int) -> dict:
    threshold = threshold_from_bound(spherical_bound)
    result = differential_evolution(
        lambda z: final_exponent((z[0], z[1]), spherical_bound)[0],
        bounds=[(0.18, 0.2499), (threshold + 1e-7, 0.7)],
        tol=2e-7,
        popsize=14,
        maxiter=90,
        polish=True,
        seed=seed,
        workers=1,
    )
    exponent, detail = final_exponent((result.x[0], result.x[1]), spherical_bound)
    detail["overall_exponent"] = exponent
    return detail


def build_report() -> dict:
    kl_half_cap, kl_half_t = cap_optimize(0.5, direct_kl)
    row_half_cap, row_half_t = cap_row_cached(0.5)
    row_half_direct, row_half_a, row_half_b = direct_row_cached(0.5)

    (
        direct_row_interpolant,
        cap_row_interpolant,
        row_grid,
        direct_row_values,
        cap_row_values,
    ) = make_row_interpolants()
    baseline = optimize_final(kl_closed, seed=11)
    classical_cap = optimize_final(cap_kl, seed=12)
    improved_direct = optimize_final(direct_row_interpolant, seed=13)
    improved = optimize_final(cap_row_interpolant, seed=14)
    direct_exact_exponent, direct_exact_detail = final_exponent(
        (improved_direct["r"], improved_direct["R"]), direct_row
    )
    direct_exact_detail["overall_exponent"] = direct_exact_exponent
    improved_direct["exact_nested_bound_check"] = direct_exact_detail
    # Re-evaluate the winning point with the un-interpolated nested bound.
    exact_exponent, exact_detail = final_exponent(
        (improved["r"], improved["R"]), cap_row
    )
    exact_detail["overall_exponent"] = exact_exponent
    improved["exact_nested_bound_check"] = exact_detail
    return {
        "experiment": "walsh_spherical_bound_transfer",
        "bound_checks": {
            "kl_closed_at_s_half": kl_closed(0.5),
            "kl_cap_optimization_at_s_half": kl_half_cap,
            "kl_cap_slice_t_at_s_half": kl_half_t,
            "row_direct_at_s_half": row_half_direct,
            "row_direct_a_at_s_half": row_half_a,
            "row_direct_b_at_s_half": row_half_b,
            "row_cap_at_s_half": row_half_cap,
            "row_cap_slice_t_at_s_half": row_half_t,
            "row_improvement_at_s_half": kl_closed(0.5) - row_half_cap,
        },
        "baseline_KL": baseline,
        "classical_cap_optimized_KL": classical_cap,
        "moving_one_row_direct": improved_direct,
        "moving_one_row": improved,
        "overall_exponent_improvement": (
            baseline["overall_exponent"] - exact_exponent
        ),
        "row_interpolation": {
            "points": len(row_grid),
            "maximum_s": float(row_grid[-1]),
            "minimum_direct_value": float(direct_row_values.min()),
            "minimum_cap_value": float(cap_row_values.min()),
        },
        "interpretation": (
            "This is a rigorous transfer conditional only on using the cited "
            "asymptotic spherical-code theorem. It implements the one-row "
            "subfamily, not the stronger full hierarchy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
