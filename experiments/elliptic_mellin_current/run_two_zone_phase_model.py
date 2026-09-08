"""Test a coherent negative-prime-chamber / neutral-tail phase model."""

from __future__ import annotations

import json
from math import exp, log, pi, sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from run_coherent_phase_model import MAXIMUM_LIMIT, SPF
from run_relaxed_cone import moment_kernel


def phase_coefficients(inner_cosine: float, cutoff_prime: int, limit: int) -> np.ndarray:
    # Cache U_k(c_p) separately for the negative chamber and neutral tail.
    maximum_exponent = int(log(limit, 2)) + 1
    tables = {}
    for label, cosine in (("inner", inner_cosine), ("outer", 0.0)):
        values = np.empty(maximum_exponent + 1, dtype=np.float64)
        values[0] = 1.0
        values[1] = 2.0 * cosine
        for exponent in range(2, maximum_exponent + 1):
            values[exponent] = (
                2.0 * cosine * values[exponent - 1] - values[exponent - 2]
            )
        tables[label] = values

    normalized = np.ones(limit + 1, dtype=np.float64)
    for n in range(2, limit + 1):
        prime = SPF[n]
        remainder = n
        exponent = 0
        while remainder % prime == 0:
            remainder //= prime
            exponent += 1
        table = tables["inner" if prime <= cutoff_prime else "outer"]
        normalized[n] = table[exponent] * normalized[remainder]
    indices = np.arange(limit + 1, dtype=np.float64)
    return np.sqrt(indices) * normalized


def residual(parameters: np.ndarray, cutoff_prime: int) -> np.ndarray:
    conductor = exp(float(parameters[0]))
    inner_cosine = float(parameters[1])
    alpha = 2.0 * pi / sqrt(conductor)
    limit = min(MAXIMUM_LIMIT, max(32, int(42.0 / alpha)))
    indices = np.arange(1, limit + 1, dtype=np.float64)
    coefficients = phase_coefficients(inner_cosine, cutoff_prime, limit)[1:]
    arguments = alpha * indices
    w1 = moment_kernel(arguments, 1)
    w3 = moment_kernel(arguments, 3)
    return np.asarray(
        (
            np.dot(coefficients, w1) / w1[0],
            np.dot(coefficients, w3) / w3[0],
        )
    )


def main() -> None:
    solutions = []
    for cutoff in (
        47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
        107, 109, 113, 127, 149, 173, 197, 233,
    ):
        for starting_conductor in (3_000_000, 20_000_000):
            fit = least_squares(
                lambda parameters: residual(parameters, cutoff),
                np.asarray((log(starting_conductor), -0.75)),
                bounds=((log(100.0), -1.0), (log(100_000_000.0), 0.0)),
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
                max_nfev=120,
            )
            solutions.append(
                {
                    "cutoff_prime": cutoff,
                    "start_conductor": starting_conductor,
                    "conductor": exp(float(fit.x[0])),
                    "inner_cosine": float(fit.x[1]),
                    "residual": residual(fit.x, cutoff).tolist(),
                    "cost": float(fit.cost),
                    "success": bool(fit.success),
                }
            )
    target = Path("/tmp/elliptic_two_zone_phase.json")
    target.write_text(json.dumps(solutions, indent=2))
    for solution in solutions:
        if solution["cost"] < 1e-12:
            print("ROOT", solution)
    print("best")
    for solution in sorted(solutions, key=lambda item: item["cost"])[:10]:
        print(solution)
    print("wrote", target)


if __name__ == "__main__":
    main()
