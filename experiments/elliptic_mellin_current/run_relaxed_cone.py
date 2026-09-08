"""Test the universal Hasse-envelope moment cone as a conductor obstruction.

This intentionally relaxes away integrality, multiplicativity, root number,
and realization by a curve.  Failure is therefore a valid universal
obstruction; success identifies how much arithmetic structure the next cone
must restore.
"""

from __future__ import annotations

import json
from math import pi, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import linprog


LAGUERRE_NODES, LAGUERRE_WEIGHTS = laggauss(64)


def moment_kernel(arguments: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate ``W_k(a)=2 int_1^inf log(x)^k exp(-a*x) dx``."""

    arguments = np.asarray(arguments, dtype=np.float64)
    scaled = LAGUERRE_NODES[:, None] / arguments[None, :]
    return (
        2.0
        * np.exp(-arguments)
        / arguments
        * np.sum(
            LAGUERRE_WEIGHTS[:, None] * np.log1p(scaled) ** degree,
            axis=0,
        )
    )


def divisor_counts(limit: int) -> np.ndarray:
    counts = np.zeros(limit + 1, dtype=np.int64)
    for divisor in range(1, limit + 1):
        counts[divisor::divisor] += 1
    return counts


def relaxed_feasibility(conductor: int, cutoff_exponent: float = 38.0) -> dict[str, object]:
    alpha = 2.0 * pi / sqrt(conductor)
    limit = max(16, int(cutoff_exponent / alpha))
    indices = np.arange(1, limit + 1, dtype=np.float64)
    arguments = alpha * indices
    w1 = moment_kernel(arguments, 1)
    w3 = moment_kernel(arguments, 3)
    bounds = divisor_counts(limit)[2:] * np.sqrt(indices[1:])
    matrix = np.vstack((w1[1:], w3[1:]))
    result = linprog(
        np.zeros(limit - 1),
        A_eq=matrix,
        b_eq=-np.asarray((w1[0], w3[0])),
        bounds=list(zip(-bounds, bounds)),
        method="highs",
    )
    report: dict[str, object] = {
        "conductor": conductor,
        "alpha": alpha,
        "coefficient_limit": limit,
        "feasible": bool(result.success),
        "solver_status": result.message,
    }
    if result.success:
        relative = np.abs(result.x) / bounds
        report["maximum_bound_fraction"] = float(np.max(relative))
        report["active_coefficients"] = [
            {
                "n": int(index + 2),
                "value": float(result.x[index]),
                "bound": float(bounds[index]),
            }
            for index in np.argsort(relative)[-12:][::-1]
            if relative[index] > 1e-10
        ]
        report["moment_residual"] = (
            matrix @ result.x + np.asarray((w1[0], w3[0]))
        ).tolist()
    return report


def main() -> None:
    conductors = (
        10,
        20,
        37,
        100,
        300,
        1_000,
        3_000,
        10_000,
        30_000,
        100_000,
        300_000,
        1_000_000,
        3_000_000,
        10_000_000,
        19_047_851,
    )
    reports = [relaxed_feasibility(conductor) for conductor in conductors]
    target = Path("/tmp/elliptic_relaxed_moment_cone.json")
    target.write_text(json.dumps(reports, indent=2))
    for report in reports:
        print(
            report["conductor"],
            report["coefficient_limit"],
            "feasible" if report["feasible"] else "infeasible",
            report.get("maximum_bound_fraction", ""),
        )
    print("wrote", target)


if __name__ == "__main__":
    main()

