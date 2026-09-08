"""Build the dual one-kernel certificate for the relaxed moment cone.

For ``K_rho(a)=W3(a)-rho*W1(a)``, rank five requires

    K_rho(alpha) + sum_{n>=2} a_n K_rho(alpha*n) = 0.

The universal coefficient bound ``|a_n| <= d(n)*sqrt(n)`` makes this
impossible whenever the first term is larger than the complete tail envelope.
"""

from __future__ import annotations

import json
from math import pi, sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from run_relaxed_cone import divisor_counts, moment_kernel


def separator(conductor: int, main_exponent: float = 42.0, tail_exponent: float = 70.0) -> dict[str, float | int]:
    alpha = 2.0 * pi / sqrt(conductor)
    main_limit = max(16, int(main_exponent / alpha))
    tail_limit = max(main_limit + 1, int(tail_exponent / alpha))
    indices = np.arange(1, tail_limit + 1, dtype=np.float64)
    arguments = alpha * indices
    w1 = moment_kernel(arguments, 1)
    w3 = moment_kernel(arguments, 3)
    ratios = w3 / w1
    exact_bounds = divisor_counts(main_limit)[2:] * np.sqrt(
        np.arange(2, main_limit + 1, dtype=np.float64)
    )

    def normalized_margin(rho: float) -> float:
        first = abs(w3[0] - rho * w1[0])
        main = np.sum(
            exact_bounds
            * np.abs(w3[1:main_limit] - rho * w1[1:main_limit])
        )
        # For the omitted arithmetic coefficients use d(n)*sqrt(n) <= 2n.
        # This deliberately enlarges the tail.  Terms beyond tail_exponent
        # are smaller than exp(-70) times a polynomial factor and are reported
        # separately rather than silently treated as zero.
        tail_indices = indices[main_limit:]
        tail = np.sum(
            2.0
            * tail_indices
            * np.abs(
                w3[main_limit:] - rho * w1[main_limit:]
            )
        )
        return (first - main - tail) / w1[0]

    result = minimize_scalar(
        lambda rho: -normalized_margin(float(rho)),
        bounds=(float(ratios[main_limit - 1]), float(ratios[0])),
        method="bounded",
        options={"xatol": 1e-13},
    )
    rho = float(result.x)
    relative_margin = normalized_margin(rho)
    return {
        "conductor": conductor,
        "alpha": alpha,
        "rho": rho,
        "main_limit": main_limit,
        "tail_limit": tail_limit,
        "relative_margin_before_exp70_remainder": relative_margin,
        "excluded": bool(relative_margin > 0.0),
    }


def main() -> None:
    coarse = [separator(conductor) for conductor in range(2_000, 12_001, 250)]
    transition = next(
        (
            (left["conductor"], right["conductor"])
            for left, right in zip(coarse, coarse[1:])
            if left["excluded"] and not right["excluded"]
        ),
        None,
    )
    reports = {
        "coarse": coarse,
        "transition": transition,
        "selected": [
            separator(conductor)
            for conductor in (3_000, 5_000, 7_000, 10_000, 19_047_851)
        ],
    }
    target = Path("/tmp/elliptic_separator_bound.json")
    target.write_text(json.dumps(reports, indent=2))
    print("transition", transition)
    for report in reports["selected"]:
        print(report)
    print("wrote", target)


if __name__ == "__main__":
    main()
