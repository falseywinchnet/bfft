"""Solve a multiplicatively coherent extremal Frobenius-phase model.

At every prime set the normalized local trace to ``2*cos(theta)``.  The
prime-power recurrence is then the Chebyshev recurrence

    U_0=1, U_1=2c, U_k=2c U_{k-1}-U_{k-2}.

Consequently ``a_n=sqrt(n)*prod_p U_{v_p(n)}(c)`` respects the full coprime
multiplicativity and every good-prime recurrence.  It is not claimed to be an
elliptic curve; it is the smallest coherent replacement for the independent
Hasse box.
"""

from __future__ import annotations

import json
from math import exp, log, pi, sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from mellin_current import prime_sieve
from run_relaxed_cone import moment_kernel


MAXIMUM_LIMIT = 80_000
_, SPF = prime_sieve(MAXIMUM_LIMIT)


def normalized_coefficients(cosine: float, limit: int) -> np.ndarray:
    max_exponent = int(log(limit, 2)) + 1
    chebyshev = np.empty(max_exponent + 1, dtype=np.float64)
    chebyshev[0] = 1.0
    chebyshev[1] = 2.0 * cosine
    for exponent in range(2, max_exponent + 1):
        chebyshev[exponent] = (
            2.0 * cosine * chebyshev[exponent - 1] - chebyshev[exponent - 2]
        )
    values = np.ones(limit + 1, dtype=np.float64)
    for n in range(2, limit + 1):
        prime = SPF[n]
        remainder = n
        exponent = 0
        while remainder % prime == 0:
            remainder //= prime
            exponent += 1
        values[n] = chebyshev[exponent] * values[remainder]
    indices = np.arange(limit + 1, dtype=np.float64)
    return np.sqrt(indices) * values


def residual(parameters: np.ndarray) -> np.ndarray:
    conductor = exp(float(parameters[0]))
    cosine = float(parameters[1])
    alpha = 2.0 * pi / sqrt(conductor)
    limit = min(MAXIMUM_LIMIT, max(32, int(42.0 / alpha)))
    indices = np.arange(1, limit + 1, dtype=np.float64)
    coefficients = normalized_coefficients(cosine, limit)[1:]
    w1 = moment_kernel(alpha * indices, 1)
    w3 = moment_kernel(alpha * indices, 3)
    # Normalize separately so the solver follows cancellation, not magnitude.
    return np.asarray(
        (
            np.dot(coefficients, w1) / w1[0],
            np.dot(coefficients, w3) / w3[0],
        )
    )


def main() -> None:
    starts = (
        (10_000, -0.95),
        (100_000, -0.95),
        (1_000_000, -0.95),
        (10_000_000, -0.95),
        (20_000_000, -0.75),
        (20_000_000, -0.5),
    )
    solutions = []
    for conductor, cosine in starts:
        fit = least_squares(
            residual,
            np.asarray((log(conductor), cosine)),
            bounds=((log(100.0), -1.0), (log(100_000_000.0), 1.0)),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=120,
        )
        solutions.append(
            {
                "start": {"conductor": conductor, "cosine": cosine},
                "conductor": exp(float(fit.x[0])),
                "cosine": float(fit.x[1]),
                "residual": residual(fit.x).tolist(),
                "cost": float(fit.cost),
                "success": bool(fit.success),
                "message": fit.message,
            }
        )
    target = Path("/tmp/elliptic_coherent_phase.json")
    target.write_text(json.dumps(solutions, indent=2))
    for solution in solutions:
        print(solution)
    print("wrote", target)


if __name__ == "__main__":
    main()

