"""Mellin-current coordinates for an elliptic curve over Q.

For a weight-two newform ``f`` of conductor ``N``, put

    G(t) = exp(t) f(i exp(t) / sqrt(N)).

Then the completed L-function centered at one is the bilateral Laplace
transform of G.  When the functional-equation sign is -1, G is odd and

    Lambda(1+z) = 2 integral_0^infinity G(t) sinh(z t) dt.

Thus analytic rank at least five is exactly the vanishing of the first and
third odd Mellin-current moments.  The module evaluates the positive half of
that current directly from the curve's Euler coefficients.  It is a probe of
one supplied curve, not a search through Weierstrass models.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isqrt, pi, sqrt

import numpy as np


@dataclass(frozen=True)
class EllipticCurve:
    conductor: int
    a1: int
    a2: int
    a3: int
    a4: int
    a6: int
    name: str = "curve"

    def rhs_residual(self, x: int, y: int, modulus: int) -> int:
        return (
            y * y + self.a1 * x * y + self.a3 * y
            - x**3 - self.a2 * x * x - self.a4 * x - self.a6
        ) % modulus


def prime_sieve(limit: int) -> tuple[list[int], list[int]]:
    """Return primes and a smallest-prime-factor table through ``limit``."""

    spf = list(range(limit + 1))
    if limit >= 1:
        spf[1] = 1
    for p in range(2, isqrt(limit) + 1):
        if spf[p] != p:
            continue
        for multiple in range(p * p, limit + 1, p):
            if spf[multiple] == multiple:
                spf[multiple] = p
    return [n for n in range(2, limit + 1) if spf[n] == n], spf


def trace_at_prime(curve: EllipticCurve, prime: int) -> int:
    """Return ``a_p = p + 1 - #E(F_p)`` by an exact finite-field count."""

    if prime == 2:
        affine = sum(
            curve.rhs_residual(x, y, prime) == 0
            for x in range(prime)
            for y in range(prime)
        )
        return prime - affine

    # For fixed x the y-equation has 1 + chi(D_x) roots, where
    # D_x=(a1*x+a3)^2+4*(x^3+a2*x^2+a4*x+a6).
    is_square = bytearray(prime)
    for value in range(1, (prime + 1) // 2):
        is_square[(value * value) % prime] = 1
    character_sum = 0
    for x in range(prime):
        linear = (curve.a1 * x + curve.a3) % prime
        cubic = (x**3 + curve.a2 * x * x + curve.a4 * x + curve.a6) % prime
        discriminant = (linear * linear + 4 * cubic) % prime
        if discriminant:
            character_sum += 1 if is_square[discriminant] else -1
    return -character_sum


def dirichlet_coefficients(curve: EllipticCurve, limit: int) -> np.ndarray:
    """Compute ``a_n`` through ``limit`` from exact prime traces."""

    primes, spf = prime_sieve(limit)
    traces = {prime: trace_at_prime(curve, prime) for prime in primes}
    coefficients = np.zeros(limit + 1, dtype=np.int64)
    coefficients[1] = 1
    prime_power: dict[tuple[int, int], int] = {}
    for prime in primes:
        prime_power[(prime, 0)] = 1
        prime_power[(prime, 1)] = traces[prime]
        power = prime * prime
        exponent = 2
        while power <= limit:
            if curve.conductor % prime == 0:
                value = traces[prime] ** exponent
            else:
                value = (
                    traces[prime] * prime_power[(prime, exponent - 1)]
                    - prime * prime_power[(prime, exponent - 2)]
                )
            prime_power[(prime, exponent)] = value
            power *= prime
            exponent += 1

    for n in range(2, limit + 1):
        prime = spf[n]
        remainder = n
        exponent = 0
        while remainder % prime == 0:
            remainder //= prime
            exponent += 1
        coefficients[n] = prime_power[(prime, exponent)] * coefficients[remainder]
    return coefficients


def positive_current(
    curve: EllipticCurve,
    coefficients: np.ndarray,
    t_values: np.ndarray,
) -> np.ndarray:
    """Evaluate ``G(t)`` for nonnegative log-distance values."""

    indices = np.arange(1, len(coefficients), dtype=np.float64)
    values = coefficients[1:].astype(np.float64)
    alpha = 2.0 * pi / sqrt(curve.conductor)
    result = np.empty_like(t_values, dtype=np.float64)
    for index, t in enumerate(t_values):
        et = exp(float(t))
        result[index] = et * np.dot(values, np.exp(-alpha * et * indices))
    return result


def _simpson(values: np.ndarray, spacing: float) -> float:
    if len(values) % 2 == 0:
        raise ValueError("Simpson quadrature needs an odd sample count")
    return spacing / 3.0 * (
        values[0]
        + values[-1]
        + 4.0 * np.sum(values[1:-1:2])
        + 2.0 * np.sum(values[2:-1:2])
    )


def current_report(
    curve: EllipticCurve,
    coefficient_limit: int,
    t_max: float = 11.0,
    sample_count: int = 2201,
) -> dict[str, object]:
    """Return moments, cancellation ratios, and cumulative-current landmarks."""

    if sample_count % 2 == 0:
        sample_count += 1
    coefficients = dirichlet_coefficients(curve, coefficient_limit)
    t_values = np.linspace(0.0, t_max, sample_count)
    current = positive_current(curve, coefficients, t_values)
    spacing = float(t_values[1] - t_values[0])

    moments: dict[int, float] = {}
    absolute_moments: dict[int, float] = {}
    cancellation: dict[int, float] = {}
    for degree in (1, 3, 5, 7):
        weighted = current * t_values**degree
        moments[degree] = 2.0 * _simpson(weighted, spacing)
        absolute_moments[degree] = 2.0 * _simpson(np.abs(weighted), spacing)
        cancellation[degree] = abs(moments[degree]) / max(absolute_moments[degree], 1e-300)

    # Cumulative first and third moments reveal where cancellation is carried.
    cumulative: dict[int, list[float]] = {}
    for degree in (1, 3):
        weighted = 2.0 * current * t_values**degree
        trapezoids = 0.5 * spacing * (weighted[:-1] + weighted[1:])
        cumulative[degree] = np.concatenate(([0.0], np.cumsum(trapezoids))).tolist()

    sign_changes = [
        float(t_values[index])
        for index in range(1, len(current))
        if current[index - 1] * current[index] < 0.0
    ]
    return {
        "curve": curve.name,
        "conductor": curve.conductor,
        "coefficient_limit": coefficient_limit,
        "moments": moments,
        "absolute_moments": absolute_moments,
        "cancellation_ratios": cancellation,
        "current_sign_changes": sign_changes,
        "t": t_values.tolist(),
        "current": current.tolist(),
        "cumulative": cumulative,
        "selected_coefficients": {
            str(index): int(coefficients[index])
            for index in range(1, min(40, len(coefficients)))
        },
    }

