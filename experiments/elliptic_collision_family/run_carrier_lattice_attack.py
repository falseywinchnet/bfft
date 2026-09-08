"""Exact low-height lattice on the hidden ``x=3*r/2`` carrier.

At ``t=1/9`` the carrier is the biquadratic

    d^2 = (k^2+4799)/(k^2-601),       c=(d-1)/2.

Writing ``w=d*(k^2-601)`` and

    X=2*(w+k^2)+4198,  Y=2*k*X,  x=X-4198

gives

    Y^2=x^3+4198*x^2+11536796*x+48431469608.

Exact 2-descent proves rank three, with the generators below.  They pull
back to the near miss, the interpolation anchor, and a support point.  The
functions here expose complete coefficient shells in this rank-three lattice;
they do not enumerate rational carrier parameters.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Optional

from run_weight_current_attack import WeightCandidate, _build_linear_candidate


Q = Fraction
Point = Optional[tuple[Fraction, Fraction]]

A2 = 4198
A4 = 11536796
A6 = 48431469608
GENERATORS: tuple[tuple[Fraction, Fraction], ...] = (
    (Q(-3598), Q(121200)),
    (Q(4802), Q(558000)),
    (Q(12002), Q(1587600)),
)


def add_points(left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2 and y1 == -y2:
        return None
    if left == right:
        slope = (3 * x1 * x1 + 2 * A2 * x1 + A4) / (2 * y1)
    else:
        slope = (y2 - y1) / (x2 - x1)
    x3 = slope * slope - A2 - x1 - x2
    return x3, slope * (x1 - x3) - y1


def multiply(index: int, point: Point) -> Point:
    if index < 0:
        if point is None:
            return None
        return multiply(-index, (point[0], -point[1]))
    result: Point = None
    while index:
        if index & 1:
            result = add_points(result, point)
        point = add_points(point, point)
        index //= 2
    return result


def carrier_coordinates(point: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    """Return ``(c,k)`` from a nonexceptional carrier Weierstrass point."""

    x, y = point
    capital_x = x + 4198
    if capital_x == 0:
        raise ValueError("the rational two-torsion is a carrier infinity")
    k = y / (2 * capital_x)
    denominator = k * k - 601
    if denominator == 0:
        raise ValueError("carrier coordinate has a pole")
    w = x / 2 - k * k
    d = w / denominator
    if d * d * denominator != k * k + 4799:
        raise AssertionError("inverse carrier map failed")
    return (d - 1) / 2, k


def shell(level: int) -> tuple[tuple[tuple[int, int, int], Fraction, Fraction], ...]:
    """Return the exact max-norm coefficient shell at ``level``."""

    if level < 1:
        return ()
    values = []
    for coefficients in product(range(-level, level + 1), repeat=3):
        if max(map(abs, coefficients)) != level:
            continue
        point: Point = None
        for coefficient, generator in zip(coefficients, GENERATORS):
            point = add_points(point, multiply(coefficient, generator))
        if point is None:
            continue
        try:
            c, k = carrier_coordinates(point)
        except ValueError:
            continue
        values.append((coefficients, c, k))
    return tuple(values)


def build_carrier_candidate(
    c: Fraction, k: Fraction, saturate: bool = False
) -> WeightCandidate:
    """Build the corresponding hidden-half-root surface fiber."""

    u = c * (c + 1)
    root = Q(2, 81) * k * u
    return _build_linear_candidate(c, Q(1, 9), Q(3, 2), Q(0), root, saturate)


def main() -> None:
    seen: set[Fraction] = set()
    for coefficients, c, k in shell(1):
        canonical_c = min(c, -1 - c)
        if canonical_c in seen:
            continue
        seen.add(canonical_c)
        candidate = build_carrier_candidate(c, k)
        print(
            "coefficients=", coefficients,
            "c=", c,
            "model=", candidate.model,
            "screen=", candidate.tame_conductor_lower_bound,
            "factors=", candidate.discriminant_factors,
        )


if __name__ == "__main__":
    main()
