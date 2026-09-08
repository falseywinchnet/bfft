"""Rank-four carrier lattice for the hidden ``x=r-5*s`` section.

At ``t=1/11`` the universal linear carrier has a square ``c^2`` factor.
Dividing the ordinate by ``c`` leaves an elliptic cubic.  After the exact
change ``X=a*c, Y=a*(root/c)`` it is

    Y^2 = X^3 + B*X^2 + C*X + D.

PARI 2-descent proves rank four and rational two-torsion.  The four points in
``GENERATORS`` are an independent basis for the low-height carrier lattice;
their shapes are ``1/4,1/5,17/88,1/6``.  The proven rank-six seed ``c=6/49``
is ``G1+G2+G3+T``.  Keeping the torsion coset is therefore essential.

This module constructs exact coefficient shells in that lattice.  It does
not scan rational shape parameters.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Optional

from run_weight_current_attack import WeightCandidate, _build_linear_candidate


Q = Fraction
Point = Optional[tuple[Fraction, Fraction]]

LEADING = Q(-57600, 1331)
A2 = Q(-201596, 14641)
A4 = Q(-22395340800, 19487171)
A6 = Q(-143313960960000, 25937424601)
GENERATORS: tuple[tuple[Fraction, Fraction], ...] = (
    (Q(-14400, 1331), Q(10224000, 161051)),
    (Q(-11520, 1331), Q(8432640, 161051)),
    (Q(-122400, 14641), Q(89208000, 1771561)),
    (Q(-9600, 1331), Q(6585600, 161051)),
)
TORSION = (Q(57600, 1331), Q(0))


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


def shell(
    level: int, torsion_coset: bool
) -> tuple[tuple[tuple[int, int, int, int], Fraction, Fraction], ...]:
    """Return ``(coefficients,c,root)`` on one exact coefficient shell."""

    if level < 1:
        return ()
    values = []
    for coefficients in product(range(-level, level + 1), repeat=4):
        if max(map(abs, coefficients)) != level:
            continue
        point: Point = TORSION if torsion_coset else None
        for coefficient, generator in zip(coefficients, GENERATORS):
            point = add_points(point, multiply(coefficient, generator))
        if point is None:
            continue
        x, y = point
        c = x / LEADING
        root = c * y / LEADING
        values.append((coefficients, c, root))
    return tuple(values)


def build_candidate(
    c: Fraction, root: Fraction, saturate: bool = False
) -> WeightCandidate:
    return _build_linear_candidate(c, Q(1, 11), Q(1), Q(-5), root, saturate)


def main() -> None:
    seen: set[Fraction] = set()
    for coefficients, c, root in shell(1, torsion_coset=True):
        if c in seen or c.denominator > 256:
            continue
        seen.add(c)
        candidate = build_candidate(c, root)
        print(
            "coefficients=", coefficients,
            "c=", c,
            "model=", candidate.model,
            "screen=", candidate.tame_conductor_lower_bound,
            "factors=", candidate.discriminant_factors,
        )


if __name__ == "__main__":
    main()
