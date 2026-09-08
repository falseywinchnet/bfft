"""Galois-trace audit of the smooth ramification endpoint ``w=4``.

The root-lattice rank-six cover ramifies at ``t=1+sqrt(2)``.  Its curve is
defined over Q because the true shape is ``c=-25/12``.  This script transports
the seven support sections and the root-lattice section to the common rational
model, adds each point to its quadratic conjugate, and certifies the rank of
the descended trace subgroup by reductions at good primes.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from pathlib import Path
import sys
from typing import Optional

import sympy as sp


HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / "elliptic_rank_support"
sys.path.insert(0, str(SUPPORT))

from elliptic_rank_support import reduction_dependency_masks  # noqa: E402


Q = Fraction
Point = Optional[tuple[sp.Expr, sp.Expr]]


def add_short(left: Point, right: Point, a4: sp.Expr) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if sp.simplify(x1 - x2) == 0 and sp.simplify(y1 + y2) == 0:
        return None
    if sp.simplify(x1 - x2) == 0 and sp.simplify(y1 - y2) == 0:
        slope = sp.simplify((3 * x1 * x1 + a4) / (2 * y1))
    else:
        slope = sp.simplify((y2 - y1) / (x2 - x1))
    x3 = sp.simplify(slope * slope - x1 - x2)
    y3 = sp.simplify(-y1 + slope * (x1 - x3))
    return x3, y3


def _fraction(value: sp.Expr) -> Fraction:
    value = sp.cancel(sp.simplify(value))
    if not value.is_Rational:
        raise ArithmeticError(f"quadratic trace did not descend: {value}")
    return Q(int(value.p), int(value.q))


def endpoint_model_and_points() -> tuple[
    tuple[int, int, int], tuple[tuple[Fraction, Fraction], ...]
]:
    radical = sp.sqrt(2)
    t = 1 + radical
    c = sp.Rational(-25, 12)
    transport = sp.expand(t * (1 - t * t))
    linear = 1 + t * t
    r = sp.expand(transport * c * (1 + c))
    s = sp.expand(c * r)
    height_scale = sp.expand(transport * c * c * (1 + c) ** 2)
    lower = sp.expand(height_scale * (1 - 2 * t - t * t))
    center = sp.expand(height_scale * linear)
    upper = sp.expand(height_scale * (1 + 2 * t - t * t))

    standard_points = [
        (sp.Integer(0), center / 2),
        (r, lower / 2),
        (s, lower / 2),
        (-r - s, lower / 2),
        (-r, upper / 2),
        (-s, upper / 2),
        (r + s, upper / 2),
    ]
    root = sp.expand(
        -c * (1 + c) * (1 - 2 * t - t * t) * (6 * c + 13) / 5
    )
    extra_x = sp.expand((3 + 2 * c) * r)
    extra_y = sp.expand(transport * c * (1 + c) * root / 2)
    standard_points.append((extra_x, extra_y))

    # Divide by u=T/(1+t^2) to reach the c-only generalized model
    # [0,0,q^2,-q^2*(c^2+c+1),0], then use the net 36/5 dilation to
    # reach its primitive integral presentation.
    u = sp.cancel(transport / linear)
    q = sp.cancel(-2 * (c + 1) * (6 * c + 25) / 3)
    canonical_a3 = sp.cancel(q * q)
    canonical_a4 = sp.cancel(-q * q * (c * c + c + 1))
    net_scale = sp.Rational(36, 5)
    a3 = int(sp.cancel(canonical_a3 * net_scale**3))
    a4 = int(sp.cancel(canonical_a4 * net_scale**4))
    assert (a3, a4) == (30_420, -713_349)
    short_model = (0, 16 * a4, 16 * a3 * a3)

    quadratic_points: list[Point] = []
    for x, y in standard_points:
        generalized_x = sp.cancel(x * net_scale**2 / u**2)
        generalized_y = sp.cancel((y - center / 2) * net_scale**3 / u**3)
        short_x = sp.cancel(4 * generalized_x)
        short_y = sp.cancel(4 * (2 * generalized_y + a3))
        assert sp.simplify(
            short_y**2
            - (
                short_x**3
                + short_model[1] * short_x
                + short_model[2]
            )
        ) == 0
        quadratic_points.append((short_x, short_y))

    traces: list[tuple[Fraction, Fraction]] = []
    short_a4 = sp.Integer(short_model[1])
    for point in quadratic_points:
        assert point is not None
        conjugate = tuple(
            sp.expand(coordinate.xreplace({radical: -radical}))
            for coordinate in point
        )
        trace = add_short(point, conjugate, short_a4)
        if trace is None:
            continue
        rational_trace = (_fraction(trace[0]), _fraction(trace[1]))
        if rational_trace not in traces and (
            rational_trace[0], -rational_trace[1]
        ) not in traces:
            traces.append(rational_trace)
    return short_model, tuple(traces)


def independent_trace_rank() -> tuple[int, tuple[int, ...] | None]:
    model, points = endpoint_model_and_points()
    for rank in range(min(6, len(points)), 0, -1):
        for indices in combinations(range(len(points)), rank):
            selected = tuple(points[index] for index in indices)
            if not reduction_dependency_masks(model, selected, count=rank):
                return rank, indices
    return 0, None


def main() -> None:
    model, points = endpoint_model_and_points()
    print("short model:", model)
    for index, point in enumerate(points):
        print(index, point)
    print("independent trace rank:", independent_trace_rank())


if __name__ == "__main__":
    main()
