"""Factor the first exact rank-loss relation on the split-current surface.

The low-conductor specialization ``s=2`` has exact rank five.  Its six
visible section representatives satisfy

    P1-P2+P3-P4+P5-4*P6 = O.

This script performs the group law over ``Q(s)`` and factors the equality
divisor.  It is a symbolic relation calculation, not a parameter walk.
"""

from __future__ import annotations

import sympy as sp


s = sp.symbols("s")
m = sp.Integer(1_217_307)
k1 = sp.Integer(-428_024_806)
k2 = sp.Integer(59_530_394)
k3 = sp.Integer(350_352_794)
triples = (
    (-1249, 407, 842),
    (-1078, -49, 1127),
    (-913, -313, 1226),
)

a_difference = k3 - k1
b_difference = k2 - k1
increment = sp.cancel(
    2 * (b_difference * s - a_difference)
    / (a_difference - b_difference * s**2)
)
height_ratios = (sp.Integer(1), 1 + increment, 1 + s * increment)
dilation = sp.cancel((height_ratios[1] ** 2 - 1) / b_difference)
a4 = -m * dilation**2
a0 = 1 - dilation * k1
a6 = dilation**2 * a0


Point = tuple[sp.Expr, sp.Expr]


def negate(point: Point) -> Point:
    return point[0], -point[1]


def add(left: Point, right: Point) -> Point:
    """Generic short-Weierstrass chord addition over ``Q(s)``."""

    x1, y1 = left
    x2, y2 = right
    slope = sp.cancel((y2 - y1) / (x2 - x1))
    x3 = sp.cancel(slope**2 - x1 - x2)
    y3 = sp.cancel(slope * (x1 - x3) - y1)
    return x3, y3


def double(point: Point) -> Point:
    x, y = point
    slope = sp.cancel((3 * x**2 + a4) / (2 * y))
    x2 = sp.cancel(slope**2 - 2 * x)
    y2 = sp.cancel(slope * (x - x2) - y)
    return x2, y2


def primitive_factor(expression: sp.Expr) -> sp.Expr:
    numerator = sp.together(expression).as_numer_denom()[0]
    return sp.factor(sp.primitive(sp.Poly(numerator, s).as_expr())[1])


def relation_divisors() -> tuple[sp.Expr, sp.Expr]:
    points = tuple(
        (dilation * root, dilation * height)
        for triple, height in zip(triples, height_ratios)
        for root in triple[:2]
    )
    left = add(
        add(add(add(points[0], negate(points[1])), points[2]), negate(points[3])),
        points[4],
    )
    right = double(double(points[5]))
    return primitive_factor(left[0] - right[0]), primitive_factor(
        left[1] - right[1]
    )


def main() -> None:
    x_divisor, y_divisor = relation_divisors()
    print("P1-P2+P3-P4+P5 = 4*P6")
    print("x divisor:", x_divisor)
    print("y divisor:", y_divisor)
    print("gcd:", sp.factor(sp.gcd(x_divisor, y_divisor)))


if __name__ == "__main__":
    main()

