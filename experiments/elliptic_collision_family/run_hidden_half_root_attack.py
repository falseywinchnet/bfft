"""Exact recurrence on the hidden constant-half-root carrier.

The rank-six ``t=1/9`` characteristic contains an independent point at
``x=3*r/2``.  Its square current is

    z^2 = 4*c*(c+1)*(601*c^2+601*c+1350)/6561.

The involution ``c -> -1-c`` makes the first square gate a conic in
``u=c*(c+1)``; asking that ``c`` itself remain rational gives a genus-one
double cover.  This script follows one exact group direction on that cover.
It deliberately does not enumerate rational shapes.

The finite point ``O=(3/2,155/54)`` anchors the interpolation.  Starting from
the sub-record rank-five point ``P=(1/8,-101/288)``, the unique quadratic
through ``O`` tangent at ``P`` gives the first descendant.  Repeating that
same tangent construction at the descendant gives a canonical one-branch
recurrence.  No rational parameter grid is traversed.
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp


Q = Fraction
CarrierPoint = tuple[Fraction, Fraction]


def carrier(c: Fraction) -> Fraction:
    """The exact ``x=3*r/2, t=1/9`` square current."""

    c = Q(c)
    return Q(4, 6561) * c * (c + 1) * (601 * c * c + 601 * c + 1350)


ORIGIN: CarrierPoint = (Q(3, 2), Q(155, 54))
NEAR_MISS: CarrierPoint = (Q(1, 8), Q(-101, 288))


def _interpolating_quadratic(points: tuple[CarrierPoint, ...]) -> sp.Poly:
    """Return the quadratic through three points, with multiplicity allowed."""

    x = sp.symbols("x")
    a, b, d = sp.symbols("a b d")
    polynomial = a * x * x + b * x + d
    equations: list[sp.Expr] = []
    if len(points) == 2:
        # The second point is tangent and therefore counted twice.
        origin, tangent = points
        equations.extend(
            (
                polynomial.subs(x, sp.Rational(*origin[0].as_integer_ratio()))
                - sp.Rational(*origin[1].as_integer_ratio()),
                polynomial.subs(x, sp.Rational(*tangent[0].as_integer_ratio()))
                - sp.Rational(*tangent[1].as_integer_ratio()),
            )
        )
        cx = sp.Rational(*tangent[0].as_integer_ratio())
        cz = sp.Rational(*tangent[1].as_integer_ratio())
        current = (
            sp.Rational(4, 6561)
            * x
            * (x + 1)
            * (601 * x * x + 601 * x + 1350)
        )
        equations.append(sp.diff(polynomial, x).subs(x, cx) - sp.diff(current, x).subs(x, cx) / (2 * cz))
    elif len(points) == 3:
        for cx, cz in points:
            equations.append(
                polynomial.subs(x, sp.Rational(*cx.as_integer_ratio()))
                - sp.Rational(*cz.as_integer_ratio())
            )
    else:
        raise ValueError("a quadratic needs three intersections including multiplicity")
    solution = sp.solve(equations, (a, b, d), dict=True)
    if len(solution) != 1:
        raise ValueError("carrier interpolation is singular")
    return sp.Poly(sp.factor(polynomial.subs(solution[0])), x, domain=sp.QQ)


def _fraction(value: sp.Rational) -> Fraction:
    return Q(int(value.p), int(value.q))


def fourth_intersection(
    quadratic: sp.Poly, known_abscissas: tuple[Fraction, ...]
) -> CarrierPoint:
    """Recover the remaining intersection of ``z=q(c)`` with the carrier."""

    x = quadratic.gens[0]
    current = sp.Poly(
        sp.Rational(4, 6561)
        * x
        * (x + 1)
        * (601 * x * x + 601 * x + 1350),
        x,
        domain=sp.QQ,
    )
    defect = current - quadratic * quadratic
    divisor = sp.Poly(1, x, domain=sp.QQ)
    for value in known_abscissas:
        divisor *= sp.Poly(x - sp.Rational(*value.as_integer_ratio()), x, domain=sp.QQ)
    quotient, remainder = sp.div(defect, divisor)
    if not remainder.is_zero or quotient.degree() != 1:
        raise AssertionError("the prescribed carrier intersections did not divide")
    root = -quotient.nth(0) / quotient.nth(1)
    ordinate = quadratic.eval(root)
    point = (_fraction(root), _fraction(ordinate))
    assert point[1] * point[1] == carrier(point[0])
    return point


def first_descendant() -> CarrierPoint:
    quadratic = _interpolating_quadratic((ORIGIN, NEAR_MISS))
    return fourth_intersection(
        quadratic, (ORIGIN[0], NEAR_MISS[0], NEAR_MISS[0])
    )


def recurrence(count: int) -> tuple[CarrierPoint, ...]:
    """Return ``P`` and ``count-1`` points in its exact tangent recurrence."""

    if count < 1:
        return ()
    points = [NEAR_MISS]
    if count == 1:
        return tuple(points)
    points.append(first_descendant())
    while len(points) < count:
        quadratic = _interpolating_quadratic((ORIGIN, points[-1]))
        points.append(
            fourth_intersection(
                quadratic, (ORIGIN[0], points[-1][0], points[-1][0])
            )
        )
    return tuple(points)


def main() -> None:
    for index, (c, z) in enumerate(recurrence(8), start=1):
        print(
            index,
            "c=", c,
            "z=", z,
            "bits=", max(c.numerator.bit_length(), c.denominator.bit_length()),
        )


if __name__ == "__main__":
    main()
