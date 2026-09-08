"""Exact recurrence on the Pythagorean lift quotients of two A2 weights.

Both shape-discriminant quartics have the involution

    (u, v) -> (4/u, 4*v/u^2).

On the further cover ``d^2=u^2+4`` this lifts to

    (u, v, d) -> (4/u, 4*v/u^2, 2*d/u).

The invariant coordinates

    w = u + 4/u,  z = v/u,  D = d*(u+2)/u

satisfy ``D^2=w*(w+4)``.  Parametrizing that conic by
``rho=D/w`` and converting the resulting even quartic to a 2-isogeny model
gives the two small elliptic curves below.  A point on one of these quotient
curves lifts back to rational ``u`` precisely when ``w^2-16`` is a square.

This module follows the resulting Mordell--Weil recurrence exactly.  It does
not search coefficient boxes or rational parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isqrt
from typing import Optional, Tuple


Q = Fraction
Point = Optional[Tuple[Fraction, Fraction]]


@dataclass(frozen=True)
class LiftQuotient:
    name: str
    quartic_a: int
    jacobi_scale: int
    torsion_x: int
    generator: tuple[Fraction, Fraction]
    second_generator: Optional[Tuple[Fraction, Fraction]] = None
    third_generator: Optional[Tuple[Fraction, Fraction]] = None
    shell_basis: Tuple[Tuple[Fraction, Fraction], ...] = ()

    @property
    def a2(self) -> Fraction:
        return Q(-self.torsion_x)

    @property
    def a4(self) -> Fraction:
        return Q(-(self.jacobi_scale**2) * self.quartic_a)

    @property
    def a6(self) -> Fraction:
        return Q(
            (self.jacobi_scale**2) * self.quartic_a * self.torsion_x
        )

    @property
    def torsion(self) -> tuple[Fraction, Fraction]:
        return Q(self.torsion_x), Q(0)


DOUBLED = LiftQuotient(
    "doubled",
    409,
    42,
    594,
    (Q(1074), Q(14400)),
    (Q(882), Q(4032)),
    shell_basis=((Q(1074), Q(14400)), (Q(1554), Q(-40320))),
)
MIXED = LiftQuotient(
    "mixed",
    241,
    42,
    666,
    (Q(618), Q(-1440)),
    (Q(630), Q(-1008)),
    shell_basis=((Q(618), Q(-1440)), (Q(714), Q(-2016))),
)
NEXT_PRIMITIVE = LiftQuotient(
    "next_primitive",
    349,
    54,
    1053,
    (Q(1728), Q(36450)),
    (Q(972), Q(-2430)),
    shell_basis=(
        (Q(1728), Q(36450)),
        (Q(48897, 49), Q(376650, 343)),
    ),
)
FOUR_OMEGA = LiftQuotient(
    "four_omega",
    23017,
    378,
    58482,
    (Q(77490), Q(7185024)),
    (Q(53298), Q(-1524096)),
    (Q(81810), Q(8911296)),
    (
        (Q(77490), Q(7185024)),
        (Q(53298), Q(-1524096)),
        (Q(81810), Q(8911296)),
    ),
)
THREE_TWO = LiftQuotient(
    "three_two",
    445,
    60,
    1296,
    (Q(2220), Q(55440)),
    (Q(1260), Q(-720)),
    shell_basis=(
        (Q(2220), Q(55440)),
        (Q(4953, 4), Q(15939, 8)),
    ),
)


def rational_square_root(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    if numerator * numerator != value.numerator:
        return None
    if denominator * denominator != value.denominator:
        return None
    return Q(numerator, denominator)


def on_curve(spec: LiftQuotient, point: Point) -> bool:
    if point is None:
        return True
    x, y = point
    return y * y == (x - spec.torsion_x) * (
        x * x - spec.jacobi_scale**2 * spec.quartic_a
    )


def add_points(spec: LiftQuotient, left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2:
        if y1 == -y2:
            return None
        slope = (3 * x1 * x1 + 2 * spec.a2 * x1 + spec.a4) / (2 * y1)
    else:
        slope = (y2 - y1) / (x2 - x1)
    x3 = slope * slope - spec.a2 - x1 - x2
    result = x3, -y1 + slope * (x1 - x3)
    assert on_curve(spec, result)
    return result


def multiply_point(spec: LiftQuotient, point: Point, multiplier: int) -> Point:
    if multiplier < 0:
        if point is None:
            return None
        return multiply_point(spec, (point[0], -point[1]), -multiplier)
    result = None
    addend = point
    while multiplier:
        if multiplier & 1:
            result = add_points(spec, result, addend)
        addend = add_points(spec, addend, addend)
        multiplier >>= 1
    return result


def quotient_coordinates(
    spec: LiftQuotient, point: Point
) -> tuple[Fraction, Fraction]:
    """Recover ``(rho,w)`` from a nonexceptional quotient-curve point."""

    if point is None:
        raise ValueError("the identity is exceptional in quotient coordinates")
    x, y = point
    capital_x = x / spec.jacobi_scale
    denominator = capital_x * capital_x - spec.quartic_a
    if denominator == 0:
        raise ValueError("point is exceptional in the Jacobi-quartic map")
    rho = (y / spec.jacobi_scale) / denominator
    if rho * rho == 1:
        raise ValueError("point maps to infinity on the w-conic")
    w = 4 / (rho * rho - 1)
    return rho, w


def split_u(w: Fraction) -> tuple[Fraction, Fraction] | None:
    """Split ``u^2-w*u+4``; this is the final rational-lift current."""

    root = rational_square_root(w * w - 16)
    if root is None:
        return None
    return (w + root) / 2, (w - root) / 2


def split_multiples(
    spec: LiftQuotient,
    count: int,
    torsion_coset: bool = False,
) -> tuple[tuple[int, Fraction, tuple[Fraction, Fraction]], ...]:
    """Return the exactly split points among the first ``count`` multiples."""

    hits = []
    start = 0 if torsion_coset else 1
    for multiplier in range(start, count + 1):
        point = multiply_point(spec, spec.generator, multiplier)
        if torsion_coset:
            point = add_points(spec, point, spec.torsion)
        try:
            _rho, w = quotient_coordinates(spec, point)
        except ValueError:
            continue
        split = split_u(w)
        if split is not None:
            hits.append((multiplier, w, split))
    return tuple(hits)


def split_lattice_shell(
    spec: LiftQuotient,
    radius: int,
) -> tuple[
    tuple[tuple[int, ...], bool, Fraction, tuple[Fraction, Fraction]], ...
]:
    """Test the exact Mordell--Weil shell in the stored short basis.

    Opposite points and duplicate presentations are identified before the
    final split test.  The basis contains every generator exposed by the
    homogeneous 2-isogeny covers, so the audit works without assuming rank
    one or two.
    """

    if not spec.shell_basis:
        raise ValueError("the lift quotient has no certified shell basis")
    seen = set()
    hits = []
    coefficient_range = range(-radius, radius + 1)
    for coefficients in product(
        coefficient_range, repeat=len(spec.shell_basis)
    ):
        if all(coefficient == 0 for coefficient in coefficients):
            continue
        point = None
        for coefficient, generator in zip(coefficients, spec.shell_basis):
            point = add_points(
                spec,
                point,
                multiply_point(spec, generator, coefficient),
            )
        if point is None:
            continue
        key = (point[0], abs(point[1]))
        if key in seen:
            continue
        seen.add(key)
        for torsion_coset in (False, True):
            candidate = (
                add_points(spec, point, spec.torsion)
                if torsion_coset
                else point
            )
            try:
                _rho, w = quotient_coordinates(spec, candidate)
            except ValueError:
                continue
            split = split_u(w)
            if split is not None:
                hits.append((coefficients, torsion_coset, w, split))
    return tuple(hits)


def main() -> None:
    for spec in (
        DOUBLED,
        MIXED,
        NEXT_PRIMITIVE,
        FOUR_OMEGA,
        THREE_TWO,
    ):
        print(spec.name, "curve generator", spec.generator)
        count = 64 if spec in (NEXT_PRIMITIVE, FOUR_OMEGA) else 28
        print("ordinary", split_multiples(spec, count))
        print(
            "torsion coset",
            split_multiples(spec, count, torsion_coset=True),
        )
        radius = 8 if spec is FOUR_OMEGA else 16 if spec is THREE_TWO else 12
        print("Mordell--Weil shell", split_lattice_shell(spec, radius))


if __name__ == "__main__":
    main()
