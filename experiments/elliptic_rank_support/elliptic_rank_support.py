"""Exact support pencils through the conductor-19047851 rank-five curve.

The module deliberately does not enumerate Weierstrass coefficient boxes.  It
starts with integral points on one known curve, finds degree-three functions
whose signed values agree at six of those points, and transports the resulting
degree-six divisor.  Every output curve therefore comes with six exact rational
sections and one forced principal-divisor relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from math import isqrt
from typing import Optional, Tuple


Q = Fraction
RECORD_A = -316
RECORD_B = 1369
RECORD_CONDUCTOR = 19_047_851


Point = Optional[Tuple[Fraction, Fraction]]


def record_rhs(x: int) -> int:
    """Right side of ``Y^2 = 4*x^3 - 316*x + 1369``."""

    return 4 * x**3 + RECORD_A * x + RECORD_B


def integral_record_points(lo: int = -100, hi: int = 100) -> tuple[tuple[int, int], ...]:
    """Return nonnegative-Y integral points in a diagnostic interval."""

    points = []
    for x in range(lo, hi + 1):
        value = record_rhs(x)
        if value < 0:
            continue
        y = isqrt(value)
        if y * y == value:
            points.append((x, y))
    return tuple(points)


def cubic_interpolate(points: tuple[tuple[int, int], ...]) -> tuple[Fraction, ...]:
    """Interpolate four points, returning ascending power coefficients."""

    if len(points) != 4 or len({x for x, _ in points}) != 4:
        raise ValueError("four distinct x-coordinates are required")
    xs = [Q(x) for x, _ in points]
    divided = [Q(y) for _, y in points]
    for order in range(1, 4):
        for index in range(3, order - 1, -1):
            divided[index] = (
                (divided[index] - divided[index - 1])
                / (xs[index] - xs[index - order])
            )
    coefficients = [Q(0)] * 4
    basis = [Q(1)]
    for order in range(4):
        for index, value in enumerate(basis):
            coefficients[index] += divided[order] * value
        if order == 3:
            break
        next_basis = [Q(0)] * (len(basis) + 1)
        for index, value in enumerate(basis):
            next_basis[index] -= xs[order] * value
            next_basis[index + 1] += value
        basis = next_basis
    return tuple(coefficients)


def polynomial_value(coefficients: tuple[Fraction, ...], x: int | Fraction) -> Fraction:
    value = Q(0)
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


@dataclass(frozen=True)
class SupportPencil:
    cubic: tuple[Fraction, Fraction, Fraction, Fraction]
    support: tuple[tuple[int, int], ...]

    def shift_for_square_lead(self, u: int | Fraction) -> Fraction:
        """Shift ``g -> g+c`` so ``g_c^2-(g^2-f)`` leads with ``u^2``."""

        leading = self.cubic[3]
        if not leading:
            raise ValueError("the pencil cubic must have degree three")
        return (Q(u) ** 2 - 4) / (2 * leading)

    def integral_model(self, u: int) -> tuple[int, int, int] | None:
        """Return integral ``(a2,a4,a6)`` for the monic square-leading fiber.

        The model is ``Y^2=X^3+a2*X^2+a4*X+a6`` after ``X=u^2*x`` and
        ``Y=u^2*y``.  ``None`` means this presentation is not integral; no
        attempt is made to hide a local-minimal-model search here.
        """

        c = self.shift_for_square_lead(u)
        d0, d1, d2, _ = self.cubic
        a2 = 2 * c * d2
        linear = RECORD_A + 2 * c * d1
        constant = RECORD_B + 2 * c * d0 + c * c
        scale = Q(u) ** 2
        values = (a2, scale * linear, scale * scale * constant)
        if any(value.denominator != 1 for value in values):
            return None
        return tuple(int(value) for value in values)

    def transported_points(self, u: int) -> tuple[tuple[int, int], ...] | None:
        model = self.integral_model(u)
        if model is None:
            return None
        c = self.shift_for_square_lead(u)
        scale = u * u
        result = []
        for x, signed_y in self.support:
            y = scale * (Q(signed_y) + c)
            if y.denominator != 1:
                return None
            result.append((scale * x, int(y)))
        return tuple(result)


def support_pencils(points: tuple[tuple[int, int], ...]) -> tuple[SupportPencil, ...]:
    """Find all cubics meeting at least six signed points in ``points``.

    Global sign reversal gives the same principal divisor with the opposite
    sheet, so the first interpolation sign is fixed positive.
    """

    cubics: dict[tuple[Fraction, ...], SupportPencil] = {}
    for indices in combinations(range(len(points)), 4):
        base = tuple(points[index] for index in indices)
        for tail_signs in product((-1, 1), repeat=3):
            signed = ((base[0][0], base[0][1]),) + tuple(
                (base[index][0], tail_signs[index - 1] * base[index][1])
                for index in range(1, 4)
            )
            cubic = cubic_interpolate(signed)
            if cubic in cubics or cubic[3] == 0:
                continue
            support = []
            for x, y in points:
                value = polynomial_value(cubic, x)
                if value == y or value == -y:
                    support.append((x, int(value)))
            if len(support) >= 6:
                cubics[cubic] = SupportPencil(cubic, tuple(support))
    return tuple(cubics.values())


def invariants(model: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return ``(c4,c6,discriminant)`` for ``[0,a2,0,a4,a6]``."""

    a2, a4, a6 = model
    b2, b4, b6 = 4 * a2, 2 * a4, 4 * a6
    b8 = 4 * a2 * a6 - a4 * a4
    c4 = b2 * b2 - 24 * b4
    c6 = -b2**3 + 36 * b2 * b4 - 216 * b6
    discriminant = -b2 * b2 * b8 - 8 * b4**3 - 27 * b6**2 + 9 * b2 * b4 * b6
    return c4, c6, discriminant


def add_points(point: Point, other: Point, model: tuple[int, int, int]) -> Point:
    """Exact group law for ``Y^2=X^3+a2*X^2+a4*X+a6``."""

    if point is None:
        return other
    if other is None:
        return point
    x1, y1 = point
    x2, y2 = other
    a2, a4, _ = map(Q, model)
    if x1 == x2 and y1 == -y2:
        return None
    if point == other:
        if y1 == 0:
            return None
        slope = (3 * x1 * x1 + 2 * a2 * x1 + a4) / (2 * y1)
    else:
        slope = (y2 - y1) / (x2 - x1)
    x3 = slope * slope - a2 - x1 - x2
    return x3, -y1 + slope * (x1 - x3)


def sum_points(points: tuple[tuple[int, int], ...], model: tuple[int, int, int]) -> Point:
    total: Point = None
    for x, y in points:
        total = add_points(total, (Q(x), Q(y)), model)
    return total


def verify_fiber(pencil: SupportPencil, u: int) -> bool:
    model = pencil.integral_model(u)
    points = pencil.transported_points(u)
    if model is None or points is None:
        return False
    a2, a4, a6 = model
    on_curve = all(y * y == x**3 + a2 * x * x + a4 * x + a6 for x, y in points)
    # div(y-g-c) is the six-section divisor minus 6*O.
    return on_curve and sum_points(points, model) is None


def _add_mod_p(
    point: tuple[int, int] | None,
    other: tuple[int, int] | None,
    model: tuple[int, int, int],
    prime: int,
) -> tuple[int, int] | None:
    if point is None:
        return other
    if other is None:
        return point
    x1, y1 = point
    x2, y2 = other
    a2, a4, _ = model
    if x1 == x2 and (y1 + y2) % prime == 0:
        return None
    if point == other:
        if y1 % prime == 0:
            return None
        slope = (3 * x1 * x1 + 2 * a2 * x1 + a4) * pow(2 * y1 % prime, -1, prime)
    else:
        slope = (y2 - y1) * pow((x2 - x1) % prime, -1, prime)
    slope %= prime
    x3 = (slope * slope - a2 - x1 - x2) % prime
    return x3, (-y1 + slope * (x1 - x3)) % prime


@lru_cache(maxsize=None)
def _finite_curve_points(model: tuple[int, int, int], prime: int) -> frozenset[tuple[int, int] | None]:
    a2, a4, a6 = model
    square_roots: dict[int, list[int]] = {}
    for y in range(prime):
        square_roots.setdefault(y * y % prime, []).append(y)
    points: set[tuple[int, int] | None] = {None}
    for x in range(prime):
        rhs = (x**3 + a2 * x * x + a4 * x + a6) % prime
        points.update((x, y) for y in square_roots.get(rhs, ()))
    return frozenset(points)


def reduction_dependency_masks(
    model: tuple[int, int, int],
    points: tuple[tuple[int, int], ...],
    primes: tuple[int, ...] = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47),
    count: int = 5,
) -> set[int]:
    """Return mod-2 dependencies not killed by the supplied good primes.

    If the returned set is empty, the first ``count`` rational points are
    independent in ``E(Q)/2E(Q)`` and hence prove rank at least ``count``.
    A surviving mask is only an unresolved dependency, not proof of one.
    """

    if count > len(points):
        raise ValueError("not enough points")
    discriminant = invariants(model)[2]
    remaining = set(range(1, 1 << count))
    for prime in primes:
        if discriminant % prime == 0:
            continue
        rational_points = tuple((Q(x), Q(y)) for x, y in points[:count])
        if any(
            coordinate.denominator % prime == 0
            for point in rational_points
            for coordinate in point
        ):
            continue
        group = _finite_curve_points(model, prime)
        doubles = {_add_mod_p(point, point, model, prime) for point in group}
        reductions = tuple(
            (
                x.numerator * pow(x.denominator % prime, -1, prime) % prime,
                y.numerator * pow(y.denominator % prime, -1, prime) % prime,
            )
            for x, y in rational_points
        )
        for mask in tuple(remaining):
            total = None
            for index, point in enumerate(reductions):
                if mask & (1 << index):
                    total = _add_mod_p(total, point, model, prime)
            if total not in doubles:
                remaining.remove(mask)
        if not remaining:
            break
    return remaining
