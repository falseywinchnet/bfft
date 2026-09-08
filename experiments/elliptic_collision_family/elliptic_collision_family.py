"""Seven-section elliptic curves from two cubic-support triples.

The construction starts from two support locations r,s and a rational right
triangle of area 4*r*s*(r+s).  Its three-square arithmetic progression gives
two horizontal triples on one elliptic curve.  The seven visible points have
two forced collinearity relations, leaving room for rank five.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from typing import Optional, Tuple


Q = Fraction
Point = Tuple[Fraction, Fraction]
RECORD_GENERALIZED_MODEL = (0, 0, 1, -79, 342)


def generalized_weierstrass_invariants(
    model: tuple[int, int, int, int, int],
) -> tuple[int, int, int]:
    """Return ``(c4,c6,Delta)`` for ``[a1,a2,a3,a4,a6]``."""

    a1, a2, a3, a4, a6 = model
    b2 = a1 * a1 + 4 * a2
    b4 = a1 * a3 + 2 * a4
    b6 = a3 * a3 + 4 * a6
    b8 = (
        a1 * a1 * a6
        + 4 * a2 * a6
        - a1 * a3 * a4
        + a2 * a3 * a3
        - a4 * a4
    )
    c4 = b2 * b2 - 24 * b4
    c6 = -b2**3 + 36 * b2 * b4 - 216 * b6
    discriminant = (
        -b2 * b2 * b8
        - 8 * b4**3
        - 27 * b6**2
        + 9 * b2 * b4 * b6
    )
    return c4, c6, discriminant


def is_prime_by_trial_division(value: int) -> bool:
    """Deterministic primality certificate adequate for the record conductor."""

    value = int(value)
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def record_has_no_rational_two_torsion() -> bool:
    """Certify trivial rational 2-torsion using the completed model mod 5."""

    # Completing the square and scaling by 2 gives
    # Y^2 = X^3 - 1264*X + 21904.  A rational 2-torsion point would give an
    # integral root of this monic cubic, hence a root modulo 5.
    return all((x**3 - 1_264 * x + 21_904) % 5 for x in range(5))


@dataclass(frozen=True)
class CollisionFiber:
    r: Fraction
    s: Fraction
    lower_height: Fraction
    center_height: Fraction
    upper_height: Fraction

    @property
    def support_product(self) -> Fraction:
        return self.r * self.s * (self.r + self.s)

    @property
    def m(self) -> Fraction:
        return self.r * self.r + self.r * self.s + self.s * self.s

    def verify(self) -> bool:
        q = self.center_height
        k = self.support_product
        return (
            self.lower_height**2 == q * q - 4 * k
            and self.upper_height**2 == q * q + 4 * k
        )

    def seven_points(self) -> tuple[Point, ...]:
        """Points on ``y^2=x^3-m*x+q^2/4``."""

        a = self.lower_height / 2
        q = self.center_height / 2
        c = self.upper_height / 2
        return (
            (Q(0), q),
            (self.r, a),
            (self.s, a),
            (-self.r - self.s, a),
            (-self.r, c),
            (-self.s, c),
            (self.r + self.s, c),
        )

    def integral_presentation(
        self, scale: int
    ) -> tuple[tuple[int, int, int], tuple[tuple[int, int], ...]]:
        """Scale ``X=scale^2*x, Y=scale^3*y`` into an integral model."""

        a4 = -self.m * scale**4
        a6 = self.center_height**2 * scale**6 / 4
        points = tuple(
            (x * scale**2, y * scale**3) for x, y in self.seven_points()
        )
        values = (a4, a6, *(coordinate for point in points for coordinate in point))
        if any(value.denominator != 1 for value in values):
            raise ValueError("the requested scale does not give an integral presentation")
        return (
            (0, int(a4), int(a6)),
            tuple((int(x), int(y)) for x, y in points),
        )


def from_right_triangle(
    r: int | Fraction,
    s: int | Fraction,
    leg_a: int | Fraction,
    leg_b: int | Fraction,
    hypotenuse: int | Fraction,
) -> CollisionFiber:
    """Build a fiber from a right triangle of area ``4*r*s*(r+s)``."""

    r, s, leg_a, leg_b, hypotenuse = map(Q, (r, s, leg_a, leg_b, hypotenuse))
    if leg_a * leg_a + leg_b * leg_b != hypotenuse * hypotenuse:
        raise ValueError("the supplied sides are not a right triangle")
    if leg_a * leg_b / 2 != 4 * r * s * (r + s):
        raise ValueError("triangle area does not match the support product")
    fiber = CollisionFiber(
        r=r,
        s=s,
        lower_height=(leg_b - leg_a) / 2,
        center_height=hypotenuse / 2,
        upper_height=(leg_a + leg_b) / 2,
    )
    if not fiber.verify():
        raise ArithmeticError("three-square progression failed")
    return fiber


def pythagorean_rank_five_family(z: int | Fraction) -> CollisionFiber:
    """The record-passing rational family; ``z=1`` is the record curve."""

    z = Q(z)
    denominator = 4 * z - 3
    if denominator == 0:
        raise ValueError("z=3/4 is the pole of this chart")
    euclid_m = 2 * z * (2 * z + 1) / denominator
    r = euclid_m / 2
    s = euclid_m + z
    # d=2 Euclidean right triangle.
    leg_a = 2 * (euclid_m * euclid_m - z * z)
    leg_b = 4 * euclid_m * z
    hypotenuse = 2 * (euclid_m * euclid_m + z * z)
    return from_right_triangle(r, s, leg_a, leg_b, hypotenuse)


def integral_right_triangles_with_area(area: int) -> tuple[tuple[int, int, int], ...]:
    """Classify positive integral right triangles of a fixed integer area."""

    results = set()
    for dilation in range(1, isqrt(area) + 1):
        if area % (dilation * dilation):
            continue
        # dilation^2*m*n*(m^2-n^2)=area bounds m^4 by area/dilation^2.
        for m in range(2, isqrt(area // (dilation * dilation)) + 2):
            for n in range(1, m):
                if gcd(m, n) != 1 or (m - n) % 2 == 0:
                    continue
                if dilation * dilation * m * n * (m * m - n * n) != area:
                    continue
                a = dilation * (m * m - n * n)
                b = 2 * dilation * m * n
                c = dilation * (m * m + n * n)
                results.add((min(a, b), max(a, b), c))
    return tuple(sorted(results))


def add_points(
    point: Optional[Point], other: Optional[Point], model: tuple[int, int, int]
) -> Optional[Point]:
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


def congruent_base_add(
    point: Optional[Point], other: Optional[Point], area: int
) -> Optional[Point]:
    return add_points(point, other, (0, -(area * area), 0))


def base_point_to_fiber(point: Point, r: int = 3, s: int = 7) -> CollisionFiber:
    """Map a point on ``v^2=u^3-area^2*u`` back to a collision fiber."""

    area = 4 * r * s * (r + s)
    x, y = point
    if y == 0:
        raise ValueError("two-torsion does not define a nondegenerate triangle")
    leg_a = (x * x - area * area) / y
    leg_b = 2 * area * x / y
    hypotenuse = (x * x + area * area) / y
    return from_right_triangle(r, s, leg_a, leg_b, hypotenuse)
