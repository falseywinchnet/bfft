"""Exact fixed-current height recurrence above the ``m=3*19^2`` wall.

The three split support currents

    K0=-13718, K1=-10582, K2=10582

have rational heights precisely when

    u^2-c^2=3136,       v^2-c^2=24300.

Writing ``p=u+c`` eliminates the first hyperbola.  The remaining gate is

    Y^2=p^4+90928*p^2+3136^2,

an exact rank-two genus-one recurrence.  This module carries its group law
directly on the quartic and maps every point to

    E_c: y^2=x^3-1083*x+(c^2+13718).

It also records the conductor-first local law.  If ``c=e/d`` is reduced and
``q`` is a prime away from ``2,3,19``, then a denominator valuation
``k=v_q(d)`` contributes according to ``k mod 3``: cube valuations disappear
under minimalization, while the other two classes have additive conductor
exponent two.  Raw denominator size is therefore not a valid rejection gate.

The natural fourth support has current ``+13718`` and would require
``w^2=c^2+27436``.  Any such lift maps to the pointed quartic

    Z^2=(c^2+3136)*(c^2+27436).

Its Jacobian has exact rank zero and torsion ``Z/2``.  Its two rational
points are the two points at infinity, so the quartic has no rational affine
point.  This closes the fourth-height lift globally, not merely on a bounded
coefficient shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

try:
    from .run_support_reservoir_attack import QuarticGroup
    from .run_three_current_height_conic import binary_quartic_invariants
except ImportError:  # Direct execution from this directory.
    from run_support_reservoir_attack import QuarticGroup
    from run_three_current_height_conic import binary_quartic_invariants


Q = Fraction
Point = tuple[Fraction, Fraction]

M = 1_083
DEGENERATE_CURRENT = 13_718
FIRST_DIFFERENCE = 3_136
SECOND_DIFFERENCE = 24_300
FOURTH_DIFFERENCE = FIRST_DIFFERENCE + SECOND_DIFFERENCE

HEIGHT_QUARTIC = (
    Q(FIRST_DIFFERENCE**2),
    Q(0),
    Q(4 * SECOND_DIFFERENCE - 2 * FIRST_DIFFERENCE),
    Q(0),
    Q(1),
)
ORIGIN: Point = (Q(196), Q(70_560))
GENERATOR_P: Point = (Q(784), Q(658_560))
GENERATOR_Q: Point = (Q(-196), Q(70_560))

PARAMETER_JACOBIAN_MINIMAL_MODEL = (0, 1, 0, -10_919_160, 13_009_804_308)
PARAMETER_JACOBIAN_CONDUCTOR = 8_888_880

FOURTH_HEIGHT_PAIR_MINIMAL_MODEL = (0, -1, 0, -40_981_640, -76_514_264_400)
FOURTH_HEIGHT_PAIR_CONDUCTOR = 31_920


@dataclass(frozen=True)
class FixedCurrentFiber:
    """One rational point of the recurrence and its target short curve."""

    parameter: Fraction
    quartic_ordinate: Fraction
    base_height: Fraction
    first_height: Fraction
    second_height: Fraction

    @property
    def a6(self) -> Fraction:
        return self.base_height**2 + DEGENERATE_CURRENT

    @property
    def short_model(self) -> tuple[int, int, int, Fraction]:
        return 0, 0, -M, self.a6

    @property
    def primitive_height(self) -> tuple[int, int]:
        return self.base_height.numerator, self.base_height.denominator

    @property
    def discriminant(self) -> Fraction:
        """Return ``Delta(E_c)=-432*c^2*(c^2+27436)``."""

        c = self.base_height
        return -432 * c * c * (c * c + FOURTH_DIFFERENCE)

    @property
    def integral_model(self) -> tuple[int, int, int, int, int]:
        """Return a canonical integral model, not necessarily minimal.

        Scaling the rational short model by ``u=1/d`` for reduced
        ``c=e/d`` gives these integral coefficients.
        """

        e, d = self.primitive_height
        return (
            0,
            0,
            0,
            -M * d**4,
            (e * e + DEGENERATE_CURRENT * d * d) * d**4,
        )

    @property
    def integral_discriminant(self) -> int:
        e, d = self.primitive_height
        return -432 * e * e * (e * e + FOURTH_DIFFERENCE * d * d) * d**8

    def verify(self) -> bool:
        p = self.parameter
        if not p:
            return False
        if self.quartic_ordinate**2 != sum(
            coefficient * p**degree
            for degree, coefficient in enumerate(HEIGHT_QUARTIC)
        ):
            return False
        c = self.base_height
        u = self.first_height
        v = self.second_height
        if p != u + c:
            return False
        if u * u - c * c != FIRST_DIFFERENCE:
            return False
        if v * v - c * c != SECOND_DIFFERENCE:
            return False
        a6 = self.a6
        return all(
            height * height == a6 + current
            for current, height in (
                (-DEGENERATE_CURRENT, c),
                (-10_582, u),
                (10_582, v),
            )
        )


def recurrence_group() -> QuarticGroup:
    return QuarticGroup(HEIGHT_QUARTIC, ORIGIN)


def fiber_from_point(point: Point) -> FixedCurrentFiber:
    p, ordinate = point
    if not p:
        raise ValueError("the p=0 boundary does not define a finite height")
    c = (p - FIRST_DIFFERENCE / p) / 2
    u = (p + FIRST_DIFFERENCE / p) / 2
    v = ordinate / (2 * p)
    fiber = FixedCurrentFiber(p, ordinate, c, u, v)
    if not fiber.verify():
        raise ArithmeticError("fixed-current recurrence point failed verification")
    return fiber


def coefficient_box_shell(level: int) -> tuple[tuple[int, int, FixedCurrentFiber], ...]:
    """Return one exact Mordell--Weil coefficient shell.

    This is a group-law shell, not a rational-height or coefficient scan.
    Opposite height signs and repeated target curves are retained so the deck
    symmetry remains visible to callers.
    """

    if level < 0:
        raise ValueError("the shell level must be nonnegative")
    if level == 0:
        return ((0, 0, fiber_from_point(ORIGIN)),)
    group = recurrence_group()
    result = []
    for first in range(-level, level + 1):
        for second in range(-level, level + 1):
            if max(abs(first), abs(second)) != level:
                continue
            point = group.add(
                group.multiply(first, GENERATOR_P),
                group.multiply(second, GENERATOR_Q),
            )
            result.append((first, second, fiber_from_point(point)))
    return tuple(result)


def denominator_minimal_delta_exponent(valuation: int) -> int:
    """Minimal discriminant exponent at ``q not in {2,3,19}``.

    Before minimalization the canonical integral model has valuations
    ``v(a4)=4*k``, ``v(a6)=4*k`` and ``v(Delta)=8*k``.  It can be scaled
    down ``floor(2*k/3)`` times.
    """

    if valuation < 0:
        raise ValueError("a denominator valuation cannot be negative")
    return 8 * valuation - 12 * (2 * valuation // 3)


def denominator_conductor_exponent(valuation: int) -> int:
    """Return the away-from-``2,3,19`` conductor exponent.

    The residual types are good, ``IV*``, and ``IV`` for valuation classes
    zero, one, and two modulo three, respectively.
    """

    return 0 if valuation % 3 == 0 else 2


def pair_quartic_coefficients(left: int, right: int) -> tuple[int, int, int, int, int]:
    """Coefficients of ``z^2=(c^2+left)*(c^2+right)``."""

    return 1, 0, left + right, 0, left * right


def pair_quartic_raw_jacobian(left: int, right: int) -> tuple[int, int, int, int, int]:
    """Return the classical Jacobian ``y^2=x^3-27*I*x-27*J``."""

    invariant_i, invariant_j = binary_quartic_invariants(
        pair_quartic_coefficients(left, right)
    )
    return 0, 0, 0, -27 * invariant_i, -27 * invariant_j


def fourth_height_pair_raw_jacobian() -> tuple[int, int, int, int, int]:
    return pair_quartic_raw_jacobian(FIRST_DIFFERENCE, FOURTH_DIFFERENCE)


def main() -> None:
    print("parameter Jacobian:", PARAMETER_JACOBIAN_MINIMAL_MODEL)
    print("parameter conductor:", PARAMETER_JACOBIAN_CONDUCTOR)
    print("fourth-height quotient:", FOURTH_HEIGHT_PAIR_MINIMAL_MODEL)
    print("fourth-height conductor:", FOURTH_HEIGHT_PAIR_CONDUCTOR)
    print("first Mordell-Weil box shell")
    for first, second, fiber in coefficient_box_shell(1):
        e, d = fiber.primitive_height
        print(
            f"({first:+d},{second:+d}) p={fiber.parameter} "
            f"c={e}/{d} a6={fiber.a6}"
        )


if __name__ == "__main__":
    main()
