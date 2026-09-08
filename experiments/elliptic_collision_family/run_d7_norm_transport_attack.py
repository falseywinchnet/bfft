"""Norm-current transport from the ``d=13`` wall to the ``d=7`` surface.

The first non-wall ``d=13`` fiber has discriminant norm primes ``7`` and
``373``.  Treating the smaller split prime ``7`` as a new Eisenstein support
direction gives ``m=3*7^2=147`` and the four currents

    -686, -286, +286, +686.

The first three heights are compatible on the pointed quartic

    V^2=p^4+3088*p^2+400^2.

Its minimal Jacobian is

    Y^2=X^3+X^2-14916*X+205884,

with Mordell--Weil group ``Z + (Z/2)^2``.  If ``G=(-30,792)``, the exact
height map is

    c = (24300-(X-114)^2)/Y.

All torsion translates of ``n*G`` give the same target curve up to the
irrelevant sign ``c -> -c``.  Thus there is one curve class per ``|n|``.

The first class has conductor ``50121`` and rank three.  The second class,
``c=76719/1148``, has exact rank six.  It is not a conductor snipe, but its
sixth direction is genuinely invisible: the five horizontal sections have
rank five and the rational point recorded below supplies the sixth direction.
The opposite horizontal current remains impossible globally because its
rank-zero quotient has only its two rational points at infinity.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod

try:
    from .run_support_reservoir_attack import QuarticGroup
    from .run_three_current_height_conic import binary_quartic_invariants
except ImportError:  # Direct execution from this directory.
    from run_support_reservoir_attack import QuarticGroup
    from run_three_current_height_conic import binary_quartic_invariants


Q = Fraction
FinitePoint = tuple[int, int]
ProjectivePoint = FinitePoint | None

DIRECTION = 7
M = 3 * DIRECTION**2
DEGENERATE_CURRENT = 2 * DIRECTION**3
OTHER_CURRENT = 286
FIRST_DIFFERENCE = DEGENERATE_CURRENT - OTHER_CURRENT  # 400
SECOND_DIFFERENCE = DEGENERATE_CURRENT + OTHER_CURRENT  # 972
FOURTH_DIFFERENCE = 2 * DEGENERATE_CURRENT  # 1372

HEIGHT_QUARTIC = (
    Q(FIRST_DIFFERENCE**2),
    Q(0),
    Q(4 * SECOND_DIFFERENCE - 2 * FIRST_DIFFERENCE),
    Q(0),
    Q(1),
)
QUARTIC_ORIGIN = (Q(0), Q(FIRST_DIFFERENCE))
QUARTIC_GENERATOR = (Q(25), Q(-1_575))

PARAMETER_MODEL = (0, 1, 0, -14_916, 205_884)
PARAMETER_CONDUCTOR = 34_320
PARAMETER_GENERATOR: FinitePoint = (-30, 792)
PARAMETER_TORSION = ((-129, 0), (14, 0))

FIRST_HEIGHT = Q(9, 2)
SECOND_HEIGHT = Q(76_719, 1_148)
FIRST_TARGET_MODEL = (0, 0, 1, -147, 706)
FIRST_TARGET_CONDUCTOR = 50_121
SECOND_TARGET_MODEL = (
    0,
    0,
    0,
    -15_957_501_882_672,
    184_268_088_879_537_135_620,
)
SECOND_TARGET_CONDUCTOR = 1_750_328_941_213_617_099_804
SECOND_TARGET_FACTORISATION = (
    (2, 2),
    (3, 3),
    (7, 2),
    (41, 2),
    (107, 1),
    (239, 1),
    (7_693_969_249, 1),
)

MISSING_HEIGHT_QUOTIENT_MODEL = (0, -1, 0, -202_616, -34_012_320)
MISSING_HEIGHT_QUOTIENT_CONDUCTOR = 1_680

VISIBLE_POINTS = (
    (Q(7), SECOND_HEIGHT),
    (Q(-13), Q(80_081, 1_148)),
    (Q(2), Q(80_081, 1_148)),
    (Q(-11), Q(84_657, 1_148)),
    (Q(-2), Q(84_657, 1_148)),
)
INVISIBLE_POINT = (Q(1_187_446, 82_369), Q(7_342_183_887, 94_559_612))


@dataclass(frozen=True)
class TargetFiber:
    index: int
    parameter_point: ProjectivePoint
    height: Fraction | None

    @property
    def a6(self) -> Fraction | None:
        if self.height is None:
            return None
        return self.height * self.height + DEGENERATE_CURRENT

    @property
    def discriminant(self) -> Fraction | None:
        if self.height is None:
            return None
        c = self.height
        return -432 * c * c * (c * c + FOURTH_DIFFERENCE)


def recurrence_group() -> QuarticGroup:
    return QuarticGroup(HEIGHT_QUARTIC, QUARTIC_ORIGIN)


def quartic_height(point: tuple[Fraction, Fraction]) -> Fraction:
    p, _ordinate = point
    if not p:
        raise ValueError("the p=0 point is the infinite-height boundary")
    return (p - FIRST_DIFFERENCE / p) / 2


def parameter_rhs(x: Fraction) -> Fraction:
    return x**3 + x**2 - 14_916 * x + 205_884


def height_from_parameter_point(point: tuple[Fraction, Fraction]) -> Fraction:
    x, y = point
    if y == 0:
        raise ValueError("a parameter 2-torsion point is a height pole")
    if y * y != parameter_rhs(x):
        raise ValueError("point is not on the parameter curve")
    return (24_300 - (x - 114) ** 2) / y


def target_rhs(x: Fraction, height: Fraction = SECOND_HEIGHT) -> Fraction:
    return x**3 - M * x + height**2 + DEGENERATE_CURRENT


def verify_target_point(
    point: tuple[Fraction, Fraction], height: Fraction = SECOND_HEIGHT
) -> bool:
    x, y = point
    return y * y == target_rhs(x, height)


def target_discriminant_currents(height: Fraction) -> tuple[int, int, int]:
    """Return primitive ``(e,d,e^2+1372*d^2)`` for ``c=e/d``."""

    e, d = height.numerator, height.denominator
    return e, d, e * e + FOURTH_DIFFERENCE * d * d


def pair_quartic_raw_jacobian(left: int, right: int) -> tuple[int, int, int, int, int]:
    coefficients = (1, 0, left + right, 0, left * right)
    invariant_i, invariant_j = binary_quartic_invariants(coefficients)
    return 0, 0, 0, -27 * invariant_i, -27 * invariant_j


def missing_height_raw_jacobian() -> tuple[int, int, int, int, int]:
    return pair_quartic_raw_jacobian(FIRST_DIFFERENCE, FOURTH_DIFFERENCE)


def _inverse_mod(value: int, modulus: int) -> int:
    return pow(value % modulus, -1, modulus)


def finite_add(
    left: ProjectivePoint, right: ProjectivePoint, modulus: int
) -> ProjectivePoint:
    """Group law for the parameter model over ``F_modulus``."""

    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 % modulus == x2 % modulus and (y1 + y2) % modulus == 0:
        return None
    if x1 % modulus == x2 % modulus:
        slope = (
            (3 * x1 * x1 + 2 * x1 - 14_916)
            * _inverse_mod(2 * y1, modulus)
        ) % modulus
    else:
        slope = ((y2 - y1) * _inverse_mod(x2 - x1, modulus)) % modulus
    x3 = (slope * slope - 1 - x1 - x2) % modulus
    y3 = (slope * (x1 - x3) - y1) % modulus
    return x3, y3


def finite_multiply(
    coefficient: int, point: ProjectivePoint, modulus: int
) -> ProjectivePoint:
    if coefficient < 0:
        if point is None:
            return None
        return finite_multiply(-coefficient, (point[0], -point[1]), modulus)
    result: ProjectivePoint = None
    addend = point
    while coefficient:
        if coefficient & 1:
            result = finite_add(result, addend, modulus)
        addend = finite_add(addend, addend, modulus)
        coefficient //= 2
    return result


def finite_order(point: ProjectivePoint, modulus: int) -> int:
    if point is None:
        return 1
    current: ProjectivePoint = None
    for order in range(1, modulus + 2 + 2 * int(modulus**0.5) + 1):
        current = finite_add(current, point, modulus)
        if current is None:
            return order
    raise ArithmeticError("point order exceeded the Hasse bound")


def bad_divisor_type(point: ProjectivePoint, modulus: int) -> str | None:
    """Classify reduction on ``c*(c^2+1372)=0`` or a height pole."""

    if point is None:
        return "pole"
    x, y = point
    numerator = (24_300 - (x - 114) ** 2) % modulus
    if y % modulus == 0:
        return "pole"
    if numerator == 0:
        return "zero"
    if (numerator * numerator + FOURTH_DIFFERENCE * y * y) % modulus == 0:
        return "norm"
    return None


def local_bad_orbit(modulus: int) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
    generator = (PARAMETER_GENERATOR[0] % modulus, PARAMETER_GENERATOR[1] % modulus)
    order = finite_order(generator, modulus)
    indices = []
    types = []
    for index in range(1, order + 1):
        kind = bad_divisor_type(finite_multiply(index, generator, modulus), modulus)
        if kind is not None:
            indices.append(index)
            types.append(kind)
    return order, tuple(indices), tuple(types)


def main() -> None:
    print("parameter:", PARAMETER_MODEL, "N=", PARAMETER_CONDUCTOR)
    print("G height:", height_from_parameter_point(tuple(map(Q, PARAMETER_GENERATOR))))
    print("2G height:", SECOND_HEIGHT)
    print("rank-six model:", SECOND_TARGET_MODEL)
    print("rank-six conductor:", SECOND_TARGET_CONDUCTOR)
    print("factorisation:", SECOND_TARGET_FACTORISATION)
    print("currents (e,d,norm):", target_discriminant_currents(SECOND_HEIGHT))
    print("invisible point verified:", verify_target_point(INVISIBLE_POINT))
    for prime in (7, 41, 107, 239):
        print(prime, local_bad_orbit(prime))


if __name__ == "__main__":
    main()
