"""Norm-preserving recurrence behind the two non-Weyl ``m=481`` supports.

The ``13``-adic split support and the integral currents ``-3600,-2280``
define a three-current height conic through the rank-five wall.  Its residual
curve-discriminant current is a quadratic norm over ``Q(sqrt(1443))``.  The
fundamental unit ``38+sqrt(1443)`` gives an exact squareclass-preserving
recurrence in the dilation ``lambda``.

Imposing the two height squares on the fixed-squareclass curve gives a
genus-three ``(Z/2)^3`` cover.  Its Jacobian splits into three elliptic
quotients.  The useful quotient has a wall-point recurrence; the remaining
square gate is a rank-two Prym.  PARI proves the Prym rank exactly, and the
first two coefficient shells below contain only wall deck returns.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import isqrt


Q = Fraction
Point = tuple[Fraction, Fraction] | None
Quadratic = tuple[Fraction, Fraction]

M = 481
H = Q(11_383, 2_197)
K1 = Q(-18_989_417_760, 4_826_809)
K2 = Q(-3_600)
K3 = Q(-2_280)
A = K3 - K1
B = K2 - K1

NORM_FIELD = 1_443
FUNDAMENTAL_UNIT: Quadratic = (Q(38), Q(1))
NORM_FACTOR_RATIONAL = Q(
    -16_608_441_971_526_007_320,
    158_682_627_512_066_045_521,
)
NORM_FACTOR_ROOT = Q(
    -902_484_842_708_195_643,
    317_365_255_024_132_091_042,
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


def residual_current(dilation: Fraction) -> Fraction:
    return (
        27 * H**4
        - 54 * H * H * K1 * dilation
        + (27 * K1 * K1 - 4 * M**3) * dilation * dilation
    )


def _quadratic_multiply(left: Quadratic, right: Quadratic) -> Quadratic:
    return (
        left[0] * right[0] + NORM_FIELD * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _unit_power(index: int) -> Quadratic:
    unit = FUNDAMENTAL_UNIT if index >= 0 else (Q(38), Q(-1))
    index = abs(index)
    result: Quadratic = (Q(1), Q(0))
    while index:
        if index & 1:
            result = _quadratic_multiply(result, unit)
        unit = _quadratic_multiply(unit, unit)
        index //= 2
    return result


def norm_unit_dilation(index: int) -> Fraction:
    """Return the squareclass-preserving dilation from one unit power."""

    initial = (Q(1) + NORM_FACTOR_RATIONAL, NORM_FACTOR_ROOT)
    rational, root = _quadratic_multiply(_unit_power(index), initial)
    scale = NORM_FACTOR_ROOT / root
    dilation = scale * rational - NORM_FACTOR_RATIONAL
    left = (dilation + NORM_FACTOR_RATIONAL, NORM_FACTOR_ROOT)
    norm = lambda value: value[0] ** 2 - NORM_FIELD * value[1] ** 2
    if norm(left) != scale * scale * norm(initial):
        raise ArithmeticError("quadratic-unit norm recurrence failed")
    return dilation


def height_roots(dilation: Fraction) -> tuple[Fraction | None, Fraction | None]:
    return (
        rational_square_root(H * H + B * dilation),
        rational_square_root(H * H + A * dilation),
    )


def quotient_cubic_coefficients() -> tuple[Fraction, ...]:
    """Coefficients of ``Y^2=R(1)*(H^2+B*l)*R(l)``."""

    r0 = 27 * H**4
    r1 = -54 * H * H * K1
    r2 = 27 * K1 * K1 - 4 * M**3
    r_at_one = r0 + r1 + r2
    return (
        r_at_one * H * H * r0,
        r_at_one * (H * H * r1 + B * r0),
        r_at_one * (H * H * r2 + B * r1),
        r_at_one * B * r2,
    )


QUOTIENT_CUBIC = quotient_cubic_coefficients()
R_AT_ONE = residual_current(Q(1))
QUOTIENT_WALL_POINT: Point = (Q(1), R_AT_ONE * 19)


def _cubic_value(abscissa: Fraction) -> Fraction:
    return sum(
        coefficient * abscissa**degree
        for degree, coefficient in enumerate(QUOTIENT_CUBIC)
    )


def quotient_add(left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2 and y1 == -y2:
        return None
    if left == right:
        derivative = (
            QUOTIENT_CUBIC[1]
            + 2 * QUOTIENT_CUBIC[2] * x1
            + 3 * QUOTIENT_CUBIC[3] * x1 * x1
        )
        slope = derivative / (2 * y1)
    else:
        slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    x3 = (
        (slope * slope - QUOTIENT_CUBIC[2]) / QUOTIENT_CUBIC[3]
        - x1
        - x2
    )
    result = x3, -(slope * x3 + intercept)
    if result[1] * result[1] != _cubic_value(result[0]):
        raise ArithmeticError("quotient cubic addition failed")
    return result


def quotient_multiply(index: int, point: Point = QUOTIENT_WALL_POINT) -> Point:
    if index < 0:
        if point is None:
            return None
        return quotient_multiply(-index, (point[0], -point[1]))
    result: Point = None
    while index:
        if index & 1:
            result = quotient_add(result, point)
        point = quotient_add(point, point)
        index //= 2
    return result


# Exact rank-two minimal Prym and the two generators returned by PARI effort 2.
PRYM_A2 = Q(1)
PRYM_A4 = Q(-1_088_332_527_181_202_692_852_209_675_776)
PRYM_A6 = Q(-357_197_621_008_566_738_848_718_735_836_593_172_099_301_360)
PRYM_WALL_GENERATOR: Point = (
    Q(-755_770_786_337_594_288, 1_681),
    Q(442_559_305_798_748_856_484_411_260, 68_921),
)
PRYM_HIDDEN_GENERATOR: Point = (
    Q(571_761_479_955_471_952, 361),
    Q(298_355_729_449_735_109_402_929_140, 6_859),
)
PRYM_TORSION: Point = (Q(-801_756_698_426_385), Q(0))

# Change from the minimal Prym back to the inversion cubic.
PRYM_CHANGE_U = Q(129_572_689, 3_992_146_620)
PRYM_CHANGE_R = Q(
    139_299_167_436_175_097_842_224_320_761_275_503,
    281_048_579_302_797_714_315_600,
)
PRYM_INVERSION_LEADING = Q(
    72_802_417_171_831_187_355_459_597_873_828_809_737,
    1_967_340_055_119_584_000_209_200,
)


def prym_add(left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2 and y1 == -y2:
        return None
    if left == right:
        slope = (3 * x1 * x1 + 2 * PRYM_A2 * x1 + PRYM_A4) / (2 * y1)
    else:
        slope = (y2 - y1) / (x2 - x1)
    x3 = slope * slope - PRYM_A2 - x1 - x2
    return x3, slope * (x1 - x3) - y1


def prym_multiply(index: int, point: Point) -> Point:
    if index < 0:
        if point is None:
            return None
        return prym_multiply(-index, (point[0], -point[1]))
    result: Point = None
    while index:
        if index & 1:
            result = prym_add(result, point)
        point = prym_add(point, point)
        index //= 2
    return result


def prym_lift_gates(
    wall_coefficient: int,
    hidden_coefficient: int,
    torsion_coset: bool = False,
) -> tuple[Fraction, Fraction | None, Fraction | None] | None:
    """Map a Prym lattice point to ``lambda`` and its two square gates."""

    point: Point = PRYM_TORSION if torsion_coset else None
    point = prym_add(
        point, prym_multiply(wall_coefficient, PRYM_WALL_GENERATOR)
    )
    point = prym_add(
        point, prym_multiply(hidden_coefficient, PRYM_HIDDEN_GENERATOR)
    )
    if point is None:
        return None
    minimal_x, _minimal_y = point
    raw_x = PRYM_CHANGE_U**2 * minimal_x + PRYM_CHANGE_R
    if raw_x == 0:
        return None
    height_square = PRYM_INVERSION_LEADING / raw_x
    dilation = (height_square - H * H) / A
    second_square = H * H + B * dilation
    return (
        dilation,
        rational_square_root(height_square),
        rational_square_root(second_square),
    )


def prym_shell(level: int) -> tuple[tuple[bool, int, int, Fraction, Fraction, Fraction], ...]:
    """Return full lifts on one exact rank-two coefficient shell."""

    hits = []
    seen: set[Point] = set()
    for torsion_coset in (False, True):
        for wall_coefficient, hidden_coefficient in product(
            range(-level, level + 1), repeat=2
        ):
            if max(abs(wall_coefficient), abs(hidden_coefficient)) != level:
                continue
            point: Point = PRYM_TORSION if torsion_coset else None
            point = prym_add(
                point,
                prym_multiply(wall_coefficient, PRYM_WALL_GENERATOR),
            )
            point = prym_add(
                point,
                prym_multiply(hidden_coefficient, PRYM_HIDDEN_GENERATOR),
            )
            if point in seen:
                continue
            seen.add(point)
            gates = prym_lift_gates(
                wall_coefficient, hidden_coefficient, torsion_coset
            )
            if gates is None or gates[1] is None or gates[2] is None:
                continue
            hits.append(
                (
                    torsion_coset,
                    wall_coefficient,
                    hidden_coefficient,
                    gates[0],
                    gates[1],
                    gates[2],
                )
            )
    return tuple(hits)


def main() -> None:
    print("NORM-UNIT SHELL")
    for index in range(-4, 5):
        dilation = norm_unit_dilation(index)
        print(index, dilation, height_roots(dilation))
    print("\nQUOTIENT WALL-POINT RECURRENCE")
    for index in range(1, 9):
        point = quotient_multiply(index)
        assert point is not None
        print(index, point[0], height_roots(point[0]))
    print("\nPRYM SHELLS")
    for level in (1, 2):
        print(level, prym_shell(level))


if __name__ == "__main__":
    main()
