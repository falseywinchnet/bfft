"""The rank-two ``d=37`` transport and its joint invisible-current cover.

The norm-current chain ``7 -> 157 -> 37`` reaches the smallest arithmetic
fiber found by the conductor-first attack.  With

    m = 3*37^2,
    A = 12100,
    B = 190512,

the simultaneous support heights ``u^2=c^2+A`` and ``v^2=c^2+B`` form an
exact rank-two elliptic curve.  Its first coefficient shell contains target
ranks ``4,5,5,6``.  The rank-four member has conductor 1,795,858,911, below
the current rank-six record, but lies on a visible rank-loss wall.

Two non-support basis points on that fiber occur at ``x=4`` and ``x=70``.
Their currents are

    D4  = 84942,
    D70 = 156816.

Preserving both currents first gives another exact rank-two elliptic curve.
The base fiber is one of its primitive generators.  Imposing the two support
squares on this quotient is therefore the smallest exact remaining problem:
any non-base lift escapes the visible wall while retaining both invisible
directions, and is a structural rank-six candidate.

This module contains no height or curve search.  It implements the two group
laws, their exact maps to ``c``, and finite-field Mordell--Weil sieves used to
order the next coefficient vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

try:
    from .elliptic_collision_family import add_points
except ImportError:  # Direct execution from this directory.
    from elliptic_collision_family import add_points


Q = Fraction
Point = tuple[Fraction, Fraction]
ProjectivePoint = Point | None
FinitePoint = tuple[int, int]
FiniteProjectivePoint = FinitePoint | None

DIRECTION = 37
M = 3 * DIRECTION**2
NODAL_CURRENT = 2 * DIRECTION**3
SUPPORT_CURRENT = 89_206
SUPPORT_SQUARE_A = NODAL_CURRENT - SUPPORT_CURRENT  # 12100 = 110^2
SUPPORT_SQUARE_B = NODAL_CURRENT + SUPPORT_CURRENT  # 190512

PARAMETER_MODEL = (0, 1, 0, -711_167_436, 7_193_953_224_864)
PARAMETER_CONDUCTOR = 412_131_720
PARAMETER_GENERATORS = (
    (Q(-15_516), Q(3_807_000)),
    (Q(223_461, 4), Q(95_270_175, 8)),
)
PARAMETER_GROUP_MODEL = (Q(1), Q(-711_167_436), Q(7_193_953_224_864))


@dataclass(frozen=True)
class FirstShellFiber:
    coefficients: tuple[int, int]
    height: Fraction
    minimal_model: tuple[int, int, int, int, int]
    conductor: int
    rank: int


FIRST_SHELL = (
    FirstShellFiber(
        (0, 1),
        Q(231, 2),
        (0, 0, 1, -4_107, 114_646),
        1_795_858_911,
        4,
    ),
    FirstShellFiber(
        (1, 0),
        Q(2_379, 10),
        (0, 0, 1, -2_566_875, 2_467_225_156),
        13_874_778_166_275,
        5,
    ),
    FirstShellFiber(
        (1, 1),
        Q(50_589, 68),
        (0, 0, 0, -5_488_331_952, 1_011_501_404_522_660),
        204_456_469_404_651_156,
        5,
    ),
    FirstShellFiber(
        (1, -1),
        Q(1_361_899, 28_272),
        (
            0,
            0,
            1,
            -640_605_825_251_723_952,
            201_869_167_863_825_640_662_818_726,
        ),
        18_806_397_523_562_338_875_666_045_873,
        6,
    ),
)

BASE_HEIGHT = Q(231, 2)
BASE_SHORT_A6 = BASE_HEIGHT**2 + NODAL_CURRENT
BASE_SHORT_POINTS = (
    (Q(4), Q(627, 2)),
    (Q(26), Q(319, 2)),
    (Q(37), Q(231, 2)),
    (Q(70), Q(825, 2)),
)
CHORD_THIRD_POINT = (Q(-287, 4), Q(1_599, 8))

INVISIBLE_CURRENT_4 = 84_942
INVISIBLE_CURRENT_70 = 156_816
JOINT_LEFT = INVISIBLE_CURRENT_4
JOINT_RIGHT = INVISIBLE_CURRENT_70
JOINT_QUARTIC_MIDDLE = 4 * JOINT_RIGHT - 2 * JOINT_LEFT  # 457380
JOINT_MODEL = (
    0,
    JOINT_QUARTIC_MIDDLE,
    0,
    -4 * JOINT_LEFT**2,
    -4 * JOINT_QUARTIC_MIDDLE * JOINT_LEFT**2,
)
JOINT_CONDUCTOR = 82_368
JOINT_GENERATORS = (
    (Q(-387_684), Q(91_998_720)),
    (Q(-339_768), Q(100_911_096)),
)
JOINT_TORSION = (
    (Q(-JOINT_QUARTIC_MIDDLE), Q(0)),
    (Q(2 * JOINT_LEFT), Q(0)),
    (Q(-2 * JOINT_LEFT), Q(0)),
)
JOINT_GROUP_MODEL = (
    Q(JOINT_QUARTIC_MIDDLE),
    Q(-4 * JOINT_LEFT**2),
    Q(-4 * JOINT_QUARTIC_MIDDLE * JOINT_LEFT**2),
)
JOINT_BASE_POINT = (Q(405_108), Q(341_545_248))

LOCAL_SIEVE_PRIMES = (31, 37, 41, 47, 73, 107, 157, 239, 503)


def _multiply(
    coefficient: int,
    point: ProjectivePoint,
    model: tuple[Fraction, Fraction, Fraction],
) -> ProjectivePoint:
    if coefficient < 0:
        if point is None:
            return None
        return _multiply(-coefficient, (point[0], -point[1]), model)
    result: ProjectivePoint = None
    addend = point
    while coefficient:
        if coefficient & 1:
            result = add_points(result, addend, model)
        addend = add_points(addend, addend, model)
        coefficient //= 2
    return result


def parameter_point(coefficients: tuple[int, int]) -> ProjectivePoint:
    result: ProjectivePoint = None
    for coefficient, generator in zip(coefficients, PARAMETER_GENERATORS):
        result = add_points(
            result,
            _multiply(coefficient, generator, PARAMETER_GROUP_MODEL),
            PARAMETER_GROUP_MODEL,
        )
    return result


def parameter_height(point: Point) -> Fraction:
    """Map the minimal parameter curve to the support height ``c``."""

    x, y = point
    raw_u = 16 * x - 245_944
    raw_v = 64 * y
    scale = Q(24_200)
    denominator = raw_u * raw_u - scale * scale
    if denominator == 0:
        raise ValueError("parameter point is a height pole")
    p = raw_v * scale / denominator
    if p == 0:
        raise ValueError("parameter point is a height pole")
    return (p - Q(SUPPORT_SQUARE_A, p)) / 2


def target_rhs(x: Fraction, height: Fraction = BASE_HEIGHT) -> Fraction:
    return x**3 - M * x + height**2 + NODAL_CURRENT


def verify_target_point(point: Point, height: Fraction = BASE_HEIGHT) -> bool:
    x, y = point
    return y * y == target_rhs(x, height)


def chord_residual(index: Fraction) -> Fraction:
    """Residual of the line through the two non-support basis points."""

    x = 66 * index - 128
    y = BASE_HEIGHT + 99 * index
    return y * y - target_rhs(x)


def chord_factor(index: Fraction) -> Fraction:
    return -3_267 * (index - 3) * (index - 2) * (88 * index - 75)


def joint_point(coefficients: tuple[int, int]) -> ProjectivePoint:
    result: ProjectivePoint = None
    for coefficient, generator in zip(coefficients, JOINT_GENERATORS):
        result = add_points(
            result,
            _multiply(coefficient, generator, JOINT_GROUP_MODEL),
            JOINT_GROUP_MODEL,
        )
    return result


def joint_height(point: Point) -> Fraction:
    """Map the joint invisible-current curve back to ``|c|``."""

    x, y = point
    u = x / (2 * JOINT_LEFT)
    denominator = u * u - 1
    if denominator == 0:
        raise ValueError("joint point is a height pole")
    p = (y / (2 * JOINT_LEFT)) / denominator
    if p == 0:
        raise ValueError("joint point is a height pole")
    return abs((p - Q(JOINT_LEFT, p)) / 2)


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


def is_full_joint_lift(coefficients: tuple[int, int]) -> bool:
    point = joint_point(coefficients)
    if point is None:
        return False
    try:
        height = joint_height(point)
    except ValueError:
        return False
    return (
        rational_square_root(height * height + SUPPORT_SQUARE_A) is not None
        and rational_square_root(height * height + SUPPORT_SQUARE_B) is not None
    )


def _finite_inverse(value: int, modulus: int) -> int:
    return pow(value % modulus, -1, modulus)


def _finite_add(
    left: FiniteProjectivePoint,
    right: FiniteProjectivePoint,
    modulus: int,
) -> FiniteProjectivePoint:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left[0] % modulus, left[1] % modulus
    x2, y2 = right[0] % modulus, right[1] % modulus
    if x1 == x2 and (y1 + y2) % modulus == 0:
        return None
    if x1 == x2:
        slope = (
            (3 * x1 * x1 + 2 * JOINT_QUARTIC_MIDDLE * x1 - 4 * JOINT_LEFT**2)
            * _finite_inverse(2 * y1, modulus)
        ) % modulus
    else:
        slope = ((y2 - y1) * _finite_inverse(x2 - x1, modulus)) % modulus
    x3 = (slope * slope - JOINT_QUARTIC_MIDDLE - x1 - x2) % modulus
    return x3, (-y1 + slope * (x1 - x3)) % modulus


def _finite_multiply(
    coefficient: int, point: FiniteProjectivePoint, modulus: int
) -> FiniteProjectivePoint:
    if coefficient < 0:
        if point is None:
            return None
        return _finite_multiply(-coefficient, (point[0], -point[1]), modulus)
    result: FiniteProjectivePoint = None
    addend = point
    while coefficient:
        if coefficient & 1:
            result = _finite_add(result, addend, modulus)
        addend = _finite_add(addend, addend, modulus)
        coefficient //= 2
    return result


def _finite_height(
    point: FiniteProjectivePoint, modulus: int
) -> int | None:
    """Return the finite reduction of ``c``; ``None`` means a height pole."""

    if point is None:
        return None
    x, y = point
    denominator = (x * x - (2 * JOINT_LEFT) ** 2) % modulus
    if denominator == 0:
        return None
    p = (
        2 * JOINT_LEFT * y * _finite_inverse(denominator, modulus)
    ) % modulus
    if p == 0:
        return None
    return (
        (p - JOINT_LEFT * _finite_inverse(p, modulus))
        * _finite_inverse(2, modulus)
    ) % modulus


def _finite_square(value: int, modulus: int) -> bool:
    value %= modulus
    return value == 0 or pow(value, (modulus - 1) // 2, modulus) == 1


def local_lift_status(coefficients: tuple[int, int], prime: int) -> str:
    """Classify the two missing support squares modulo a good prime.

    ``S`` and ``N`` denote square and nonsquare.  A height pole is retained,
    since negative valuation can make both rational square conditions locally
    soluble after clearing the denominator.
    """

    point: FiniteProjectivePoint = None
    for coefficient, generator in zip(coefficients, JOINT_GENERATORS):
        reduced = (int(generator[0]) % prime, int(generator[1]) % prime)
        point = _finite_add(
            point, _finite_multiply(coefficient, reduced, prime), prime
        )
    height = _finite_height(point, prime)
    if height is None:
        return "pole"
    return "".join(
        "S" if _finite_square(height * height + current, prime) else "N"
        for current in (SUPPORT_SQUARE_A, SUPPORT_SQUARE_B)
    )


def passes_local_sieve(
    coefficients: tuple[int, int], primes: tuple[int, ...] = LOCAL_SIEVE_PRIMES
) -> bool:
    return all("N" not in local_lift_status(coefficients, prime) for prime in primes)


def main() -> None:
    print("parameter:", PARAMETER_MODEL, "N=", PARAMETER_CONDUCTOR)
    for fiber in FIRST_SHELL:
        point = parameter_point(fiber.coefficients)
        assert point is not None
        print(
            fiber.coefficients,
            "c=", abs(parameter_height(point)),
            "rank=", fiber.rank,
            "N=", fiber.conductor,
        )
    print("chord third point:", CHORD_THIRD_POINT)
    print("joint quotient:", JOINT_MODEL, "N=", JOINT_CONDUCTOR)
    print("joint base height:", joint_height(JOINT_BASE_POINT))
    for coefficients in ((0, 1), (18, 31), (120, 90)):
        print(
            coefficients,
            tuple(
                (prime, local_lift_status(coefficients, prime))
                for prime in LOCAL_SIEVE_PRIMES
            ),
        )


if __name__ == "__main__":
    main()
