"""Directed conductor descent on the compatible rank-six current family.

This is deliberately not a coefficient-box search.  On the ``t=1/2`` slice,
the primitive-support and extra-section square currents have the common conic

    X^2 + 47 V^2 = 1128.

Lines of slope ``k`` through ``(47/2, 7/2)`` give

    c = 2 (k^2 + 47)^2 /
        (k^4 + 84 k^3 + 118 k^2 - 3948 k + 2209).

The script isolates the pullback of the discriminant wall and evaluates only
continued-fraction convergents to its real branches.  Every tested fiber is
therefore a best rational approximation to a vanishing current.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations
from math import floor, isqrt
from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / "elliptic_rank_support"
sys.path.insert(0, str(SUPPORT))

from elliptic_rank_support import add_points, reduction_dependency_masks  # noqa: E402
from run_factor_screen import halve_point  # noqa: E402
from seven_section_surface import (  # noqa: E402
    extra_section_r_minus_s,
    extra_section_r_minus_s_conic,
    short_discriminant,
    surface_fiber,
)


Q = Fraction
RANK_SIX_CONDUCTOR_TARGET = 5_187_563_742
GOOD_PRIMES = (
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
)


@dataclass(frozen=True)
class WallFiber:
    branch: int
    slope: Fraction
    c: Fraction
    scale: int
    a3: int
    a4: int
    discriminant: int
    independent_indices: tuple[int, ...] | None
    short_model: tuple[int, int, int]
    short_points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CurrentSlice:
    name: str
    t: Fraction
    norm_coefficient: int
    seed_x: Fraction
    seed_v: Fraction


CURRENT_SLICES = {
    "half": CurrentSlice("half", Q(1, 2), 47, Q(47, 2), Q(7, 2)),
    "quarter": CurrentSlice("quarter", Q(1, 4), 431, Q(431, 5), Q(13, 5)),
    "fifth": CurrentSlice("fifth", Q(1, 5), 764, Q(382, 3), Q(5, 3)),
    "sixth": CurrentSlice("sixth", Q(1, 6), 1151, Q(1151, 7), Q(5, 7)),
    "seventh": CurrentSlice("seventh", Q(1, 7), 1532, Q(383, 2), Q(1, 4)),
    "eighth": CurrentSlice("eighth", Q(1, 8), 1823, Q(1823, 9), Q(11, 9)),
    "ninth": CurrentSlice("ninth", Q(1, 9), 1916, Q(958, 5), Q(11, 5)),
    "tenth": CurrentSlice("tenth", Q(1, 10), 1679, Q(1679, 14), Q(55, 14)),
    "eleventh": CurrentSlice("eleventh", Q(1, 11), 956, Q(478, 5), Q(19, 5)),
    "twelfth": CurrentSlice("twelfth", Q(1, 12), -433, Q(433, 2), Q(23, 2)),
    "thirteenth": CurrentSlice(
        "thirteenth", Q(1, 13), -2692, Q(1346, 7), Q(43, 7)
    ),
    "fourteenth": CurrentSlice(
        "fourteenth", Q(1, 14), -6049, Q(6049, 7), Q(85, 7)
    ),
    "fifteenth": CurrentSlice(
        "fifteenth", Q(1, 15), -10756, Q(2689, 4), Q(65, 8)
    ),
    "sixteenth": CurrentSlice(
        "sixteenth", Q(1, 16), -17089, Q(17089, 17), Q(155, 17)
    ),
    "seventeenth": CurrentSlice(
        "seventeenth", Q(1, 17), -25348, Q(12674, 9), Q(91, 9)
    ),
    "eighteenth": CurrentSlice(
        "eighteenth", Q(1, 18), -35857, Q(35857, 19), Q(211, 19)
    ),
    "nineteenth": CurrentSlice(
        "nineteenth", Q(1, 19), -48964, Q(12241, 5), Q(121, 10)
    ),
    "twentieth": CurrentSlice(
        "twentieth", Q(1, 20), -65041, Q(65041, 55), Q(371, 55)
    ),
}


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


def compatible_shape(
    slope: Fraction, current_slice: CurrentSlice = CURRENT_SLICES["half"]
) -> Fraction:
    """Return ``c`` on the simultaneous support/extra-section current conic."""

    k = Q(slope)
    norm = current_slice.norm_coefficient
    x0, v0 = current_slice.seed_x, current_slice.seed_v
    denominator = k * k + norm
    v = 2 * k * (k * v0 - x0) / denominator - v0
    t = current_slice.t
    transport = t * (1 - t * t)
    area = (1 + t * t) ** 2
    shape_denominator = transport * v * v - area - 12 * transport
    if shape_denominator == 0:
        raise ZeroDivisionError("current parameter is at infinity")
    return (area - 12 * transport) / shape_denominator


def _factor_denominator(value: int) -> dict[int, int]:
    return {int(prime): int(exponent) for prime, exponent in sp.factorint(value).items()}


def generalized_integral_scale(
    m: Fraction,
    q: Fraction,
    points: tuple[tuple[Fraction, Fraction], ...],
) -> int:
    """Smallest integer dilation for ``y^2+q*y=x^3-m*x`` and its points."""

    weighted = [(m, 4), (q, 3)]
    weighted.extend((x, 2) for x, _ in points)
    weighted.extend((y, 3) for _, y in points)
    requirements: dict[int, int] = {}
    for value, weight in weighted:
        for prime, exponent in _factor_denominator(value.denominator).items():
            needed = (exponent + weight - 1) // weight
            requirements[prime] = max(requirements.get(prime, 0), needed)
    scale = 1
    for prime, exponent in requirements.items():
        scale *= prime**exponent
    return scale


def wall_fiber(
    branch: int,
    slope: Fraction,
    current_slice: CurrentSlice = CURRENT_SLICES["half"],
    certify_independence: bool = False,
) -> WallFiber | None:
    """Build and certify the exact fiber at one wall convergent."""

    t = current_slice.t
    c = compatible_shape(slope, current_slice)
    support_root = rational_square_root(t * (1 - t * t) * c * (c + 1))
    extra_root = rational_square_root(extra_section_r_minus_s_conic(c, t))
    if support_root in (None, 0) or extra_root is None:
        raise AssertionError("the compatible-current parametrization lost a square")
    fiber = surface_fiber(c, t, 1 / support_root)
    extra = extra_section_r_minus_s(c, t, extra_root, 1 / support_root)

    q = fiber.center_height
    generalized_points = tuple(
        (x, y - q / 2) for x, y in fiber.seven_points()
    ) + ((extra[0], extra[1] - q / 2),)
    if len(set(generalized_points)) < 8:
        return None

    scale = generalized_integral_scale(fiber.m, q, generalized_points)
    a3 = int(q * scale**3)
    a4 = int(-fiber.m * scale**4)
    discriminant = int(short_discriminant(fiber) * scale**12)

    # The integral generalized model is [0,0,a3,a4,0].  For the existing
    # exact mod-2 oracle use X=4*x and Y=4*(2*y+a3), which gives the monic
    # short model Y^2=X^3+16*a4*X+16*a3^2.
    short_model = (0, 16 * a4, 16 * a3 * a3)
    short_points = tuple(
        (
            int(4 * x * scale**2),
            int(4 * (2 * y * scale**3 + a3)),
        )
        for x, y in generalized_points
    )
    independent = None
    if certify_independence:
        for indices in combinations(range(8), 6):
            selected = tuple(short_points[index] for index in indices)
            if not reduction_dependency_masks(
                short_model, selected, primes=GOOD_PRIMES, count=6
            ):
                independent = indices
                break
    return WallFiber(
        branch=branch,
        slope=slope,
        c=c,
        scale=scale,
        a3=a3,
        a4=a4,
        discriminant=discriminant,
        independent_indices=independent,
        short_model=short_model,
        short_points=short_points,
    )


def saturate_six_sections(
    fiber: WallFiber,
    indices: tuple[int, ...] = (0, 1, 2, 4, 5, 7),
) -> tuple[str, tuple[tuple[Fraction, Fraction], ...], tuple[object, ...]]:
    """Saturate the geometric six-section basis at 2.

    A mod-2 dependency is not a rank failure when its sum is twice another
    rational point.  Replace one generator by that exact half and repeat.
    """

    if len(indices) != 6:
        raise ValueError("six section indices are required")
    generators = [
        (Q(fiber.short_points[index][0]), Q(fiber.short_points[index][1]))
        for index in indices
    ]
    saturation_primes = tuple(int(prime) for prime in sp.primerange(5, 1000))
    halvings: list[object] = []
    seen_states = set()
    for _ in range(16):
        state = tuple(generators)
        if state in seen_states:
            return "dependent", state, tuple(halvings + [("cycle", None)])
        seen_states.add(state)
        dependencies = reduction_dependency_masks(
            fiber.short_model,
            tuple(generators),
            primes=saturation_primes,
            count=6,
        )
        if not dependencies:
            return "rank6", tuple(generators), tuple(halvings)
        progressed = False
        for mask in sorted(dependencies):
            total = None
            selected = []
            for index, generator in enumerate(generators):
                if mask & (1 << index):
                    total = add_points(total, generator, fiber.short_model)
                    selected.append(index)
            if total is None:
                return "dependent", tuple(generators), tuple(halvings + [(mask, None)])
            half = halve_point(total, fiber.short_model)
            if half is None:
                continue
            negative_half = (half[0], -half[1])
            if half in generators or negative_half in generators:
                return (
                    "dependent",
                    tuple(generators),
                    tuple(halvings + [(mask, half)]),
                )
            generators[selected[0]] = half
            halvings.append((mask, half))
            progressed = True
            break
        if not progressed:
            return "unresolved", tuple(generators), tuple(halvings)
    return "unresolved", tuple(generators), tuple(halvings)


def _sympy_rational(value: Fraction) -> sp.Rational:
    return sp.Rational(value.numerator, value.denominator)


def discriminant_wall_polynomial(
    current_slice: CurrentSlice = CURRENT_SLICES["half"],
) -> sp.Poly:
    k = sp.symbols("k")
    norm = current_slice.norm_coefficient
    x0 = _sympy_rational(current_slice.seed_x)
    v0 = _sympy_rational(current_slice.seed_v)
    v = 2 * k * (k * v0 - x0) / (k * k + norm) - v0
    t = _sympy_rational(current_slice.t)
    transport = t * (1 - t * t)
    area = (1 + t * t) ** 2
    c = (area - 12 * transport) / (transport * v * v - area - 12 * transport)
    m = 1 + c + c * c
    q_squared = area * c * (c + 1) / transport
    wall = 64 * m**3 - 27 * q_squared**2
    numerator = sp.factor(sp.together(wall).as_numer_denom()[0])
    return sp.Poly(numerator, k)


def real_wall_roots(
    current_slice: CurrentSlice = CURRENT_SLICES["half"], digits: int = 80
) -> tuple[Decimal, ...]:
    roots = sp.nroots(
        discriminant_wall_polynomial(current_slice), n=digits, maxsteps=300
    )
    result = []
    for root in roots:
        real, imaginary = root.as_real_imag()
        if abs(imaginary) < sp.Float(10) ** (-(digits // 2)):
            result.append(Decimal(str(real)))
    return tuple(sorted(result))


def continued_fraction_convergents(
    value: Decimal, maximum_denominator: int
) -> tuple[Fraction, ...]:
    """Return the characteristic rational approximants to one real branch."""

    p_minus_two, p_minus_one = 0, 1
    q_minus_two, q_minus_one = 1, 0
    convergents: list[Fraction] = []
    with localcontext() as context:
        context.prec = 90
        current = +value
        for _ in range(200):
            coefficient = floor(current)
            p = coefficient * p_minus_one + p_minus_two
            q = coefficient * q_minus_one + q_minus_two
            if q > maximum_denominator:
                break
            convergents.append(Q(p, q))
            remainder = current - coefficient
            if not remainder:
                break
            current = 1 / remainder
            p_minus_two, p_minus_one = p_minus_one, p
            q_minus_two, q_minus_one = q_minus_one, q
    return tuple(convergents)


def descent(
    maximum_denominator: int,
    certify_count: int = 20,
    current_slice: CurrentSlice = CURRENT_SLICES["half"],
) -> tuple[WallFiber, ...]:
    candidates: dict[Fraction, int] = {}
    for branch, root in enumerate(real_wall_roots(current_slice)):
        for slope in continued_fraction_convergents(root, maximum_denominator):
            candidates.setdefault(slope, branch)
    fibers = []
    for slope, branch in candidates.items():
        fiber = wall_fiber(branch, slope, current_slice)
        if fiber is not None:
            fibers.append(fiber)
    fibers.sort(key=lambda item: abs(item.discriminant))
    for index, fiber in enumerate(fibers[:certify_count]):
        certified = wall_fiber(
            fiber.branch,
            fiber.slope,
            current_slice,
            certify_independence=True,
        )
        assert certified is not None
        fibers[index] = certified
    return tuple(fibers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-denominator", type=int, default=200)
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--certify-count", type=int, default=20)
    parser.add_argument("--slice", choices=tuple(CURRENT_SLICES), default="half")
    args = parser.parse_args()
    current_slice = CURRENT_SLICES[args.slice]
    fibers = descent(args.max_denominator, args.certify_count, current_slice)
    print("slice=", current_slice.name)
    print("wall branches=", len(real_wall_roots(current_slice)))
    print("noncollision convergents=", len(fibers))
    for fiber in fibers[: args.show]:
        print(
            "branch=", fiber.branch,
            "k=", fiber.slope,
            "c=", fiber.c,
            "scale=", fiber.scale,
            "a3=", fiber.a3,
            "a4=", fiber.a4,
            "Delta=", fiber.discriminant,
            "rank6=", fiber.independent_indices,
            "sub_target_discriminant=",
            abs(fiber.discriminant) < RANK_SIX_CONDUCTOR_TARGET,
        )


if __name__ == "__main__":
    main()
