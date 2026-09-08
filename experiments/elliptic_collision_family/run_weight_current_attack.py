"""Exact first attack on the non-root A2 weight-current surface.

The lower ``omega_2=(r+2s)/3`` interpolation reduces its rational-shape
condition to

    v^2 = 25u^4 + 20u^3 - 84u^2 + 80u + 400,
    u = t - 1/t.

The first lifted point and its first nontrivial quotient chord give
``t=1/2`` and ``t=1/3``.  Both split the residual shape quadratic as
``(c+9)(5c-4)``.  This module reconstructs and saturates those four fibers;
it does not scan rational parameters.

The next A2 orbit, ``2*omega_2=(2r+4s)/3``, has the distinct quotient

    v^2 = 16u^4 + 56u^3 + 537u^2 + 224u + 256.

Exact secants through its small support points produce the simultaneous
Pythagorean lifts ``u=3/2,8/3``, hence ``t=2,3``.  At either parameter the
remaining shape quadratic splits as ``(7c+9)(14c+5)``.  These sections are
also reconstructed and saturated below.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt, prod

import sympy as sp

from elliptic_collision_family import generalized_weierstrass_invariants
from run_rank_six_wall_descent import (
    WallFiber,
    generalized_integral_scale,
    saturate_six_sections,
)
from seven_section_surface import (
    linear_abscissa_section,
    short_discriminant,
    surface_fiber,
)


Q = Fraction
QuotientPoint = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class WeightCandidate:
    c: Fraction
    t: Fraction
    carrier_root: Fraction
    model: tuple[int, int, int, int, int]
    discriminant_factors: dict[int, int]
    discriminant_radical: int
    tame_conductor_lower_bound: int
    saturation_status: str
    saturation_trace: tuple[object, ...]


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


def dilation_minimize_generalized_model(
    model: tuple[int, int, int, int, int],
) -> tuple[tuple[int, int, int, int, int], int]:
    """Remove every integral dilation visible in ``[0,0,a3,a4,0]``.

    This is an exact global change ``x=d^2*x', y=d^3*y'``.  It is not a
    replacement for Tate's algorithm, but it prevents the conductor screen
    from counting primes introduced solely by the chosen integral
    presentation.
    """

    a1, a2, a3, a4, a6 = model
    if (a1, a2, a6) != (0, 0, 0):
        raise ValueError("the dilation minimizer expects [0,0,a3,a4,0]")
    common = gcd(abs(a3), abs(a4))
    dilation = 1
    for prime, _ in sp.factorint(common).items():
        prime = int(prime)
        v3 = 0
        v4 = 0
        left = abs(a3)
        right = abs(a4)
        while left and left % prime == 0:
            v3 += 1
            left //= prime
        while right and right % prime == 0:
            v4 += 1
            right //= prime
        dilation *= prime ** min(v3 // 3, v4 // 4)
    return (0, 0, a3 // dilation**3, a4 // dilation**4, 0), dilation


def lower_omega2_shape_residual(c: Fraction, t: Fraction) -> Fraction:
    """The opposite-sign lower-support residual quadratic."""

    return (
        5 * c * c * t**4
        + 20 * c * c * t**3
        + 10 * c * c * t * t
        - 20 * c * c * t
        + 5 * c * c
        + 5 * c * t**4
        + 14 * c * t**3
        + 10 * c * t * t
        - 14 * c * t
        + 5 * c
        + 6 * t**3
        - 6 * t
    )


def lower_omega2_carrier_root(c: Fraction, t: Fraction) -> Fraction:
    """Interpolated square root on the residual-zero surface."""

    if lower_omega2_shape_residual(c, t) != 0:
        raise ValueError("shape is not on the lower omega_2 residual")
    return c * (1 + c) * (1 - 2 * t - t * t) * (10 * c - 1) / 9


def lower_omega2_quotient(u: Fraction) -> Fraction:
    return 25 * u**4 + 20 * u**3 - 84 * u * u + 80 * u + 400


def doubled_omega2_shape_residual(c: Fraction, t: Fraction) -> Fraction:
    """Residual quadratic for the lower ``2*omega_2`` interpolation."""

    return (
        4 * c * c * t**4
        + 16 * c * c * t**3
        + 8 * c * c * t * t
        - 16 * c * c * t
        + 4 * c * c
        + 4 * c * t**4
        + 37 * c * t**3
        + 8 * c * t * t
        - 37 * c * t
        + 4 * c
        + 15 * t**3
        - 15 * t
    )


def doubled_omega2_carrier_root(c: Fraction, t: Fraction) -> Fraction:
    """Interpolated square root on the residual-zero doubled-weight surface."""

    if doubled_omega2_shape_residual(c, t) != 0:
        raise ValueError("shape is not on the lower doubled omega_2 residual")
    return c * (1 + c) * (1 - 2 * t - t * t) * (8 * c + 7) / 9


def doubled_omega2_quotient(u: Fraction) -> Fraction:
    return 16 * u**4 + 56 * u**3 + 537 * u * u + 224 * u + 256


def mixed_weight_shape_residual(c: Fraction, t: Fraction) -> Fraction:
    """Residual quadratic for ``2*omega_1+omega_2=(5r+4s)/3``."""

    return (
        c * c * t**4
        + 4 * c * c * t**3
        + 2 * c * c * t * t
        - 4 * c * c * t
        + c * c
        + c * t**4
        + 25 * c * t**3
        + 2 * c * t * t
        - 25 * c * t
        + c
        + 24 * t**3
        - 24 * t
    )


def mixed_weight_carrier_root(c: Fraction, t: Fraction) -> Fraction:
    if mixed_weight_shape_residual(c, t) != 0:
        raise ValueError("shape is not on the lower mixed-weight residual")
    return c * (1 + c) * (1 - 2 * t - t * t) * (4 * c + 11) / 9


def mixed_weight_quotient(u: Fraction) -> Fraction:
    return u**4 - 46 * u**3 + 249 * u * u - 184 * u + 16


def root_lattice_shape(t: Fraction) -> Fraction:
    """Rank-six root-lattice component at ``x=(3+2*c)*r``.

    Opposite-sign interpolation between the lower coincidences ``c=-3``
    and ``c=-4/3`` leaves a residual that is linear in ``c``.  Its
    nondegenerate component is this rational curve.
    """

    return adjacent_root_shape(Q(3), t)


def root_lattice_shape_residual(c: Fraction, t: Fraction) -> Fraction:
    """The sole residual factor of the root-lattice carrier defect."""

    c, t = map(Q, (c, t))
    return 3 * c * (t * t + 2 * t - 1) ** 2 + 50 * t * (t * t - 1)


def root_lattice_carrier_root(c: Fraction, t: Fraction) -> Fraction:
    """Interpolated square root on the rank-six root-lattice component."""

    c, t = map(Q, (c, t))
    if root_lattice_shape_residual(c, t) != 0:
        raise ValueError("shape is not on the root-lattice component")
    return adjacent_root_carrier_root(Q(3), c, t)


def adjacent_root_shape(index: Fraction, t: Fraction) -> Fraction:
    """Rational component for ``x=(index+(index-1)*c)*r``.

    This is the full adjacent-root ray in the A2 lattice.  The first useful
    member is ``index=3`` (norm 7), followed by norms 13, 21, 31, ... .
    """

    index, t = map(Q, (index, t))
    if index in (0, 2, -2):
        raise ValueError("degenerate adjacent-root index")
    denominator = index * (index - 2) * (t * t + 2 * t - 1) ** 2
    if denominator == 0:
        raise ValueError("adjacent-root shape has a pole")
    return (
        (index - 1)
        * (index + 2) ** 2
        * t
        * (1 - t * t)
        / denominator
    )


def adjacent_root_shape_residual(
    index: Fraction, c: Fraction, t: Fraction
) -> Fraction:
    index, c, t = map(Q, (index, c, t))
    return (
        c * index * (index - 2) * (t * t + 2 * t - 1) ** 2
        + (index - 1) * (index + 2) ** 2 * (t**3 - t)
    )


def adjacent_root_carrier_root(
    index: Fraction, c: Fraction, t: Fraction
) -> Fraction:
    """Opposite-sign lower-support root on the adjacent-root component."""

    index, c, t = map(Q, (index, c, t))
    if adjacent_root_shape_residual(index, c, t) != 0:
        raise ValueError("shape is not on the adjacent-root component")
    interpolation = -(
        2 * c * index * index
        - 4 * c * index
        + 2 * index * index
        - index
        - 2
    ) / (index + 2)
    return c * (1 + c) * (1 - 2 * t - t * t) * interpolation


def pythagorean_lift(u: Fraction) -> tuple[Fraction, Fraction] | None:
    root = rational_square_root(u * u + 4)
    if root is None:
        return None
    return (u + root) / 2, (u - root) / 2


def quotient_chord(left: QuotientPoint, right: QuotientPoint) -> QuotientPoint:
    """Third intersection with ``v=5u^2+B*u+C`` on the quotient quartic."""

    x1, y1 = left
    x2, y2 = right
    if x1 == x2:
        raise ValueError("use distinct quotient points for this chord")
    slope = ((y2 - 5 * x2 * x2) - (y1 - 5 * x1 * x1)) / (x2 - x1)
    intercept = y1 - 5 * x1 * x1 - slope * x1
    cubic = 20 - 10 * slope
    quadratic = -84 - slope * slope - 10 * intercept
    if cubic == 0:
        raise ValueError("the quotient chord meets the selected infinity")
    x3 = -quadratic / cubic - x1 - x2
    y3 = 5 * x3 * x3 + slope * x3 + intercept
    assert y3 * y3 == lower_omega2_quotient(x3)
    return x3, y3


def doubled_quotient_chord(
    left: QuotientPoint, right: QuotientPoint
) -> QuotientPoint:
    """Third intersection with ``v=4u^2+B*u+C`` on the doubled quotient."""

    x1, y1 = left
    x2, y2 = right
    if x1 == x2:
        raise ValueError("use distinct quotient points for this chord")
    slope = ((y2 - 4 * x2 * x2) - (y1 - 4 * x1 * x1)) / (x2 - x1)
    intercept = y1 - 4 * x1 * x1 - slope * x1
    cubic = 56 - 8 * slope
    quadratic = 537 - slope * slope - 8 * intercept
    if cubic == 0:
        raise ValueError("the quotient chord meets the selected infinity")
    x3 = -quadratic / cubic - x1 - x2
    y3 = 4 * x3 * x3 + slope * x3 + intercept
    assert y3 * y3 == doubled_omega2_quotient(x3)
    return x3, y3


def _build_linear_candidate(
    c: Fraction,
    t: Fraction,
    a: Fraction,
    b: Fraction,
    root: Fraction,
    saturate: bool = True,
) -> WeightCandidate:
    fiber = surface_fiber(c, t)
    extra = linear_abscissa_section(c, t, a, b, root)
    center = fiber.center_height
    generalized_points = tuple(
        (x, y - center / 2) for x, y in fiber.seven_points()
    ) + ((extra[0], extra[1] - center / 2),)
    scale = generalized_integral_scale(fiber.m, center, generalized_points)
    a3 = int(center * scale**3)
    a4 = int(-fiber.m * scale**4)
    discriminant = int(short_discriminant(fiber) * scale**12)
    short_model = (0, 16 * a4, 16 * a3 * a3)
    short_points = tuple(
        (
            int(4 * x * scale**2),
            int(4 * (2 * y * scale**3 + a3)),
        )
        for x, y in generalized_points
    )
    wall_fiber = WallFiber(
        branch=0,
        slope=Q(0),
        c=c,
        scale=scale,
        a3=a3,
        a4=a4,
        discriminant=discriminant,
        independent_indices=None,
        short_model=short_model,
        short_points=short_points,
    )
    if saturate:
        status, _generators, trace = saturate_six_sections(wall_fiber)
    else:
        status, trace = "unchecked", ()
    model, _dilation = dilation_minimize_generalized_model(
        (0, 0, a3, a4, 0)
    )
    c4, _c6, checked_discriminant = generalized_weierstrass_invariants(model)
    factors = {
        int(prime): int(exponent)
        for prime, exponent in sp.factorint(abs(checked_discriminant)).items()
    }
    radical = prod(factors)
    tame_lower_bound = radical * prod(
        prime for prime in factors if prime >= 5 and c4 % prime == 0
    )
    return WeightCandidate(
        c=c,
        t=t,
        carrier_root=root,
        model=model,
        discriminant_factors=factors,
        discriminant_radical=radical,
        tame_conductor_lower_bound=tame_lower_bound,
        saturation_status=status,
        saturation_trace=trace,
    )


def build_candidate(c: Fraction, t: Fraction) -> WeightCandidate:
    root = lower_omega2_carrier_root(c, t)
    return _build_linear_candidate(c, t, Q(1, 3), Q(2, 3), root)


def build_doubled_candidate(c: Fraction, t: Fraction) -> WeightCandidate:
    root = doubled_omega2_carrier_root(c, t)
    return _build_linear_candidate(c, t, Q(2, 3), Q(4, 3), root)


def build_mixed_candidate(c: Fraction, t: Fraction) -> WeightCandidate:
    root = mixed_weight_carrier_root(c, t)
    return _build_linear_candidate(c, t, Q(5, 3), Q(4, 3), root)


def build_root_lattice_candidate(
    t: Fraction, saturate: bool = True
) -> WeightCandidate:
    """Build the generic rank-six section at ``x=(3+2*c)*r``."""

    t = Q(t)
    c = root_lattice_shape(t)
    root = root_lattice_carrier_root(c, t)
    return _build_linear_candidate(c, t, Q(3), Q(2), root, saturate)


def build_adjacent_root_candidate(
    index: Fraction, t: Fraction, saturate: bool = True
) -> WeightCandidate:
    index, t = map(Q, (index, t))
    c = adjacent_root_shape(index, t)
    root = adjacent_root_carrier_root(index, c, t)
    return _build_linear_candidate(c, t, index, index - 1, root, saturate)


def first_lifted_parameters() -> tuple[Fraction, ...]:
    first = (Q(-3, 2), Q(49, 4))
    assert first[1] * first[1] == lower_omega2_quotient(first[0])
    second = quotient_chord(first, (Q(0), Q(20)))
    assert second == (Q(-8, 3), Q(196, 9))
    parameters = []
    for point in (first, second):
        lift = pythagorean_lift(point[0])
        assert lift is not None
        parameters.extend(value for value in lift if 0 < value < 1)
    return tuple(parameters)


def first_doubled_lifted_parameters() -> tuple[Fraction, ...]:
    """Two lifts obtained by exact quotient secants, without a parameter scan."""

    first = doubled_quotient_chord((Q(-4), Q(-92)), (Q(0), Q(-16)))
    assert first == (Q(3, 2), Q(91, 2))
    second = doubled_quotient_chord((Q(-1), Q(23)), (Q(0), Q(-16)))
    assert second == (Q(8, 3), Q(-728, 9))
    parameters = []
    for point in (first, second):
        lift = pythagorean_lift(point[0])
        assert lift is not None
        parameters.extend(value for value in lift if value > 0)
    return tuple(parameters)


def main() -> None:
    print("fundamental omega_2")
    for t in first_lifted_parameters():
        for c in (Q(-9), Q(4, 5)):
            candidate = build_candidate(c, t)
            print(candidate)
    print("doubled omega_2")
    for t in first_doubled_lifted_parameters():
        for c in (Q(-9, 7), Q(-5, 14)):
            candidate = build_doubled_candidate(c, t)
            print(candidate)


if __name__ == "__main__":
    main()
