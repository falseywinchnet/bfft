"""Exact conductor descent on the rational A2 root-lattice rank-six family.

The section at ``x=(3+2*c)r`` is rational on

    c = 50*t*(1-t^2) / (3*(t^2+2*t-1)^2).

Six-section saturation is generically successful, so the remaining problem
is purely arithmetic: make the essential discriminant current small.  Two
involutions reduce its degree-24 factor first to a degree-12 polynomial and
then to the sextic ``S6`` below.  This file records those reductions and a
small primitive support-height shell; it is not a rational-grid search.
"""

from __future__ import annotations

from argparse import ArgumentParser
from fractions import Fraction
from math import gcd

from run_weight_current_attack import build_root_lattice_candidate


Q = Fraction

# Descending coefficient order.
H24 = (
    1355211, -23389236, -30155328, 1135258956, 918075006,
    18860855364, -136940504704, -48680769852, 815682247269,
    2771064504, -2047805107008, 69154041528, 2741899123364,
    -69154041528, -2047805107008, -2771064504, 815682247269,
    48680769852, -136940504704, -18860855364, 918075006,
    -1135258956, -30155328, 23389236, 1355211,
)

P12 = (
    1355211, -23389236, -13892796, 877977360, 689703120,
    28049059584, -130499557504, 112196238336, 11035249920,
    56190551040, -3556555776, -23950577664, 5550944256,
)

S6 = (
    1355211, -23389236, -46417860, 1345762080, 1107138240,
    15642192384, -136635218944,
)


def polynomial_value(coefficients: tuple[int, ...], x: Fraction) -> Fraction:
    value = Q(0)
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def first_quotient(t: Fraction) -> Fraction:
    """Coordinate for the involution ``t -> -1/t``."""

    t = Q(t)
    if t == 0:
        raise ValueError("the first quotient has a pole at t=0")
    return t - 1 / t


def second_quotient(u: Fraction) -> Fraction:
    """Coordinate for the residual involution ``u -> 4/u``."""

    u = Q(u)
    if u == 0:
        raise ValueError("the second quotient has a pole at u=0")
    return u + 4 / u


def verify_current_reduction(t: Fraction) -> None:
    """Check ``H24=t^12 P12`` and ``P12=u^6 S6`` exactly."""

    t = Q(t)
    u = first_quotient(t)
    w = second_quotient(u)
    assert polynomial_value(H24, t) == t**12 * polynomial_value(P12, u)
    assert polynomial_value(P12, u) == u**6 * polynomial_value(S6, w)


def primitive_support_shell(height: int) -> tuple[Fraction, ...]:
    """Reduced positive ``t=m/n>1`` ordered by Pythagorean support height."""

    parameters: list[tuple[int, Fraction]] = []
    for n in range(1, height + 1):
        for m in range(n + 1, height + 1):
            if gcd(m, n) != 1:
                continue
            parameters.append((m * m + n * n, Q(m, n)))
    parameters.sort()
    return tuple(t for _, t in parameters)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--saturate-top", type=int, default=2)
    args = parser.parse_args()

    screened = []
    for t in primitive_support_shell(args.height):
        verify_current_reduction(t)
        candidate = build_root_lattice_candidate(t, saturate=False)
        screened.append((candidate.tame_conductor_lower_bound, t, candidate))
    screened.sort(key=lambda item: item[0])

    for bound, t, candidate in screened:
        print(
            f"t={t} c={candidate.c} radical={candidate.discriminant_radical} "
            f"tame-lower-bound={bound}"
        )
    print("saturated leaders")
    for _, t, _ in screened[: args.saturate_top]:
        print(build_root_lattice_candidate(t, saturate=True))


if __name__ == "__main__":
    main()
