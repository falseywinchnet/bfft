"""First exact split-support lift shell on the rank-surplus ``m=481`` wall.

The rank-five wall

    y^2 = x^3 - 481*x + 3961

has three integral horizontal supports, at currents ``-3600,-2280,2280``.
Its support-reservoir Jacobian has exact rank four, one more than the visible
current count.  The four directions below are the two other supports and the
two smallest height-reflection directions from the integral deck.

The adjacent ``{-1,0,1}`` coefficient shell is finite.  Each reservoir point
is accepted only when PARI factors ``x^3-481*x-K`` into three rational linear
factors.  Thus square-discriminant cyclic cubics never masquerade as rational
support triples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product

try:
    from .run_support_reservoir_attack import QuarticGroup
    from .run_three_current_height_conic import (
        ThreeCurrentHeightFiber,
        build_three_current_height_fiber,
        support_current,
    )
except ImportError:  # Direct execution from this directory.
    from run_support_reservoir_attack import QuarticGroup
    from run_three_current_height_conic import (
        ThreeCurrentHeightFiber,
        build_three_current_height_fiber,
        support_current,
    )


Q = Fraction
M = 481
A6 = 3_961
ORIGIN = (Q(19), Q(9_758))
GROUP = QuarticGroup(
    (
        Q(4 * M**3 - 27 * A6**2),
        Q(0),
        Q(54 * A6),
        Q(0),
        Q(-27),
    ),
    ORIGIN,
)
DIRECTIONS = (
    (Q(41), Q(17_458)),
    (Q(79), Q(17_458)),
    (Q(-19), Q(9_758)),
    (Q(-41), Q(17_458)),
)
SPLIT_SUPPORTS = tuple(
    sorted(
        (
            (
                (Q(2_441, 169), Q(1_824, 169), Q(-4_265, 169)),
                Q(11_383, 2_197),
                "K13",
            ),
            ((Q(-25), Q(9), Q(16)), Q(19), "K-3600"),
            ((Q(-24), Q(5), Q(19)), Q(41), "K-2280"),
            ((Q(-19), Q(-5), Q(24)), Q(79), "K2280"),
            (
                (Q(34_176, 1_369), Q(-12_041, 1_369), Q(-22_135, 1_369)),
                Q(4_389_953, 50_653),
                "K37",
            ),
        ),
        key=lambda support: support_current(support[0]),
    )
)


@dataclass(frozen=True)
class LiftCandidate:
    coefficients: tuple[int, int, int, int]
    height: Fraction
    discriminant_root: Fraction

    @property
    def current(self) -> Fraction:
        return self.height * self.height - A6

    @property
    def parameter_bits(self) -> int:
        return max(
            self.height.numerator.bit_length(),
            self.height.denominator.bit_length(),
            self.discriminant_root.numerator.bit_length(),
            self.discriminant_root.denominator.bit_length(),
        )


def adjacent_shell() -> tuple[LiftCandidate, ...]:
    candidates = []
    seen = set()
    for coefficients in product((-1, 0, 1), repeat=len(DIRECTIONS)):
        if not any(coefficients):
            continue
        point = ORIGIN
        try:
            for coefficient, direction in zip(coefficients, DIRECTIONS):
                point = GROUP.add(point, GROUP.multiply(coefficient, direction))
        except ArithmeticError:
            continue
        if point in seen:
            continue
        seen.add(point)
        candidates.append(LiftCandidate(coefficients, point[0], point[1]))
    return tuple(sorted(candidates, key=lambda candidate: candidate.parameter_bits))


def _gp_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def gp_split_program() -> str:
    candidates = adjacent_shell()
    currents = ",".join(_gp_fraction(candidate.current) for candidate in candidates)
    labels = ",".join(
        '"' + ",".join(map(str, candidate.coefficients)) + '"'
        for candidate in candidates
    )


def primitive_farey_neighbors(slope: Fraction) -> tuple[Fraction, Fraction]:
    numerator, denominator = slope.numerator, slope.denominator
    if denominator == 1:
        return Q(numerator - 1), Q(numerator + 1)
    neighbors = []
    for determinant in (1, -1):
        neighbor_denominator = (
            determinant * pow(numerator, -1, denominator)
        ) % denominator
        if neighbor_denominator == 0:
            neighbor_denominator = denominator
        neighbor_numerator = (
            numerator * neighbor_denominator - determinant
        ) // denominator
        neighbors.append(Q(neighbor_numerator, neighbor_denominator))
    return neighbors[0], neighbors[1]


@dataclass(frozen=True)
class TransverseCandidate:
    label: str
    wall_slope: Fraction
    exit_slope: Fraction
    fiber: ThreeCurrentHeightFiber


def new_support_transverse_candidates(
    new_index: int,
) -> tuple[TransverseCandidate, ...]:
    """The six exits using one selected new support and two integral ones."""

    candidates = []
    for integral_pair in combinations((1, 2, 3), 2):
        indices = tuple(sorted((new_index,) + integral_pair))
        supports = tuple(SPLIT_SUPPORTS[index] for index in indices)
        heights = tuple(support[1] for support in supports)
        wall_slope = (heights[2] - heights[0]) / (heights[1] - heights[0])
        for exit_slope in primitive_farey_neighbors(wall_slope):
            fiber = build_three_current_height_fiber(
                tuple(support[0] for support in supports),
                exit_slope,
                heights[0],
            )
            label = "-".join(support[2] for support in supports)
            candidates.append(
                TransverseCandidate(
                    f"{label}-s{exit_slope}", wall_slope, exit_slope, fiber
                )
            )
    return tuple(candidates)


def k13_transverse_candidates() -> tuple[TransverseCandidate, ...]:
    return new_support_transverse_candidates(0)


def k37_transverse_candidates() -> tuple[TransverseCandidate, ...]:
    return new_support_transverse_candidates(4)


def mixed_transverse_candidates() -> tuple[TransverseCandidate, ...]:
    """The six exits using both new supports and one integral support."""

    candidates = []
    for integral_index in (1, 2, 3):
        indices = tuple(sorted((0, integral_index, 4)))
        supports = tuple(SPLIT_SUPPORTS[index] for index in indices)
        heights = tuple(support[1] for support in supports)
        wall_slope = (heights[2] - heights[0]) / (heights[1] - heights[0])
        for exit_slope in primitive_farey_neighbors(wall_slope):
            fiber = build_three_current_height_fiber(
                tuple(support[0] for support in supports),
                exit_slope,
                heights[0],
            )
            label = "-".join(support[2] for support in supports)
            candidates.append(
                TransverseCandidate(
                    f"{label}-s{exit_slope}", wall_slope, exit_slope, fiber
                )
            )
    return tuple(candidates)


def gp_candidates_program(candidates: tuple[TransverseCandidate, ...]) -> str:
    models = []
    for candidate in candidates:
        _a2, a4, a6 = candidate.fiber.short_model
        models.append(
            "[0,0,0," + _gp_fraction(a4) + "," + _gp_fraction(a6) + "]"
        )
    labels = ",".join(f'"{candidate.label}"' for candidate in candidates)
    return "\n".join(
        (
            "default(parisizemax,4000000000);",
            "models=[" + ",".join(models) + "];",
            "labels=[" + labels + "];",
            "for(i=1,#models,E=ellinit(models[i]);M=ellminimalmodel(E,&v);"
            "N=ellglobalred(M)[1];print(labels[i],\" model=\","
            "[M.a1,M.a2,M.a3,M.a4,M.a6],\" conductor=\",N);"
            "if(N<5187563742,r=ellrank(M);print(\" rank=\",r[1..2])));",
            "quit;",
        )
    )


def gp_transverse_program(new_index: int) -> str:
    return gp_candidates_program(new_support_transverse_candidates(new_index))
    return "\n".join(
        (
            "x='x;",
            "currents=[" + currents + "];",
            "labels=[" + labels + "];",
            "for(i=1,#currents,F=factor(x^3-481*x-currents[i]);"
            "if(matsize(F)[1]==3,print(labels[i],\" current=\",currents[i],"
            "\" factors=\",F)));",
            "quit;",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transverse", action="store_true")
    parser.add_argument("--k37-transverse", action="store_true")
    parser.add_argument("--mixed-transverse", action="store_true")
    args = parser.parse_args()
    if args.mixed_transverse:
        print(gp_candidates_program(mixed_transverse_candidates()))
    elif args.k37_transverse:
        print(gp_transverse_program(4))
    elif args.transverse:
        print(gp_transverse_program(0))
    else:
        print(gp_split_program())


if __name__ == "__main__":
    main()
