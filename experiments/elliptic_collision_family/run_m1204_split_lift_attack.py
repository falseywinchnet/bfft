"""First exact split-support lift shell on the rank-surplus ``m=1204`` wall.

The wall ``y^2=x^3-1204*x+15844`` has four integral split supports, while
its support-reservoir Jacobian has exact rank five.  The directions below are
the other three supports and the two smallest height-reflection directions.
The complete adjacent coefficient shell is filtered by exact factorization
of ``x^3-1204*x-K``; no rational-height cyclic cubic is accepted as a split
support.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product

try:
    from .run_support_reservoir_attack import QuarticGroup
except ImportError:  # Direct execution from this directory.
    from run_support_reservoir_attack import QuarticGroup


Q = Fraction
M = 1_204
A6 = 15_844
ORIGIN = (Q(2), Q(14_384))
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
    (Q(82), Q(68_816)),
    (Q(158), Q(68_816)),
    (Q(178), Q(14_384)),
    (Q(-2), Q(14_384)),
    (Q(-82), Q(68_816)),
)


@dataclass(frozen=True)
class LiftCandidate:
    coefficients: tuple[int, int, int, int, int]
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
    return "\n".join(
        (
            "x='x;",
            "currents=[" + currents + "];",
            "labels=[" + labels + "];",
            "for(i=1,#currents,F=factor(x^3-1204*x-currents[i]);"
            "if(matsize(F)[1]==3,print(labels[i],\" current=\",currents[i],"
            "\" factors=\",F)));",
            "quit;",
        )
    )


def main() -> None:
    print(gp_split_program())


if __name__ == "__main__":
    main()
