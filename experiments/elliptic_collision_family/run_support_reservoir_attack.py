"""Exact group-law attack on the ``m=217`` horizontal-support reservoir.

For a short curve ``y^2=x^3-m*x+a6``, a horizontal cubic with product
current ``K`` has discriminant

    delta^2 = 4*m^3 - 27*K^2.

Its height is rational exactly when ``h^2=a6+K``.  Eliminating ``K`` gives
the genus-one reservoir

    delta^2 = 4*m^3 - 27*(h^2-a6)^2.

The rank-five subrecord ``(m,a6)=(217,1585)`` supplies four integral split
supports on this reservoir.  This module performs its group law directly on
the quartic, with ``(19,650)`` as origin.  Adjacent coefficient shells are
therefore Mordell--Weil shells, not a scan through rational heights.

The square-discriminant condition alone permits cyclic irreducible cubics.
Consequently every output also records the exact depressed cubic
``x^3-m*x-K``; PARI can certify which reservoir points lift through the
split-support three-cover.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product

try:
    from .run_three_current_height_conic import (
        RANK_FIVE_217_TRIPLES,
        ThreeCurrentHeightFiber,
        build_three_current_height_fiber,
        support_current,
    )
except ImportError:  # Direct execution from this directory.
    from run_three_current_height_conic import (
        RANK_FIVE_217_TRIPLES,
        ThreeCurrentHeightFiber,
        build_three_current_height_fiber,
        support_current,
    )


Q = Fraction
Point = tuple[Fraction, Fraction]
Polynomial = list[Fraction]

M = 217
A6 = 1_585
ORIGIN: Point = (Q(19), Q(650))
KNOWN_SPLIT_POINTS: tuple[Point, ...] = (
    (Q(31), Q(5_510)),
    (Q(47), Q(5_510)),
    (Q(53), Q(650)),
    (Q(-19), Q(650)),
)
NEW_SPLIT_TRIPLE = (Q(5_407, 361), Q(-183, 361), Q(-5_224, 361))
NEW_SPLIT_HEIGHT = Q(282_377, 6_859)
WALL_SUPPORTS = tuple(
    sorted(
        tuple(
            zip(
                RANK_FIVE_217_TRIPLES,
                (Q(19), Q(31), Q(47), Q(53)),
                ("K-1224", "K-624", "K624", "K1224"),
            )
        )
        + ((NEW_SPLIT_TRIPLE, NEW_SPLIT_HEIGHT, "Knew"),),
        key=lambda support: support_current(support[0]),
    )
)


def reservoir_polynomial() -> Polynomial:
    """Return coefficients, low degree first, of the support quartic."""

    return [Q(4 * M**3 - 27 * A6**2), Q(0), Q(54 * A6), Q(0), Q(-27)]


F = reservoir_polynomial()


def _evaluate(polynomial: Polynomial, value: Fraction) -> Fraction:
    return sum(coefficient * value**degree for degree, coefficient in enumerate(polynomial))


def _first_derivative(value: Fraction) -> Fraction:
    return sum(
        degree * coefficient * value ** (degree - 1)
        for degree, coefficient in enumerate(F)
        if degree
    )


def _second_derivative(value: Fraction) -> Fraction:
    return sum(
        degree * (degree - 1) * coefficient * value ** (degree - 2)
        for degree, coefficient in enumerate(F)
        if degree >= 2
    )


def _multiply_polynomials(left: Polynomial, right: Polynomial) -> Polynomial:
    result = [Q(0)] * (len(left) + len(right) - 1)
    for left_degree, left_coefficient in enumerate(left):
        for right_degree, right_coefficient in enumerate(right):
            result[left_degree + right_degree] += left_coefficient * right_coefficient
    return result


def _subtract_polynomials(left: Polynomial, right: Polynomial) -> Polynomial:
    degree = max(len(left), len(right))
    return [
        (left[index] if index < len(left) else Q(0))
        - (right[index] if index < len(right) else Q(0))
        for index in range(degree)
    ]


def _divide_polynomials(dividend: Polynomial, divisor: Polynomial) -> Polynomial:
    remainder = dividend[:]
    while len(remainder) > 1 and not remainder[-1]:
        remainder.pop()
    quotient = [Q(0)] * max(1, len(remainder) - len(divisor) + 1)
    while len(remainder) >= len(divisor):
        coefficient = remainder[-1] / divisor[-1]
        shift = len(remainder) - len(divisor)
        quotient[shift] = coefficient
        for index, value in enumerate(divisor):
            remainder[shift + index] -= coefficient * value
        while len(remainder) > 1 and not remainder[-1]:
            remainder.pop()
    if any(remainder):
        raise ArithmeticError("quartic intersection did not divide exactly")
    return quotient


def _solve_three(rows: list[list[Fraction]], values: list[Fraction]) -> Polynomial:
    matrix = [row[:] + [value] for row, value in zip(rows, values)]
    for column in range(3):
        pivot = next(
            (row for row in range(column, 3) if matrix[row][column]), None
        )
        if pivot is None:
            raise ArithmeticError("degenerate quartic interpolation chart")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        for row in range(3):
            if row == column:
                continue
            scale = matrix[row][column]
            if scale:
                matrix[row] = [
                    value - scale * pivot_value
                    for value, pivot_value in zip(matrix[row], matrix[column])
                ]
    return [matrix[row][-1] for row in range(3)]


@dataclass(frozen=True)
class QuarticGroup:
    """Affine group law on ``y^2=quartic(x)`` with a rational origin."""

    polynomial: tuple[Fraction, Fraction, Fraction, Fraction, Fraction]
    origin: Point

    def evaluate(self, value: Fraction) -> Fraction:
        return _evaluate(list(self.polynomial), value)

    def verify(self, point: Point) -> bool:
        return point[1] * point[1] == self.evaluate(point[0])

    def derivative(self, value: Fraction, order: int = 1) -> Fraction:
        if order not in (1, 2):
            raise ValueError("only first and second derivatives are required")
        return sum(
            degree
            * (degree - 1 if order == 2 else 1)
            * coefficient
            * value ** (degree - order)
            for degree, coefficient in enumerate(self.polynomial)
            if degree >= order
        )

    def fourth_intersection(self, left: Point, right: Point) -> Point:
        grouped: list[list[object]] = []
        for point in (left, right, self.origin):
            for group in grouped:
                if group[0] == point:
                    group[1] = int(group[1]) + 1
                    break
            else:
                grouped.append([point, 1])

        rows: list[list[Fraction]] = []
        values: list[Fraction] = []
        divisor: Polynomial = [Q(1)]
        for point, raw_multiplicity in grouped:
            abscissa, ordinate = point  # type: ignore[misc]
            multiplicity = int(raw_multiplicity)
            if not self.verify((abscissa, ordinate)):
                raise ValueError("point is not on the quartic")
            rows.append([Q(1), abscissa, abscissa * abscissa])
            values.append(ordinate)
            if multiplicity >= 2:
                tangent = self.derivative(abscissa) / (2 * ordinate)
                rows.append([Q(0), Q(1), 2 * abscissa])
                values.append(tangent)
            if multiplicity >= 3:
                curvature = (
                    self.derivative(abscissa, 2) / (2 * ordinate)
                    - self.derivative(abscissa) ** 2 / (4 * ordinate**3)
                )
                rows.append([Q(0), Q(0), Q(2)])
                values.append(curvature)
            for _ in range(multiplicity):
                divisor = _multiply_polynomials(divisor, [-abscissa, Q(1)])

        quadratic = _solve_three(rows, values)
        intersection = _subtract_polynomials(
            list(self.polynomial), _multiply_polynomials(quadratic, quadratic)
        )
        residual = _divide_polynomials(intersection, divisor)
        while len(residual) > 1 and not residual[-1]:
            residual.pop()
        if len(residual) != 2:
            raise ArithmeticError("quartic group result left the affine chart")
        abscissa = -residual[0] / residual[1]
        result = abscissa, _evaluate(quadratic, abscissa)
        if not self.verify(result):
            raise ArithmeticError("quartic group result failed verification")
        return result

    @property
    def translation(self) -> Point:
        return self.fourth_intersection(self.origin, self.origin)

    def add(self, left: Point, right: Point) -> Point:
        return self.fourth_intersection(
            self.fourth_intersection(left, right), self.origin
        )

    def negate(self, point: Point) -> Point:
        return self.fourth_intersection(point, self.translation)

    def multiply(self, coefficient: int, point: Point) -> Point:
        if coefficient < 0:
            return self.multiply(-coefficient, self.negate(point))
        result = self.origin
        addend = point
        while coefficient:
            if coefficient & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            coefficient //= 2
        return result


def verify_point(point: Point) -> bool:
    height, discriminant_root = point
    return discriminant_root * discriminant_root == _evaluate(F, height)


def _fourth_intersection(left: Point, right: Point) -> Point:
    """Return ``T-left-right`` using a quadratic through left, right, origin."""

    grouped: list[list[object]] = []
    for point in (left, right, ORIGIN):
        for group in grouped:
            if group[0] == point:
                group[1] = int(group[1]) + 1
                break
        else:
            grouped.append([point, 1])

    rows: list[list[Fraction]] = []
    values: list[Fraction] = []
    divisor: Polynomial = [Q(1)]
    for point, raw_multiplicity in grouped:
        height, discriminant_root = point  # type: ignore[misc]
        multiplicity = int(raw_multiplicity)
        if not verify_point((height, discriminant_root)):
            raise ValueError("point is not on the support reservoir")
        rows.append([Q(1), height, height * height])
        values.append(discriminant_root)
        if multiplicity >= 2:
            tangent = _first_derivative(height) / (2 * discriminant_root)
            rows.append([Q(0), Q(1), 2 * height])
            values.append(tangent)
        if multiplicity >= 3:
            curvature = (
                _second_derivative(height) / (2 * discriminant_root)
                - _first_derivative(height) ** 2 / (4 * discriminant_root**3)
            )
            rows.append([Q(0), Q(0), Q(2)])
            values.append(curvature)
        for _ in range(multiplicity):
            divisor = _multiply_polynomials(divisor, [-height, Q(1)])

    quadratic = _solve_three(rows, values)
    intersection = _subtract_polynomials(
        F, _multiply_polynomials(quadratic, quadratic)
    )
    residual = _divide_polynomials(intersection, divisor)
    while len(residual) > 1 and not residual[-1]:
        residual.pop()
    if len(residual) != 2:
        raise ArithmeticError("quartic group result left the affine chart")
    height = -residual[0] / residual[1]
    result = height, _evaluate(quadratic, height)
    if not verify_point(result):
        raise ArithmeticError("quartic group result failed verification")
    return result


TRANSLATION = _fourth_intersection(ORIGIN, ORIGIN)


def add_points(left: Point, right: Point) -> Point:
    """Group addition on the quartic with :data:`ORIGIN` as identity."""

    return _fourth_intersection(_fourth_intersection(left, right), ORIGIN)


def negate_point(point: Point) -> Point:
    """Group inverse in the generic affine interpolation chart."""

    return _fourth_intersection(point, TRANSLATION)


def multiply_point(coefficient: int, point: Point) -> Point:
    if coefficient < 0:
        return multiply_point(-coefficient, negate_point(point))
    result = ORIGIN
    addend = point
    while coefficient:
        if coefficient & 1:
            result = add_points(result, addend)
        addend = add_points(addend, addend)
        coefficient //= 2
    return result


@dataclass(frozen=True)
class ReservoirCandidate:
    coefficients: tuple[int, ...]
    height: Fraction
    discriminant_root: Fraction

    @property
    def current(self) -> Fraction:
        return self.height * self.height - A6

    @property
    def depressed_cubic(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return Q(1), Q(0), Q(-M), -self.current

    @property
    def parameter_bits(self) -> int:
        values = (self.height, self.discriminant_root, self.current)
        return max(
            max(value.numerator.bit_length(), value.denominator.bit_length())
            for value in values
        )


def adjacent_shell() -> tuple[ReservoirCandidate, ...]:
    """Return the first exact coefficient shell of the four known directions."""

    candidates = []
    seen: set[Point] = set()
    for coefficients in product((-1, 0, 1), repeat=len(KNOWN_SPLIT_POINTS)):
        if not any(coefficients):
            continue
        point = ORIGIN
        try:
            for coefficient, generator in zip(coefficients, KNOWN_SPLIT_POINTS):
                point = add_points(point, multiply_point(coefficient, generator))
        except ArithmeticError:
            # A vertical conjugate can leave this particular quadratic chart.
            # It represents a known deck boundary, not a missing parameter walk.
            continue
        if point in seen:
            continue
        seen.add(point)
        candidates.append(
            ReservoirCandidate(coefficients, point[0], point[1])
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.parameter_bits))


def primitive_farey_neighbors(slope: Fraction) -> tuple[Fraction, Fraction]:
    """Return the two finite shortest Farey neighbors of a reduced slope."""

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
        neighbor = Q(neighbor_numerator, neighbor_denominator)
        if numerator * neighbor.denominator - denominator * neighbor.numerator != determinant:
            raise ArithmeticError("Farey neighbor determinant failed")
        neighbors.append(neighbor)
    return neighbors[0], neighbors[1]


@dataclass(frozen=True)
class TransverseCandidate:
    label: str
    wall_slope: Fraction
    exit_slope: Fraction
    fiber: ThreeCurrentHeightFiber


def primitive_transverse_candidates() -> tuple[TransverseCandidate, ...]:
    """Bound the first transverse shell opened by the new split support.

    There are exactly six choices of two old supports together with the new
    one, and two determinant-one Farey exits from each wall point.  No other
    slopes belong to this adjacent shell.
    """

    new_index = next(
        index for index, support in enumerate(WALL_SUPPORTS) if support[2] == "Knew"
    )
    candidates = []
    for indices in combinations(range(len(WALL_SUPPORTS)), 3):
        if new_index not in indices:
            continue
        supports = tuple(WALL_SUPPORTS[index] for index in indices)
        heights = tuple(support[1] for support in supports)
        wall_slope = (heights[2] - heights[0]) / (heights[1] - heights[0])
        for exit_slope in primitive_farey_neighbors(wall_slope):
            fiber = build_three_current_height_fiber(
                tuple(support[0] for support in supports),
                exit_slope,
                heights[0],
            )
            labels = "-".join(support[2] for support in supports)
            candidates.append(
                TransverseCandidate(
                    f"{labels}-s{exit_slope}", wall_slope, exit_slope, fiber
                )
            )
    return tuple(candidates)


def _gp_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def gp_transverse_program() -> str:
    """Return a PARI program for exact reduction of the bounded shell."""

    candidates = primitive_transverse_candidates()
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
            "if(N<5187563742,r=ellrank(M);print(\"  rank=\",r[1..2])));",
            "quit;",
        )
    )


def gp_split_shell_program() -> str:
    """Return a PARI program certifying all split cubics in the first shell."""

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
            "for(i=1,#currents,F=factor(x^3-217*x-currents[i]);"
            "if(matsize(F)[1]==3,print(labels[i],\" current=\",currents[i],"
            "\" factors=\",F)));",
            "quit;",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gp-transverse", action="store_true")
    parser.add_argument("--gp-split-shell", action="store_true")
    args = parser.parse_args()
    if args.gp_transverse:
        print(gp_transverse_program())
        return
    if args.gp_split_shell:
        print(gp_split_shell_program())
        return
    print("quartic=", F)
    print("origin=", ORIGIN)
    print("translation=", TRANSLATION)
    for candidate in adjacent_shell():
        print(
            "coefficients=", candidate.coefficients,
            "height=", candidate.height,
            "current=", candidate.current,
            "delta=", candidate.discriminant_root,
            "bits=", candidate.parameter_bits,
            "cubic=", candidate.depressed_cubic,
        )

    print("\nNEW-SUPPORT PRIMITIVE TRANSVERSE SHELL")
    for candidate in primitive_transverse_candidates():
        print(
            "label=", candidate.label,
            "wall=", candidate.wall_slope,
            "exit=", candidate.exit_slope,
            "lambda=", candidate.fiber.dilation,
        )


if __name__ == "__main__":
    main()
