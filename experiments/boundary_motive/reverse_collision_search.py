"""Reverse search for exceptional, nested collision degenerations.

The forward family fixes a target arc and asks for its root clusters.  Here we
enumerate tropical target arcs first and retain Newton faces with repeated
nonzero roots.  We also record the lowest-height exceptional m=4 strata found
by solving the face-discriminant and discriminant-singularity conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys

import sympy as sp


GENUS2_DIRECTORY = Path(__file__).resolve().parents[1] / "genus2_collision"
if str(GENUS2_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GENUS2_DIRECTORY))

from symmetric_collision_tower import inverse_coordinate_polynomial  # noqa: E402


A, B, C, x, X = sp.symbols("A B C x X")
a, b, c = sp.symbols("a b c", nonzero=True)
rho, q, z = sp.symbols("rho q z")


def _lower_hull(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    hull: list[tuple[int, int]] = []
    for point in sorted(points):
        while len(hull) >= 2:
            left, middle = hull[-2], hull[-1]
            cross = (middle[0] - left[0]) * (point[1] - left[1]) - (
                middle[1] - left[1]
            ) * (point[0] - left[0])
            if cross > 0:
                break
            hull.pop()
        hull.append(point)
    return tuple(hull)


@dataclass(frozen=True)
class FaceCandidate:
    valuation_arc: tuple[int, int, int]
    root_depth: sp.Rational
    face: sp.Expr
    generic_repeated_degree: int


@lru_cache(maxsize=None)
def monomial_face_candidates(m: int, valuation_bound: int = 3) -> tuple[FaceCandidate, ...]:
    """Enumerate distinct generic faces for A=a*r^p,B=b*r^q,C=c*r^s.

    The enumeration is a tropical reconnaissance, not an assertion that a
    bounded integer box computes the complete Gröbner fan.  For m=4 and m=5
    the list stabilizes at ten faces already at bound three and remains ten
    through bound six.
    """

    inverse = sp.Poly(inverse_coordinate_polynomial(m, A, B, C, x), x)
    coefficient_data: dict[int, list[tuple[tuple[int, ...], sp.Expr]]] = {}
    for power in range(inverse.degree() + 1):
        coefficient = inverse.coeff_monomial(x**power)
        if coefficient:
            coefficient_data[power] = sp.Poly(coefficient, A, B, C).terms()

    # Store faces as primitive term signatures while scanning.  Constructing
    # and gcd-reducing a symbolic rational function in every cone is vastly
    # more expensive and supplies no additional tropical information.
    unique: dict[
        tuple[tuple[int, int, int, int, int], ...],
        tuple[tuple[int, int, int], sp.Rational],
    ] = {}
    values = range(-valuation_bound, valuation_bound + 1)
    for p, q_value, s in product(values, repeat=3):
        if (p, q_value, s) == (0, 0, 0):
            continue
        leading: dict[int, tuple[int, sp.Expr]] = {}
        points: list[tuple[int, int]] = []
        for power, terms in coefficient_data.items():
            valuations = [
                monomial[0] * p + monomial[1] * q_value + monomial[2] * s
                for monomial, _ in terms
            ]
            valuation = min(valuations)
            initial = sum(
                coefficient
                * a ** monomial[0]
                * b ** monomial[1]
                * c ** monomial[2]
                for (monomial, coefficient), term_valuation in zip(terms, valuations)
                if term_valuation == valuation
            )
            leading[power] = (valuation, sp.expand(initial))
            points.append((power, valuation))

        hull = _lower_hull(points)
        for (left_power, left_value), (right_power, right_value) in zip(hull, hull[1:]):
            depth = -sp.Rational(
                right_value - left_value, right_power - left_power
            )
            edge_value = left_value + left_power * depth
            raw_terms: list[tuple[int, int, int, int, int]] = []
            for power, (valuation, _) in leading.items():
                if valuation + power * depth != edge_value:
                    continue
                for monomial, coefficient in coefficient_data[power]:
                    term_value = (
                        monomial[0] * p
                        + monomial[1] * q_value
                        + monomial[2] * s
                    )
                    if term_value == valuation:
                        raw_terms.append(
                            (
                                power - left_power,
                                monomial[0],
                                monomial[1],
                                monomial[2],
                                int(coefficient),
                            )
                        )

            min_a = min(term[1] for term in raw_terms)
            min_b = min(term[2] for term in raw_terms)
            min_c = min(term[3] for term in raw_terms)
            content = abs(int(sp.gcd_list([term[4] for term in raw_terms])))
            primitive = [
                (
                    term[0],
                    term[1] - min_a,
                    term[2] - min_b,
                    term[3] - min_c,
                    term[4] // content,
                )
                for term in raw_terms
            ]
            primitive.sort()
            if primitive[0][4] < 0:
                primitive = [term[:-1] + (-term[-1],) for term in primitive]
            signature = tuple(primitive)
            unique.setdefault(signature, ((p, q_value, s), depth))

    candidates = []
    for signature, (valuation_arc, depth) in unique.items():
        face = sum(
            coefficient * X**x_power * a**a_power * b**b_power * c**c_power
            for x_power, a_power, b_power, c_power, coefficient in signature
        )
        # Two exact torus specializations avoid the pathological cost of a
        # multivariate symbolic gcd.  The only surviving repeated face is
        # separately identified and proved in the general cluster theorem.
        repeated_degrees = []
        for constants in ((1, 1, 1), (2, 3, 5)):
            specialized = sp.Poly(face.subs(dict(zip((a, b, c), constants))), X)
            repeated_degrees.append(sp.gcd(specialized, specialized.diff()).degree())
        candidates.append(
            FaceCandidate(
                valuation_arc=valuation_arc,
                root_depth=depth,
                face=sp.factor(face),
                generic_repeated_degree=min(repeated_degrees),
            )
        )
    return tuple(candidates)


def small_height_tuned_faces(
    m: int, valuation_bound: int = 3, coefficient_height: int = 3
) -> tuple[tuple[FaceCandidate, tuple[int, int, int], sp.Expr], ...]:
    """Find exceptional repetitions at small nonzero integer leading ratios."""

    values = tuple(range(-coefficient_height, 0)) + tuple(
        range(1, coefficient_height + 1)
    )
    hits = []
    for candidate in monomial_face_candidates(m, valuation_bound):
        if candidate.generic_repeated_degree:
            continue
        for constants in product(values, repeat=3):
            specialized = sp.Poly(
                sp.cancel(candidate.face.subs(dict(zip((a, b, c), constants)))), X
            )
            repeated = sp.gcd(specialized, specialized.diff())
            if repeated.degree() > 0:
                hits.append((candidate, constants, sp.factor(repeated.as_expr())))
    return tuple(hits)


def rho_polynomial() -> sp.Expr:
    """Cubic factor in the simplest algebraically tuned m=4 face."""

    return rho**3 - 4 * rho**2 + 10 * rho - 20


def first_tuning_polynomial(variable: sp.Expr = c) -> sp.Expr:
    """Minimal condition making two outer-face factors share a root."""

    return 875 * variable**3 + 92176 * variable**2 + 3236800 * variable + 37888000


def tuned_outer_factors() -> tuple[sp.Expr, sp.Expr]:
    """Factors whose resultant produces ``first_tuning_polynomial``."""

    cubic = X**3 - 4 * X**2 + 10 * X - 20
    quartic = (c + 35) * (X + 1) ** 4 - 35
    return cubic, quartic


def second_tuning_polynomial(variable: sp.Expr = q) -> sp.Expr:
    """Condition on the first jet that makes the inner quadratic repeat."""

    return 12950 * variable**3 - 14343 * variable**2 + 5292 * variable - 652


def rational_two_node_target() -> tuple[sp.Rational, int, sp.Rational]:
    """The low-height singular point of the m=4 discriminant surface."""

    return -sp.Rational(7, 2), 1, -sp.Rational(160, 7)


def rational_two_node_factorization() -> sp.Expr:
    inverse = inverse_coordinate_polynomial(4, A, B, C, x)
    target = dict(zip((A, B, C), rational_two_node_target()))
    return sp.factor(inverse.subs(target))


def rational_two_node_smoothing() -> sp.Expr:
    """Remainder modulo the double-root quadratic along the A direction."""

    inverse = inverse_coordinate_polynomial(4, A, B, C, x)
    A0, B0, C0 = rational_two_node_target()
    family = inverse.subs({A: A0 + z, B: B0, C: C0})
    node = 9 * x**2 - 3 * x + 2
    return sp.factor(sp.rem(sp.Poly(family, x), sp.Poly(node, x)).as_expr())


def rational_normalization_j() -> sp.Rational:
    """j-invariant of the elliptic normalization at the two-node point."""

    # Divide 27*x^3+18*x^2-3*x-8 by 27 to get a1=a3=0,
    # a2=2/3, a4=-1/9, a6=-8/27.
    a2 = sp.Rational(2, 3)
    a4 = -sp.Rational(1, 9)
    a6 = -sp.Rational(8, 27)
    b2, b4, b6 = 4 * a2, 2 * a4, 4 * a6
    b8 = 4 * a2 * a6 - a4**2
    c4 = b2**2 - 24 * b4
    discriminant = (
        -b2**2 * b8 - 8 * b4**3 - 27 * b6**2 + 9 * b2 * b4 * b6
    )
    return sp.factor(c4**3 / discriminant)


if __name__ == "__main__":
    for m in (4, 5):
        faces = monomial_face_candidates(m)
        print(
            f"m={m}: {len(faces)} generic faces, "
            f"{sum(face.generic_repeated_degree > 0 for face in faces)} "
            "generically repeated, "
            f"{len(small_height_tuned_faces(m))} small-height exceptional"
        )
    print("first algebraic tuning:", first_tuning_polynomial())
    print("second jet tuning:", second_tuning_polynomial())
    print("rational two-node fiber:", rational_two_node_factorization())
    print("rational normalization j:", rational_normalization_j())
