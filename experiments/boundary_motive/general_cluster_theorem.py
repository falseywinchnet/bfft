"""Closed-form Newton-cluster theorem for the natural collision line.

For every m >= 3, the projective inverse-coordinate boundary has one cluster
of m branches at U=-1.  Its depth is 2/m and its minimal saturated thickness
is 2/gcd(m,2).  The proof is recorded in README.md; this module exposes the
closed formulas and supplies exact finite-m verification against the original
resultant construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
import sys

import sympy as sp


GENUS2_DIRECTORY = Path(__file__).resolve().parents[1] / "genus2_collision"
if str(GENUS2_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GENUS2_DIRECTORY))

from symmetric_collision_tower import (  # noqa: E402
    inverse_coordinate_polynomial,
    r_polynomial,
)


A, B, C, x, r, u, U, V, W = sp.symbols("A B C x r u U V W")


def parameter_polynomial(m: int, variable: sp.Expr = V) -> sp.Expr:
    """A_m(variable)=R_m(0,variable)."""

    return sp.expand(r_polynomial(m, sp.Integer(0), variable))


def parameter_constant(m: int) -> sp.Integer:
    """A_m(0)=(-1)^(m-1) binomial(2m-1,m-1)."""

    return (-1) ** (m - 1) * sp.binomial(2 * m - 1, m - 1)


def exact_boundary_polynomial(m: int) -> sp.Expr:
    """Compute r*H_m(1,1/r,1/r,rU) from the defining resultant."""

    inverse = inverse_coordinate_polynomial(m, A, B, C, x)
    expression = sp.cancel(
        r * inverse.subs({A: 1, B: 1 / r, C: 1 / r, x: r * U})
    )
    if sp.denom(expression) != 1:
        raise AssertionError("boundary expression was not polynomial")
    return sp.expand(expression)


def central_polynomial(m: int) -> sp.Expr:
    """Closed central fiber after V=U+1."""

    return sp.factor(-V**m * parameter_polynomial(m, V))


def face_polynomial(m: int) -> sp.Expr:
    """Closed Newton face after the minimal root-separating base change."""

    constant = parameter_constant(m)
    return sp.factor(-constant * (W**m + constant))


def coefficient_valuation(coefficient: sp.Expr) -> int:
    polynomial = sp.Poly(coefficient, r)
    return min(power[0] for power, value in polynomial.terms() if value)


def explicit_newton_hull(m: int) -> tuple[tuple[int, int], ...]:
    """Compute the lower hull from the explicit resultant."""

    shifted = sp.Poly(sp.expand(exact_boundary_polynomial(m).subs(U, V - 1)), V)
    points: list[tuple[int, int]] = []
    for power in range(shifted.degree() + 1):
        coefficient = shifted.coeff_monomial(V**power)
        if coefficient:
            points.append((power, coefficient_valuation(coefficient)))

    hull: list[tuple[int, int]] = []
    for point in points:
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
class GeneralClusterReport:
    m: int
    total_branches: int
    clustered_branches: int
    newton_hull: tuple[tuple[int, int], ...]
    depth_in_r: sp.Rational
    minimal_base_ramification: int
    saturated_thickness: int
    cluster_levels: int
    stable_edges: int
    regular_model_unit_edges: int
    face: sp.Expr

    @property
    def has_four_edge_chain(self) -> bool:
        return self.regular_model_unit_edges >= 4


def general_cluster_report(m: int) -> GeneralClusterReport:
    """The proved cluster data, valid for every integer m >= 3."""

    if m < 3:
        raise ValueError("the closed proof is stated for m >= 3")
    common = gcd(m, 2)
    thickness = 2 // common
    return GeneralClusterReport(
        m=m,
        total_branches=2 * m - 1,
        clustered_branches=m,
        newton_hull=((0, 2), (m, 0), (2 * m - 1, 0)),
        depth_in_r=sp.Rational(2, m),
        minimal_base_ramification=m // common,
        saturated_thickness=thickness,
        cluster_levels=1,
        stable_edges=1,
        regular_model_unit_edges=thickness,
        face=face_polynomial(m),
    )


if __name__ == "__main__":
    for degree_parameter in range(3, 11):
        report = general_cluster_report(degree_parameter)
        print(
            f"m={degree_parameter}: hull={report.newton_hull}, "
            f"depth={report.depth_in_r}, "
            f"base ramification={report.minimal_base_ramification}, "
            f"thickness={report.saturated_thickness}"
        )
