"""Newton-cluster screen for the m=4 and m=5 collision boundaries.

We use the natural projective line through the symmetric collision target

    (A,B,C) = (1, 1/r, 1/r),

and inspect the inverse-coordinate branch polynomial at target infinity
``r=0``.  Writing ``x=r*U`` gives a finite boundary polynomial.  The repeated
root at ``U=-1`` is then resolved by its Newton polygon.
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

from symmetric_collision_tower import inverse_coordinate_polynomial  # noqa: E402


A, B, C, x, r, u, U, V, W = sp.symbols("A B C x r u U V W")


def boundary_polynomial(m: int) -> sp.Expr:
    """The finite r=0 model r*H_m(1,1/r,1/r,rU)."""

    inverse = inverse_coordinate_polynomial(m, A, B, C, x)
    expression = sp.cancel(
        r * inverse.subs({A: 1, B: 1 / r, C: 1 / r, x: r * U})
    )
    if sp.denom(expression) != 1:
        raise AssertionError("projective boundary expression was not polynomial")
    return sp.expand(expression)


def coefficient_valuation(coefficient: sp.Expr) -> tuple[int, sp.Expr]:
    """Return the r-adic valuation and initial coefficient."""

    polynomial = sp.Poly(coefficient, r)
    valuation = min(power[0] for power, value in polynomial.terms() if value)
    return valuation, sp.expand(coefficient).coeff(r, valuation)


def shifted_valuation_points(m: int) -> tuple[tuple[int, int], ...]:
    """Valuation points after centering the repeated root with V=U+1."""

    shifted = sp.Poly(sp.expand(boundary_polynomial(m).subs(U, V - 1)), V)
    points = []
    for power in range(shifted.degree() + 1):
        coefficient = shifted.coeff_monomial(V**power)
        if coefficient:
            valuation, _ = coefficient_valuation(coefficient)
            points.append((power, valuation))
    return tuple(points)


def lower_newton_hull(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Lower convex hull, omitting collinear interior points."""

    hull: list[tuple[int, int]] = []
    for point in sorted(points):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (b[0] - a[0]) * (point[1] - a[1]) - (
                b[1] - a[1]
            ) * (point[0] - a[0])
            if cross > 0:
                break
            hull.pop()
        hull.append(point)
    return tuple(hull)


def cluster_face_polynomial(m: int) -> sp.Expr:
    """Initial polynomial after the minimal root-separating base change."""

    common = gcd(m, 2)
    ramification = m // common
    thickness = 2 // common
    shifted = boundary_polynomial(m).subs(U, V - 1)
    transformed = sp.cancel(
        shifted.subs({r: u**ramification, V: u**thickness * W})
        / u ** (2 * ramification)
    )
    return sp.factor(transformed.subs(u, 0))


@dataclass(frozen=True)
class ClusterReport:
    m: int
    total_branches: int
    clustered_branches: int
    cluster_depth_in_r: sp.Rational
    base_change_ramification: int
    stable_node_thickness: int
    cluster_levels: int
    stable_edges: int
    regular_model_unit_edges: int
    newton_hull: tuple[tuple[int, int], ...]
    central_factorization: sp.Expr
    face_polynomial: sp.Expr

    @property
    def has_four_edge_chain(self) -> bool:
        return self.regular_model_unit_edges >= 4


def cluster_report(m: int) -> ClusterReport:
    """Return the exact one-cluster stable-reduction data for m=4 or m=5."""

    if m not in (4, 5):
        raise ValueError("this screen is intentionally restricted to m=4 and m=5")
    common = gcd(m, 2)
    hull = lower_newton_hull(shifted_valuation_points(m))
    return ClusterReport(
        m=m,
        total_branches=2 * m - 1,
        clustered_branches=m,
        cluster_depth_in_r=sp.Rational(2, m),
        base_change_ramification=m // common,
        stable_node_thickness=2 // common,
        cluster_levels=1,
        stable_edges=1,
        regular_model_unit_edges=2 // common,
        newton_hull=hull,
        central_factorization=sp.factor(boundary_polynomial(m).subs(r, 0)),
        face_polynomial=cluster_face_polynomial(m),
    )


if __name__ == "__main__":
    for degree_parameter in (4, 5):
        report = cluster_report(degree_parameter)
        print(f"m={degree_parameter}")
        print("  central factorization:", report.central_factorization)
        print("  Newton hull:", report.newton_hull)
        print("  face polynomial:", report.face_polynomial)
        print("  base ramification:", report.base_change_ramification)
        print("  stable thickness:", report.stable_node_thickness)
