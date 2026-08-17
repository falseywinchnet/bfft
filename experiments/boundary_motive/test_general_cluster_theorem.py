from __future__ import annotations

import sympy as sp

from general_cluster_theorem import (
    U,
    V,
    W,
    central_polynomial,
    exact_boundary_polynomial,
    explicit_newton_hull,
    face_polynomial,
    general_cluster_report,
    parameter_constant,
    parameter_polynomial,
    r,
)


def test_parameter_constant_formula() -> None:
    for m in range(3, 13):
        assert parameter_polynomial(m, V).subs(V, 0) == parameter_constant(m)
        assert parameter_constant(m) == (
            (-1) ** (m - 1) * sp.binomial(2 * m - 1, m - 1)
        )


def test_central_factor_and_newton_hull_through_m10() -> None:
    for m in range(3, 11):
        boundary = exact_boundary_polynomial(m)
        explicit_central = sp.factor(boundary.subs({r: 0, U: V - 1}))
        assert sp.expand(explicit_central - central_polynomial(m)) == 0
        assert explicit_newton_hull(m) == ((0, 2), (m, 0), (2 * m - 1, 0))


def test_closed_face_matches_explicit_newton_face_through_m10() -> None:
    for m in range(3, 11):
        report = general_cluster_report(m)
        ramification = report.minimal_base_ramification
        thickness = report.saturated_thickness
        shifted = exact_boundary_polynomial(m).subs(U, V - 1)
        transformed = sp.cancel(
            shifted.subs({r: sp.Symbol("q") ** ramification,
                          V: sp.Symbol("q") ** thickness * W})
            / sp.Symbol("q") ** (2 * ramification)
        )
        explicit_face = sp.factor(transformed.subs(sp.Symbol("q"), 0))
        assert sp.expand(explicit_face - face_polynomial(m)) == 0


def test_face_is_squarefree_and_thickness_never_exceeds_two() -> None:
    for m in range(3, 25):
        report = general_cluster_report(m)
        assert sp.discriminant(report.face, W) != 0
        assert report.cluster_levels == 1
        assert report.saturated_thickness in (1, 2)
        assert report.saturated_thickness == (2 if m % 2 else 1)
        assert not report.has_four_edge_chain


if __name__ == "__main__":
    tests = [
        test_parameter_constant_formula,
        test_central_factor_and_newton_hull_through_m10,
        test_closed_face_matches_explicit_newton_face_through_m10,
        test_face_is_squarefree_and_thickness_never_exceeds_two,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
