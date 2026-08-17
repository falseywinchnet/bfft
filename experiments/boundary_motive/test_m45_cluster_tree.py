from __future__ import annotations

import sympy as sp

from m45_cluster_tree import (
    U,
    W,
    boundary_polynomial,
    cluster_face_polynomial,
    cluster_report,
    lower_newton_hull,
    r,
    shifted_valuation_points,
)


def test_m4_boundary_cluster() -> None:
    central = sp.factor(boundary_polynomial(4))
    assert sp.factor(central.subs(r, 0)) == -(
        U + 1
    ) ** 4 * (U**3 - 4 * U**2 + 10 * U - 20)

    points = shifted_valuation_points(4)
    assert points == (
        (0, 2), (1, 2), (2, 2), (3, 2),
        (4, 0), (5, 0), (6, 0), (7, 0),
    )
    assert lower_newton_hull(points) == ((0, 2), (4, 0), (7, 0))
    assert sp.expand(cluster_face_polynomial(4) - 35 * (W**4 - 35)) == 0

    report = cluster_report(4)
    assert report.cluster_depth_in_r == sp.Rational(1, 2)
    assert report.base_change_ramification == 2
    assert report.stable_node_thickness == 1
    assert report.cluster_levels == 1
    assert report.regular_model_unit_edges == 1
    assert not report.has_four_edge_chain


def test_m5_boundary_cluster() -> None:
    central = boundary_polynomial(5).subs(r, 0)
    assert sp.factor(central) == -(
        U + 1
    ) ** 5 * (U**4 - 5 * U**3 + 15 * U**2 - 35 * U + 70)

    points = shifted_valuation_points(5)
    assert points == (
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
        (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
    )
    assert lower_newton_hull(points) == ((0, 2), (5, 0), (9, 0))
    assert sp.expand(cluster_face_polynomial(5) + 126 * (W**5 + 126)) == 0

    report = cluster_report(5)
    assert report.cluster_depth_in_r == sp.Rational(2, 5)
    assert report.base_change_ramification == 5
    assert report.stable_node_thickness == 2
    assert report.cluster_levels == 1
    assert report.stable_edges == 1
    assert report.regular_model_unit_edges == 2
    assert not report.has_four_edge_chain


def test_face_polynomials_separate_every_clustered_branch() -> None:
    for m in (4, 5):
        face = cluster_face_polynomial(m)
        assert sp.discriminant(face, W) != 0


if __name__ == "__main__":
    tests = [
        test_m4_boundary_cluster,
        test_m5_boundary_cluster,
        test_face_polynomials_separate_every_clustered_branch,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
