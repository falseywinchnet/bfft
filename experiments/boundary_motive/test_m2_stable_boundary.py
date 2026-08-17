from __future__ import annotations

import sympy as sp

from m2_stable_boundary import (
    T,
    denominator,
    epsilon,
    bubble_polynomials,
    numerator,
    pole_square_roots,
    source_coordinate,
    stable_boundary_report,
    t,
    target_infinity_coordinate,
    u,
)


def test_five_marked_branches_are_recovered() -> None:
    assert sp.expand(numerator() - t * (t**2 + epsilon)) == 0
    assert sp.expand(
        denominator() - (4 * t**4 - (4 * epsilon + 3) * t**2 - epsilon)
    ) == 0
    assert sp.cancel(
        source_coordinate()
        - 4 * t * (t**2 + epsilon)
        / (4 * t**4 - (4 * epsilon + 3) * t**2 - epsilon)
    ) == 0

    small, distant = pole_square_roots()
    assert sp.series(small, epsilon, 0, 3) == (
        -epsilon / 3 + 16 * epsilon**2 / 27 + sp.Order(epsilon**3)
    )
    assert sp.series(distant, epsilon, 0, 3) == (
        sp.Rational(3, 4)
        + 4 * epsilon / 3
        - 16 * epsilon**2 / 27
        + sp.Order(epsilon**3)
    )


def test_one_blowup_separates_the_cluster() -> None:
    zero_bubble, pole_bubble = bubble_polynomials()
    assert sp.expand(zero_bubble - T * (T**2 + 1)) == 0
    assert pole_bubble == 4 * T**4 * u**2 - 4 * T**2 * u**2 - 3 * T**2 - 1

    # On u=0 the three zeros are T=0,+/-i and the two poles obey
    # T^2=-1/3.  They are five distinct geometric points.
    assert sp.factor(zero_bubble.subs(u, 0)) == T * (T**2 + 1)
    assert sp.factor(pole_bubble.subs(u, 0)) == -3 * T**2 - 1
    assert sp.resultant(zero_bubble.subs(u, 0), pole_bubble.subs(u, 0), T) != 0


def test_plumbing_is_one_edge_of_thickness_one() -> None:
    report = stable_boundary_report()
    assert report.vertices == 2
    assert report.edges == 1
    assert report.plumbing_order_after_base_change == 1
    assert report.plumbing_order_in_epsilon == sp.Rational(1, 2)
    assert not report.has_four_edge_chain


def test_target_ramification_is_not_plumbing_thickness() -> None:
    v = target_infinity_coordinate(t)
    assert sp.series(v, t, 0, 6) == -2 * t**3 - 8 * t**5 + sp.Order(t**6)
    assert stable_boundary_report().target_ramification_order == 3


if __name__ == "__main__":
    tests = [
        test_five_marked_branches_are_recovered,
        test_one_blowup_separates_the_cluster,
        test_plumbing_is_one_edge_of_thickness_one,
        test_target_ramification_is_not_plumbing_thickness,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
