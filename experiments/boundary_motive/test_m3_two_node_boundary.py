from __future__ import annotations

import sympy as sp

from m3_two_node_boundary import (
    family_polynomial,
    local_node_coefficients,
    node_polynomial,
    two_node_report,
    x,
    z,
)


def test_central_fiber_has_exactly_two_nodes() -> None:
    polynomial = family_polynomial()
    assert sp.factor(polynomial.subs(z, 0)) == (
        (x + 2) * node_polynomial() ** 2 / 4
    )
    assert sp.factor(sp.discriminant(polynomial, x)) == (
        sp.Rational(20503125, 16) * z**2 * (z - 1) ** 2
    )


def test_each_node_is_smoothed_transversely() -> None:
    A, B = local_node_coefficients()
    assert sp.expand(A + 5 * (x + 2)) == 0
    assert sp.expand(B - (5 * x - 12) / 4) == 0
    assert sp.resultant(node_polynomial(), A, x) == 375
    assert sp.resultant(node_polynomial(), B, x) == sp.Rational(243, 16)

    report = two_node_report()
    assert report.plumbing_orders == (1, 1)


def test_dual_graph_has_loops_not_a_four_edge_chain() -> None:
    report = two_node_report()
    assert (report.vertices, report.loop_edges, report.chain_edges) == (1, 2, 0)
    assert not report.has_four_edge_chain


if __name__ == "__main__":
    tests = [
        test_central_fiber_has_exactly_two_nodes,
        test_each_node_is_smoothed_transversely,
        test_dual_graph_has_loops_not_a_four_edge_chain,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
