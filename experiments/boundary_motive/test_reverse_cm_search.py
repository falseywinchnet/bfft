"""Tests for the reverse maximal-node/CM search."""

import sympy as sp

from reverse_cm_search import (
    beta,
    generic_branch_is_empty,
    m4_factor_shape,
    m4_normalization_j,
    m4_shape_resultant,
    m4_unique_elliptic_fiber,
    m4_unique_factorization,
    m5_factor_shape,
    maximal_elliptic_expected_dimension,
    sparse_equianharmonic_ghost,
    t,
    x,
)


def test_dimension_threshold() -> None:
    assert maximal_elliptic_expected_dimension(4) == 1
    assert maximal_elliptic_expected_dimension(5) == 0
    assert maximal_elliptic_expected_dimension(6) == -1


def test_high_coefficient_shape() -> None:
    q, e, constraint = m5_factor_shape()
    assert constraint == 0
    product = sp.Poly(sp.expand(q**2 * e), x)
    for power in range(5, 9):
        assert product.coeff_monomial(x**power) == 0
    assert product.coeff_monomial(x**9) == 1


def test_complete_m4_elliptic_survivor() -> None:
    q, e = m4_factor_shape()
    parameter, _, _, _, leading, j_value = m4_unique_elliptic_fiber()
    expected = leading * (q**2 * e).subs(t, parameter)
    assert sp.expand(m4_unique_factorization() - expected) == 0
    assert m4_normalization_j(parameter) == j_value == -sp.Rational(224, 3)
    assert sp.factor(m4_shape_resultant()).subs(t, parameter) == 0


def test_equianharmonic_branch_is_degree_drop() -> None:
    assert sp.factor(sparse_equianharmonic_ghost() + 9800 / beta) == 0


def test_generic_candidate_fails_last_equation() -> None:
    assert generic_branch_is_empty()


if __name__ == "__main__":
    for test in (
        test_dimension_threshold,
        test_high_coefficient_shape,
        test_complete_m4_elliptic_survivor,
        test_equianharmonic_branch_is_degree_drop,
        test_generic_candidate_fails_last_equation,
    ):
        test()
    print("reverse CM search tests passed")
