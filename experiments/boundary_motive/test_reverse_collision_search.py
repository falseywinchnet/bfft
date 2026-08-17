from __future__ import annotations

import sympy as sp

from reverse_collision_search import (
    X,
    c,
    first_tuning_polynomial,
    monomial_face_candidates,
    rational_normalization_j,
    rational_two_node_factorization,
    rational_two_node_smoothing,
    rho,
    rho_polynomial,
    second_tuning_polynomial,
    small_height_tuned_faces,
    tuned_outer_factors,
    x,
    z,
)


def test_tropical_reverse_scan_stabilizes_to_ten_faces() -> None:
    for m in (4, 5):
        faces = monomial_face_candidates(m, valuation_bound=3)
        assert len(faces) == 10
        assert sum(face.generic_repeated_degree > 0 for face in faces) == 1
        assert small_height_tuned_faces(m, 3, 3) == ()


def test_first_algebraic_tuning_creates_a_second_cluster_level() -> None:
    cubic, quartic = tuned_outer_factors()
    resultant = sp.factor(sp.resultant(cubic, quartic, X))
    assert sp.expand(resultant - 1715 * first_tuning_polynomial(c)) == 0

    # Equivalently choose rho on the cubic and
    # c=35/(rho+1)^4-35.  This forces the two factors to share rho.
    c_of_rho = 35 / (rho + 1) ** 4 - 35
    assert sp.rem(
        sp.Poly(quartic.subs({X: rho, c: c_of_rho}).as_numer_denom()[0], rho),
        sp.Poly(rho_polynomial(), rho),
    ).as_expr() == 0


def test_second_jet_tuning_is_again_cubic() -> None:
    # Elimination from the inner quadratic discriminant gives a square of
    # this cubic, showing that one further jet can force a third level.
    q = sp.Symbol("q")
    condition = (
        (17269 * rho**2 - 20468 * rho + 122948) * q**2
        + (-14452 * rho**2 + 16736 * rho - 102512) * q
        + 3028 * rho**2 - 3440 * rho + 21392
    )
    assert sp.expand(
        sp.factor(sp.resultant(rho_polynomial(), condition, rho))
        - 44800 * second_tuning_polynomial(q) ** 2
    ) == 0


def test_low_height_rational_discriminant_singularity() -> None:
    expected = (
        sp.Rational(100, 7)
        * (9 * x**2 - 3 * x + 2) ** 2
        * (27 * x**3 + 18 * x**2 - 3 * x - 8)
    )
    assert sp.expand(rational_two_node_factorization() - expected) == 0

    smoothing = rational_two_node_smoothing()
    expected_smoothing = (
        sp.Rational(4, 107163)
        * z**2
        * (21 * x - 10)
        * (6400 * z**2 - 78624 * z + 346969)
    )
    assert sp.expand(smoothing - expected_smoothing) == 0
    assert sp.resultant(9 * x**2 - 3 * x + 2, 21 * x - 10, x) == 1152
    assert rational_normalization_j() == -sp.Rational(224, 3)


if __name__ == "__main__":
    tests = [
        test_tropical_reverse_scan_stabilizes_to_ten_faces,
        test_first_algebraic_tuning_creates_a_second_cluster_level,
        test_second_jet_tuning_is_again_cubic,
        test_low_height_rational_discriminant_singularity,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
