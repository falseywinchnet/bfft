from __future__ import annotations

import sympy as sp

from symmetric_collision_tower import (
    c4_collision_tangent_ranks,
    inverse_coordinate_polynomial,
    point_counts,
    real_weil_polynomial,
    rho_polynomial,
    symmetric_curve_polynomial,
    symmetric_q_polynomial,
    t,
    T,
    u,
    weil_polynomial,
    x,
)


def test_symmetric_curve_construction() -> None:
    assert rho_polynomial(3) == 8 * t**2 + 4 * t + 3
    assert symmetric_q_polynomial(3) == 64 * u**2 + 45
    assert symmetric_curve_polynomial(3) == 64 * x**5 + 45 * x
    assert symmetric_q_polynomial(4) == 256 * u**3 - 140 * u + 175
    assert symmetric_curve_polynomial(4) == 256 * x**7 - 140 * x**3 + 175 * x
    assert symmetric_q_polynomial(5) == 16384 * u**4 - 16800 * u + 11025
    assert symmetric_curve_polynomial(5) == 16384 * x**9 - 16800 * x**3 + 11025 * x


def test_m4_absolute_simplicity_certificate() -> None:
    counts = point_counts(4, 13)
    assert counts == [10, 210, 2170]
    weil = weil_polynomial(counts, 13)
    assert weil == T**6 - 4 * T**5 + 28 * T**4 - 100 * T**3 + 364 * T**2 - 676 * T + 2197
    assert sp.Poly(weil, T).is_irreducible
    assert -100 % 13 != 0
    real = real_weil_polynomial(weil, 13)
    assert real == u**3 - 4 * u**2 - 11 * u + 4
    assert sp.Poly(real, u).is_irreducible
    assert sp.discriminant(real, u) == 11020


def test_m5_absolute_simplicity_certificate() -> None:
    counts = point_counts(5, 13)
    assert counts == [18, 180, 2094, 28288]
    weil = weil_polynomial(counts, 13)
    assert weil == (
        T**8 + 4 * T**7 + 13 * T**6 - 4 * T**5 - 144 * T**4
        - 52 * T**3 + 2197 * T**2 + 8788 * T + 28561
    )
    assert sp.Poly(weil, T).is_irreducible
    assert -144 % 13 != 0
    real = real_weil_polynomial(weil, 13)
    assert real == u**4 + 4 * u**3 - 39 * u**2 - 160 * u - 144
    real_poly = sp.Poly(real, u)
    assert real_poly.is_irreducible
    group, _ = real_poly.galois_group()
    assert group.order() == 24


def test_no_higher_m_collision_native_eigensystem() -> None:
    A, B, C = sp.symbols("A B C", nonzero=True)
    for m, constant_coefficient, x2_after_c0, fixed_axis in (
        (4, 20 * C, 2800 * B, 16 * symmetric_curve_polynomial(4)),
        (5, -70 * C, 66150 * B, 4 * symmetric_curve_polynomial(5)),
    ):
        inverse = inverse_coordinate_polynomial(m, A, B, C)
        assert inverse.coeff(x, 0) == constant_coefficient
        assert sp.factor(inverse.coeff(x, 2).subs(C, 0)) == x2_after_c0
        assert sp.expand(inverse.subs({A: 1, B: 0, C: 0}) - fixed_axis) == 0

    # Including the full infinitesimal PGL_2 orbit, the C4 and collision
    # tangents meet only in A-scaling.  After quotienting it, the intersection
    # in moduli has tangent rank zero.
    assert c4_collision_tangent_ranks(4) == (6, 3, 1)
    assert c4_collision_tangent_ranks(5) == (7, 3, 1)


if __name__ == "__main__":
    tests = [
        test_symmetric_curve_construction,
        test_m4_absolute_simplicity_certificate,
        test_m5_absolute_simplicity_certificate,
        test_no_higher_m_collision_native_eigensystem,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
