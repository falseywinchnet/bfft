from __future__ import annotations

import sympy as sp

from genus2_collision import (
    collision_axis_quintic,
    c4_moving_branch_obstruction,
    de_rham_reduce_matrix,
    inverse_quintic,
    matrix_lie_algebra_dimension,
    normalized_symmetric_cube_obstructions,
    point_counts_mod_7,
    primitive_wedge_connection,
    r3,
    reduce_alpha,
    s3,
    scalar_operator,
    simple_slice_quintic,
    simple_slice_operator4,
    two_node_slice_quintic,
    two_node_slice_target,
    x,
    z,
)


def test_poster_m3_hensel_lift() -> None:
    t, alpha = sp.symbols("t alpha")
    p = sp.Symbol("p")
    assert sp.expand(r3(t, p) - (p**2 - 5 * (1 + t) * p + 10 * (1 + t + t**2))) == 0
    assert sp.expand(r3(t, 1 - t) - (16 * t**2 + 8 * t + 6)) == 0
    lifted = reduce_alpha(r3(t, (t - 1) ** 2 * s3(t, alpha)), alpha)
    assert all(sp.expand(lifted).coeff(t, degree) == 0 for degree in range(3))


def test_generic_inverse_resultant() -> None:
    A, B, C, t = sp.symbols("A B C t")
    equation1 = t**2 - (1 + B * x) * t + A * x**2
    p = A * x**2 / t
    equation2 = sp.expand(C * t**5 - x * sp.together(r3(t, p) * t**2))
    resultant = sp.factor(sp.resultant(equation1, equation2, t))
    assert sp.expand(resultant - A**2 * x**5 * inverse_quintic(A, B, C)) == 0


def test_collision_axis_is_split_cm() -> None:
    c = sp.symbols("c", nonzero=True)
    assert collision_axis_quintic(c) == x * (64 * c**2 * x**4 + 45)

    # Normalize to v^2=u(u^4+1).  The non-hyperelliptic involution
    # (u,v)->(1/u,v/u^3) has elliptic quotient
    # V^2=(U+2)(U^2-2), U=u+1/u, V=v(u+1)/u^2.
    u = sp.symbols("u", nonzero=True)
    U = u + 1 / u
    quotient_identity = sp.factor(
        u * (u**4 + 1) * (u + 1) ** 2 / u**4 - (U + 2) * (U**2 - 2)
    )
    assert quotient_identity == 0

    # y^2 = U^3+2U^2-2U-4 has c4=160, Delta=512, j=8000.
    a2, a4, a6 = 2, -2, -4
    b2, b4, b6 = 4 * a2, 2 * a4, 4 * a6
    b8 = 4 * a2 * a6 - a4**2
    c4 = b2**2 - 24 * b4
    delta = -b2**2 * b8 - 8 * b4**3 - 27 * b6**2 + 9 * b2 * b4 * b6
    assert (c4, delta, c4**3 // delta) == (160, 512, 8000)


def test_nonsymmetric_slice_and_absolute_simplicity_certificate() -> None:
    f = simple_slice_quintic()
    expected_factorization = -(x**2 - 3 * x + 6) * (
        x**3 * z - 10 * x**3 + 3 * x**2 * z - 30 * x**2 + 3 * x * z - 30 * x + z
    )
    assert sp.expand(f - expected_factorization) == 0
    assert sp.factor(sp.discriminant(f, x)) == 101250000 * (z - 10) ** 2 * (
        20 * z**2 - 405 * z + 2052
    ) ** 2

    # At z=1, good reduction modulo 7 has N1=9 and N2=45.  Thus its
    # Frobenius polynomial is T^4+T^3-2T^2+7T+49.  It is ordinary,
    # irreducible, and misses all four Howe-Zhu splitting equalities.
    assert point_counts_mod_7() == (9, 45)
    T = sp.symbols("T")
    frobenius = T**4 + T**3 - 2 * T**2 + 7 * T + 49
    assert sp.Poly(frobenius, T, domain=sp.QQ).is_irreducible
    a, b, q = 1, -2, 7
    assert b % q != 0
    assert a != 0
    assert a**2 != q + b
    assert a**2 != 2 * b
    assert a**2 != 3 * b - 3 * q


def test_picard_fuchs_is_genuinely_rank_four_and_wedge_rank_five() -> None:
    f = simple_slice_quintic()
    connection4 = de_rham_reduce_matrix(f)
    operator4 = simple_slice_operator4()
    obstruction1, obstruction2 = normalized_symmetric_cube_obstructions(operator4, point=0)
    assert obstruction1 != 0
    assert obstruction2 != 0

    # Check the precomputed scalar operator against the connection at z=0.
    cyclic_rows = [sp.Matrix([[1, 0, 0, 0]])]
    for _ in range(4):
        previous = cyclic_rows[-1]
        cyclic_rows.append(
            (previous.applyfunc(lambda value: sp.diff(value, z)) + previous * connection4).applyfunc(sp.cancel)
        )
    scalar_residual = cyclic_rows[4]
    for i, coefficient in enumerate(operator4):
        scalar_residual += coefficient * cyclic_rows[i]
    assert all(sp.cancel(value).subs(z, 0) == 0 for value in scalar_residual)

    # The cup-product matrix for x^i dx/y has only J03=4/(3K), J12=4/K.
    K = 10 - z
    cup = sp.zeros(4)
    cup[0, 3] = sp.Rational(4, 3) / K
    cup[3, 0] = -cup[0, 3]
    cup[1, 2] = 4 / K
    cup[2, 1] = -cup[1, 2]
    assert (sp.diff(cup, z) - connection4 * cup - cup * connection4.T).applyfunc(sp.cancel) == sp.zeros(4)

    connection5 = primitive_wedge_connection(connection4)
    assert connection5.shape == (5, 5)
    # A generic evaluation suffices to certify that the first primitive
    # component is cyclic of the full rank five.
    rows = [sp.Matrix([[1, 0, 0, 0, 0]])]
    for _ in range(4):
        previous = rows[-1]
        rows.append(
            (previous.applyfunc(lambda value: sp.diff(value, z)) + previous * connection5).applyfunc(sp.cancel)
        )
    assert sp.Matrix.vstack(*rows).subs(z, 0).det() != 0


def test_projective_two_node_boundary_and_monodromy() -> None:
    A, B, C = two_node_slice_target()
    f = two_node_slice_quintic()
    scale = sp.Rational(1296, 125) * z
    assert sp.cancel(inverse_quintic(A, B, C) - scale * f) == 0
    assert sp.factor(f.subs(z, 0)) == (x + 2) * (2 * x**2 - 2 * x + 3) ** 2 / 4
    assert sp.factor(sp.discriminant(f, x)) == sp.Rational(20503125, 16) * z**2 * (z - 1) ** 2

    connection4 = de_rham_reduce_matrix(f)
    connection5 = primitive_wedge_connection(connection4)
    residue4 = connection4.applyfunc(lambda value: sp.limit(z * value, z, 0))
    residue5 = connection5.applyfunc(lambda value: sp.limit(z * value, z, 0))
    lam = sp.symbols("lam")
    assert residue4.charpoly(lam).as_expr() == lam**4
    assert (residue4.rank(), (residue4**2).rank()) == (2, 0)
    assert residue5.charpoly(lam).as_expr() == lam**5
    assert (residue5.rank(), (residue5**2).rank(), (residue5**3).rank()) == (2, 1, 0)

    # The exact connection has only the three true singularities 0, 1, infinity.
    denominator = sp.lcm([sp.denom(sp.cancel(value)) for value in connection5])
    assert sp.factor(denominator) == 240 * z * (z - 1)

    # Removing the scalar trace, the two finite residues generate all so(5).
    # Hence Sym^2 of this rank-5 K3 system has an irreducible 14-dimensional
    # traceless part; it cannot secretly collapse to a cheap rank-5 CY system.
    residue1 = connection5.applyfunc(lambda value: sp.limit((z - 1) * value, z, 1))
    residue1_traceless = residue1 - sp.trace(residue1) * sp.eye(5) / 5
    assert matrix_lie_algebra_dimension([residue5, residue1_traceless]) == 10


def test_no_collision_native_c4_deformation() -> None:
    A, B, C = sp.symbols("A B C", nonzero=True)
    f = inverse_quintic(A, B, C)

    # Preserving the displayed sigma:(x,y)->(-x,iy) requires f to be odd.
    # Its constant term first forces C=0; then its x^2 term forces B=0.
    assert f.coeff(x, 0) == -6 * C
    assert sp.factor(f.coeff(x, 2).subs(C, 0)) == 90 * B
    assert sp.factor(f.subs({B: 0, C: 0})) == 4 * x * (64 * A**2 * x**4 + 45)

    # This also closes the coordinate-change loophole.  A general C4 curve is
    # x*(x^4+b*x^2-1).  If a nonfixed branch r is moved to infinity and the
    # quintic is depressed, the collision condition [x^3]=0 gives r^4=1,
    # hence b=0.  The apparent first-order deformation is only quadratic
    # tangency at the already-known split-CM point.
    r = sp.symbols("r", nonzero=True)
    b, cubic = c4_moving_branch_obstruction(r)
    assert sp.cancel(b - (1 - r**4) / r**2) == 0
    assert cubic == -4 * (r - 1) ** 2 * (r + 1) ** 2 * (r**2 + 1) ** 2 / (
        5 * r**2 * (r**4 + 1)
    )
    numerator = sp.together(b).as_numer_denom()[0]
    assert sp.expand(numerator + (r - 1) * (r + 1) * (r**2 + 1)) == 0


if __name__ == "__main__":
    tests = [
        test_poster_m3_hensel_lift,
        test_generic_inverse_resultant,
        test_collision_axis_is_split_cm,
        test_nonsymmetric_slice_and_absolute_simplicity_certificate,
        test_picard_fuchs_is_genuinely_rank_four_and_wedge_rank_five,
        test_projective_two_node_boundary_and_monodromy,
        test_no_collision_native_c4_deformation,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
