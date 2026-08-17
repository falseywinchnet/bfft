"""Exact symbolic audit of the m=3 Aldred collision family.

This module uses SymPy for polynomial elimination and de Rham reduction.  It
keeps the derivation close to the formulas on Cal Aldred's July 2026 poster:

    R_3(t,p) = p^2 - 5(1+t)p + 10(1+t+t^2).

Target coordinates are denoted (A, B, C), while x is the first source
coordinate.  The five x-coordinates in a generic fiber are the roots of
``inverse_quintic``.  Adding infinity gives the six-point branch divisor of a
genus-2 double cover.
"""

from __future__ import annotations

from itertools import combinations

import sympy as sp


x, z = sp.symbols("x z")


def r3(t: sp.Expr, p: sp.Expr) -> sp.Expr:
    return sp.expand(p**2 - 5 * (1 + t) * p + 10 * (1 + t + t**2))


def s3(t: sp.Expr, alpha: sp.Expr) -> sp.Expr:
    """The degree < 3 Hensel lift, modulo alpha^2-5 alpha+10."""

    return sp.expand(
        alpha
        + sp.Rational(5, 3) * (alpha + 2) * t
        + sp.Rational(25, 9) * (alpha + 2) * t**2
    )


def reduce_alpha(expr: sp.Expr, alpha: sp.Symbol) -> sp.Expr:
    modulus = sp.Poly(alpha**2 - 5 * alpha + 10, alpha)
    return sp.Poly(sp.expand(expr), alpha).rem(modulus).as_expr()


def inverse_quintic(A: sp.Expr, B: sp.Expr, C: sp.Expr, X: sp.Expr = x) -> sp.Expr:
    """Polynomial whose roots are generic first source coordinates.

    Elimination starts from

        t^2 - (1+Bx)t + A x^2 = 0,
        C t^5 = x [A^2 x^4 - 5 A x^2 t(1+t)
                    + 10 t^2(1+t+t^2)].

    Their resultant is A^2*x^5 times the quintic below.  The first factor is
    extraneous on the generic chart A*x*t != 0.
    """

    K = (
        A**3 * C**2
        - 30 * A**2 * B * C
        + 256 * A**2
        + 10 * A * B**3 * C
        - 95 * A * B**2
        - B**5 * C
        + 10 * B**4
    )
    L = 10 * A * C - 10 * B**2 * C + 90 * B
    M = 180 - 15 * B * C
    N = -6 * C
    return sp.expand(K * X**5 + L * X**2 + M * X + N)


def collision_axis_quintic(c: sp.Expr, X: sp.Expr = x) -> sp.Expr:
    """The special target (A,B,C)=(c,0,0)."""

    return sp.factor(inverse_quintic(c, 0, 0, X) / 4)


def simple_slice_quintic(parameter: sp.Expr = z, X: sp.Expr = x) -> sp.Expr:
    """A non-symmetric target slice: (A,B,C)=(0,1,z)."""

    return inverse_quintic(0, 1, parameter, X)


def two_node_slice_target(parameter: sp.Expr = z) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """A rational target arc approaching the two-node boundary at z=0.

    The target itself goes to infinity in the affine (A,B,C) chart.  After
    removing the irrelevant scalar 1296*z/125 from its inverse quintic, the
    associated genus-2 curve has the especially simple model returned by
    :func:`two_node_slice_quintic`.
    """

    A = -5 * (2916 * parameter**2 - 3375 * parameter - 3125) / (26244 * parameter**2)
    B = -(27 * parameter + 125) / (81 * parameter)
    C = -sp.Rational(972, 125) * parameter
    return tuple(sp.factor(value) for value in (A, B, C))


def two_node_slice_quintic(parameter: sp.Expr = z, X: sp.Expr = x) -> sp.Expr:
    """Normalized boundary family with two nodes at z=0."""

    return sp.expand(
        (1 - parameter) * X**5
        + 5 * X**2
        - sp.Rational(15, 4) * X
        + sp.Rational(9, 2)
    )


def c4_moving_branch_obstruction(r: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """C4 modulus and depressed-cubic obstruction after changing infinity.

    Every smooth genus-2 curve with an order-four automorphism squaring to the
    hyperelliptic involution can be written over C as

        y^2 = x*(x^4 + b*x^2 - 1).

    Besides the two fixed branch points 0 and infinity, choose a branch r with
    r^4+b*r^2-1=0 and send it to infinity.  After depressing the resulting
    quintic, the returned second expression is its x^3 coefficient.  The
    collision inverse quintic has no x^3 term, so its vanishing is the exact
    test for a conjugate C4 symmetry.
    """

    b = sp.factor((1 - r**4) / r**2)
    X, Z, U, V, q = sp.symbols("X Z U V q")
    binary = X * Z * (X**4 + b * X**2 * Z**2 - Z**4)
    moved = sp.expand(binary.subs({X: r * U + V, Z: U}, simultaneous=True))
    affine = sp.Poly(moved.subs({U: q, V: 1}), q)
    leading = affine.coeff_monomial(q**5)
    quartic = affine.coeff_monomial(q**4)
    shift = sp.factor(-quartic / (5 * leading))
    depressed = sp.Poly(sp.expand(affine.as_expr().subs(q, q + shift)), q)
    cubic = sp.factor(depressed.coeff_monomial(q**3))
    return b, cubic


def simple_slice_operator4(parameter: sp.Symbol = z) -> tuple[sp.Expr, ...]:
    """Precomputed scalar Picard-Fuchs operator for the first de Rham period.

    The return order is c0,c1,c2,c3 in
    ``y'''' + c3*y''' + c2*y'' + c1*y' + c0*y = 0``.  Keeping this exact
    closed form avoids repeatedly expanding the cyclic-vector determinant.
    """

    q = 20 * parameter**2 - 405 * parameter + 2052
    apparent = 10 * parameter - 99
    c0 = 5 * (
        9600 * parameter**3
        - 285260 * parameter**2
        + 2825820 * parameter
        - 9332199
    ) / (144 * (parameter - 10) ** 3 * apparent**2 * q)
    c1 = (
        294000 * parameter**4
        - 11683600 * parameter**3
        + 174113370 * parameter**2
        - 1153188225 * parameter
        + 2864145258
    ) / (18 * (parameter - 10) ** 3 * apparent**2 * q)
    c2 = (
        341500 * parameter**4
        - 13605300 * parameter**3
        + 203256765 * parameter**2
        - 1349545455 * parameter
        + 3360078126
    ) / (9 * (parameter - 10) ** 2 * apparent**2 * q)
    c3 = 10 * (
        180 * parameter**3 - 5408 * parameter**2 + 54162 * parameter - 180819
    ) / ((parameter - 10) * apparent * q)
    return tuple(sp.factor(value) for value in (c0, c1, c2, c3))


def de_rham_reduce_matrix(f: sp.Expr, X: sp.Symbol = x, parameter: sp.Symbol = z) -> sp.Matrix:
    """Gauss-Manin matrix for x^i dx/y, i=0..3, on y^2=f(x,z)."""

    fx = sp.diff(f, X)
    fz = sp.diff(f, parameter)
    rows: list[list[sp.Expr]] = []
    for i in range(4):
        rs = sp.symbols("r0:4")
        qs = sp.symbols("q0:5")
        r = sum(rs[j] * X**j for j in range(4))
        q = sum(qs[j] * X**j for j in range(5))
        p = -sp.Rational(1, 2) * X**i * fz
        identity = sp.Poly(
            sp.expand((r + sp.diff(q, X)) * f - sp.Rational(1, 2) * q * fx - p),
            X,
        )
        equations = [identity.coeff_monomial(X**k) for k in range(9)]
        solution = sp.solve(equations, rs + qs, dict=True, simplify=False)[0]
        rows.append([sp.factor(solution[v]) for v in rs])
    return sp.Matrix(rows)


def scalar_operator(connection: sp.Matrix, parameter: sp.Symbol = z) -> tuple[sp.Expr, ...]:
    """Return c_0,...,c_{n-1} for y^(n)+sum c_k y^(k)=0."""

    n = connection.rows
    rows = [sp.Matrix([[1] + [0] * (n - 1)])]
    for _ in range(n):
        previous = rows[-1]
        rows.append(
            (previous.applyfunc(lambda value: sp.diff(value, parameter)) + previous * connection)
            .applyfunc(sp.cancel)
        )
    cyclic = sp.Matrix.vstack(*rows[:n])
    coefficients = cyclic.T.inv() * (-rows[n].T)
    return tuple(sp.factor(sp.cancel(value)) for value in coefficients)


def wedge_connection(connection: sp.Matrix) -> tuple[sp.Matrix, tuple[tuple[int, int], ...]]:
    """Connection induced on exterior two-forms."""

    pairs = tuple(combinations(range(connection.rows), 2))
    result = sp.zeros(len(pairs))
    for row, (i, j) in enumerate(pairs):
        for a in range(connection.rows):
            if a != j:
                pair = tuple(sorted((a, j)))
                result[row, pairs.index(pair)] += (1 if a < j else -1) * connection[i, a]
            if a != i:
                pair = tuple(sorted((i, a)))
                result[row, pairs.index(pair)] += (1 if i < a else -1) * connection[j, a]
    return result.applyfunc(sp.factor), pairs


def primitive_wedge_connection(connection: sp.Matrix) -> sp.Matrix:
    """Rank-5 primitive exterior square for a depressed quintic.

    In the ordered wedge basis 01,02,03,12,13,23, the cup product is
    proportional to (0,0,1/3,1,0,0).  Its kernel has the constant basis
    01, 02, 3*03-12, 13, 23.
    """

    wedge, pairs = wedge_connection(connection)
    assert pairs == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    basis = sp.Matrix(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 3, -1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    differentiated = (basis * wedge).applyfunc(sp.cancel)
    output = sp.zeros(5)
    for i in range(5):
        # The preservation assertion is the Riemann-bilinear/polarization check.
        assert sp.cancel(differentiated[i, 2] / 3 + differentiated[i, 3]) == 0
        output[i, 0] = differentiated[i, 0]
        output[i, 1] = differentiated[i, 1]
        output[i, 2] = sp.cancel(differentiated[i, 2] / 3)
        output[i, 3] = differentiated[i, 4]
        output[i, 4] = differentiated[i, 5]
    return output.applyfunc(sp.factor)


def normalized_symmetric_cube_obstructions(
    coefficients: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr],
    parameter: sp.Symbol = z,
    point: sp.Expr | None = None,
) -> tuple[sp.Expr, sp.Expr]:
    """Necessary symmetric-cube identities after removing the D^3 term.

    A symmetric cube of u''=r*u has reduced operator

        D^4 - 10r D^2 - 10r' D + 9r^2 - 3r''.

    Therefore b1-b2' and b0-(9 b2^2/100+3 b2''/10) must both vanish.
    """

    a0, a1, a2, a3 = coefficients
    h = -a3 / 4
    hp = sp.diff(h, parameter)
    hpp = sp.diff(h, parameter, 2)
    b2 = 6 * h**2 + 6 * hp + 3 * a3 * h + a2
    b1 = (
        4 * h**3
        + 12 * h * hp
        + 4 * hpp
        + 3 * a3 * (h**2 + hp)
        + 2 * a2 * h
        + a1
    )
    b0 = (
        h**4
        + 6 * h**2 * hp
        + 3 * hp**2
        + 4 * h * hpp
        + sp.diff(h, parameter, 3)
        + a3 * (h**3 + 3 * h * hp + hpp)
        + a2 * (hp + h**2)
        + a1 * h
        + a0
    )
    first = b1 - sp.diff(b2, parameter)
    second = b0 - sp.Rational(9, 100) * b2**2 - sp.Rational(3, 10) * sp.diff(b2, parameter, 2)
    if point is not None:
        return sp.simplify(first.subs(parameter, point)), sp.simplify(second.subs(parameter, point))
    first = sp.factor(sp.cancel(first))
    second = sp.factor(sp.cancel(second))
    return first, second


def matrix_lie_algebra_dimension(generators: list[sp.Matrix]) -> int:
    """Dimension of the matrix Lie algebra generated over the rationals."""

    if not generators:
        return 0
    size = generators[0].rows * generators[0].cols
    basis: list[sp.Matrix] = []

    def flattened(matrix: sp.Matrix) -> sp.Matrix:
        return matrix.reshape(size, 1)

    def add_if_independent(matrix: sp.Matrix) -> bool:
        candidates = basis + [matrix]
        columns = sp.Matrix.hstack(*(flattened(value) for value in candidates))
        if columns.rank() > len(basis):
            basis.append(matrix)
            return True
        return False

    for generator in generators:
        add_if_independent(generator)
    while True:
        old_basis = list(basis)
        changed = False
        for i, left in enumerate(old_basis):
            for right in old_basis[i + 1 :]:
                changed = add_if_independent(left * right - right * left) or changed
        if not changed:
            return len(basis)


def point_counts_mod_7() -> tuple[int, int]:
    """Counts for y^2=9x^5+80x^2+165x-6 over F_7 and F_49."""

    p = 7
    coefficients = [-6, 165, 80, 0, 0, 9]

    def legendre(value: int) -> int:
        value %= p
        if value == 0:
            return 0
        return 1 if pow(value, (p - 1) // 2, p) == 1 else -1

    n1 = p + 1
    for a in range(p):
        value = sum(coefficient * pow(a, i, p) for i, coefficient in enumerate(coefficients))
        n1 += legendre(value)

    # F_49 = F_7[u]/(u^2-3); 3 is a nonsquare modulo 7.
    d = 3

    def multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        a, b = left
        c, e = right
        return ((a * c + d * b * e) % p, (a * e + b * c) % p)

    def power(value: tuple[int, int], exponent: int) -> tuple[int, int]:
        result = (1, 0)
        while exponent:
            if exponent & 1:
                result = multiply(result, value)
            value = multiply(value, value)
            exponent >>= 1
        return result

    def evaluate(value: tuple[int, int]) -> tuple[int, int]:
        result = (0, 0)
        for coefficient in reversed(coefficients):
            result = multiply(result, value)
            result = ((result[0] + coefficient) % p, result[1])
        return result

    n2 = p * p + 1
    for a in range(p):
        for b in range(p):
            value = evaluate((a, b))
            if value != (0, 0):
                n2 += 1 if power(value, (p * p - 1) // 2) == (1, 0) else -1
    return n1, n2


def main() -> None:
    f = simple_slice_quintic()
    connection4 = de_rham_reduce_matrix(f)
    operator4 = simple_slice_operator4()
    connection5 = primitive_wedge_connection(connection4)
    print("inverse quintic:", inverse_quintic(sp.Symbol("A"), sp.Symbol("B"), sp.Symbol("C")))
    print("slice discriminant:", sp.factor(sp.discriminant(f, x)))
    print("rank-4 operator coefficients c0..c3:")
    for coefficient in operator4:
        print(sp.factor(coefficient))
    print("symmetric-cube obstructions at z=0:", normalized_symmetric_cube_obstructions(operator4, point=0))
    print("primitive exterior-square connection:")
    print(connection5)
    print("point counts over F_7,F_49:", point_counts_mod_7())
    boundary = two_node_slice_quintic()
    boundary4 = de_rham_reduce_matrix(boundary)
    boundary5 = primitive_wedge_connection(boundary4)
    residue4 = boundary4.applyfunc(lambda value: sp.limit(z * value, z, 0))
    residue5 = boundary5.applyfunc(lambda value: sp.limit(z * value, z, 0))
    lam = sp.Symbol("lambda")
    print("two-node slice discriminant:", sp.factor(sp.discriminant(boundary, x)))
    print("rank-4 residue at z=0:", sp.factor(residue4.charpoly(lam).as_expr()), residue4.rank())
    print(
        "rank-5 residue at z=0:",
        sp.factor(residue5.charpoly(lam).as_expr()),
        (residue5.rank(), (residue5**2).rank(), (residue5**3).rank()),
    )


if __name__ == "__main__":
    main()
