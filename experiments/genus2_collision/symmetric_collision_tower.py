"""Exact screen for CM elliptic factors in symmetric collision fibers.

For a root tau of rho_m, the symmetric fiber contains the two first-source
coordinates x with x^2=tau*(1-tau)/c. After scaling c away, its curve is

    C_m: y^2 = x Q_m(x^2),

where Q_m(u) is the resultant of rho_m(t) and u-t(1-t).
"""

from __future__ import annotations

from itertools import product

import sympy as sp


t, u, x, T = sp.symbols("t u x T")


def r_polynomial(m: int, t_variable: sp.Symbol = t, p_variable: sp.Expr | None = None) -> sp.Expr:
    """Aldred's general polynomial R_m(t,p)."""

    if m < 2:
        raise ValueError("m must be at least 2")
    if p_variable is None:
        p_variable = sp.Symbol("p")
    n = m - 1
    result = 0
    for j in range(n + 1):
        for i in range(n - j + 1):
            result += (
                (-1) ** (n - j)
                * sp.binomial(2 * n + 1, n - j)
                * sp.binomial(i + j, j)
                * sp.binomial(n - i, j)
                / sp.binomial(n, j)
                * t_variable**i
                * p_variable**j
            )
    return sp.expand(result)


def _positive_primitive(poly: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    primitive = sp.primitive(sp.Poly(poly, variable))[1]
    if primitive.LC() < 0:
        primitive = -primitive
    return primitive.as_expr()


def rho_polynomial(m: int, variable: sp.Symbol = t) -> sp.Expr:
    """Primitive integral normalization of rho_m."""

    if m < 2:
        raise ValueError("m must be at least 2")
    series = sum(
        sp.rf(1 - m, k)
        * sp.rf(1, k)
        / (sp.rf(sp.Rational(3, 2) - m, k) * sp.factorial(k))
        * variable**k
        for k in range(m)
    )
    scaled = (-1) ** (m - 1) * sp.binomial(2 * m - 2, m - 1) * series
    return _positive_primitive(sp.expand(scaled), variable)


def symmetric_q_polynomial(m: int, variable: sp.Symbol = u) -> sp.Expr:
    """Q_m(u)=prod_{rho_m(tau)=0}(u-tau*(1-tau)), up to content."""

    resultant = sp.resultant(rho_polynomial(m, t), variable - t * (1 - t), t)
    return _positive_primitive(sp.expand(resultant), variable)


def symmetric_curve_polynomial(m: int, variable: sp.Symbol = x) -> sp.Expr:
    """Odd-degree model for the genus-(m-1) symmetric collision curve."""

    return sp.expand(variable * symmetric_q_polynomial(m, u).subs(u, variable**2))


def inverse_coordinate_polynomial(
    m: int,
    A: sp.Expr,
    B: sp.Expr,
    C: sp.Expr,
    variable: sp.Symbol = x,
) -> sp.Expr:
    """Generic first-source-coordinate polynomial for the F_m inverse fiber."""

    local_t = sp.Symbol("local_t")
    equation1 = local_t**2 - (1 + B * variable) * local_t + A * variable**2
    numerator = sp.cancel(
        r_polynomial(m, local_t, A * variable**2 / local_t) * local_t ** (m - 1)
    )
    equation2 = C * local_t ** (2 * m - 1) - variable * numerator
    resultant = sp.resultant(equation1, equation2, local_t)
    chart_factor = A ** (m - 1) * variable ** (2 * m - 1)
    return sp.expand(sp.cancel(resultant / chart_factor))


def c4_collision_tangent_ranks(m: int) -> tuple[int, int, int]:
    """Ranks of the C4, collision, and intersection tangent spaces.

    Binary forms include every infinitesimal PGL_2 coordinate change.  The
    intersection rank is measured before quotienting the one-dimensional
    isotrivial A-scaling direction.
    """

    A, B, C = sp.symbols("A B C", nonzero=True)
    X, Z = sp.symbols("X Z")
    degree = 2 * m
    inverse = inverse_coordinate_polynomial(m, A, B, C, x)

    def homogenize(polynomial: sp.Expr) -> sp.Expr:
        source = sp.Poly(polynomial, x)
        return sp.expand(
            sum(
                source.coeff_monomial(x**power) * X**power * Z ** (degree - power)
                for power in range(2 * m)
            )
        )

    symmetric = homogenize(inverse.subs({B: 0, C: 0}))
    collision_derivatives = [
        homogenize(sp.diff(inverse, parameter).subs({B: 0, C: 0}))
        for parameter in (A, B, C)
    ]
    orbit = [
        X * sp.diff(symmetric, Z),
        Z * sp.diff(symmetric, X),
        X * sp.diff(symmetric, X) - Z * sp.diff(symmetric, Z),
    ]
    odd_deformations = [X**power * Z ** (degree - power) for power in range(1, degree, 2)]

    def vector(polynomial: sp.Expr) -> sp.Matrix:
        expanded = sp.expand(polynomial)
        return sp.Matrix(
            [expanded.coeff(X, degree - power).coeff(Z, power) for power in range(degree + 1)]
        )

    c4_space = sp.Matrix.hstack(*(vector(value) for value in orbit + odd_deformations))
    collision_space = sp.Matrix.hstack(*(vector(value) for value in collision_derivatives))
    c4_rank = c4_space.rank()
    collision_rank = collision_space.rank()
    joined_rank = c4_space.row_join(collision_space).rank()
    return c4_rank, collision_rank, c4_rank + collision_rank - joined_rank


def _irreducible_modulus(p: int, degree: int) -> list[int]:
    if degree == 1:
        return [0, 1]
    X = sp.Symbol("X")
    for coefficients in product(range(p), repeat=degree):
        if coefficients[0] == 0:
            continue
        candidate = X**degree + sum(coefficients[i] * X**i for i in range(degree))
        if sp.Poly(candidate, X, modulus=p).is_irreducible:
            return list(coefficients) + [1]
    raise RuntimeError("failed to find an irreducible modulus")


def count_hyperelliptic_points(coefficients: list[int], p: int, degree: int) -> int:
    """Count an odd-degree y^2=f(x) over F_(p^degree)."""

    modulus = _irreducible_modulus(p, degree)
    q = p**degree
    vectors: list[list[int]] = []
    for encoded in range(q):
        value = encoded
        vector = []
        for _ in range(degree):
            vector.append(value % p)
            value //= p
        vectors.append(vector)

    def encode(vector: list[int]) -> int:
        value = 0
        for coefficient in reversed(vector):
            value = value * p + coefficient % p
        return value

    def add(left: int, right: int) -> int:
        return encode([(vectors[left][i] + vectors[right][i]) % p for i in range(degree)])

    def multiply(left: int, right: int) -> int:
        accumulator = [0] * (2 * degree - 1)
        for i, a in enumerate(vectors[left]):
            for j, b in enumerate(vectors[right]):
                accumulator[i + j] = (accumulator[i + j] + a * b) % p
        for power in range(2 * degree - 2, degree - 1, -1):
            leading = accumulator[power]
            for i in range(degree):
                accumulator[power - degree + i] = (
                    accumulator[power - degree + i] - leading * modulus[i]
                ) % p
        return encode(accumulator[:degree])

    def field_power(value: int, exponent: int) -> int:
        result = 1
        while exponent:
            if exponent & 1:
                result = multiply(result, value)
            value = multiply(value, value)
            exponent >>= 1
        return result

    count = q + 1
    for argument in range(q):
        value = 0
        for coefficient in reversed(coefficients):
            value = add(multiply(value, argument), coefficient % p)
        if value:
            count += 1 if field_power(value, (q - 1) // 2) == 1 else -1
    return count


def point_counts(m: int, p: int) -> list[int]:
    """Counts over F_p,...,F_(p^(m-1)) for C_m."""

    polynomial = sp.Poly(symmetric_curve_polynomial(m, x), x)
    coefficients = [int(polynomial.coeff_monomial(x**i)) for i in range(polynomial.degree() + 1)]
    return [count_hyperelliptic_points(coefficients, p, degree) for degree in range(1, m)]


def weil_polynomial(counts: list[int], p: int, variable: sp.Symbol = T) -> sp.Expr:
    """Recover the Frobenius characteristic polynomial from genus-many counts."""

    genus = len(counts)
    power_sums = [p**degree + 1 - counts[degree - 1] for degree in range(1, genus + 1)]
    elementary = [sp.Integer(1)]
    for degree in range(1, genus + 1):
        elementary.append(
            sp.simplify(
                sum(
                    (-1) ** (i - 1) * elementary[degree - i] * power_sums[i - 1]
                    for i in range(1, degree + 1)
                )
                / degree
            )
        )
    coefficients = [(-1) ** degree * elementary[degree] for degree in range(genus + 1)]
    coefficients.extend([sp.Integer(0)] * genus)
    for degree in range(genus):
        coefficients[2 * genus - degree] = p ** (genus - degree) * coefficients[degree]
    return sp.expand(
        sum(coefficients[degree] * variable ** (2 * genus - degree) for degree in range(2 * genus + 1))
    )


def real_weil_polynomial(weil: sp.Expr, p: int, variable: sp.Symbol = u) -> sp.Expr:
    """Return g with Weil(X)=X^genus*g(X+p/X)."""

    X = sp.Symbol("X")
    genus = sp.Poly(weil, T).degree() // 2
    unknowns = sp.symbols(f"a0:{genus}")
    candidate = variable**genus + sum(unknowns[i] * variable**i for i in range(genus))
    identity = sp.Poly(sp.expand(X**genus * candidate.subs(variable, X + p / X) - weil.subs(T, X)), X)
    solution = sp.solve(identity.all_coeffs(), unknowns, dict=True)[0]
    return sp.expand(candidate.subs(solution))


def main() -> None:
    for m in range(2, 6):
        print(f"m={m}: rho={rho_polynomial(m)}, Q={symmetric_q_polynomial(m)}")
        print(" curve:", symmetric_curve_polynomial(m))
    for m in (4, 5):
        counts = point_counts(m, 13)
        weil = weil_polynomial(counts, 13)
        real = real_weil_polynomial(weil, 13)
        print(f"m={m}, p=13: counts={counts}")
        print(" Weil:", weil)
        print(" real Weil:", real)
        print(" real Galois group:", sp.Poly(real, u).galois_group())


if __name__ == "__main__":
    main()
