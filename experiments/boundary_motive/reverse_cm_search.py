"""Reverse search for maximally nodal CM fibers in the collision tower.

For ``m=5`` the inverse curve has arithmetic genus four.  Three simultaneous
nodes would leave an elliptic normalization, so the most direct reverse CM
ansatz is

    H_5(A,B,C,x) = K Q_3(x)^2 E_3(x).

This module records an exact elimination certificate showing that no
nondegenerate fiber of this form exists.  It also records the tempting
equianharmonic special case and verifies that it is only a degree-drop ghost.
"""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp


GENUS2_DIRECTORY = Path(__file__).resolve().parents[1] / "genus2_collision"
if str(GENUS2_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GENUS2_DIRECTORY))

from symmetric_collision_tower import inverse_coordinate_polynomial  # noqa: E402


A, B, C, K, beta, t, x = sp.symbols("A B C K beta t x")


def maximal_elliptic_expected_dimension(m: int) -> int:
    """Naive dimension before quotienting the unavoidable scaling orbit.

    A monic factorization ``Q_(m-2)^2 E_3`` has ``m+1`` shape coefficients.
    The ``m-1`` missing high coefficients of ``H_m`` leave two shape
    parameters.  Together with ``A,B,C,K`` there are six unknowns to match
    the leading coefficient and the ``m`` lower coefficients.
    """

    if m < 3:
        raise ValueError("m must be at least three")
    return 5 - m


def m4_factor_shape() -> tuple[sp.Expr, sp.Expr]:
    """The complete double-node shape after cancelling high coefficients."""

    q = x**2 + x + t
    e = x**3 - 2 * x**2 + (3 - 2 * t) * x + 6 * t - 4
    return q, e


def m4_shape_resultant() -> sp.Expr:
    """Primitive shape obstruction for ``H_4=K Q_2^2 E_3``."""

    return (
        t**4
        * (t - 2) ** 6
        * (3 * t - 2) ** 6
        * (4 * t - 1) ** 5
        * (8 * t**3 - 167 * t**2 + 180 * t - 50) ** 4
    )


def m4_normalization_j(parameter: sp.Expr = t) -> sp.Expr:
    """j-invariant of the residual cubic in the normalized shape chart."""

    return sp.factor(
        64
        * (6 * parameter - 5) ** 3
        / (8 * parameter**3 - 167 * parameter**2 + 180 * parameter - 50)
    )


def m4_unique_elliptic_fiber() -> tuple[sp.Rational, ...]:
    """Return ``(t,A,B,C,K,j)`` for the sole nonsingular survivor."""

    return (
        sp.Integer(2),
        -sp.Rational(7, 18),
        -sp.Rational(1, 3),
        sp.Rational(480, 7),
        sp.Rational(300, 7),
        -sp.Rational(224, 3),
    )


def m4_unique_factorization() -> sp.Expr:
    """Factorization of the sole nonsingular elliptic normalization."""

    parameter, a_value, b_value, c_value, _, _ = m4_unique_elliptic_fiber()
    inverse = inverse_coordinate_polynomial(4, A, B, C, x)
    return sp.factor(
        inverse.subs({A: a_value, B: b_value, C: c_value, t: parameter})
    )


def m5_factor_shape() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``(Q,E,constraint)`` after cancelling the five high powers.

    The chart ``u=1`` is used for the quadratic coefficient of ``Q``.  The
    remaining shape parameter is ``t``.
    """

    w = (-5 + 12 * t - 3 * t**2) / 6
    q = x**3 + x**2 + t * x + w
    e = x**3 - 2 * x**2 + (3 - 2 * t) * x - 2 * (2 - 3 * t + w)
    constraint = -5 + 12 * t - 6 * w - 3 * t**2
    return q, e, sp.expand(constraint)


def sparse_equianharmonic_ghost() -> sp.Expr:
    """The apparent ``j=0`` stratum collapses ``H_5`` to a constant."""

    inverse = inverse_coordinate_polynomial(5, A, B, C, x)
    target = {A: beta**2 / 4, B: beta, C: 140 / beta}
    return sp.factor(inverse.subs(target))


def generic_candidate_polynomial() -> sp.Expr:
    """Only non-pole factor surviving the first five matching equations."""

    return (
        243 * t**10
        + 30114 * t**9
        - 359991 * t**8
        + 1759248 * t**7
        - 4833150 * t**6
        + 8561628 * t**5
        - 10142346 * t**4
        + 8003344 * t**3
        - 3988437 * t**2
        + 1117410 * t
        - 132575
    )


def final_leading_obstruction() -> sp.Expr:
    """Remainder of the last coefficient equation modulo the candidate."""

    return (
        728267634527350922304087052711330621272 * t**9
        + 90642640607159994243053055852468950364111 * t**8
        - 1030175120567576227239268809620422805439624 * t**7
        + 4718535361368913607635141233322336799122332 * t**6
        - 11946244887587121472785635000496178295605320 * t**5
        + 19227011118421561190982377999591122905858762 * t**4
        - 20035789797061494650092102290946129008396504 * t**3
        + 13177901161264670501152186752779555846325116 * t**2
        - 4834697285342329077474609317016562069149808 * t
        + 731327045949227804729261814368764830341695
    )


def generic_branch_is_empty() -> bool:
    """Certify that the candidate and final obstruction have no common root."""

    candidate = sp.Poly(generic_candidate_polynomial(), t, domain=sp.QQ)
    obstruction = sp.Poly(final_leading_obstruction(), t, domain=sp.QQ)
    return sp.gcd(candidate, obstruction).degree() == 0


def exceptional_shape_factors() -> tuple[sp.Expr, ...]:
    """Factors removed by divisions in the generic triangular elimination."""

    return (
        3 * t**2 - 12 * t + 5,
        3 * t**2 + 6 * t - 7,
        3 * t**2 - 4 * t + 3,
        81 * t**4 + 12 * t**3 + 126 * t**2 - 300 * t + 125,
    )


def exceptional_strata_are_empty() -> tuple[bool, ...]:
    """Run the exact saturated number-field Groebner checks.

    This is the expensive part of the certificate.  ``K*L-1`` removes
    degree-drop solutions, so a one-element basis ``[1]`` means that the
    corresponding exceptional shape contains no nondegenerate fiber.
    """

    L = sp.symbols("L")
    q, e, _ = m5_factor_shape()
    product = sp.Poly(sp.expand(q**2 * e), x)
    inverse = sp.Poly(inverse_coordinate_polynomial(5, A, B, C, x), x)
    equations = [
        sp.expand(
            inverse.coeff_monomial(x**power)
            - K * product.coeff_monomial(x**power)
        )
        for power in range(5)
    ]
    equations.extend(
        [sp.expand(inverse.coeff_monomial(x**9) - K), K * L - 1]
    )

    results = []
    for factor in exceptional_shape_factors():
        root = sp.CRootOf(factor, 0)
        basis = sp.groebner(
            [equation.subs(t, root) for equation in equations],
            A,
            B,
            C,
            K,
            L,
            extension=root,
            order="grevlex",
        )
        results.append(len(basis.polys) == 1 and basis.polys[0].as_expr() == 1)
    return tuple(results)


if __name__ == "__main__":
    print("m=4 unique elliptic fiber:", m4_unique_elliptic_fiber())
    print("m=4 factorization:", m4_unique_factorization())
    print("m=5 expected dimension:", maximal_elliptic_expected_dimension(5))
    print("equianharmonic ghost:", sparse_equianharmonic_ghost())
    print("generic branch empty:", generic_branch_is_empty())
    print("exceptional strata empty:", exceptional_strata_are_empty())
