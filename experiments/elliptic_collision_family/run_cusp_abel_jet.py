"""Expose the five section currents as Abel jets at the rational cusp.

This script uses separated scale variables before substituting ``R=c(1+c)``.
That keeps the generic determinant factorization small enough for exact SymPy
arithmetic.  It is intended for the repository's M4 execution path.
"""

from __future__ import annotations

import sympy as sp


def abel_jet_data(order: int = 6):
    t, x, c = sp.symbols("t x c")
    radius, support_norm = sp.symbols("R h", nonzero=True)
    fiber_square = (
        radius**4 / 4
        + t * (x**3 - radius**2 * support_norm * x)
        + 2 * t**3 * radius**2 * support_norm * x
        - radius**4 * t**4 / 2
        - t**5 * radius**2 * support_norm * x
    )
    differential = sp.series(
        1 / (2 * sp.sqrt(fiber_square)), t, 0, order
    ).removeO()
    primitive = sp.integrate(differential, x)
    roles = (sp.Integer(1), c, -1 - c, -1, -c, 1 + c)
    rows = []
    for role in roles:
        endpoint = role * radius * (1 - t * t)
        current = sp.series(primitive.subs(x, endpoint), t, 0, order).removeO()
        rows.append([sp.factor(current.coeff(t, degree)) for degree in range(order)])
    matrix = sp.Matrix(rows)
    record_matrix = matrix.subs(
        {
            c: sp.Rational(7, 3),
            radius: sp.Rational(70, 9),
            support_norm: sp.Rational(79, 9),
        }
    )
    rank_growth = tuple(
        record_matrix[:, :width].rank() for width in range(1, order + 1)
    )

    # Orders 0,1,2,3,5 are the first five independent current layers.
    minor = sp.Matrix(
        [
            [rows[row][degree] for degree in (0, 1, 2, 3, 5)]
            for row in range(5)
        ]
    )
    fourth_layer = sp.Matrix([rows[row][4] for row in range(5)])
    closure = minor.inv() * fourth_layer
    closure = tuple(
        sp.factor(
            coefficient.subs(
                {
                    radius: c * (1 + c),
                    support_norm: 1 + c + c * c,
                }
            )
        )
        for coefficient in closure
    )
    determinant = sp.factor(minor.det())
    determinant = sp.factor(
        determinant.subs(
            {
                radius: c * (1 + c),
                support_norm: 1 + c + c * c,
            }
        )
    )
    return rank_growth, record_matrix.T.nullspace(), determinant, closure


def main() -> None:
    ranks, relations, determinant, closure = abel_jet_data()
    print("rank growth through t^0,...,t^5:", ranks)
    print("section-current relations:", relations)
    print("generic five-current determinant:")
    print(determinant)
    print("fourth-layer closure in orders 0,1,2,3,5:")
    print(closure)


if __name__ == "__main__":
    main()
