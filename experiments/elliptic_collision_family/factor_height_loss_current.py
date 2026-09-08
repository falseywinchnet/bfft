"""Symbolically factor the first shell of the sixth-section height current.

This is a derivation script, not a specialization search.  It works in the
function field Q(c,t), applies the exact chord law, removes only the known
degenerate boundary components, and then pulls the surviving relation
divisors back to the compatible-current norm conic.
"""

from __future__ import annotations

import sympy as sp


c, t, v = sp.symbols("c t v")
T = t * (1 - t**2)
A = (1 + t**2) ** 2
r = T * c * (1 + c)
s = c * r
H = T * c**2 * (1 + c) ** 2
q = H * (1 + t**2)
h_minus = H * (1 - 2 * t - t**2)
h_plus = H * (1 + 2 * t - t**2)
x_extra = r - s


def x_sum_with_center(x: sp.Expr, y_twice: sp.Expr, sign: int) -> sp.Expr:
    """x-coordinate of P0 + sign*P, with twice-y coordinates supplied."""

    slope = (sign * y_twice - q) / (2 * x)
    return sp.cancel(slope**2 - x)


def primitive_numerator(expression: sp.Expr) -> sp.Expr:
    """Return the primitive factored numerator of a rational expression."""

    numerator = sp.together(expression).as_numer_denom()[0]
    return sp.factor(sp.primitive(sp.Poly(numerator, c, t).as_expr())[1])


def nondegenerate_part(expression: sp.Expr) -> sp.Expr:
    """Strip the support and angle boundary components from a numerator."""

    boundary = {c, c + 1, t, t - 1, t + 1}
    coefficient, factors = sp.factor_list(primitive_numerator(expression))
    kept = [factor**power for factor, power in factors if factor not in boundary]
    return sp.factor(sp.prod(kept))


def center_shell_factors() -> dict[str, sp.Expr]:
    points = {
        "lower_r": (r, h_minus),
        "lower_s": (s, h_minus),
        "upper_minus_r": (-r, h_plus),
        "upper_minus_s": (-s, h_plus),
    }
    return {
        f"P0{'+' if sign == 1 else '-'}{name}": nondegenerate_part(
            x_sum_with_center(x, y_twice, sign) - x_extra
        )
        for name, (x, y_twice) in points.items()
        for sign in (1, -1)
    }


def compatible_conic_pullback(form: sp.Expr) -> sp.Expr:
    """Pull a c,t divisor back through c=(A-12T)/(T*v^2-A-12T)."""

    conic_c = (A - 12 * T) / (T * v**2 - A - 12 * T)
    numerator = sp.together(form.subs(c, conic_c)).as_numer_denom()[0]
    primitive = sp.primitive(sp.Poly(numerator, v, t).as_expr())[1]
    coefficient, factors = sp.factor_list(primitive)
    boundary = {t, t - 1, t + 1}
    return sp.factor(
        sp.prod(factor**power for factor, power in factors if factor not in boundary)
    )


def base_seventh_relation() -> sp.Expr:
    """Divisor seen at t=1/7: P0+P_r has the abscissa of P_s."""

    return nondegenerate_part(x_sum_with_center(r, h_minus, 1) - s)


def add(
    left: tuple[sp.Expr, sp.Expr], right: tuple[sp.Expr, sp.Expr]
) -> tuple[sp.Expr, sp.Expr]:
    """Generic chord addition on y^2=x^3-m*x+q^2/4."""

    x1, y1 = left
    x2, y2 = right
    slope = sp.cancel((y2 - y1) / (x2 - x1))
    x3 = sp.cancel(slope**2 - x1 - x2)
    y3 = sp.cancel(slope * (x1 - x3) - y1)
    return x3, y3


def base_fifth_relation() -> tuple[sp.Expr, sp.Expr]:
    """Divisors for P0+Q_-r = P_r+P_s+Q_-s, seen at t=1/5."""

    p0 = (sp.Integer(0), q / 2)
    p_r = (r, h_minus / 2)
    p_s = (s, h_minus / 2)
    q_minus_r = (-r, h_plus / 2)
    q_minus_s = (-s, h_plus / 2)
    left = add(p0, q_minus_r)
    right = add(add(p_r, p_s), q_minus_s)
    return nondegenerate_part(left[0] - right[0]), nondegenerate_part(
        left[1] - right[1]
    )


def base_seventh_y_residues() -> tuple[sp.Expr, sp.Expr]:
    """Check which sign accompanies the t=1/7 abscissa coincidence."""

    p0_plus_r = add((sp.Integer(0), q / 2), (r, h_minus / 2))
    return (
        nondegenerate_part(p0_plus_r[1] - h_minus / 2),
        nondegenerate_part(p0_plus_r[1] + h_minus / 2),
    )


def main() -> None:
    factors = center_shell_factors()
    print("CENTER-SHELL X-COINCIDENCE FACTORS")
    for name, factor in factors.items():
        print(f"{name}: {factor}")

    print("\nCOMPATIBLE-CURRENT PULLBACKS")
    for name, factor in factors.items():
        print(f"{name}: {compatible_conic_pullback(factor)}")

    seventh = base_seventh_relation()
    print("\nBASE-FIVE t=1/7 X-COINCIDENCE")
    print(seventh)
    print("pullback:", compatible_conic_pullback(seventh))
    print("y-P_s, y+P_s:", base_seventh_y_residues())

    fifth_x, fifth_y = base_fifth_relation()
    print("\nBASE-FIVE t=1/5 RELATION")
    print("x divisor:", fifth_x)
    print("y divisor:", fifth_y)
    print("x pullback:", compatible_conic_pullback(fifth_x))
    print("y pullback:", compatible_conic_pullback(fifth_y))


if __name__ == "__main__":
    main()
