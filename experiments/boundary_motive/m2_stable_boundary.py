"""Stable marked-boundary audit for the m=2 collision slice.

The symmetric slice hides five marked places at t=0: three zeros and two
poles of the first source coordinate.  This module resolves that collision
under the smallest semistable base change and records the actual plumbing
thickness.  It also keeps the unrelated cubic ramification of the target map
separate from the plumbing parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


t, epsilon, u, T = sp.symbols("t epsilon u T")


def numerator(parameter: sp.Expr = epsilon, variable: sp.Expr = t) -> sp.Expr:
    """Numerator divisor of x_epsilon, with its scalar 4 removed."""

    return sp.expand(variable * (variable**2 + parameter))


def denominator(parameter: sp.Expr = epsilon, variable: sp.Expr = t) -> sp.Expr:
    """Denominator divisor of x_epsilon."""

    return sp.expand(
        4 * variable**4 - (4 * parameter + 3) * variable**2 - parameter
    )


def source_coordinate(
    parameter: sp.Expr = epsilon, variable: sp.Expr = t
) -> sp.Expr:
    """The perturbed first source coordinate x_epsilon(t)."""

    return sp.cancel(4 * numerator(parameter, variable) / denominator(parameter, variable))


def pole_square_roots(parameter: sp.Expr = epsilon) -> tuple[sp.Expr, sp.Expr]:
    """Roots in w=t^2 of the pole equation, small root first."""

    radical = sp.sqrt(16 * parameter**2 + 40 * parameter + 9)
    return (
        sp.cancel((4 * parameter + 3 - radical) / 8),
        sp.cancel((4 * parameter + 3 + radical) / 8),
    )


def bubble_polynomials() -> tuple[sp.Expr, sp.Expr]:
    """Strict-transform equations after epsilon=u^2 and t=u*T.

    The powers of u common to each divisor are removed.  Setting u=0 then
    displays all five formerly coincident markings on the bubble.
    """

    zero_transform = sp.cancel(numerator(u**2, u * T) / u**3)
    pole_transform = sp.cancel(denominator(u**2, u * T) / u**2)
    return sp.expand(zero_transform), sp.expand(pole_transform)


def target_infinity_coordinate(variable: sp.Expr = t) -> sp.Expr:
    """Local target coordinate v=1/c on the unperturbed C-axis slice."""

    # c(t)=(4t^2-1)/(2t^3), so target infinity is v=1/c=0.
    return sp.cancel(2 * variable**3 / (4 * variable**2 - 1))


def order_at_zero(expression: sp.Expr, variable: sp.Symbol) -> int:
    """Exact valuation at variable=0 for a nonzero rational expression."""

    numerator_part, denominator_part = sp.cancel(expression).as_numer_denom()
    numerator_poly = sp.Poly(numerator_part, variable)
    denominator_poly = sp.Poly(denominator_part, variable)

    def polynomial_order(polynomial: sp.Poly) -> int:
        return min(monomial[0] for monomial, coefficient in polynomial.terms() if coefficient)

    return polynomial_order(numerator_poly) - polynomial_order(denominator_poly)


@dataclass(frozen=True)
class StableBoundaryReport:
    """Combinatorial and valuation data for the resolved marked curve."""

    base_change: str
    bubble_marks: tuple[str, ...]
    main_marks: tuple[str, ...]
    vertices: int
    edges: int
    plumbing_equation: str
    plumbing_order_after_base_change: int
    plumbing_order_in_epsilon: sp.Rational
    target_ramification_order: int

    @property
    def has_four_edge_chain(self) -> bool:
        return self.edges >= 4


def stable_boundary_report() -> StableBoundaryReport:
    """Return the minimal stable reduction of the fivefold t=0 cluster.

    On the bubble T=t/u, the five labels specialize to

        0, +/-i, +/-i/sqrt(3).

    The main and bubble charts meet with t*(1/T)=u.  Hence the sole node has
    q=u after epsilon=u^2: order one on the semistable base, not order five.
    """

    return StableBoundaryReport(
        base_change="epsilon=u^2",
        bubble_marks=("0", "+i", "-i", "+i/sqrt(3)", "-i/sqrt(3)"),
        main_marks=("infinity (zero)", "+sqrt(3)/2 (pole)", "-sqrt(3)/2 (pole)"),
        vertices=2,
        edges=1,
        plumbing_equation="t*(1/T)=u",
        plumbing_order_after_base_change=1,
        plumbing_order_in_epsilon=sp.Rational(1, 2),
        target_ramification_order=order_at_zero(target_infinity_coordinate(t), t),
    )


if __name__ == "__main__":
    small_pole, distant_pole = pole_square_roots()
    zero_bubble, pole_bubble = bubble_polynomials()
    report = stable_boundary_report()
    print("x_epsilon(t) =", source_coordinate())
    print("small pole t^2 =", sp.series(small_pole, epsilon, 0, 3))
    print("distant pole t^2 =", sp.series(distant_pole, epsilon, 0, 3))
    print("bubble zeros:", sp.factor(zero_bubble.subs(u, 0)))
    print("bubble poles:", sp.factor(pole_bubble.subs(u, 0)))
    print(report)
