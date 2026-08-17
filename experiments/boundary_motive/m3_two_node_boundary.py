"""Plumbing orders in the known m=3 two-node boundary degeneration."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


x, z = sp.symbols("x z")


def family_polynomial(parameter: sp.Expr = z, variable: sp.Expr = x) -> sp.Expr:
    """Normalized genus-two family at the projective two-node boundary."""

    return sp.expand(
        (1 - parameter) * variable**5
        + 5 * variable**2
        - sp.Rational(15, 4) * variable
        + sp.Rational(9, 2)
    )


def node_polynomial(variable: sp.Expr = x) -> sp.Expr:
    """Quadratic whose two roots are the nodes of the central fiber."""

    return 2 * variable**2 - 2 * variable + 3


def local_node_coefficients() -> tuple[sp.Expr, sp.Expr]:
    """Return the X^2 and z coefficients modulo the node equation.

    At a root r of ``node_polynomial``, put x=r+X.  To lowest order,

        y^2 = A(r) X^2 + B(r) z + higher terms.

    Both A(r) and B(r) are nonzero, so the local product coordinates have
    plumbing parameter a nonzero unit times z.
    """

    polynomial = family_polynomial()
    central = polynomial.subs(z, 0)
    node = node_polynomial()
    quadratic = sp.rem(sp.diff(central, x, 2) / 2, node, x)
    smoothing = sp.rem(sp.diff(polynomial, z).subs(z, 0), node, x)
    return sp.factor(quadratic), sp.factor(smoothing)


@dataclass(frozen=True)
class TwoNodeReport:
    nodes: int
    plumbing_orders: tuple[int, ...]
    vertices: int
    loop_edges: int
    chain_edges: int

    @property
    def has_four_edge_chain(self) -> bool:
        return self.chain_edges >= 4


def two_node_report() -> TwoNodeReport:
    """Stable graph and thickness data for z=0."""

    return TwoNodeReport(
        nodes=2,
        plumbing_orders=(1, 1),
        vertices=1,
        loop_edges=2,
        chain_edges=0,
    )


if __name__ == "__main__":
    polynomial = family_polynomial()
    A, B = local_node_coefficients()
    print("central fiber =", sp.factor(polynomial.subs(z, 0)))
    print("discriminant =", sp.factor(sp.discriminant(polynomial, x)))
    print("local X^2 coefficient modulo node equation =", A)
    print("local z coefficient modulo node equation =", B)
    print(two_node_report())
