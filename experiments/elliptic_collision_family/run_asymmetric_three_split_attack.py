"""Asymmetric three-horizontal-cubic rank-six construction.

Let a normalized split support have product current ``K`` and split root
``S``, so ``S^2+27*K^2=4``.  Its torus double has current ``S*K``.  We put
the three support triples with currents

    +K, -K, S*K

on one elliptic curve.  Unlike the four-split construction, this asks for
only three square heights and leaves one rational conic free.

Writing ``rho=c*K`` and ``z=1/(2*c)``, the first two heights are
``rho^2*(1+/-z)``.  The third is rational precisely when

    eta^2 = 4*c^2+4*S*c+1.

The line ``eta=1+ell*c`` through the boundary point ``c=0`` gives

    c = 2*(2*S-ell)/(ell^2-4).

Thus every rational ``(t,ell)`` away from the explicit boundaries gives
three complete horizontal cubics and generically six independent sections.
The exceptional chord ``ell=4`` is the first equianharmonic cusp blow-up;
at ``t=1/2`` it gives an exact rank-six curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from run_four_split_torus_attack import (
    current_and_split_root,
    support_coordinates,
)


Q = Fraction


@dataclass(frozen=True)
class AsymmetricThreeSplitFiber:
    parameter: Fraction
    chord_slope: Fraction
    support_multiplier: int
    rho: Fraction
    a6: Fraction
    currents: tuple[Fraction, Fraction, Fraction]
    heights: tuple[Fraction, Fraction, Fraction]
    triples: tuple[tuple[Fraction, Fraction, Fraction], ...]

    @property
    def short_model(self) -> tuple[Fraction, Fraction, Fraction]:
        return Q(0), -(self.rho * self.rho), self.a6

    @property
    def points(self) -> tuple[tuple[Fraction, Fraction], ...]:
        return tuple(
            (root, height)
            for triple, height in zip(self.triples, self.heights)
            for root in triple
        )

    def verify(self) -> bool:
        m = self.rho * self.rho
        if len(set(self.currents)) != 3:
            return False
        for current, height, triple in zip(
            self.currents, self.heights, self.triples
        ):
            if sum(triple) != 0:
                return False
            if sum(root * root for root in triple) != 2 * m:
                return False
            if -triple[0] * triple[1] * (
                triple[0] + triple[1]
            ) != current:
                return False
            for root in triple:
                if height * height != root**3 - m * root + self.a6:
                    return False
        return True


def build_asymmetric_three_split_fiber(
    t: Fraction, chord_slope: Fraction
) -> AsymmetricThreeSplitFiber:
    """Build the torus-double three-split fiber at ``(t,chord_slope)``."""

    return build_asymmetric_torus_multiplier_fiber(t, 2, chord_slope)


def torus_current_multiplier(split_root: Fraction, multiple: int) -> Fraction:
    """Return ``U_(multiple-1)(split_root/2)`` by its exact recurrence."""

    if multiple < 1:
        raise ValueError("the torus multiple must be positive")
    previous, current = Q(0), Q(1)
    for _index in range(1, multiple):
        previous, current = current, split_root * current - previous
    return current


def torus_multiple_support(
    t: Fraction, multiple: int
) -> tuple[Fraction, Fraction, Fraction]:
    """Return the zero-sum support triple of the torus multiple."""

    if multiple < 1:
        raise ValueError("the torus multiple must be positive")
    _a, b, x = support_coordinates(Q(t))
    multiple_x, multiple_b = Q(2), Q(0)
    for _index in range(multiple):
        multiple_x, multiple_b = (
            (multiple_x * x - 3 * multiple_b * b) / 2,
            (multiple_x * b + multiple_b * x) / 2,
        )
    multiple_a = (multiple_x - multiple_b) / 2
    return multiple_a, multiple_b, -multiple_a - multiple_b


def build_asymmetric_torus_multiplier_fiber(
    t: Fraction, support_multiplier: int, chord_slope: Fraction
) -> AsymmetricThreeSplitFiber:
    """Build the ``(+K,-K,K_n)`` fiber on the adjacent multiplier ray."""

    t, ell = Q(t), Q(chord_slope)
    if ell in (Q(-2), Q(2)):
        raise ValueError("the height-conic chord is tangent at infinity")
    k, split_root = current_and_split_root(t)
    if not k:
        raise ValueError("the support current is degenerate")
    multiplier = torus_current_multiplier(split_root, support_multiplier)
    c = 2 * (2 * multiplier - ell) / (ell * ell - 4)
    if not c:
        raise ValueError("the height-conic point is degenerate")
    eta = 1 + ell * c
    if eta * eta != 4 * c * c + 4 * multiplier * c + 1:
        raise ArithmeticError("height-conic parametrization failed")

    rho = c * k
    z = Q(1, 2) / c
    u = rho * rho
    a6 = rho**4 * (1 + z * z)
    base = tuple(rho * value for value in support_coordinates(t)[:2])
    base_triple = (base[0], base[1], -base[0] - base[1])
    multiple_triple = tuple(
        rho * value for value in torus_multiple_support(t, support_multiplier)
    )
    current = rho**3 * k
    multiple_current = -multiple_triple[0] * multiple_triple[1] * (
        multiple_triple[0] + multiple_triple[1]
    )
    if multiple_current != multiplier * current:
        raise ArithmeticError("torus current multiplier identity failed")
    currents = (current, -current, multiple_current)
    heights = (
        u * (1 + z),
        u * (1 - z),
        u * eta / (2 * c),
    )
    triples = (
        base_triple,
        tuple(-value for value in base_triple),
        multiple_triple,
    )
    fiber = AsymmetricThreeSplitFiber(
        t, ell, support_multiplier, rho, a6, currents, heights, triples
    )
    if not fiber.verify():
        raise ArithmeticError("asymmetric three-split verification failed")
    return fiber


def cusp_sextic_coefficients(p: int, q: int) -> tuple[int, int, int]:
    """Return ``(A,B,A^2-7*B^2)`` for the ``ell=4`` sextic current.

    If ``t=p/q`` and ``U=t+1/t=P/Q`` with

        P=p^2+q^2, Q=p*q,

    the first essential discriminant factor is the quadratic norm

        A^2-7*B^2,
        A=P^3+3P^2Q-21PQ^2-47Q^3,
        B=6(P+2Q)Q^2.

    This is the dihedral sextic current in its useful representation over
    ``Q(sqrt(7))``.
    """

    if not p and not q:
        raise ValueError("the homogeneous parameter cannot vanish")
    P = p * p + q * q
    Qh = p * q
    a = P**3 + 3 * P * P * Qh - 21 * P * Qh * Qh - 47 * Qh**3
    b = 6 * (P + 2 * Qh) * Qh * Qh
    return a, b, a * a - 7 * b * b


def triple_current_cancellation_mod_11() -> bool:
    """Projective local obstruction for the third torus-current square.

    Making the discriminant current the third torus multiple reduces the
    remaining rational gates to

        H^2=-3*(p^4-12*p^2*q^2+4*q^4),
        W^2= p^4+2*p^3*q+4*p*q^3+4*q^4.

    There is no nonzero projective pair ``(p:q)`` modulo 11 satisfying both.
    Hence this apparent universal square-discriminant branch has no rational
    fiber at all.
    """

    modulus = 11
    squares = {value * value % modulus for value in range(modulus)}
    for p in range(modulus):
        for q in range(modulus):
            if p == q == 0:
                continue
            first = -3 * (p**4 - 12 * p * p * q * q + 4 * q**4)
            second = p**4 + 2 * p**3 * q + 4 * p * q**3 + 4 * q**4
            if first % modulus in squares and second % modulus in squares:
                return False
    return True


def fourth_current_mod_5_boundary_classes() -> tuple[
    tuple[int, int, int, int, int, int], ...
]:
    """All local classes for the fourth torus-current cancellation.

    Put ``L=S^3-2*S``.  The fourth-current discriminant cancellation asks

        S^2+27*K^2=4,
        d^2=L^2-1,
        2*c=L+d,
        h^2=4*c*(L+S).

    Complete enumeration modulo five leaves only the four signed classes
    returned below.  In all of them ``S=+/-1``, ``L=-S`` and ``d=h=0``.
    The accompanying 5-adic square-class argument is exact: if
    ``delta=S-epsilon`` were nonzero, then

        d^2/h^2 = 1/2 (mod 5),

    after removing the common even valuation of ``delta``.  Since ``1/2``
    is a nonsquare modulo five, every Q_5 lift has ``delta=0`` identically.
    Pulling ``S=+/-1`` back to the split-support parameter gives four monic
    irreducible cubics, so no rational support fiber survives.
    """

    modulus = 5
    squares = {value * value % modulus for value in range(modulus)}
    classes = []
    for split_root in range(modulus):
        for current in range(modulus):
            if (split_root**2 + 27 * current**2 - 4) % modulus:
                continue
            multiplier = (split_root**3 - 2 * split_root) % modulus
            for c in range(1, modulus):
                if (4 * c * c - 4 * multiplier * c + 1) % modulus:
                    continue
                for d in range(modulus):
                    if d * d % modulus != (multiplier**2 - 1) % modulus:
                        continue
                    height_square = 4 * c * (multiplier + split_root) % modulus
                    if height_square in squares:
                        classes.append(
                            (
                                split_root,
                                current,
                                multiplier,
                                c,
                                d,
                                height_square,
                            )
                        )
    result = tuple(classes)
    expected = (
        (1, 2, 4, 2, 0, 0),
        (1, 3, 4, 2, 0, 0),
        (4, 2, 1, 3, 0, 0),
        (4, 3, 1, 3, 0, 0),
    )
    if result != expected:
        raise ArithmeticError("unexpected fourth-current local classes")
    return result


def fourth_current_boundary_support_polynomials() -> tuple[
    tuple[int, int, int, int], ...
]:
    """The irreducible cubic factors above ``S=+/-1``.

    Coefficients are in descending order.  Every monic cubic has possible
    rational roots only ``+/-1``; direct evaluation excludes both.
    """

    polynomials = (
        (1, 0, -3, -1),
        (1, 3, 0, -1),
        (1, -3, -6, -1),
        (1, 6, 3, -1),
    )
    for coefficients in polynomials:
        for root in (-1, 1):
            value = sum(
                coefficient * root ** (3 - index)
                for index, coefficient in enumerate(coefficients)
            )
            if value == 0:
                raise ArithmeticError("fourth-current boundary cubic split")
    return polynomials


def fifth_current_projection_jacobian() -> tuple[int, int, int, int, int]:
    """Rank-zero projection closing the fifth torus-current multiplier.

    For ``L=S^4-3*S^2+1``, remove the visible square ``S^2`` from
    ``d^2=L^2-1`` and put ``X=S^2``.  The remaining gate is

        Y^2=(X-1)*(X-2)*(X-3).

    Translating ``X`` by two gives ``y^2=x^3-x``.  Its rational group is
    ``(Z/2)^2``: the only finite ``X`` values are 1, 2 and 3.  Of these only
    1 is a rational square, forcing ``S=+/-1`` and hence one of the four
    irreducible boundary support cubics returned above.
    """

    return 0, 0, 0, -1, 0


def sixth_current_mod_11_obstruction() -> bool:
    """Complete local obstruction for the sixth torus multiplier.

    The current multiplier is ``L=S^5-4*S^3+3*S``.  Enumerating the support
    conic, the quadratic root gate and the final height gate over ``F_11``
    gives no class at all.  This is a projective support calculation:
    ``-27`` is a nonsquare modulo eleven, so the homogeneous support conic
    has no points at infinity.  Its affine chart therefore contains every
    possible rational reduction.
    """

    modulus = 11
    squares = {value * value % modulus for value in range(modulus)}
    for split_root in range(modulus):
        for current in range(modulus):
            if (split_root**2 + 27 * current**2 - 4) % modulus:
                continue
            multiplier = (
                split_root**5 - 4 * split_root**3 + 3 * split_root
            ) % modulus
            for d in range(modulus):
                if d * d % modulus != (multiplier**2 - 1) % modulus:
                    continue
                c = (multiplier + d) * pow(2, -1, modulus) % modulus
                if 4 * c * (multiplier + split_root) % modulus in squares:
                    return False
    return True


def square_discriminant_recurrence_jacobian() -> tuple[int, int, int, int, int]:
    """The rank-one Jacobian for the second torus-current cancellation.

    Setting the curve discriminant current equal to the torus-double current
    leaves the double lift

        eta^2=2*(x^2+1),
        H^2=-3*(x^4-14*x^2+1).

    Eliminating ``x^2`` gives

        Z^2=-3*(eta^4-32*eta^2+64).

    Its useful 2-isogenous recurrence is the model returned here.  A point
    ``(X,Y)`` lifts only if both ``-X/3`` and ``(-X/3)/2-1`` are rational
    squares.  PARI proves rank one and torsion Z/4; the exact recurrence
    audit is in ``asymmetric_three_split_pari.gp``.
    """

    return 0, 96, 0, 576, 0


def square_discriminant_projection_jacobian() -> tuple[int, int, int, int, int]:
    """Rank-zero quotient that globally closes the double lift.

    Before eliminating either square, every rational lift projects to

        H^2=-3*(x^4-14*x^2+1).

    This genus-one quartic has the four visible points
    ``(x,H)=(+/-1,+/-6)``.  Its Jacobian minimalizes to the model returned
    here, with exact rank zero and torsion ``(Z/2)^2``.  The quartic has no
    rational points at infinity because its leading coefficient is ``-3``.
    Choosing any visible point as origin therefore identifies its rational
    points bijectively with the four Jacobian torsion points.  The visible
    four are complete, and all have ``x^2=1``: they are precisely the two
    rank-loss cusp classes.  This closes the formerly bounded recurrence at
    every height.
    """

    return 0, 0, 0, -39, 70


def main() -> None:
    for ell in (Q(0), Q(1), Q(4)):
        fiber = build_asymmetric_three_split_fiber(Q(1, 2), ell)
        print(
            "ell=", ell,
            "model=", fiber.short_model,
            "points=", len(fiber.points),
        )
    print("cusp norm at t=1/2:", cusp_sextic_coefficients(1, 2))
    print("third-current obstruction mod 11:", triple_current_cancellation_mod_11())


if __name__ == "__main__":
    main()
