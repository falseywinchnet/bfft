"""Exact unit-flow attack on the residual simultaneous-Pell problem.

The module deliberately does not search the negative-Pell orbit.  It exposes
three exact structures instead:

* each of the two Pell equations has one positive orbit and a strict
  predecessor;
* equality of their shared ``u`` coordinate has strong index congruences;
* the mismatch between the two predecessors is an integral descent defect.

These statements turn the first possible counterexample into a theorem-sized
object (at least 25 decimal digits), rather than a larger finite search.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd


DEFECT_SQUAREFREE_DIVISORS = (1, 2, 5, 7, 10, 14, 35, 70)
SQRT2_NORM_SEEDS = {
    5: ((5, 0),),
    6: ((6, 0),),
    14: ((14, 0), (22, 12), (22, -12)),
    21: ((21, 0), (33, 18), (33, -18)),
    70: ((70, 0), (110, 60), (110, -60)),
    105: ((105, 0), (165, 90), (165, -90)),
}
# level -> (valuation parity of the sqrt(210) unit,
#           valuation parity of the sqrt(105) unit,
#           square root in the basis 1,sqrt(2),sqrt(105),sqrt(210))
CORE_SQUARECLASS_ROOTS = {
    5: (0, 1, (-10, 0, 1, 0)),
    6: (1, 1, (-42, -30, 4, 3)),
    14: (1, 0, (-14, 0, 0, 1)),
    21: (0, 1, (21, 0, -2, 0)),
    70: (1, 1, (140, 105, -14, -10)),
    105: (0, 0, (0, 0, 1, 0)),
}
CURRENT_DETERMINANT_MULTIPLIERS = {
    5: -10,
    6: -6,
    14: -14,
    21: 42,
    70: 70,
    105: -210,
}


def biquadratic_multiply(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Multiply in the basis ``1,sqrt(2),sqrt(105),sqrt(210)``."""

    a, b, c, d = left
    e, f, g, h = right
    return (
        a * e + 2 * b * f + 105 * c * g + 210 * d * h,
        a * f + b * e + 105 * c * h + 105 * d * g,
        a * g + c * e + 2 * b * h + 2 * d * f,
        a * h + d * e + b * g + c * f,
    )


def biquadratic_square(value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return biquadratic_multiply(value, value)


def squareclass_radicand(level: int) -> tuple[int, int, int, int]:
    """Return ``level*delta210^p*delta105^q`` for its viable parity pair."""

    p, q, _ = CORE_SQUARECLASS_ROOTS[level]
    value = (level, 0, 0, 0)
    if p:
        value = biquadratic_multiply(value, (29, 0, 0, -2))
    if q:
        value = biquadratic_multiply(value, (41, 0, -4, 0))
    return value


def quadratic_unit_power(
    discriminant: int,
    rational: int,
    radical: int,
    exponent: int,
) -> tuple[int, int]:
    """Return ``(a,b)`` for ``(rational-radical*sqrt(D))^exponent``.

    The returned pair represents ``a-b*sqrt(D)``.  Binary powering keeps the
    calculation exact even at the very large indices relevant here.
    """

    if exponent < 0:
        raise ValueError("exponent must be nonnegative")
    result = (1, 0)
    base = (rational, radical)
    power = exponent
    while power:
        if power & 1:
            a, b = result
            c, d = base
            result = (a * c + discriminant * b * d, a * d + b * c)
        c, d = base
        base = (c * c + discriminant * d * d, 2 * c * d)
        power //= 2
    return result


def unit_lattice_coefficients(
    level: int,
    n: int,
    i: int,
    ell: int,
) -> tuple[int, int, int, int]:
    """Coefficient vector of the global unit equation's right side.

    Here ``i=floor((j+1)/2)`` and ``ell=floor((k+1)/2)``.  The result is in
    the basis ``1,sqrt(2),sqrt(105),sqrt(210)`` and includes the viable
    squareclass root for ``level``.
    """

    if level not in CORE_SQUARECLASS_ROOTS:
        raise ValueError("not a core quartic level")
    if min(n, i, ell) < 0:
        raise ValueError("unit exponents must be nonnegative")
    a, b = quadratic_unit_power(2, 7, 5, n)
    c, d = quadratic_unit_power(210, 29, 2, i)
    e, f = quadratic_unit_power(105, 41, 4, ell)
    value = CORE_SQUARECLASS_ROOTS[level][2]
    value = biquadratic_multiply(value, (a, -b, 0, 0))
    value = biquadratic_multiply(value, (c, 0, 0, -d))
    return biquadratic_multiply(value, (e, 0, -f, 0))


def vanishing_currents(
    level: int,
    n: int,
    i: int,
    ell: int,
) -> tuple[int, int]:
    """Return the two lattice currents that a quartic solution must kill."""

    _, q2, q105, q210 = unit_lattice_coefficients(level, n, i, ell)
    return 2 * q105 - 7 * q210, 2 * q2 + 21 * q210


def vanishing_current_form(
    level: int,
    n: int,
    i: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the two currents as linear forms in ``(e,f)``.

    If ``(41-4sqrt(105))^ell=e-fsqrt(105)``, the returned rows
    ``((r0,r1),(s0,s1))`` represent ``r0*e+r1*f`` and ``s0*e+s1*f``.
    """

    if level not in CORE_SQUARECLASS_ROOTS:
        raise ValueError("not a core quartic level")
    a, b = quadratic_unit_power(2, 7, 5, n)
    c, d = quadratic_unit_power(210, 29, 2, i)
    prefix = CORE_SQUARECLASS_ROOTS[level][2]
    prefix = biquadratic_multiply(prefix, (a, -b, 0, 0))
    p0, p2, p105, p210 = biquadratic_multiply(prefix, (c, 0, 0, -d))
    return (
        (2 * p105 - 7 * p210, -2 * p0 + 7 * p2),
        (2 * p2 + 21 * p210, -210 * p210 - 21 * p2),
    )


def vanishing_current_determinant(level: int, n: int, i: int) -> int:
    """Eliminate the last quadratic unit from the two current equations."""

    (r0, r1), (s0, s1) = vanishing_current_form(level, n, i)
    return r0 * s1 - r1 * s0


def reduced_vanishing_current(
    level: int,
    a: int,
    b: int,
    c: int,
    d: int,
) -> int:
    """Eliminate the sqrt(105) ratio and reduce by the two Pell norms.

    ``a-b*sqrt(2)=(7-5sqrt(2))^n`` and
    ``c-d*sqrt(210)=(29-2sqrt(210))^i``.  Simultaneous vanishing forces this
    integer to be zero.  Levels 5 and 21 share the first expression; levels 6
    and 14 share the second.
    """

    if level not in CORE_SQUARECLASS_ROOTS:
        raise ValueError("not a core quartic level")
    sign = a * a - 2 * b * b
    if sign not in (-1, 1):
        raise ValueError("the sqrt(2) coefficients do not have unit norm")
    if c * c - 210 * d * d != 1:
        raise ValueError("the sqrt(210) coefficients do not have unit norm")
    if level in (5, 21, 105):
        return -2 * a * b - 7 * b * b - sign * (21 * c * d - 735 * d * d)
    return -4 * a * b - 14 * b * b - sign * (
        1_722 * c * d + 24_990 * d * d + 63
    )


def two_adic_valuation(value: int) -> int:
    """Return ``v_2(value)`` for a nonzero integer."""

    if value == 0:
        raise ValueError("the 2-adic valuation of zero is infinite")
    value = abs(value)
    return (value & -value).bit_length() - 1


def level5_two_adic_balance(n: int, i: int, ell: int) -> tuple[int, int]:
    """Return the two sides of the level-5 current's necessary v2 identity.

    For positive ``n,i,ell`` with ``n`` even, ``i`` a multiple of 5, and
    ``ell`` a multiple of 7, the second current can vanish only if

    ``v2(n) = v2(i) + v2(ell) + 3``.
    """

    if n <= 0 or i <= 0 or ell <= 0 or n % 2 or i % 5 or ell % 7:
        raise ValueError("not an admissible positive level-5 exponent triple")
    return two_adic_valuation(n), two_adic_valuation(i) + two_adic_valuation(ell) + 3


def level5_first_current_two_adic_balance(n: int, i: int) -> tuple[int, int]:
    """Return the valuations forced equal by the first level-5 current.

    Once the second current has forced ``v2(n)>=3``, the factors
    ``2a+7b`` and ``35d-c`` have valuations 1 and 0 respectively.  Hence
    ``b(2a+7b)=21d(35d-c)`` would require the returned pair to be equal.
    """

    if n <= 0 or i <= 0 or n % 2 or i % 5:
        raise ValueError("not an admissible positive level-5 exponent pair")
    if two_adic_valuation(n) < 3:
        raise ValueError("the second current has not passed its necessary gate")
    return two_adic_valuation(n) + 1, two_adic_valuation(i) + 1


def level5_currents_have_two_adic_contradiction(n: int, i: int, ell: int) -> bool:
    """Certify that the two level-5 currents cannot vanish together."""

    second_left, second_right = level5_two_adic_balance(n, i, ell)
    if second_left != second_right:
        return True
    first_left, first_right = level5_first_current_two_adic_balance(n, i)
    return first_left != first_right


def first_forward(u: int, v: int) -> tuple[int, int]:
    """Multiply ``sqrt(15)u + sqrt(14)v`` by ``29 + 2sqrt(210)``."""

    return 29 * u + 28 * v, 30 * u + 29 * v


def first_predecessor(u: int, v: int) -> tuple[int, int]:
    return 29 * u - 28 * v, 29 * v - 30 * u


def second_forward(u: int, w: int) -> tuple[int, int]:
    """Multiply ``sqrt(21)u + sqrt(20)w`` by ``41 + 2sqrt(420)``."""

    return 41 * u + 40 * w, 42 * u + 41 * w


def second_predecessor(u: int, w: int) -> tuple[int, int]:
    return 41 * u - 40 * w, 41 * w - 42 * u


def first_orbit(index: int) -> tuple[int, int]:
    if index < 0:
        raise ValueError("index must be nonnegative")
    point = (1, 1)
    for _ in range(index):
        point = first_forward(*point)
    return point


def second_orbit(index: int) -> tuple[int, int]:
    if index < 0:
        raise ValueError("index must be nonnegative")
    point = (1, 1)
    for _ in range(index):
        point = second_forward(*point)
    return point


def negative_pell_orbit(index: int) -> tuple[int, int]:
    """Return the residual orbit satisfying ``49v^2 - 50w^2 = -1``."""

    if index < 0:
        raise ValueError("index must be nonnegative")
    v, w = 1, 1
    for _ in range(index):
        v, w = 99 * v + 100 * w, 98 * v + 99 * w
    return v, w


def pell2_power(exponent: int) -> tuple[int, int]:
    """Return ``(a,b)`` with ``(1 + sqrt(2))^exponent = a + b sqrt(2)``."""

    if exponent < 0:
        raise ValueError("exponent must be nonnegative")
    result = (1, 0)
    base = (1, 1)
    power = exponent
    while power:
        if power & 1:
            a, b = result
            c, d = base
            result = (a * c + 2 * b * d, a * d + b * c)
        c, d = base
        base = (c * c + 2 * d * d, 2 * c * d)
        power //= 2
    return result


def compressed_negative_pell_orbit(index: int) -> tuple[int, int]:
    """Recover the residual orbit from one ordinary sqrt(2) Pell current."""

    a, b = pell2_power(6 * index + 3)
    if a % 7 or b % 5:
        raise ArithmeticError("the 6n+3 divisibility law failed")
    return a // 7, b // 5


def first_index_gate(index: int) -> bool:
    """Necessary condition imposed by the second orbit's ``u == 1 mod 40``."""

    return index % 10 in (0, 9)


def second_index_gate(index: int) -> bool:
    """Necessary condition imposed by the first orbit's ``u == 1 mod 28``."""

    return index % 14 in (0, 13)


def admissible_defect_residues() -> tuple[int, ...]:
    """CRT classes modulo 70 forced on the predecessor defect."""

    return tuple(
        h
        for h in range(70)
        if h % 2 == 0 and h % 5 in (0, 4) and h % 7 in (0, 1)
    )


def defect_cover_quartic(d: int, a: int, b: int) -> int:
    """Quartic obtained from the almost-coprime defect factors."""

    if d not in DEFECT_SQUAREFREE_DIVISORS:
        raise ValueError("d must be a squarefree divisor of 70")
    e = 70 // d
    u_form = e * b * b - d * a * a
    v_form = e * b * b + d * a * a + 20 * a * b
    return 15 * u_form * u_form - 14 * v_form * v_form


def universal_quartic(A: int, b: int) -> int:
    """The common binary quartic underlying every surviving defect cover."""

    return (
        A**4
        - 560 * A**3 * b
        - 9_660 * A * A * b * b
        - 39_200 * A * b**3
        + 4_900 * b**4
    )


def biquadratic_relative_norms(A: int, b: int) -> dict[int, tuple[int, int]]:
    """Return ``(p,q)`` for the three norms ``p + q*sqrt(D)``.

    Each pair has rational norm equal to ``universal_quartic(A,b)``.
    They are the relative norms of ``A - beta*b`` in
    ``Q(sqrt(2),sqrt(105))``, where

    ``beta = 140 + 105sqrt(2) - 14sqrt(105) - 10sqrt(210)``.
    """

    return {
        2: (A * A - 280 * A * b + 70 * b * b, -210 * A * b),
        105: (A * A - 280 * A * b - 2_870 * b * b, 28 * A * b + 280 * b * b),
        210: (A * A - 280 * A * b - 2_030 * b * b, 20 * A * b + 140 * b * b),
    }


def negative_pell_unit_shadow(index: int) -> tuple[int, int]:
    """Return ``(L,M)`` with ``(99-70sqrt(2))^index=L-Msqrt(2)``."""

    if index < 0:
        raise ValueError("index must be nonnegative")
    L, M = 1, 0
    for _ in range(index):
        L, M = 99 * L + 140 * M, 70 * L + 99 * M
    return L, M


def shadow_from_negative_pell(v: int, w: int) -> tuple[int, int]:
    """Project a residual Pell point onto its norm-one sqrt(2) shadow."""

    if 49 * v * v - 50 * w * w != -1:
        raise ValueError("not a residual negative-Pell point")
    return 50 * w - 49 * v, 35 * (v - w)


def defect_cover_locally_soluble(g: int, d: int, prime: int) -> bool:
    """Test one finite defect cover over ``F_prime`` with coprimality."""

    if g not in (1, 3):
        raise ValueError("the defect gcd multiplier must be 1 or 3")
    if d not in DEFECT_SQUAREFREE_DIVISORS:
        raise ValueError("d must be a squarefree divisor of 70")
    e = 70 // d
    target = 9 // (g * g)
    for a in range(prime):
        for b in range(prime):
            if a == b == 0:
                continue
            # gcd(d*a^2, e*b^2)=1 on an integral cover.
            if e % prime == 0 and a == 0:
                continue
            if d % prime == 0 and b == 0:
                continue
            if defect_cover_quartic(d, a, b) % prime == target % prime:
                return True
    return False


def mod5_defect_covers() -> tuple[tuple[int, int], ...]:
    return tuple(
        (g, d)
        for g in (1, 3)
        for d in DEFECT_SQUAREFREE_DIVISORS
        if defect_cover_locally_soluble(g, d, 5)
    )


def defect_cover_mod210_soluble(g: int, d: int) -> bool:
    """Apply the combined 2,3,5,7 cover and nonzero-defect conditions."""

    e = 70 // d
    target = 9 // (g * g)
    for a in range(210):
        for b in range(210):
            if any(a % prime == b % prime == 0 for prime in (2, 3, 5, 7)):
                continue
            if any(e % prime == 0 and a % prime == 0 for prime in (2, 5, 7)):
                continue
            if any(d % prime == 0 and b % prime == 0 for prime in (2, 5, 7)):
                continue
            u_form = e * b * b - d * a * a
            v_form = e * b * b + d * a * a + 20 * a * b
            w_form = e * b * b + d * a * a + 14 * a * b
            if (g * u_form) % 3 or (g * v_form) % 3 or (g * w_form) % 3:
                continue
            if defect_cover_quartic(d, a, b) % 210 != target % 210:
                continue
            if (2 * g * d * a * a) % 70 not in (0, 14, 50, 64):
                continue
            return True
    return False


def core_defect_covers() -> tuple[tuple[int, int], ...]:
    return tuple(
        cover for cover in mod5_defect_covers() if defect_cover_mod210_soluble(*cover)
    )


def core_cover_level(g: int, d: int) -> int:
    """Return ``c`` such that the universal equation is ``G(A,b)=c^2``."""

    if (g, d) not in core_defect_covers():
        raise ValueError("not a core defect cover")
    return 3 * d // g


@dataclass(frozen=True)
class DefectCertificate:
    first_u: int
    first_v: int
    second_u: int
    second_w: int
    h: int
    half_h: int
    half_gap: int
    factor_gcd: int


def descent_defect(u: int, v: int, w: int) -> DefectCertificate:
    """Return and verify the exact mismatch of the two strict predecessors.

    For a simultaneous solution, ``h = 10w - 7v - 3u`` and

    ``first_u - second_u = 4h``
    ``3(u*first_u - 1) = 2h(h + 14v + 6u)``
    ``3(u*second_u - 1) = 2h(h + 14v)``.

    Thus ``h=0`` is exactly the record solution.  The mathematical proof that
    ``h>0`` away from the record uses positivity of both Pell predecessors;
    this function checks the algebraic certificate only.
    """

    if 15 * u * u - 14 * v * v != 1:
        raise ValueError("(u,v) does not satisfy the first Pell equation")
    if 21 * u * u - 20 * w * w != 1:
        raise ValueError("(u,w) does not satisfy the second Pell equation")
    first_u, first_v = first_predecessor(u, v)
    second_u, second_w = second_predecessor(u, w)
    h = 10 * w - 7 * v - 3 * u
    assert first_u - second_u == 4 * h
    assert 3 * (u * first_u - 1) == 2 * h * (h + 14 * v + 6 * u)
    assert 3 * (u * second_u - 1) == 2 * h * (h + 14 * v)
    assert h % 2 == 0 and (v - w) % 2 == 0
    half_h = h // 2
    half_gap = (v - w) // 2
    # The normalized defect curve drops from genus one to this conic gate.
    assert half_h * (half_h + 3 * u) == 70 * half_gap * half_gap
    assert 3 % gcd(h, u) == 0
    factor_gcd = gcd(half_h, half_h + 3 * u)
    assert 9 % factor_gcd == 0
    return DefectCertificate(
        first_u,
        first_v,
        second_u,
        second_w,
        h,
        half_h,
        half_gap,
        factor_gcd,
    )


def first_counterexample_floor() -> int:
    """Exact lower bound for ``u`` after the two index gates."""

    return max(first_orbit(9)[0], second_orbit(13)[0])


def main() -> None:
    print("sqrt(2) compression:")
    for index in range(4):
        print(index, negative_pell_orbit(index), compressed_negative_pell_orbit(index))
    print("sqrt(2) norm seeds:", SQRT2_NORM_SEEDS)
    print("first allowed nonrecord indices: j=9, k=13")
    print("defect residues mod 70:", admissible_defect_residues())
    print("defect covers surviving mod 5:", mod5_defect_covers())
    print("core defect covers:", core_defect_covers())
    print(
        "universal levels:",
        tuple((cover, core_cover_level(*cover)) for cover in core_defect_covers()),
    )
    print("counterexample floor for u:", first_counterexample_floor())
    print("record defect:", descent_defect(1, 1, 1))


if __name__ == "__main__":
    main()
