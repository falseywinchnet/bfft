"""Local classification of the factorized angle-wall squareclass.

On the rational conductor-factorization locus, the remaining wall current is

    F(p)=p^4-8*p^3+2*p^2+8*p+1.

Writing ``F(p)=D*y^2`` with squarefree ``D`` isolates the only new
squarefree conductor contribution from this wall.  Rational ``p=P/Q`` gives
the homogeneous necessary condition

    F(P,Q)=D*Y^2,

where ``P,Q`` are primitive at every local prime.  This module checks that
condition projectively modulo prime powers.  It is a finite local-solubility
classification, not a height search for rational points.

Every nontrivial squarefree ``|D|<23`` is obstructed.  The sole survivor
``D=1`` has rank-zero Jacobian and only the boundary/torsion classes.  The
seeded class ``D=-23`` is therefore the smallest possible nondegenerate wall
squareclass; its Jacobian has exact rank one.
"""

from __future__ import annotations


LOCAL_MODULI = (
    (2, 2**8),
    (3, 3**5),
    (5, 5**3),
    (7, 7**2),
    (11, 11**2),
    (13, 13**2),
    (17, 17**2),
    (19, 19**2),
    (23, 23**2),
)


def binary_wall_current(p: int, q: int, modulus: int) -> int:
    """Return the homogeneous quartic ``F(P,Q)`` modulo ``modulus``."""

    return (
        p**4
        - 8 * p**3 * q
        + 2 * p * p * q * q
        + 8 * p * q**3
        + q**4
    ) % modulus


def projectively_locally_soluble(
    squareclass: int, prime: int, modulus: int
) -> bool:
    """Necessary solubility of ``F(P,Q)=D*Y^2`` at one prime power."""

    square_values = {
        squareclass * value * value % modulus for value in range(modulus)
    }
    for p in range(modulus):
        for q in range(modulus):
            if p % prime == 0 and q % prime == 0:
                continue
            if binary_wall_current(p, q, modulus) in square_values:
                return True
    return False


def is_squarefree(value: int) -> bool:
    value = abs(value)
    divisor = 2
    while divisor * divisor <= value:
        if value % (divisor * divisor) == 0:
            return False
        divisor += 1
    return True


def local_obstructions(squareclass: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (prime, modulus)
        for prime, modulus in LOCAL_MODULI
        if not projectively_locally_soluble(squareclass, prime, modulus)
    )


def sub_23_survivors() -> tuple[int, ...]:
    return tuple(
        squareclass
        for squareclass in range(-22, 23)
        if squareclass and is_squarefree(squareclass)
        and not local_obstructions(squareclass)
    )


def main() -> None:
    for squareclass in range(-22, 23):
        if not squareclass or not is_squarefree(squareclass):
            continue
        print(squareclass, local_obstructions(squareclass))
    print("survivors", sub_23_survivors())
    print("D=-23 obstructions", local_obstructions(-23))


if __name__ == "__main__":
    main()
