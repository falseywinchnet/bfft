"""Exact norm-class sieve for the equianharmonic cusp sextic.

For the ``ell=4`` asymmetric three-split family, the first essential
conductor current at ``t=p/q`` is

    A^2-7*B^2,

where ``(A,B)`` is the degree-six split-support map implemented below.  A
fixed norm has finitely many representatives modulo the fundamental unit
``8+3*sqrt(7)``.  Comparing each complete unit period modulo small primes
with the exact image of the coupled support map obstructs the entire
infinite orbit.

Below 55151, only norms 1 and 729 survive the local sieve.  Exact Thue
equations in ``asymmetric_three_split_pari.gp`` show both are cusp classes.
At -55151 the first nonboundary orbit appears, exactly at ``t=1/2`` (and its
deck-equivalent presentations).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import subprocess

from run_asymmetric_three_split_attack import cusp_sextic_coefficients


PRIMES = (
    5,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
)


@dataclass(frozen=True)
class NormOrbit:
    norm: int
    a: int
    b: int


@dataclass(frozen=True)
class LocalObstruction:
    orbit: NormOrbit
    prime: int
    period: int


def admissible_norms(bound: int) -> tuple[int, ...]:
    """Apply the exact primitive support residues at 2, 3, 5 and 7."""

    return tuple(
        value
        for value in range(-bound, bound + 1)
        if value % 8 == 1
        and value % 9 in (0, 1)
        and value % 5 in (1, 4)
        and value % 7 in (1, 2)
    )


def split_support_image(modulus: int) -> frozenset[tuple[int, int]]:
    """Complete primitive image ``(p,q)->(A,B)`` modulo a prime."""

    image: set[tuple[int, int]] = set()
    for p in range(modulus):
        for q in range(modulus):
            if p == q == 0:
                continue
            a, b, _norm = cusp_sextic_coefficients(p, q)
            image.add((a % modulus, b % modulus))
    return frozenset(image)


def enumerate_norm_orbits(
    bound: int, gp: str = "/opt/homebrew/bin/gp"
) -> tuple[NormOrbit, ...]:
    classes = admissible_norms(bound)
    if not classes:
        return ()
    program = (
        "K=bnfinit(x^2-7);C=["
        + ",".join(str(value) for value in classes)
        + "];for(i=1,#C,c=C[i];v=bnfisintnorm(K,c);"
        "for(j=1,#v,print(c,\",\",polcoeff(v[j],0),\",\","
        "polcoeff(v[j],1))));quit;\n"
    )
    completed = subprocess.run(
        [gp, "-q"],
        input=program,
        text=True,
        capture_output=True,
        check=True,
    )
    if completed.stderr.strip():
        raise RuntimeError(completed.stderr.strip())
    orbits: set[NormOrbit] = set()
    for line in completed.stdout.splitlines():
        norm, a, b = (int(field) for field in line.split(","))
        orbits.add(NormOrbit(norm, a, b))
        orbits.add(NormOrbit(norm, -a, -b))
    return tuple(sorted(orbits, key=lambda item: (item.norm, item.a, item.b)))


def unit_orbit_obstruction(
    orbit: NormOrbit,
    modulus: int,
    image: frozenset[tuple[int, int]] | None = None,
) -> LocalObstruction | None:
    """Test a complete period under multiplication by ``8+3*sqrt(7)``."""

    support = image or split_support_image(modulus)
    a, b = orbit.a % modulus, orbit.b % modulus
    start = a, b
    period = 0
    while True:
        if (a, b) in support:
            return None
        a, b = (8 * a + 21 * b) % modulus, (3 * a + 8 * b) % modulus
        period += 1
        if (a, b) == start:
            return LocalObstruction(orbit, modulus, period)
        if period > modulus * modulus + 1:
            raise ArithmeticError("unit orbit failed to close")


def certify_orbits(
    orbits: tuple[NormOrbit, ...],
) -> tuple[tuple[LocalObstruction, ...], tuple[NormOrbit, ...]]:
    images = {prime: split_support_image(prime) for prime in PRIMES}
    certificates: list[LocalObstruction] = []
    survivors: list[NormOrbit] = []
    for orbit in orbits:
        for prime in PRIMES:
            certificate = unit_orbit_obstruction(
                orbit, prime, images[prime]
            )
            if certificate is not None:
                certificates.append(certificate)
                break
        else:
            survivors.append(orbit)
    return tuple(certificates), tuple(survivors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound", type=int, default=55_150)
    parser.add_argument("--gp", default="/opt/homebrew/bin/gp")
    args = parser.parse_args()
    orbits = enumerate_norm_orbits(args.bound, args.gp)
    certificates, survivors = certify_orbits(orbits)
    print("admissible norm classes:", len(admissible_norms(args.bound)))
    print("signed unit orbits:", len(orbits))
    print("locally obstructed:", len(certificates))
    print("survivors:")
    for orbit in survivors:
        print(f"  {orbit.norm}: {orbit.a}+({orbit.b})*sqrt(7)")


if __name__ == "__main__":
    main()
