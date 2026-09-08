"""Exact local certificate for the second four-split conductor current.

The second quartic conductor core can be written

    c = D^6 - 3*(18*r^2)^2

where, for a primitive split support ``t=p/q``,

    D = p^2+p*q+q^2,       r = p*q*(p+q)/2.

Thus ``D^3 + 18*r^2*sqrt(3)`` is an element of norm ``c`` in
``Z[sqrt(3)]``.  PARI's ``bnfisintnorm`` gives a complete finite set of
representatives for every fixed norm modulo the unit group.  Multiplication
by the fundamental unit ``2+sqrt(3)`` then makes each representative a
single deterministic recurrence.

For an odd prime ell, this recurrence has a finite exact period.  We compare
the whole period with the image modulo ell of the *actual split-support map*

    (p,q) -> ((p^2+p*q+q^2)^3, 18*(p*q*(p+q)/2)^2).

If the two finite sets do not meet, that entire infinite unit orbit is
impossible over Q.  This is a norm-class classification, not a walk through
Weierstrass coefficients or rational parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess


DEFAULT_PRIMES = (
    5,
    7,
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
    """Return the canonical-deck norm classes ``c=1 (mod 36)``."""

    if bound < 1:
        return ()
    return tuple(c for c in range(-bound, bound + 1) if c % 36 == 1)


def split_support_image(modulus: int) -> frozenset[tuple[int, int]]:
    """The complete homogeneous split-support image modulo an odd prime."""

    if modulus < 3 or modulus % 2 == 0:
        raise ValueError("modulus must be odd and at least three")
    inverse_two = pow(2, -1, modulus)
    image: set[tuple[int, int]] = set()
    for p in range(modulus):
        for q in range(modulus):
            d = (p * p + p * q + q * q) % modulus
            r = p * q * (p + q) * inverse_two % modulus
            image.add((d**3 % modulus, 18 * r * r % modulus))
    return frozenset(image)


def unit_orbit_obstruction(
    orbit: NormOrbit,
    modulus: int,
    support_image: frozenset[tuple[int, int]] | None = None,
) -> LocalObstruction | None:
    """Return a certificate when a complete local unit orbit misses support."""

    support = support_image or split_support_image(modulus)
    a, b = orbit.a % modulus, orbit.b % modulus
    start = a, b
    period = 0
    meets_support = False
    while True:
        meets_support = meets_support or (a, b) in support
        a, b = (2 * a + 3 * b) % modulus, (a + 2 * b) % modulus
        period += 1
        if (a, b) == start:
            break
        if period > modulus * modulus + 1:
            raise ArithmeticError("unit orbit failed to close")
    if meets_support:
        return None
    return LocalObstruction(orbit=orbit, prime=modulus, period=period)


def parse_bnfisintnorm_output(output: str) -> tuple[NormOrbit, ...]:
    """Parse ``norm,a,b`` rows and add the independent signed associates."""

    orbits: set[NormOrbit] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) != 3:
            raise ValueError(f"unexpected PARI row: {raw_line!r}")
        norm, a, b = (int(field) for field in fields)
        orbits.add(NormOrbit(norm, a, b))
        orbits.add(NormOrbit(norm, -a, -b))
    return tuple(sorted(orbits, key=lambda item: (item.norm, item.a, item.b)))


def enumerate_norm_orbits(
    bound: int,
    gp: str = "/opt/homebrew/bin/gp",
) -> tuple[NormOrbit, ...]:
    """Use PARI to enumerate all fixed-norm unit orbits exactly."""

    classes = admissible_norms(bound)
    if not classes:
        return ()
    start, stop = classes[0], classes[-1]
    program = (
        "K=bnfinit(x^2-3);"
        f"forstep(c={start},{stop},36,"
        "v=bnfisintnorm(K,c);"
        "for(j=1,#v,print(c,\",\",polcoeff(v[j],0),\",\","
        "polcoeff(v[j],1))));quit\n"
    )
    result = subprocess.run(
        [gp, "-q"],
        input=program,
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stderr.strip():
        raise RuntimeError(result.stderr.strip())
    return parse_bnfisintnorm_output(result.stdout)


def certify_orbits(
    orbits: tuple[NormOrbit, ...],
    primes: tuple[int, ...] = DEFAULT_PRIMES,
) -> tuple[tuple[LocalObstruction, ...], tuple[NormOrbit, ...]]:
    """Find the first local split-support obstruction for every orbit."""

    images = {prime: split_support_image(prime) for prime in primes}
    certificates: list[LocalObstruction] = []
    survivors: list[NormOrbit] = []
    for orbit in orbits:
        certificate = None
        for prime in primes:
            certificate = unit_orbit_obstruction(orbit, prime, images[prime])
            if certificate is not None:
                certificates.append(certificate)
                break
        if certificate is None:
            survivors.append(orbit)
    return tuple(certificates), tuple(survivors)


def write_certificate(
    path: Path,
    bound: int,
    orbits: tuple[NormOrbit, ...],
    certificates: tuple[LocalObstruction, ...],
    survivors: tuple[NormOrbit, ...],
) -> None:
    """Write a small auditable text certificate."""

    lines = [
        f"bound={bound}",
        f"admissible_classes={len(admissible_norms(bound))}",
        f"signed_unit_orbits={len(orbits)}",
        f"obstructed_orbits={len(certificates)}",
        f"surviving_orbits={len(survivors)}",
    ]
    lines.extend(
        f"survivor={orbit.norm},{orbit.a},{orbit.b}" for orbit in survivors
    )
    lines.append("certificates:")
    lines.extend(
        f"{item.orbit.norm},{item.orbit.a},{item.orbit.b},"
        f"{item.prime},{item.period}"
        for item in certificates
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound", type=int, default=38_916)
    parser.add_argument("--gp", default="/opt/homebrew/bin/gp")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    orbits = enumerate_norm_orbits(args.bound, args.gp)
    certificates, survivors = certify_orbits(orbits)
    print(f"admissible classes: {len(admissible_norms(args.bound))}")
    print(f"signed unit orbits: {len(orbits)}")
    print(f"locally obstructed: {len(certificates)}")
    print("survivors:")
    for orbit in survivors:
        print(f"  c={orbit.norm}: {orbit.a}+({orbit.b})*sqrt(3)")
    if args.out is not None:
        write_certificate(
            args.out, args.bound, orbits, certificates, survivors
        )


if __name__ == "__main__":
    main()
