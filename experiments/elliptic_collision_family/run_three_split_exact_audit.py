"""Exact closed audit of the three-horizontal-cubic rank-six layer.

For ``m=a^2+a*b+b^2``, the zero-sum triple ``(a,b,-a-b)`` has product
current ``K=-a*b*(a+b)``.  Three such triples lie on

    y^2 = x^3 - m*x + a6

precisely when three values ``a6+K`` are integer squares.  Difference-of-
squares factorization classifies this finite layer without walking over
Weierstrass coefficients.  The optional PARI audit computes exact global
conductors and algebraic rank bounds for every classified curve.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from math import isqrt
from shutil import which
import subprocess


RANK_SIX_TARGET = 5_187_563_742


@dataclass(frozen=True)
class ThreeSplitCurve:
    m: int
    a6: int
    currents: tuple[int, ...]
    model_discriminant: int


@dataclass(frozen=True)
class PariAudit:
    curve: ThreeSplitCurve
    conductor: int
    root_number: int
    rank_lower: int
    rank_upper: int


def current_catalog(maximum_norm: int) -> dict[int, tuple[int, ...]]:
    catalog: dict[int, set[int]] = {}
    bound = isqrt(4 * maximum_norm // 3) + 2
    for a in range(-bound, bound + 1):
        for b in range(-bound, bound + 1):
            m = a * a + a * b + b * b
            if not 0 < m <= maximum_norm:
                continue
            current = -a * b * (a + b)
            if current == 0:
                # A zero product is the degenerate support boundary, not a
                # third nondegenerate split cubic.
                continue
            catalog.setdefault(m, set()).add(current)
    return {m: tuple(sorted(values)) for m, values in catalog.items()}


def _positive_divisors(value: int) -> tuple[int, ...]:
    divisors = set()
    for divisor in range(1, isqrt(value) + 1):
        if value % divisor == 0:
            divisors.add(divisor)
            divisors.add(value // divisor)
    return tuple(sorted(divisors))


def classify_three_split_curves(
    maximum_norm: int = 2000,
    discriminant_ceiling: int = RANK_SIX_TARGET,
) -> tuple[ThreeSplitCurve, ...]:
    curves: dict[tuple[int, int], ThreeSplitCurve] = {}
    for m, currents in current_catalog(maximum_norm).items():
        if len(currents) < 3:
            continue
        candidate_a6 = set()
        for left_index, left in enumerate(currents):
            for right in currents[left_index + 1 :]:
                difference = right - left
                for small in _positive_divisors(difference):
                    large = difference // small
                    if small > large or (small + large) % 2:
                        continue
                    left_height = (large - small) // 2
                    candidate_a6.add(left_height * left_height - left)
        for a6 in candidate_a6:
            supported = tuple(
                current
                for current in currents
                if a6 + current >= 0
                and isqrt(a6 + current) ** 2 == a6 + current
            )
            if len(supported) < 3:
                continue
            discriminant = 64 * m**3 - 432 * a6 * a6
            if discriminant == 0 or abs(discriminant) >= discriminant_ceiling:
                continue
            curves[(m, a6)] = ThreeSplitCurve(m, a6, supported, discriminant)
    return tuple(
        sorted(curves.values(), key=lambda curve: abs(curve.model_discriminant))
    )


def pari_audit(curves: tuple[ThreeSplitCurve, ...]) -> tuple[PariAudit, ...]:
    gp = which("gp")
    if gp is None:
        raise RuntimeError("PARI/GP is required for --pari")
    models = ",".join(
        f"[0,0,0,{-curve.m},{curve.a6}]" for curve in curves
    )
    program = (
        f"models=[{models}];"
        "for(i=1,#models,E=ellinit(models[i]);r=ellrank(E);"
        "print(i,\",\",ellglobalred(E)[1],\",\",ellrootno(E),"
        "\",\",r[1],\",\",r[2]));quit;\n"
    )
    completed = subprocess.run(
        [gp, "-q", "-f"],
        input=program,
        text=True,
        capture_output=True,
        check=True,
    )
    audits = []
    for line in completed.stdout.splitlines():
        index, conductor, root, lower, upper = map(int, line.split(","))
        audits.append(
            PariAudit(curves[index - 1], conductor, root, lower, upper)
        )
    if len(audits) != len(curves):
        raise ArithmeticError(completed.stderr or "incomplete PARI audit")
    return tuple(sorted(audits, key=lambda item: item.conductor))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--maximum-norm", type=int, default=2000)
    parser.add_argument("--pari", action="store_true")
    args = parser.parse_args()
    curves = classify_three_split_curves(args.maximum_norm)
    print("classified curves:", len(curves))
    if not args.pari:
        for curve in curves:
            print(curve)
        return
    audits = pari_audit(curves)
    for audit in audits:
        curve = audit.curve
        print(
            f"m={curve.m} a6={curve.a6} conductor={audit.conductor} "
            f"root={audit.root_number} rank=[{audit.rank_lower},{audit.rank_upper}] "
            f"currents={curve.currents} delta={curve.model_discriminant}"
        )
    winners = [
        audit
        for audit in audits
        if audit.conductor < RANK_SIX_TARGET and audit.rank_lower >= 6
    ]
    print("rank-six winners:", len(winners))


if __name__ == "__main__":
    main()
