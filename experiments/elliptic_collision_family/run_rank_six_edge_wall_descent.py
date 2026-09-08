"""Discriminant-wall descent for the second support-edge section current.

The extra abscissa ``x=r+2*s`` has the linear square carrier

    z^2 = (1+t^2)^2 + 12*t*(1-t^2)*(1+2*c).

Thus ``z`` parametrizes the entire rank-six-capacity locus at fixed ``t``.
Only continued-fraction characteristics of its pulled-back discriminant wall
are evaluated here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / "elliptic_rank_support"
sys.path.insert(0, str(SUPPORT))

from elliptic_rank_support import reduction_dependency_masks  # noqa: E402
from run_rank_six_wall_descent import (  # noqa: E402
    GOOD_PRIMES,
    RANK_SIX_CONDUCTOR_TARGET,
    continued_fraction_convergents,
    generalized_integral_scale,
)
from seven_section_surface import (  # noqa: E402
    extra_section_r_minus_s,
    extra_section_r_minus_s_parameter,
    extra_section_r_plus_2s,
    extra_section_r_plus_2s_parameter,
    short_discriminant,
    surface_fiber,
)


Q = Fraction


@dataclass(frozen=True)
class EdgeWallFiber:
    branch: int
    current: str
    q: int
    root: Fraction
    c: Fraction
    scale: int
    a3: int
    a4: int
    discriminant: int
    independent_indices: tuple[int, ...] | None
    short_model: tuple[int, int, int]
    short_points: tuple[tuple[int, int], ...]


def edge_wall_fiber(
    branch: int,
    q_denominator: int,
    root: Fraction,
    certify_independence: bool = False,
    current: str = "edge",
) -> EdgeWallFiber | None:
    t = Q(1, q_denominator)
    if current == "edge":
        c = extra_section_r_plus_2s_parameter(t, root)
        extra_builder = lambda: extra_section_r_plus_2s(c, t, root)
    elif current == "r-minus-s":
        c, section_root = extra_section_r_minus_s_parameter(t, root)
        extra_builder = lambda: extra_section_r_minus_s(c, t, section_root)
    else:
        raise ValueError("unknown extra-section current")
    if c in (0, -1):
        return None
    fiber = surface_fiber(c, t)
    extra = extra_builder()
    center = fiber.center_height
    points = tuple((x, y - center / 2) for x, y in fiber.seven_points())
    points += ((extra[0], extra[1] - center / 2),)
    if len(set(points)) < 8:
        return None
    scale = generalized_integral_scale(fiber.m, center, points)
    a3 = int(center * scale**3)
    a4 = int(-fiber.m * scale**4)
    discriminant = int(short_discriminant(fiber) * scale**12)
    short_model = (0, 16 * a4, 16 * a3 * a3)
    short_points = tuple(
        (
            int(4 * x * scale**2),
            int(4 * (2 * y * scale**3 + a3)),
        )
        for x, y in points
    )
    independent = None
    if certify_independence:
        for indices in combinations(range(8), 6):
            selected = tuple(short_points[index] for index in indices)
            if not reduction_dependency_masks(
                short_model, selected, primes=GOOD_PRIMES, count=6
            ):
                independent = indices
                break
    return EdgeWallFiber(
        branch,
        current,
        q_denominator,
        root,
        c,
        scale,
        a3,
        a4,
        discriminant,
        independent,
        short_model,
        short_points,
    )


def edge_wall_polynomial(
    q_denominator: int, current: str = "edge"
) -> sp.Poly:
    z = sp.symbols("z")
    t = sp.Rational(1, q_denominator)
    transport = t * (1 - t * t)
    area = (1 + t * t) ** 2
    if current == "edge":
        c = (z * z - area - 12 * transport) / (24 * transport)
    elif current == "r-minus-s":
        offset = (4 * area + 24 * transport - 4 * (1 + t * t) * z)
        offset /= z * z - area - 12 * transport
        c = 1 + offset
    else:
        raise ValueError("unknown extra-section current")
    wall = 64 * transport**2 * (1 + c + c * c) ** 3
    wall -= 27 * c * c * (1 + c) ** 2 * area**2
    numerator = sp.factor(sp.together(wall).as_numer_denom()[0])
    return sp.Poly(numerator, z)


def real_edge_wall_roots(
    q_denominator: int, current: str = "edge", digits: int = 80
) -> tuple[Decimal, ...]:
    roots = sp.nroots(
        edge_wall_polynomial(q_denominator, current), n=digits, maxsteps=300
    )
    result = []
    for root in roots:
        real, imaginary = root.as_real_imag()
        if abs(imaginary) < sp.Float(10) ** (-(digits // 2)):
            result.append(Decimal(str(real)))
    return tuple(sorted(result))


def descent(
    q_denominator: int,
    maximum_denominator: int,
    certify_count: int,
    current: str = "edge",
) -> tuple[EdgeWallFiber, ...]:
    candidates: dict[Fraction, int] = {}
    for branch, wall_root in enumerate(real_edge_wall_roots(q_denominator, current)):
        for root in continued_fraction_convergents(wall_root, maximum_denominator):
            candidates.setdefault(root, branch)
    fibers = []
    for root, branch in candidates.items():
        fiber = edge_wall_fiber(branch, q_denominator, root, current=current)
        if fiber is not None:
            fibers.append(fiber)
    fibers.sort(key=lambda item: abs(item.discriminant))
    for index, fiber in enumerate(fibers[:certify_count]):
        certified = edge_wall_fiber(
            fiber.branch,
            fiber.q,
            fiber.root,
            certify_independence=True,
            current=current,
        )
        assert certified is not None
        fibers[index] = certified
    return tuple(fibers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--max-denominator", type=int, default=100)
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument("--certify-count", type=int, default=0)
    parser.add_argument("--current", choices=("edge", "r-minus-s"), default="edge")
    args = parser.parse_args()
    fibers = descent(args.q, args.max_denominator, args.certify_count, args.current)
    print("current=", args.current)
    print("t=1/", args.q, sep="")
    print("wall branches=", len(real_edge_wall_roots(args.q, args.current)))
    print("noncollision convergents=", len(fibers))
    for fiber in fibers[: args.show]:
        print(
            "branch=", fiber.branch,
            "z=", fiber.root,
            "c=", fiber.c,
            "scale=", fiber.scale,
            "a3=", fiber.a3,
            "a4=", fiber.a4,
            "Delta=", fiber.discriminant,
            "rank6=", fiber.independent_indices,
            "sub_target_discriminant=",
            abs(fiber.discriminant) < RANK_SIX_CONDUCTOR_TARGET,
        )


if __name__ == "__main__":
    main()
