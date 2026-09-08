"""First-survivor audit after removing the proved height-loss walls.

This is deliberately not a box search.  On each already-constructed rational
norm-conic slice it follows only the continued-fraction characteristics of the
discriminant-zero branches, orders them by integral model discriminant, and
2-saturates the first specialization surviving every known symbolic wall.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction

from run_rank_six_wall_descent import (
    CURRENT_SLICES,
    RANK_SIX_CONDUCTOR_TARGET,
    CurrentSlice,
    WallFiber,
    descent,
    saturate_six_sections,
)
from seven_section_surface import (
    base_second_height_forms,
    base_height_shell_forms,
    cross_relation_forms,
    exposed_base_height_forms,
    sixth_height_shell_forms,
)


@dataclass(frozen=True)
class SafeSurvivor:
    current_slice: CurrentSlice
    fiber: WallFiber
    ordinal: int
    saturation_status: str
    saturation_trace: tuple[object, ...]


def known_height_safe(c: Fraction, t: Fraction) -> bool:
    """Reject exactly the rank-loss divisors proved in the current theory."""

    return all(
        cross_relation_forms(c, t)
        + base_height_shell_forms(c, t)
        + base_second_height_forms(c, t)
        + exposed_base_height_forms(c, t)
        + sixth_height_shell_forms(c, t)
    )


def first_safe_survivor(
    current_slice: CurrentSlice, maximum_denominator: int
) -> SafeSurvivor | None:
    """Return and saturate only the first discriminant-ordered safe fiber."""

    for ordinal, fiber in enumerate(
        descent(maximum_denominator, certify_count=0, current_slice=current_slice),
        start=1,
    ):
        if not known_height_safe(fiber.c, current_slice.t):
            continue
        status, _generators, trace = saturate_six_sections(fiber)
        return SafeSurvivor(current_slice, fiber, ordinal, status, trace)
    return None


def audit(maximum_denominator: int) -> tuple[SafeSurvivor, ...]:
    survivors = []
    for current_slice in CURRENT_SLICES.values():
        survivor = first_safe_survivor(current_slice, maximum_denominator)
        if survivor is not None:
            survivors.append(survivor)
    return tuple(sorted(survivors, key=lambda item: abs(item.fiber.discriminant)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-denominator", type=int, default=100)
    args = parser.parse_args()
    survivors = audit(args.max_denominator)
    for item in survivors:
        fiber = item.fiber
        print(
            "slice=", item.current_slice.name,
            "ordinal=", item.ordinal,
            "t=", item.current_slice.t,
            "k=", fiber.slope,
            "c=", fiber.c,
            "Delta=", fiber.discriminant,
            "sub_target=", abs(fiber.discriminant) < RANK_SIX_CONDUCTOR_TARGET,
            "status=", item.saturation_status,
            "trace=", item.saturation_trace,
        )


if __name__ == "__main__":
    main()
