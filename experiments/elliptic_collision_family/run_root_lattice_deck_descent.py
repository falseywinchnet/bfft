"""Klein-four descent of the root-lattice rank-six section system.

The true shape ``c(t)`` is invariant under

    a(t) = -1/t,       b(t) = (t+1)/(t-1).

Consequently the four presentations in a deck orbit define one elliptic
curve.  This script transports every visible section to the same c-only
model and adds its four conjugates.  The resulting orbit sums measure the
rational Mordell--Weil directions that survive when c is rational but t is
not.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / "elliptic_rank_support"
sys.path.insert(0, str(SUPPORT))

from elliptic_rank_support import add_points, reduction_dependency_masks  # noqa: E402
from run_weight_current_attack import (  # noqa: E402
    root_lattice_carrier_root,
    root_lattice_shape,
)
from seven_section_surface import (  # noqa: E402
    linear_abscissa_section,
    surface_fiber,
)


Q = Fraction


def deck_orbit(t: Fraction) -> tuple[Fraction, ...]:
    t = Q(t)
    pending = [t]
    orbit = set()
    while pending:
        value = pending.pop()
        if value in orbit:
            continue
        if value in (0, 1):
            raise ValueError("deck orbit meets a pole")
        orbit.add(value)
        pending.extend((-1 / value, (value + 1) / (value - 1)))
    if len(orbit) != 4:
        raise ArithmeticError(f"expected a four-element orbit, got {orbit}")
    return tuple(sorted(orbit))


def canonical_short_model(
    c: Fraction, net_scale: Fraction
) -> tuple[int, int, int]:
    c, net_scale = map(Q, (c, net_scale))
    q = -2 * (c + 1) * (6 * c + 25) / 3
    a3 = q * q * net_scale**3
    a4 = -q * q * (c * c + c + 1) * net_scale**4
    if a3.denominator != 1 or a4.denominator != 1:
        raise ValueError("net scale does not produce an integral model")
    return 0, 16 * int(a4), 16 * int(a3) ** 2


def canonical_short_points(
    t: Fraction, net_scale: Fraction
) -> tuple[tuple[Fraction, Fraction], ...]:
    t, net_scale = map(Q, (t, net_scale))
    c = root_lattice_shape(t)
    fiber = surface_fiber(c, t)
    root = root_lattice_carrier_root(c, t)
    extra = linear_abscissa_section(c, t, Q(3), Q(2), root)
    standard_points = fiber.seven_points() + (extra,)
    transport = t * (1 - t * t)
    u = transport / (1 + t * t)
    q = -2 * (c + 1) * (6 * c + 25) / 3
    canonical_a3 = q * q
    points = []
    for x, y in standard_points:
        generalized_x = x * net_scale**2 / u**2
        generalized_y = (y - fiber.center_height / 2) * net_scale**3 / u**3
        short_x = 4 * generalized_x
        short_y = 4 * (2 * generalized_y + canonical_a3 * net_scale**3)
        points.append((short_x, short_y))
    return tuple(points)


def orbit_sums(
    t: Fraction = Q(2), net_scale: Fraction = Q(343, 25)
) -> tuple[
    tuple[int, int, int],
    tuple[tuple[Fraction, Fraction], ...],
    tuple[tuple[Fraction, Fraction], ...],
]:
    orbit = deck_orbit(t)
    c = root_lattice_shape(t)
    assert all(root_lattice_shape(value) == c for value in orbit)
    model = canonical_short_model(c, net_scale)
    presentations = tuple(canonical_short_points(value, net_scale) for value in orbit)
    for points in presentations:
        for x, y in points:
            assert y * y == x**3 + model[1] * x + model[2]

    sums = []
    for section_index in range(8):
        total = None
        for points in presentations:
            total = add_points(total, points[section_index], model)
        if total is None:
            continue
        if total not in sums and (total[0], -total[1]) not in sums:
            sums.append(total)
    return model, tuple(sums), presentations[0]


def independent_orbit_sum_rank() -> tuple[int, tuple[int, ...] | None]:
    model, sums, _ = orbit_sums()
    for rank in range(min(6, len(sums)), 0, -1):
        for indices in combinations(range(len(sums)), rank):
            selected = tuple(sums[index] for index in indices)
            if not reduction_dependency_masks(model, selected, count=rank):
                return rank, indices
    return 0, None


def main() -> None:
    model, sums, _ = orbit_sums()
    print("deck orbit:", deck_orbit(Q(2)))
    print("short model:", model)
    for index, point in enumerate(sums):
        print(index, point)
    print("independent orbit-sum rank:", independent_orbit_sum_rank())


if __name__ == "__main__":
    main()
