"""Deterministic control scenes for the photonic transport-field experiment."""

from __future__ import annotations

import math

import numpy as np

from .transport import Patch, SphereOccluder


def facing_patch_grids(
    *,
    side: int = 8,
    separation: float = 8.0,
    extent: float = 2.0,
    occluder_radius: float | None = None,
) -> tuple[tuple[Patch, ...], tuple[SphereOccluder, ...]]:
    """Return two inward-facing square patch lattices."""
    if side < 1:
        raise ValueError("grid side must be positive")
    step = extent / side
    coordinates = -0.5 * extent + (np.arange(side) + 0.5) * step
    area = step * step
    radius = 0.5 * math.sqrt(2.0) * step
    patches: list[Patch] = []
    for x, normal, albedo in (
        (0.0, (1.0, 0.0, 0.0), (0.72, 0.10, 0.08)),
        (separation, (-1.0, 0.0, 0.0), (0.76, 0.76, 0.76)),
    ):
        for y in coordinates:
            for z in coordinates:
                emission = (
                    (2.0, 1.6, 1.2)
                    if x == separation and abs(y) < step and abs(z) < step
                    else (0.0, 0.0, 0.0)
                )
                patches.append(Patch(
                    center=np.array((x, y, z)),
                    normal=np.array(normal),
                    area=area,
                    radius=radius,
                    albedo=np.array(albedo),
                    emission=np.array(emission),
                ))
    blockers: tuple[SphereOccluder, ...] = ()
    if occluder_radius is not None:
        blockers = (SphereOccluder(
            center=np.array((0.5 * separation, 0.0, 0.0)),
            radius=float(occluder_radius),
        ),)
    return tuple(patches), blockers


def two_patch_color_bleed() -> tuple[Patch, Patch]:
    """White emitter facing a spectrally red diffuse receiver."""
    return (
        Patch(
            center=np.array((0.0, 0.0, 0.0)),
            normal=np.array((1.0, 0.0, 0.0)),
            area=0.25,
            radius=0.2,
            albedo=np.array((0.0, 0.0, 0.0)),
            emission=np.array((1.0, 1.0, 1.0)),
        ),
        Patch(
            center=np.array((1.0, 0.0, 0.0)),
            normal=np.array((-1.0, 0.0, 0.0)),
            area=0.25,
            radius=0.2,
            albedo=np.array((0.8, 0.1, 0.05)),
            emission=np.zeros(3),
        ),
    )
