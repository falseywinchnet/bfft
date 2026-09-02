"""Render a Cornell-style colored-object scene from the transport field."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .budgeted import BudgetedTransportGeometry, solve_budgeted_transport
from .transport import Patch, SphereOccluder


@dataclass(frozen=True)
class SphereSpec:
    center: np.ndarray
    radius: float
    albedo: np.ndarray
    object_id: int
    group: str


def _plane_patches(
    patches: list[Patch],
    groups: dict[str, list[int]],
    *,
    group: str,
    origin: tuple[float, float, float],
    span_u: tuple[float, float, float],
    span_v: tuple[float, float, float],
    count_u: int,
    count_v: int,
    normal: tuple[float, float, float],
    albedo: tuple[float, float, float],
    emission_density: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    origin_value = np.asarray(origin, dtype=np.float64)
    u = np.asarray(span_u, dtype=np.float64)
    v = np.asarray(span_v, dtype=np.float64)
    du = u / count_u
    dv = v / count_v
    area = float(np.linalg.norm(np.cross(du, dv)))
    radius = 0.5 * math.sqrt(
        float(np.dot(du, du)) + float(np.dot(dv, dv))
    )
    indices = groups.setdefault(group, [])
    for i in range(count_u):
        for j in range(count_v):
            center = origin_value + (i + 0.5) * du + (j + 0.5) * dv
            indices.append(len(patches))
            patches.append(Patch(
                center=center,
                normal=np.asarray(normal),
                area=area,
                radius=radius,
                albedo=np.asarray(albedo),
                emission=area * np.asarray(emission_density),
            ))


def _sphere_patches(
    patches: list[Patch],
    groups: dict[str, list[int]],
    sphere: SphereSpec,
    count: int,
) -> None:
    area = 4.0 * math.pi * sphere.radius * sphere.radius / count
    radius = 1.15 * math.sqrt(area / math.pi)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    indices = groups.setdefault(sphere.group, [])
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radial = math.sqrt(max(1.0 - y * y, 0.0))
        angle = golden * index
        normal = np.array((
            radial * math.cos(angle),
            y,
            radial * math.sin(angle),
        ))
        indices.append(len(patches))
        patches.append(Patch(
            center=sphere.center + sphere.radius * normal,
            normal=normal,
            area=area,
            radius=radius,
            albedo=sphere.albedo,
            emission=np.zeros(3),
            object_id=sphere.object_id,
        ))


def build_standard_scene() -> tuple[
    tuple[Patch, ...],
    tuple[SphereOccluder, ...],
    dict[str, np.ndarray],
    tuple[SphereSpec, ...],
]:
    patches: list[Patch] = []
    raw_groups: dict[str, list[int]] = {}
    white = (0.72, 0.74, 0.76)
    _plane_patches(
        patches,
        raw_groups,
        group="floor",
        origin=(-3.0, 0.0, -5.0),
        span_u=(6.0, 0.0, 0.0),
        span_v=(0.0, 0.0, 6.0),
        count_u=6,
        count_v=6,
        normal=(0.0, 1.0, 0.0),
        albedo=white,
    )
    _plane_patches(
        patches,
        raw_groups,
        group="back",
        origin=(-3.0, 0.0, -5.0),
        span_u=(6.0, 0.0, 0.0),
        span_v=(0.0, 4.0, 0.0),
        count_u=6,
        count_v=4,
        normal=(0.0, 0.0, 1.0),
        albedo=white,
    )
    _plane_patches(
        patches,
        raw_groups,
        group="left_wall",
        origin=(-3.0, 0.0, -5.0),
        span_u=(0.0, 0.0, 6.0),
        span_v=(0.0, 4.0, 0.0),
        count_u=5,
        count_v=4,
        normal=(1.0, 0.0, 0.0),
        albedo=(0.58, 0.16, 0.12),
    )
    _plane_patches(
        patches,
        raw_groups,
        group="right_wall",
        origin=(3.0, 0.0, 1.0),
        span_u=(0.0, 0.0, -6.0),
        span_v=(0.0, 4.0, 0.0),
        count_u=5,
        count_v=4,
        normal=(-1.0, 0.0, 0.0),
        albedo=(0.12, 0.22, 0.62),
    )
    _plane_patches(
        patches,
        raw_groups,
        group="ceiling",
        origin=(-3.0, 4.0, 1.0),
        span_u=(6.0, 0.0, 0.0),
        span_v=(0.0, 0.0, -6.0),
        count_u=6,
        count_v=6,
        normal=(0.0, -1.0, 0.0),
        albedo=white,
    )
    _plane_patches(
        patches,
        raw_groups,
        group="area_light",
        origin=(-1.15, 3.92, -3.0),
        span_u=(2.3, 0.0, 0.0),
        span_v=(0.0, 0.0, 1.7),
        count_u=3,
        count_v=2,
        normal=(0.0, -1.0, 0.0),
        albedo=(0.02, 0.02, 0.02),
        emission_density=(34.0, 31.0, 27.0),
    )
    spheres = (
        SphereSpec(
            center=np.array((-1.35, 1.05, -2.65)),
            radius=0.72,
            albedo=np.array((0.78, 0.08, 0.055)),
            object_id=10,
            group="red_sphere",
        ),
        SphereSpec(
            center=np.array((1.22, 0.92, -3.15)),
            radius=0.62,
            albedo=np.array((0.055, 0.20, 0.82)),
            object_id=11,
            group="blue_sphere",
        ),
        SphereSpec(
            center=np.array((0.15, 0.77, -1.20)),
            radius=0.48,
            albedo=np.array((0.08, 0.72, 0.20)),
            object_id=12,
            group="green_sphere",
        ),
    )
    for sphere in spheres:
        _sphere_patches(patches, raw_groups, sphere, 24)
    occluders = tuple(
        SphereOccluder(
            center=sphere.center,
            radius=sphere.radius,
            object_id=sphere.object_id,
        )
        for sphere in spheres
    )
    groups = {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in raw_groups.items()
    }
    return tuple(patches), occluders, groups, spheres


def _camera_rays(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    camera = np.array((0.0, 2.15, 6.8))
    target = np.array((0.0, 1.55, -2.15))
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array((0.0, 1.0, 0.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    aspect = width / height
    scale = math.tan(math.radians(48.0) * 0.5)
    xx = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * aspect * scale
    yy = (1.0 - 2.0 * (np.arange(height) + 0.5) / height) * scale
    x_grid, y_grid = np.meshgrid(xx, yy)
    directions = (
        forward[None, None, :]
        + x_grid[:, :, None] * right[None, None, :]
        + y_grid[:, :, None] * up[None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    return camera, directions


def _bounded_plane_hit(
    origin: np.ndarray,
    directions: np.ndarray,
    axis: int,
    coordinate: float,
    bounds: tuple[tuple[int, float, float], tuple[int, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    denominator = directions[:, :, axis]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (coordinate - origin[axis]) / denominator
    point = origin[None, None, :] + distance[:, :, None] * directions
    valid = distance > 1.0e-6
    for bound_axis, lower, upper in bounds:
        valid &= (point[:, :, bound_axis] >= lower) & (
            point[:, :, bound_axis] <= upper
        )
    return np.where(valid, distance, np.inf), point


def _nearest_patch_radiance(
    points: np.ndarray,
    patch_centers: np.ndarray,
    patch_radiance: np.ndarray,
) -> np.ndarray:
    result = np.empty((points.shape[0], 3), dtype=np.float64)
    neighbor_count = min(4, patch_centers.shape[0])
    for start in range(0, points.shape[0], 16384):
        stop = min(start + 16384, points.shape[0])
        difference = (
            points[start:stop, None, :] - patch_centers[None, :, :]
        )
        distance_squared = np.sum(difference * difference, axis=2)
        nearest = np.argpartition(
            distance_squared, neighbor_count - 1, axis=1
        )[:, :neighbor_count]
        selected_distance = np.take_along_axis(
            distance_squared, nearest, axis=1
        )
        weight = 1.0 / np.maximum(selected_distance, 1.0e-8)
        weight /= np.sum(weight, axis=1, keepdims=True)
        result[start:stop] = np.sum(
            weight[:, :, None] * patch_radiance[nearest], axis=1
        )
    return result


def render_scene(
    patches: tuple[Patch, ...],
    groups: dict[str, np.ndarray],
    spheres: tuple[SphereSpec, ...],
    outgoing: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    origin, directions = _camera_rays(width, height)
    nearest_t = np.full((height, width), np.inf)
    surface_id = np.full((height, width), -1, dtype=np.int32)
    hit_point = np.zeros((height, width, 3), dtype=np.float64)
    names = [
        "floor",
        "back",
        "left_wall",
        "right_wall",
        "ceiling",
        "area_light",
    ] + [sphere.group for sphere in spheres]
    name_id = {name: index for index, name in enumerate(names)}
    plane_specs = (
        ("floor", 1, 0.0, ((0, -3.0, 3.0), (2, -5.0, 1.0))),
        ("back", 2, -5.0, ((0, -3.0, 3.0), (1, 0.0, 4.0))),
        ("left_wall", 0, -3.0, ((1, 0.0, 4.0), (2, -5.0, 1.0))),
        ("right_wall", 0, 3.0, ((1, 0.0, 4.0), (2, -5.0, 1.0))),
        ("ceiling", 1, 4.0, ((0, -3.0, 3.0), (2, -5.0, 1.0))),
        ("area_light", 1, 3.92, ((0, -1.15, 1.15), (2, -3.0, -1.3))),
    )
    for name, axis, coordinate, bounds in plane_specs:
        distance, point = _bounded_plane_hit(
            origin, directions, axis, coordinate, bounds
        )
        selected = distance < nearest_t
        nearest_t[selected] = distance[selected]
        surface_id[selected] = name_id[name]
        hit_point[selected] = point[selected]

    flat_direction = directions.reshape(-1, 3)
    for sphere in spheres:
        relative = origin - sphere.center
        b = flat_direction @ relative
        c = float(np.dot(relative, relative) - sphere.radius * sphere.radius)
        discriminant = b * b - c
        distance = np.full(flat_direction.shape[0], np.inf)
        visible = discriminant >= 0.0
        root = np.sqrt(np.maximum(discriminant[visible], 0.0))
        first = -b[visible] - root
        second = -b[visible] + root
        distance[visible] = np.where(first > 1.0e-6, first, second)
        distance = distance.reshape(height, width)
        selected = (distance > 1.0e-6) & (distance < nearest_t)
        point = origin[None, None, :] + distance[:, :, None] * directions
        nearest_t[selected] = distance[selected]
        surface_id[selected] = name_id[sphere.group]
        hit_point[selected] = point[selected]

    area = np.array([patch.area for patch in patches])
    radiance = outgoing / (math.pi * area[:, None])
    linear = np.zeros((height, width, 3), dtype=np.float64)
    centers = np.stack([patch.center for patch in patches])
    for name in names:
        selected = surface_id == name_id[name]
        if not np.any(selected):
            continue
        indices = groups[name]
        linear[selected] = _nearest_patch_radiance(
            hit_point[selected], centers[indices], radiance[indices]
        )
    # A filmic exponential map preserves the transport ratios while placing
    # the broad emitter and diffuse room within one display range.
    mapped = 1.0 - np.exp(-0.72 * np.maximum(linear, 0.0))
    mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2)
    return np.asarray(np.round(255.0 * mapped), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/photonic_transport_field/standard_scene.png"),
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--error-fraction", type=float, default=2.0e-3)
    args = parser.parse_args()
    patches, occluders, groups, spheres = build_standard_scene()
    emission = np.stack([patch.emission for patch in patches])
    albedo = np.stack([patch.albedo for patch in patches])
    requested_error = args.error_fraction * float(np.sum(emission))
    geometry = BudgetedTransportGeometry(
        patches,
        occluders=occluders,
        admissibility=0.08,
        # The room has an open camera face.  This reciprocal global factor is
        # unresolved escape plus a conservative guard for centroid quadrature;
        # unlike row normalization it preserves every pair ratio.
        transport_scale=0.80,
    )
    started = time.perf_counter()
    result = solve_budgeted_transport(
        geometry,
        emission,
        albedo,
        outgoing_error_budget=requested_error,
    )
    solve_seconds = time.perf_counter() - started
    image = render_scene(
        patches,
        groups,
        spheres,
        result.outgoing,
        width=args.width,
        height=args.height,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(args.out)
    diagnostics = {
        "scene": "Cornell-style room, three colored diffuse spheres, global area light",
        "patches": len(patches),
        "sphere_occluders": len(occluders),
        "solve_seconds": solve_seconds,
        "transport_depths": result.depth_count,
        "requested_outgoing_l1_error": requested_error,
        "certified_outgoing_l1_error_bound": result.outgoing_error_bound,
        "depth_work": [{
            "depth": int(record["depth"]),
            "frontier_energy": float(record["frontier_energy"]),
            "cluster_pair_visits": int(record["cluster_pair_visits"]),
            "blocks": int(record["block_count"]),
            "visibility_tests": int(record["exact_visibility_tests"]),
            "next_frontier_energy": float(record["next_frontier_energy"]),
        } for record in result.depth_diagnostics],
        "image": str(args.out),
    }
    diagnostics_path = args.out.with_suffix(".json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
