#!/usr/bin/env python3
"""Complete one-shot transport-measure cell reconstruction.

Pipeline:

1. build the BFFT pass/transport stack;
2. collapse it into one transported support measure and precision tensor;
3. let the integral of that measure determine the cell count;
4. quantize the measure once, without candidates or residual feedback;
5. obtain every ellipse from the transported precision tensor;
6. solve one global affine partition-of-unity reconstruction.

The quantizer has no requested count, cell creation loop, ownership, ranking,
top-k selection, deletion, or geometry refinement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import LinearOperator, cg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
import bfft  # noqa: E402
from bfft._core import _meyer_padded  # noqa: E402
from bfft.vision import (  # noqa: E402
    SingleStageDecompositionObjective,
    compact_support_operators,
)
from bfft_flow_stage_geometry import (  # noqa: E402
    _event_geometry,
    _normal_transport,
    build_flow_volume,
)
from flow_support_measure import infer_support_measure, push_measure  # noqa: E402
from flow_volume_cells import (  # noqa: E402
    Population,
    fit_layered_population,
    fit_population,
    r2_control,
    support_samples,
)
from resource_transport_cells import _r2_sites  # noqa: E402
from transport_voronoi import _fit_rgb, srgb_to_lab  # noqa: E402


def _load_image(path: str | None, gallery_key: str) -> tuple[np.ndarray, str]:
    if path:
        from skimage.io import imread

        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def _unit_r2(count: int) -> np.ndarray:
    plastic = 1.324717957244746
    index = np.arange(1, count + 1, dtype=np.float64)
    return np.column_stack([
        np.mod(0.5 + index / plastic, 1.0),
        np.mod(0.5 + index / (plastic * plastic), 1.0),
    ])


def quantize_rosenblatt(density: np.ndarray) -> np.ndarray:
    """Push an R2 sequence through the exact separable density CDF."""
    density = np.maximum(np.asarray(density, dtype=np.float64), 1e-15)
    height, width = density.shape
    count = max(1, int(round(float(np.sum(density)))))
    unit = _unit_r2(count)

    row_mass = np.sum(density, axis=1)
    row_cdf = np.cumsum(row_mass)
    row_cdf = (row_cdf - 0.5 * row_mass) / row_cdf[-1]
    y = np.interp(
        unit[:, 1],
        np.concatenate(([0.0], row_cdf, [1.0])),
        np.concatenate(([0.0], np.arange(height) + 0.5, [height - 1e-9])),
    )
    row = np.clip(np.floor(y).astype(np.intp), 0, height - 1)
    x = np.empty(count, dtype=np.float64)
    coordinate_x = np.arange(width, dtype=np.float64) + 0.5
    for y_index in np.unique(row):
        selected = row == y_index
        mass = density[y_index]
        cdf = np.cumsum(mass)
        cdf = (cdf - 0.5 * mass) / cdf[-1]
        x[selected] = np.interp(
            unit[selected, 0],
            np.concatenate(([0.0], cdf, [1.0])),
            np.concatenate(([0.0], coordinate_x, [width - 1e-9])),
        )
    return np.column_stack([
        np.clip(x, 0.0, width - 1.0),
        np.clip(y, 0.0, height - 1.0),
    ])


def quantize_poisson_map(density: np.ndarray) -> tuple[np.ndarray, dict]:
    """One closed-form linearized Monge map of uniform R2 sites."""
    density = np.maximum(np.asarray(density, dtype=np.float64), 1e-15)
    height, width = density.shape
    count = max(1, int(round(float(np.sum(density)))))
    centers = _r2_sites(count, width, height)
    desired = density / max(float(np.mean(density)), 1e-30)
    source = 1.0 - desired
    source_hat = dctn(source, type=2, norm="ortho")
    eig_y = 2.0 * np.cos(np.pi * np.arange(height) / height) - 2.0
    eig_x = 2.0 * np.cos(np.pi * np.arange(width) / width) - 2.0
    eigenvalue = eig_y[:, None] + eig_x[None, :]
    potential_hat = np.zeros_like(source_hat)
    nonzero = np.abs(eigenvalue) > 1e-14
    potential_hat[nonzero] = source_hat[nonzero] / eigenvalue[nonzero]
    potential = idctn(potential_hat, type=2, norm="ortho")
    velocity_y, velocity_x = np.gradient(potential)

    # Prevent the linearized map from folding.  This damping is determined by
    # its own Jacobian, not by reconstruction error or a site-count target.
    dvx_dx = ndi.sobel(velocity_x, axis=1, mode="reflect") / 8.0
    dvy_dy = ndi.sobel(velocity_y, axis=0, mode="reflect") / 8.0
    dvx_dy = ndi.sobel(velocity_x, axis=0, mode="reflect") / 8.0
    dvy_dx = ndi.sobel(velocity_y, axis=1, mode="reflect") / 8.0
    symmetric_xy = 0.5 * (dvx_dy + dvy_dx)
    trace = dvx_dx + dvy_dy
    disc = np.hypot(dvx_dx - dvy_dy, 2.0 * symmetric_xy)
    minimum_eigenvalue = float(np.min(0.5 * (trace - disc)))
    damping = min(
        1.0,
        0.8 / max(-minimum_eigenvalue, 1e-30),
    )
    coordinates = np.vstack([centers[:, 1], centers[:, 0]])
    dx = damping * ndi.map_coordinates(
        velocity_x, coordinates, order=1, mode="nearest")
    dy = damping * ndi.map_coordinates(
        velocity_y, coordinates, order=1, mode="nearest")
    centers = centers + np.column_stack([dx, dy])
    centers[:, 0] = np.clip(centers[:, 0], 0.0, width - 1.0)
    centers[:, 1] = np.clip(centers[:, 1], 0.0, height - 1.0)
    return centers, {
        "map_damping": damping,
        "displacement_rms": float(np.sqrt(np.mean(dx * dx + dy * dy))),
        "displacement_q95": float(np.percentile(np.hypot(dx, dy), 95.0)),
        "minimum_linearized_jacobian_eigenvalue": minimum_eigenvalue,
    }


def _hash01(x: np.ndarray, y: np.ndarray, salt: float) -> np.ndarray:
    value = np.sin(
        (x + 1.0) * 12.9898 + (y + 1.0) * 78.233 + salt
    ) * 43758.5453123
    return value - np.floor(value)


def quantize_phase(
    density: np.ndarray,
    phase_shift: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Locally quantize measure with a fixed blue-noise-like phase field.

    Interleaved gradient noise has unusually even local threshold strata.
    The image changes only the threshold, never the phase ordering.  Integer
    mass emits that many sites locally; fractional mass emits when it crosses
    the fixed phase.  This is a parallel field quantization, not selection
    from a candidate list.
    """
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    height, width = density.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    phase = np.mod(
        52.9829189 * np.mod(
            0.06711056 * xx + 0.00583715 * yy + phase_shift, 1.0),
        1.0,
    )
    whole = np.floor(density).astype(np.int32)
    fractional = density - whole
    count_at = whole + (phase < fractional)
    flat_count = count_at.ravel()
    pixel = np.repeat(np.arange(height * width), flat_count)
    if pixel.size == 0:
        maximum = int(np.argmax(density))
        pixel = np.array([maximum], dtype=np.int64)
    occurrence = np.concatenate([
        np.arange(value, dtype=np.float64) for value in flat_count
        if value > 0
    ])
    py = pixel // width
    px = pixel % width
    jitter_x = _hash01(px, py, 17.123 + occurrence * 31.7) - 0.5
    jitter_y = _hash01(px, py, 93.731 + occurrence * 47.3) - 0.5
    centers = np.column_stack([
        np.clip(px + 0.5 + 0.8 * jitter_x, 0.0, width - 1.0),
        np.clip(py + 0.5 + 0.8 * jitter_y, 0.0, height - 1.0),
    ])
    return centers, {
        "quantization_error": float(len(centers) - np.sum(density)),
        "maximum_pixel_measure": float(np.max(density)),
    }


def population_from_scale_measure(
    result: dict,
    volume: dict,
    overlap: float,
    advect: bool = True,
) -> tuple[Population, dict]:
    """Quantize the mutually exclusive scale components of one measure."""
    component = np.asarray(
        result["newly_binding"], dtype=np.float64).copy()
    local_requirement = np.asarray(
        result["local_requirement"], dtype=np.float64)
    local_qxx = np.asarray(
        result["local_precision_xx"], dtype=np.float64)
    local_qxy = np.asarray(
        result["local_precision_xy"], dtype=np.float64)
    local_qyy = np.asarray(
        result["local_precision_yy"], dtype=np.float64)
    tx = np.asarray(volume["transport_x"], dtype=np.float64)
    ty = np.asarray(volume["transport_y"], dtype=np.float64)
    confidence = np.asarray(
        volume["transport_confidence"], dtype=np.float64)
    persistence = np.asarray(
        volume["transport_persistence"], dtype=np.float64)

    height, width = component.shape[1:]
    broad_density = (
        float(result["broad_horizon_count"]) / (height * width))
    component[0] = np.maximum(component[0] - broad_density, 0.0)
    broad_count = max(1, int(round(result["broad_horizon_count"])))
    broad_centers = _r2_sites(broad_count, width, height)
    broad_radius = (
        float(result["max_support_px"]) * math.sqrt(float(overlap)))

    centers_parts = [broad_centers]
    major_parts = [np.full(broad_count, broad_radius)]
    minor_parts = [np.full(broad_count, broad_radius)]
    angle_parts = [np.zeros(broad_count)]
    stage_parts = [np.zeros(broad_count, dtype=np.int16)]
    implied_by_stage = []
    realized_by_stage = []
    displacement_parts = [np.zeros(broad_count)]
    golden_shift = 0.6180339887498949
    for stage in range(component.shape[0]):
        centers, _ = quantize_phase(
            component[stage],
            phase_shift=np.mod(stage * golden_shift, 1.0),
        )
        implied = float(np.sum(component[stage]))
        # ``quantize_phase`` guarantees one fallback site for an empty field;
        # empty components are not support obligations and must stay empty.
        if implied < 1e-12:
            centers = np.empty((0, 2), dtype=np.float64)
        implied_by_stage.append(implied)
        realized_by_stage.append(int(len(centers)))
        if not len(centers):
            continue

        qxx = _sample(local_qxx[stage], centers)
        qxy = _sample(local_qxy[stage], centers)
        qyy = _sample(local_qyy[stage], centers)
        trace = qxx + qyy
        disc = np.hypot(qxx - qyy, 2.0 * qxy)
        high = np.maximum(0.5 * (trace + disc), 1e-12)
        low = np.maximum(0.5 * (trace - disc), 1e-12)
        ratio = np.sqrt(high / low)
        density = np.maximum(
            _sample(local_requirement[stage], centers), 1e-12)
        area = float(overlap) / density
        major = np.sqrt(area * ratio / math.pi)
        minor = np.sqrt(area / (math.pi * ratio))
        normal_angle = 0.5 * np.arctan2(2.0 * qxy, qxx - qyy)
        tangent_angle = normal_angle + 0.5 * math.pi

        original = centers.copy()
        if advect:
            for next_stage in range(stage + 1, component.shape[0]):
                gate = _sample(confidence[next_stage], centers)
                gate *= _sample(persistence, centers)
                centers[:, 0] += gate * _sample(tx[next_stage], centers)
                centers[:, 1] += gate * _sample(ty[next_stage], centers)
                centers[:, 0] = np.clip(
                    centers[:, 0], 0.0, component.shape[2] - 1.0)
                centers[:, 1] = np.clip(
                    centers[:, 1], 0.0, component.shape[1] - 1.0)
        displacement_parts.append(np.linalg.norm(
            centers - original, axis=1))
        centers_parts.append(centers)
        major_parts.append(major)
        minor_parts.append(minor)
        angle_parts.append(tangent_angle)
        stage_parts.append(np.full(
            len(centers), stage + 1, dtype=np.int16))

    centers = np.concatenate(centers_parts, axis=0)
    major = np.concatenate(major_parts)
    minor = np.concatenate(minor_parts)
    angle = np.concatenate(angle_parts)
    stages = np.concatenate(stage_parts)
    displacement = np.concatenate(displacement_parts)
    population = Population(
        centers=centers,
        major=major,
        minor=minor,
        angle=angle,
        stage=stages,
        base_cells=0,
    )
    return population, {
        "quantizer": "phase stack",
        "cells": int(len(centers)),
        "measure_integral": float(
            np.sum(component) + result["broad_horizon_count"]),
        "broad_cells": broad_count,
        "overlap": float(overlap),
        "advected": bool(advect),
        "implied_by_stage": implied_by_stage,
        "realized_by_stage": realized_by_stage,
        "displacement_rms": float(np.sqrt(np.mean(displacement**2))),
        "major_p10_p50_p90": [
            float(value) for value in np.percentile(major, (10, 50, 90))
        ],
        "minor_p10_p50_p90": [
            float(value) for value in np.percentile(minor, (10, 50, 90))
        ],
        "ratio_p10_p50_p90": [
            float(value)
            for value in np.percentile(
                major / np.maximum(minor, 1e-12), (10, 50, 90))
        ],
    }


def build_streaming_population(
    rgb: np.ndarray,
    passes: int = 24,
    lam: float = 0.05,
    mu: float = 40.0,
    flow_sweeps: int = 4,
    max_support_fraction: float = 0.18,
    overlap: float = 5.0,
    tensor_sigma: float = 1.0,
    threads: int = 4,
) -> tuple[Population, dict, dict]:
    """Build the complete population with O(image + cells) live memory.

    Two native linear traces are used.  The first measures directional
    persistence.  The second reduces the transported support envelope and
    immediately quantizes each mutually exclusive scale contribution.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    height, width = rgb.shape[:2]
    light = srgb_to_lab(rgb)[..., 0] * 255.0
    plan, padded, top, left, _, _ = _meyer_padded(
        light, lam, mu, passes, 1, 0.0, threads)
    max_length = max(
        float(max_support_fraction) * max(height, width), 1.0)
    frequency_floor = 1.0 / (max_length * max_length)
    broad_density = frequency_floor / math.pi
    broad_count_measure = height * width * broad_density

    signed_x = np.zeros((height, width), dtype=np.float64)
    signed_y = np.zeros((height, width), dtype=np.float64)
    path_length = np.zeros((height, width), dtype=np.float64)
    previous_cartoon = light / 255.0

    def persistence_visit(stage, cartoon_padded, texture_padded):
        nonlocal previous_cartoon
        cartoon = cartoon_padded[
            top:top + height, left:left + width]
        vx, vy, confidence = _normal_transport(
            previous_cartoon, cartoon / 255.0)
        signed_x[:] += confidence * vx
        signed_y[:] += confidence * vy
        path_length[:] += confidence * np.hypot(vx, vy)
        previous_cartoon = (cartoon / 255.0).copy()

    plan.visit(padded, persistence_visit)
    persistence = np.hypot(signed_x, signed_y) / np.maximum(
        path_length, 1e-30)

    centers_parts = []
    major_parts = []
    minor_parts = []
    angle_parts = []
    stage_parts = []
    implied_by_stage = []
    realized_by_stage = []
    scale_mass = np.zeros((height, width), dtype=np.float64)
    scale_moment = np.zeros((height, width), dtype=np.float64)

    broad_count = max(1, int(round(broad_count_measure)))
    centers_parts.append(_r2_sites(broad_count, width, height))
    broad_radius = max_length * math.sqrt(float(overlap))
    major_parts.append(np.full(broad_count, broad_radius))
    minor_parts.append(np.full(broad_count, broad_radius))
    angle_parts.append(np.zeros(broad_count))
    stage_parts.append(np.zeros(broad_count, dtype=np.int16))

    previous_cartoon = light / 255.0
    previous_texture = np.zeros((height, width), dtype=np.float64)
    previous_defect = np.zeros((height, width), dtype=np.float64)
    envelope = None
    qxx = qxy = qyy = None
    golden_shift = 0.6180339887498949

    def reduction_visit(stage, cartoon_padded, texture_padded):
        nonlocal previous_cartoon, previous_texture, previous_defect
        nonlocal envelope, qxx, qxy, qyy
        cartoon = (
            cartoon_padded[top:top + height, left:left + width] / 255.0)
        texture = (
            texture_padded[top:top + height, left:left + width] / 255.0)
        projected = bfft.rof(
            light - texture * 255.0,
            c=lam,
            eta=2.0 * lam,
            sweeps=flow_sweeps,
            tol=0.0,
            threads=threads,
        )
        defect = (cartoon * 255.0 - projected) / 255.0
        vx, vy, confidence = _normal_transport(
            previous_cartoon, cartoon)
        geometry = _event_geometry(
            cartoon - previous_cartoon,
            texture - previous_texture,
            defect - previous_defect,
            tensor_sigma,
            max_length,
        )
        amplitude = np.sqrt(np.maximum(geometry["energy"], 0.0))
        scale = max(float(np.percentile(amplitude, 99.5)), 1e-30)
        reliability = amplitude / (amplitude + 1e-5 * scale)
        high = reliability * geometry["high_frequency"] + frequency_floor
        low = reliability * geometry["low_frequency"] + frequency_floor
        required = np.sqrt(high * low) / math.pi
        tangent = geometry["angle"]
        tangent_x = np.cos(tangent)
        tangent_y = np.sin(tangent)
        normal_x = -tangent_y
        normal_y = tangent_x
        local_qxx = (
            low * tangent_x * tangent_x + high * normal_x * normal_x)
        local_qxy = (
            low * tangent_x * tangent_y + high * normal_x * normal_y)
        local_qyy = (
            low * tangent_y * tangent_y + high * normal_y * normal_y)

        if envelope is None:
            binding = required.copy()
            binding -= broad_density
            np.maximum(binding, 0.0, out=binding)
            envelope = required.copy()
            qxx = local_qxx.copy()
            qxy = local_qxy.copy()
            qyy = local_qyy.copy()
        else:
            gate = confidence * persistence
            pushed = push_measure(envelope, gate * vx, gate * vy)
            safe = np.maximum(pushed, 1e-30)
            pushed_qxx = push_measure(
                envelope * qxx, gate * vx, gate * vy) / safe
            pushed_qxy = push_measure(
                envelope * qxy, gate * vx, gate * vy) / safe
            pushed_qyy = push_measure(
                envelope * qyy, gate * vx, gate * vy) / safe
            binding = np.maximum(required - pushed, 0.0)
            local_binds = required >= pushed
            envelope = np.where(local_binds, required, pushed)
            qxx = np.where(local_binds, local_qxx, pushed_qxx)
            qxy = np.where(local_binds, local_qxy, pushed_qxy)
            qyy = np.where(local_binds, local_qyy, pushed_qyy)

        implied = float(np.sum(binding))
        centers, _ = quantize_phase(
            binding,
            phase_shift=np.mod((stage - 1) * golden_shift, 1.0),
        )
        if implied < 1e-12:
            centers = np.empty((0, 2), dtype=np.float64)
        implied_by_stage.append(implied)
        realized_by_stage.append(int(len(centers)))
        scale_mass[:] += binding
        scale_moment[:] += stage * binding
        if len(centers):
            sample_qxx = _sample(local_qxx, centers)
            sample_qxy = _sample(local_qxy, centers)
            sample_qyy = _sample(local_qyy, centers)
            trace = sample_qxx + sample_qyy
            disc = np.hypot(
                sample_qxx - sample_qyy, 2.0 * sample_qxy)
            sample_high = np.maximum(
                0.5 * (trace + disc), 1e-12)
            sample_low = np.maximum(
                0.5 * (trace - disc), 1e-12)
            ratio = np.sqrt(sample_high / sample_low)
            sampled_density = np.maximum(
                _sample(required, centers), 1e-12)
            area = float(overlap) / sampled_density
            major_parts.append(np.sqrt(area * ratio / math.pi))
            minor_parts.append(np.sqrt(area / (math.pi * ratio)))
            angle_parts.append(
                0.5 * np.arctan2(
                    2.0 * sample_qxy, sample_qxx - sample_qyy)
                + 0.5 * math.pi)
            centers_parts.append(centers)
            stage_parts.append(np.full(
                len(centers), stage, dtype=np.int16))
        previous_cartoon = cartoon.copy()
        previous_texture = texture.copy()
        previous_defect = defect

    plan.visit(padded, reduction_visit)
    centers = np.concatenate(centers_parts, axis=0)
    major = np.concatenate(major_parts)
    minor = np.concatenate(minor_parts)
    angle = np.concatenate(angle_parts)
    stages = np.concatenate(stage_parts)
    population = Population(
        centers=centers,
        major=major,
        minor=minor,
        angle=angle,
        stage=stages,
        base_cells=broad_count,
    )
    support = {
        "envelope": envelope[None].astype(np.float32),
        "precision_xx": qxx.astype(np.float32),
        "precision_xy": qxy.astype(np.float32),
        "precision_yy": qyy.astype(np.float32),
        "scale_mean": (
            scale_moment / np.maximum(scale_mass, 1e-30)
        ).astype(np.float32),
        "transported_count": float(np.sum(envelope)),
        "broad_horizon_count": broad_count_measure,
        "max_support_px": max_length,
    }
    report = {
        "cells": int(len(centers)),
        "measure_integral": float(
            np.sum(implied_by_stage) + broad_count_measure),
        "broad_cells": broad_count,
        "implied_by_stage": implied_by_stage,
        "realized_by_stage": realized_by_stage,
        "major_p10_p50_p90": [
            float(value) for value in np.percentile(major, (10, 50, 90))
        ],
        "minor_p10_p50_p90": [
            float(value) for value in np.percentile(minor, (10, 50, 90))
        ],
    }
    return population, support, report


def _sample(field: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ndi.map_coordinates(
        np.asarray(field, dtype=np.float64),
        np.vstack([centers[:, 1], centers[:, 0]]),
        order=1,
        mode="nearest",
    )


def site_id_colours(count: int) -> np.ndarray:
    """Return SAD-compatible deterministic colours for integer site IDs."""
    site = np.arange(max(int(count), 0), dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF)

    h = site.copy()
    h = ((h ^ np.uint64(61)) ^ (h >> np.uint64(16))) & mask
    h = (h + (h << np.uint64(3))) & mask
    h = (h ^ (h >> np.uint64(4))) & mask
    red = (h * np.uint64(0x27D4EB2D)) & mask

    h = (site * np.uint64(2654435761)) & mask
    h = ((h ^ np.uint64(61)) ^ (h >> np.uint64(16))) & mask
    h = (h + (h << np.uint64(3))) & mask
    green = (h * np.uint64(0x27D4EB2D)) & mask

    h = (site * np.uint64(1103515245)) & mask
    h = ((h ^ np.uint64(61)) ^ (h >> np.uint64(16))) & mask
    h = (h ^ (h >> np.uint64(4))) & mask
    blue = (h * np.uint64(0x27D4EB2D)) & mask

    scale = 1.0 / float(0xFFFFFF)
    return np.column_stack([
        (red & np.uint64(0xFFFFFF)).astype(np.float64) * scale,
        (green & np.uint64(0xFFFFFF)).astype(np.float64) * scale,
        (blue & np.uint64(0xFFFFFF)).astype(np.float64) * scale,
    ])


def population_geometry_views(
    population: Population,
    height: int,
    width: int,
    power: float = 2.0,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """Rasterize the actual compact-support partition for inspection.

    ``soft_site_ids`` replaces each fitted affine jet with a hashed ID colour
    and blends those colours with the exact partition-of-unity weights used by
    reconstruction.  This is the direct analogue of SAD's Site IDs view.
    ``dominant_site_ids`` rounds that partition only for diagnosis; ownership
    is not used anywhere in the fitting method.  ``cell_outlines`` displays
    the literal transported ellipses before blending.
    """
    height, width = int(height), int(width)
    cells = len(population.centers)
    colours = site_id_colours(cells)
    denominator = np.zeros((height, width), dtype=np.float64)
    square_sum = np.zeros((height, width), dtype=np.float64)
    largest = np.zeros((height, width), dtype=np.float64)
    dominant = np.full((height, width), -1, dtype=np.int32)
    numerator = np.zeros((height, width, 3), dtype=np.float64)
    outlines = np.zeros((height, width, 3), dtype=np.float32)

    for site, ((cx, cy), major, minor, theta) in enumerate(zip(
        population.centers,
        population.major,
        population.minor,
        population.angle,
    )):
        major = max(float(major), 0.75)
        minor = max(float(minor), 0.75)
        ct, st = math.cos(float(theta)), math.sin(float(theta))
        extent_x = math.sqrt((major * ct) ** 2 + (minor * st) ** 2)
        extent_y = math.sqrt((major * st) ** 2 + (minor * ct) ** 2)
        x0 = max(0, int(math.floor(cx - extent_x)))
        x1 = min(width, int(math.ceil(cx + extent_x)) + 1)
        y0 = max(0, int(math.floor(cy - extent_y)))
        y1 = min(height, int(math.ceil(cy + extent_y)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = xx - cx
        dy = yy - cy
        along = dx * ct + dy * st
        across = -dx * st + dy * ct
        q = (along / major) ** 2 + (across / minor) ** 2
        visible = q < 1.0
        if not np.any(visible):
            continue
        phi = np.maximum(1.0 - q, 0.0) ** (
            float(power) * float(temperature))

        denominator_patch = denominator[y0:y1, x0:x1]
        square_patch = square_sum[y0:y1, x0:x1]
        numerator_patch = numerator[y0:y1, x0:x1]
        denominator_patch[visible] += phi[visible]
        square_patch[visible] += phi[visible] * phi[visible]
        numerator_patch[visible] += (
            phi[visible, None] * colours[site][None, :])

        largest_patch = largest[y0:y1, x0:x1]
        dominant_patch = dominant[y0:y1, x0:x1]
        stronger = visible & (phi > largest_patch)
        largest_patch[stronger] = phi[stronger]
        dominant_patch[stronger] = site

        # Draw the final pixel-wide interior rim of the actual compact support.
        rim_width = min(0.9 / min(major, minor), 0.8)
        rim = visible & (np.sqrt(np.maximum(q, 0.0)) >= 1.0 - rim_width)
        outline_patch = outlines[y0:y1, x0:x1]
        outline_patch[rim] = colours[site].astype(np.float32)

    covered = denominator > 1e-30
    soft = np.zeros_like(numerator, dtype=np.float32)
    soft[covered] = (
        numerator[covered] / denominator[covered, None]).astype(np.float32)
    hard = np.zeros_like(soft)
    hard[covered] = colours[dominant[covered]].astype(np.float32)
    dominance = np.zeros((height, width), dtype=np.float32)
    dominance[covered] = (
        largest[covered] / denominator[covered]).astype(np.float32)
    effective = np.zeros((height, width), dtype=np.float32)
    effective[covered] = (
        denominator[covered] ** 2
        / np.maximum(square_sum[covered], 1e-30)).astype(np.float32)

    # Site dots distinguish very small ellipses whose rims collapse to a pixel.
    for cx, cy in population.centers:
        xi = int(np.clip(round(cx), 0, width - 1))
        yi = int(np.clip(round(cy), 0, height - 1))
        outlines[yi, xi] = 1.0

    return {
        "soft_site_ids": soft,
        "dominant_site_ids": hard,
        "cell_outlines": outlines,
        "dominance": dominance,
        "effective_contributors": effective,
        "covered": covered,
    }


def population_from_measure(
    result: dict,
    quantizer: str,
    overlap: float,
) -> tuple[Population, dict]:
    density = np.asarray(result["envelope"][-1], dtype=np.float64)
    height, width = density.shape
    if quantizer == "rosenblatt":
        centers = quantize_rosenblatt(density)
        quantizer_report = {}
    elif quantizer == "phase":
        centers, quantizer_report = quantize_phase(density)
    elif quantizer == "poisson":
        centers, quantizer_report = quantize_poisson_map(density)
    elif quantizer == "uniform":
        count = max(1, int(round(float(np.sum(density)))))
        centers = _r2_sites(count, width, height)
        quantizer_report = {}
    else:
        raise ValueError(f"unknown quantizer {quantizer!r}")

    qxx = _sample(result["precision_xx"], centers)
    qxy = _sample(result["precision_xy"], centers)
    qyy = _sample(result["precision_yy"], centers)
    trace = qxx + qyy
    disc = np.hypot(qxx - qyy, 2.0 * qxy)
    high = np.maximum(0.5 * (trace + disc), 1e-12)
    low = np.maximum(0.5 * (trace - disc), 1e-12)
    ratio = np.sqrt(high / low)
    sampled_density = np.maximum(_sample(density, centers), 1e-12)
    area = float(overlap) / sampled_density
    major = np.sqrt(area * ratio / math.pi)
    minor = np.sqrt(area / (math.pi * ratio))
    normal_angle = 0.5 * np.arctan2(2.0 * qxy, qxx - qyy)
    tangent_angle = normal_angle + 0.5 * math.pi
    population = Population(
        centers=centers,
        major=major,
        minor=minor,
        angle=tangent_angle,
        stage=np.zeros(len(centers), dtype=np.int16),
        base_cells=0,
    )
    return population, {
        "quantizer": quantizer,
        "cells": int(len(centers)),
        "measure_integral": float(np.sum(density)),
        "overlap": float(overlap),
        "sampled_density_p10_p50_p90": [
            float(value)
            for value in np.percentile(sampled_density, (10, 50, 90))
        ],
        "major_p10_p50_p90": [
            float(value) for value in np.percentile(major, (10, 50, 90))
        ],
        "minor_p10_p50_p90": [
            float(value) for value in np.percentile(minor, (10, 50, 90))
        ],
        "ratio_p10_p50_p90": [
            float(value) for value in np.percentile(ratio, (10, 50, 90))
        ],
        **quantizer_report,
    }


def _serializable(record: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in record.items()
        if key != "rgb"
    }


def fit_population_local(
    population: Population,
    target_lab: np.ndarray,
    objective: SingleStageDecompositionObjective | None = None,
) -> tuple[dict, np.ndarray, dict]:
    """Independent affine jets plus owner-free partition-of-unity render.

    This is the HD finish: its memory and arithmetic are linear in compact
    support samples.  It is also a useful control for how much the exact
    global coupling contributes at research resolution.
    """
    height, width = target_lab.shape[:2]
    samples = support_samples(population, height, width)
    cells = len(population.centers)
    basis = np.asarray(samples["basis"], dtype=np.float64)
    phi = np.asarray(samples["phi"], dtype=np.float64)
    sites = np.asarray(samples["sites"], dtype=np.intp)
    rows = np.asarray(samples["rows"], dtype=np.intp)
    target = np.asarray(target_lab, dtype=np.float64).reshape(-1, 3)

    normal = np.zeros((cells, 3, 3), dtype=np.float64)
    rhs = np.zeros((cells, 3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            np.add.at(normal[:, a, b], sites, phi * basis[:, a] * basis[:, b])
        for channel in range(3):
            np.add.at(
                rhs[:, a, channel],
                sites,
                phi * basis[:, a] * target[rows, channel],
            )
    scale = np.maximum(np.trace(normal, axis1=1, axis2=2), 1.0)
    normal[:, 0, 0] += 1e-5 * scale
    normal[:, 1, 1] += 2e-3 * scale
    normal[:, 2, 2] += 2e-3 * scale
    coeff = np.linalg.solve(normal, rhs)

    prediction = np.einsum(
        "ni,nic->nc", basis, coeff[sites], optimize=False)
    denominator = np.bincount(
        rows, weights=phi, minlength=height * width)
    rendered = np.empty((height * width, 3), dtype=np.float64)
    for channel in range(3):
        rendered[:, channel] = np.bincount(
            rows,
            weights=phi * prediction[:, channel],
            minlength=height * width,
        )
    background = np.mean(target, axis=0)
    uncovered = denominator <= 1e-30
    rendered[~uncovered] /= denominator[~uncovered, None]
    rendered[uncovered] = background
    reconstruction = rendered.reshape(target_lab.shape)
    if objective is None:
        record = {"rgb": np.clip(
            __import__("bfft").lab_to_srgb(reconstruction), 0.0, 1.0)}
    else:
        from flow_volume_cells import score

        record = score(objective, objective.target_rgb, reconstruction)
    diagnostic = {
        "samples": int(len(rows)),
        "uncovered_fraction": float(np.mean(uncovered)),
        "solver": "local affine",
    }
    return record, reconstruction, diagnostic


def fit_population_cg(
    population: Population,
    target_lab: np.ndarray,
    objective: SingleStageDecompositionObjective | None = None,
    iterations: int = 80,
    sparse_sample_limit: int = 2_000_000,
    temperature: float = 1.0,
) -> tuple[dict, np.ndarray, dict]:
    """Matrix-free preconditioned global fit for large images."""
    from dual_aperture_support import aperture, design_matrix

    height, width = target_lab.shape[:2]
    samples = support_samples(population, height, width)
    weight, dominance, effective = aperture(
        samples, height * width, float(temperature))
    rows = np.asarray(samples["rows"], dtype=np.intp)
    sites = np.asarray(samples["sites"], dtype=np.intp)
    basis_x = np.asarray(samples["basis"][:, 1], dtype=np.float64)
    basis_y = np.asarray(samples["basis"][:, 2], dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    pixels = height * width
    cells = len(population.centers)
    target = np.asarray(target_lab, dtype=np.float64).reshape(-1, 3)
    regularization = np.tile(
        np.array([1e-5, 2e-3, 2e-3], dtype=np.float64),
        cells,
    )

    sparse_backend = len(rows) <= int(sparse_sample_limit)
    if sparse_backend:
        design = design_matrix(samples, pixels, weight)

        def apply_design(vector):
            return design @ vector

        def apply_transpose(pixel_field):
            return design.T @ pixel_field

        weight_square = None
    else:
        native_operators = compact_support_operators(
            rows, sites, weight, basis_x, basis_y, pixels, cells)
        if native_operators is not None:
            apply_design, apply_transpose, apply_normal = native_operators
        else:
            apply_normal = None

            def apply_design(vector):
                coefficient = vector.reshape(cells, 3)
                prediction = (
                    coefficient[sites, 0]
                    + basis_x * coefficient[sites, 1]
                    + basis_y * coefficient[sites, 2])
                return np.bincount(
                    rows, weights=weight * prediction, minlength=pixels)

            def apply_transpose(pixel_field):
                sampled = weight * pixel_field[rows]
                result = np.empty((cells, 3), dtype=np.float64)
                result[:, 0] = np.bincount(
                    sites, weights=sampled, minlength=cells)
                result[:, 1] = np.bincount(
                    sites, weights=sampled * basis_x, minlength=cells)
                result[:, 2] = np.bincount(
                    sites, weights=sampled * basis_y, minlength=cells)
                return result.ravel()

        weight_square = weight * weight

    rhs = np.column_stack([
        apply_transpose(target[:, channel])
        for channel in range(3)
    ])

    def normal_product(vector):
        if not sparse_backend and apply_normal is not None:
            product = apply_normal(vector)
        else:
            product = apply_transpose(apply_design(vector))
        return product + regularization * vector

    normal = LinearOperator(
        (3 * cells, 3 * cells),
        matvec=normal_product,
        dtype=np.float64,
    )
    if sparse_backend:
        diagonal = np.asarray(
            design.power(2).sum(axis=0)).ravel() + regularization
    else:
        diagonal = np.column_stack([
            np.bincount(sites, weights=weight_square, minlength=cells),
            np.bincount(
                sites, weights=weight_square * basis_x * basis_x,
                minlength=cells),
            np.bincount(
                sites, weights=weight_square * basis_y * basis_y,
                minlength=cells),
        ]).ravel() + regularization
    preconditioner = LinearOperator(
        normal.shape,
        matvec=lambda vector: vector / diagonal,
        dtype=np.float64,
    )
    coefficient = np.column_stack([
        cg(
            normal,
            rhs[:, channel],
            M=preconditioner,
            rtol=0.0,
            atol=0.0,
            maxiter=max(int(iterations), 1),
        )[0]
        for channel in range(3)
    ])
    reconstruction = np.column_stack([
        apply_design(coefficient[:, channel])
        for channel in range(3)
    ]).reshape(target_lab.shape)
    if objective is None:
        record = {"rgb": np.clip(
            __import__("bfft").lab_to_srgb(reconstruction), 0.0, 1.0)}
    else:
        from flow_volume_cells import score

        record = score(objective, objective.target_rgb, reconstruction)
    covered = np.bincount(rows, minlength=pixels) > 0
    diagnostic = {
        "samples": int(len(samples["rows"])),
        "uncovered_fraction": float(np.mean(~covered)),
        "dominance_mean": float(np.mean(dominance[covered])),
        "effective_contributors_median": float(
            np.median(effective[covered])),
        "solver": "matrix-free preconditioned CG",
        "operator_backend": (
            "sparse design"
            if sparse_backend
            else (
                "native direct support scatter"
                if native_operators is not None
                else "NumPy direct support scatter")),
        "iterations": int(iterations),
        "partition_temperature": float(temperature),
    }
    return record, reconstruction, diagnostic


def save_panel(
    rgb: np.ndarray,
    density: np.ndarray,
    variants: list[tuple[str, Population, dict, dict]],
    output: Path,
) -> None:
    columns = 1 + len(variants)
    fig, axes = plt.subplots(3, columns, figsize=(4 * columns, 11.5))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("target")
    axes[1, 0].imshow(density, cmap="viridis")
    axes[1, 0].set_title(
        f"transport support measure\nintegral {np.sum(density):.1f}")
    axes[2, 0].axis("off")
    for column, (name, population, record, diagnostic) in enumerate(
        variants, start=1
    ):
        axes[0, column].imshow(record["rgb"])
        axes[0, column].set_title(
            f"{name}\n{record['psnr']:.2f} dB  "
            f"obj {record['objective']:.4g}")
        axes[1, column].imshow(rgb, alpha=0.22)
        axes[1, column].scatter(
            population.centers[:, 0],
            population.centers[:, 1],
            s=np.clip(population.major * population.minor, 1.0, 18.0),
            c=np.log1p(population.major / np.maximum(
                population.minor, 1e-12)),
            cmap="turbo",
            alpha=0.62,
            linewidths=0,
        )
        uncovered = float(diagnostic.get("uncovered_fraction", 0.0))
        axes[1, column].set_title(
            f"{len(population.centers)} supports; "
            f"uncovered {100*uncovered:.2f}%")
        error = np.sqrt(np.mean((rgb - record["rgb"]) ** 2, axis=2))
        axes[2, column].imshow(
            error,
            cmap="inferno",
            vmin=0.0,
            vmax=max(float(np.percentile(error, 99.5)), 1e-12),
        )
        axes[2, column].set_title("RGB error")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--passes", type=int, default=24)
    parser.add_argument(
        "--flow-sweeps",
        type=int,
        default=4,
        help="auxiliary defect projection sweeps; four is the validated default",
    )
    parser.add_argument("--max-support-fraction", type=float, default=0.18)
    parser.add_argument(
        "--quantizers",
        nargs="+",
        choices=(
            "phase-stack", "phase-stack-static",
            "phase", "rosenblatt", "poisson", "uniform"),
        default=(
            "phase-stack", "phase-stack-static",
            "phase", "rosenblatt", "poisson", "uniform"),
    )
    parser.add_argument(
        "--overlaps", type=float, nargs="+", default=(1.5, 2.0, 3.0))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/out/transport_measure_cells.png",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "experiments/out/transport_measure_cells.json",
    )
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    rgb = _fit_rgb(image, args.side)
    started = time.perf_counter()
    volume = build_flow_volume(
        rgb, passes=args.passes, flow_sweeps=args.flow_sweeps)
    support = infer_support_measure(
        volume, max_support_fraction=args.max_support_fraction)
    geometry_ms = (time.perf_counter() - started) * 1000.0
    objective = SingleStageDecompositionObjective(rgb)
    target_lab = srgb_to_lab(rgb)

    candidates = []
    all_records = {}
    solve_started = time.perf_counter()
    for quantizer in args.quantizers:
        for overlap in args.overlaps:
            if quantizer in ("phase-stack", "phase-stack-static"):
                population, population_report = population_from_scale_measure(
                    support,
                    volume,
                    overlap,
                    advect=quantizer == "phase-stack",
                )
            else:
                population, population_report = population_from_measure(
                    support, quantizer, overlap)
            record, _, diagnostic = fit_population(
                population, target_lab, objective)
            name = f"{quantizer} overlap={overlap:g}"
            candidates.append((name, population, record, diagnostic))
            all_records[name] = {
                **_serializable(record),
                "population": population_report,
                "diagnostic": diagnostic,
            }
            if quantizer == "phase-stack-static":
                layered_record, _, layered_diagnostic = (
                    fit_layered_population(population, rgb, objective))
                layered_name = (
                    f"round cartoon + transport texture overlap={overlap:g}")
                candidates.append((
                    layered_name,
                    population,
                    layered_record,
                    layered_diagnostic,
                ))
                all_records[layered_name] = {
                    **_serializable(layered_record),
                    "population": {
                        **population_report,
                        "quantizer": "phase stack, dual aperture",
                    },
                    "diagnostic": layered_diagnostic,
                }
                circular = population.circularized()
                circular_record, _, circular_diagnostic = fit_population(
                    circular, target_lab, objective)
                circular_name = (
                    f"same centers circular overlap={overlap:g}")
                candidates.append((
                    circular_name,
                    circular,
                    circular_record,
                    circular_diagnostic,
                ))
                all_records[circular_name] = {
                    **_serializable(circular_record),
                    "population": {
                        **population_report,
                        "quantizer": "phase stack, circularized",
                    },
                    "diagnostic": circular_diagnostic,
                }

    count = max(1, int(round(support["transported_count"])))
    for overlap in args.overlaps:
        control = r2_control(count, rgb.shape[0], rgb.shape[1], overlap)
        record, _, diagnostic = fit_population(
            control, target_lab, objective)
        name = f"uniform circular overlap={overlap:g}"
        candidates.append((name, control, record, diagnostic))
        all_records[name] = {
            **_serializable(record),
            "population": {
                "quantizer": "uniform circular",
                "cells": count,
                "measure_integral": support["transported_count"],
                "overlap": overlap,
            },
            "diagnostic": diagnostic,
        }
    solve_ms = (time.perf_counter() - solve_started) * 1000.0
    candidates.sort(key=lambda item: item[2]["objective"])
    shown = candidates[:4]
    save_panel(
        rgb,
        np.asarray(support["envelope"][-1]),
        shown,
        args.output,
    )
    report = {
        "source": source,
        "shape": list(rgb.shape),
        "passes": args.passes,
        "flow_sweeps": args.flow_sweeps,
        "max_support_fraction": args.max_support_fraction,
        "support": {
            "static_count": support["static_count"],
            "transported_count": support["transported_count"],
            "broad_horizon_count": support["broad_horizon_count"],
        },
        "best": shown[0][0],
        "scores": all_records,
        "geometry_ms": geometry_ms,
        "solve_ms": solve_ms,
        "output": str(args.output.resolve()),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
