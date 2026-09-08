#!/usr/bin/env python3
"""Locate high-frequency loss in CONV raster and warp transport.

The audit separates two questions that are otherwise easily conflated:

1. Nodal semantics.  A procedural field is sampled at endpoint-aligned nodes,
   reconstructed by successive CONV atlas stages, and averaged over target
   Voronoi basins.  This isolates proposal, support-range, joint-admission,
   and final-aperture losses under the paper's present direct-warp model.
2. Raster semantics.  Source entries are exact uniform pixel-cell averages.
   The present nodal-atlas/pixel-average composition is compared with a
   conservative cell-average transport obtained by zero-detail CONV moment
   refinement followed by exact affine overlap.  Identity must reproduce the
   source cell averages exactly in the latter representation.

No competitor output is used as truth.  Every reference is an analytic
Fourier value or Fourier cell integral.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.conv_conservative_multiresolution import (
    conv_moments_2d,
    face_sign_certificate_2d,
    synthesize_block_moments_2d,
)
from experiments.conv_warp.geometry import ProjectiveMap, affine_about_center
from experiments.conv_warp.joint_reference import (
    _gradient_cone_constraints,
    _global_lattice_from_atlas,
    _q1_global_lattice,
    _shared_support_bounds,
    canonical_sampled_control_net,
    finite_joint_control_nets,
    raw_tensor_control_net,
)
from experiments.conv_distilled_core import nodal_current_geometry
from experiments.convstar import convstar_resize
from experiments.conv_warp.synthetic import FourierField
from experiments.conv_warp.warped_pixels import (
    _clip_rectangle,
    _signed_polygon_area,
    exact_affine_warped_pixel_average,
)
from standalone_conv_resize_demo.conservative import zero_detail_synthesis


Array = np.ndarray


def _uniform_cell_average(
    shape: tuple[int, int], wave: tuple[float, float], phase: float
) -> Array:
    """Exact averages of ``cos(2 pi wave.x + phase)`` on uniform pixels."""

    height, width = map(int, shape)
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    yy, xx = np.meshgrid(y, x, indexing="ij")
    kx, ky = map(float, wave)
    aperture = np.sinc(kx / width) * np.sinc(ky / height)
    return aperture * np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phase)


def _warped_uniform_cell_average(
    transform: ProjectiveMap,
    shape: tuple[int, int],
    wave: tuple[float, float],
    phase: float,
) -> Array:
    """Exact affine target-pixel averages of one source Fourier mode."""

    linear, offset = transform.affine_parts()
    height, width = map(int, shape)
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    yy, xx = np.meshgrid(y, x, indexing="ij")
    source_x = linear[0, 0] * xx + linear[0, 1] * yy + offset[0]
    source_y = linear[1, 0] * xx + linear[1, 1] * yy + offset[1]
    transformed_wave = np.asarray(wave, dtype=np.float64) @ linear
    aperture = (
        np.sinc(transformed_wave[0] / width)
        * np.sinc(transformed_wave[1] / height)
    )
    return aperture * np.cos(
        2.0 * np.pi * (wave[0] * source_x + wave[1] * source_y) + phase
    )


def _cell_validity_mask(
    transform: ProjectiveMap, shape: tuple[int, int]
) -> Array:
    """Target pixels whose complete affine image lies inside the source."""

    height, width = map(int, shape)
    result = np.ones((height, width), dtype=bool)
    for row in range(height):
        y0, y1 = row / height, (row + 1) / height
        for column in range(width):
            x0, x1 = column / width, (column + 1) / width
            x, y = transform.map(
                np.array((x0, x1, x1, x0)),
                np.array((y0, y0, y1, y1)),
            )
            result[row, column] = bool(
                np.min(x) >= 0.0 and np.max(x) <= 1.0
                and np.min(y) >= 0.0 and np.max(y) <= 1.0
            )
    return result


def _exact_affine_cell_overlap(
    values: Array, transform: ProjectiveMap, target_shape: tuple[int, int]
) -> Array:
    """Push uniform source-cell averages through an affine map exactly.

    ``values`` defines a piecewise-constant density on equal source cells.
    For an affine inverse map, target averaging is therefore the area-weighted
    overlap of one source parallelogram with those cells.  This is the exact
    positive raster functional for that represented density.
    """

    if not transform.is_affine:
        raise ValueError("the exact overlap audit currently requires an affine map")
    source = np.asarray(values, dtype=np.float64)
    scalar = source.ndim == 2
    if scalar:
        source = source[..., None]
    if source.ndim != 3:
        raise ValueError("values must be HxW or HxWxC")
    target_height, target_width = map(int, target_shape)
    source_height, source_width, channels = source.shape
    output = np.full((target_height, target_width, channels), np.nan)
    for row in range(target_height):
        y0, y1 = row / target_height, (row + 1) / target_height
        for column in range(target_width):
            x0, x1 = column / target_width, (column + 1) / target_width
            px, py = transform.map(
                np.array((x0, x1, x1, x0)),
                np.array((y0, y0, y1, y1)),
            )
            polygon = np.stack((source_width * px, source_height * py), axis=1)
            area = abs(_signed_polygon_area(polygon))
            if area <= 64.0 * np.finfo(np.float64).eps:
                continue
            xmin, xmax = float(np.min(polygon[:, 0])), float(np.max(polygon[:, 0]))
            ymin, ymax = float(np.min(polygon[:, 1])), float(np.max(polygon[:, 1]))
            if xmin < 0.0 or ymin < 0.0 or xmax > source_width or ymax > source_height:
                continue
            total = np.zeros(channels, dtype=np.float64)
            for sy in range(max(0, int(math.floor(ymin))), min(source_height - 1, int(math.floor(ymax))) + 1):
                for sx in range(max(0, int(math.floor(xmin))), min(source_width - 1, int(math.floor(xmax))) + 1):
                    clipped = _clip_rectangle(polygon, sx, sx + 1.0, sy, sy + 1.0)
                    overlap = abs(_signed_polygon_area(clipped))
                    if overlap > 256.0 * np.finfo(np.float64).eps:
                        total += overlap * source[sy, sx]
            output[row, column] = total / area
    return output[..., 0] if scalar else output


def _atlas_from_lattice(lattice: Array, shape: tuple[int, int]) -> Array:
    height, width = shape
    value = np.asarray(lattice, dtype=np.float64)
    scalar = value.ndim == 2
    if scalar:
        value = value[..., None]
    atlas = np.empty((height - 1, width - 1, 6, 6, value.shape[-1]))
    for row in range(height - 1):
        for column in range(width - 1):
            atlas[row, column] = value[
                5 * row:5 * row + 6, 5 * column:5 * column + 6
            ]
    return atlas[..., 0] if scalar else atlas


def _q1_integer_refine(values: Array, scale: int) -> Array:
    """Bilinearly refine an endpoint field by an integer factor."""

    source = np.asarray(values, dtype=np.float64)
    height, width = source.shape
    out_height = scale * (height - 1) + 1
    out_width = scale * (width - 1) + 1
    x = np.arange(out_width, dtype=np.float64) / scale
    y = np.arange(out_height, dtype=np.float64) / scale
    x0 = np.minimum(np.floor(x).astype(np.intp), width - 2)
    y0 = np.minimum(np.floor(y).astype(np.intp), height - 2)
    ux = x - x0
    uy = y - y0
    horizontal = (
        (1.0 - ux)[None, :] * source[:, x0]
        + ux[None, :] * source[:, x0 + 1]
    )
    return (
        (1.0 - uy)[:, None] * horizontal[y0]
        + uy[:, None] * horizontal[y0 + 1]
    )


def _canonical_sampled_control_net_f64(values: Array) -> Array:
    """Double-precision realization of the sampled joint proposal."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("the precision audit currently expects one channel")
    scale = 5
    forward = convstar_resize(source, scale=scale)
    reverse = np.swapaxes(
        convstar_resize(np.swapaxes(source, 0, 1), scale=scale), 0, 1
    )
    beta = nodal_current_geometry(source)[3]
    beta_fine = _q1_integer_refine(beta, scale)
    sampled = (1.0 - beta_fine) * forward + beta_fine * reverse
    phase = np.linspace(0.0, 1.0, 6)
    collocation = np.array([
        [math.comb(5, k) * u**k * (1.0 - u)**(5 - k) for k in range(6)]
        for u in phase
    ])
    inverse = np.linalg.inv(collocation)
    height, width = source.shape
    output = np.empty((height - 1, width - 1, 6, 6))
    for row in range(height - 1):
        for column in range(width - 1):
            local = sampled[
                5 * row:5 * row + 6, 5 * column:5 * column + 6
            ]
            output[row, column] = np.einsum(
                "ai,ij,bj->ab", inverse, local, inverse, optimize=True
            )
    return output


def _minimum_support_gradient_margin(source: Array, atlas: Array) -> float:
    minimum = float("inf")
    for row in range(source.shape[0] - 1):
        for column in range(source.shape[1] - 1):
            for coefficient in _gradient_cone_constraints(
                source[..., None], row, column, 0, support=True
            ):
                minimum = min(
                    minimum,
                    float(np.sum(coefficient * atlas[row, column])),
                )
    return 0.0 if minimum == float("inf") else minimum


def _metrics(result: Array, truth: Array, mask: Array) -> dict[str, float]:
    value = np.asarray(result, dtype=np.float64)[mask]
    reference = np.asarray(truth, dtype=np.float64)[mask]
    error = value - reference
    centered = reference - np.mean(reference)
    denominator = float(centered @ centered)
    gain = float((value - np.mean(value)) @ centered / denominator) if denominator else 1.0
    return {
        "mse": float(np.mean(error * error)),
        "maximum_absolute_error": float(np.max(np.abs(error), initial=0.0)),
        "signed_mode_gain": gain,
        "range_excursion": max(
            float(np.min(reference) - np.min(value)),
            float(np.max(value) - np.max(reference)),
            0.0,
        ),
    }


def _nodal_stage_record(
    shape: tuple[int, int], wave: tuple[float, float], phase: float
) -> dict[str, object]:
    field = FourierField(
        np.array((wave,), dtype=np.float64),
        np.array((1.0,), dtype=np.float64),
        np.array((phase,), dtype=np.float64),
        0.0,
    )
    source = field.sample_nodes(shape)
    transform = ProjectiveMap(np.eye(3))
    truth = field.affine_basin_average(transform, shape)
    raw = raw_tensor_control_net(source)
    canonical = canonical_sampled_control_net(source)
    canonical_f64 = _canonical_sampled_control_net_f64(source)
    lattice = _global_lattice_from_atlas(canonical)[..., 0]
    lattice[::5, ::5] = source
    lower, upper = _shared_support_bounds(source)
    lower = lower[..., 0]
    upper = upper[..., 0]
    clipped_lattice = np.clip(lattice, lower, upper)
    clipped = _atlas_from_lattice(clipped_lattice, shape)
    lattice_f64 = _global_lattice_from_atlas(canonical_f64)[..., 0]
    lattice_f64[::5, ::5] = source
    clipped_f64 = _atlas_from_lattice(
        np.clip(lattice_f64, lower, upper), shape
    )
    joint, diagnostic = finite_joint_control_nets(source)
    stages = {
        "raw_tensor": raw,
        "canonical_sampled": canonical,
        "support_range": clipped,
        "support_range_f64": clipped_f64,
        "joint_admitted": joint,
    }
    output: dict[str, object] = {
        "admission": diagnostic,
        "minimum_support_gradient_margin": {
            "float32_sampled": _minimum_support_gradient_margin(source, clipped),
            "float64_sampled": _minimum_support_gradient_margin(source, clipped_f64),
        },
        "stages": {},
    }
    mask = np.ones(shape, dtype=bool)
    for name, atlas in stages.items():
        value, _ = exact_affine_warped_pixel_average(
            source, transform, shape, control=atlas
        )
        output["stages"][name] = _metrics(value, truth, mask)
    return output


def _raster_record(
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    wave: tuple[float, float],
    phase: float,
    transform: ProjectiveMap,
    levels: int,
) -> dict[str, object]:
    source = _uniform_cell_average(source_shape, wave, phase)
    truth = _warped_uniform_cell_average(transform, target_shape, wave, phase)
    mask = _cell_validity_mask(transform, target_shape)
    nodal, _ = exact_affine_warped_pixel_average(
        source, transform, target_shape
    )
    methods: dict[str, Array] = {
        "present_nodal_atlas_then_pixel_average": nodal,
        "source_cells_piecewise_constant": _exact_affine_cell_overlap(
            source, transform, target_shape
        ),
    }
    refined = source
    native_refined = source.astype(np.float32)
    conservation: list[float] = []
    native_conservation: list[float] = []
    native_formal_difference: list[float] = []
    topology: list[dict[str, int]] = []
    refinement_range_excursion: list[float] = []
    for level in range(1, levels + 1):
        moments = conv_moments_2d(refined)
        topology.append(face_sign_certificate_2d(refined, moments))
        refined = synthesize_block_moments_2d(moments)
        native_refined = zero_detail_synthesis(native_refined)
        native_formal_difference.append(float(
            np.max(np.abs(native_refined.astype(np.float64) - refined))
        ))
        recovered = refined.reshape(
            source.shape[0], 2**level, source.shape[1], 2**level
        ).mean(axis=(1, 3))
        native_recovered = native_refined.reshape(
            source.shape[0], 2**level, source.shape[1], 2**level
        ).mean(axis=(1, 3), dtype=np.float64)
        conservation.append(float(np.max(np.abs(recovered - source))))
        native_conservation.append(float(
            np.max(np.abs(native_recovered - source))
        ))
        refinement_range_excursion.append(max(
            float(np.min(source) - np.min(refined)),
            float(np.max(refined) - np.max(source)),
            0.0,
        ))
        methods[f"conservative_moment_refinement_{2**level}x"] = (
            _exact_affine_cell_overlap(refined, transform, target_shape)
        )
        methods[f"native_fused_moment_atlas_{2**level}x"] = (
            _exact_affine_cell_overlap(
                native_refined, transform, target_shape
            )
        )
    return {
        "valid_pixels": int(np.sum(mask)),
        "conservation_residual_by_level": conservation,
        "native_conservation_residual_by_level": native_conservation,
        "native_formal_maximum_difference_by_level": native_formal_difference,
        "topology_by_level": topology,
        "refinement_source_range_excursion_by_level": refinement_range_excursion,
        "methods": {
            name: _metrics(value, truth, mask & np.isfinite(value))
            for name, value in methods.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=13)
    parser.add_argument("--raster-source-size", type=int, default=25)
    parser.add_argument("--raster-target-size", type=int, default=13)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument(
        "--out", type=Path,
        default=Path("/tmp/conv_high_frequency_transport_audit.json"),
    )
    args = parser.parse_args()
    shape = (args.size, args.size)
    raster_source_shape = (
        args.raster_source_size, args.raster_source_size
    )
    raster_target_shape = (
        args.raster_target_size, args.raster_target_size
    )
    modes = ((2.0, 1.0), (4.0, 0.0), (4.0, 3.0), (5.0, 4.0))
    transforms = {
        "identity": ProjectiveMap(np.eye(3)),
        "affine_17deg": affine_about_center(
            angle_degrees=17.0, scale_x=0.76, scale_y=0.72, shear_x=0.08
        ),
    }
    result: dict[str, object] = {
        "semantics": {
            "nodal": "analytic endpoint-node samples and endpoint Voronoi truth",
            "raster": "analytic uniform source/target pixel-cell averages",
        },
        "shape": list(shape),
        "raster_source_shape": list(raster_source_shape),
        "raster_target_shape": list(raster_target_shape),
        "nodal_stage_census": {},
        "raster_transport_census": {},
    }
    for wave in modes:
        key = f"kx={wave[0]:g},ky={wave[1]:g}"
        result["nodal_stage_census"][key] = _nodal_stage_record(
            shape, wave, phase=0.37
        )
        result["raster_transport_census"][key] = {
            name: _raster_record(
                raster_source_shape, raster_target_shape, wave, 0.37,
                transform, max(0, int(args.levels))
            )
            for name, transform in transforms.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
