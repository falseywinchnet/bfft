"""Two-times single-image super-resolution from the v3 support basis.

The benchmark is honest by construction: an original image is cropped to an
even lattice and Lanczos-reduced by exactly two.  Every method receives only
that reduced observation.  The original is retained solely for scoring.

The v3 method lifts the fitted continuous design, not just its raster:

* affine cell coordinates are evaluated exactly on the dense lattice;
* graph-phase and paired one-sided columns are Lanczos-lifted;
* fixed per-cell coefficients are recovered from the low-resolution fit;
* uncertain high-frequency basis energy receives Gaussian posterior shrinkage;
* two fixed owner-respecting back-projections enforce the observation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfft.effects import lab_to_srgb
from experiments.segmenting_v3 import SegmentingV3Config, build_segmenting_v3
from port_needed.fast_image_ops import gaussian_filter


DEFAULT_IMAGE = (
    Path.home() / "Downloads"
    / "487056612_18488677060002753_2143459132310970747_n.jpg"
)


def _identity(function):  # pragma: no cover
    return function


_compile_parallel = (
    njit(cache=True, parallel=True, fastmath=False)
    if njit is not None else _identity
)


@_compile_parallel
def _eikonal_lanczos2_kernel(
    image,
    labels,
    tensor_xx,
    tensor_xy,
    tensor_yy,
    anisotropy,
    clamp_range,
):
    """Fixed local eikonal-chart tensor product of Lanczos-3 kernels."""
    height, width, channels = image.shape
    output = np.empty((2 * height, 2 * width, channels), dtype=np.float64)
    pi = math.pi
    for output_y in prange(2 * height):
        source_y = (output_y + 0.5) * 0.5 - 0.5
        center_y = int(math.floor(source_y + 0.5))
        center_y = min(max(center_y, 0), height - 1)
        base_y = int(math.floor(source_y)) - 4
        for output_x in range(2 * width):
            source_x = (output_x + 0.5) * 0.5 - 0.5
            center_x = int(math.floor(source_x + 0.5))
            center_x = min(max(center_x, 0), width - 1)
            base_x = int(math.floor(source_x)) - 4
            owner = labels[center_y, center_x]

            xx = tensor_xx[center_y, center_x]
            xy = tensor_xy[center_y, center_x]
            yy = tensor_yy[center_y, center_x]
            trace = max(xx + yy, 1e-15)
            difference = xx - yy
            discriminant = math.sqrt(
                max(difference * difference + 4.0 * xy * xy, 0.0))
            coherence = min(max(discriminant / trace, 0.0), 1.0)
            angle = (
                0.5 * math.atan2(2.0 * xy, difference)
                if coherence > 0.05 else 0.0
            )
            normal_x = math.cos(angle)
            normal_y = math.sin(angle)
            tangent_x = -normal_y
            tangent_y = normal_x
            stretch = 1.0 + anisotropy * coherence
            normal_scale = 1.0 / stretch
            tangent_scale = stretch

            total_weight = 0.0
            accum0 = 0.0
            accum1 = 0.0
            accum2 = 0.0
            minimum0 = 1e300
            minimum1 = 1e300
            minimum2 = 1e300
            maximum0 = -1e300
            maximum1 = -1e300
            maximum2 = -1e300
            for local_y in range(10):
                sample_y = min(max(base_y + local_y, 0), height - 1)
                dy = sample_y - source_y
                for local_x in range(10):
                    sample_x = min(max(base_x + local_x, 0), width - 1)
                    if labels[sample_y, sample_x] != owner:
                        continue
                    dx = sample_x - source_x
                    normal_distance = (
                        dx * normal_x + dy * normal_y) / normal_scale
                    tangent_distance = (
                        dx * tangent_x + dy * tangent_y) / tangent_scale
                    if (
                        abs(normal_distance) >= 3.0
                        or abs(tangent_distance) >= 3.0
                    ):
                        continue
                    if abs(normal_distance) < 1e-12:
                        normal_weight = 1.0
                    else:
                        normal_weight = (
                            math.sin(pi * normal_distance)
                            / (pi * normal_distance)
                            * math.sin(pi * normal_distance / 3.0)
                            / (pi * normal_distance / 3.0)
                        )
                    if abs(tangent_distance) < 1e-12:
                        tangent_weight = 1.0
                    else:
                        tangent_weight = (
                            math.sin(pi * tangent_distance)
                            / (pi * tangent_distance)
                            * math.sin(pi * tangent_distance / 3.0)
                            / (pi * tangent_distance / 3.0)
                        )
                    weight = normal_weight * tangent_weight
                    total_weight += weight
                    value0 = image[sample_y, sample_x, 0]
                    value1 = image[sample_y, sample_x, 1]
                    value2 = image[sample_y, sample_x, 2]
                    accum0 += weight * value0
                    accum1 += weight * value1
                    accum2 += weight * value2
                    minimum0 = min(minimum0, value0)
                    minimum1 = min(minimum1, value1)
                    minimum2 = min(minimum2, value2)
                    maximum0 = max(maximum0, value0)
                    maximum1 = max(maximum1, value1)
                    maximum2 = max(maximum2, value2)
            if abs(total_weight) < 1e-12:
                output[output_y, output_x] = image[center_y, center_x]
            else:
                value0 = accum0 / total_weight
                value1 = accum1 / total_weight
                value2 = accum2 / total_weight
                if clamp_range:
                    value0 = min(max(value0, minimum0), maximum0)
                    value1 = min(max(value1, minimum1), maximum1)
                    value2 = min(max(value2, minimum2), maximum2)
                output[output_y, output_x, 0] = value0
                output[output_y, output_x, 1] = value1
                output[output_y, output_x, 2] = value2
    return output


def eikonal_lanczos2(observed, result, anisotropy=0.75, clamp_range=True):
    """V3-metric radial Lanczos with structural geodesic barriers."""
    geometry = result["texture_geometry"]
    if geometry is None:
        geometry = result["cartoon_geometry"]
    return np.clip(_eikonal_lanczos2_kernel(
        np.ascontiguousarray(observed, dtype=np.float64),
        np.ascontiguousarray(result["labels"], dtype=np.int32),
        np.ascontiguousarray(geometry["boundary_xx"], dtype=np.float64),
        np.ascontiguousarray(geometry["boundary_xy"], dtype=np.float64),
        np.ascontiguousarray(geometry["boundary_yy"], dtype=np.float64),
        float(anisotropy),
        bool(clamp_range),
    ), 0.0, 1.0)


def _rgb(image):
    value = np.asarray(image, dtype=np.float64)[..., :3]
    if float(np.max(value, initial=0.0)) > 1.5:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def _resize(image, shape, resample):
    """Float-preserving Pillow resize, channel by channel."""
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        value = value[..., None]
    height, width = map(int, shape)
    output = np.empty((height, width, value.shape[2]), dtype=np.float64)
    for channel in range(value.shape[2]):
        plane = Image.fromarray(
            np.asarray(value[..., channel], dtype=np.float32), mode="F")
        output[..., channel] = np.asarray(
            plane.resize((width, height), resample=resample),
            dtype=np.float64,
        )
    return output[..., 0] if np.asarray(image).ndim == 2 else output


def _downsample2(image):
    height, width = image.shape[:2]
    return _resize(
        image, (height // 2, width // 2), Image.Resampling.LANCZOS)


def _upsample2(image, resample):
    height, width = image.shape[:2]
    return _resize(image, (2 * height, 2 * width), resample)


def _fit_cell_coefficients(result):
    """Recover the fixed small-system coefficients used by the v3 fit."""
    labels = np.asarray(result["texture_labels"], dtype=np.int32).ravel()
    design = np.asarray(
        result["texture_active_basis"], dtype=np.float64).reshape(
            labels.size, -1)
    target = np.asarray(
        result["texture_fit_lab"], dtype=np.float64).reshape(-1, 3)
    count = np.asarray(result["texture_basis_count"], dtype=np.float64)
    radius = np.asarray(result["texture_basis_radius"], dtype=np.float64)
    cells = len(count)
    terms = design.shape[1]
    coefficient = np.zeros((cells, terms, 3), dtype=np.float64)

    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    boundaries = np.searchsorted(
        sorted_labels, np.arange(cells + 1), side="left")
    for cell in range(cells):
        indices = order[boundaries[cell]:boundaries[cell + 1]]
        if indices.size == 0:
            continue
        local = design[indices]
        normal = local.T @ local
        regularization = np.full(terms, 2e-5 * count[cell])
        regularization[0] = 1e-7 * count[cell]
        if terms > 2:
            gradient = (
                1e-5 * count[cell] / max(radius[cell] ** 2, 1e-30))
            regularization[1:3] = gradient
        normal.flat[::terms + 1] += regularization
        rhs = local.T @ target[indices]
        try:
            coefficient[cell] = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            coefficient[cell] = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    return coefficient


def _lift_v3_texture(result, coefficient):
    """Evaluate the low-resolution v3 design on the doubled lattice."""
    labels = np.asarray(result["texture_labels"], dtype=np.int32)
    high_labels = np.repeat(np.repeat(labels, 2, axis=0), 2, axis=1)
    height, width = high_labels.shape
    low_height, low_width = labels.shape
    centroid = np.asarray(result["texture_basis_centroid"], dtype=np.float64)
    radius = np.asarray(result["texture_basis_radius"], dtype=np.float64)
    active = np.asarray(result["texture_active_basis"], dtype=np.float64)

    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    px = (xx + 0.5) / width - 0.5
    py = (yy + 0.5) / height - 0.5
    cell_coefficient = coefficient[high_labels]
    texture = cell_coefficient[..., 0, :].copy()
    texture += (
        (px - centroid[high_labels, 0]) / radius[high_labels]
    )[..., None] * cell_coefficient[..., 1, :]
    texture += (
        (py - centroid[high_labels, 1]) / radius[high_labels]
    )[..., None] * cell_coefficient[..., 2, :]

    for term in range(3, active.shape[2]):
        lifted = _resize(
            active[..., term], (height, width), Image.Resampling.LANCZOS)
        low_min = float(np.min(active[..., term]))
        low_max = float(np.max(active[..., term]))
        lifted = np.clip(lifted, low_min, low_max)
        texture += lifted[..., None] * cell_coefficient[..., term, :]
    assert texture.shape == (2 * low_height, 2 * low_width, 3)
    return texture, high_labels


def _cell_confidence(result, high_labels):
    labels = np.asarray(result["texture_labels"], dtype=np.int32).ravel()
    residual = (
        np.asarray(result["texture_target_lab"], dtype=np.float64)
        - np.asarray(result["texture_fit_lab"], dtype=np.float64)
    ).reshape(-1, 3)
    energy = np.mean(residual * residual, axis=1)
    cells = len(result["texture_centers"])
    count = np.bincount(labels, minlength=cells)
    total = np.bincount(labels, weights=energy, minlength=cells)
    cell_error = total / np.maximum(count, 1)
    positive = cell_error[cell_error > 0.0]
    scale = (
        float(np.quantile(positive, 0.80))
        if positive.size else 1.0
    )
    confidence = np.exp(-cell_error / max(scale, 1e-12))
    return confidence[high_labels]


def _owner_diffuse(field, labels, passes=2):
    """Small fixed diffusion that never crosses a lifted v3 owner."""
    value = np.asarray(field, dtype=np.float64).copy()
    labels = np.asarray(labels)
    for _ in range(int(passes)):
        total = value.copy()
        weight = np.ones(labels.shape, dtype=np.float64)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted_labels = np.roll(labels, (dy, dx), axis=(0, 1))
            shifted = np.roll(value, (dy, dx), axis=(0, 1))
            valid = shifted_labels == labels
            if dy == -1:
                valid[-1] = False
            elif dy == 1:
                valid[0] = False
            elif dx == -1:
                valid[:, -1] = False
            else:
                valid[:, 0] = False
            total += shifted * valid[..., None]
            weight += valid
        value = total / weight[..., None]
    return value


def _back_project(high, observed, labels=None, uncertainty=None, passes=2):
    """Fixed observation projection; this is not a convergence loop."""
    value = np.asarray(high, dtype=np.float64).copy()
    for _ in range(int(passes)):
        residual = observed - _downsample2(value)
        correction = _upsample2(residual, Image.Resampling.LANCZOS)
        if labels is not None:
            diffused = _owner_diffuse(correction, labels, passes=1)
            if uncertainty is None:
                correction = diffused
            else:
                correction = (
                    (1.0 - uncertainty[..., None]) * correction
                    + uncertainty[..., None] * diffused
                )
        value = np.clip(value + correction, 0.0, 1.0)
    return value


def v3_super_resolve(observed, config=None, return_components=False):
    """Return raw basis lift, Bayesian lift, and the v3 diagnostic."""
    if config is None:
        config = SegmentingV3Config(
            structural_topology="canonical_v2",
            structural_flow_sweeps=1,
            structural_characteristic_passes=1,
            texture_safety_cells=max(131072, observed.shape[0] * observed.shape[1]),
            diagnostic_return_basis=True,
            threads=4,
        )
    started = time.perf_counter()
    result = build_segmenting_v3(observed, config)
    coefficient = _fit_cell_coefficients(result)
    high_texture, high_labels = _lift_v3_texture(result, coefficient)
    high_cartoon = _upsample2(
        result["cartoon_lab"], Image.Resampling.BICUBIC)
    raw = np.clip(lab_to_srgb(high_cartoon + high_texture), 0.0, 1.0)

    confidence = _cell_confidence(result, high_labels)
    texture_density = gaussian_filter(
        np.mean(np.abs(high_texture), axis=2), 1.0)
    density_scale = max(float(np.quantile(texture_density, 0.90)), 1e-12)
    density = np.clip(texture_density / density_scale, 0.0, 1.0)
    retention = confidence * (0.55 + 0.45 * density)

    # Do not replace the observed image with the basis model: its low-order
    # approximation is deliberately lossy.  Isolate only what continuous
    # evaluation adds beyond raster interpolation of that same model, then
    # treat that difference as the speculative super-resolution posterior.
    observed_lanczos = _upsample2(observed, Image.Resampling.LANCZOS)
    rasterized_model = _upsample2(
        result["reconstruction_rgb"], Image.Resampling.LANCZOS)
    innovation = raw - rasterized_model
    magnitude = np.max(np.abs(innovation), axis=2)
    innovation_limit = max(float(np.quantile(magnitude, 0.95)), 1e-12)
    innovation = np.clip(
        innovation, -innovation_limit, innovation_limit)
    posterior = np.clip(
        observed_lanczos + retention[..., None] * innovation, 0.0, 1.0)
    posterior = _back_project(
        posterior,
        observed,
        labels=high_labels,
        uncertainty=1.0 - retention,
        passes=2,
    )
    elapsed = 1000.0 * (time.perf_counter() - started)
    diagnostic = {
        "v3_ms": float(result["timing"]["total_ms"]),
        "super_resolution_total_ms": elapsed,
        "texture_cells": int(len(result["texture_centers"])),
        "mean_retention": float(np.mean(retention)),
        "retention_p10": float(np.quantile(retention, 0.10)),
        "retention_p90": float(np.quantile(retention, 0.90)),
        "innovation_clip_p95": innovation_limit,
    }
    if return_components:
        return raw, posterior, diagnostic, {
            "observed_lanczos": observed_lanczos,
            "innovation": innovation,
            "retention": retention,
            "high_labels": high_labels,
            "result": result,
        }
    return raw, posterior, diagnostic


def _psnr(reference, estimate):
    mse = float(np.mean((reference - estimate) ** 2))
    return -10.0 * math.log10(max(mse, 1e-15))


def _edge_psnr(reference, estimate):
    grey = np.mean(reference, axis=2)
    gx = np.zeros_like(grey)
    gy = np.zeros_like(grey)
    gx[:, 1:-1] = 0.5 * (grey[:, 2:] - grey[:, :-2])
    gy[1:-1] = 0.5 * (grey[2:] - grey[:-2])
    threshold = float(np.quantile(np.hypot(gx, gy), 0.80))
    mask = np.hypot(gx, gy) >= threshold
    mse = float(np.mean((reference[mask] - estimate[mask]) ** 2))
    return -10.0 * math.log10(max(mse, 1e-15))


def _score(reference, methods):
    return {
        name: {
            "psnr_db": _psnr(reference, image),
            "edge_psnr_db": _edge_psnr(reference, image),
            "observation_rmse": float(np.sqrt(np.mean(
                (_downsample2(image) - _downsample2(reference)) ** 2))),
        }
        for name, image in methods.items()
    }


def halo_edge_benchmark():
    """Known oblique step: distinguish ringing from squared-error ranking."""
    height, width, supersample = 256, 320, 8
    y, x = np.mgrid[
        :height * supersample, :width * supersample].astype(np.float64)
    signed = (
        (x + 0.5) / supersample
        - (0.62 * (y + 0.5) / supersample + 62.35)
    )
    reference = (
        0.2 + 0.6 * (signed > 0.0)
    ).reshape(height, supersample, width, supersample).mean(axis=(1, 3))
    reference = np.repeat(reference[..., None], 3, axis=2)
    observed = _downsample2(reference)
    result = build_segmenting_v3(
        observed,
        SegmentingV3Config(
            structural_topology="canonical_v2",
            structural_flow_sweeps=1,
            texture_safety_cells=131072,
            diagnostic_return_basis=True,
            threads=4,
        ),
    )
    methods = {
        "analytic ground truth": reference,
        "Lanczos": _upsample2(observed, Image.Resampling.LANCZOS),
        "local-eikonal Lanczos": eikonal_lanczos2(
            observed, result, anisotropy=0.75, clamp_range=True),
    }
    lower = float(np.min(observed))
    upper = float(np.max(observed))
    report = {}
    for name, image in methods.items():
        if name == "analytic ground truth":
            continue
        grey = image[..., 0]
        halo = np.maximum(lower - image, 0.0) + np.maximum(
            image - upper, 0.0)
        transition = (
            (grey > lower + 0.1 * (upper - lower))
            & (grey < lower + 0.9 * (upper - lower))
        )
        report[name] = {
            "psnr_db": _psnr(reference, image),
            "out_of_support_halo_mean": float(np.mean(halo)),
            "transition_fraction": float(np.mean(transition)),
            "minimum": float(np.min(image)),
            "maximum": float(np.max(image)),
        }
    return report, methods


def _labelled_montage(path, methods):
    names = list(methods)
    height, width = next(iter(methods.values())).shape[:2]
    columns = 3
    rows = int(math.ceil(len(names) / columns))
    header = 24
    canvas = Image.new("RGB", (columns * width, rows * (height + header)))
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        row, column = divmod(index, columns)
        x = column * width
        y = row * (height + header)
        pixels = np.clip(
            np.rint(methods[name] * 255.0), 0, 255).astype(np.uint8)
        canvas.paste(Image.fromarray(pixels), (x, y + header))
        draw.text((x + 5, y + 5), name, fill=(255, 255, 255))
    canvas.save(path)


def benchmark(image_path=DEFAULT_IMAGE, output_dir=None):
    image_path = Path(image_path)
    reference = _rgb(Image.open(image_path).convert("RGB"))
    reference = reference[:reference.shape[0] // 2 * 2,
                          :reference.shape[1] // 2 * 2]
    observed = _downsample2(reference)
    shape = reference.shape[:2]

    methods = {
        "ground truth": reference,
        "nearest": _resize(observed, shape, Image.Resampling.NEAREST),
        "bilinear": _resize(observed, shape, Image.Resampling.BILINEAR),
        "bicubic": _resize(observed, shape, Image.Resampling.BICUBIC),
        "lanczos": _resize(observed, shape, Image.Resampling.LANCZOS),
    }
    methods["lanczos + 2 fixed back-projections"] = _back_project(
        methods["lanczos"], observed, passes=2)
    raw, posterior, diagnostic, components = v3_super_resolve(
        observed, return_components=True)
    methods["v3 continuous basis lift"] = raw
    methods["v3 Bayesian support posterior"] = posterior
    eikonal_lanczos = eikonal_lanczos2(
        observed, components["result"], anisotropy=0.75, clamp_range=True)
    methods["v3 local-eikonal Lanczos"] = eikonal_lanczos
    methods["v3 local-eikonal Lanczos + projection"] = _back_project(
        eikonal_lanczos,
        observed,
        labels=components["high_labels"],
        uncertainty=1.0 - components["retention"],
        passes=2,
    )
    posterior_ladder = {}
    for gain in (0.0, 0.125, 0.25, 0.5, 1.0):
        candidate = np.clip(
            components["observed_lanczos"]
            + gain
            * components["retention"][..., None]
            * components["innovation"],
            0.0,
            1.0,
        )
        candidate = _back_project(
            candidate,
            observed,
            labels=components["high_labels"],
            uncertainty=1.0 - components["retention"],
            passes=2,
        )
        posterior_ladder[f"{gain:.3f}"] = candidate
    methods["v3 owner-conditioned projection"] = posterior_ladder["0.000"]

    scores = _score(reference, {
        key: value for key, value in methods.items() if key != "ground truth"})
    report = {
        "input": str(image_path),
        "ground_truth_shape": list(reference.shape),
        "observed_shape": list(observed.shape),
        "observation": "2x Pillow Lanczos reduction",
        "scores": scores,
        "v3": diagnostic,
        "posterior_gain_ladder": _score(reference, posterior_ladder),
    }
    halo_report, halo_methods = halo_edge_benchmark()
    report["analytic_oblique_edge"] = halo_report
    if output_dir is None:
        output_dir = ROOT / "experiments" / "out" / "v3_super_resolution"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    _labelled_montage(output_dir / "comparison.png", methods)
    detail = {
        name: image[
            int(0.08 * shape[0]):int(0.62 * shape[0]),
            int(0.00 * shape[1]):int(0.72 * shape[1]),
        ]
        for name, image in methods.items()
        if name in (
            "ground truth",
            "lanczos",
            "lanczos + 2 fixed back-projections",
            "v3 Bayesian support posterior",
            "v3 owner-conditioned projection",
            "v3 local-eikonal Lanczos",
            "v3 local-eikonal Lanczos + projection",
        )
    }
    _labelled_montage(output_dir / "bridge_detail.png", detail)
    _labelled_montage(output_dir / "oblique_edge.png", halo_methods)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = benchmark(args.image, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
