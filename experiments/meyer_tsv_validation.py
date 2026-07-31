#!/usr/bin/env python3
"""Validate plain BFFT Meyer contour selectivity against a TSV probe.

This does not implement He--Huska--Liu's weighted G-norm solver.  It asks a
more precise question: does BFFT's unweighted Gilles--Osher Meyer split
already show the outcome their TSV weight is designed to enforce--texture
interior capture without object-contour capture?

Two additive scenes have known cartoon and texture truth.  A discrete
four-direction approximation of the paper's TSV equation (20) supplies an
independent symmetry/contour probe.  One native Meyer trace then scores
passes 1, 2, 4, 8, 16, 32, and 64 without rerunning earlier passes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import bfft  # noqa: E402
from port_needed.frozen_meyer_geometry import build_frozen_geometry  # noqa: E402


PAPER_URL = "https://arxiv.org/abs/2503.22560"
DEFAULT_PASSES = (1, 2, 4, 8, 16, 32, 64)


def _forward_difference(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(image, (-dy, -dx), axis=(0, 1)) - image


def _periodic_kernel(
    shape: tuple[int, int],
    theta: float,
    sigma_long: float,
    sigma_width: float,
    radius: int,
) -> np.ndarray:
    """Rotated anisotropic Gaussian from paper equations (19)--(20)."""
    cosine = math.cos(theta)
    sine = math.sin(theta)
    a = cosine * cosine / (2.0 * sigma_long) + (
        sine * sine / (2.0 * sigma_width))
    b = math.sin(2.0 * theta) / (4.0 * sigma_long) - (
        math.sin(2.0 * theta) / (4.0 * sigma_width))
    c = sine * sine / (2.0 * sigma_long) + (
        cosine * cosine / (2.0 * sigma_width))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    local = np.exp(-(a * xx * xx + 2.0 * b * xx * yy + c * yy * yy))
    local /= np.sum(local)
    kernel = np.zeros(shape, dtype=np.float64)
    for local_y, dy in enumerate(range(-radius, radius + 1)):
        for local_x, dx in enumerate(range(-radius, radius + 1)):
            kernel[dy % shape[0], dx % shape[1]] += local[local_y, local_x]
    return kernel


def tsv_four_direction(
    image: np.ndarray,
    *,
    sigma_long: float = 2.75,
    sigma_width: float = 0.75,
    radius: int = 10,
) -> np.ndarray:
    """Clean-image four-direction TSV diagnostic from paper equation (20)."""
    value = np.ascontiguousarray(image, dtype=np.float64)
    directions = (
        (1, 0, 0.0),
        (0, 1, math.pi / 2.0),
        (1, 1, math.pi / 4.0),
        (1, -1, 3.0 * math.pi / 4.0),
    )
    total = np.zeros_like(value)
    for dy, dx, theta in directions:
        difference = _forward_difference(value, dy, dx)
        kernel = _periodic_kernel(
            value.shape,
            theta,
            float(sigma_long),
            float(sigma_width),
            int(radius),
        )
        integrated = np.fft.ifft2(
            np.fft.fft2(difference) * np.fft.fft2(kernel)
        ).real
        total += np.abs(integrated)
    return total


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(radius)):
        result = (
            result
            | np.roll(result, 1, axis=0)
            | np.roll(result, -1, axis=0)
            | np.roll(result, 1, axis=1)
            | np.roll(result, -1, axis=1)
        )
    return result


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = _forward_difference(image, 0, 1)
    gy = _forward_difference(image, 1, 0)
    return np.hypot(gx, gy)


def symmetric_support_scene(size: int = 256) -> dict:
    """Isolated contours plus a tapered, truly oscillatory interior."""
    y, x = np.mgrid[:size, :size].astype(np.float64)
    cartoon = 92.0 + 10.0 * x / size
    circle = (x - 0.27 * size) ** 2 + (y - 0.29 * size) ** 2 < (
        0.155 * size) ** 2
    cartoon[circle] += 82.0

    x0, x1 = 0.43 * size, 0.92 * size
    y0, y1 = 0.16 * size, 0.86 * size
    rectangle = (x > x0) & (x < x1) & (y > y0) & (y < y1)
    cartoon[rectangle] += 52.0

    distance = np.minimum.reduce((x - x0, x1 - x, y - y0, y1 - y))
    taper = np.clip((distance - 5.0) / 13.0, 0.0, 1.0)
    taper *= rectangle
    carrier = (
        np.cos(2.0 * np.pi * (x + 0.55 * y) / 8.0)
        + 0.45 * np.cos(2.0 * np.pi * (x - 1.8 * y) / 13.0 + 0.7)
    )
    texture = 22.0 * taper * carrier
    source = cartoon + texture

    structural_gradient = _gradient_magnitude(cartoon)
    contour = _dilate(structural_gradient > 4.0, 3)
    texture_interior = (taper > 0.995) & ~_dilate(contour, 5)
    return {
        "name": "symmetric_support",
        "source": source,
        "cartoon": cartoon,
        "texture": texture,
        "contour": contour,
        "texture_interior": texture_interior,
    }


def multiscale_crossing_scene(size: int = 256) -> dict:
    """Authored truth: two scales cross two independently known objects."""
    y, x = np.mgrid[:size, :size].astype(np.float64)
    xn, yn = x / size, y / size
    smooth_cartoon = 88.0 + 13.0 * xn - 7.0 * yn
    circle = (xn - 0.28) ** 2 + (yn - 0.31) ** 2 < 0.145 ** 2
    rectangle = (
        (xn > 0.56) & (xn < 0.87) & (yn > 0.55) & (yn < 0.82)
    )
    authored_jump = np.zeros_like(smooth_cartoon)
    authored_jump[circle] += 76.0
    authored_jump[rectangle] -= 49.0
    hard_composition = smooth_cartoon + authored_jump

    coarse_radius = np.sqrt(
        ((xn - 0.43) / 0.36) ** 2 + ((yn - 0.58) / 0.31) ** 2
    )
    coarse_mask = np.clip((1.0 - coarse_radius) / 0.12, 0.0, 1.0)
    fine_distance = np.minimum.reduce((
        xn - 0.37, 0.94 - xn, yn - 0.12, 0.66 - yn,
    ))
    fine_mask = np.clip(fine_distance / 0.055, 0.0, 1.0)
    coarse = 19.0 * coarse_mask * np.cos(
        2.0 * np.pi * (x + 0.42 * y) / 17.0 + 0.31
    )
    fine = 11.0 * fine_mask * (
        np.cos(2.0 * np.pi * (x - 1.35 * y) / 7.0 - 0.47)
        + 0.35 * np.cos(2.0 * np.pi * (x + 0.75 * y) / 5.0 + 0.19)
    )
    material_texture = coarse + fine
    # Texture is divergence-generated and therefore has no DC mode.  Move
    # the jump potential's one undetermined integration constant to cartoon.
    # The first Meyer cartoon resolvent declares the continuous transition:
    # cartoon retains the objects with smooth boundaries, while texture owns
    # exactly the complementary energy that sharpens those transitions.
    jump_mean = float(np.mean(authored_jump))
    jump_potential = authored_jump - jump_mean
    wy = 2.0 * np.cos(2.0 * np.pi * np.arange(size) / size) - 2.0
    wx = 2.0 * np.cos(2.0 * np.pi * np.arange(size) / size) - 2.0
    laplacian = wy[:, None] + wx[None, :]
    first_cartoon_resolvent = 1.0 / (1.0 - 2.0 * laplacian)
    smooth_jump = np.fft.ifft2(
        np.fft.fft2(jump_potential) * first_cartoon_resolvent
    ).real
    boundary_texture = jump_potential - smooth_jump
    cartoon = smooth_cartoon + jump_mean + smooth_jump
    texture = boundary_texture + material_texture
    source = cartoon + texture
    structural_gradient = _gradient_magnitude(hard_composition)
    contour = _dilate(structural_gradient > 4.0, 3)
    texture_interior = (
        ((coarse_mask > 0.995) | (fine_mask > 0.995))
        & ~_dilate(contour, 5)
    )
    return {
        "name": "multiscale_crossing",
        "source": source,
        "cartoon": cartoon,
        "smooth_cartoon": smooth_cartoon,
        "hard_composition": hard_composition,
        "jump_potential": jump_potential,
        "smooth_jump": smooth_jump,
        "boundary_texture": boundary_texture,
        "texture": texture,
        "material_texture": material_texture,
        "coarse_texture": coarse,
        "fine_texture": fine,
        "coarse_support": coarse_mask,
        "fine_support": fine_mask,
        "contour": contour,
        "texture_interior": texture_interior,
    }


def _rms(value: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(value, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(selected * selected))) if selected.size else 0.0


def _linear_gain(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    target = truth[mask]
    denominator = float(target @ target)
    return float(estimate[mask] @ target / max(denominator, 1e-30))


def _ordering_auc(high: np.ndarray, low: np.ndarray) -> float:
    """Probability that a random `high` sample exceeds a random `low`."""
    a = np.sort(np.asarray(low, dtype=np.float64).ravel())
    b = np.asarray(high, dtype=np.float64).ravel()
    if not a.size or not b.size:
        return 0.5
    below = np.searchsorted(a, b, side="left")
    through = np.searchsorted(a, b, side="right")
    return float(np.mean((below + through) * 0.5 / a.size))


def score_split(
    cartoon_estimate: np.ndarray,
    texture_estimate: np.ndarray,
    scene: dict,
    tsv: np.ndarray | None = None,
) -> dict:
    truth_cartoon = scene["cartoon"]
    truth_texture = scene["texture"]
    source = scene["source"]
    contour = scene["contour"]
    interior = scene["texture_interior"]
    texture_scale = max(_rms(truth_texture, interior), 1e-12)
    gain = _linear_gain(texture_estimate, truth_texture, interior)
    texture_error = _rms(
        texture_estimate - truth_texture, interior) / texture_scale
    contour_excess = _rms(
        texture_estimate - truth_texture, contour) / texture_scale
    truth_gradient = _gradient_magnitude(truth_cartoon)
    estimated_gradient = _gradient_magnitude(cartoon_estimate)
    edge_gain = _linear_gain(estimated_gradient, truth_gradient, contour)
    residual = source - cartoon_estimate - texture_estimate
    effective_structure = source - texture_estimate
    effective_edge_gain = _linear_gain(
        _gradient_magnitude(effective_structure),
        truth_gradient,
        contour,
    )
    allocation_auc = _ordering_auc(
        np.abs(texture_estimate[interior]),
        np.abs((texture_estimate - truth_texture)[contour]),
    )
    result = {
        "interior_texture_gain": gain,
        "interior_texture_relative_rms_error": texture_error,
        "contour_excess_texture_rms": contour_excess,
        "cartoon_edge_gradient_gain": edge_gain,
        "structure_plus_residual_edge_gradient_gain": effective_edge_gain,
        "model_residual_contour_rms": _rms(residual, contour) / texture_scale,
        "model_residual_energy_fraction": float(
            np.sum(residual * residual)
            / max(float(np.sum((source - np.mean(source)) ** 2)), 1e-30)
        ),
        "texture_over_contour_allocation_auc": allocation_auc,
        "cartoon_relative_rms_error": (
            float(np.linalg.norm(cartoon_estimate - truth_cartoon))
            / max(float(np.linalg.norm(truth_cartoon)), 1e-30)
        ),
        "texture_relative_rms_error": (
            float(np.linalg.norm(texture_estimate - truth_texture))
            / max(float(np.linalg.norm(truth_texture)), 1e-30)
        ),
    }
    if tsv is not None:
        high_tsv = tsv >= np.percentile(tsv, 90.0)
        texture_error_field = np.square(texture_estimate - truth_texture)
        texture_energy_field = np.square(texture_estimate)
        result.update({
            "high_tsv_pixel_fraction": float(np.mean(high_tsv)),
            "texture_error_energy_in_high_tsv_fraction": float(
                np.sum(texture_error_field[high_tsv])
                / max(float(np.sum(texture_error_field)), 1e-30)
            ),
            "texture_energy_in_high_tsv_fraction": float(
                np.sum(texture_energy_field[high_tsv])
                / max(float(np.sum(texture_energy_field)), 1e-30)
            ),
        })
    return result


def _normalize_map(value: np.ndarray) -> np.ndarray:
    scale = max(float(np.percentile(value, 99.5)), 1e-12)
    return np.clip(value / scale, 0.0, 1.0)


def evaluate_scene(
    scene: dict,
    *,
    lam: float,
    mu: float,
    passes: tuple[int, ...],
    threads: int,
    sigma_long: float,
    sigma_width: float,
) -> tuple[dict, dict]:
    source = np.asarray(scene["source"], dtype=np.float64)
    t0 = time.perf_counter()
    tsv = tsv_four_direction(
        source,
        sigma_long=sigma_long,
        sigma_width=sigma_width,
    )
    tsv_ms = 1000.0 * (time.perf_counter() - t0)
    contour = scene["contour"]
    interior = scene["texture_interior"]
    tsv_probe = {
        "edge_mean": float(np.mean(tsv[contour])),
        "texture_interior_mean": float(np.mean(tsv[interior])),
        "edge_to_texture_mean_ratio": float(
            np.mean(tsv[contour]) / max(float(np.mean(tsv[interior])), 1e-30)
        ),
        "edge_over_texture_auc": _ordering_auc(tsv[contour], tsv[interior]),
        "milliseconds": tsv_ms,
    }

    requested = set(map(int, passes))
    selected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    scores = {}
    plan = bfft.MeyerPlan(
        source.shape,
        lam=float(lam),
        mu=float(mu),
        passes=max(requested),
        rung_sweeps=1,
        rung_tol=0.0,
        threads=int(threads),
        solver=0,
    )
    t0 = time.perf_counter()

    def visit(pass_number, cartoon, texture):
        if pass_number in requested:
            cartoon_copy = cartoon.copy()
            texture_copy = texture.copy()
            selected[pass_number] = (cartoon_copy, texture_copy)
            scores[str(pass_number)] = score_split(
                cartoon_copy, texture_copy, scene, tsv=tsv)

    plan.visit(source, visit)
    meyer_ms = 1000.0 * (time.perf_counter() - t0)

    # The segmenter's frozen support is downstream of Meyer, not part of the
    # G-norm. Probe it separately because its determinant/coherence geometry
    # explicitly turns coherent contours into long, low-density cells.
    rgb = np.repeat(
        np.clip(source / 255.0, 0.0, 1.0)[..., None], 3, axis=2)
    t0 = time.perf_counter()
    geometry = build_frozen_geometry(
        rgb,
        tgfd_sweeps=1,
        flow_sweeps=24,
        texture_support_weight=0.65,
        glass_support_weight=0.70,
        null_evidence_strength=0.5,
        threads=int(threads),
        meyer_solver=1,
    )
    geometry_ms = 1000.0 * (time.perf_counter() - t0)
    geometry_probe = {"milliseconds": geometry_ms, "fields": {}}
    for field_name in (
        "measure",
        "boundary_confidence",
        "source_reliability",
        "energy",
    ):
        field = np.asarray(geometry[field_name], dtype=np.float64)
        edge_mean = float(np.mean(field[contour]))
        texture_mean = float(np.mean(field[interior]))
        geometry_probe["fields"][field_name] = {
            "contour_mean": edge_mean,
            "texture_interior_mean": texture_mean,
            "contour_to_texture_mean_ratio": (
                edge_mean / max(texture_mean, 1e-30)
            ),
            "texture_over_contour_auc": _ordering_auc(
                field[interior], field[contour]),
        }
    record = {
        "tsv_probe": tsv_probe,
        "meyer_trace_ms": meyer_ms,
        "segmenter_frozen_geometry_probe": geometry_probe,
        "passes": scores,
        "masks": {
            "contour_pixels": int(np.count_nonzero(contour)),
            "texture_interior_pixels": int(np.count_nonzero(interior)),
            "truth_texture_rms_interior": _rms(scene["texture"], interior),
            "truth_texture_rms_at_contour": _rms(scene["texture"], contour),
        },
    }
    artifacts = {
        "tsv": _normalize_map(tsv),
        "selected": selected,
        "support_measure": _normalize_map(geometry["measure"]),
        "boundary_confidence": _normalize_map(
            geometry["boundary_confidence"]),
    }
    return record, artifacts


def _to_rgb(value: np.ndarray, *, signed: bool = False) -> np.ndarray:
    field = np.asarray(value, dtype=np.float64)
    if signed:
        scale = max(float(np.percentile(np.abs(field), 99.0)), 1e-12)
        field = 0.5 + 0.48 * field / scale
    else:
        low, high = np.percentile(field, (0.2, 99.8))
        field = (field - low) / max(float(high - low), 1e-12)
    grey = np.clip(np.rint(field * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2)


def _heat(value: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return np.stack((
        np.clip(2.2 * z, 0.0, 1.0),
        np.clip(2.0 - 3.0 * np.abs(z - 0.55), 0.0, 1.0),
        np.clip(1.1 - 1.7 * z, 0.0, 1.0),
    ), axis=-1) * 255.0


def _save_montage(scene: dict, artifacts: dict, path: Path, display_pass: int):
    cartoon, texture = artifacts["selected"][display_pass]
    tiles = (
        ("source", _to_rgb(scene["source"])),
        ("true cartoon", _to_rgb(scene["cartoon"])),
        ("true texture", _to_rgb(scene["texture"], signed=True)),
        ("TSV symmetry probe", _heat(artifacts["tsv"]).astype(np.uint8)),
        (f"Meyer cartoon p={display_pass}", _to_rgb(cartoon)),
        (f"Meyer texture p={display_pass}", _to_rgb(texture, signed=True)),
        (
            "cartoon error",
            _to_rgb(cartoon - scene["cartoon"], signed=True),
        ),
        (
            "texture error",
            _to_rgb(texture - scene["texture"], signed=True),
        ),
    )
    height, width = scene["source"].shape
    label_height = 24
    canvas = Image.new("RGB", (4 * width, 2 * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, pixels) in enumerate(tiles):
        row, column = divmod(index, 4)
        left = column * width
        top = row * (height + label_height)
        draw.text((left + 6, top + 5), label, fill="black")
        canvas.paste(Image.fromarray(pixels, "RGB"), (left, top + label_height))
    canvas.save(path)


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    passes = tuple(sorted(set(args.passes)))
    scenes = (
        symmetric_support_scene(args.size),
        multiscale_crossing_scene(args.size),
    )
    report = {
        "paper": PAPER_URL,
        "hypothesis": (
            "test whether unweighted TV/G competition empirically rejects "
            "contours like TSV weighting, despite having no eta(x) field"
        ),
        "parameters": {
            "size": args.size,
            "lambda": args.lam,
            "mu": args.mu,
            "passes": list(passes),
            "threads": args.threads,
            "tsv_sigma_long": args.tsv_sigma_long,
            "tsv_sigma_width": args.tsv_sigma_width,
        },
        "scenes": {},
    }
    for scene in scenes:
        record, artifacts = evaluate_scene(
            scene,
            lam=args.lam,
            mu=args.mu,
            passes=passes,
            threads=args.threads,
            sigma_long=args.tsv_sigma_long,
            sigma_width=args.tsv_sigma_width,
        )
        report["scenes"][scene["name"]] = record
        _save_montage(
            scene,
            artifacts,
            output / f"{scene['name']}_validation.png",
            args.display_pass,
        )
        Image.fromarray(
            _heat(artifacts["support_measure"]).astype(np.uint8), "RGB"
        ).save(output / f"{scene['name']}_support_measure.png")
        Image.fromarray(
            _heat(artifacts["boundary_confidence"]).astype(np.uint8), "RGB"
        ).save(output / f"{scene['name']}_boundary_confidence.png")
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "experiments/out/meyer_tsv_validation"),
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=40.0)
    parser.add_argument("--passes", type=int, nargs="+", default=DEFAULT_PASSES)
    parser.add_argument("--display-pass", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    # The paper explicitly makes TSV scale dependent. The validation's
    # carriers have periods 8 and 13 pixels, so the long support must span
    # that interval; 2.75 is one example from the paper, not a universal
    # constant.
    parser.add_argument("--tsv-sigma-long", type=float, default=12.0)
    parser.add_argument("--tsv-sigma-width", type=float, default=0.75)
    args = parser.parse_args()
    if args.size < 16 or args.size & (args.size - 1):
        parser.error("--size must be a power of two >= 16")
    if args.display_pass not in args.passes:
        parser.error("--display-pass must be included in --passes")
    return args


def main() -> int:
    print(json.dumps(run(parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
