#!/usr/bin/env python3
"""Fixed-population canopy descent through the BFFT transport stack.

The final transported support measure determines the population once.  Those
sites exist at the first pass and are never created, selected, split, deleted,
or ranked.  Every BFFT pass then performs one simultaneous continuation step:

1. transport the sites with the measured inter-pass flow;
2. form a soft anisotropic canopy from their current compact neighborhoods;
3. move each site toward the support-measure centroid beneath its canopy;
4. update its area and covariance from that same shared partition;
5. adjust one additive site potential toward one quantum of support measure;
6. anneal the canopy from diffuse cooperation toward a sharper diagram.

This is a soft, sparse, anisotropic capacity-constrained centroidal diagram.
It is not target-image gradient descent: geometry sees only the BFFT support
measure, transport, and event tensors.  Image values enter only in the final
linear coefficient solve.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from bfft.vision import SingleStageDecompositionObjective  # noqa: E402
from bfft_flow_stage_geometry import build_flow_volume  # noqa: E402
from dual_aperture_support import design_matrix, score, solve_field  # noqa: E402
from flow_support_measure import infer_support_measure  # noqa: E402
from flow_volume_cells import Population, support_samples  # noqa: E402
from resource_transport_cells import _r2_sites  # noqa: E402
from transport_measure_cells import (  # noqa: E402
    population_geometry_views,
    quantize_rosenblatt,
    site_id_colours,
)
from transport_voronoi import _fit_rgb, srgb_to_lab  # noqa: E402


def _sample_nearest(field: np.ndarray, centers: np.ndarray) -> np.ndarray:
    height, width = field.shape
    x = np.clip(np.rint(centers[:, 0]).astype(np.intp), 0, width - 1)
    y = np.clip(np.rint(centers[:, 1]).astype(np.intp), 0, height - 1)
    return np.asarray(field, dtype=np.float64)[y, x]


def _canopy_partition(
    samples: dict,
    pixels: int,
    sharpness: float,
    log_gain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize the sparse anisotropic metric canopy over every pixel."""
    phi = np.asarray(samples["phi"], dtype=np.float64)
    rows = np.asarray(samples["rows"], dtype=np.intp)
    sites = np.asarray(samples["sites"], dtype=np.intp)
    # support_samples uses phi=(1-q)^2. Recover normalized ellipse distance q
    # and use the compact ellipse only as a sparse neighborhood, not as the
    # partition profile itself.
    q = 1.0 - np.sqrt(np.clip(phi, 0.0, 1.0))
    raw = np.exp(
        -float(sharpness) * q
        + np.clip(np.asarray(log_gain)[sites], -20.0, 20.0))
    denominator = np.bincount(rows, weights=raw, minlength=pixels)
    weight = raw / np.maximum(denominator[rows], 1e-30)
    return weight, denominator, q


def _blend_angle(
    old: np.ndarray,
    new: np.ndarray,
    fraction: float,
) -> np.ndarray:
    x = (
        (1.0 - fraction) * np.cos(2.0 * old)
        + fraction * np.cos(2.0 * new))
    y = (
        (1.0 - fraction) * np.sin(2.0 * old)
        + fraction * np.sin(2.0 * new))
    return 0.5 * np.arctan2(y, x)


def evolve_canopy_population(
    volume: dict,
    support: dict,
    overlap: float = 8.0,
    sharpness_start: float = 1.5,
    sharpness_end: float = 7.0,
    centroid_step: float = 0.58,
    shape_step: float = 0.38,
    capacity_step: float = 0.70,
    flow_step: float = 0.65,
    initialization: str = "density",
) -> tuple[Population, np.ndarray, dict, list[dict]]:
    """Descend one fixed population through the complete support flow."""
    envelopes = np.asarray(support["envelope"], dtype=np.float64)
    stages, height, width = envelopes.shape
    pixels = height * width
    count = max(1, int(round(float(support["transported_count"]))))
    if initialization == "density":
        initial_density = envelopes[0].copy()
        initial_density *= count / max(float(np.sum(initial_density)), 1e-30)
        centers = quantize_rosenblatt(initial_density)
    elif initialization == "uniform":
        centers = _r2_sites(count, width, height)
    else:
        raise ValueError("initialization must be 'density' or 'uniform'")

    spacing = math.sqrt(pixels / count)
    radius = math.sqrt(float(overlap) * pixels / (math.pi * count))
    major = np.full(count, radius, dtype=np.float64)
    minor = np.full(count, radius, dtype=np.float64)
    angle = np.zeros(count, dtype=np.float64)
    log_gain = np.zeros(count, dtype=np.float64)
    trace: list[dict] = []

    tx = np.asarray(volume["transport_x"], dtype=np.float64)
    ty = np.asarray(volume["transport_y"], dtype=np.float64)
    confidence = np.asarray(
        volume["transport_confidence"], dtype=np.float64)
    persistence = np.asarray(
        volume["transport_persistence"], dtype=np.float64)
    local_qxx = np.asarray(
        support["local_precision_xx"], dtype=np.float64)
    local_qxy = np.asarray(
        support["local_precision_xy"], dtype=np.float64)
    local_qyy = np.asarray(
        support["local_precision_yy"], dtype=np.float64)
    yy_flat, xx_flat = np.mgrid[:height, :width]
    pixel_x = xx_flat.ravel().astype(np.float64)
    pixel_y = yy_flat.ravel().astype(np.float64)

    for stage in range(stages):
        progress = stage / max(stages - 1, 1)
        sharpness = (
            float(sharpness_start)
            * (float(sharpness_end) / float(sharpness_start)) ** progress)

        if stage:
            gate = confidence[stage] * persistence
            centers[:, 0] += float(flow_step) * _sample_nearest(
                gate * tx[stage], centers)
            centers[:, 1] += float(flow_step) * _sample_nearest(
                gate * ty[stage], centers)
            centers[:, 0] = np.clip(centers[:, 0], 0.0, width - 1.0)
            centers[:, 1] = np.clip(centers[:, 1], 0.0, height - 1.0)

        population = Population(
            centers=centers,
            major=major,
            minor=minor,
            angle=angle,
            stage=np.full(count, stage + 1, dtype=np.int16),
            base_cells=0,
        )
        samples = support_samples(population, height, width)
        rows = np.asarray(samples["rows"], dtype=np.intp)
        sites = np.asarray(samples["sites"], dtype=np.intp)
        weight, denominator, _ = _canopy_partition(
            samples, pixels, sharpness, log_gain)

        density = envelopes[stage].ravel().copy()
        density *= count / max(float(np.sum(density)), 1e-30)
        responsibility = weight * density[rows]
        mass = np.bincount(
            sites, weights=responsibility, minlength=count)
        safe_mass = np.maximum(mass, 1e-30)
        centroid_x = np.bincount(
            sites,
            weights=responsibility * pixel_x[rows],
            minlength=count,
        ) / safe_mass
        centroid_y = np.bincount(
            sites,
            weights=responsibility * pixel_y[rows],
            minlength=count,
        ) / safe_mass
        valid = mass > 1e-12
        displacement_x = np.where(valid, centroid_x - centers[:, 0], 0.0)
        displacement_y = np.where(valid, centroid_y - centers[:, 1], 0.0)
        displacement = np.hypot(displacement_x, displacement_y)
        limit = max(0.9 * spacing, 0.75)
        limiter = np.minimum(1.0, limit / np.maximum(displacement, 1e-30))
        centers[:, 0] += (
            float(centroid_step) * limiter * displacement_x)
        centers[:, 1] += (
            float(centroid_step) * limiter * displacement_y)
        centers[:, 0] = np.clip(centers[:, 0], 0.0, width - 1.0)
        centers[:, 1] = np.clip(centers[:, 1], 0.0, height - 1.0)

        dx = pixel_x[rows] - centers[sites, 0]
        dy = pixel_y[rows] - centers[sites, 1]
        cxx = np.bincount(
            sites, weights=responsibility * dx * dx,
            minlength=count) / safe_mass
        cxy = np.bincount(
            sites, weights=responsibility * dx * dy,
            minlength=count) / safe_mass
        cyy = np.bincount(
            sites, weights=responsibility * dy * dy,
            minlength=count) / safe_mass
        covariance_trace = cxx + cyy
        covariance_disc = np.hypot(cxx - cyy, 2.0 * cxy)
        covariance_high = np.maximum(
            0.5 * (covariance_trace + covariance_disc), 1e-4)
        covariance_low = np.maximum(
            0.5 * (covariance_trace - covariance_disc), 1e-4)
        covariance_ratio = np.sqrt(covariance_high / covariance_low)
        covariance_angle = 0.5 * np.arctan2(
            2.0 * cxy, cxx - cyy)

        qxx = _sample_nearest(local_qxx[stage], centers)
        qxy = _sample_nearest(local_qxy[stage], centers)
        qyy = _sample_nearest(local_qyy[stage], centers)
        qtrace = qxx + qyy
        qdisc = np.hypot(qxx - qyy, 2.0 * qxy)
        qhigh = np.maximum(0.5 * (qtrace + qdisc), 1e-12)
        qlow = np.maximum(0.5 * (qtrace - qdisc), 1e-12)
        flow_ratio = np.sqrt(qhigh / qlow)
        flow_angle = (
            0.5 * np.arctan2(2.0 * qxy, qxx - qyy)
            + 0.5 * math.pi)
        coherence = np.clip(
            (qhigh - qlow) / np.maximum(qhigh + qlow, 1e-30),
            0.0, 1.0)
        desired_ratio = np.exp(
            (1.0 - coherence) * np.log(np.clip(
                covariance_ratio, 1.0, 12.0))
            + coherence * np.log(np.clip(flow_ratio, 1.0, 12.0)))
        desired_angle = _blend_angle(
            covariance_angle, flow_angle, coherence)

        # The canopy's current uniform-area mass is its emergent territory.
        territory = np.bincount(
            sites, weights=weight, minlength=count)
        desired_area = np.maximum(float(overlap) * territory, math.pi * 0.75**2)
        desired_major = np.sqrt(
            desired_area * desired_ratio / math.pi)
        desired_minor = np.sqrt(
            desired_area / (math.pi * desired_ratio))
        major = np.exp(
            (1.0 - float(shape_step)) * np.log(np.maximum(major, 0.75))
            + float(shape_step) * np.log(np.maximum(desired_major, 0.75)))
        minor = np.exp(
            (1.0 - float(shape_step)) * np.log(np.maximum(minor, 0.75))
            + float(shape_step) * np.log(np.maximum(desired_minor, 0.75)))
        major = np.clip(major, 0.75, 0.45 * max(height, width))
        minor = np.clip(minor, 0.75, 0.45 * max(height, width))
        angle = _blend_angle(angle, desired_angle, float(shape_step))

        correction = -np.log(np.maximum(mass, 1e-30))
        correction -= np.mean(correction[np.isfinite(correction)])
        log_gain += float(capacity_step) * correction
        log_gain -= np.mean(log_gain)
        log_gain = np.clip(log_gain, -12.0, 12.0)

        covered = denominator > 1e-30
        effective = 1.0 / np.maximum(
            np.bincount(
                rows, weights=weight * weight, minlength=pixels),
            1e-30,
        )
        trace.append({
            "stage": stage + 1,
            "sharpness": sharpness,
            "coverage": float(np.mean(covered)),
            "support_mass_cv": float(np.std(mass)),
            "support_mass_min": float(np.min(mass)),
            "effective_median": float(np.median(effective[covered])),
            "centroid_motion_rms": float(np.sqrt(np.mean(
                displacement_x * displacement_x
                + displacement_y * displacement_y))),
            "ratio_median": float(np.median(major / minor)),
            "centers": centers.copy(),
            "major": major.copy(),
            "minor": minor.copy(),
            "angle": angle.copy(),
        })

    final = Population(
        centers=centers,
        major=major,
        minor=minor,
        angle=angle,
        stage=np.full(count, stages, dtype=np.int16),
        base_cells=0,
    )
    samples = support_samples(final, height, width)
    weights, _, _ = _canopy_partition(
        samples, pixels, sharpness_end, log_gain)
    return final, weights, samples, trace


def fit_canopy_population(
    population: Population,
    samples: dict,
    weight: np.ndarray,
    target_lab: np.ndarray,
    objective: SingleStageDecompositionObjective,
) -> tuple[dict, np.ndarray, dict]:
    height, width = target_lab.shape[:2]
    pixels = height * width
    design = design_matrix(samples, pixels, weight)
    reconstruction = solve_field(design, target_lab)
    record = score(objective, objective.target_rgb, reconstruction)
    rows = np.asarray(samples["rows"], dtype=np.intp)
    covered = np.asarray(design.getnnz(axis=1) > 0)
    effective = 1.0 / np.maximum(
        np.bincount(rows, weights=weight * weight, minlength=pixels),
        1e-30,
    )
    return record, reconstruction, {
        "cells": len(population.centers),
        "samples": len(rows),
        "coverage": float(np.mean(covered)),
        "effective_median": float(np.median(effective[covered])),
    }


def canopy_site_ids(
    population: Population,
    samples: dict,
    weight: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    colours = site_id_colours(len(population.centers))
    rows = np.asarray(samples["rows"], dtype=np.intp)
    sites = np.asarray(samples["sites"], dtype=np.intp)
    result = np.zeros((height * width, 3), dtype=np.float64)
    for channel in range(3):
        result[:, channel] = np.bincount(
            rows,
            weights=weight * colours[sites, channel],
            minlength=height * width,
        )
    return result.reshape(height, width, 3)


def canopy_geometry_views(
    population: Population,
    samples: dict,
    weight: np.ndarray,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    """Expose the evolved partition and literal neighborhoods."""
    pixels = int(height) * int(width)
    rows = np.asarray(samples["rows"], dtype=np.intp)
    sites = np.asarray(samples["sites"], dtype=np.intp)
    colours = site_id_colours(len(population.centers))
    soft = canopy_site_ids(
        population, samples, weight, height, width)
    largest = np.zeros(pixels, dtype=np.float64)
    np.maximum.at(largest, rows, weight)
    dominant = np.full(pixels, -1, dtype=np.int32)
    strongest = weight >= largest[rows] - 1e-14
    dominant[rows[strongest]] = sites[strongest]
    covered = dominant >= 0
    hard = np.zeros((pixels, 3), dtype=np.float32)
    hard[covered] = colours[dominant[covered]].astype(np.float32)
    effective = 1.0 / np.maximum(
        np.bincount(rows, weights=weight * weight, minlength=pixels),
        1e-30,
    )
    outlines = population_geometry_views(
        population, height, width)["cell_outlines"]
    return {
        "soft_site_ids": soft.astype(np.float32),
        "dominant_site_ids": hard.reshape(height, width, 3),
        "cell_outlines": outlines,
        "dominance": largest.reshape(height, width).astype(np.float32),
        "effective_contributors": effective.reshape(
            height, width).astype(np.float32),
        "covered": covered.reshape(height, width),
    }


def save_panel(
    rgb: np.ndarray,
    population: Population,
    record: dict,
    samples: dict,
    weight: np.ndarray,
    trace: list[dict],
    output: Path,
) -> None:
    height, width = rgb.shape[:2]
    site_ids = canopy_site_ids(
        population, samples, weight, height, width)
    outlines = population_geometry_views(
        population, height, width)["cell_outlines"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("target")
    axes[0, 1].imshow(record["rgb"])
    axes[0, 1].set_title(
        f"canopy reconstruction\n{record['psnr']:.2f} dB")
    axes[0, 2].imshow(site_ids)
    axes[0, 2].set_title("soft site IDs")
    axes[1, 0].imshow(outlines)
    axes[1, 0].set_title("final compact neighborhoods")
    axes[1, 1].plot(
        [item["stage"] for item in trace],
        [item["support_mass_cv"] for item in trace],
        label="capacity CV")
    axes[1, 1].plot(
        [item["stage"] for item in trace],
        [item["effective_median"] for item in trace],
        label="effective contributors")
    axes[1, 1].legend()
    axes[1, 1].set_title("canopy continuation")
    selected = [
        trace[index]
        for index in sorted(set([
            0, len(trace) // 3, 2 * len(trace) // 3, len(trace) - 1]))
    ]
    axes[1, 2].imshow(rgb, alpha=0.20)
    for item in selected:
        centers = item["centers"]
        axes[1, 2].scatter(
            centers[:, 0], centers[:, 1], s=1.5,
            label=f"pass {item['stage']}", alpha=0.45)
    axes[1, 2].legend(markerscale=3, fontsize=7)
    axes[1, 2].set_title("fixed sites descending through flow")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--passes", type=int, default=24)
    parser.add_argument("--flow-sweeps", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=8.0)
    parser.add_argument("--sharpness-start", type=float, default=1.5)
    parser.add_argument("--sharpness-end", type=float, default=10.0)
    parser.add_argument(
        "--initialization", choices=("density", "uniform"),
        default="density")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/out/transport_canopy_cells.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/transport_canopy_cells.json")
    args = parser.parse_args()

    if args.image:
        from skimage.io import imread

        source = imread(Path(args.image).expanduser())
        source_name = str(Path(args.image).expanduser())
    else:
        source = gallery.load(args.gallery)
        source_name = f"gallery:{args.gallery}"
    rgb = _fit_rgb(source, args.side)
    started = time.perf_counter()
    volume = build_flow_volume(
        rgb, passes=args.passes, flow_sweeps=args.flow_sweeps)
    support = infer_support_measure(volume)
    geometry_started = time.perf_counter()
    population, weights, samples, trace = evolve_canopy_population(
        volume,
        support,
        overlap=args.overlap,
        sharpness_start=args.sharpness_start,
        sharpness_end=args.sharpness_end,
        initialization=args.initialization,
    )
    geometry_seconds = time.perf_counter() - geometry_started
    objective = SingleStageDecompositionObjective(
        rgb, passes=args.passes)
    record, _, diagnostic = fit_canopy_population(
        population, samples, weights, srgb_to_lab(rgb), objective)
    save_panel(
        rgb, population, record, samples, weights, trace, args.output)
    report = {
        "source": source_name,
        "shape": list(rgb.shape),
        "passes": args.passes,
        "cells": len(population.centers),
        "overlap": args.overlap,
        "sharpness_start": args.sharpness_start,
        "sharpness_end": args.sharpness_end,
        "initialization": args.initialization,
        "geometry_seconds": geometry_seconds,
        "total_seconds": time.perf_counter() - started,
        "record": {
            key: float(value)
            for key, value in record.items() if key != "rgb"
        },
        "diagnostic": diagnostic,
        "trace": [{
            key: value for key, value in item.items()
            if key not in ("centers", "major", "minor", "angle")
        } for item in trace],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "cells": report["cells"],
        "psnr": report["record"]["psnr"],
        "objective": report["record"]["objective"],
        "geometry_seconds": geometry_seconds,
        "coverage": diagnostic["coverage"],
        "effective_median": diagnostic["effective_median"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
