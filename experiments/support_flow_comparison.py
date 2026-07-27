#!/usr/bin/env python3
"""Compare support flow in resource cells and the known-good reference.

The experiment matches the final resource-cell population with a
ReceiverGuidedVoronoi model, then measures support geometry independently of
PSNR:

* contributors per pixel and maximum support dominance;
* ambiguous boundary area;
* cell-domain area and anisotropy;
* site bias toward BFFT texture activity;
* cartoon/detail aperture width in the reference.

Run:
    .venv/bin/python experiments/support_flow_comparison.py \
        ~/Downloads/25.png --side 128 --resource-rounds 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from receiver_guided_graph import ReceiverGuidedVoronoi  # noqa: E402
from resource_transport_cells import (  # noqa: E402
    ResourceConfig, ResourceTransportCells,
)
from transport_voronoi import Config, _fit_rgb  # noqa: E402


def _safe_percentiles(value):
    finite = np.asarray(value)[np.isfinite(value)]
    if finite.size == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    p10, p50, p90 = np.percentile(finite, (10.0, 50.0, 90.0))
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def resource_support_fields(model):
    sum_phi = np.zeros((model.h, model.w), dtype=np.float64)
    sum_phi2 = np.zeros_like(sum_phi)
    sum_phi_log_phi = np.zeros_like(sum_phi)
    max_phi = np.zeros_like(sum_phi)
    for site in range(len(model.centers)):
        patch = model._patch(site)
        if patch is None:
            continue
        y0, y1, x0, x1, _, _, _, phi = patch
        target = np.s_[y0:y1, x0:x1]
        sum_phi[target] += phi
        sum_phi2[target] += phi * phi
        sum_phi_log_phi[target] += (
            phi * np.log(np.maximum(phi, 1e-30)))
        max_phi[target] = np.maximum(max_phi[target], phi)
    safe_sum = np.maximum(sum_phi, 1e-30)
    effective_l2 = sum_phi * sum_phi / np.maximum(sum_phi2, 1e-30)
    entropy = (
        np.log(safe_sum) -
        sum_phi_log_phi / safe_sum)
    effective_entropy = np.exp(np.clip(entropy, 0.0, 20.0))
    dominance = max_phi / safe_sum
    return {
        "sum": sum_phi,
        "effective_l2": effective_l2,
        "effective_entropy": effective_entropy,
        "dominance": dominance,
    }


def reference_support_fields(model, cartoon_softness=4.0,
                             texture_softness=16.0):
    gap = (model.d2 - model.d1).reshape(model.h, model.w)
    valid = (model.second >= 0).reshape(model.h, model.w)

    def aperture(softness):
        z = np.clip(0.5 * float(softness) * gap, -50.0, 50.0)
        first = 1.0 / (1.0 + np.exp(-z))
        first[~valid] = 1.0
        second = 1.0 - first
        dominance = np.maximum(first, second)
        effective = 1.0 / np.maximum(
            first * first + second * second, 1e-30)
        return first, dominance, effective

    base_first, base_dominance, base_effective = aperture(
        cartoon_softness)
    detail_first, detail_dominance, detail_effective = aperture(
        texture_softness)
    return {
        "base_first": base_first,
        "base_dominance": base_dominance,
        "base_effective": base_effective,
        "detail_first": detail_first,
        "detail_dominance": detail_dominance,
        "detail_effective": detail_effective,
    }


def _ellipse_ratios_from_owner(owner, cells, width, height):
    ids = owner.ravel()
    x = np.tile(np.arange(width, dtype=np.float64), height)
    y = np.repeat(np.arange(height, dtype=np.float64), width)
    count = np.bincount(ids, minlength=cells).astype(np.float64)
    safe = np.maximum(count, 1.0)
    mx = np.bincount(ids, weights=x, minlength=cells) / safe
    my = np.bincount(ids, weights=y, minlength=cells) / safe
    dx = x - mx[ids]
    dy = y - my[ids]
    xx = np.bincount(ids, weights=dx * dx, minlength=cells) / safe
    yy = np.bincount(ids, weights=dy * dy, minlength=cells) / safe
    xy = np.bincount(ids, weights=dx * dy, minlength=cells) / safe
    disc = np.hypot(xx - yy, 2.0 * xy)
    high = np.maximum(0.5 * (xx + yy + disc), 1e-12)
    low = np.maximum(0.5 * (xx + yy - disc), 1e-12)
    return count, np.sqrt(high / low)


def _site_texture_bias(points, texture_activity):
    h, w = texture_activity.shape
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, h - 1)
    sample = texture_activity[y, x]
    baseline = max(float(np.mean(texture_activity)), 1e-12)
    return {
        "mean_at_sites": float(np.mean(sample)),
        "image_mean": float(np.mean(texture_activity)),
        "bias_ratio": float(np.mean(sample) / baseline),
        **_safe_percentiles(sample),
    }


def summarize_resource(model, fields, texture_activity):
    areas = math.pi * model.major * model.minor
    ratios = model.major / np.maximum(model.minor, 1e-12)
    dominance = fields["dominance"]
    return {
        "cells": int(len(model.centers)),
        "psnr": float(model.psnr),
        "contributors_l2": {
            "mean": float(np.mean(fields["effective_l2"])),
            **_safe_percentiles(fields["effective_l2"]),
        },
        "contributors_entropy": {
            "mean": float(np.mean(fields["effective_entropy"])),
            **_safe_percentiles(fields["effective_entropy"]),
        },
        "dominance": {
            "mean": float(np.mean(dominance)),
            "fraction_below_0_8": float(np.mean(dominance < 0.8)),
            "fraction_below_0_6": float(np.mean(dominance < 0.6)),
            **_safe_percentiles(dominance),
        },
        "area": {
            "mean": float(np.mean(areas)),
            "cv": float(np.std(areas) / max(np.mean(areas), 1e-12)),
            **_safe_percentiles(areas),
        },
        "anisotropy": {
            "mean": float(np.mean(ratios)),
            **_safe_percentiles(ratios),
        },
        "site_texture": _site_texture_bias(
            model.centers, texture_activity),
    }


def summarize_reference(model, fields):
    owner = model.owner.reshape(model.h, model.w)
    area, ratio = _ellipse_ratios_from_owner(
        owner, len(model.seeds), model.w, model.h)
    result = {
        "cells": int(len(model.seeds)),
        "psnr": float(model.psnr),
        "area": {
            "mean": float(np.mean(area)),
            "cv": float(np.std(area) / max(np.mean(area), 1e-12)),
            **_safe_percentiles(area),
        },
        "anisotropy": {
            "mean": float(np.mean(ratio)),
            **_safe_percentiles(ratio),
        },
        "site_texture": _site_texture_bias(
            model.seeds, model.texture_activity),
    }
    for name in ("base", "detail"):
        dominance = fields[f"{name}_dominance"]
        effective = fields[f"{name}_effective"]
        result[f"{name}_aperture"] = {
            "contributors_mean": float(np.mean(effective)),
            "dominance_mean": float(np.mean(dominance)),
            "fraction_below_0_8": float(np.mean(dominance < 0.8)),
            "fraction_below_0_6": float(np.mean(dominance < 0.6)),
            "dominance": _safe_percentiles(dominance),
        }
    return result


def _normalize(field, percentile=99.0):
    field = np.asarray(field, dtype=np.float64)
    scale = max(float(np.percentile(field, percentile)), 1e-12)
    return np.clip(field / scale, 0.0, 1.0)


def _site_overlay(rgb, points, color):
    image = rgb.copy()
    h, w = image.shape[:2]
    radius = max(1, int(round(max(h, w) / 220.0)))
    for x, y in points:
        ix = int(np.clip(round(x), 0, w - 1))
        iy = int(np.clip(round(y), 0, h - 1))
        image[
            max(0, iy - radius):min(h, iy + radius + 1),
            max(0, ix - radius):min(w, ix + radius + 1),
        ] = color
    return image


def save_panel(resource, reference, resource_fields, reference_fields,
               path):
    import matplotlib.pyplot as plt

    from bfft.effects import lab_to_srgb

    reference_rgb = np.clip(
        lab_to_srgb(reference.reconstruction), 0.0, 1.0)
    figure, axes = plt.subplots(2, 5, figsize=(20, 8))
    products = [
        (resource.rgb, "target", None),
        (resource.rgb_reconstruction,
         f"resource ({resource.psnr:.2f} dB)", None),
        (reference_rgb, f"reference ({reference.psnr:.2f} dB)", None),
        (_site_overlay(
            resource.rgb_reconstruction, resource.centers,
            np.array([0.0, 1.0, 1.0])),
         "resource sites", None),
        (_site_overlay(
            reference_rgb, reference.seeds,
            np.array([1.0, 0.2, 0.0])),
         "reference sites", None),
        (resource_fields["effective_entropy"],
         "resource effective contributors", "magma"),
        (resource_fields["dominance"],
         "resource max dominance", "viridis"),
        (reference_fields["base_dominance"],
         "reference cartoon dominance", "viridis"),
        (reference_fields["detail_dominance"],
         "reference detail dominance", "viridis"),
        (_normalize(np.abs(reference.texture)),
         "BFFT texture magnitude", "magma"),
    ]
    for axis, (image, title, cmap) in zip(axes.ravel(), products):
        axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def load_image(path, gallery_key):
    if path:
        from skimage.io import imread
        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--initial-cells", type=int, default=180)
    parser.add_argument("--resource-rounds", type=int, default=30)
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/support_flow_comparison.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/support_flow_comparison.json")
    args = parser.parse_args()

    image, source = load_image(args.image, args.gallery)
    started = time.perf_counter()
    resource = ResourceTransportCells(
        image, ResourceConfig(
            max_side=args.side, cells=args.initial_cells))
    for _ in range(args.resource_rounds):
        resource.step()
    resource_seconds = time.perf_counter() - started
    matched_cells = len(resource.centers)

    reference_cfg = Config(
        max_side=args.side,
        passes=24,
        flow_sweeps=64,
        lam=0.05,
        mu=40.0,
        initial_cells=args.initial_cells,
        max_cells=matched_cells,
        split_batch=min(48, max(1, matched_cells - args.initial_cells)),
        anisotropy=5.0,
        edge_density=4.0,
        texture_density=3.0,
        edge_barrier=12.0,
        site_reach=1.5,
        softness=10.0,
        allocation_mode="Expected affine gain",
    )
    started = time.perf_counter()
    reference = ReceiverGuidedVoronoi(image, reference_cfg)
    while len(reference.seeds) < matched_cells:
        reference.step_direct(
            split=True, cartoon_softness=4.0,
            texture_softness=16.0)
        if reference.stagnation >= 3:
            break
    reference.solve_direct_coupled(
        cartoon_softness=4.0, texture_softness=16.0)
    reference_seconds = time.perf_counter() - started

    resource_fields = resource_support_fields(resource)
    reference_fields = reference_support_fields(reference)
    resource_summary = summarize_resource(
        resource, resource_fields, reference.texture_activity)
    reference_summary = summarize_reference(
        reference, reference_fields)
    report = {
        "source": source,
        "shape": [resource.h, resource.w],
        "resource_rounds": args.resource_rounds,
        "resource_seconds": resource_seconds,
        "reference_seconds": reference_seconds,
        "resource": resource_summary,
        "reference": reference_summary,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    save_panel(
        resource, reference, resource_fields, reference_fields,
        args.save)
    print(json.dumps(report, indent=2))
    print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
