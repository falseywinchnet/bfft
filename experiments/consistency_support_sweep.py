#!/usr/bin/env python3
"""Redistribute initial support area by local gradient consistency.

Total geometric support area is conserved.  Uniform sites on coherent,
energetic structure receive tighter initial supports; sites in incoherent or
flat regions receive broader supports:

    pressure = coherence(J_grad) * normalized_trace(J_grad)
    area_i ∝ exp(-gain * pressure_i).

The factors are normalized to geometric mean one, so the trial does not win
by globally changing overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from dual_aperture_support import aperture, support_samples  # noqa: E402
from receiver_guided_graph import ReceiverGuidedVoronoi  # noqa: E402
from resource_transport_cells import (  # noqa: E402
    ResourceConfig, ResourceTransportCells,
)
from transport_voronoi import Config  # noqa: E402


def _load_image(path, gallery_key):
    if path:
        from skimage.io import imread
        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def _consistency_pressure(rgb):
    light = (
        0.2126 * rgb[..., 0] +
        0.7152 * rgb[..., 1] +
        0.0722 * rgb[..., 2])
    gx = ndi.sobel(light, axis=1, mode="reflect") / 8.0
    gy = ndi.sobel(light, axis=0, mode="reflect") / 8.0
    jxx = ndi.gaussian_filter(gx * gx, 1.4, mode="reflect")
    jyy = ndi.gaussian_filter(gy * gy, 1.4, mode="reflect")
    jxy = ndi.gaussian_filter(gx * gy, 1.4, mode="reflect")
    trace = jxx + jyy
    disc = np.sqrt(np.maximum(
        (jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    coherence = disc / np.maximum(trace, 1e-12)
    scale = max(float(np.percentile(trace, 95.0)), 1e-12)
    energy = np.clip(trace / scale, 0.0, 1.0)
    return coherence * np.sqrt(energy)


def _sample(field, points):
    x = np.clip(
        np.rint(points[:, 0]).astype(np.int64), 0, field.shape[1] - 1)
    y = np.clip(
        np.rint(points[:, 1]).astype(np.int64), 0, field.shape[0] - 1)
    return field[y, x]


def _support_stats(model, initial_cells, high_texture):
    samples = support_samples(model)
    weight, dominance, effective = aperture(samples, model.npix, 1.0)
    share = np.bincount(
        samples["rows"],
        weights=weight * (samples["sites"] < initial_cells),
        minlength=model.npix).reshape(model.h, model.w)
    return (
        share, float(np.mean(share[high_texture])),
        float(np.mean(dominance)), float(np.mean(effective)))


def _save_panel(models, records, pressure, path):
    import matplotlib.pyplot as plt

    columns = len(models)
    figure, axes = plt.subplots(2, columns, figsize=(4 * columns, 8))
    for column, (model, record) in enumerate(zip(models, records)):
        axes[0, column].imshow(model.rgb_reconstruction)
        axes[0, column].set_title(
            f"consistency gain {record['gain']:g}\n"
            f"{record['psnr']:.2f} dB, {record['cells']} cells")
        axes[0, column].axis("off")
        if column == 0:
            axes[1, column].imshow(pressure, cmap="magma")
            axes[1, column].set_title("consistency × energy")
        else:
            axes[1, column].imshow(
                record["_share"], cmap="magma", vmin=0.0, vmax=1.0)
            axes[1, column].set_title(
                "initial support share\n"
                f"detail {record['initial_share_high_texture']:.2f}")
        axes[1, column].axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--initial-cells", type=int, default=180)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--initial-overlap", type=float, default=5.0)
    parser.add_argument("--birth-scale", type=float, default=0.25)
    parser.add_argument(
        "--gains", type=float, nargs="*",
        default=(0.0, 0.5, 1.0, 2.0, 3.0))
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/consistency_support.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/consistency_support.json")
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    probe = ResourceTransportCells(
        image, ResourceConfig(
            max_side=args.side, cells=args.initial_cells,
            initial_overlap=args.initial_overlap,
            germination_initial_scale=args.birth_scale))
    pressure = _consistency_pressure(probe.rgb)
    geometry = ReceiverGuidedVoronoi(
        image, Config(
            max_side=args.side, initial_cells=args.initial_cells,
            max_cells=args.initial_cells))
    high_texture = geometry.texture_activity >= np.percentile(
        geometry.texture_activity, 90.0)

    models = []
    records = []
    for gain in args.gains:
        model = ResourceTransportCells(
            image, ResourceConfig(
                max_side=args.side, cells=args.initial_cells,
                initial_overlap=args.initial_overlap,
                germination_initial_scale=args.birth_scale))
        site_pressure = _sample(pressure, model.centers)
        log_area_factor = -float(gain) * site_pressure
        log_area_factor -= float(np.mean(log_area_factor))
        area_factor = np.exp(np.clip(log_area_factor, -2.0, 2.0))
        radius_factor = np.sqrt(area_factor)
        model.major *= radius_factor
        model.minor *= radius_factor
        model._render()
        for _ in range(args.rounds):
            model.step()
        share, high_share, dominance, effective = _support_stats(
            model, args.initial_cells, high_texture)
        objective = model.decomposition_metrics()
        record = {
            "gain": float(gain),
            "cells": int(len(model.centers)),
            "psnr": float(model.psnr),
            "objective": {
                key: float(value) for key, value in objective.items()
            },
            "initial_share_high_texture": high_share,
            "dominance_mean": dominance,
            "effective_mean": effective,
            "initial_area_factor": {
                "p10": float(np.percentile(area_factor, 10.0)),
                "p50": float(np.percentile(area_factor, 50.0)),
                "p90": float(np.percentile(area_factor, 90.0)),
            },
            "_share": share,
        }
        models.append(model)
        records.append(record)
        print(
            f"gain {gain:g}: {model.psnr:.3f} dB, "
            f"{len(model.centers)} cells, detail old share "
            f"{high_share:.3f}",
            flush=True)

    report = {
        "source": source,
        "shape": [models[0].h, models[0].w],
        "rounds": args.rounds,
        "initial_overlap": args.initial_overlap,
        "birth_scale": args.birth_scale,
        "results": [
            {
                key: value for key, value in record.items()
                if not key.startswith("_")
            }
            for record in records
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    _save_panel(models, records, pressure, args.save)
    print(json.dumps(report, indent=2))
    print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
