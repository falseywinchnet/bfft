#!/usr/bin/env python3
"""Test a two-scale local resource for fine-cell germination.

Coarse support remains in the reconstruction occupancy denominator, but only
later/fine support inhibits another fine germ:

    resource_fine =
        error / total_occupancy / (1 + k * fine_occupancy).

This makes a resolved residual peak locally refractory at the same resolution
while leaving adjacent under-resolved support available.  It is a local
reaction law, not a selection pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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


def _support_stats(model, initial_cells, high_texture):
    samples = support_samples(model)
    weight, dominance, effective = aperture(samples, model.npix, 1.0)
    share = np.bincount(
        samples["rows"],
        weights=weight * (samples["sites"] < initial_cells),
        minlength=model.npix).reshape(model.h, model.w)
    fine_coverage = model.fine_occupancy > 1e-4
    return {
        "share": share,
        "initial_share_high_texture": float(np.mean(
            share[high_texture])),
        "fine_coverage": float(np.mean(fine_coverage)),
        "fine_coverage_high_texture": float(np.mean(
            fine_coverage[high_texture])),
        "dominance_mean": float(np.mean(dominance)),
        "effective_mean": float(np.mean(effective)),
    }


def _save_panel(models, records, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 8))
    for column, (model, record) in enumerate(zip(models, records)):
        axes[0, column].imshow(model.rgb_reconstruction)
        axes[0, column].set_title(
            f"fine inhibition {record['fine_inhibition']:g}\n"
            f"{record['psnr']:.2f} dB, {record['cells']} cells")
        axes[0, column].axis("off")
        axes[1, column].imshow(
            record["_share"], cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, column].set_title(
            "initial support share\n"
            f"detail {record['initial_share_high_texture']:.2f}, "
            f"fine cover {record['fine_coverage_high_texture']:.2f}")
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
    parser.add_argument("--birth-scale", type=float, default=0.25)
    parser.add_argument(
        "--inhibitions", type=float, nargs="*",
        default=(0.0, 1.0, 4.0, 12.0, 32.0))
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/resolution_selective.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/resolution_selective.json")
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    geometry = ReceiverGuidedVoronoi(
        image, Config(
            max_side=args.side, initial_cells=args.initial_cells,
            max_cells=args.initial_cells))
    high_texture = geometry.texture_activity >= np.percentile(
        geometry.texture_activity, 90.0)

    models = []
    records = []
    for inhibition in args.inhibitions:
        model = ResourceTransportCells(
            image, ResourceConfig(
                max_side=args.side, cells=args.initial_cells,
                germination_initial_scale=args.birth_scale,
                resolution_selective_germination=(
                    float(inhibition) > 0.0),
                fine_germination_inhibition=float(inhibition)))
        births = []
        for _ in range(args.rounds):
            births.append(int(model.step()["births"]))
        support = _support_stats(
            model, args.initial_cells, high_texture)
        objective = model.decomposition_metrics()
        record = {
            "fine_inhibition": float(inhibition),
            "cells": int(len(model.centers)),
            "birth_rounds": int(np.count_nonzero(births)),
            "largest_birth": int(max(births, default=0)),
            "psnr": float(model.psnr),
            "objective": {
                key: float(value) for key, value in objective.items()
            },
            "initial_share_high_texture": (
                support["initial_share_high_texture"]),
            "fine_coverage": support["fine_coverage"],
            "fine_coverage_high_texture": (
                support["fine_coverage_high_texture"]),
            "dominance_mean": support["dominance_mean"],
            "effective_mean": support["effective_mean"],
            "_share": support["share"],
        }
        models.append(model)
        records.append(record)
        print(
            f"inhibition {inhibition:g}: {model.psnr:.3f} dB, "
            f"{len(model.centers)} cells, detail old share "
            f"{record['initial_share_high_texture']:.3f}, "
            f"fine coverage {record['fine_coverage_high_texture']:.3f}",
            flush=True)

    report = {
        "source": source,
        "shape": [models[0].h, models[0].w],
        "rounds": args.rounds,
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
    _save_panel(models, records, args.save)
    print(json.dumps(report, indent=2))
    print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
