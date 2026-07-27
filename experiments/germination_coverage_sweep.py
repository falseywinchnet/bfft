#!/usr/bin/env python3
"""Sweep newborn support scale after the support-trajectory diagnosis.

New resource cells currently start at 15% of the initial radius.  On Pikachu,
278 later cells collectively own only 6% as much raw geometric area as the
180-cell initial coat.  This experiment changes only the birth radius.

New cells still inherit the existing rendered plane, so a larger footprint is
initially invisible and may paint only after a measured residual update.
There is no ranked allocation, ownership, or deletion.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _initial_share(model, initial_cells):
    samples = support_samples(model)
    weight, dominance, effective = aperture(
        samples, model.npix, 1.0)
    share = np.bincount(
        samples["rows"],
        weights=weight * (samples["sites"] < initial_cells),
        minlength=model.npix).reshape(model.h, model.w)
    return share, dominance.reshape(model.h, model.w), effective


def _save_panel(models, records, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 8))
    for column, (model, record) in enumerate(zip(models, records)):
        axes[0, column].imshow(model.rgb_reconstruction)
        axes[0, column].set_title(
            f"birth radius {record['birth_scale']:g}\n"
            f"{record['psnr']:.2f} dB, {record['cells']} cells")
        axes[0, column].axis("off")
        axes[1, column].imshow(
            record["_initial_share"], cmap="magma",
            vmin=0.0, vmax=1.0)
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
    parser.add_argument(
        "--birth-scales", type=float, nargs="*",
        default=(0.15, 0.25, 0.4, 0.6))
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/germination_coverage.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/germination_coverage.json")
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    geometry = ReceiverGuidedVoronoi(
        image, Config(
            max_side=args.side, initial_cells=args.initial_cells,
            max_cells=args.initial_cells))
    texture = geometry.texture_activity
    high_texture = texture >= np.percentile(texture, 90.0)

    models = []
    records = []
    for birth_scale in args.birth_scales:
        model = ResourceTransportCells(
            image, ResourceConfig(
                max_side=args.side, cells=args.initial_cells,
                germination_initial_scale=float(birth_scale)))
        for _ in range(args.rounds):
            model.step()
        share, dominance, effective = _initial_share(
            model, args.initial_cells)
        old_area = float(np.sum(
            math.pi * model.major[:args.initial_cells] *
            model.minor[:args.initial_cells]))
        new_area = float(np.sum(
            math.pi * model.major[args.initial_cells:] *
            model.minor[args.initial_cells:]))
        objective = model.decomposition_metrics()
        record = {
            "birth_scale": float(birth_scale),
            "cells": int(len(model.centers)),
            "psnr": float(model.psnr),
            "objective": {
                key: float(value) for key, value in objective.items()
            },
            "initial_share_mean": float(np.mean(share)),
            "initial_share_high_texture": float(np.mean(
                share[high_texture])),
            "dominance_mean": float(np.mean(dominance)),
            "effective_mean": float(np.mean(effective)),
            "new_to_old_geometric_area": new_area / max(old_area, 1e-12),
            "new_radius": {
                "p10": float(np.percentile(
                    np.sqrt(
                        model.major[args.initial_cells:] *
                        model.minor[args.initial_cells:]), 10.0)),
                "p50": float(np.percentile(
                    np.sqrt(
                        model.major[args.initial_cells:] *
                        model.minor[args.initial_cells:]), 50.0)),
                "p90": float(np.percentile(
                    np.sqrt(
                        model.major[args.initial_cells:] *
                        model.minor[args.initial_cells:]), 90.0)),
            },
            "_initial_share": share,
        }
        models.append(model)
        records.append(record)
        print(
            f"birth {birth_scale:g}: {model.psnr:.3f} dB, "
            f"{len(model.centers)} cells, detail old share "
            f"{record['initial_share_high_texture']:.3f}, "
            f"new/old area {record['new_to_old_geometric_area']:.3f}",
            flush=True)

    report = {
        "source": source,
        "shape": [models[0].h, models[0].w],
        "rounds": args.rounds,
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
