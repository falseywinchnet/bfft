#!/usr/bin/env python3
"""Learn cell existence as a continuous support activity.

At birth a copied atom has zero support derivative because p_new == I.  This
experiment instead stores a target-directed affine hypothesis at low activity
and learns log(activity) from the exact receiver derivative

    d I / d log(activity_i)
        = phi_i (p_i - I) / Z.

Initial cells are held at activity one.  New cells can grow toward the target
and locally starve the old coat, or remain nearly latent when their hypothesis
is not useful.  No owner, deletion, candidate ranking, or site-pair graph is
introduced.
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
    return (
        share, float(np.mean(share[high_texture])),
        float(np.mean(dominance)), float(np.mean(effective)))


def _save_panel(models, records, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 8))
    for column, (model, record) in enumerate(zip(models, records)):
        axes[0, column].imshow(model.rgb_reconstruction)
        axes[0, column].set_title(
            f"{record['name']}\n{record['psnr']:.2f} dB, "
            f"{record['cells']} cells")
        axes[0, column].axis("off")
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
    parser.add_argument("--birth-scale", type=float, default=0.25)
    parser.add_argument(
        "--trials", nargs="*", default=(
            "control", "0.02:0.3", "0.05:0.3",
            "0.1:0.3", "0.05:0.6"))
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/support_activity.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/support_activity.json")
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
    for trial in args.trials:
        if trial == "control":
            adaptive = False
            birth_activity = 1.0
            activity_rate = 0.0
            name = "control"
        else:
            birth_activity, activity_rate = (
                float(value) for value in trial.split(":", 1))
            adaptive = True
            name = f"a0={birth_activity:g}, rate={activity_rate:g}"
        model = ResourceTransportCells(
            image, ResourceConfig(
                max_side=args.side, cells=args.initial_cells,
                germination_initial_scale=args.birth_scale,
                adaptive_activity=adaptive,
                birth_activity=birth_activity,
                activity_rate=activity_rate,
                birth_target_hypothesis=adaptive,
                lock_initial_activity=True))
        for _ in range(args.rounds):
            model.step()
        share, high_share, dominance, effective = _support_stats(
            model, args.initial_cells, high_texture)
        born_activity = model.activity[args.initial_cells:]
        objective = model.decomposition_metrics()
        record = {
            "name": name,
            "adaptive_activity": adaptive,
            "birth_activity": float(birth_activity),
            "activity_rate": float(activity_rate),
            "cells": int(len(model.centers)),
            "psnr": float(model.psnr),
            "objective": {
                key: float(value) for key, value in objective.items()
            },
            "initial_share_high_texture": high_share,
            "dominance_mean": dominance,
            "effective_mean": effective,
            "born_activity": {
                "p10": (
                    float(np.percentile(born_activity, 10.0))
                    if len(born_activity) else 0.0),
                "p50": (
                    float(np.percentile(born_activity, 50.0))
                    if len(born_activity) else 0.0),
                "p90": (
                    float(np.percentile(born_activity, 90.0))
                    if len(born_activity) else 0.0),
                "max": (
                    float(np.max(born_activity))
                    if len(born_activity) else 0.0),
            },
            "_share": share,
        }
        models.append(model)
        records.append(record)
        print(
            f"{name}: {model.psnr:.3f} dB, "
            f"{len(model.centers)} cells, detail old share "
            f"{high_share:.3f}, activity p50 "
            f"{record['born_activity']['p50']:.3f}",
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
