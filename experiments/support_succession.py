#!/usr/bin/env python3
"""Test whether new resource cells must locally displace the coarse coat.

The trajectory audit shows that the original 180 cells still contribute more
than 80% of the normalized support in high-texture regions after 278 births.
This frozen-geometry experiment changes only that support age balance.

For raw compact support phi_i and a site gain g_i,

    w_i(x) = g_i phi_i(x) / sum_j g_j phi_j(x).

The control has g_i=1.  Succession uses g_i=r for the initial coat and one for
all later germs.  This is a smooth local attenuation: where no later support
exists, the common factor r cancels and the initial coat remains a complete
partition.  No site is deleted and no pixel receives an owner.

Two fits are measured:

1. direct: the age-weighted support fits the whole target;
2. layered: all supports fit the BFFT cartoon field, while succession applies
   only to the BFFT detail field.
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
from bfft.vision import SingleStageDecompositionObjective  # noqa: E402
from dual_aperture_support import (  # noqa: E402
    aperture, design_matrix, score, solve_field, support_samples,
    target_layers,
)
from resource_transport_cells import (  # noqa: E402
    ResourceConfig, ResourceTransportCells,
)


def _load_image(path, gallery_key):
    if path:
        from skimage.io import imread
        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def _fit(samples, target, objective, rgb, gain):
    weight, dominance, effective = aperture(
        samples, target.shape[0] * target.shape[1],
        1.0, site_gain=gain)
    design = design_matrix(
        samples, target.shape[0] * target.shape[1], weight)
    field = solve_field(design, target)
    return (
        score(objective, rgb, field),
        field, dominance.reshape(target.shape[:2]),
        effective.reshape(target.shape[:2]),
        weight,
    )


def _initial_share(samples, weight, initial_cells, shape):
    value = np.bincount(
        samples["rows"],
        weights=weight * (samples["sites"] < initial_cells),
        minlength=shape[0] * shape[1])
    return value.reshape(shape)


def _save_panel(model, results, path):
    import matplotlib.pyplot as plt

    columns = 2 + len(results)
    figure, axes = plt.subplots(2, columns, figsize=(4 * columns, 8))
    axes[0, 0].imshow(model.rgb)
    axes[0, 0].set_title("target")
    axes[0, 1].imshow(model.rgb_reconstruction)
    axes[0, 1].set_title(f"live local fit\n{model.psnr:.2f} dB")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    for column, result in enumerate(results, start=2):
        axes[0, column].imshow(result["layered"]["rgb"])
        axes[0, column].set_title(
            f"old gain {result['old_gain']:g}\n"
            f"layered {result['layered']['psnr']:.2f} dB")
        axes[1, column].imshow(
            result["initial_share"], cmap="magma",
            vmin=0.0, vmax=1.0)
        axes[1, column].set_title(
            "detail initial share\n"
            f"mean {np.mean(result['initial_share']):.2f}")
        axes[1, column].axis("off")
    for axis in axes[0]:
        axis.axis("off")
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
    parser.add_argument("--resource-rounds", type=int, default=30)
    parser.add_argument(
        "--old-gains", type=float, nargs="*",
        default=(1.0, 0.5, 0.25, 0.1, 0.03))
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/support_succession.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/support_succession.json")
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    model = ResourceTransportCells(
        image, ResourceConfig(
            max_side=args.side, cells=args.initial_cells))
    for _ in range(args.resource_rounds):
        model.step()
    samples = support_samples(model)
    lab, base_target, detail_target = target_layers(model.rgb)
    objective = SingleStageDecompositionObjective(model.rgb)

    all_gain = np.ones(len(model.centers), dtype=np.float64)
    base_record, base_field, _, _, _ = _fit(
        samples, base_target, objective, model.rgb, all_gain)

    results = []
    serializable = []
    for old_gain in args.old_gains:
        gain = np.ones(len(model.centers), dtype=np.float64)
        gain[:args.initial_cells] = float(old_gain)
        direct, _direct_field, _direct_dom, _direct_eff, _ = _fit(
            samples, lab, objective, model.rgb, gain)
        detail, detail_field, dominance, effective, weight = _fit(
            samples, detail_target, objective, model.rgb, gain)
        layered_field = base_field + detail_field
        layered = score(
            objective, model.rgb, layered_field)
        initial_share = _initial_share(
            samples, weight, args.initial_cells, model.rgb.shape[:2])
        results.append({
            "old_gain": float(old_gain),
            "direct": direct,
            "detail": detail,
            "layered": layered,
            "initial_share": initial_share,
            "dominance": dominance,
            "effective": effective,
        })
        serializable.append({
            "old_gain": float(old_gain),
            "direct": {
                key: float(value) for key, value in direct.items()
                if key != "rgb"
            },
            "layered": {
                key: float(value) for key, value in layered.items()
                if key != "rgb"
            },
            "initial_share": {
                "mean": float(np.mean(initial_share)),
                "p10": float(np.percentile(initial_share, 10.0)),
                "p50": float(np.percentile(initial_share, 50.0)),
                "p90": float(np.percentile(initial_share, 90.0)),
            },
            "dominance_mean": float(np.mean(dominance)),
            "effective_mean": float(np.mean(effective)),
        })
        print(
            f"old gain {old_gain:g}: "
            f"direct {direct['psnr']:.3f}, "
            f"layered {layered['psnr']:.3f}, "
            f"initial share {np.mean(initial_share):.3f}",
            flush=True)

    report = {
        "source": source,
        "shape": [model.h, model.w],
        "cells": int(len(model.centers)),
        "live_local_psnr": float(model.psnr),
        "base_fit": {
            key: float(value) for key, value in base_record.items()
            if key != "rgb"
        },
        "results": serializable,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    _save_panel(model, results, args.save)
    print(json.dumps(report, indent=2))
    print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
