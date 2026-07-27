#!/usr/bin/env python3
"""Stage-by-stage support audit: resource cells versus the reference.

The study deliberately measures representation geometry rather than trusting
PSNR.  At matched population sizes it records:

* where sites are born relative to texture, cartoon edges, and current error;
* how much final support still comes from the initial coarse coat;
* the effective support radius over high-detail and low-detail pixels;
* contributors and dominance per pixel;
* site density and the support-age field through time.

The key diagnostic is ``initial_share``.  If old broad cells continue to
explain a high-detail pixel after fine cells germinate there, the fine cells
are not learning a local basis; they are merely perturbing an ancestral
mixture.  In an ownership-style refined partition, old support is displaced
locally by new sites.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _normalize(field, percentile=99.0):
    scale = max(float(np.percentile(field, percentile)), 1e-12)
    return np.clip(np.asarray(field, dtype=np.float64) / scale, 0.0, 1.0)


def _sample(field, points):
    h, w = field.shape
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, h - 1)
    return field[y, x]


def _birth_record(method, stage, points, edge, texture, error):
    if len(points) == 0:
        return None
    values = {}
    for name, field in (
        ("cartoon_edge", edge),
        ("texture", texture),
        ("prebirth_error", error),
    ):
        sample = _sample(field, points)
        order = np.sort(field.ravel())
        rank = np.searchsorted(order, sample, side="right") / order.size
        values[name] = {
            "mean": float(np.mean(sample)),
            "image_mean": float(np.mean(field)),
            "bias": float(
                np.mean(sample) / max(float(np.mean(field)), 1e-12)),
            "mean_percentile": float(np.mean(rank)),
        }
    return {
        "method": method,
        "stage": int(stage),
        "births": int(len(points)),
        "signals": values,
    }


def _site_density(points, h, w):
    impulses = np.zeros((h, w), dtype=np.float64)
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, h - 1)
    np.add.at(impulses, (y, x), 1.0)
    return ndi.gaussian_filter(impulses, 2.0, mode="reflect")


def _resource_snapshot(model, birth_round, initial_cells, stage):
    samples = support_samples(model)
    weight, dominance, effective = aperture(samples, model.npix, 1.0)
    initial = (samples["sites"] < initial_cells).astype(np.float64)
    initial_share = np.bincount(
        samples["rows"], weights=weight * initial,
        minlength=model.npix)
    radius = np.sqrt(model.major * model.minor)
    scale = np.bincount(
        samples["rows"],
        weights=weight * radius[samples["sites"]],
        minlength=model.npix)
    support_age = np.bincount(
        samples["rows"],
        weights=weight * birth_round[samples["sites"]],
        minlength=model.npix)
    return {
        "method": "resource",
        "stage": int(stage),
        "cells": int(len(model.centers)),
        "psnr": float(model.psnr),
        "sites": model.centers.copy(),
        "birth_round": birth_round.copy(),
        "rgb": model.rgb_reconstruction.copy(),
        "error": model.error.copy(),
        "initial_share": initial_share.reshape(model.h, model.w),
        "support_radius": scale.reshape(model.h, model.w),
        "support_age": support_age.reshape(model.h, model.w),
        "dominance": dominance.reshape(model.h, model.w),
        "effective": effective.reshape(model.h, model.w),
        "site_density": _site_density(
            model.centers, model.h, model.w),
    }


def _reference_snapshot(model, birth_round, initial_cells, stage):
    owner = model.owner.astype(np.int64)
    valid = model.second >= 0
    runner = np.where(valid, model.second, model.owner).astype(np.int64)
    gap = model.d2 - model.d1
    z = np.clip(0.5 * 16.0 * gap, -50.0, 50.0)
    first = 1.0 / (1.0 + np.exp(-z))
    first[~valid] = 1.0
    other = 1.0 - first
    area = np.bincount(owner, minlength=len(model.seeds)).astype(np.float64)
    radius = np.sqrt(np.maximum(area, 1.0) / math.pi)
    initial_share = (
        first * (owner < initial_cells) +
        other * (runner < initial_cells))
    scale = first * radius[owner] + other * radius[runner]
    support_age = (
        first * birth_round[owner] +
        other * birth_round[runner])
    dominance = np.maximum(first, other)
    effective = 1.0 / np.maximum(first * first + other * other, 1e-30)
    from bfft.effects import lab_to_srgb
    return {
        "method": "reference",
        "stage": int(stage),
        "cells": int(len(model.seeds)),
        "psnr": float(model.psnr),
        "sites": model.seeds.copy(),
        "birth_round": birth_round.copy(),
        "rgb": np.clip(lab_to_srgb(model.reconstruction), 0.0, 1.0),
        "error": model.error.copy(),
        "initial_share": initial_share.reshape(model.h, model.w),
        "support_radius": scale.reshape(model.h, model.w),
        "support_age": support_age.reshape(model.h, model.w),
        "dominance": dominance.reshape(model.h, model.w),
        "effective": effective.reshape(model.h, model.w),
        "site_density": _site_density(
            model.seeds, model.h, model.w),
    }


def _region_summary(snapshot, texture, edge):
    high_texture = texture >= np.percentile(texture, 90.0)
    low_texture = texture <= np.percentile(texture, 50.0)
    high_edge = edge >= np.percentile(edge, 90.0)
    sites = snapshot["sites"]

    def mean(field, mask):
        return float(np.mean(field[mask]))

    return {
        "stage": snapshot["stage"],
        "cells": snapshot["cells"],
        "psnr": snapshot["psnr"],
        "initial_share": {
            "all": float(np.mean(snapshot["initial_share"])),
            "high_texture": mean(
                snapshot["initial_share"], high_texture),
            "low_texture": mean(
                snapshot["initial_share"], low_texture),
            "high_edge": mean(snapshot["initial_share"], high_edge),
        },
        "support_radius": {
            "all": float(np.mean(snapshot["support_radius"])),
            "high_texture": mean(
                snapshot["support_radius"], high_texture),
            "low_texture": mean(
                snapshot["support_radius"], low_texture),
            "high_to_low": (
                mean(snapshot["support_radius"], high_texture) /
                max(mean(
                    snapshot["support_radius"], low_texture), 1e-12)),
        },
        "support_age": {
            "all": float(np.mean(snapshot["support_age"])),
            "high_texture": mean(snapshot["support_age"], high_texture),
            "low_texture": mean(snapshot["support_age"], low_texture),
        },
        "dominance": {
            "mean": float(np.mean(snapshot["dominance"])),
            "ambiguous_below_0_8": float(np.mean(
                snapshot["dominance"] < 0.8)),
        },
        "effective_contributors_mean": float(np.mean(
            snapshot["effective"])),
        "site_texture_bias": float(
            np.mean(_sample(texture, sites)) /
            max(float(np.mean(texture)), 1e-12)),
        "site_edge_bias": float(
            np.mean(_sample(edge, sites)) /
            max(float(np.mean(edge)), 1e-12)),
    }


def _site_overlay(rgb, snapshot):
    image = rgb.copy()
    points = snapshot["sites"]
    ages = snapshot["birth_round"]
    maximum = max(float(np.max(ages)), 1.0)
    for (x, y), age in zip(points, ages):
        ix = int(np.clip(round(x), 0, image.shape[1] - 1))
        iy = int(np.clip(round(y), 0, image.shape[0] - 1))
        t = float(age) / maximum
        color = np.array([t, 1.0 - t, 1.0])
        image[
            max(0, iy - 1):min(image.shape[0], iy + 2),
            max(0, ix - 1):min(image.shape[1], ix + 2),
        ] = color
    return image


def _save_trajectory(target, resource, reference, path):
    import matplotlib.pyplot as plt

    columns = max(len(resource), len(reference))
    figure, axes = plt.subplots(4, columns, figsize=(3.2 * columns, 12))
    for column in range(columns):
        for row, snapshots in ((0, resource), (2, reference)):
            if column >= len(snapshots):
                axes[row, column].axis("off")
                axes[row + 1, column].axis("off")
                continue
            snapshot = snapshots[column]
            axes[row, column].imshow(
                _site_overlay(snapshot["rgb"], snapshot))
            axes[row, column].set_title(
                f"{snapshot['method']} {snapshot['cells']} cells\n"
                f"{snapshot['psnr']:.2f} dB")
            axes[row, column].axis("off")
            axes[row + 1, column].imshow(
                snapshot["initial_share"], cmap="magma",
                vmin=0.0, vmax=1.0)
            axes[row + 1, column].set_title("initial-coat support share")
            axes[row + 1, column].axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _save_final(target, texture, edge, resource, reference, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 6, figsize=(20, 7))
    for row, snapshot in enumerate((resource, reference)):
        products = (
            (snapshot["rgb"], f"{snapshot['method']} reconstruction", None),
            (_normalize(snapshot["error"]), "remaining error", "magma"),
            (snapshot["initial_share"], "initial-coat share", "magma"),
            (snapshot["support_radius"], "effective support radius", "viridis"),
            (_normalize(snapshot["site_density"]), "site density", "magma"),
            (snapshot["dominance"], "max support dominance", "viridis"),
        )
        for axis, (image, title, cmap) in zip(axes[row], products):
            axis.imshow(image, cmap=cmap)
            axis.set_title(title)
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
        "--checkpoints", type=int, nargs="*", default=(0, 1, 3, 8, 15, 30))
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/support_trajectory.png")
    parser.add_argument(
        "--final-save", type=Path,
        default=ROOT / "experiments/out/support_trajectory_final.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/support_trajectory.json")
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    checkpoints = sorted(set(
        int(np.clip(value, 0, args.resource_rounds))
        for value in args.checkpoints))

    resource = ResourceTransportCells(
        image, ResourceConfig(
            max_side=args.side, cells=args.initial_cells))
    resource_birth_round = np.zeros(
        len(resource.centers), dtype=np.float64)
    resource_snapshots = []
    resource_births = []
    if 0 in checkpoints:
        resource_snapshots.append(_resource_snapshot(
            resource, resource_birth_round, args.initial_cells, 0))
    for stage in range(1, args.resource_rounds + 1):
        old_count = len(resource.centers)
        pre_error = resource.error.copy()
        resource.step()
        new_count = len(resource.centers)
        if new_count > old_count:
            resource_birth_round = np.concatenate([
                resource_birth_round,
                np.full(new_count - old_count, stage, dtype=np.float64)])
            resource_births.append((stage, old_count, new_count, pre_error))
        if stage in checkpoints:
            resource_snapshots.append(_resource_snapshot(
                resource, resource_birth_round,
                args.initial_cells, stage))

    final_cells = len(resource.centers)
    reference = ReceiverGuidedVoronoi(
        image, Config(
            max_side=args.side, initial_cells=args.initial_cells,
            max_cells=final_cells, split_batch=24,
            passes=24, flow_sweeps=64, lam=0.05, mu=40.0,
            anisotropy=5.0, edge_density=4.0, texture_density=3.0,
            edge_barrier=12.0, site_reach=1.5,
            allocation_mode="Expected affine gain"))
    reference.solve_direct_coupled(4.0, 16.0)
    edge = reference.edge_strength.copy()
    texture = reference.texture_activity.copy()

    reference_birth_round = np.zeros(
        len(reference.seeds), dtype=np.float64)
    reference_snapshots = []
    reference_births = []
    target_counts = [snapshot["cells"] for snapshot in resource_snapshots]
    if target_counts and target_counts[0] == len(reference.seeds):
        reference_snapshots.append(_reference_snapshot(
            reference, reference_birth_round,
            args.initial_cells, 0))
        target_counts = target_counts[1:]
    reference_stage = 0
    for target_count in target_counts:
        while len(reference.seeds) < target_count:
            old_count = len(reference.seeds)
            pre_error = reference.error.copy()
            reference.cfg.split_batch = target_count - old_count
            reference.step_direct(True, 4.0, 16.0)
            reference_stage += 1
            new_count = len(reference.seeds)
            if new_count <= old_count:
                break
            reference_birth_round = np.concatenate([
                reference_birth_round,
                np.full(
                    new_count - old_count, reference_stage,
                    dtype=np.float64)])
            reference_births.append((
                reference_stage, old_count, new_count, pre_error))
        reference_snapshots.append(_reference_snapshot(
            reference, reference_birth_round,
            args.initial_cells, reference_stage))

    birth_records = []
    for method, snapshots, births in (
        ("resource", resource_snapshots, resource_births),
        ("reference", reference_snapshots, reference_births),
    ):
        site_source = (
            resource.centers if method == "resource" else reference.seeds)
        for stage, old_count, new_count, pre_error in births:
            record = _birth_record(
                method, stage, site_source[old_count:new_count],
                edge, texture, pre_error)
            if record is not None:
                birth_records.append(record)

    report = {
        "source": source,
        "shape": [resource.h, resource.w],
        "initial_cells": args.initial_cells,
        "final_cells": final_cells,
        "resource": [
            _region_summary(snapshot, texture, edge)
            for snapshot in resource_snapshots
        ],
        "reference": [
            _region_summary(snapshot, texture, edge)
            for snapshot in reference_snapshots
        ],
        "births": birth_records,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    _save_trajectory(
        resource.rgb, resource_snapshots, reference_snapshots,
        args.save)
    _save_final(
        resource.rgb, texture, edge,
        resource_snapshots[-1], reference_snapshots[-1],
        args.final_save)
    print(json.dumps(report, indent=2))
    print(f"saved {args.save}")
    print(f"saved {args.final_save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
