#!/usr/bin/env python3
"""Freeze resource-cell geometry and test independent detail apertures.

This is a support-law diagnostic, not a proposed final global solver.  It
answers two questions without changing one cell center or ellipse:

1. How much error comes from local block-Jacobi coefficient fitting?
2. How much detail returns when cartoon and texture use different normalized
   powers of the same owner-free supports?

For compact support ``phi_i``:

    w_i^(tau)(x) = phi_i(x)^tau / sum_j phi_j(x)^tau.

``tau=1`` is the current broad aperture.  Larger texture ``tau`` approaches a
winner-take-most detail aperture continuously, without assigning an owner.
The affine basis uses the current population spacing, matching the known-good
reference instead of retaining the initial 180-cell scale forever.
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
from scipy import sparse
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import bfft  # noqa: E402
import gallery  # noqa: E402
from bfft.effects import lab_to_srgb  # noqa: E402
from bfft.vision import SingleStageDecompositionObjective  # noqa: E402
from resource_transport_cells import (  # noqa: E402
    ResourceConfig, ResourceTransportCells,
)
from transport_voronoi import srgb_to_lab  # noqa: E402


def target_layers(rgb):
    lab = srgb_to_lab(rgb)
    light = lab[..., 0] * 255.0
    cartoon, _ = bfft.meyer_split(
        light, lam=0.05, mu=40.0, passes=24, threads=4)
    base = lab.copy()
    base[..., 0] = np.clip(cartoon / 255.0, 0.0, 1.0)
    base[..., 1] = ndi.gaussian_filter(
        lab[..., 1], 2.0, mode="reflect")
    base[..., 2] = ndi.gaussian_filter(
        lab[..., 2], 2.0, mode="reflect")
    return lab, base, lab - base


def support_samples(model, spacing=None):
    """Return sparse support samples and current-scale affine bases."""
    n = len(model.centers)
    spacing = max(
        math.sqrt(model.npix / max(n, 1))
        if spacing is None else float(spacing),
        1e-9)
    rows = []
    sites = []
    phis = []
    bases = []
    for site in range(n):
        patch = model._patch(site)
        if patch is None:
            continue
        y0, y1, x0, x1, dx, dy, _, phi = patch
        yy, xx = np.mgrid[y0:y1, x0:x1]
        visible = phi > 1e-9
        if not np.any(visible):
            continue
        rows.append((yy[visible] * model.w + xx[visible]).astype(np.int32))
        sites.append(np.full(np.sum(visible), site, dtype=np.int32))
        phis.append(phi[visible])
        bases.append(np.column_stack([
            np.ones(np.sum(visible), dtype=np.float64),
            np.broadcast_to(dx, phi.shape)[visible] / spacing,
            np.broadcast_to(dy, phi.shape)[visible] / spacing,
        ]))
    return {
        "rows": np.concatenate(rows),
        "sites": np.concatenate(sites),
        "phi": np.concatenate(phis),
        "basis": np.concatenate(bases, axis=0),
        "cells": n,
        "spacing": spacing,
    }


def aperture(samples, npix, temperature, site_gain=None):
    power = np.power(
        np.maximum(samples["phi"], 1e-30), float(temperature))
    if site_gain is not None:
        power *= np.asarray(site_gain)[samples["sites"]]
    denominator = np.bincount(
        samples["rows"], weights=power, minlength=npix)
    weight = power / np.maximum(denominator[samples["rows"]], 1e-30)
    dominance = np.zeros(npix, dtype=np.float64)
    np.maximum.at(dominance, samples["rows"], weight)
    effective_denominator = np.bincount(
        samples["rows"], weights=weight * weight, minlength=npix)
    effective = 1.0 / np.maximum(effective_denominator, 1e-30)
    return weight, dominance, effective


def design_matrix(samples, npix, weight):
    width = 3
    rows = np.repeat(samples["rows"], width)
    parts = np.tile(np.arange(width, dtype=np.int32), len(weight))
    columns = width * np.repeat(samples["sites"], width) + parts
    data = (samples["basis"] * weight[:, None]).ravel()
    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(npix, width * samples["cells"])).tocsr()


def solve_field(design, target):
    nvars = design.shape[1]
    regularization = np.tile(
        np.array([1e-5, 2e-3, 2e-3], dtype=np.float64),
        nvars // 3)
    gram = (design.T @ design).tocsc()
    gram += sparse.diags(regularization, format="csc")
    factor = splu(
        gram, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
        options={"SymmetricMode": True})
    rhs = design.T @ target.reshape(-1, 3)
    coeff = factor.solve(rhs)
    return (design @ coeff).reshape(target.shape)


def score(objective, target_rgb, reconstruction):
    rgb = np.clip(lab_to_srgb(reconstruction), 0.0, 1.0)
    record = objective.evaluate(rgb)
    record["rgb"] = rgb
    return record


def _serializable(record):
    return {
        key: float(value)
        for key, value in record.items()
        if key != "rgb"
    }


def save_panel(target, current, single, variants, dominance, path):
    import matplotlib.pyplot as plt

    columns = 3 + len(variants)
    figure, axes = plt.subplots(2, columns, figsize=(4 * columns, 8))
    top = [
        (target, "target"),
        (current["rgb"], f"local fit {current['psnr']:.2f} dB"),
        (single["rgb"], f"global tau=1 {single['psnr']:.2f} dB"),
    ] + [
        (record["rgb"], f"detail tau={tau:g}\\n{record['psnr']:.2f} dB")
        for tau, record in variants
    ]
    for axis, (image, title) in zip(axes[0], top):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    diagnostic = [
        (np.zeros(target.shape[:2]), "local broad aperture"),
        (dominance[1.0].reshape(target.shape[:2]), "tau=1 dominance"),
        (dominance[1.0].reshape(target.shape[:2]),
         "single-fit dominance"),
    ] + [
        (dominance[tau].reshape(target.shape[:2]),
         f"detail tau={tau:g} dominance")
        for tau, _ in variants
    ]
    for axis, (image, title) in zip(axes[1], diagnostic):
        axis.imshow(image, cmap="viridis", vmin=0.0, vmax=1.0)
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
        "--detail-temperatures", type=float, nargs="+",
        default=[2.0, 4.0, 8.0, 16.0])
    parser.add_argument(
        "--fine-scale-gammas", type=float, nargs="+",
        default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/dual_aperture_support.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/dual_aperture_support.json")
    args = parser.parse_args()

    image, source = load_image(args.image, args.gallery)
    model = ResourceTransportCells(
        image, ResourceConfig(
            max_side=args.side, cells=args.initial_cells))
    for _ in range(args.resource_rounds):
        model.step()
    objective = SingleStageDecompositionObjective(model.rgb)
    current = score(objective, model.rgb, model.reconstruction)

    target, base, detail = target_layers(model.rgb)
    samples = support_samples(model)
    initial_scale_samples = support_samples(
        model, spacing=model.spacing)
    dominance = {}
    base_weight, dominance[1.0], effective = aperture(
        samples, model.npix, 1.0)
    base_design = design_matrix(samples, model.npix, base_weight)
    single_field = solve_field(base_design, target)
    single = score(objective, model.rgb, single_field)
    initial_weight, _, _ = aperture(
        initial_scale_samples, model.npix, 1.0)
    initial_design = design_matrix(
        initial_scale_samples, model.npix, initial_weight)
    initial_scale_field = solve_field(initial_design, target)
    initial_scale_global = score(
        objective, model.rgb, initial_scale_field)
    base_field = solve_field(base_design, base)

    variants = []
    started = time.perf_counter()
    for temperature in args.detail_temperatures:
        detail_weight, dominance[temperature], _ = aperture(
            samples, model.npix, temperature)
        detail_design = design_matrix(
            samples, model.npix, detail_weight)
        detail_field = solve_field(detail_design, detail)
        record = score(
            objective, model.rgb, base_field + detail_field)
        variants.append((temperature, record))
    areas = math.pi * model.major * model.minor
    median_area = max(float(np.median(areas)), 1e-12)
    scale_variants = []
    for gamma in args.fine_scale_gammas:
        detail_gain = np.clip(
            np.power(median_area / np.maximum(areas, 1e-12), gamma),
            0.02, 50.0)
        detail_weight, scale_dominance, _ = aperture(
            samples, model.npix, 1.0, site_gain=detail_gain)
        detail_design = design_matrix(
            samples, model.npix, detail_weight)
        detail_field = solve_field(detail_design, detail)
        record = score(
            objective, model.rgb, base_field + detail_field)
        scale_variants.append((
            gamma, record, scale_dominance, detail_gain))
    paired_scale_variants = []
    for gamma in args.fine_scale_gammas:
        coarse_gain = np.clip(
            np.power(np.maximum(areas, 1e-12) / median_area, gamma),
            0.02, 50.0)
        fine_gain = np.clip(
            np.power(median_area / np.maximum(areas, 1e-12), gamma),
            0.02, 50.0)
        coarse_weight, coarse_dominance, _ = aperture(
            samples, model.npix, 1.0, site_gain=coarse_gain)
        fine_weight, fine_dominance, _ = aperture(
            samples, model.npix, 1.0, site_gain=fine_gain)
        coarse_field = solve_field(
            design_matrix(samples, model.npix, coarse_weight), base)
        fine_field = solve_field(
            design_matrix(samples, model.npix, fine_weight), detail)
        record = score(
            objective, model.rgb, coarse_field + fine_field)
        paired_scale_variants.append((
            gamma, record, coarse_dominance, fine_dominance))
    solve_seconds = time.perf_counter() - started

    report = {
        "source": source,
        "shape": [model.h, model.w],
        "cells": int(len(model.centers)),
        "resource_rounds": args.resource_rounds,
        "current_local": _serializable(current),
        "single_aperture_global": _serializable(single),
        "single_aperture_global_initial_scale": _serializable(
            initial_scale_global),
        "tau1_effective_contributors": {
            "mean": float(np.mean(effective)),
            "median": float(np.median(effective)),
        },
        "dual_aperture": {
            str(temperature): {
                **_serializable(record),
                "dominance_mean": float(np.mean(dominance[temperature])),
                "ambiguous_below_0_8": float(
                    np.mean(dominance[temperature] < 0.8)),
            }
            for temperature, record in variants
        },
        "fine_scale_aperture": {
            str(gamma): {
                **_serializable(record),
                "dominance_mean": float(np.mean(scale_dominance)),
                "ambiguous_below_0_8": float(
                    np.mean(scale_dominance < 0.8)),
                "site_gain_p10_p50_p90": [
                    float(value) for value in np.percentile(
                        detail_gain, (10.0, 50.0, 90.0))],
            }
            for gamma, record, scale_dominance, detail_gain
            in scale_variants
        },
        "paired_coarse_fine_aperture": {
            str(gamma): {
                **_serializable(record),
                "coarse_dominance_mean": float(
                    np.mean(coarse_dominance)),
                "fine_dominance_mean": float(
                    np.mean(fine_dominance)),
            }
            for gamma, record, coarse_dominance, fine_dominance
            in paired_scale_variants
        },
        "solve_seconds": solve_seconds,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    save_panel(
        model.rgb, current, single, variants, dominance, args.save)
    print(json.dumps(report, indent=2))
    print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
