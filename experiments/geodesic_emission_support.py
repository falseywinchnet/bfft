#!/usr/bin/env python3
"""Replace frozen ellipses by independent BFFT-geodesic emissions.

This is an exact support diagnostic.  Every resource cell retains its center,
metabolic area, concentration, and crystallinity.  Only its radial coordinate
changes:

    Euclidean ellipse q_i(x)  ->  (d_G(center_i, x) / radius_i)^2.

All cells emit independently and normalize softly.  No owner or runner is
used by the renderer.  The all-source Dijkstra call is intentionally a
research oracle; a successful result would motivate a local heat/eikonal
implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from bfft.vision import SingleStageDecompositionObjective  # noqa: E402
from dual_aperture_support import (  # noqa: E402
    aperture, design_matrix, score, solve_field, support_samples,
)
from receiver_guided_graph import ReceiverGuidedVoronoi  # noqa: E402
from resource_transport_cells import (  # noqa: E402
    ResourceConfig, ResourceTransportCells,
)
from transport_voronoi import Config  # noqa: E402


DIRECTIONS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)


def metric_graph(reference):
    h, w = reference.h, reference.w
    source_rows = []
    destination_rows = []
    weights = []
    for direction, (dy, dx) in enumerate(DIRECTIONS):
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        source = (yy * w + xx).ravel()
        destination = ((yy + dy) * w + (xx + dx)).ravel()
        cost = reference._edge_cost_volume[
            direction, y0:y1, x0:x1].ravel()
        finite = np.isfinite(cost)
        source_rows.append(source[finite])
        destination_rows.append(destination[finite])
        weights.append(cost[finite].astype(np.float64))
    graph = sparse.coo_matrix((
        np.concatenate(weights),
        (np.concatenate(source_rows), np.concatenate(destination_rows))),
        shape=(h * w, h * w)).tocsr()
    graph.sum_duplicates()
    return graph


def _cell_kernel(model, site, q):
    hardness = float(model.hardness[site])
    family = model.cfg.kernel_family
    power_amplitude = (
        (hardness + 1.0) /
        (model.cfg.kernel_power + 1.0))
    power = (
        power_amplitude *
        np.power(np.maximum(1.0 - q, 0.0), hardness))
    power[q >= 1.0] = 0.0
    if family == "power":
        return power

    softplus = float(np.logaddexp(0.0, hardness))
    radial_mass = softplus / max(hardness, 1e-12)
    logistic_amplitude = (
        1.0 /
        ((model.cfg.kernel_power + 1.0) * radial_mass))
    logistic = logistic_amplitude / (
        1.0 + np.exp(np.clip(hardness * (q - 1.0), -60.0, 60.0)))
    if family == "logistic":
        logistic[q >= 1.0 + 8.0 / max(hardness, 1e-6)] = 0.0
        return logistic

    crystallinity = 1.0 / (
        1.0 + math.exp(-float(model.crystallinity_logit[site])))
    tail_widths = float(np.clip(
        math.log(max(crystallinity, 1e-12) / 1e-3), 0.0, 8.0))
    logistic[
        q >= 1.0 + tail_widths / max(hardness, 1e-6)] = 0.0
    return (1.0 - crystallinity) * power + crystallinity * logistic


def _kernel_extent_q(model, site, threshold=1e-9):
    """Largest normalized squared radius retained by a cell kernel."""
    lo = 0.0
    hi = 64.0
    for _ in range(48):
        middle = 0.5 * (lo + hi)
        value = float(_cell_kernel(
            model, site, np.array([middle], dtype=np.float64))[0])
        if value > threshold:
            lo = middle
        else:
            hi = middle
    return max(lo, 1e-9)


def _mass_matched_radii(model, distances, radial_samples):
    """Choose one geodesic radius per cell with the same support count.

    Matching support count removes the units/coverage confound from the
    geometric comparison.  It does not alter centers, kernel family,
    hardness, or crystallinity.
    """
    target_count = np.bincount(
        radial_samples["sites"], minlength=len(model.centers))
    radii = np.empty(len(model.centers), dtype=np.float64)
    for site, count in enumerate(target_count):
        count = int(np.clip(count, 1, model.npix))
        kth = np.partition(distances[site], count - 1)[count - 1]
        radii[site] = max(
            float(kth) / math.sqrt(_kernel_extent_q(model, site)),
            1e-6)
    return radii


def geodesic_samples(model, distances, reach_scale=1.0, radii=None):
    n = len(model.centers)
    spacing = max(math.sqrt(model.npix / max(n, 1)), 1e-9)
    rows = []
    sites = []
    phis = []
    bases = []
    flat_x = np.tile(np.arange(model.w, dtype=np.float64), model.h)
    flat_y = np.repeat(np.arange(model.h, dtype=np.float64), model.w)
    for site in range(n):
        if radii is None:
            radius = max(
                math.sqrt(float(model.major[site] * model.minor[site])),
                1.0)
        else:
            radius = max(float(radii[site]), 1e-6)
        radius *= float(reach_scale)
        q = np.square(distances[site] / radius)
        phi = _cell_kernel(model, site, q)
        visible = np.isfinite(phi) & (phi > 1e-9)
        if not np.any(visible):
            continue
        index = np.flatnonzero(visible).astype(np.int32)
        rows.append(index)
        sites.append(np.full(len(index), site, dtype=np.int32))
        phis.append(phi[visible])
        bases.append(np.column_stack([
            np.ones(len(index), dtype=np.float64),
            (flat_x[visible] - model.centers[site, 0]) / spacing,
            (flat_y[visible] - model.centers[site, 1]) / spacing,
        ]))
    return {
        "rows": np.concatenate(rows),
        "sites": np.concatenate(sites),
        "phi": np.concatenate(phis),
        "basis": np.concatenate(bases, axis=0),
        "cells": n,
        "spacing": spacing,
    }


def support_fit(model, samples, target, objective):
    weight, dominance, effective = aperture(
        samples, model.npix, 1.0)
    design = design_matrix(samples, model.npix, weight)
    field = solve_field(design, target)
    record = score(objective, model.rgb, field)
    return record, dominance.reshape(model.h, model.w), effective


def support_summary(samples, dominance, effective, npix):
    covered = np.bincount(
        samples["rows"], minlength=npix) > 0
    return {
        "samples": int(len(samples["phi"])),
        "coverage": float(np.mean(covered)),
        "dominance_mean": float(np.mean(dominance.ravel()[covered])),
        "effective_mean": float(np.mean(effective[covered])),
    }


def save_panel(model, radial, variants, radial_dom, path):
    import matplotlib.pyplot as plt

    columns = 3 + len(variants)
    figure, axes = plt.subplots(2, columns, figsize=(4 * columns, 8))
    products = [
        (model.rgb, "target", None),
        (model.rgb_reconstruction, f"local ellipse {model.psnr:.2f}", None),
        (radial["rgb"], f"coupled ellipse {radial['psnr']:.2f}", None),
    ] + [
        (record["rgb"], f"{name}\n{record['psnr']:.2f} dB", None)
        for name, record, _dominance, _effective, _samples in variants
    ]
    for axis, (image, title, cmap) in zip(axes[0], products):
        axis.imshow(
            image, cmap=cmap,
            vmin=-0.5 if cmap else None,
            vmax=0.5 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    diagnostics = [
        (np.zeros_like(radial_dom), "target"),
        (np.zeros_like(radial_dom), "local ellipse"),
        (radial_dom, "ellipse dominance"),
    ] + [
        (dominance, f"{name}\ndominance")
        for name, _record, dominance, _effective, _samples in variants
    ]
    for axis, (image, title) in zip(axes[1], diagnostics):
        axis.imshow(image, cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
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
        "--reach-scales", type=float, nargs="*", default=(1.0, 1.5, 2.0))
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/geodesic_emission_support.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/geodesic_emission_support.json")
    args = parser.parse_args()

    image, source = load_image(args.image, args.gallery)
    model = ResourceTransportCells(
        image, ResourceConfig(
            max_side=args.side, cells=args.initial_cells))
    for _ in range(args.resource_rounds):
        model.step()
    objective = SingleStageDecompositionObjective(model.rgb)

    radial_samples = support_samples(model)
    radial, radial_dom, radial_effective = support_fit(
        model, radial_samples, model.lab, objective)

    geometry = ReceiverGuidedVoronoi(
        image, Config(
            max_side=args.side, initial_cells=args.initial_cells,
            max_cells=args.initial_cells, passes=24, flow_sweeps=64,
            lam=0.05, mu=40.0, anisotropy=5.0,
            edge_density=4.0, texture_density=3.0,
            edge_barrier=12.0, site_reach=1.5))
    graph = metric_graph(geometry)
    seed_x = np.clip(
        np.rint(model.centers[:, 0]).astype(np.int64), 0, model.w - 1)
    seed_y = np.clip(
        np.rint(model.centers[:, 1]).astype(np.int64), 0, model.h - 1)
    seed_index = seed_y * model.w + seed_x
    distances = dijkstra(
        graph, directed=True, indices=seed_index,
        return_predecessors=False)
    variants = []
    for scale in args.reach_scales:
        samples = geodesic_samples(
            model, distances, reach_scale=scale)
        record, dominance, effective = support_fit(
            model, samples, model.lab, objective)
        variants.append((
            f"geodesic x{scale:g}", record, dominance,
            effective, samples))

    matched_radii = _mass_matched_radii(
        model, distances, radial_samples)
    matched_samples = geodesic_samples(
        model, distances, radii=matched_radii)
    matched, matched_dom, matched_effective = support_fit(
        model, matched_samples, model.lab, objective)
    variants.append((
        "geodesic mass-matched", matched, matched_dom,
        matched_effective, matched_samples))

    report = {
        "source": source,
        "shape": [model.h, model.w],
        "cells": int(len(model.centers)),
        "local_ellipse": {
            "psnr": float(model.psnr),
        },
        "coupled_ellipse": {
            key: float(value) for key, value in radial.items()
            if key != "rgb"
        },
        "ellipse_support": support_summary(
            radial_samples, radial_dom, radial_effective, model.npix),
        "geodesic_variants": {
            name: {
                "score": {
                    key: float(value) for key, value in record.items()
                    if key != "rgb"
                },
                "support": support_summary(
                    samples, dominance, effective, model.npix),
            }
            for name, record, dominance, effective, samples in variants
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    save_panel(model, radial, variants, radial_dom, args.save)
    print(json.dumps(report, indent=2))
    print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
