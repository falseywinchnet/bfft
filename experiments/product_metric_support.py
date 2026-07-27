#!/usr/bin/env python3
"""Transport each learned ellipse through the BFFT metric.

This experiment keeps the resource model's centers, anisotropy, support
kernel, hardness, crystallinity, and per-cell support count fixed.  It changes
only the path used to measure the cell's normalized radius.

For cell i with learned inverse-radius tensor S_i and BFFT edge cost m(x,e),

    ds_i(x,e) = ((1-alpha)|e| + alpha m(x,e))
                sqrt(e^T S_i e) / |e|.

At alpha=0 this is the cell's fixed tangent ellipse.  At alpha>0 its own
directional frame is accumulated through the image transport geometry.  A
per-cell scalar is selected so that the number of retained support pixels is
exactly the same as the original ellipse.  The comparison therefore cannot
win by growing more support or lose merely because the two metrics use
different units.

All-source, per-cell Dijkstra is deliberately an oracle.  It is not proposed
as a production implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
    score, solve_field, support_samples,
)
from geodesic_emission_support import (  # noqa: E402
    _cell_kernel, _kernel_extent_q, metric_graph, save_panel,
    support_fit, support_summary,
)
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


def _graph_arrays(reference):
    graph = metric_graph(reference)
    rows = np.repeat(
        np.arange(graph.shape[0], dtype=np.int32),
        np.diff(graph.indptr))
    cols = graph.indices.astype(np.int32, copy=False)
    row_y, row_x = np.divmod(rows, reference.w)
    col_y, col_x = np.divmod(cols, reference.w)
    dx = (col_x - row_x).astype(np.float64)
    dy = (col_y - row_y).astype(np.float64)
    step = np.hypot(dx, dy)
    return graph, dx, dy, step


def _product_samples(
    model, graph, dx, dy, step, target_count, alpha,
):
    """Emit mass-matched supports from cell-specific product metrics."""
    rows_out = []
    sites_out = []
    phi_out = []
    basis_out = []
    flat_x = np.tile(np.arange(model.w, dtype=np.float64), model.h)
    flat_y = np.repeat(np.arange(model.h, dtype=np.float64), model.w)
    spacing = max(
        math.sqrt(model.npix / max(len(model.centers), 1)), 1e-9)
    seed_x = np.clip(
        np.rint(model.centers[:, 0]).astype(np.int64), 0, model.w - 1)
    seed_y = np.clip(
        np.rint(model.centers[:, 1]).astype(np.int64), 0, model.h - 1)
    seed_index = seed_y * model.w + seed_x
    base_cost = (
        (1.0 - float(alpha)) * step +
        float(alpha) * graph.data)

    for site in range(len(model.centers)):
        theta = float(model.angle[site])
        ct, st = math.cos(theta), math.sin(theta)
        major = max(float(model.major[site]), 1.5)
        minor = max(float(model.minor[site]), 1.5)
        along = dx * ct + dy * st
        across = -dx * st + dy * ct
        tangent_cost = np.sqrt(
            np.square(along / major) +
            np.square(across / minor))
        edge_cost = base_cost * tangent_cost / np.maximum(step, 1e-12)
        cell_graph = sparse.csr_matrix(
            (edge_cost, graph.indices, graph.indptr),
            shape=graph.shape, copy=False)
        distance = dijkstra(
            cell_graph, directed=True, indices=int(seed_index[site]),
            return_predecessors=False)

        count = int(np.clip(target_count[site], 1, model.npix))
        cutoff_distance = np.partition(distance, count - 1)[count - 1]
        reach = max(
            float(cutoff_distance) /
            math.sqrt(_kernel_extent_q(model, site)),
            1e-9)
        q = np.square(distance / reach)
        phi = _cell_kernel(model, site, q)
        visible = np.isfinite(phi) & (phi > 1e-9)
        if not np.any(visible):
            continue
        index = np.flatnonzero(visible).astype(np.int32)
        rows_out.append(index)
        sites_out.append(np.full(len(index), site, dtype=np.int32))
        phi_out.append(phi[visible])
        basis_out.append(np.column_stack([
            np.ones(len(index), dtype=np.float64),
            (flat_x[visible] - model.centers[site, 0]) / spacing,
            (flat_y[visible] - model.centers[site, 1]) / spacing,
        ]))

    return {
        "rows": np.concatenate(rows_out),
        "sites": np.concatenate(sites_out),
        "phi": np.concatenate(phi_out),
        "basis": np.concatenate(basis_out, axis=0),
        "cells": len(model.centers),
        "spacing": spacing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--initial-cells", type=int, default=180)
    parser.add_argument("--resource-rounds", type=int, default=30)
    parser.add_argument(
        "--metric-strengths", type=float, nargs="*",
        default=(0.1, 0.25, 0.5, 1.0))
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/product_metric_support.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/product_metric_support.json")
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    model = ResourceTransportCells(
        image, ResourceConfig(
            max_side=args.side, cells=args.initial_cells))
    for _ in range(args.resource_rounds):
        model.step()
    objective = SingleStageDecompositionObjective(model.rgb)

    radial_samples = support_samples(model)
    radial, radial_dom, radial_effective = support_fit(
        model, radial_samples, model.lab, objective)
    target_count = np.bincount(
        radial_samples["sites"], minlength=len(model.centers))

    geometry = ReceiverGuidedVoronoi(
        image, Config(
            max_side=args.side, initial_cells=args.initial_cells,
            max_cells=args.initial_cells, passes=24, flow_sweeps=64,
            lam=0.05, mu=40.0, anisotropy=5.0,
            edge_density=4.0, texture_density=3.0,
            edge_barrier=12.0, site_reach=1.5))
    graph, dx, dy, step = _graph_arrays(geometry)

    variants = []
    timings = {}
    for alpha in args.metric_strengths:
        started = time.perf_counter()
        samples = _product_samples(
            model, graph, dx, dy, step, target_count, alpha)
        record, dominance, effective = support_fit(
            model, samples, model.lab, objective)
        name = f"product alpha={alpha:g}"
        variants.append((name, record, dominance, effective, samples))
        timings[name] = time.perf_counter() - started
        print(
            f"{name}: {record['psnr']:.3f} dB, "
            f"{len(samples['phi'])} samples, "
            f"{timings[name]:.2f} s", flush=True)

    report = {
        "source": source,
        "shape": [model.h, model.w],
        "cells": int(len(model.centers)),
        "local_ellipse_psnr": float(model.psnr),
        "coupled_ellipse": {
            key: float(value) for key, value in radial.items()
            if key != "rgb"
        },
        "ellipse_support": support_summary(
            radial_samples, radial_dom, radial_effective, model.npix),
        "variants": {
            name: {
                "elapsed_s": float(timings[name]),
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
