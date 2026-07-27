#!/usr/bin/env python3
"""Measurement driver for the alpha trials.  Writes to nothing it imports.

Reports the log's three-error objective (RGB, one-stage cartoon, one-stage
texture MSE) plus wall time, so a trial can only be archived after it has
actually run.

    PYTHONPATH=.:viewer .venv/bin/python viewer/claude_trial_alpha_run.py \
        --max-side 128 --cells 700

Trials:
  exact_solve   exact normal-equation coupled fit vs the lsmr coupled fit at
                identical sites (a renderer change; allocation untouched)
  prices        rank agreement between the incumbent death score and the
                exact deletion price, and the complementarity of adjacent
                pairs
  births        persistence-ranked births vs pressure arg-max births
  deaths        exact-price victims vs integrated-score victims
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402
import gallery  # noqa: E402
from transport_voronoi import Config, TransportVoronoi  # noqa: E402
import claude_trial_alpha_normal as alpha_normal  # noqa: E402
import claude_trial_alpha_market as alpha_market  # noqa: E402


PIKACHU = Path.home() / "Downloads" / "25.png"


def load(key):
    if key == "pikachu":
        import imageio.v3 as iio
        return iio.imread(PIKACHU)
    return gallery.load(key)


def native_components(image, cfg):
    split = bfft.meyer_channels(
        image, space="oklab_lc", lam=cfg.lam, mu=cfg.mu,
        passes=cfg.passes, threads=4)
    scale = np.maximum(split.scale[None, None, :], 1e-12)
    return split.cartoon / scale, split.texture / scale


def score(model, cfg):
    from bfft.effects import lab_to_srgb
    recon = np.clip(lab_to_srgb(model.reconstruction), 0.0, 1.0)
    target_c, target_t = native_components(model.rgb, cfg)
    recon_c, recon_t = native_components(recon, cfg)
    rgb_mse = float(np.mean((model.rgb - recon) ** 2))
    return {
        "rgb_mse": rgb_mse,
        "psnr": -10.0 * math.log10(max(rgb_mse, 1e-12)),
        "cartoon_mse": float(np.mean((target_c - recon_c) ** 2)),
        "texture_mse": float(np.mean((target_t - recon_t) ** 2)),
    }


def grow(model, cfg, exchanges=3):
    while len(model.seeds) < cfg.max_cells:
        model.step()
    for _ in range(exchanges):
        model.step()
    return model


def make_config(args):
    return Config(
        max_side=args.max_side, passes=args.passes,
        flow_sweeps=args.flow_sweeps, initial_cells=args.initial_cells,
        max_cells=args.cells, split_batch=args.split_batch,
        allocation_mode=args.allocation_mode)


# -- trials ----------------------------------------------------------
def trial_exact_solve(image, cfg, args):
    model = grow(TransportVoronoi(image, cfg), cfg)
    t0 = time.perf_counter()
    model.solve_coupled(True, args.cartoon_softness, args.texture_softness)
    lsmr_ms = (time.perf_counter() - t0) * 1000.0
    lsmr = score(model, cfg)

    t0 = time.perf_counter()
    solver, _ = alpha_normal.solve_coupled_exact(
        model, args.cartoon_softness, args.texture_softness)
    exact_ms = (time.perf_counter() - t0) * 1000.0
    exact = score(model, cfg)
    return {
        "lsmr": dict(lsmr, ms=lsmr_ms),
        "exact": dict(exact, ms=exact_ms),
        "delta_db": exact["psnr"] - lsmr["psnr"],
        "speedup": lsmr_ms / max(exact_ms, 1e-9),
        "graph": solver.graph_report(),
    }


def trial_prices(image, cfg, args):
    model = grow(TransportVoronoi(image, cfg), cfg)
    alpha_normal.solve_coupled_exact(
        model, args.cartoon_softness, args.texture_softness)
    t0 = time.perf_counter()
    report = alpha_market.price_correlation(
        model, shortlist=args.shortlist,
        cartoon_softness=args.cartoon_softness)
    price_ms = (time.perf_counter() - t0) * 1000.0
    solver = report.pop("solver")
    columns = report.pop("columns")
    candidates = report.pop("candidates")
    exact = report.pop("exact")

    shortlist = set(int(i) for i in candidates)
    pairs = [(int(i), int(j)) for i, j in solver.edges
             if int(i) in shortlist and int(j) in shortlist]
    pairs = pairs[:4000]
    prices = {int(i): float(p) for i, p in zip(candidates, exact)}
    coupling = solver.complementarity(pairs, columns, prices)
    values = np.array(list(coupling.values())) if coupling else np.zeros(1)
    report.update({
        "price_ms": price_ms,
        "priced_pairs": len(coupling),
        "redundant_pairs": int(np.sum(values < 0.0)),
        "complementary_pairs": int(np.sum(values > 0.0)),
        "coupling_median": float(np.median(values)),
    })
    return report


def trial_allocator(image, cfg, args, exact_deaths, persistence_births):
    model = alpha_market.MarketVoronoi(
        image, cfg, exact_deaths=exact_deaths,
        persistence_births=persistence_births,
        price_shortlist=args.shortlist,
        cartoon_softness=args.cartoon_softness)
    t0 = time.perf_counter()
    grow(model, cfg)
    grow_s = time.perf_counter() - t0
    local = score(model, cfg)
    alpha_normal.solve_coupled_exact(
        model, args.cartoon_softness, args.texture_softness)
    coupled = score(model, cfg)
    return {
        "local": local,
        "coupled": coupled,
        "grow_s": grow_s,
        "cells": len(model.seeds),
        "price_report": model.price_report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-side", type=int, default=128)
    parser.add_argument("--cells", type=int, default=700)
    parser.add_argument("--initial-cells", type=int, default=96)
    parser.add_argument("--split-batch", type=int, default=36)
    parser.add_argument("--passes", type=int, default=6)
    parser.add_argument("--flow-sweeps", type=int, default=24)
    parser.add_argument("--shortlist", type=int, default=240)
    parser.add_argument("--cartoon-softness", type=float, default=4.0)
    parser.add_argument("--texture-softness", type=float, default=16.0)
    parser.add_argument(
        "--allocation-mode", default="Expected affine gain")
    parser.add_argument(
        "--images", nargs="+",
        default=["pikachu", "camera", "coins", "grass", "chelsea"])
    parser.add_argument(
        "--trials", nargs="+",
        default=["exact_solve", "prices", "births", "deaths", "both"])
    args = parser.parse_args()

    record = {"protocol": vars(args), "results": []}
    for key in args.images:
        image = load(key)
        cfg = make_config(args)
        for trial in args.trials:
            t0 = time.perf_counter()
            if trial == "exact_solve":
                result = trial_exact_solve(image, cfg, args)
                line = (f"{result['exact']['psnr']:6.3f} dB exact vs "
                        f"{result['lsmr']['psnr']:6.3f} lsmr  "
                        f"({result['speedup']:.1f}x faster)")
            elif trial == "prices":
                result = trial_prices(image, cfg, args)
                line = (f"rho {result['spearman_incumbent_vs_exact']:+.3f}  "
                        f"redundant {result['redundant_pairs']}/"
                        f"{result['priced_pairs']}")
            else:
                exact_deaths = trial in ("deaths", "both")
                births = trial in ("births", "both")
                result = trial_allocator(
                    image, cfg, args, exact_deaths, births)
                line = (f"{result['coupled']['psnr']:6.3f} dB coupled  "
                        f"({result['grow_s']:.1f} s grow)")
            result.update({
                "image": key, "trial": trial,
                "elapsed_s": time.perf_counter() - t0,
            })
            record["results"].append(result)
            print(f"{key:9s} {trial:12s} {line}",
                  file=sys.stderr, flush=True)
    print(json.dumps(record, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
