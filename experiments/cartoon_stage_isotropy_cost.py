#!/usr/bin/env python3
"""What does the anisotropic substitution cost the pipeline?

`notes/cartoon_stage_problem_statement.md` §7 showed that exact 1-D
subproblems find the jump set 2.8-3.9x faster than a shrink, and are cheaper
per iteration. That result is for **anisotropic** TV. The shipped cartoon is
isotropic. This file measures the only thing that decides whether the trade
is worth taking: what the substitution does to the transport pipeline's own
objective.

Design, so the comparison cannot flatter either arm:

* The substitution happens at the module boundary. `bfft.meyer_split` and
  `bfft.rof` are swapped for anisotropic Douglas-Rachford versions while the
  model is constructed, so **every** downstream consumer -- the metric, the
  edge and texture fields, `base_lab`, the allocation pressure -- sees the
  substituted cartoon. Nothing else differs.
* `bfft.meyer_channels` is **not** patched. It is the scoring operator, and
  it is applied to the target and to the reconstruction alike in both arms.
  Scoring with the thing under test would be circular.
* Both arms grow to the same cell budget under the same currency and finish
  with the same coupled solve.

    PYTHONPATH=.:viewer .venv/bin/python \
        experiments/cartoon_stage_isotropy_cost.py
"""

from __future__ import annotations

import contextlib
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
from bfft.effects import lab_to_srgb  # noqa: E402
from transport_voronoi import Config, TransportVoronoi  # noqa: E402

from cartoon_stage_tautstring import (  # noqa: E402
    _div, _grad, _solve_neumann, tv_cols, tv_rows)


def iso_neumann_split(image, lam=0.05, mu=40.0, passes=24, threads=0):
    """Isotropic TGFD, but with Neumann boundaries instead of periodic.

    This isolates the confound.  The shipped kernel wraps its differences --
    defect 7 of `viewer/TRANSPORT_CELL_MATH.md` -- while the taut-string
    route treats each line with free ends.  Any gain the anisotropic arm
    shows could be the boundary fix rather than the anisotropy, and this arm
    separates them: same isotropic shrink as the kernel, same alternation,
    only the boundary condition changed.
    """
    f = np.ascontiguousarray(np.asarray(image, dtype=np.float64))
    plans = ((lam, 2.0 * lam), (1.0 / mu, 10.0 / mu))
    state = [[np.zeros_like(f) for _ in range(4)] for _ in range(2)]
    u = np.zeros_like(f)
    w = np.zeros_like(f)
    for p in range(max(1, int(passes))):
        for side in (0, 1):
            c, eta = plans[side]
            bx, by, px, py = state[side]
            if side == 0:
                g = f if p == 0 else u + w
            else:
                g = f - u
            x = _solve_neumann(c * g - eta * _div(px, py), c, eta)
            gx, gy = _grad(x)
            tx, ty = gx + bx, gy + by
            magnitude = np.sqrt(tx * tx + ty * ty)
            shrink = np.where(
                magnitude > 1.0 / eta,
                1.0 - (1.0 / eta) / np.maximum(magnitude, 1e-300), 0.0)
            dx, dy = tx * shrink, ty * shrink
            bx, by = tx - dx, ty - dy
            state[side] = [bx, by, dx - bx, dy - by]
            if side == 0:
                u = x
            else:
                w = x
    return u, f - u - w


# ----------------------------------------------------------------------
# The anisotropic stand-ins
# ----------------------------------------------------------------------

def dr_step(g, c, z, gamma=1.0):
    """One Douglas-Rachford iteration of anisotropic ROF."""
    lam = 1.0 / c
    target = (gamma * g + z) / (1.0 + gamma)
    u = tv_rows(target, gamma * lam / (1.0 + gamma))
    v = tv_cols(2.0 * u - z, gamma * lam)
    return u, z + v - u


def aniso_rof(image, c=0.025, eta=0.0, sweeps=20, tol=0.0, threads=0):
    """Stand-in for `bfft.rof`: anisotropic, solved by exact 1-D steps."""
    g = np.ascontiguousarray(np.asarray(image, dtype=np.float64))
    z = g.copy()
    u = g
    for _ in range(max(1, int(min(sweeps, 24)))):
        u, z = dr_step(g, c, z)
    return u


def aniso_meyer_split(image, lam=0.05, mu=40.0, passes=24, threads=0):
    """Stand-in for `bfft.meyer_split`: the same TGFD alternation.

    Structure is the kernel's exactly -- at pass zero the texture is defined
    to be zero so the cartoon step sees `f`, and thereafter it sees `u + w`.
    Only the inner solver changes.
    """
    f = np.ascontiguousarray(np.asarray(image, dtype=np.float64))
    c_u, c_v = lam, 1.0 / mu
    u = np.zeros_like(f)
    w = np.zeros_like(f)
    z_u = None
    z_v = None
    for p in range(max(1, int(passes))):
        g = f if p == 0 else u + w
        if z_u is None:
            z_u = g.copy()
        u, z_u = dr_step(g, c_u, z_u)
        g = f - u
        if z_v is None:
            z_v = g.copy()
        w, z_v = dr_step(g, c_v, z_v)
    return u, f - u - w


@contextlib.contextmanager
def substituted_cartoon(passes, kind="anisotropic"):
    """Swap the cartoon stage for every downstream consumer at once."""
    original_split, original_rof = bfft.meyer_split, bfft.rof
    maker = (aniso_meyer_split if kind == "anisotropic"
             else iso_neumann_split)

    def split(image, lam=0.05, mu=40.0, passes_=passes, threads=0, **kw):
        return maker(image, lam=lam, mu=mu, passes=passes, threads=threads)

    bfft.meyer_split = split
    bfft.rof = aniso_rof
    try:
        yield
    finally:
        bfft.meyer_split, bfft.rof = original_split, original_rof


# ----------------------------------------------------------------------
# Scoring, with the unpatched operator
# ----------------------------------------------------------------------

def native_components(image, cfg):
    split = bfft.meyer_channels(
        image, space="oklab_lc", lam=cfg.lam, mu=cfg.mu,
        passes=cfg.passes, threads=4)
    scale = np.maximum(split.scale[None, None, :], 1e-12)
    return split.cartoon / scale, split.texture / scale


def score(model, cfg):
    reconstruction = np.clip(lab_to_srgb(model.reconstruction), 0.0, 1.0)
    target_cartoon, target_texture = native_components(model.rgb, cfg)
    recon_cartoon, recon_texture = native_components(reconstruction, cfg)
    rgb_mse = float(np.mean((model.rgb - reconstruction) ** 2))
    cartoon_mse = float(np.mean((target_cartoon - recon_cartoon) ** 2))
    texture_mse = float(np.mean((target_texture - recon_texture) ** 2))
    return {
        "cells": int(len(model.seeds)),
        "psnr": float(-10.0 * math.log10(max(rgb_mse, 1e-12))),
        "rgb_mse": rgb_mse,
        "cartoon_mse": cartoon_mse,
        "texture_mse": texture_mse,
        "objective": rgb_mse + cartoon_mse + texture_mse,
    }


def run_arm(image, cfg, label, passes=None, kind="anisotropic"):
    started = time.perf_counter()
    context = (substituted_cartoon(passes, kind) if passes
               else contextlib.nullcontext())
    with context:
        model = TransportVoronoi(image, cfg)
        while len(model.seeds) < cfg.max_cells:
            model.step()
        for _ in range(3):
            model.step()
        grow_s = time.perf_counter() - started
        model.solve_coupled(multiscale=True, cartoon_softness=4.0,
                            texture_softness=16.0)
    record = score(model, cfg)
    record.update({"arm": label, "grow_s": grow_s})
    return record, model


def compare_splits(light):
    """How different are the two cartoons, before any pipeline runs?"""
    print("\n== the two cartoons, side by side ==")
    t0 = time.perf_counter()
    iso_cartoon, iso_texture = bfft.meyer_split(
        light, lam=0.05, mu=40.0, passes=24, threads=4)
    iso_s = time.perf_counter() - t0
    for passes in (8, 24):
        t0 = time.perf_counter()
        ani_cartoon, ani_texture = aniso_meyer_split(
            light, lam=0.05, mu=40.0, passes=passes)
        ani_s = time.perf_counter() - t0
        scale = float(np.linalg.norm(iso_cartoon))
        gap = float(np.linalg.norm(ani_cartoon - iso_cartoon)) / scale
        print(f"  anisotropic, {passes:2d} passes: relative distance to the "
              f"isotropic cartoon {gap:.4f}   "
              f"({ani_s * 1e3:.0f} ms vs {iso_s * 1e3:.0f} ms for the "
              f"shipped 24-pass split)")


def main():
    for key in ("pikachu", "chelsea"):
        run_image(key)


def run_image(key):
    image = gallery.load(key)
    cfg = Config(max_side=128, initial_cells=120, max_cells=700,
                 split_batch=24, allocation_mode="Expected affine gain",
                 recursive_memory_stages=1, residual_memory_weight=0.0,
                 composition_discrepancy_weight=0.0)

    probe = TransportVoronoi(image, Config(max_side=128, initial_cells=8,
                                           max_cells=8))
    compare_splits(probe.lab[..., 0] * 255.0)

    print(f"\n== full pipeline, {key}, "
          f"{cfg.max_side} px / {cfg.max_cells} cells ==")
    results = []
    for label, passes, kind in (
            ("isotropic, periodic (shipped)", None, ""),
            ("isotropic, Neumann", 24, "isotropic"),
            ("anisotropic, 24 passes", 24, "anisotropic"),
            ("anisotropic, 8 passes", 8, "anisotropic")):
        record, _ = run_arm(image, cfg, label, passes, kind)
        results.append(record)
        print(f"  {label:30s} {record['psnr']:6.2f} dB  "
              f"obj {record['objective']:.4e}  "
              f"cartoon {record['cartoon_mse']:.3e}  "
              f"texture {record['texture_mse']:.3e}  "
              f"grow {record['grow_s']:.1f}s")

    control = results[0]
    print()
    for record in results[1:]:
        change = (record["objective"] - control["objective"]) / (
            control["objective"])
        verdict = "better" if change < 0 else "worse"
        print(f"  {record['arm']:30s} objective {change:+7.1%} "
              f"({verdict}), PSNR "
              f"{record['psnr'] - control['psnr']:+.2f} dB")

    out = ROOT / "experiments" / "out" / "cartoon_isotropy_cost.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
