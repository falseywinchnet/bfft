#!/usr/bin/env python3
"""Component 4: the objective evaluation inside the Newton line search.

`receiver_guided_graph.receiver_newton_step` evaluates the objective seven
times per step -- once for the baseline, five times for the backtracking
sweep, once for the final state.  At 256 px / 2,400 cells that is 533 ms of
scoring against 588 ms of actually solving.

Two formal wastes, both removable without changing a single reported number:

**The target decomposition is invariant.**  `score` calls
`native_components(model.rgb, cfg)` on every invocation.  `model.rgb` is the
source image; it does not change between line-search probes, between Newton
steps, or ever.  That is 25 ms of Meyer decomposition recomputed six times
per step to produce the same array.  Memoizing on the array's identity and
the parameters that reach the decomposition makes it exact by construction.

**The diffuseness diagnostic is not part of the objective.**  The line
search reads `objective` alone, but `score` also runs `residual_structure`,
which is nine Gaussian filters over the full image.  Computing a diagnostic
six times to read it zero times is the same waste in a different place.
Making it lazy leaves it available and stops paying for it during a search.

Neither change touches the arithmetic.  The test below asserts every field
is bit-identical to the baseline scorer, including on a model whose
reconstruction has been perturbed between calls, which is the only way a
stale cache could hide.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_common import bench, fixture  # noqa: E402

import bfft  # noqa: E402
from bfft.effects import lab_to_srgb  # noqa: E402
from claude_trial_sigma import residual_structure  # noqa: E402


class TargetSplit:
    """Memoized Meyer decomposition of an unchanging image.

    Keyed on the array's identity *and* its contents' checksum, so an
    in-place edit cannot silently return a stale split.  The checksum costs
    a fraction of a millisecond against the 25 ms it guards.
    """

    def __init__(self):
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def components(self, image, cfg):
        key = (id(image), image.shape, float(image.sum()),
               cfg.lam, cfg.mu, cfg.passes)
        hit = self._cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        split = bfft.meyer_channels(
            image, space="oklab_lc", lam=cfg.lam, mu=cfg.mu,
            passes=cfg.passes, threads=4)
        scale = np.maximum(split.scale[None, None, :], 1e-12)
        value = (split.cartoon / scale, split.texture / scale)
        self._cache[key] = value
        return value


_TARGETS = TargetSplit()


def score_fast(model, cfg, extra=None, with_structure=False,
               targets=_TARGETS):
    """Same numbers as the harness scorer, minus the recomputation."""
    reconstruction = np.clip(lab_to_srgb(model.reconstruction), 0.0, 1.0)
    target_cartoon, target_texture = targets.components(model.rgb, cfg)
    split = bfft.meyer_channels(
        reconstruction, space="oklab_lc", lam=cfg.lam, mu=cfg.mu,
        passes=cfg.passes, threads=4)
    scale = np.maximum(split.scale[None, None, :], 1e-12)
    recon_cartoon = split.cartoon / scale
    recon_texture = split.texture / scale
    rgb_mse = float(np.mean((model.rgb - reconstruction) ** 2))
    record = {
        "cells": int(len(model.seeds)),
        "rgb_mse": rgb_mse,
        "psnr": float(-10.0 * math.log10(max(rgb_mse, 1e-12))),
        "cartoon_mse": float(np.mean(
            (target_cartoon - recon_cartoon) ** 2)),
        "texture_mse": float(np.mean(
            (target_texture - recon_texture) ** 2)),
    }
    record["objective"] = (
        record["rgb_mse"] + record["cartoon_mse"] + record["texture_mse"])
    if with_structure:
        record["residual_structure"] = float(residual_structure(model))
    if extra:
        record.update(extra)
    return record


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import receiver_guided_graph as rg

    for image, side, cells in (("camera", 128, 700), ("pikachu", 256, 2400)):
        model = fixture(image, side, cells)
        fields = model._solve_direct_pair()
        model._apply_direct_fields(
            fields["base"][1]["field"], fields["detail"][1]["field"],
            "bench")
        print(f"\n{image} {side} px / {cells} cells")

        score_fast(model, model.cfg)
        base_time, base_record = bench(
            "baseline score()", lambda: rg.score(model, model.cfg))
        search_time, search_record = bench(
            "search score (cached target, lazy struct)",
            lambda: score_fast(model, model.cfg))
        full_time, full_record = bench(
            "full score (cached target, with struct)",
            lambda: score_fast(model, model.cfg, with_structure=True))
        print(f"  speedup  search {base_time / search_time:.1f}x   "
              f"full {base_time / full_time:.1f}x")

        for key in ("rgb_mse", "psnr", "cartoon_mse", "texture_mse",
                    "objective"):
            assert base_record[key] == search_record[key], (
                key, base_record[key], search_record[key])
        assert (full_record["residual_structure"] ==
                float(residual_structure(model)))
        print("    every shared field bit-identical")

        # A stale cache would survive the check above.  Change the
        # reconstruction and confirm the score moves with it.
        before = score_fast(model, model.cfg)["objective"]
        model.reconstruction = model.reconstruction * 0.98
        after = score_fast(model, model.cfg)["objective"]
        reference = rg.score(model, model.cfg)["objective"]
        assert after == reference and after != before
        print(f"    responds to a changed reconstruction "
              f"({before:.4e} -> {after:.4e}, matches baseline)")
        model.reconstruction = model.reconstruction / 0.98

        per_step = 7
        saved = (base_time - search_time) * (per_step - 1) + (
            base_time - full_time)
        print(f"    per Newton step: {saved * 1000:.0f} ms removed "
              f"from {base_time * per_step * 1000:.0f} ms of scoring")
        print(f"    target splits computed: {_TARGETS.misses}, "
              f"reused: {_TARGETS.hits}")


if __name__ == "__main__":
    main()
