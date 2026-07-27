"""Shared fixture and timing for the sigma component optimizations.

Every optimizer in this folder is judged on two things only: it must
reproduce the baseline's output exactly (or to stated tolerance), and it must
be faster.  Nothing here changes what the algorithms compute.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import gallery  # noqa: E402
from transport_voronoi import Config  # noqa: E402
from claude_trial_sigma import SigmaVoronoi  # noqa: E402

_CACHE = {}


def fixture(image="pikachu", max_side=256, cells=2400, initial=180):
    key = (image, max_side, cells, initial)
    if key in _CACHE:
        return _CACHE[key]
    cfg = Config(
        max_side=max_side, initial_cells=initial, max_cells=cells,
        split_batch=36, allocation_mode="Expected affine gain",
        recursive_memory_stages=1, residual_memory_weight=0.0,
        composition_discrepancy_weight=0.0)
    model = SigmaVoronoi(gallery.load(image), cfg)
    while len(model.seeds) < cells:
        model.step()
    for _ in range(3):
        model.step()
    _CACHE[key] = model
    return model


def walk_inputs(model, scale=None):
    """Exactly the arguments `walk_with_predecessors` builds."""
    density = model.density.ravel()
    scale_ref = max(float(np.median(density)), 1e-9)
    sx = np.clip(np.rint(model.seeds[:, 0]).astype(np.int64), 0, model.w - 1)
    sy = np.clip(np.rint(model.seeds[:, 1]).astype(np.int64), 0, model.h - 1)
    seed_p = (sy * model.w + sx).astype(np.int64)
    reach = (model.cfg.site_reach *
             np.sqrt(density[seed_p] / scale_ref)).astype(np.float64)
    if scale is None:
        scale = np.ones(model.npix, dtype=np.float64)
    return seed_p, reach, model._edge_cost_volume, scale, model.h, model.w


def bench(label, fn, repeats=5, warmup=1):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    best = min(samples)
    print(f"  {label:<42s} {best * 1000:9.2f} ms   "
          f"(median {np.median(samples) * 1000:.2f})")
    return best, result
