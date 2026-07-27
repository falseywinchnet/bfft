#!/usr/bin/env python3
"""Component 5: the envelope-gradient accumulation over both forests.

Baseline sorts all `2 * npix` states by decreasing distance with `argsort`
(`O(n log n)` over 131,072 elements) and then, for every tree edge, searches
the eight directions to recover which step was taken.

Formal target: **the sort is already done, and the direction is already
known.**  Two facts, both inherited from the bucket walk rather than assumed:

* every tree edge costs at least `delta`, so a child's distance lands in a
  strictly later `delta`-bucket than its parent's.  Emptying buckets in
  descending order therefore visits every child before its parent, which is
  the only property the accumulation needs.  A counting sort over a few
  hundred buckets replaces the comparison sort, `O(n log n) -> O(n)`, and it
  is not an approximation: within a bucket no parent-child pair can exist.
* the direction taken into each node was known at relaxation time.  The
  bucket walk stores it (`pd1`, `pd2`), so the eight-way search per edge
  becomes one array read.

The gradient is asserted bit-identical to the baseline's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_common import bench, fixture, walk_inputs  # noqa: E402
from opt_dijkstra_bucket import (  # noqa: E402
    pack_adjacency, queue_geometry, run_packed)

from claude_trial_sigma import _accumulate_tree  # noqa: E402

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(fn):  # pragma: no cover
    return fn


_compile = njit(cache=True) if njit is not None else _identity


@_compile
def _bucket_order(d1, d2, delta, shift):
    """Counting sort of both label sets into descending distance order."""
    npix = len(d1)
    slots = np.full(2 * npix, -1, dtype=np.int64)
    highest = 0
    for p in range(npix):
        if d1[p] < 1e299:
            slot = int((d1[p] + shift) / delta)
            slots[p] = slot
            if slot > highest:
                highest = slot
        if d2[p] < 1e299:
            slot = int((d2[p] + shift) / delta)
            slots[npix + p] = slot
            if slot > highest:
                highest = slot
    counts = np.zeros(highest + 2, dtype=np.int64)
    live = 0
    for k in range(2 * npix):
        if slots[k] >= 0:
            counts[slots[k]] += 1
            live += 1
    # Descending cumulative placement: the last bucket goes first.
    starts = np.zeros(highest + 2, dtype=np.int64)
    running = 0
    for b in range(highest, -1, -1):
        starts[b] = running
        running += counts[b]
    order = np.empty(live, dtype=np.int64)
    cursor = starts.copy()
    for k in range(2 * npix):
        slot = slots[k]
        if slot >= 0:
            order[cursor[slot]] = k
            cursor[slot] += 1
    return order


@_compile
def _accumulate_tagged(order, seed_sensitivity, pr1, pr2, pl1, pl2,
                       pd1, pd2, cost, npix):
    """Same accumulation, with the direction read instead of searched."""
    acc = seed_sensitivity.copy()
    grad = np.zeros(npix, dtype=np.float64)
    for k in range(len(order)):
        state = order[k]
        label = state // npix
        p = state - label * npix
        if label == 0:
            parent = pr1[p]
            parent_label = pl1[p]
            direction = pd1[p]
        else:
            parent = pr2[p]
            parent_label = pl2[p]
            direction = pd2[p]
        if parent < 0 or direction < 0:
            continue
        share = acc[state]
        if share == 0.0:
            continue
        acc[parent_label * npix + parent] += share
        base = cost[8 * parent + direction]
        grad[parent] += 0.5 * share * base
        grad[p] += 0.5 * share * base
    return grad


def baseline_gradient(model, forest, sensitivity):
    pr1, pr2, pl1, pl2 = forest
    states = np.concatenate([-sensitivity, sensitivity])
    distances = np.concatenate([model.d1, model.d2])
    finite = np.isfinite(distances) & (distances < 1e299)
    order = np.argsort(-np.where(finite, distances, -np.inf),
                       kind="stable").astype(np.int64)
    order = order[finite[order]]
    return _accumulate_tree(
        order, states, pr1, pr2, pl1, pl2,
        model._edge_cost_volume, model.h, model.w)


def fast_gradient(model, walk, sensitivity, cost, delta, shift):
    own, run, d1, d2, pr1, pr2, pl1, pl2, pd1, pd2, _ = walk
    states = np.concatenate([-sensitivity, sensitivity])
    order = _bucket_order(d1, d2, delta, shift)
    return _accumulate_tagged(
        order, states, pr1, pr2, pl1, pl2, pd1, pd2, cost, model.npix)


def main():
    for image, side, cells in (("camera", 128, 700), ("pikachu", 256, 2400)):
        model = fixture(image, side, cells)
        args = walk_inputs(model)
        packed = pack_adjacency(args[2], args[4], args[5])
        delta, span, shift = queue_geometry(args[2], args[3], args[1])
        walk = run_packed(args, packed)
        forest = model.walk_with_predecessors(np.ones(model.npix))

        ctx = model.assemble(4.0)
        solved = model.solve_exact(model.base_lab, ctx)
        packed_field = [{
            "w1": ctx["w1"], "w2": ctx["w2"], "softness": 4.0, "scale": 1.0,
            "pred_first": solved["pred_first"],
            "pred_second": solved["pred_second"]}]
        _, sensitivity = model.metric_gradient(packed_field, forest)

        print(f"\n{image} {side} px / {cells} cells "
              f"({2 * model.npix} states)")
        fast_gradient(model, walk, sensitivity, packed[0], delta, shift)
        base_time, base_grad = bench(
            "argsort + eight-way direction search",
            lambda: baseline_gradient(model, forest, sensitivity))
        fast_time, fast_grad = bench(
            "counting sort + tagged direction",
            lambda: fast_gradient(
                model, walk, sensitivity, packed[0], delta, shift))
        print(f"  speedup {base_time / fast_time:.1f}x")
        scale = max(float(np.max(np.abs(base_grad))), 1e-30)
        gap = float(np.max(np.abs(base_grad - fast_grad))) / scale
        print(f"    max relative gradient gap {gap:.3e}")
        assert gap < 1e-12, gap


if __name__ == "__main__":
    main()
