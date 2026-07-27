#!/usr/bin/env python3
"""Exact incremental topology refresh for two-label geodesic cells.

The Meyer/TGFD scope law says to retain state on a fixed substrate while
refreshing the active topology after each useful event.  For the cell system
the fixed operator is min-plus geodesic closure.  If existing sites do not
move, inserting new sites is a monotone change: old first/second distances
are valid upper bounds, and only pixels improved by a new label need enter
the queue.

This experiment seeds a batch of new labels into an already-settled
owner/runner solution and applies exactly the production relaxation rules.
It verifies the result against a fresh global walk and measures the fraction
of the image and queue work touched by the topology event.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

from transport_voronoi import _dijkstra_two_best_packed, _fit_rgb  # noqa: E402
from transport_voronoi import srgb_to_lab  # noqa: E402
from bfft.vision import SingleStageDecompositionObjective  # noqa: E402
from wasserstein_allocation_tree import (  # noqa: E402
    _balanced_branch_barycentres,
    _edge_cost_stack,
    _physical_precision,
    _soft_transport_moments,
    _unstable_direction,
    bifurcate_allocation,
    fit_hard_regions_with_ridge,
    single_decomposition_geometry,
)


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


@_compile
def insert_sites_incremental(
    owner_in: np.ndarray,
    runner_in: np.ndarray,
    first_in: np.ndarray,
    second_in: np.ndarray,
    new_seed_x: np.ndarray,
    new_seed_y: np.ndarray,
    new_reach: np.ndarray,
    new_site_id: np.ndarray,
    costs: np.ndarray,
    height: int,
    width: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Insert a batch of sources into an exact settled two-label solution.

    This is the production heap relaxation with two differences only:

    1. the distance and label arrays begin at the previous fixed point;
    2. the queue begins with only the newly inserted sources.

    A new label that fails to improve the two best distances at a pixel is
    safely pruned there.  Every source distance and the second order statistic
    are 1-Lipschitz on the same weighted graph, so it cannot become top-two
    after crossing a non-improved pixel.
    """
    owner = owner_in.copy()
    runner = runner_in.copy()
    best = first_in.copy()
    second = second_in.copy()
    pixels = height * width
    changed = np.zeros(pixels, dtype=np.uint8)

    capacity = max(4 * pixels + len(new_seed_x) + 256, 512)
    heap_distance = np.empty(capacity, dtype=np.float64)
    heap_pixel = np.empty(capacity, dtype=np.int32)
    heap_site = np.empty(capacity, dtype=np.int32)
    size = 0
    pushes = 0
    pops = 0

    for local_site in range(len(new_seed_x)):
        site = new_site_id[local_site]
        pixel = new_seed_y[local_site] * width + new_seed_x[local_site]
        distance = -new_reach[local_site]
        if owner[pixel] == site:
            if distance < best[pixel]:
                best[pixel] = distance
                changed[pixel] = 1
        elif runner[pixel] == site:
            if distance < second[pixel]:
                second[pixel] = distance
                if second[pixel] < best[pixel]:
                    best[pixel], second[pixel] = (
                        second[pixel], best[pixel])
                    owner[pixel], runner[pixel] = (
                        runner[pixel], owner[pixel])
                changed[pixel] = 1
        elif distance < best[pixel]:
            second[pixel], runner[pixel] = (
                best[pixel], owner[pixel])
            best[pixel], owner[pixel] = distance, site
            changed[pixel] = 1
        elif distance < second[pixel]:
            second[pixel], runner[pixel] = distance, site
            changed[pixel] = 1

        heap_distance[size] = distance
        heap_pixel[size] = pixel
        heap_site[size] = site
        child = size
        size += 1
        pushes += 1
        while child > 0:
            parent = (child - 1) // 2
            if heap_distance[parent] <= heap_distance[child]:
                break
            heap_distance[parent], heap_distance[child] = (
                heap_distance[child], heap_distance[parent])
            heap_pixel[parent], heap_pixel[child] = (
                heap_pixel[child], heap_pixel[parent])
            heap_site[parent], heap_site[child] = (
                heap_site[child], heap_site[parent])
            child = parent

    dys = (-1, 1, 0, 0, -1, -1, 1, 1)
    dxs = (0, 0, -1, 1, -1, 1, -1, 1)
    tolerance = 1e-12
    while size > 0:
        distance = heap_distance[0]
        pixel = heap_pixel[0]
        site = heap_site[0]
        size -= 1
        pops += 1
        heap_distance[0] = heap_distance[size]
        heap_pixel[0] = heap_pixel[size]
        heap_site[0] = heap_site[size]
        node = 0
        while True:
            left = 2 * node + 1
            right = left + 1
            smallest = node
            if (
                left < size
                and heap_distance[left] < heap_distance[smallest]
            ):
                smallest = left
            if (
                right < size
                and heap_distance[right] < heap_distance[smallest]
            ):
                smallest = right
            if smallest == node:
                break
            heap_distance[node], heap_distance[smallest] = (
                heap_distance[smallest], heap_distance[node])
            heap_pixel[node], heap_pixel[smallest] = (
                heap_pixel[smallest], heap_pixel[node])
            heap_site[node], heap_site[smallest] = (
                heap_site[smallest], heap_site[node])
            node = smallest

        valid = (
            (owner[pixel] == site
             and distance <= best[pixel] + tolerance)
            or
            (runner[pixel] == site
             and distance <= second[pixel] + tolerance)
        )
        if not valid:
            continue

        y = pixel // width
        x = pixel - y * width
        for direction in range(8):
            ny = y + dys[direction]
            nx = x + dxs[direction]
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            q = ny * width + nx
            candidate = distance + costs[direction, y, x]
            touched = False
            if owner[q] == site:
                if candidate + tolerance < best[q]:
                    best[q] = candidate
                    touched = True
            elif runner[q] == site:
                if candidate + tolerance < second[q]:
                    second[q] = candidate
                    if second[q] < best[q]:
                        best[q], second[q] = second[q], best[q]
                        owner[q], runner[q] = runner[q], owner[q]
                    touched = True
            elif candidate + tolerance < best[q]:
                second[q], runner[q] = best[q], owner[q]
                best[q], owner[q] = candidate, site
                touched = True
            elif candidate + tolerance < second[q]:
                second[q], runner[q] = candidate, site
                touched = True
            if not touched:
                continue

            changed[q] = 1
            if size >= capacity:
                capacity *= 2
                new_distance = np.empty(capacity, dtype=np.float64)
                new_pixel = np.empty(capacity, dtype=np.int32)
                new_site = np.empty(capacity, dtype=np.int32)
                new_distance[:size] = heap_distance[:size]
                new_pixel[:size] = heap_pixel[:size]
                new_site[:size] = heap_site[:size]
                heap_distance, heap_pixel, heap_site = (
                    new_distance, new_pixel, new_site)
            heap_distance[size] = candidate
            heap_pixel[size] = q
            heap_site[size] = site
            child = size
            size += 1
            pushes += 1
            while child > 0:
                parent = (child - 1) // 2
                if heap_distance[parent] <= heap_distance[child]:
                    break
                heap_distance[parent], heap_distance[child] = (
                    heap_distance[child], heap_distance[parent])
                heap_pixel[parent], heap_pixel[child] = (
                    heap_pixel[child], heap_pixel[parent])
                heap_site[parent], heap_site[child] = (
                    heap_site[child], heap_site[parent])
                child = parent

    return owner, runner, best, second, changed, pushes, pops


def _seeds(
    centers: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.clip(
            np.rint(centers[:, 0] * width - 0.5).astype(np.int32),
            0,
            width - 1,
        ),
        np.clip(
            np.rint(centers[:, 1] * height - 0.5).astype(np.int32),
            0,
            height - 1,
        ),
    )


def _walk(
    centers: np.ndarray,
    costs: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed_x, seed_y = _seeds(centers, height, width)
    return _dijkstra_two_best_packed(
        seed_x,
        seed_y,
        np.zeros(len(centers), dtype=np.float64),
        costs,
        height,
        width,
    )


def _walk_label_anchors(
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
    anchor_site: np.ndarray,
    cells: int,
    costs: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full oracle walk allowing several source anchors per cell label."""
    pixels = height * width
    owner = np.full(pixels, -1, dtype=np.int32)
    runner = np.full(pixels, -1, dtype=np.int32)
    first = np.full(pixels, 1e300, dtype=np.float64)
    second = np.full(pixels, 1e300, dtype=np.float64)
    solved = insert_sites_incremental(
        owner,
        runner,
        first,
        second,
        np.asarray(anchor_x, dtype=np.int32),
        np.asarray(anchor_y, dtype=np.int32),
        np.zeros(len(anchor_x), dtype=np.float64),
        np.asarray(anchor_site, dtype=np.int32),
        costs,
        height,
        width,
    )
    if np.max(solved[0]) >= cells:
        raise RuntimeError("anchor label exceeds declared cell count")
    return solved[0], solved[1], solved[2], solved[3]


def compare_event(
    centers: np.ndarray,
    old_count: int,
    new_count: int,
    costs: np.ndarray,
    height: int,
    width: int,
    *,
    repeats: int = 3,
) -> dict:
    """Compare one insertion event to a complete global refresh."""
    old_count = max(1, min(int(old_count), len(centers)))
    new_count = max(old_count, min(int(new_count), len(centers)))
    old_centers = centers[:old_count]
    all_centers = centers[:new_count]
    owner, runner, first, second = _walk(
        old_centers, costs, height, width)
    new_x, new_y = _seeds(
        centers[old_count:new_count], height, width)
    reach = np.zeros(new_count - old_count, dtype=np.float64)

    # Compile before measuring.
    incremental = insert_sites_incremental(
        owner,
        runner,
        first,
        second,
        new_x,
        new_y,
        reach,
        np.arange(old_count, new_count, dtype=np.int32),
        costs,
        height,
        width,
    )
    reference = _walk(all_centers, costs, height, width)

    incremental_times = []
    full_times = []
    for _ in range(max(int(repeats), 1)):
        started = time.perf_counter()
        incremental = insert_sites_incremental(
            owner,
            runner,
            first,
            second,
            new_x,
            new_y,
            reach,
            np.arange(old_count, new_count, dtype=np.int32),
            costs,
            height,
            width,
        )
        incremental_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        reference = _walk(all_centers, costs, height, width)
        full_times.append(time.perf_counter() - started)

    trial_owner, trial_runner, trial_first, trial_second = incremental[:4]
    ref_owner, ref_runner, ref_first, ref_second = reference
    finite_first = np.isfinite(ref_first) & (ref_first < 1e299)
    finite_second = np.isfinite(ref_second) & (ref_second < 1e299)
    first_gap = float(np.max(np.abs(
        trial_first[finite_first] - ref_first[finite_first])))
    second_gap = float(np.max(np.abs(
        trial_second[finite_second] - ref_second[finite_second])))

    owner_mismatch = trial_owner != ref_owner
    runner_mismatch = trial_runner != ref_runner
    owner_tied = np.abs(ref_first - ref_second) <= 1e-9
    runner_tied = (
        np.abs(trial_second - ref_second) <= 1e-9)
    changed = incremental[4]
    incremental_seconds = float(min(incremental_times))
    full_seconds = float(min(full_times))
    return {
        "old_sites": old_count,
        "new_sites": new_count,
        "inserted_sites": new_count - old_count,
        "first_distance_gap": first_gap,
        "second_distance_gap": second_gap,
        "owner_mismatch": int(np.count_nonzero(owner_mismatch)),
        "owner_mismatch_untied": int(np.count_nonzero(
            owner_mismatch & ~owner_tied)),
        "runner_mismatch": int(np.count_nonzero(runner_mismatch)),
        "runner_mismatch_distance_disagreement": int(np.count_nonzero(
            runner_mismatch & ~runner_tied)),
        "affected_pixels": int(np.count_nonzero(changed)),
        "affected_fraction": float(np.mean(changed)),
        "queue_pushes": int(incremental[5]),
        "queue_pops": int(incremental[6]),
        "incremental_seconds": incremental_seconds,
        "full_seconds": full_seconds,
        "speedup": (
            full_seconds / incremental_seconds
            if incremental_seconds > 0.0 else math.inf
        ),
    }


def random_exactness_trials(
    costs: np.ndarray,
    height: int,
    width: int,
    *,
    trials: int = 12,
) -> list[dict]:
    """Verify insertions at random site sets before trusting image timings."""
    rng = np.random.default_rng(7127)
    reports = []
    for trial in range(max(int(trials), 1)):
        old_count = int(rng.integers(2, 25))
        added = int(rng.integers(1, 12))
        centers = np.column_stack([
            rng.uniform(0.5 / width, 1.0 - 0.5 / width,
                        old_count + added),
            rng.uniform(0.5 / height, 1.0 - 0.5 / height,
                        old_count + added),
        ])
        report = compare_event(
            centers,
            old_count,
            old_count + added,
            costs,
            height,
            width,
            repeats=1,
        )
        report["trial"] = trial
        reports.append(report)
    return reports


def monotone_interleaved_allocation(
    geometry: dict,
    *,
    threshold: float = 4.0,
    metric_extent_threshold: float = 8.0,
    softness_start: float = 0.20,
    softness_end: float = 0.0025,
    minimum_region_pixels: int = 12,
    maximum_rounds: int = 32,
    safety_cells: int = 4096,
    metric_strength: float = 1.5,
    balance_steps: int = 14,
    grow_parent_anchor: bool = False,
    initial_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, dict, list[dict]]:
    """Allocation by exact source insertion with no established-site motion.

    A stressed support contributes one new site at the farther of its two
    mass-balanced branch barycentres.  The old site remains fixed, making the
    source change monotone and the incremental topology update exact.  No
    child inherits pixels; owner/runner are always the global min-plus result.
    """
    measure_2d = np.asarray(geometry["measure"], dtype=np.float64)
    height, width = measure_2d.shape
    pixels = height * width
    yy, xx = np.mgrid[:height, :width]
    x = (xx.ravel().astype(np.float64) + 0.5) / width
    y = (yy.ravel().astype(np.float64) + 0.5) / height
    measure = measure_2d.ravel()
    qxx_p, qxy_p, qyy_p = _physical_precision(
        np.asarray(geometry["precision_xx"], dtype=np.float64),
        np.asarray(geometry["precision_xy"], dtype=np.float64),
        np.asarray(geometry["precision_yy"], dtype=np.float64),
        width,
        height,
    )
    qxx_p = qxx_p.ravel()
    qxy_p = qxy_p.ravel()
    qyy_p = qyy_p.ravel()
    costs = _edge_cost_stack(geometry, metric_strength)
    if initial_centers is None:
        centers = np.array([[
            float(np.sum(measure * x)),
            float(np.sum(measure * y)),
        ]], dtype=np.float64)
    else:
        centers = np.asarray(initial_centers, dtype=np.float64).copy()
        centers[:, 0] = np.clip(
            centers[:, 0], 0.5 / width, 1.0 - 0.5 / width)
        centers[:, 1] = np.clip(
            centers[:, 1], 0.5 / height, 1.0 - 0.5 / height)
    initial_x, initial_y = _seeds(centers, height, width)
    anchor_x = initial_x.copy()
    anchor_y = initial_y.copy()
    anchor_site = np.arange(len(centers), dtype=np.int32)
    owner, runner, first, second = _walk_label_anchors(
        anchor_x,
        anchor_y,
        anchor_site,
        len(centers),
        costs,
        height,
        width,
    )
    trace: list[dict] = []
    incremental_seconds = 0.0
    reduction_seconds = 0.0

    for round_index in range(max(int(maximum_rounds), 1)):
        cells = len(centers)
        progress = round_index / max(int(maximum_rounds) - 1, 1)
        softness = (
            float(softness_start)
            * (float(softness_end) / float(softness_start)) ** progress)
        temperature = (
            softness * max(float(geometry["max_support_px"]), 1.0))
        started = time.perf_counter()
        pixel_count = np.bincount(owner, minlength=cells)
        moments, qxx, qxy, qyy = _soft_transport_moments(
            owner,
            runner,
            first,
            second,
            temperature,
            measure,
            x,
            y,
            qxx_p,
            qxy_p,
            qyy_p,
            cells,
        )
        instability = np.zeros(cells, dtype=np.float64)
        direction = np.zeros((cells, 2), dtype=np.float64)
        for cell in range(cells):
            value, vx, vy, _ = _unstable_direction(
                moments["cxx"][cell],
                moments["cxy"][cell],
                moments["cyy"][cell],
                qxx[cell],
                qxy[cell],
                qyy[cell],
            )
            instability[cell] = value
            direction[cell] = (vx, vy)
        transport_extent = (
            moments["transport_rms"]
            / max(float(geometry["max_support_px"]), 1e-12)
        )
        split = (
            (
                (transport_extent > float(threshold))
                | (instability > float(metric_extent_threshold))
            )
            & (
                pixel_count
                >= 2 * max(int(minimum_region_pixels), 1)
            )
            & (moments["mass"] > 1e-8)
        )
        split_ids = np.flatnonzero(split)
        if len(centers) + len(split_ids) > int(safety_cells):
            break
        if not len(split_ids):
            reduction_seconds += time.perf_counter() - started
            trace.append({
                "round": round_index + 1,
                "cells_before": cells,
                "inserted": 0,
                "cells_after": cells,
                "affected_fraction": 0.0,
                "reduction_seconds": (
                    time.perf_counter() - started),
                "incremental_seconds": 0.0,
                "instability_max": float(np.max(instability)),
            })
            break

        negative, positive, valid = _balanced_branch_barycentres(
            owner,
            runner,
            first,
            second,
            temperature,
            measure,
            x,
            y,
            moments,
            direction,
            split,
            balance_steps=balance_steps,
        )
        split_ids = np.flatnonzero(valid)
        negative_distance = np.hypot(
            (negative[split_ids, 0] - centers[split_ids, 0]) * width,
            (negative[split_ids, 1] - centers[split_ids, 1]) * height,
        )
        positive_distance = np.hypot(
            (positive[split_ids, 0] - centers[split_ids, 0]) * width,
            (positive[split_ids, 1] - centers[split_ids, 1]) * height,
        )
        separation = np.maximum(
            negative_distance, positive_distance)
        valid_separation = separation >= 0.75
        split_ids = split_ids[valid_separation]
        negative_branch = negative[split_ids].copy()
        positive_branch = positive[split_ids].copy()
        negative_distance = negative_distance[valid_separation]
        positive_distance = positive_distance[valid_separation]
        choose_positive_child = (
            positive_distance >= negative_distance)
        child_candidates = negative_branch.copy()
        parent_candidates = positive_branch.copy()
        child_candidates[choose_positive_child] = (
            positive_branch[choose_positive_child])
        parent_candidates[choose_positive_child] = (
            negative_branch[choose_positive_child])
        for candidates in (child_candidates, parent_candidates):
            candidates[:, 0] = np.clip(
                candidates[:, 0], 0.5 / width, 1.0 - 0.5 / width)
            candidates[:, 1] = np.clip(
                candidates[:, 1], 0.5 / height, 1.0 - 0.5 / height)
        reduction_elapsed = time.perf_counter() - started
        reduction_seconds += reduction_elapsed
        if not len(child_candidates):
            break

        child_ids = np.arange(
            cells, cells + len(child_candidates), dtype=np.int32)
        if grow_parent_anchor:
            parent_x, parent_y = _seeds(
                parent_candidates, height, width)
            child_x, child_y = _seeds(
                child_candidates, height, width)
            new_x = np.concatenate([parent_x, child_x])
            new_y = np.concatenate([parent_y, child_y])
            new_ids = np.concatenate([
                split_ids.astype(np.int32), child_ids])
        else:
            new_x, new_y = _seeds(
                child_candidates, height, width)
            new_ids = child_ids
        started = time.perf_counter()
        inserted = insert_sites_incremental(
            owner,
            runner,
            first,
            second,
            new_x,
            new_y,
            np.zeros(len(new_x), dtype=np.float64),
            new_ids,
            costs,
            height,
            width,
        )
        event_seconds = time.perf_counter() - started
        incremental_seconds += event_seconds
        owner, runner, first, second = inserted[:4]
        anchor_x = np.concatenate([anchor_x, new_x])
        anchor_y = np.concatenate([anchor_y, new_y])
        anchor_site = np.concatenate([anchor_site, new_ids])
        centers = np.vstack([centers, child_candidates])
        trace.append({
            "round": round_index + 1,
            "cells_before": cells,
            "inserted": int(len(child_candidates)),
            "anchors_added": int(len(new_x)),
            "cells_after": int(len(centers)),
            "affected_fraction": float(np.mean(inserted[4])),
            "queue_pushes": int(inserted[5]),
            "reduction_seconds": reduction_elapsed,
            "incremental_seconds": event_seconds,
            "instability_max": float(np.max(instability)),
        })

    # This full walk is a verification oracle, not part of the allocation.
    reference = _walk_label_anchors(
        anchor_x,
        anchor_y,
        anchor_site,
        len(centers),
        costs,
        height,
        width,
    )
    finite_first = reference[2] < 1e299
    finite_second = reference[3] < 1e299
    first_gap = float(np.max(np.abs(
        first[finite_first] - reference[2][finite_first])))
    second_gap = float(np.max(np.abs(
        second[finite_second] - reference[3][finite_second])))
    second_gap_index = int(np.argmax(np.where(
        finite_second,
        np.abs(second - reference[3]),
        -1.0,
    )))
    mass = np.bincount(
        owner, weights=measure, minlength=len(centers))
    safe_mass = np.maximum(mass, 1e-30)
    centers[:, 0] = np.bincount(
        owner, weights=measure * x, minlength=len(centers)) / safe_mass
    centers[:, 1] = np.bincount(
        owner, weights=measure * y, minlength=len(centers)) / safe_mass
    return owner.reshape(height, width), {
        "centers": centers,
        "cells": int(len(centers)),
        "incremental_seconds": incremental_seconds,
        "reduction_seconds": reduction_seconds,
        "first_distance_gap": first_gap,
        "second_distance_gap": second_gap,
        "second_gap_probe": {
            "pixel": second_gap_index,
            "incremental_owner": int(owner[second_gap_index]),
            "incremental_runner": int(runner[second_gap_index]),
            "incremental_first": float(first[second_gap_index]),
            "incremental_second": float(second[second_gap_index]),
            "reference_owner": int(reference[0][second_gap_index]),
            "reference_runner": int(reference[1][second_gap_index]),
            "reference_first": float(reference[2][second_gap_index]),
            "reference_second": float(reference[3][second_gap_index]),
        },
        "anchors": int(len(anchor_x)),
        "grow_parent_anchor": bool(grow_parent_anchor),
        "anchor_state": (anchor_x, anchor_y, anchor_site),
    }, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", nargs="?",
        default=str(Path.home() / "Downloads/25.png"))
    parser.add_argument("--side", type=int, default=475)
    parser.add_argument("--metric-limit", type=float, default=8.0)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--monotone-rounds", type=int, default=32)
    parser.add_argument("--hybrid-round", type=int, default=13)
    parser.add_argument("--ridges", type=int, default=1)
    parser.add_argument("--random-trials", type=int, default=12)
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/incremental_topology.json")
    args = parser.parse_args()

    from skimage.io import imread

    rgb = _fit_rgb(imread(Path(args.image).expanduser()), args.side)
    geometry_started = time.perf_counter()
    geometry = single_decomposition_geometry(rgb)
    geometry_seconds = time.perf_counter() - geometry_started
    costs = _edge_cost_stack(geometry, metric_strength=1.5)
    height, width = rgb.shape[:2]

    random_reports = random_exactness_trials(
        costs,
        height,
        width,
        trials=args.random_trials,
    )
    for report in random_reports:
        assert report["first_distance_gap"] < 1e-9, report
        assert report["second_distance_gap"] < 1e-9, report
        assert report["owner_mismatch_untied"] == 0, report
        assert (
            report["runner_mismatch_distance_disagreement"] == 0
        ), report

    allocation_started = time.perf_counter()
    baseline_labels, allocation, trace = bifurcate_allocation(
        geometry,
        metric_extent_threshold=args.metric_limit,
        maximum_rounds=args.rounds,
        transport_queue="heap",
        capture_center_history=True,
    )
    allocation_seconds = time.perf_counter() - allocation_started
    centers = np.asarray(allocation["centers"], dtype=np.float64)

    # Final centre positions preserve binary birth order: an existing parent
    # retains its index and each positive branch is appended.  Sampling the
    # recorded population transitions therefore gives realistic site
    # densities while holding existing sources fixed, which is the exact
    # monotone-insertion case being tested.
    event_indices = np.linspace(
        2, max(len(trace) - 1, 2), 7, dtype=int)
    event_indices = np.unique(np.clip(
        event_indices, 1, len(trace) - 1))
    events = []
    for index in event_indices:
        old_count = int(trace[index]["cells_before"])
        new_count = int(trace[index]["cells_after"])
        if new_count <= old_count:
            continue
        report = compare_event(
            centers,
            old_count,
            new_count,
            costs,
            height,
            width,
            repeats=3,
        )
        report["allocation_round"] = int(trace[index]["round"])
        assert report["first_distance_gap"] < 1e-9, report
        assert report["second_distance_gap"] < 1e-9, report
        assert report["owner_mismatch_untied"] == 0, report
        assert (
            report["runner_mismatch_distance_disagreement"] == 0
        ), report
        events.append(report)

    objective = SingleStageDecompositionObjective(rgb, passes=24)
    baseline_record, _, _ = fit_hard_regions_with_ridge(
        baseline_labels,
        centers,
        srgb_to_lab(rgb),
        objective,
        ridge_count=args.ridges,
    )
    monotone_results = {}
    monotone_print = {}
    for name, grow_parent_anchor in (
        ("point_insertion", False),
        ("support_anchor_growth", True),
    ):
        monotone_started = time.perf_counter()
        monotone_labels, monotone, monotone_trace = (
            monotone_interleaved_allocation(
                geometry,
                metric_extent_threshold=args.metric_limit,
                maximum_rounds=args.monotone_rounds,
                grow_parent_anchor=grow_parent_anchor,
            )
        )
        monotone_wall_seconds = time.perf_counter() - monotone_started
        monotone_record, _, _ = fit_hard_regions_with_ridge(
            monotone_labels,
            monotone["centers"],
            srgb_to_lab(rgb),
            objective,
            ridge_count=args.ridges,
        )
        monotone_results[name] = {
            **{
                key: value
                for key, value in monotone.items()
                if key not in ("centers", "anchor_state")
            },
            "wall_seconds_with_verification": monotone_wall_seconds,
            "record": {
                key: float(value)
                for key, value in monotone_record.items()
                if key != "rgb"
            },
            "trace": monotone_trace,
        }
        monotone_print[name] = {
            "cells": monotone["cells"],
            "anchors": monotone["anchors"],
            "psnr": float(monotone_record["psnr"]),
            "objective": float(monotone_record["objective"]),
            "allocation_seconds": (
                monotone["incremental_seconds"]
                + monotone["reduction_seconds"]
            ),
            "wall_seconds_with_verification": monotone_wall_seconds,
            "first_gap": monotone["first_distance_gap"],
            "second_gap": monotone["second_distance_gap"],
        }

    history = allocation["center_history"] or []
    hybrid_cutoff = min(
        max(int(args.hybrid_round), 1),
        max(len(history) - 1, 1),
    )
    if history and hybrid_cutoff < len(trace):
        hybrid_started = time.perf_counter()
        hybrid_labels, hybrid, hybrid_trace = (
            monotone_interleaved_allocation(
                geometry,
                metric_extent_threshold=args.metric_limit,
                maximum_rounds=max(
                    int(args.rounds) - hybrid_cutoff, 1),
                softness_start=float(
                    trace[hybrid_cutoff]["softness"]),
                softness_end=0.0025,
                grow_parent_anchor=False,
                initial_centers=history[hybrid_cutoff - 1],
            )
        )
        hybrid_wall_seconds = time.perf_counter() - hybrid_started
        hybrid_record, _, _ = fit_hard_regions_with_ridge(
            hybrid_labels,
            hybrid["centers"],
            srgb_to_lab(rgb),
            objective,
            ridge_count=args.ridges,
        )
        monotone_results["mature_graph_hybrid"] = {
            **{
                key: value
                for key, value in hybrid.items()
                if key not in ("centers", "anchor_state")
            },
            "baseline_rounds": hybrid_cutoff,
            "continuation_wall_seconds_with_verification": (
                hybrid_wall_seconds),
            "record": {
                key: float(value)
                for key, value in hybrid_record.items()
                if key != "rgb"
            },
            "trace": hybrid_trace,
        }
        monotone_print["mature_graph_hybrid"] = {
            "baseline_rounds": hybrid_cutoff,
            "cells": hybrid["cells"],
            "psnr": float(hybrid_record["psnr"]),
            "objective": float(hybrid_record["objective"]),
            "continuation_seconds": (
                hybrid["incremental_seconds"]
                + hybrid["reduction_seconds"]
            ),
            "first_gap": hybrid["first_distance_gap"],
            "second_gap": hybrid["second_distance_gap"],
        }

    result = {
        "source": str(Path(args.image).expanduser()),
        "shape": list(rgb.shape),
        "geometry_seconds": geometry_seconds,
        "allocation_seconds": allocation_seconds,
        "final_cells": int(len(centers)),
        "baseline_record": {
            key: float(value)
            for key, value in baseline_record.items()
            if key != "rgb"
        },
        "metric_limit": float(args.metric_limit),
        "random_trials": random_reports,
        "events": events,
        "monotone": monotone_results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "random_trials": len(random_reports),
        "maximum_random_first_gap": max(
            item["first_distance_gap"] for item in random_reports),
        "maximum_random_second_gap": max(
            item["second_distance_gap"] for item in random_reports),
        "final_cells": len(centers),
        "baseline": {
            "cells": len(centers),
            "psnr": float(baseline_record["psnr"]),
            "objective": float(baseline_record["objective"]),
        },
        "monotone": monotone_print,
        "events": [
            {
                "round": item["allocation_round"],
                "sites": (
                    f"{item['old_sites']}->{item['new_sites']}"),
                "affected_percent": 100.0 * item["affected_fraction"],
                "speedup": item["speedup"],
                "first_gap": item["first_distance_gap"],
                "second_gap": item["second_distance_gap"],
            }
            for item in events
        ],
        "json": str(args.json),
    }, indent=2))


if __name__ == "__main__":
    main()
