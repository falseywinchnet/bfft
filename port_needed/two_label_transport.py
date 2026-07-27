"""PORT 03: exact two-label geodesic transport and final hard refresh.

This is the dominant native port.  It is a monotone shortest-path propagation,
not a candidate search.  The exact local refresh experiment has the same
owner/runner/distance contract and can later replace a global refresh when a
source is inserted or moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "viewer", ROOT / "experiments", ROOT / "experiments" / "sigma_opt"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from transport_voronoi import _dijkstra_two_best_packed  # noqa: E402


def walk_two_labels(
    centers: np.ndarray,
    costs: np.ndarray,
    *,
    queue: str = "bucket",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = costs.shape[1:]
    centers = np.asarray(centers, dtype=np.float64)
    seed_x = np.clip(
        np.rint(centers[:, 0] * width - 0.5).astype(np.int32),
        0,
        width - 1,
    )
    seed_y = np.clip(
        np.rint(centers[:, 1] * height - 0.5).astype(np.int32),
        0,
        height - 1,
    )
    reach = np.zeros(len(centers), dtype=np.float64)
    if queue == "bucket":
        from experiments.sigma_opt.opt_dijkstra_bucket import (
            _dijkstra_bucket,
            queue_geometry,
        )

        seed_pixel = seed_y.astype(np.int64) * width + seed_x.astype(np.int64)
        scale = np.ones(height * width, dtype=np.float64)
        delta, span, shift = queue_geometry(costs, scale, reach)
        result = _dijkstra_bucket(
            seed_pixel, reach, costs, scale, height, width,
            delta, span, shift,
        )
        return result[0], result[1], result[2], result[3]
    if queue != "heap":
        raise ValueError(f"unknown exact queue {queue!r}")
    return _dijkstra_two_best_packed(
        seed_x, seed_y, reach, costs, height, width)


def hard_partition(
    centers: np.ndarray,
    costs: np.ndarray,
    *,
    queue: str = "bucket",
) -> np.ndarray:
    owner, _, _, _ = walk_two_labels(centers, costs, queue=queue)
    return owner.reshape(costs.shape[1:])
