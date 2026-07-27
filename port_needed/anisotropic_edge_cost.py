"""PORT 02: assemble the fixed eight-neighbour transport metric.

This is a streaming stencil, O(pixels), with no cell-count dependence.
Output layout is direction-major ``float32[8, height, width]`` to match the
current exact walk; a native port should benchmark node-major packing too.
"""

from __future__ import annotations

import numpy as np

from experiments.wasserstein_allocation_tree import (
    _edge_cost_stack as _reference_edge_cost_stack,
)


def build_edge_costs(geometry: dict, metric_strength: float) -> np.ndarray:
    return np.ascontiguousarray(
        _reference_edge_cost_stack(geometry, metric_strength),
        dtype=np.float32,
    )
