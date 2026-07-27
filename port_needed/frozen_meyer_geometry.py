"""PORT 01: one frozen Meyer support measure and physical metric.

C++ input
    contiguous RGB image plus Meyer/support scalar parameters.
C++ output
    float32 measure, Qxx, Qxy, Qyy, cartoon, texture, glass and energy.

The target is decomposed once.  Restriction to the allocation grid samples
that finished geometry; it must never decompose a resized or changed target.
"""

from __future__ import annotations

import numpy as np

from experiments.wasserstein_allocation_tree import (
    pyramid_geometry as _pyramid_geometry,
    single_decomposition_geometry as _single_decomposition_geometry,
)


def build_frozen_geometry(rgb: np.ndarray, **parameters) -> dict:
    parameters.setdefault("meyer_solver", 1)
    geometry = _single_decomposition_geometry(rgb, **parameters)
    # The transport walk is bandwidth-bound.  Preserve float64 scalars, but
    # make image fields explicit float32 port targets.
    for key, value in tuple(geometry.items()):
        if isinstance(value, np.ndarray):
            geometry[key] = np.ascontiguousarray(value, dtype=np.float32)
    return geometry


def restrict_geometry(geometry: dict, maximum_side: int) -> dict:
    return _pyramid_geometry(geometry, int(maximum_side))
