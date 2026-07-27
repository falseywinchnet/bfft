"""PORT 05: constant-time 2x2 generalized support-instability solve.

One independent closed-form eigenpair is evaluated per cell.  There is no
global ordering, candidate list, top-k selection, or cell-pair comparison.
"""

from __future__ import annotations

import numpy as np

from experiments.wasserstein_allocation_tree import (
    _unstable_direction as _reference_unstable_direction,
)


def measure_instability(moments, qxx, qxy, qyy):
    cells = len(qxx)
    major = np.empty(cells, dtype=np.float64)
    minor = np.empty(cells, dtype=np.float64)
    direction = np.empty((cells, 2), dtype=np.float64)
    for cell in range(cells):
        value, vx, vy, other = _reference_unstable_direction(
            moments["cxx"][cell], moments["cxy"][cell],
            moments["cyy"][cell], qxx[cell], qxy[cell], qyy[cell])
        major[cell] = value
        minor[cell] = other
        direction[cell] = (vx, vy)
    return major, minor, direction
