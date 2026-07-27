"""PORT 05: constant-time 2x2 generalized support-instability solve.

One independent closed-form eigenpair is evaluated per cell.  There is no
global ordering, candidate list, top-k selection, or cell-pair comparison.
"""

from __future__ import annotations

import numpy as np


def measure_instability(moments, qxx, qxy, qyy):
    cxx = np.asarray(moments["cxx"], dtype=np.float64)
    cxy = np.asarray(moments["cxy"], dtype=np.float64)
    cyy = np.asarray(moments["cyy"], dtype=np.float64)
    qxx = np.asarray(qxx, dtype=np.float64)
    qxy = np.asarray(qxy, dtype=np.float64)
    qyy = np.asarray(qyy, dtype=np.float64)
    a = cxx * qxx + cxy * qxy
    b = cxx * qxy + cxy * qyy
    c = cxy * qxx + cyy * qxy
    d = cxy * qxy + cyy * qyy
    trace = a + d
    determinant = np.maximum(a * d - b * c, 0.0)
    disc = np.sqrt(np.maximum(
        trace * trace - 4.0 * determinant, 0.0))
    major = np.maximum(0.5 * (trace + disc), 0.0)
    minor = np.maximum(trace - major, 0.0)

    vx = b.copy()
    vy = major - a
    fallback = (np.abs(vx) + np.abs(vy)) < 1e-15
    vx = np.where(fallback, major - d, vx)
    vy = np.where(fallback, c, vy)
    norm = np.hypot(vx, vy)
    degenerate = norm < 1e-15
    safe = np.maximum(norm, 1e-300)
    vx = np.where(
        degenerate, np.where(cxx >= cyy, 1.0, 0.0), vx / safe)
    vy = np.where(
        degenerate, np.where(cxx >= cyy, 0.0, 1.0), vy / safe)
    return major, minor, np.column_stack((vx, vy))
