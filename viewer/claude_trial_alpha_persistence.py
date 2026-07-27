"""Trial alpha 2: let the pressure field's own bifurcation structure allocate.

The allocator currently answers "where is the pressure large?" with a robust
cap, a Gaussian blur, an exclusion disk, and a per-cell quota.  Every one of
those is a defence against the same failure: a tall narrow spike is not a
region, and a wide shallow mound is.  Height cannot tell them apart, so the
code suppresses height by hand.

There is a classical quantity that tells them apart with no parameters and no
iteration.  Sweep the field from its maximum downward and maintain the
connected components of the superlevel set with union-find.  A component is
born at a local maximum.  When two components meet they meet at a saddle, and
by the elder rule the younger one dies there; its persistence is the height it
survived alone.  One descending sort and one union-find pass, O(N a(N)), with
no smoothing, no threshold and no scale parameter.

Persistence is the right currency for *where* because it is a statement about
separation rather than amplitude:

  - a one-pixel edge spike is born and dies almost immediately, so the robust
    85th-percentile cap that exists to suppress it becomes unnecessary;
  - a broad under-resolved region has a deep saddle to its neighbours and
    ranks high even when its peak is unremarkable;
  - the saddle that kills a component is exactly where the parent cell is
    trying to be two things, which is a birth location with a reason.

It also subsumes the exclusion disk it replaces.  Two peaks separated by a
deep saddle are distinct components and both deserve budget however close they
are; two peaks on one plateau are one component and deserve one cell however
far apart they are.  That is what a hand-chosen radius approximates.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - the repo venv has numba
    njit = None


OFFSET_Y = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
OFFSET_X = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)


def _find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        nxt = parent[i]
        parent[i] = root
        i = nxt
    return root


def _pairs(order, values, h, w, offset_y, offset_x):
    n = order.size
    parent = np.zeros(n, dtype=np.int64)
    peak = np.zeros(n, dtype=np.int64)
    active = np.zeros(n, dtype=np.uint8)
    out_peak = np.zeros(n, dtype=np.int64)
    out_saddle = np.zeros(n, dtype=np.int64)
    out_pers = np.zeros(n, dtype=np.float64)
    roots = np.zeros(8, dtype=np.int64)
    count = 0
    for k in range(n):
        p = order[k]
        y = p // w
        x = p - y * w
        parent[p] = p
        peak[p] = p
        active[p] = 1
        found = 0
        for t in range(8):
            ny = y + offset_y[t]
            nx = x + offset_x[t]
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            q = ny * w + nx
            if active[q] == 0:
                continue
            r = _find(parent, q)
            seen = False
            for s in range(found):
                if roots[s] == r:
                    seen = True
                    break
            if not seen:
                roots[found] = r
                found += 1
        if found == 0:
            continue
        elder = roots[0]
        best = values[peak[elder]]
        for s in range(1, found):
            value = values[peak[roots[s]]]
            if value > best:
                best = value
                elder = roots[s]
        for s in range(found):
            r = roots[s]
            if r == elder:
                continue
            out_peak[count] = peak[r]
            out_saddle[count] = p
            out_pers[count] = values[peak[r]] - values[p]
            count += 1
            parent[r] = elder
        parent[p] = elder
    return out_peak[:count], out_saddle[:count], out_pers[:count]


if njit is not None:
    _find = njit(cache=True)(_find)
    _pairs = njit(cache=True)(_pairs)


def superlevel_persistence(field):
    """0-dimensional persistence of the superlevel filtration.

    Returns (peak_index, saddle_index, persistence) for every component that
    dies, with the global maximum appended carrying the full range.
    """
    values = np.ascontiguousarray(field, dtype=np.float64).ravel()
    h, w = field.shape
    order = np.ascontiguousarray(np.argsort(values)[::-1].astype(np.int64))
    peaks, saddles, pers = _pairs(order, values, h, w, OFFSET_Y, OFFSET_X)
    root = int(order[0])
    peaks = np.append(peaks, root)
    saddles = np.append(saddles, int(order[-1]))
    pers = np.append(pers, float(values[root] - values.min()))
    return peaks, saddles, pers


def persistent_sites(field, count, existing=None, min_distance=1.5,
                     use_saddle=False):
    """The `count` most persistent components as birth coordinates.

    No exclusion disk and no per-cell quota: distinctness is decided by the
    field's own saddle structure.  `min_distance` only stops a birth landing
    on a site that already exists.
    """
    peaks, saddles, pers = superlevel_persistence(field)
    order = np.argsort(pers)[::-1]
    w = field.shape[1]
    source = saddles if use_saddle else peaks
    chosen = []
    values = []
    guard = min_distance * min_distance
    for k in order:
        index = int(source[k])
        y, x = divmod(index, w)
        if existing is not None and len(existing):
            d2 = (existing[:, 0] - x) ** 2 + (existing[:, 1] - y) ** 2
            if float(d2.min()) < guard:
                continue
        if chosen:
            taken = np.asarray(chosen)
            d2 = (taken[:, 0] - x) ** 2 + (taken[:, 1] - y) ** 2
            if float(d2.min()) < guard:
                continue
        chosen.append((float(x), float(y)))
        values.append(float(pers[k]))
        if len(chosen) >= count:
            break
    return (np.asarray(chosen, dtype=np.float64).reshape(-1, 2),
            np.asarray(values, dtype=np.float64))


def persistence_field(field):
    """Display: each component's peak carries its persistence."""
    peaks, _, pers = superlevel_persistence(field)
    out = np.zeros(field.size, dtype=np.float64)
    out[peaks] = pers
    return out.reshape(field.shape)
