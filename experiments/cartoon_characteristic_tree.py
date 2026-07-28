#!/usr/bin/env python3
"""Finite characteristic-tree proposal for the isotropic cartoon stage.

The proposal has no convergence loop:

1. freeze the early isotropic-dual directions,
2. make a direction-adapted spanning tree,
3. solve quadratic tree-TV exactly by bottom-up piecewise-linear messages,
4. score (and optionally accept) the result with the original isotropic ROF
   objective.

This is deliberately an experiment, not a replacement for isotropic TV.  The
tree problem is a conditional analytical proposal whose merit is measured only
in the original objective.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import zoom
from scipy.sparse.csgraph import minimum_spanning_tree

import cartoon_radial_direction as radial


def isotropic_tv(u: np.ndarray) -> float:
    gx, gy = radial._grad(u)
    return float(np.sum(np.hypot(gx, gy)))


class _EventTreap:
    """Destructive meldable ordered slope-event store.

    An event (x, ds) contributes ``ds * max(t - x, 0)`` to a derivative
    message.  Unique serial numbers allow coincident analytical breakpoints.
    """

    def __init__(self) -> None:
        self.x = [0.0]
        self.ds = [0.0]
        self.serial = [0]
        self.priority = [0]
        self.left = [0]
        self.right = [0]
        self.size = [0]
        self.sum_ds = [0.0]
        self.sum_dsx = [0.0]
        self._next_serial = 1

    @staticmethod
    def _priority(serial: int) -> int:
        # SplitMix64: deterministic, well-distributed treap priorities.
        z = (serial + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return int(z ^ (z >> 31))

    def _new(self, x: float, ds: float) -> int:
        i = len(self.x)
        serial = self._next_serial
        self._next_serial += 1
        self.x.append(float(x))
        self.ds.append(float(ds))
        self.serial.append(serial)
        self.priority.append(self._priority(serial))
        self.left.append(0)
        self.right.append(0)
        self.size.append(1)
        self.sum_ds.append(float(ds))
        self.sum_dsx.append(float(ds * x))
        return i

    def _pull(self, i: int) -> None:
        l = self.left[i]
        r = self.right[i]
        self.size[i] = 1 + self.size[l] + self.size[r]
        self.sum_ds[i] = self.ds[i] + self.sum_ds[l] + self.sum_ds[r]
        self.sum_dsx[i] = (
            self.ds[i] * self.x[i] + self.sum_dsx[l] + self.sum_dsx[r]
        )

    def _less(self, i: int, key_x: float, key_serial: int) -> bool:
        return self.x[i] < key_x or (
            self.x[i] == key_x and self.serial[i] < key_serial
        )

    def split(self, root: int, key_x: float, key_serial: int) -> tuple[int, int]:
        """Return events with keys < key and events with keys >= key."""
        if root == 0:
            return 0, 0
        if self._less(root, key_x, key_serial):
            a, b = self.split(self.right[root], key_x, key_serial)
            self.right[root] = a
            self._pull(root)
            return root, b
        a, b = self.split(self.left[root], key_x, key_serial)
        self.left[root] = b
        self._pull(root)
        return a, root

    def unite(self, a: int, b: int) -> int:
        """Destructively meld two event sets."""
        if a == 0:
            return b
        if b == 0:
            return a
        if self.priority[a] < self.priority[b]:
            a, b = b, a
        bl, br = self.split(b, self.x[a], self.serial[a])
        self.left[a] = self.unite(self.left[a], bl)
        self.right[a] = self.unite(self.right[a], br)
        self._pull(a)
        return a

    def insert(self, root: int, x: float, ds: float) -> int:
        if abs(ds) < 1e-14:
            return root
        return self.unite(root, self._new(x, ds))

    def prefix_ds(self, root: int, x: float, *, inclusive: bool) -> float:
        """Sum slope changes at coordinates < x (or <= x)."""
        total = 0.0
        i = root
        while i:
            before = self.x[i] < x or (inclusive and self.x[i] == x)
            if before:
                total += self.sum_ds[self.left[i]] + self.ds[i]
                i = self.right[i]
            else:
                i = self.left[i]
        return total

    def find_crossing(
        self,
        root: int,
        target: float,
        base_slope: float,
        base_intercept: float,
    ) -> float:
        """Analytical root of a strictly increasing continuous PL function."""

        def descend(i: int, slope: float, intercept: float) -> float:
            l = self.left[i]
            slope_at = slope + self.sum_ds[l]
            intercept_at = intercept - self.sum_dsx[l]
            value_at = slope_at * self.x[i] + intercept_at

            if target <= value_at:
                if l:
                    return descend(l, slope, intercept)
                return (target - intercept) / slope

            slope_after = slope_at + self.ds[i]
            intercept_after = intercept_at - self.ds[i] * self.x[i]
            r = self.right[i]
            if r:
                return descend(r, slope_after, intercept_after)
            return (target - intercept_after) / slope_after

        if root == 0:
            return (target - base_intercept) / base_slope
        return descend(root, base_slope, base_intercept)

    def clip_message(
        self,
        root: int,
        lo: float,
        hi: float,
        base_slope: float,
    ) -> int:
        """Turn h into clip(h,-w,w), retaining only its slope events."""
        slope_right_lo = base_slope + self.prefix_ds(root, lo, inclusive=True)
        slope_left_hi = base_slope + self.prefix_ds(root, hi, inclusive=False)

        # Retain only lo < event.x < hi.  Coincident events are represented by
        # the two newly inserted aggregate boundary changes.
        _, above_lo = self.split(root, lo, np.iinfo(np.int64).max)
        middle, _ = self.split(above_lo, hi, -1)
        middle = self.insert(middle, lo, slope_right_lo)
        middle = self.insert(middle, hi, -slope_left_hi)
        return middle


@dataclass
class CharacteristicTree:
    parent: np.ndarray
    parent_weight: np.ndarray
    order: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_base_weight: np.ndarray


@dataclass
class TreeSolve:
    u: np.ndarray
    seconds: float
    events_created: int
    max_capacity_violation: float
    max_sign_violation: float
    stationarity_residual: float


@dataclass
class TaylorDrop:
    u: np.ndarray
    scale: float
    raw_scale: float
    accepted: bool


@radial._compile
def _raster_parent_kernel(nx, ny, diagonals, alignment_floor):
    h, w = nx.shape
    n = h * w
    parent = np.full(n, -1, dtype=np.int64)
    weight = np.zeros(n, dtype=np.float64)
    for index in range(1, n):
        y = index // w
        x = index - y * w
        best_parent = -1
        best_cost = 1e300
        # Earlier-neighbor offsets: left, up, upper-left, upper-right.
        for which in range(4):
            if which == 0:
                if x == 0:
                    continue
                candidate, dx, dy = index - 1, -1, 0
            elif which == 1:
                if y == 0:
                    continue
                candidate, dx, dy = index - w, 0, -1
            elif which == 2:
                if not diagonals or y == 0 or x == 0:
                    continue
                candidate, dx, dy = index - w - 1, -1, -1
            else:
                if not diagonals or y == 0 or x + 1 >= w:
                    continue
                candidate, dx, dy = index - w + 1, 1, -1

            cy = candidate // w
            cx = candidate - cy * w
            length = math.sqrt(dx * dx + dy * dy)
            ux, uy = dx / length, dy / length
            alignment = 0.5 * (
                abs(nx[y, x] * ux + ny[y, x] * uy)
                + abs(nx[cy, cx] * ux + ny[cy, cx] * uy)
            )
            cost = length / max(alignment, alignment_floor)
            if cost < best_cost:
                best_cost = cost
                best_parent = candidate
        parent[index] = best_parent
        weight[index] = best_cost
    return parent, weight


@radial._compile
def _event_pull(i, x, ds, left, right, sum_ds, sum_dsx):
    l = left[i]
    r = right[i]
    sum_ds[i] = ds[i] + sum_ds[l] + sum_ds[r]
    sum_dsx[i] = ds[i] * x[i] + sum_dsx[l] + sum_dsx[r]


@radial._compile
def _event_split(
    root,
    key_x,
    key_serial,
    x,
    ds,
    serial,
    left,
    right,
    sum_ds,
    sum_dsx,
):
    if root == 0:
        return 0, 0
    is_less = x[root] < key_x or (
        x[root] == key_x and serial[root] < key_serial
    )
    if is_less:
        a, b = _event_split(
            right[root],
            key_x,
            key_serial,
            x,
            ds,
            serial,
            left,
            right,
            sum_ds,
            sum_dsx,
        )
        right[root] = a
        _event_pull(root, x, ds, left, right, sum_ds, sum_dsx)
        return root, b
    a, b = _event_split(
        left[root],
        key_x,
        key_serial,
        x,
        ds,
        serial,
        left,
        right,
        sum_ds,
        sum_dsx,
    )
    left[root] = b
    _event_pull(root, x, ds, left, right, sum_ds, sum_dsx)
    return a, root


@radial._compile
def _event_unite(
    a,
    b,
    x,
    ds,
    serial,
    priority,
    left,
    right,
    sum_ds,
    sum_dsx,
):
    if a == 0:
        return b
    if b == 0:
        return a
    if priority[a] < priority[b]:
        swap = a
        a = b
        b = swap
    bl, br = _event_split(
        b,
        x[a],
        serial[a],
        x,
        ds,
        serial,
        left,
        right,
        sum_ds,
        sum_dsx,
    )
    left[a] = _event_unite(
        left[a],
        bl,
        x,
        ds,
        serial,
        priority,
        left,
        right,
        sum_ds,
        sum_dsx,
    )
    right[a] = _event_unite(
        right[a],
        br,
        x,
        ds,
        serial,
        priority,
        left,
        right,
        sum_ds,
        sum_dsx,
    )
    _event_pull(a, x, ds, left, right, sum_ds, sum_dsx)
    return a


@radial._compile
def _event_prefix_ds(root, query_x, inclusive, x, ds, left, right, sum_ds):
    total = 0.0
    i = root
    while i != 0:
        before = x[i] < query_x or (inclusive and x[i] == query_x)
        if before:
            total += sum_ds[left[i]] + ds[i]
            i = right[i]
        else:
            i = left[i]
    return total


@radial._compile
def _event_crossing(
    root,
    target,
    base_slope,
    base_intercept,
    x,
    ds,
    left,
    right,
    sum_ds,
    sum_dsx,
):
    if root == 0:
        return (target - base_intercept) / base_slope
    i = root
    slope = base_slope
    intercept = base_intercept
    while True:
        l = left[i]
        slope_at = slope + sum_ds[l]
        intercept_at = intercept - sum_dsx[l]
        value_at = slope_at * x[i] + intercept_at
        if target <= value_at:
            if l != 0:
                i = l
                continue
            return (target - intercept) / slope
        slope = slope_at + ds[i]
        intercept = intercept_at - ds[i] * x[i]
        r = right[i]
        if r != 0:
            i = r
            continue
        return (target - intercept) / slope


@radial._compile
def _event_priority(serial):
    # An integer-only mixer; overflow is the desired modulo-2^64 operation.
    z = np.uint64(serial) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@radial._compile
def _solve_tree_tv_kernel(flat_g, c, parent, parent_weight, order):
    n = flat_g.size
    counts = np.zeros(n + 1, dtype=np.int64)
    for k in range(1, n):
        counts[parent[order[k]] + 1] += 1
    for i in range(n):
        counts[i + 1] += counts[i]
    cursor = counts[:-1].copy()
    children = np.empty(n - 1, dtype=np.int64)
    for k in range(1, n):
        child = order[k]
        p = parent[child]
        children[cursor[p]] = child
        cursor[p] += 1

    maximum_events = 2 * n + 1
    event_x = np.zeros(maximum_events, dtype=np.float64)
    event_ds = np.zeros(maximum_events, dtype=np.float64)
    event_serial = np.zeros(maximum_events, dtype=np.int64)
    event_priority = np.zeros(maximum_events, dtype=np.uint64)
    event_left = np.zeros(maximum_events, dtype=np.int64)
    event_right = np.zeros(maximum_events, dtype=np.int64)
    event_sum_ds = np.zeros(maximum_events, dtype=np.float64)
    event_sum_dsx = np.zeros(maximum_events, dtype=np.float64)
    message_root = np.zeros(n, dtype=np.int64)
    lo = np.full(n, -np.inf, dtype=np.float64)
    hi = np.full(n, np.inf, dtype=np.float64)
    next_event = 1
    root_value = 0.0

    for reverse_k in range(n - 1, -1, -1):
        v = order[reverse_k]
        root = 0
        intercept = -c * flat_g[v]
        for child_position in range(counts[v], counts[v + 1]):
            child = children[child_position]
            root = _event_unite(
                root,
                message_root[child],
                event_x,
                event_ds,
                event_serial,
                event_priority,
                event_left,
                event_right,
                event_sum_ds,
                event_sum_dsx,
            )
            intercept -= parent_weight[child]

        if parent[v] >= 0:
            capacity = parent_weight[v]
            lo[v] = _event_crossing(
                root,
                -capacity,
                c,
                intercept,
                event_x,
                event_ds,
                event_left,
                event_right,
                event_sum_ds,
                event_sum_dsx,
            )
            hi[v] = _event_crossing(
                root,
                capacity,
                c,
                intercept,
                event_x,
                event_ds,
                event_left,
                event_right,
                event_sum_ds,
                event_sum_dsx,
            )
            slope_lo = c + _event_prefix_ds(
                root,
                lo[v],
                True,
                event_x,
                event_ds,
                event_left,
                event_right,
                event_sum_ds,
            )
            slope_hi = c + _event_prefix_ds(
                root,
                hi[v],
                False,
                event_x,
                event_ds,
                event_left,
                event_right,
                event_sum_ds,
            )
            _, above_lo = _event_split(
                root,
                lo[v],
                maximum_events,
                event_x,
                event_ds,
                event_serial,
                event_left,
                event_right,
                event_sum_ds,
                event_sum_dsx,
            )
            middle, _ = _event_split(
                above_lo,
                hi[v],
                -1,
                event_x,
                event_ds,
                event_serial,
                event_left,
                event_right,
                event_sum_ds,
                event_sum_dsx,
            )
            event_x[next_event] = lo[v]
            event_ds[next_event] = slope_lo
            event_serial[next_event] = next_event
            event_priority[next_event] = _event_priority(next_event)
            event_sum_ds[next_event] = slope_lo
            event_sum_dsx[next_event] = slope_lo * lo[v]
            middle = _event_unite(
                middle,
                next_event,
                event_x,
                event_ds,
                event_serial,
                event_priority,
                event_left,
                event_right,
                event_sum_ds,
                event_sum_dsx,
            )
            next_event += 1

            event_x[next_event] = hi[v]
            event_ds[next_event] = -slope_hi
            event_serial[next_event] = next_event
            event_priority[next_event] = _event_priority(next_event)
            event_sum_ds[next_event] = -slope_hi
            event_sum_dsx[next_event] = -slope_hi * hi[v]
            middle = _event_unite(
                middle,
                next_event,
                event_x,
                event_ds,
                event_serial,
                event_priority,
                event_left,
                event_right,
                event_sum_ds,
                event_sum_dsx,
            )
            next_event += 1
            message_root[v] = middle
        else:
            root_value = _event_crossing(
                root,
                0.0,
                c,
                intercept,
                event_x,
                event_ds,
                event_left,
                event_right,
                event_sum_ds,
                event_sum_dsx,
            )

    solution = np.empty(n, dtype=np.float64)
    solution[order[0]] = root_value
    for k in range(1, n):
        v = order[k]
        value = solution[parent[v]]
        solution[v] = min(max(value, lo[v]), hi[v])
    return solution, next_event - 1


def _grid_edges(h: int, w: int, diagonals: bool) -> tuple[np.ndarray, ...]:
    yy, xx = np.indices((h, w))
    index = np.arange(h * w, dtype=np.int64).reshape(h, w)
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    dxs: list[np.ndarray] = []
    dys: list[np.ndarray] = []

    steps = [(1, 0), (0, 1)]
    if diagonals:
        steps += [(1, 1), (-1, 1)]
    for dx, dy in steps:
        x0 = max(0, -dx)
        x1 = min(w, w - dx)
        y0 = max(0, -dy)
        y1 = min(h, h - dy)
        src = index[y0:y1, x0:x1].ravel()
        dst = index[y0 + dy : y1 + dy, x0 + dx : x1 + dx].ravel()
        us.append(src)
        vs.append(dst)
        dxs.append(np.full(src.size, dx, dtype=np.float64))
        dys.append(np.full(src.size, dy, dtype=np.float64))
    return (
        np.concatenate(us),
        np.concatenate(vs),
        np.concatenate(dxs),
        np.concatenate(dys),
    )


def characteristic_mst(
    nx: np.ndarray,
    ny: np.ndarray,
    reference_u: np.ndarray,
    *,
    diagonals: bool = True,
    alignment_floor: float = 0.08,
) -> CharacteristicTree:
    """Build an unoriented direction-following spanning tree.

    Edge cost is inverse directional projection.  The same geometric factor is
    globally normalized so tree variation matches isotropic TV on reference_u.
    """
    h, w = nx.shape
    n = h * w
    u, v, dx, dy = _grid_edges(h, w, diagonals)
    flat_nx = nx.ravel()
    flat_ny = ny.ravel()
    length = np.hypot(dx, dy)
    ux = dx / length
    uy = dy / length
    align_u = np.abs(flat_nx[u] * ux + flat_ny[u] * uy)
    align_v = np.abs(flat_nx[v] * ux + flat_ny[v] * uy)
    alignment = 0.5 * (align_u + align_v)
    geometric = length / np.maximum(alignment, alignment_floor)

    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    data = np.concatenate([geometric, geometric])
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    mst = minimum_spanning_tree(graph).tocoo()
    edge_u = mst.row.astype(np.int64)
    edge_v = mst.col.astype(np.int64)
    edge_base = mst.data.astype(np.float64)
    if edge_u.size != n - 1:
        raise RuntimeError(f"spanning tree has {edge_u.size} edges for {n} nodes")

    flat_ref = reference_u.ravel()
    raw_tree_tv = float(
        np.sum(edge_base * np.abs(flat_ref[edge_u] - flat_ref[edge_v]))
    )
    iso_ref = isotropic_tv(reference_u)
    scale = iso_ref / max(raw_tree_tv, 1e-12)
    edge_base *= scale

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b, wt in zip(edge_u, edge_v, edge_base):
        adjacency[int(a)].append((int(b), float(wt)))
        adjacency[int(b)].append((int(a), float(wt)))

    root = int(np.argmin(reference_u))
    parent = np.full(n, -2, dtype=np.int64)
    parent_weight = np.zeros(n, dtype=np.float64)
    parent[root] = -1
    order = np.empty(n, dtype=np.int64)
    order[0] = root
    head = 0
    tail = 1
    while head < tail:
        node = int(order[head])
        head += 1
        for child, wt in adjacency[node]:
            if parent[child] != -2:
                continue
            parent[child] = node
            parent_weight[child] = wt
            order[tail] = child
            tail += 1
    if tail != n:
        raise RuntimeError("tree traversal did not reach every pixel")

    return CharacteristicTree(
        parent=parent,
        parent_weight=parent_weight,
        order=order,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_base_weight=edge_base,
    )


def characteristic_raster_tree(
    nx: np.ndarray,
    ny: np.ndarray,
    reference_u: np.ndarray,
    *,
    diagonals: bool = True,
    alignment_floor: float = 0.08,
) -> CharacteristicTree:
    """Linear-work causal tree: each pixel selects its best earlier neighbor."""
    h, w = nx.shape
    n = h * w
    parent, parent_weight = _raster_parent_kernel(
        nx, ny, diagonals, alignment_floor
    )
    order = np.arange(n, dtype=np.int64)
    edge_u = parent[1:].copy()
    edge_v = np.arange(1, n, dtype=np.int64)
    edge_base = parent_weight[1:].copy()

    flat_ref = reference_u.ravel()
    raw_tree_tv = float(
        np.sum(edge_base * np.abs(flat_ref[edge_u] - flat_ref[edge_v]))
    )
    scale = isotropic_tv(reference_u) / max(raw_tree_tv, 1e-12)
    edge_base *= scale
    parent_weight[1:] *= scale
    return CharacteristicTree(
        parent=parent,
        parent_weight=parent_weight,
        order=order,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_base_weight=edge_base,
    )


def solve_tree_tv_direct(
    g: np.ndarray,
    c: float,
    tree: CharacteristicTree,
) -> TreeSolve:
    """Exact finite dynamic program for quadratic TV on a fixed tree."""
    started = time.perf_counter()
    flat_g = np.asarray(g, dtype=np.float64).ravel()
    n = flat_g.size
    children: list[list[int]] = [[] for _ in range(n)]
    for v in tree.order[1:]:
        children[int(tree.parent[v])].append(int(v))

    messages = np.zeros(n, dtype=np.int64)
    lo = np.full(n, -np.inf, dtype=np.float64)
    hi = np.full(n, np.inf, dtype=np.float64)
    events = _EventTreap()
    root_value = 0.0

    for vv in tree.order[::-1]:
        v = int(vv)
        event_root = 0
        intercept = -c * flat_g[v]
        for child in children[v]:
            event_root = events.unite(event_root, int(messages[child]))
            intercept -= tree.parent_weight[child]

        if tree.parent[v] >= 0:
            capacity = float(tree.parent_weight[v])
            lo[v] = events.find_crossing(event_root, -capacity, c, intercept)
            hi[v] = events.find_crossing(event_root, capacity, c, intercept)
            messages[v] = events.clip_message(event_root, lo[v], hi[v], c)
        else:
            root_value = events.find_crossing(event_root, 0.0, c, intercept)

    x = np.empty(n, dtype=np.float64)
    root = int(tree.order[0])
    x[root] = root_value
    for vv in tree.order[1:]:
        v = int(vv)
        x[v] = min(max(x[int(tree.parent[v])], lo[v]), hi[v])

    # Independent KKT certificate via reverse subtree residual accumulation.
    subtree = c * (x - flat_g)
    for vv in tree.order[:0:-1]:
        v = int(vv)
        subtree[int(tree.parent[v])] += subtree[v]
    dual_edge = -subtree
    capacity_violation = 0.0
    sign_violation = 0.0
    for vv in tree.order[1:]:
        v = int(vv)
        wt = float(tree.parent_weight[v])
        q = float(dual_edge[v])
        capacity_violation = max(capacity_violation, max(abs(q) - wt, 0.0))
        difference = x[v] - x[int(tree.parent[v])]
        if abs(difference) > 1e-8:
            sign_violation = max(
                sign_violation, abs(q - wt * math.copysign(1.0, difference))
            )

    return TreeSolve(
        u=x.reshape(g.shape),
        seconds=time.perf_counter() - started,
        events_created=len(events.x) - 1,
        max_capacity_violation=capacity_violation,
        max_sign_violation=sign_violation,
        stationarity_residual=abs(float(subtree[root])),
    )


def solve_tree_tv_compiled(
    g: np.ndarray,
    c: float,
    tree: CharacteristicTree,
) -> TreeSolve:
    """Compiled form of the same finite message calculation."""
    started = time.perf_counter()
    flat_g = np.asarray(g, dtype=np.float64).ravel()
    x, events_created = _solve_tree_tv_kernel(
        flat_g,
        float(c),
        tree.parent,
        tree.parent_weight,
        tree.order,
    )
    elapsed = time.perf_counter() - started

    subtree = c * (x - flat_g)
    for vv in tree.order[:0:-1]:
        v = int(vv)
        subtree[int(tree.parent[v])] += subtree[v]
    dual_edge = -subtree
    capacity_violation = 0.0
    sign_violation = 0.0
    for vv in tree.order[1:]:
        v = int(vv)
        wt = float(tree.parent_weight[v])
        q = float(dual_edge[v])
        capacity_violation = max(capacity_violation, max(abs(q) - wt, 0.0))
        difference = x[v] - x[int(tree.parent[v])]
        if abs(difference) > 1e-8:
            sign_violation = max(
                sign_violation, abs(q - wt * math.copysign(1.0, difference))
            )
    root = int(tree.order[0])
    return TreeSolve(
        u=x.reshape(g.shape),
        seconds=elapsed,
        events_created=int(events_created),
        max_capacity_violation=capacity_violation,
        max_sign_violation=sign_violation,
        stationarity_residual=abs(float(subtree[root])),
    )


def taylor_capacity_drop(
    result_u: np.ndarray,
    g: np.ndarray,
    c: float,
    tree: CharacteristicTree,
    *,
    minimum_scale: float = 0.25,
    maximum_scale: float = 3.0,
) -> TaylorDrop:
    """One analytical capacity correction on the current tree active set.

    Fused tree edges define plateaus.  With active signs fixed, a plateau
    value is affine in a common edge-capacity multiplier s:

        du_C/ds at s=1 = u_C - mean(g_C).

    A one-sided quadratic Taylor model of the *isotropic* objective then gives
    a scalar drop.  Zero spatial gradients contribute their exact |delta|
    kink rather than being hidden by an epsilon.
    """
    x = result_u.ravel()
    flat_g = g.ravel()
    n = x.size
    representative = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(a: int) -> int:
        while representative[a] != a:
            representative[a] = representative[representative[a]]
            a = int(representative[a])
        return a

    def unite(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        representative[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    plateau_tolerance = 1e-8 * max(float(np.ptp(x)), 1.0)
    for child in tree.order[1:]:
        v = int(child)
        p = int(tree.parent[v])
        if abs(x[v] - x[p]) <= plateau_tolerance:
            unite(v, p)

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    counts = np.bincount(roots, minlength=n).astype(np.float64)
    sums = np.bincount(roots, weights=flat_g, minlength=n)
    means = sums[roots] / counts[roots]
    velocity = (x - means).reshape(g.shape)

    gx, gy = radial._grad(result_u)
    vx, vy = radial._grad(velocity)
    magnitude = np.hypot(gx, gy)
    moving = magnitude > plateau_tolerance
    dot = gx * vx + gy * vy
    smooth_first = c * float(np.vdot(result_u - g, velocity).real)
    smooth_first += float(np.sum(dot[moving] / magnitude[moving]))
    kink = float(np.sum(np.hypot(vx[~moving], vy[~moving])))
    cross = gx * vy - gy * vx
    second = c * float(np.vdot(velocity, velocity).real)
    second += float(np.sum(cross[moving] ** 2 / magnitude[moving] ** 3))

    right_derivative = smooth_first + kink
    left_derivative = smooth_first - kink
    if right_derivative < 0.0:
        delta = -right_derivative / max(second, 1e-30)
    elif left_derivative > 0.0:
        delta = -left_derivative / max(second, 1e-30)
    else:
        delta = 0.0

    raw_scale = 1.0 + delta
    scale = float(np.clip(raw_scale, minimum_scale, maximum_scale))
    candidate = result_u + (scale - 1.0) * velocity
    baseline_objective = radial.isotropic_objective(result_u, g, c)
    candidate_objective = radial.isotropic_objective(candidate, g, c)
    accepted = candidate_objective < baseline_objective
    return TaylorDrop(
        u=candidate if accepted else result_u,
        scale=scale if accepted else 1.0,
        raw_scale=raw_scale,
        accepted=accepted,
    )


def _equivalent_pass(objective: float, bregman_objectives: np.ndarray) -> str:
    indices = np.flatnonzero(bregman_objectives <= objective)
    return str(int(indices[0] + 1)) if indices.size else f">{bregman_objectives.size}"


def _load_image(name: str, size: int) -> np.ndarray:
    if name == "synthetic":
        return radial.synthetic(size)
    if name not in {"camera", "cameraman"}:
        raise ValueError(f"unknown image {name!r}")
    image = np.asarray(radial.gallery.load("camera"), dtype=np.float64)
    if image.shape != (size, size):
        factors = (size / image.shape[0], size / image.shape[1])
        image = zoom(image, factors, order=1, prefilter=False)
    return image


def run_case(
    image_name: str,
    size: int,
    c: float,
    checkpoints: list[int],
    reference_passes: int,
    diagonals: bool,
    raster_tree: bool,
) -> None:
    image = _load_image(image_name, size)
    positive_checkpoints = [p for p in checkpoints if p > 0]
    states, bregman_objectives = radial.split_bregman_states(
        image,
        c,
        eta=0.10,
        iterations=reference_passes,
        checkpoints=positive_checkpoints,
    )
    if 0 in checkpoints:
        zeros = np.zeros_like(image)
        states[0] = (image.copy(), zeros, zeros)
    reference, _, _, _, reference_gap = radial.fgp_reference(
        image, c, max_iterations=12000, tolerance=2e-8
    )
    reference_objective = radial.isotropic_objective(reference, image, c)

    print(
        f"\n{image_name} {size}x{size}, c={c:g}, "
        f"{'8' if diagonals else '4'}-neighbor "
        f"{'causal raster tree' if raster_tree else 'characteristic MST'}"
    )
    print(f"reference relative primal-dual gap: {reference_gap:.2e}")
    print(
        "pass  build_ms  solve_ms  events/px  tree_KKT    Taylor_s  "
        "iso_obj_excess  checkpoint_excess  equiv_pass  accepted"
    )
    for pass_count in checkpoints:
        state_u, state_px, state_py = states[pass_count]
        nx, ny, _ = radial.directions_from_state(
            state_u, state_px, state_py
        )
        build_started = time.perf_counter()
        tree_builder = (
            characteristic_raster_tree if raster_tree else characteristic_mst
        )
        tree = tree_builder(
            nx,
            ny,
            state_u,
            diagonals=diagonals,
        )
        build_seconds = time.perf_counter() - build_started
        result = solve_tree_tv_compiled(image, c, tree)
        taylor = taylor_capacity_drop(result.u, image, c, tree)
        candidate_obj = radial.isotropic_objective(taylor.u, image, c)
        checkpoint_obj = radial.isotropic_objective(state_u, image, c)
        kkt = max(
            result.max_capacity_violation,
            result.max_sign_violation,
            result.stationarity_residual,
        )
        print(
            f"{pass_count:4d}  {1e3 * build_seconds:8.2f}  "
            f"{1e3 * result.seconds:8.2f}  "
            f"{result.events_created / image.size:9.3f}  {kkt:9.2e}  "
            f"{taylor.scale:8.3f}  "
            f"{candidate_obj - reference_objective:14.6g}  "
            f"{checkpoint_obj - reference_objective:17.6g}  "
            f"{_equivalent_pass(candidate_obj, bregman_objectives):>10s}  "
            f"{'yes' if candidate_obj < checkpoint_obj else 'no'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8, 16],
    )
    parser.add_argument("--reference-passes", type=int, default=128)
    parser.add_argument(
        "--images",
        nargs="+",
        default=["cameraman", "synthetic"],
    )
    parser.add_argument(
        "--four-neighbor",
        action="store_true",
        help="Use only axis grid edges when building the tree.",
    )
    parser.add_argument(
        "--raster-tree",
        action="store_true",
        help="Use a linear-work causal predecessor tree instead of an MST.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Exclude one-time Numba compilation/cache loading from per-image timings.
    warm_g = radial.synthetic(8)
    warm_zero = np.zeros_like(warm_g)
    warm_nx, warm_ny, _ = radial.directions_from_state(
        warm_g, warm_zero, warm_zero
    )
    warm_tree = characteristic_raster_tree(warm_nx, warm_ny, warm_g)
    solve_tree_tv_compiled(warm_g, args.c, warm_tree)
    for image_name in args.images:
        run_case(
            image_name,
            args.size,
            args.c,
            args.checkpoints,
            args.reference_passes,
            not args.four_neighbor,
            args.raster_tree,
        )


if __name__ == "__main__":
    main()
