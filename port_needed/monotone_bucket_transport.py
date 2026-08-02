#!/usr/bin/env python3
"""Exact monotone-bucket reference walks for transport support.

Baseline: comparison-based binary heap with lazy deletion, `O(E log V)`.

Formal target: **a monotone priority queue**.  Dijkstra pops in nondecreasing
key order, so a comparison heap is more structure than the problem needs.
With bucket width `delta` chosen no larger than the smallest edge cost, an
element in the current bucket cannot be improved by another element of the
same bucket -- any relaxation from a key in `[k*delta, (k+1)*delta)` adds at
least `delta` and therefore lands in a strictly later bucket.  So buckets may
be emptied in arbitrary internal order and the result is *exact*, not
approximate.  This is Dial's argument, and it turns `O(E log V)` into
`O(E + range/delta)`.

Two other costs go away with it, both consequences rather than tricks:

* the queue never needs to hold a key more than `c_max` beyond the current
  key, so the buckets are a small circular array, not a growing structure;
* the direction taken into each node is known at relaxation time, so storing
  it removes the eight-way search the gradient accumulator does later.

Exactness is checked against the shipped kernel on distances, not on labels:
where two sites tie exactly, either answer is correct and the two queues may
disagree.  That disagreement is reported rather than hidden.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(fn):  # pragma: no cover
    return fn


_compile = njit(cache=True) if njit is not None else _identity


@_compile
def _dijkstra_bucket(seed_p, reach, base_costs, s_field, h, w,
                     delta, span, shift):
    """Monotone-bucket two-label walk.

    `delta` must not exceed the smallest edge cost and `span` must cover the
    largest, or the queue is no longer a valid monotone structure.  Both are
    computed from the same arrays by the caller, so the precondition is a
    property of the inputs rather than a tuning choice.
    """
    npix = h * w
    inf = 1e300
    d1 = np.empty(npix)
    d2 = np.empty(npix)
    own = np.empty(npix, dtype=np.int32)
    run = np.empty(npix, dtype=np.int32)
    pr1 = np.empty(npix, dtype=np.int32)
    pr2 = np.empty(npix, dtype=np.int32)
    pl1 = np.zeros(npix, dtype=np.int8)
    pl2 = np.zeros(npix, dtype=np.int8)
    # Direction taken into each node, recorded where it is already known.
    pd1 = np.empty(npix, dtype=np.int8)
    pd2 = np.empty(npix, dtype=np.int8)
    for pixel in range(npix):
        d1[pixel] = inf
        d2[pixel] = inf
        own[pixel] = -1
        run[pixel] = -1
        pr1[pixel] = -1
        pr2[pixel] = -1
        pd1[pixel] = -1
        pd2[pixel] = -1

    buckets = span + 2
    head = np.empty(buckets, dtype=np.int32)
    for bucket in range(buckets):
        head[bucket] = -1
    cap = 4 * npix + 256
    key = np.empty(cap, dtype=np.float64)
    pix = np.empty(cap, dtype=np.int32)
    sit = np.empty(cap, dtype=np.int32)
    nxt = np.empty(cap, dtype=np.int32)
    used = 0
    alive = 0

    for site in range(len(seed_p)):
        p = seed_p[site]
        distance = -reach[site]
        if distance < d1[p]:
            d2[p] = d1[p]
            run[p] = own[p]
            pr2[p] = pr1[p]
            pl2[p] = pl1[p]
            pd2[p] = pd1[p]
            d1[p] = distance
            own[p] = site
            pr1[p] = -1
            pl1[p] = 0
            pd1[p] = -1
        elif site != own[p] and distance < d2[p]:
            d2[p] = distance
            run[p] = site
            pr2[p] = -1
            pl2[p] = 0
            pd2[p] = -1
        slot = int((distance + shift) / delta)
        key[used] = distance
        pix[used] = p
        sit[used] = site
        nxt[used] = head[slot % buckets]
        head[slot % buckets] = used
        used += 1
        alive += 1

    dys = (-1, 1, 0, 0, -1, -1, 1, 1)
    dxs = (0, 0, -1, 1, -1, 1, -1, 1)
    tolerance = 1e-12

    current = 0
    guard = 0
    limit = buckets * (npix + 16)
    while alive > 0 and guard < limit:
        index = current % buckets
        entry = head[index]
        if entry < 0:
            current += 1
            guard += 1
            continue
        head[index] = -1
        while entry >= 0:
            distance = key[entry]
            p = pix[entry]
            site = sit[entry]
            entry = nxt[entry]
            alive -= 1

            if own[p] == site and distance <= d1[p] + tolerance:
                label = 0
            elif run[p] == site and distance <= d2[p] + tolerance:
                label = 1
            else:
                continue

            y = p // w
            x = p - y * w
            sp = s_field[p]
            for direction in range(8):
                ny = y + dys[direction]
                nx = x + dxs[direction]
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                q = ny * w + nx
                step = base_costs[direction, y, x] * 0.5 * (sp + s_field[q])
                candidate = distance + step
                touched = False
                if own[q] == site:
                    if candidate + tolerance < d1[q]:
                        d1[q] = candidate
                        pr1[q] = p
                        pl1[q] = label
                        pd1[q] = direction
                        touched = True
                elif run[q] == site:
                    if candidate + tolerance < d2[q]:
                        d2[q] = candidate
                        pr2[q] = p
                        pl2[q] = label
                        pd2[q] = direction
                        if d2[q] < d1[q]:
                            swap = d1[q]
                            d1[q] = d2[q]
                            d2[q] = swap
                            iswap = own[q]
                            own[q] = run[q]
                            run[q] = iswap
                            iswap = pr1[q]
                            pr1[q] = pr2[q]
                            pr2[q] = iswap
                            bswap = pl1[q]
                            pl1[q] = pl2[q]
                            pl2[q] = bswap
                            bswap = pd1[q]
                            pd1[q] = pd2[q]
                            pd2[q] = bswap
                        touched = True
                elif candidate + tolerance < d1[q]:
                    d2[q] = d1[q]
                    run[q] = own[q]
                    pr2[q] = pr1[q]
                    pl2[q] = pl1[q]
                    pd2[q] = pd1[q]
                    d1[q] = candidate
                    own[q] = site
                    pr1[q] = p
                    pl1[q] = label
                    pd1[q] = direction
                    touched = True
                elif candidate + tolerance < d2[q]:
                    d2[q] = candidate
                    run[q] = site
                    pr2[q] = p
                    pl2[q] = label
                    pd2[q] = direction
                    touched = True
                if not touched:
                    continue
                if used >= cap:
                    cap *= 2
                    nk = np.empty(cap, dtype=np.float64)
                    np_ = np.empty(cap, dtype=np.int32)
                    ns = np.empty(cap, dtype=np.int32)
                    nn = np.empty(cap, dtype=np.int32)
                    nk[:used] = key[:used]
                    np_[:used] = pix[:used]
                    ns[:used] = sit[:used]
                    nn[:used] = nxt[:used]
                    key, pix, sit, nxt = nk, np_, ns, nn
                slot = int((candidate + shift) / delta)
                key[used] = candidate
                pix[used] = q
                sit[used] = site
                nxt[used] = head[slot % buckets]
                head[slot % buckets] = used
                used += 1
                alive += 1
        current += 1
        guard += 1
    return own, run, d1, d2, pr1, pr2, pl1, pl2, pd1, pd2, used


@_compile
def _dijkstra_first_bucket(seed_p, reach, base_costs, s_field, h, w,
                           delta, span, shift):
    """Monotone multi-source walk retaining only the achieving first label.

    For source-independent nonnegative edge costs, a runner-up can never
    become first after passing through a vertex: the first source at that
    vertex can follow the identical suffix and remain no farther away.
    Therefore first-owner geometry and its predecessor forest do not require
    the two-label state carried by `_dijkstra_bucket`.
    """
    npix = h * w
    inf = 1e300
    distance = np.empty(npix)
    owner = np.empty(npix, dtype=np.int32)
    parent = np.empty(npix, dtype=np.int32)
    for pixel in range(npix):
        distance[pixel] = inf
        owner[pixel] = -1
        parent[pixel] = -1

    buckets = span + 2
    head = np.empty(buckets, dtype=np.int32)
    for bucket in range(buckets):
        head[bucket] = -1
    cap = 2 * npix + 256
    key = np.empty(cap, dtype=np.float64)
    pix = np.empty(cap, dtype=np.int32)
    site_at_entry = np.empty(cap, dtype=np.int32)
    nxt = np.empty(cap, dtype=np.int32)
    used = 0
    alive = 0

    for site in range(len(seed_p)):
        p = seed_p[site]
        value = -reach[site]
        if value < distance[p]:
            distance[p] = value
            owner[p] = site
            parent[p] = -1
            slot = int((value + shift) / delta)
            key[used] = value
            pix[used] = p
            site_at_entry[used] = site
            nxt[used] = head[slot % buckets]
            head[slot % buckets] = used
            used += 1
            alive += 1

    dys = (-1, 1, 0, 0, -1, -1, 1, 1)
    dxs = (0, 0, -1, 1, -1, 1, -1, 1)
    tolerance = 1e-12
    current = 0
    guard = 0
    limit = buckets * (npix + 16)
    while alive > 0 and guard < limit:
        index = current % buckets
        entry = head[index]
        if entry < 0:
            current += 1
            guard += 1
            continue
        head[index] = -1
        while entry >= 0:
            value = key[entry]
            p = pix[entry]
            site = site_at_entry[entry]
            entry = nxt[entry]
            alive -= 1
            if owner[p] != site or value > distance[p] + tolerance:
                continue

            y = p // w
            x = p - y * w
            sp = s_field[p]
            for direction in range(8):
                ny = y + dys[direction]
                nx = x + dxs[direction]
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                q = ny * w + nx
                step = (
                    base_costs[direction, y, x]
                    * 0.5 * (sp + s_field[q])
                )
                candidate = value + step
                if candidate + tolerance >= distance[q]:
                    continue
                distance[q] = candidate
                owner[q] = site
                parent[q] = p
                if used >= cap:
                    cap *= 2
                    next_key = np.empty(cap, dtype=np.float64)
                    next_pix = np.empty(cap, dtype=np.int32)
                    next_site = np.empty(cap, dtype=np.int32)
                    next_link = np.empty(cap, dtype=np.int32)
                    next_key[:used] = key[:used]
                    next_pix[:used] = pix[:used]
                    next_site[:used] = site_at_entry[:used]
                    next_link[:used] = nxt[:used]
                    key = next_key
                    pix = next_pix
                    site_at_entry = next_site
                    nxt = next_link
                slot = int((candidate + shift) / delta)
                key[used] = candidate
                pix[used] = q
                site_at_entry[used] = site
                nxt[used] = head[slot % buckets]
                head[slot % buckets] = used
                used += 1
                alive += 1
        current += 1
        guard += 1
    return owner, distance, parent, used


@_compile
def _dijkstra_bucket_packed(seed_p, reach, cost, neighbour, s_field, npix,
                            delta, span, shift):
    """The same queue over a node-major adjacency.

    The shipped kernel reads `base_costs[d, y, x]`, so a node's eight step
    costs live eight full images apart and each visit touches eight cache
    lines.  Interleaving them to `cost[8 * p + d]` makes a node's whole
    neighbourhood one contiguous read.  Storing the neighbour index beside
    it removes the per-node integer division and the eight bound tests: an
    out-of-grid step is encoded as an infinite cost pointing back at itself,
    so it can never relax and never needs to be asked about.
    """
    inf = 1e300
    d1 = np.full(npix, inf)
    d2 = np.full(npix, inf)
    own = np.full(npix, -1, dtype=np.int32)
    run = np.full(npix, -1, dtype=np.int32)
    pr1 = np.full(npix, -1, dtype=np.int32)
    pr2 = np.full(npix, -1, dtype=np.int32)
    pl1 = np.zeros(npix, dtype=np.int8)
    pl2 = np.zeros(npix, dtype=np.int8)
    pd1 = np.full(npix, -1, dtype=np.int8)
    pd2 = np.full(npix, -1, dtype=np.int8)

    buckets = span + 2
    head = np.full(buckets, -1, dtype=np.int32)
    cap = 4 * npix + 256
    key = np.empty(cap, dtype=np.float64)
    pix = np.empty(cap, dtype=np.int32)
    sit = np.empty(cap, dtype=np.int32)
    nxt = np.empty(cap, dtype=np.int32)
    used = 0
    alive = 0

    for site in range(len(seed_p)):
        p = seed_p[site]
        distance = -reach[site]
        if distance < d1[p]:
            d2[p] = d1[p]
            run[p] = own[p]
            pr2[p] = pr1[p]
            pl2[p] = pl1[p]
            pd2[p] = pd1[p]
            d1[p] = distance
            own[p] = site
            pr1[p] = -1
            pl1[p] = 0
            pd1[p] = -1
        elif site != own[p] and distance < d2[p]:
            d2[p] = distance
            run[p] = site
            pr2[p] = -1
            pl2[p] = 0
            pd2[p] = -1
        slot = int((distance + shift) / delta) % buckets
        key[used] = distance
        pix[used] = p
        sit[used] = site
        nxt[used] = head[slot]
        head[slot] = used
        used += 1
        alive += 1

    tolerance = 1e-12
    current = 0
    guard = 0
    limit = buckets * (npix + 16)
    while alive > 0 and guard < limit:
        index = current % buckets
        entry = head[index]
        if entry < 0:
            current += 1
            guard += 1
            continue
        head[index] = -1
        while entry >= 0:
            distance = key[entry]
            p = pix[entry]
            site = sit[entry]
            entry = nxt[entry]
            alive -= 1
            if own[p] == site and distance <= d1[p] + tolerance:
                label = 0
            elif run[p] == site and distance <= d2[p] + tolerance:
                label = 1
            else:
                continue
            row = 8 * p
            sp = s_field[p]
            for direction in range(8):
                step = cost[row + direction]
                if step > 1e299:
                    continue
                q = neighbour[row + direction]
                candidate = distance + step * 0.5 * (sp + s_field[q])
                touched = False
                if own[q] == site:
                    if candidate + tolerance < d1[q]:
                        d1[q] = candidate
                        pr1[q] = p
                        pl1[q] = label
                        pd1[q] = direction
                        touched = True
                elif run[q] == site:
                    if candidate + tolerance < d2[q]:
                        d2[q] = candidate
                        pr2[q] = p
                        pl2[q] = label
                        pd2[q] = direction
                        if d2[q] < d1[q]:
                            swap = d1[q]
                            d1[q] = d2[q]
                            d2[q] = swap
                            iswap = own[q]
                            own[q] = run[q]
                            run[q] = iswap
                            iswap = pr1[q]
                            pr1[q] = pr2[q]
                            pr2[q] = iswap
                            bswap = pl1[q]
                            pl1[q] = pl2[q]
                            pl2[q] = bswap
                            bswap = pd1[q]
                            pd1[q] = pd2[q]
                            pd2[q] = bswap
                        touched = True
                elif candidate + tolerance < d1[q]:
                    d2[q] = d1[q]
                    run[q] = own[q]
                    pr2[q] = pr1[q]
                    pl2[q] = pl1[q]
                    pd2[q] = pd1[q]
                    d1[q] = candidate
                    own[q] = site
                    pr1[q] = p
                    pl1[q] = label
                    pd1[q] = direction
                    touched = True
                elif candidate + tolerance < d2[q]:
                    d2[q] = candidate
                    run[q] = site
                    pr2[q] = p
                    pl2[q] = label
                    pd2[q] = direction
                    touched = True
                if not touched:
                    continue
                if used >= cap:
                    cap *= 2
                    nk = np.empty(cap, dtype=np.float64)
                    np_ = np.empty(cap, dtype=np.int32)
                    ns = np.empty(cap, dtype=np.int32)
                    nn = np.empty(cap, dtype=np.int32)
                    nk[:used] = key[:used]
                    np_[:used] = pix[:used]
                    ns[:used] = sit[:used]
                    nn[:used] = nxt[:used]
                    key, pix, sit, nxt = nk, np_, ns, nn
                slot = int((candidate + shift) / delta) % buckets
                key[used] = candidate
                pix[used] = q
                sit[used] = site
                nxt[used] = head[slot]
                head[slot] = used
                used += 1
                alive += 1
        current += 1
        guard += 1
    return own, run, d1, d2, pr1, pr2, pl1, pl2, pd1, pd2, used


def pack_adjacency(base_costs, h, w):
    """Node-major costs and neighbour indices, built once per geometry."""
    npix = h * w
    dys = (-1, 1, 0, 0, -1, -1, 1, 1)
    dxs = (0, 0, -1, 1, -1, 1, -1, 1)
    cost = np.full((npix, 8), np.inf, dtype=np.float64)
    neighbour = np.tile(
        np.arange(npix, dtype=np.int32)[:, None], (1, 8))
    yy, xx = np.mgrid[0:h, 0:w]
    for direction, (dy, dx) in enumerate(zip(dys, dxs)):
        ny, nx = yy + dy, xx + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        plane = base_costs[direction]
        cost[:, direction] = np.where(
            inside & np.isfinite(plane), plane, np.inf).ravel()
        neighbour[:, direction] = np.where(
            inside, ny * w + nx, yy * w + xx).ravel()
    return cost.ravel(), neighbour.ravel()


def queue_geometry(base_costs, s_field, reach):
    """Bucket width and span forced by the inputs, not chosen.

    `delta` is a strict lower bound on any edge cost and `span` an upper
    bound on how far ahead of the current key an insertion can land.  Both
    follow from the modulated cost `base * (s(p) + s(q)) / 2`.
    """
    smallest = float(np.min(base_costs))
    widest_base = float(np.max(base_costs))
    if not np.isfinite(smallest) or not np.isfinite(widest_base):
        finite = base_costs[np.isfinite(base_costs)]
        smallest = float(np.min(finite))
        widest_base = float(np.max(finite))
    low_scale = float(np.min(s_field))
    high_scale = float(np.max(s_field))
    delta = smallest * low_scale
    widest = widest_base * high_scale
    span = int(np.ceil(widest / delta)) + 2
    shift = float(np.max(reach)) + delta
    return delta, span, shift


def run_bucket(args):
    seed_p, reach, base_costs, s_field, h, w = args
    delta, span, shift = queue_geometry(base_costs, s_field, reach)
    return _dijkstra_bucket(
        seed_p, reach, base_costs, s_field, h, w, delta, span, shift)


def run_packed(args, packed=None):
    seed_p, reach, base_costs, s_field, h, w = args
    delta, span, shift = queue_geometry(base_costs, s_field, reach)
    if packed is None:
        packed = pack_adjacency(base_costs, h, w)
    cost, neighbour = packed
    return _dijkstra_bucket_packed(
        seed_p, reach, cost, neighbour, s_field, h * w, delta, span, shift)
