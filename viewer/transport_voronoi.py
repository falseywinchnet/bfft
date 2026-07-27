"""BFFT-guided anisotropic Voronoi image reconstruction.

This is an executable research model, not a port of SAD.  SAD learns site
placement, anisotropy, temperature, and color jointly from reconstruction
error.  Here the Meyer split supplies that geometry before fitting:

* the cartoon gradient and one-step outer-map defect define coarse bounds;
* the texture structure tensor defines the tangent direction and coherence;
* cartoon edges plus texture energy define site density;
* only the remaining OKLab error drives later subdivision.

Each site owns an anisotropic Voronoi region and carries an affine OKLab
color plane in its local tangent/normal frame.  The two nearest sites may be
softly blended, giving a SAD-like partition of unity without asking an
optimizer to discover the image geometry from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import os
import time

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import lsmr
from skimage.segmentation import watershed
from skimage.transform import resize

try:
    from numba import njit, types
    from numba.typed import List
except ImportError:  # pragma: no cover - the Python solver remains valid
    njit = None

import bfft
from bfft.effects import _from_working, lab_to_srgb, srgb_to_lab
from recursive_decomposition import _native_from_working

CARTOON = np.uint8(0)
TEXTURE = np.uint8(1)


if njit is not None:
    _HEAP_ITEM_TYPE = types.Tuple((
        types.float64, types.int64, types.int64))

    @njit(cache=True)
    def _dijkstra_two_best(seed_x, seed_y, reach, costs, h, w):
        """Compiled equivalent of the two-label geodesic Dijkstra walk."""
        npix = h * w
        best = np.full(npix, np.inf)
        second = np.full(npix, np.inf)
        owner = np.full(npix, -1, dtype=np.int32)
        runner = np.full(npix, -1, dtype=np.int32)
        heap = List.empty_list(_HEAP_ITEM_TYPE)
        for site in range(len(seed_x)):
            x, y = seed_x[site], seed_y[site]
            p = y * w + x
            distance = -reach[site]
            if distance < best[p]:
                second[p], runner[p] = best[p], owner[p]
                best[p], owner[p] = distance, site
            elif site != owner[p] and distance < second[p]:
                second[p], runner[p] = distance, site
            heapq.heappush(
                heap, (np.float64(distance), np.int64(p), np.int64(site)))
        dys = (-1, 1, 0, 0, -1, -1, 1, 1)
        dxs = (0, 0, -1, 1, -1, 1, -1, 1)
        tolerance = 1e-12
        while heap:
            distance, p, site = heapq.heappop(heap)
            valid = (
                (owner[p] == site and distance <= best[p] + tolerance) or
                (runner[p] == site and distance <= second[p] + tolerance))
            if not valid:
                continue
            y, x = divmod(p, w)
            for direction in range(8):
                ny, nx = y + dys[direction], x + dxs[direction]
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                q = ny * w + nx
                candidate = (
                    np.float64(distance) +
                    np.float64(costs[direction, y, x]))
                if owner[q] == site:
                    if candidate + tolerance < best[q]:
                        best[q] = candidate
                        heapq.heappush(heap, (candidate, q, site))
                elif runner[q] == site:
                    if candidate + tolerance < second[q]:
                        second[q] = candidate
                        if second[q] < best[q]:
                            best[q], second[q] = second[q], best[q]
                            owner[q], runner[q] = runner[q], owner[q]
                        heapq.heappush(heap, (candidate, q, site))
                elif candidate + tolerance < best[q]:
                    second[q], runner[q] = best[q], owner[q]
                    best[q], owner[q] = candidate, site
                    heapq.heappush(heap, (candidate, q, site))
                elif candidate + tolerance < second[q]:
                    second[q], runner[q] = candidate, site
                    heapq.heappush(heap, (candidate, q, site))
        return owner, runner, best, second

    @njit(cache=True)
    def _dijkstra_two_best_packed(seed_x, seed_y, reach, costs, h, w):
        """Exact two-label walk with a packed array heap.

        This is the same relaxation as ``_dijkstra_two_best``.  Keeping the
        heap in three contiguous arrays removes typed-list/Python-heap
        bookkeeping and gives the compiler simple monomorphic loops.
        """
        npix = h * w
        inf = 1e300
        best = np.full(npix, inf)
        second = np.full(npix, inf)
        owner = np.full(npix, -1, dtype=np.int32)
        runner = np.full(npix, -1, dtype=np.int32)

        cap = npix + 4 * len(seed_x) + 256
        heap_distance = np.empty(cap, dtype=np.float64)
        heap_pixel = np.empty(cap, dtype=np.int32)
        heap_site = np.empty(cap, dtype=np.int32)
        size = 0

        for site in range(len(seed_x)):
            p = seed_y[site] * w + seed_x[site]
            distance = -reach[site]
            if distance < best[p]:
                second[p], runner[p] = best[p], owner[p]
                best[p], owner[p] = distance, site
            elif site != owner[p] and distance < second[p]:
                second[p], runner[p] = distance, site
            if size >= cap:
                cap *= 2
                new_distance = np.empty(cap, dtype=np.float64)
                new_pixel = np.empty(cap, dtype=np.int32)
                new_site = np.empty(cap, dtype=np.int32)
                new_distance[:size] = heap_distance[:size]
                new_pixel[:size] = heap_pixel[:size]
                new_site[:size] = heap_site[:size]
                heap_distance, heap_pixel, heap_site = (
                    new_distance, new_pixel, new_site)
            heap_distance[size] = distance
            heap_pixel[size] = p
            heap_site[size] = site
            child = size
            size += 1
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
            p = heap_pixel[0]
            site = heap_site[0]
            size -= 1
            heap_distance[0] = heap_distance[size]
            heap_pixel[0] = heap_pixel[size]
            heap_site[0] = heap_site[size]
            node = 0
            while True:
                left = 2 * node + 1
                right = left + 1
                smallest = node
                if (left < size and
                        heap_distance[left] < heap_distance[smallest]):
                    smallest = left
                if (right < size and
                        heap_distance[right] < heap_distance[smallest]):
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
                (owner[p] == site and distance <= best[p] + tolerance) or
                (runner[p] == site and distance <= second[p] + tolerance))
            if not valid:
                continue
            y = p // w
            x = p - y * w
            for direction in range(8):
                ny = y + dys[direction]
                nx = x + dxs[direction]
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                q = ny * w + nx
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
                if size >= cap:
                    cap *= 2
                    new_distance = np.empty(cap, dtype=np.float64)
                    new_pixel = np.empty(cap, dtype=np.int32)
                    new_site = np.empty(cap, dtype=np.int32)
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
        return owner, runner, best, second

    @njit(cache=True)
    def _dijkstra_two_best_bucket(
            seed_x, seed_y, reach, costs, h, w, delta, bucket_span):
        """Exact two-label Dial walk with a reusable circular queue.

        Every edge costs at least ``delta``. A relaxation therefore lands in
        a later bucket and entries within the current bucket need no ordering.
        Queue slots are recycled as they are popped, avoiding the several-P
        append-only allocation used by the original experiment.
        """
        npix = h * w
        inf = 1e300
        best = np.full(npix, inf)
        second = np.full(npix, inf)
        owner = np.full(npix, -1, dtype=np.int32)
        runner = np.full(npix, -1, dtype=np.int32)

        minimum_reach = np.min(reach)
        maximum_reach = np.max(reach)
        reach_buckets = int(
            math.ceil((maximum_reach - minimum_reach) / delta)) + 2
        buckets = max(bucket_span + 2, reach_buckets)
        head = np.full(buckets, -1, dtype=np.int32)
        cap = npix + len(seed_x) + 256
        key = np.empty(cap, dtype=np.float64)
        pixel = np.empty(cap, dtype=np.int32)
        site_id = np.empty(cap, dtype=np.int32)
        next_entry = np.empty(cap, dtype=np.int32)
        used = 0
        free_head = -1
        alive = 0
        shift = maximum_reach + delta

        for site in range(len(seed_x)):
            p = seed_y[site] * w + seed_x[site]
            distance = -reach[site]
            if distance < best[p]:
                second[p], runner[p] = best[p], owner[p]
                best[p], owner[p] = distance, site
            elif site != owner[p] and distance < second[p]:
                second[p], runner[p] = distance, site
            slot = int((distance + shift) / delta) % buckets
            key[used] = distance
            pixel[used] = p
            site_id[used] = site
            next_entry[used] = head[slot]
            head[slot] = used
            used += 1
            alive += 1

        dys = (-1, 1, 0, 0, -1, -1, 1, 1)
        dxs = (0, 0, -1, 1, -1, 1, -1, 1)
        tolerance = 1e-12
        current = 0
        while alive > 0:
            slot = current % buckets
            entry = head[slot]
            if entry < 0:
                current += 1
                continue
            head[slot] = -1
            while entry >= 0:
                following = next_entry[entry]
                distance = key[entry]
                p = pixel[entry]
                site = site_id[entry]
                next_entry[entry] = free_head
                free_head = entry
                entry = following
                alive -= 1

                valid = (
                    (owner[p] == site and
                     distance <= best[p] + tolerance) or
                    (runner[p] == site and
                     distance <= second[p] + tolerance))
                if not valid:
                    continue
                y = p // w
                x = p - y * w
                for direction in range(8):
                    ny = y + dys[direction]
                    nx = x + dxs[direction]
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    q = ny * w + nx
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
                                owner[q], runner[q] = (
                                    runner[q], owner[q])
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

                    if free_head >= 0:
                        new_entry = free_head
                        free_head = next_entry[new_entry]
                    else:
                        if used >= cap:
                            cap *= 2
                            new_key = np.empty(cap, dtype=np.float64)
                            new_pixel = np.empty(cap, dtype=np.int32)
                            new_site = np.empty(cap, dtype=np.int32)
                            new_next = np.empty(cap, dtype=np.int32)
                            new_key[:used] = key[:used]
                            new_pixel[:used] = pixel[:used]
                            new_site[:used] = site_id[:used]
                            new_next[:used] = next_entry[:used]
                            key, pixel, site_id, next_entry = (
                                new_key, new_pixel, new_site, new_next)
                        new_entry = used
                        used += 1
                    destination = int(
                        (candidate + shift) / delta) % buckets
                    key[new_entry] = candidate
                    pixel[new_entry] = q
                    site_id[new_entry] = site
                    next_entry[new_entry] = head[destination]
                    head[destination] = new_entry
                    alive += 1
            current += 1
        return owner, runner, best, second

    @njit(cache=True)
    def _cell_pressure_argmax(owner, pressure, cells):
        """Highest-pressure pixel of every cell in one linear image pass."""
        best_value = np.full(cells, -np.inf)
        best_index = np.full(cells, -1, dtype=np.int64)
        for pixel in range(owner.size):
            cell = owner[pixel]
            if (cell >= 0 and cell < cells and
                    pressure[pixel] > best_value[cell]):
                best_value[cell] = pressure[pixel]
                best_index[cell] = pixel
        return best_index
else:
    _dijkstra_two_best = None
    _dijkstra_two_best_packed = None
    _dijkstra_two_best_bucket = None

    def _cell_pressure_argmax(owner, pressure, cells):
        best_index = np.full(cells, -1, dtype=np.int64)
        order = np.argsort(pressure)
        best_index[owner[order]] = order
        return best_index


def _normalize(a, percentile=99.0):
    scale = float(np.percentile(np.abs(a), percentile))
    return np.clip(a / max(scale, 1e-12), 0.0, 1.0)


def _fit_rgb(image, max_side):
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 2:
        if a.max() > 1.5:
            a = a / 255.0
        a = np.stack([a, a, a], axis=-1)
    else:
        a = a[..., :3]
        if a.max() > 1.5:
            a = a / 255.0
    h, w = a.shape[:2]
    if max_side is None or int(max_side) <= 0:
        return np.clip(a, 0.0, 1.0)
    scale = min(1.0, float(max_side) / max(h, w))
    oh, ow = max(16, round(h * scale)), max(16, round(w * scale))
    if (oh, ow) != (h, w):
        a = resize(a, (oh, ow), anti_aliasing=True, preserve_range=True)
    return np.clip(a, 0.0, 1.0)


@dataclass
class Config:
    max_side: int = 256
    passes: int = 24
    lam: float = 0.05
    mu: float = 40.0
    flow_sweeps: int = 64
    marked_cells: bool = True
    territory_count: int = 24
    initial_cells: int = 180
    max_cells: int = 1200
    split_batch: int = 24
    edge_density: float = 4.0
    texture_density: float = 3.0
    seed_bias: float = 0.0
    anisotropy: float = 5.0
    density_anisotropy: float = 2.0
    entropy_threshold: float = 0.38
    entropy_gain: float = 9.0
    consistency_weight: float = 0.65
    texture_reach_scale: float = 0.65
    cross_inhibition: float = 0.0
    edge_barrier: float = 12.0
    site_reach: float = 1.5
    softness: float = 10.0
    lloyd: float = 0.0
    shade_c: float = 0.02
    detail_precision: float = 1.0
    bounded_gradients: bool = True
    recursive_memory_stages: int = 1
    residual_memory_weight: float = 0.0
    composition_discrepancy_weight: float = 0.0
    allocation_mode: str = "robust integrated error"
    legacy_cartoon_gain: float = 1.0
    legacy_texture_gain: float = 1.0
    legacy_shade_gain: float = 0.0


class TransportVoronoi:
    """Coarse-to-fine anisotropic cell reconstruction."""

    def __init__(self, image, config: Config | None = None):
        init_started = time.perf_counter()
        phase_started = init_started
        self.init_timings = {}
        trace_init = os.environ.get("BFFT_TRACE_INIT", "").lower() in {
            "1", "true", "yes", "on"}

        def mark(name):
            nonlocal phase_started
            now = time.perf_counter()
            self.init_timings[name] = (now - phase_started) * 1000.0
            phase_started = now
            if trace_init:
                print(
                    f"BFFT init {name:24s} "
                    f"{self.init_timings[name]:9.1f} ms",
                    flush=True)

        self.cfg = config or Config()
        self.rgb = _fit_rgb(image, self.cfg.max_side)
        self.h, self.w = self.rgb.shape[:2]
        self.npix = self.h * self.w
        self.xf = np.tile(
            np.arange(self.w, dtype=np.float64), self.h)
        self.yf = np.repeat(
            np.arange(self.h, dtype=np.float64), self.w)
        self.lab = srgb_to_lab(self.rgb)
        mark("input + OKLab")

        t0 = time.perf_counter()
        light = self.lab[..., 0] * 255.0
        self.cartoon, texture = bfft.meyer_split(
            light, lam=self.cfg.lam, mu=self.cfg.mu,
            passes=self.cfg.passes, threads=4)
        # One accurate evaluation of the cartoon-side outer map. This is the
        # state-to-state defect toward the Gilles fixed point, rather than a
        # second smoothing of the cartoon itself.
        projected = bfft.rof(
            light - texture, c=self.cfg.lam, eta=2.0 * self.cfg.lam,
            sweeps=self.cfg.flow_sweeps, tol=0.0, threads=4)
        self.projected_cartoon = projected
        self.flow = self.cartoon - projected
        self.texture = texture
        mark("luma split + TV flow")

        # These products are expensive full colour decompositions and do not
        # participate in the default transport solve.  Build them only when a
        # caller requests the legacy view or enables residual-memory guidance.
        self.legacy_split = None
        self.legacy_shade = None
        if float(self.cfg.residual_memory_weight) > 0.0:
            self.residual_memory = self._build_residual_memory()
        else:
            self.residual_memory = np.zeros(
                (self.h, self.w), dtype=np.float32)
        self.composition_discrepancy = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.cartoon_discrepancy_signed = np.zeros(
            (self.h, self.w), dtype=np.float32)
        self.texture_discrepancy_signed = np.zeros(
            (self.h, self.w), dtype=np.float32)
        self.cartoon_decomp_mse = float("nan")
        self.texture_decomp_mse = float("nan")
        self.decomp_metrics_fresh = False
        self.split_ms = (time.perf_counter() - t0) * 1000.0
        mark("legacy + residual memory")

        self.base_lab = self.lab.copy()
        self.base_lab[..., 0] = np.clip(self.cartoon / 255.0, 0.0, 1.0)
        self.base_lab[..., 1] = ndi.gaussian_filter(
            self.lab[..., 1], 2.0, mode="reflect")
        self.base_lab[..., 2] = ndi.gaussian_filter(
            self.lab[..., 2], 2.0, mode="reflect")
        self.detail_lab = self.lab - self.base_lab

        (self.angle, self.coherence, self.density,
         self.metric_xx, self.metric_xy, self.metric_yy) = self._geometry()
        mark("support geometry")
        edge_table = self._graph_edge_table()
        self._edge_cost_volume = self._edge_cost_stack(edge_table)
        finite_edge_cost = self._edge_cost_volume[
            np.isfinite(self._edge_cost_volume)]
        self._edge_bucket_delta = float(np.min(finite_edge_cost))
        self._edge_bucket_span = int(math.ceil(
            float(np.max(finite_edge_cost)) / self._edge_bucket_delta)) + 2
        # The compiled walks read the packed volume.  Retaining the eight
        # cropped source arrays duplicates almost the entire HD graph.
        self._edge_table_texture = (
            None if _dijkstra_two_best_packed is not None else edge_table)
        self._edge_table_cartoon = None  # historical, never consumed
        mark("graph costs")
        self.flow_sites, self.territories = self._build_flow_territories()
        mark("flow territories")
        self.seeds = self._initial_seed_set()
        mark("initial sites")
        self.marks = np.full(
            len(self.seeds), CARTOON, dtype=np.uint8)
        self.parents = np.full(len(self.seeds), -1, dtype=np.int32)
        self.generations = np.zeros(len(self.seeds), dtype=np.int16)
        self.support_maps = []
        self.support_map_indices = np.full(
            len(self.seeds), -1, dtype=np.int32)
        self.site_regions = np.zeros(len(self.seeds), dtype=np.int32)
        self.owner = np.full(self.npix, -1, dtype=np.int32)
        self.second = np.full(self.npix, -1, dtype=np.int32)
        self.d1 = np.full(self.npix, np.inf)
        self.d2 = np.full(self.npix, np.inf)
        self.base_coeff = np.zeros((len(self.seeds), 3, 3), dtype=np.float64)
        self.base_lo = np.full((len(self.seeds), 3), -np.inf)
        self.base_hi = np.full((len(self.seeds), 3), np.inf)
        self.detail_coeff = np.zeros((len(self.seeds), 3, 3),
                                     dtype=np.float64)
        self.detail_lo = np.full((len(self.seeds), 3), -np.inf)
        self.detail_hi = np.full((len(self.seeds), 3), np.inf)
        self.reconstruction = np.zeros_like(self.lab)
        self.cartoon_reconstruction = np.zeros_like(self.lab)
        self.texture_reconstruction = np.zeros_like(self.lab)
        self.error = np.zeros((self.h, self.w), dtype=np.float64)
        self.cartoon_error = np.zeros((self.h, self.w), dtype=np.float64)
        self.texture_error = np.zeros((self.h, self.w), dtype=np.float64)
        self.allocation_pressure = np.zeros((self.h, self.w),
                                            dtype=np.float64)
        self.cartoon_pressure = np.zeros((self.h, self.w), dtype=np.float64)
        self.texture_pressure = np.zeros((self.h, self.w), dtype=np.float64)
        self.iteration = 0
        self.last_ms = 0.0
        self.psnr = 0.0
        self.last_gain = 0.0
        self.last_action = "initialize"
        self.stagnation = 0
        mark("state allocation")
        self.step(split=False)
        mark("assign + fit + score")
        self.init_timings["total"] = (
            time.perf_counter() - init_started) * 1000.0

    def _build_residual_memory(self):
        """Collect recursive residual support without importing its pixels."""
        current = self.rgb.copy()
        memory = np.zeros((self.h, self.w), dtype=np.float64)
        for stage in range(max(1, int(self.cfg.recursive_memory_stages))):
            split = bfft.meyer_channels(
                current, space="oklab_lc", lam=self.cfg.lam,
                mu=self.cfg.mu, passes=self.cfg.passes, threads=4)
            native_residual = (
                split.residual /
                np.maximum(split.scale[None, None, :], 1e-12))
            memory += (
                0.72 ** stage) * _normalize(
                    np.mean(np.abs(native_residual), axis=2))
            anchor = np.mean(
                split.cartoon, axis=(0, 1), keepdims=True)
            working = (
                split.residual + anchor +
                0.5 * (split.cartoon - anchor) +
                0.5 * split.texture)
            current = _native_from_working(working, split)
        return _normalize(
            ndi.gaussian_filter(memory, 0.7, mode="reflect"))

    # ------------------------------------------------------------------
    # Geometry supplied by BFFT
    # ------------------------------------------------------------------

    def _geometry(self):
        cu = self.cartoon / 255.0
        tv = self.texture / 255.0
        fl = self.flow / 255.0
        cgx = ndi.sobel(cu, axis=1, mode="reflect") / 8.0
        cgy = ndi.sobel(cu, axis=0, mode="reflect") / 8.0
        tgx = ndi.sobel(tv, axis=1, mode="reflect") / 8.0
        tgy = ndi.sobel(tv, axis=0, mode="reflect") / 8.0
        fgx = ndi.sobel(fl, axis=1, mode="reflect") / 8.0
        fgy = ndi.sobel(fl, axis=0, mode="reflect") / 8.0

        # Cartoon and TV-flow normals carry region topology; texture adds the
        # fine orientation that the cells should yield toward.
        gx = cgx + 0.7 * fgx
        gy = cgy + 0.7 * fgy
        jxx = ndi.gaussian_filter(gx * gx + 0.65 * tgx * tgx, 1.5,
                                  mode="reflect")
        jyy = ndi.gaussian_filter(gy * gy + 0.65 * tgy * tgy, 1.5,
                                  mode="reflect")
        jxy = ndi.gaussian_filter(gx * gy + 0.65 * tgx * tgy, 1.5,
                                  mode="reflect")
        disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy ** 2, 0.0))
        normal = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        tangent = normal + np.pi * 0.5
        coherence = disc / np.maximum(jxx + jyy, 1e-12)

        edge = _normalize(np.hypot(cgx, cgy) + 0.8 * np.hypot(fgx, fgy))
        self.edge_strength = edge
        tex_energy = _normalize(
            np.sqrt(ndi.gaussian_filter(tv * tv, 2.0, mode="reflect")))
        oriented = []
        for theta in np.linspace(0.0, np.pi, 4, endpoint=False):
            response = math.cos(theta) * tgx + math.sin(theta) * tgy
            oriented.append(ndi.gaussian_filter(
                response * response, 2.0, mode="reflect"))
        oriented = np.stack(oriented, axis=-1)
        probabilities = oriented / np.maximum(
            oriented.sum(axis=-1, keepdims=True), 1e-15)
        entropy = -np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-15)),
            axis=-1) / math.log(oriented.shape[-1])
        self.texture_activity = tex_energy
        self.texture_entropy = np.clip(entropy, 0.0, 1.0)
        self.gradient_consistency = np.clip(coherence, 0.0, 1.0)
        self.texture_demand = tex_energy * (
            0.35 + 0.65 * self.texture_entropy)
        density = (1.0 + self.cfg.edge_density * edge +
                   self.cfg.texture_density * tex_energy)

        # Spatial Riemannian metric. Texture penalizes travel across its
        # principal gradient; cartoon/TV edges add a stronger local barrier.
        coherence = np.clip(coherence, 0.0, 1.0)
        nvx, nvy = np.cos(normal), np.sin(normal)
        cnorm = np.hypot(gx, gy)
        ncx = gx / np.maximum(cnorm, 1e-12)
        ncy = gy / np.maximum(cnorm, 1e-12)
        # Classic checkpoint: density chooses where sites are useful, but it
        # does not amplify their metric a second time.
        self.anisotropy_pressure = np.ones_like(density)
        texture_cost = self.cfg.anisotropy * coherence
        barrier_cost = self.cfg.edge_barrier * edge
        self.cartoon_metric_xx = 1.0 + barrier_cost * ncx * ncx
        self.cartoon_metric_xy = barrier_cost * ncx * ncy
        self.cartoon_metric_yy = 1.0 + barrier_cost * ncy * ncy
        mxx = (1.0 + texture_cost * nvx * nvx +
               barrier_cost * ncx * ncx)
        mxy = (texture_cost * nvx * nvy +
               barrier_cost * ncx * ncy)
        myy = (1.0 + texture_cost * nvy * nvy +
               barrier_cost * ncy * ncy)
        return tangent, coherence, density, mxx, mxy, myy

    def _graph_edge_table(self, metric_xx=None, metric_xy=None,
                          metric_yy=None):
        """Eight-neighbor costs induced by the spatial metric."""
        metric_xx = self.metric_xx if metric_xx is None else metric_xx
        metric_xy = self.metric_xy if metric_xy is None else metric_xy
        metric_yy = self.metric_yy if metric_yy is None else metric_yy
        table = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ys = slice(max(0, -dy), min(self.h, self.h - dy))
            xs = slice(max(0, -dx), min(self.w, self.w - dx))
            yd = slice(max(0, dy), min(self.h, self.h + dy))
            xd = slice(max(0, dx), min(self.w, self.w + dx))
            mxx = 0.5 * (metric_xx[ys, xs] + metric_xx[yd, xd])
            mxy = 0.5 * (metric_xy[ys, xs] + metric_xy[yd, xd])
            myy = 0.5 * (metric_yy[ys, xs] + metric_yy[yd, xd])
            cost = np.sqrt(np.maximum(
                dx * dx * mxx + 2.0 * dx * dy * mxy +
                dy * dy * myy, 1e-8))
            table.append((dy, dx, max(0, -dy), max(0, -dx), cost))
        return table

    def _edge_cost_stack(self, table):
        """Expand cropped direction costs for the compiled solver.

        The graph weights come from image-derived tensors and do not carry
        useful float64 precision.  A packed float32 graph halves the largest
        persistent HD allocation and reduces Dijkstra's memory traffic; the
        accumulated path distances remain float64.
        """
        stack = np.full((8, self.h, self.w), np.inf, dtype=np.float32)
        for direction, (dy, dx, y0, x0, cost) in enumerate(table):
            ys = slice(max(0, -dy), min(self.h, self.h - dy))
            xs = slice(max(0, -dx), min(self.w, self.w - dx))
            stack[direction, ys, xs] = cost
        return stack

    def _weighted_farthest(self, count):
        """Deterministic density-weighted farthest-point seeding."""
        count = max(1, min(int(count), self.npix))
        raw_weight = self.density.ravel()
        weight = 1.0 + self.cfg.seed_bias * (raw_weight - 1.0)
        first = int(np.argmax(weight))
        chosen = [first]
        min_d2 = ((self.xf - self.xf[first]) ** 2 +
                  (self.yf - self.yf[first]) ** 2)
        blocked = np.zeros(self.npix, dtype=bool)
        blocked[first] = True
        for _ in range(1, count):
            # In two dimensions, density rho implies spacing² proportional
            # to 1/rho. Squaring rho over-populates exactly the high-energy
            # boundaries we are trying to describe economically.
            score = min_d2 * weight
            score[blocked] = -1.0
            idx = int(np.argmax(score))
            if score[idx] < 0:
                break
            chosen.append(idx)
            blocked[idx] = True
            d2 = ((self.xf - self.xf[idx]) ** 2 +
                  (self.yf - self.yf[idx]) ** 2)
            np.minimum(min_d2, d2, out=min_d2)
        return np.column_stack(
            [self.xf[chosen], self.yf[chosen]]).astype(np.float64)

    def _candidate_farthest(self, count, weighted=True):
        """Blue-noise-like coverage without an O(cells * pixels) scan.

        A deterministic R2 sequence supplies a dense, resolution-independent
        candidate pool.  Greedy farthest selection is then O(cells²) with a
        small constant instead of O(cells * full-HD-pixels).
        """
        count = max(1, min(int(count), self.npix))
        candidate_count = min(
            self.npix, max(2048, 16 * count))
        index = np.arange(candidate_count, dtype=np.float64)
        u = np.mod(0.5 + 0.7548776662466927 * index, 1.0)
        v = np.mod(0.5 + 0.5698402909980532 * index, 1.0)
        x = np.clip(
            np.rint(u * max(self.w - 1, 0)).astype(np.int64),
            0, self.w - 1)
        y = np.clip(
            np.rint(v * max(self.h - 1, 0)).astype(np.int64),
            0, self.h - 1)
        encoded = np.unique(y * self.w + x)
        anchors = np.array([
            0, self.w - 1, (self.h - 1) * self.w,
            self.npix - 1, (self.h // 2) * self.w + self.w // 2,
        ], dtype=np.int64)
        encoded = np.unique(np.concatenate((encoded, anchors)))
        x = (encoded % self.w).astype(np.float64)
        y = (encoded // self.w).astype(np.float64)
        if weighted:
            density = self.density.ravel()[encoded]
            weight = 1.0 + float(self.cfg.seed_bias) * (density - 1.0)
        else:
            weight = np.ones(encoded.size, dtype=np.float64)

        centre = np.array([0.5 * (self.w - 1), 0.5 * (self.h - 1)])
        first = int(np.argmin(
            (x - centre[0]) ** 2 + (y - centre[1]) ** 2))
        chosen = [first]
        minimum = (x - x[first]) ** 2 + (y - y[first]) ** 2
        blocked = np.zeros(encoded.size, dtype=bool)
        blocked[first] = True
        for _ in range(1, count):
            score = minimum * weight
            score[blocked] = -1.0
            selected = int(np.argmax(score))
            if score[selected] < 0.0:
                break
            chosen.append(selected)
            blocked[selected] = True
            distance = (
                (x - x[selected]) ** 2 + (y - y[selected]) ** 2)
            np.minimum(minimum, distance, out=minimum)
        return np.column_stack((x[chosen], y[chosen]))

    def _uniform_farthest(self, count):
        """Uniform deterministic blue-noise-like coverage."""
        count = max(1, min(int(count), self.npix))
        first = (self.h // 2) * self.w + self.w // 2
        chosen = [first]
        min_d2 = ((self.xf - self.xf[first]) ** 2 +
                  (self.yf - self.yf[first]) ** 2)
        blocked = np.zeros(self.npix, dtype=bool)
        blocked[first] = True
        for _ in range(1, count):
            score = min_d2.copy()
            score[blocked] = -1.0
            idx = int(np.argmax(score))
            if score[idx] < 0:
                break
            chosen.append(idx)
            blocked[idx] = True
            d2 = ((self.xf - self.xf[idx]) ** 2 +
                  (self.yf - self.yf[idx]) ** 2)
            np.minimum(min_d2, d2, out=min_d2)
        return np.column_stack(
            [self.xf[chosen], self.yf[chosen]]).astype(np.float64)

    def _build_flow_territories(self):
        """Freeze coarse transport compartments before marked germination."""
        count = max(1, min(int(self.cfg.territory_count),
                           int(self.cfg.initial_cells), self.npix))
        if count == 1:
            sites = np.array(
                [[float(self.w // 2), float(self.h // 2)]],
                dtype=np.float64)
            return sites, np.ones(
                (self.h, self.w), dtype=np.int32)
        sites = self._candidate_farthest(count, weighted=False)
        markers = np.zeros((self.h, self.w), dtype=np.int32)
        for label, (x, y) in enumerate(sites, start=1):
            markers[int(round(y)), int(round(x))] = label
        wave = _normalize(np.abs(self.flow), 99.0)
        barrier = ndi.gaussian_filter(
            0.72 * self.edge_strength + 0.28 * wave,
            0.8, mode="reflect")
        regions = watershed(
            barrier, markers=markers,
            connectivity=np.ones((3, 3), dtype=np.uint8))
        return sites, regions.astype(np.int32)

    def _initial_seed_set(self):
        """Uniform blue noise by default; content bias remains optional."""
        return self._candidate_farthest(self.cfg.initial_cells)

    # ------------------------------------------------------------------
    # Anisotropic ownership and local color models
    # ------------------------------------------------------------------

    def _site_frames(self):
        xi = np.clip(np.rint(self.seeds[:, 0]).astype(int), 0, self.w - 1)
        yi = np.clip(np.rint(self.seeds[:, 1]).astype(int), 0, self.h - 1)
        angle = self.angle[yi, xi]
        coherence = self.coherence[yi, xi]
        # Anisotropy is a continuous local propensity. It is not a discrete
        # cartoon/texture species decision.
        ratio = 1.0 + self.cfg.anisotropy * coherence
        return angle, np.clip(ratio, 1.0, 12.0)

    def _assign(self):
        """Two-best additively weighted geodesic power assignment.

        A multi-label Dijkstra walk samples the BFFT metric at every pixel
        crossed. This replaces the former frozen ellipse at each centroid.
        """
        n = len(self.seeds)
        best = np.full(self.npix, np.inf, dtype=np.float64)
        second = np.full(self.npix, np.inf, dtype=np.float64)
        owner = np.full(self.npix, -1, dtype=np.int32)
        runner = np.full(self.npix, -1, dtype=np.int32)
        density_flat = self.density.ravel()
        territory_flat = self.territories.ravel()
        density_scale = max(float(np.median(density_flat)), 1e-9)
        seed_x = np.clip(
            np.rint(self.seeds[:, 0]).astype(np.int64), 0, self.w - 1)
        seed_y = np.clip(
            np.rint(self.seeds[:, 1]).astype(np.int64), 0, self.h - 1)
        reach = self.cfg.site_reach * np.sqrt(
            density_flat[seed_y * self.w + seed_x] / density_scale)
        if _dijkstra_two_best_bucket is not None:
            owner, runner, best, second = _dijkstra_two_best_bucket(
                seed_x, seed_y, reach.astype(np.float64),
                self._edge_cost_volume, self.h, self.w,
                self._edge_bucket_delta, self._edge_bucket_span)
            self.owner, self.second = owner, runner
            self.d1, self.d2 = best, second
            self.site_regions = np.zeros(n, dtype=np.int32)
            return
        heap = []
        site_regions = np.empty(n, dtype=np.int32)
        for i in range(n):
            x = int(np.clip(round(self.seeds[i, 0]), 0, self.w - 1))
            y = int(np.clip(round(self.seeds[i, 1]), 0, self.h - 1))
            p = y * self.w + x
            site_regions[i] = territory_flat[p]
            reach = self.cfg.site_reach * math.sqrt(
                density_flat[p] / density_scale)
            d = -reach
            if d < best[p]:
                second[p], runner[p] = best[p], owner[p]
                best[p], owner[p] = d, i
            elif i != owner[p] and d < second[p]:
                second[p], runner[p] = d, i
            heapq.heappush(heap, (d, p, i))

        tolerance = 1e-12
        while heap:
            d, p, site = heapq.heappop(heap)
            valid = ((owner[p] == site and d <= best[p] + tolerance) or
                     (runner[p] == site and d <= second[p] + tolerance))
            if not valid:
                continue
            y, x = divmod(p, self.w)
            # TV flow controls where germs originate. Ownership itself is
            # bounded by the texture-aware metric, as in the original model;
            # turning the TV watershed into a hard wall creates seams and
            # prevents useful cells from filling across nucleation basins.
            table = self._edge_table_texture
            for dy, dx, y0, x0, costs in table:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= self.h or nx < 0 or nx >= self.w:
                    continue
                q = ny * self.w + nx
                nd = d + float(costs[y - y0, x - x0])
                if owner[q] == site:
                    if nd + tolerance < best[q]:
                        best[q] = nd
                        heapq.heappush(heap, (nd, q, site))
                elif runner[q] == site:
                    if nd + tolerance < second[q]:
                        second[q] = nd
                        if second[q] < best[q]:
                            best[q], second[q] = second[q], best[q]
                            owner[q], runner[q] = runner[q], owner[q]
                        heapq.heappush(heap, (nd, q, site))
                elif nd + tolerance < best[q]:
                    second[q], runner[q] = best[q], owner[q]
                    best[q], owner[q] = nd, site
                    heapq.heappush(heap, (nd, q, site))
                elif nd + tolerance < second[q]:
                    second[q], runner[q] = nd, site
                    heapq.heappush(heap, (nd, q, site))
        self.owner, self.second = owner, runner
        self.d1, self.d2 = best, second
        self.site_regions = np.zeros(n, dtype=np.int32)

    def _fit_models(self):
        n = len(self.seeds)
        angles, _ = self._site_frames()
        ids = self.owner
        count = np.bincount(ids, minlength=n).astype(np.float64)
        count_safe = np.maximum(count, 1.0)
        sx = self.seeds[ids, 0]
        sy = self.seeds[ids, 1]
        ct = np.cos(angles[ids])
        st = np.sin(angles[ids])
        dx, dy = self.xf - sx, self.yf - sy
        q = dx * ct + dy * st
        r = -dx * st + dy * ct
        sum_q = np.bincount(ids, weights=q, minlength=n)
        sum_r = np.bincount(ids, weights=r, minlength=n)
        sum_q2 = np.bincount(ids, weights=q * q, minlength=n)
        sum_r2 = np.bincount(ids, weights=r * r, minlength=n)
        sum_qr = np.bincount(ids, weights=q * r, minlength=n)
        normal = np.zeros((n, 3, 3), dtype=np.float64)
        normal[:, 0, 0] = count_safe
        normal[:, 0, 1] = normal[:, 1, 0] = sum_q
        normal[:, 0, 2] = normal[:, 2, 0] = sum_r
        normal[:, 1, 1] = sum_q2 + 1e-4
        normal[:, 2, 2] = sum_r2 + 1e-4
        normal[:, 1, 2] = normal[:, 2, 1] = sum_qr
        def fit_target(target, slope_limit):
            coeff = np.zeros((n, 3, 3), dtype=np.float64)
            bounds_lo = np.zeros((n, 3), dtype=np.float64)
            bounds_hi = np.zeros((n, 3), dtype=np.float64)
            flat = target.reshape(-1, 3)
            for channel in range(3):
                y = flat[:, channel]
                sum_y = np.bincount(ids, weights=y, minlength=n)
                sum_y2 = np.bincount(ids, weights=y * y, minlength=n)
                sum_qy = np.bincount(ids, weights=q * y, minlength=n)
                sum_ry = np.bincount(ids, weights=r * y, minlength=n)
                rhs = np.column_stack([sum_y, sum_qy, sum_ry])
                try:
                    fitted = np.linalg.solve(
                        normal, rhs[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    fitted = np.zeros((n, 3), dtype=np.float64)
                    fitted[:, 0] = sum_y / count_safe
                fitted[:, 1:] = np.clip(
                    fitted[:, 1:], -slope_limit, slope_limit)
                coeff[:, channel, :] = fitted
                mean = sum_y / count_safe
                std = np.sqrt(np.maximum(
                    sum_y2 / count_safe - mean * mean, 0.0))
                margin = 2.5 * std + (
                    0.008 if channel == 0 else 0.003)
                bounds_lo[:, channel] = mean - margin
                bounds_hi[:, channel] = mean + margin
            return coeff, bounds_lo, bounds_hi

        (self.base_coeff, self.base_lo,
         self.base_hi) = fit_target(self.base_lab, 0.055)
        (self.detail_coeff, self.detail_lo,
         self.detail_hi) = fit_target(self.detail_lab, 0.08)

        # Both species participate in the partition of unity. Their mark
        # changes birth likelihood, reach, and geometry—not whether they are
        # allowed to explain the signal under their footprint. Suppressing
        # detail on cartoon-marked owners creates literal reconstruction
        # holes whenever no texture germ is among a pixel's two neighbors.

        # Backward-compatible research handles.
        self.coeff = self.detail_coeff
        self.model_lo = self.detail_lo
        self.model_hi = self.detail_hi
        return q, r

    def _predict_for(self, ids, layer="detail"):
        angles, _ = self._site_frames()
        sx = self.seeds[ids, 0]
        sy = self.seeds[ids, 1]
        dx, dy = self.xf - sx, self.yf - sy
        ct, st = np.cos(angles[ids]), np.sin(angles[ids])
        q = dx * ct + dy * st
        r = -dx * st + dy * ct
        if layer == "base":
            coeff, bounds_lo, bounds_hi = (
                self.base_coeff, self.base_lo, self.base_hi)
        else:
            coeff, bounds_lo, bounds_hi = (
                self.detail_coeff, self.detail_lo, self.detail_hi)
        pred = (coeff[ids, :, 0] +
                coeff[ids, :, 1] * q[:, None] +
                coeff[ids, :, 2] * r[:, None])
        if self.cfg.bounded_gradients:
            pred = np.minimum(np.maximum(pred, bounds_lo[ids]),
                              bounds_hi[ids])
        return pred

    def _render(self):
        base_first = self._predict_for(self.owner, "base")
        detail_first = self._predict_for(self.owner, "detail")
        valid_second = self.second >= 0
        if self.cfg.softness <= 0.0 or not np.any(valid_second):
            base, detail = base_first, detail_first
        else:
            safe_second = np.where(valid_second, self.second, self.owner)
            base_other = self._predict_for(safe_second, "base")
            detail_other = self._predict_for(safe_second, "detail")
            gap = self.d2 - self.d1
            z = np.clip(0.5 * self.cfg.softness * gap, -50.0, 50.0)
            w = 1.0 / (1.0 + np.exp(-z))
            w[~valid_second] = 1.0
            base = (base_first * w[:, None] +
                    base_other * (1.0 - w[:, None]))
            detail = (detail_first * w[:, None] +
                      detail_other * (1.0 - w[:, None]))
        precision = float(np.clip(self.cfg.detail_precision, 0.0, 1.5))
        self.cartoon_reconstruction = base.reshape(self.h, self.w, 3)
        self.texture_reconstruction = detail.reshape(self.h, self.w, 3)
        self.reconstruction = (
            self.cartoon_reconstruction +
            precision * self.texture_reconstruction)
        base_delta = self.base_lab - self.cartoon_reconstruction
        detail_delta = self.detail_lab - self.texture_reconstruction
        delta = self.lab - self.reconstruction
        self.cartoon_error = np.sqrt(
            base_delta[..., 0] ** 2 +
            1.5 * (base_delta[..., 1] ** 2 + base_delta[..., 2] ** 2))
        self.texture_error = np.sqrt(
            detail_delta[..., 0] ** 2 +
            1.5 * (detail_delta[..., 1] ** 2 +
                   detail_delta[..., 2] ** 2))
        err = np.sqrt(delta[..., 0] ** 2 +
                      1.5 * (delta[..., 1] ** 2 + delta[..., 2] ** 2))
        self.error = err
        self._update_allocation_pressure()
        mse = float(np.mean((self.rgb - np.clip(
            lab_to_srgb(self.reconstruction), 0.0, 1.0)) ** 2))
        self.rgb_mse = mse
        self.psnr = -10.0 * math.log10(max(mse, 1e-12))

    def _update_allocation_pressure(self):
        """Estimate reducible error rather than chasing edge outliers.

        A capped robust loss prevents a one-pixel discontinuity from owning
        the entire budget. Integrated pressure rewards large under-resolved
        regions. Geodesic clearance restores the space-filling behavior of
        blue noise, while a soft edge penalty avoids proposing every child
        directly on the same contour.
        """
        def robust_loss(error):
            cap = max(float(np.percentile(error, 85.0)),
                      2.5 * float(np.median(error)), 1e-4)
            loss = np.minimum(error * error, cap * cap)
            return ndi.gaussian_filter(loss, 1.25, mode="reflect")

        cartoon_loss = robust_loss(self.cartoon_error)
        texture_loss = robust_loss(self.texture_error)
        clearance = np.maximum(
            self.d1.reshape(self.h, self.w), 0.0)
        clearance_scale = max(float(np.percentile(clearance, 90.0)), 1e-6)
        coverage = 1.0 + 0.75 * np.sqrt(
            np.clip(clearance / clearance_scale, 0.0, 4.0))
        edge_penalty = 1.0 / (1.0 + 2.5 * self.edge_strength)
        self.texture_likelihood = self.texture_activity
        cartoon_pressure = cartoon_loss * coverage * edge_penalty
        texture_pressure = texture_loss * coverage * edge_penalty

        mode = str(self.cfg.allocation_mode).lower()
        needs_decomposition = (
            mode == "rgb + decomposition gain" or
            float(self.cfg.composition_discrepancy_weight) > 0.0)
        if needs_decomposition:
            self.refresh_decomposition_metrics()
        else:
            self.decomp_metrics_fresh = False
        guidance = (
            1.0 +
            max(float(self.cfg.residual_memory_weight), 0.0) *
            self.residual_memory +
            max(float(self.cfg.composition_discrepancy_weight), 0.0) *
            self.composition_discrepancy)

        self.cartoon_pressure = cartoon_pressure * guidance
        self.texture_pressure = texture_pressure * guidance
        robust_pressure = (
            robust_loss(self.error) * coverage * edge_penalty * guidance)
        self.expected_gain = np.zeros_like(robust_pressure)
        if mode == "robust integrated error":
            self.allocation_pressure = robust_pressure
            return

        # Closed-form gain of a Gaussian-windowed constant-plus-gradient
        # correction.  It prices what the actual affine model can remove,
        # rather than treating every large residual as equally reducible.
        spacing = math.sqrt(self.npix / max(len(self.seeds), 1))
        sigma = max(0.55 * spacing, 0.8)
        residual = self.lab - self.reconstruction
        gain = self._affine_gain_field(
            residual, (1.0, 1.5, 1.5), sigma)

        if mode == "rgb + decomposition gain":
            rgb_scale = max(
                float(np.median(gain[gain > 0]))
                if np.any(gain > 0) else 0.0, 1e-12)
            for plane in (
                    self.cartoon_discrepancy_signed,
                    self.texture_discrepancy_signed):
                component = self._affine_gain_field(
                    plane[..., None], (1.0,), sigma)
                component_scale = max(
                    float(np.median(component[component > 0]))
                    if np.any(component > 0) else 0.0, 1e-12)
                gain += 0.35 * rgb_scale * component / component_scale

        self.expected_gain = gain
        self.allocation_pressure = (
            gain * coverage * edge_penalty * guidance)

    def _affine_gain_field(self, fields, channel_weights, sigma):
        """Price a local affine correction on a bounded working grid.

        Direct FIR Gaussian filtering costs ``O(P * sigma)`` because SciPy's
        kernel radius grows with the cell spacing.  At HD and a coarse cell
        count that silently dominates rendering.  Sampling the support field
        at its own physical scale keeps the working sigma at most six pixels.
        Derivatives are converted back to native-pixel units, so
        ``sigma² |grad|²`` retains the same dimensional meaning.
        """
        fields = np.asarray(fields, dtype=np.float64)
        sigma = max(float(sigma), 0.0)
        reduction = max(1, int(math.ceil(sigma / 6.0)))
        if reduction > 1:
            oh = max(8, int(math.ceil(self.h / reduction)))
            ow = max(8, int(math.ceil(self.w / reduction)))
            working = resize(
                fields, (oh, ow, fields.shape[2]),
                order=1, anti_aliasing=True, preserve_range=True)
        else:
            oh, ow = self.h, self.w
            working = fields

        scale_y = self.h / oh
        scale_x = self.w / ow
        sigma_y = sigma / scale_y
        sigma_x = sigma / scale_x
        gain = np.zeros((oh, ow), dtype=np.float64)
        for channel, channel_weight in enumerate(channel_weights):
            plane = working[..., channel]
            mean = ndi.gaussian_filter(
                plane, (sigma_y, sigma_x), mode="reflect")
            gx = ndi.gaussian_filter(
                plane, (sigma_y, sigma_x), order=(0, 1), mode="reflect")
            gy = ndi.gaussian_filter(
                plane, (sigma_y, sigma_x), order=(1, 0), mode="reflect")
            gain += float(channel_weight) * (
                mean * mean +
                (sigma / scale_x) ** 2 * gx * gx +
                (sigma / scale_y) ** 2 * gy * gy)
        if reduction > 1:
            gain = resize(
                gain, (self.h, self.w), order=1,
                anti_aliasing=False, preserve_range=True)
        return gain

    def refresh_decomposition_metrics(self):
        """Compute the expensive decomposition discrepancy on demand."""
        current_light = np.clip(
            self.reconstruction[..., 0], 0.0, 1.0) * 255.0
        current_cartoon, current_texture = bfft.meyer_split(
            current_light, lam=self.cfg.lam, mu=self.cfg.mu,
            passes=self.cfg.passes, threads=4)
        self.cartoon_discrepancy_signed = (
            self.cartoon - current_cartoon) / 255.0
        self.texture_discrepancy_signed = (
            self.texture - current_texture) / 255.0
        self.composition_discrepancy = _normalize(
            np.abs(self.cartoon - current_cartoon) +
            1.25 * np.abs(self.texture - current_texture))
        self.cartoon_decomp_mse = float(np.mean(
            ((self.cartoon - current_cartoon) / 255.0) ** 2))
        self.texture_decomp_mse = float(np.mean(
            ((self.texture - current_texture) / 255.0) ** 2))
        self.decomp_metrics_fresh = True
        return self.cartoon_decomp_mse, self.texture_decomp_mse

    def _coupled_affine_field(self, target, softness):
        """Joint least-squares solve under the actual overlap renderer."""
        n = len(self.seeds)
        spacing = math.sqrt(self.npix / max(n, 1))
        angles, _ = self._site_frames()
        valid = self.second >= 0
        other = np.where(valid, self.second, self.owner)
        gap = self.d2 - self.d1
        z = np.clip(0.5 * float(softness) * gap, -50.0, 50.0)
        first_weight = 1.0 / (1.0 + np.exp(-z))
        first_weight[~valid] = 1.0
        other_weight = 1.0 - first_weight

        def basis(site_ids):
            sx, sy = self.seeds[site_ids, 0], self.seeds[site_ids, 1]
            ct, st = np.cos(angles[site_ids]), np.sin(angles[site_ids])
            dx, dy = self.xf - sx, self.yf - sy
            return np.column_stack([
                np.ones(self.npix),
                (dx * ct + dy * st) / max(spacing, 1e-9),
                (-dx * st + dy * ct) / max(spacing, 1e-9),
            ])

        first_basis = basis(self.owner)
        other_basis = basis(other)
        rows = np.repeat(np.arange(self.npix, dtype=np.int32), 3)
        components = np.tile(
            np.arange(3, dtype=np.int32), self.npix)
        first_cols = 3 * np.repeat(self.owner, 3) + components
        first_data = (
            first_basis * first_weight[:, None]).ravel()
        visible = np.flatnonzero(valid)
        other_rows = np.repeat(visible, 3)
        other_components = np.tile(
            np.arange(3, dtype=np.int32), len(visible))
        other_cols = (
            3 * np.repeat(other[valid], 3) + other_components)
        other_data = (
            other_basis[valid] *
            other_weight[valid, None]).ravel()
        regularization = np.tile(
            np.array([1e-5, 2e-3, 2e-3], dtype=np.float64), n)
        regularizer_rows = np.arange(
            self.npix, self.npix + 3 * n, dtype=np.int32)
        regularizer_cols = np.arange(3 * n, dtype=np.int32)
        design = sparse.coo_matrix((
            np.concatenate([
                first_data, other_data, np.sqrt(regularization)]),
            (
                np.concatenate([rows, other_rows, regularizer_rows]),
                np.concatenate([
                    first_cols, other_cols, regularizer_cols]),
            )), shape=(self.npix + 3 * n, 3 * n)).tocsr()
        padded = np.zeros(self.npix + 3 * n, dtype=np.float64)
        fitted = np.zeros((self.npix, 3), dtype=np.float64)
        for channel in range(3):
            padded[:self.npix] = target[..., channel].ravel()
            coeff = lsmr(
                design, padded, atol=2e-6, btol=2e-6,
                maxiter=160)[0].reshape(n, 3)
            first_prediction = np.sum(
                coeff[self.owner] * first_basis, axis=1)
            other_prediction = np.sum(
                coeff[other] * other_basis, axis=1)
            fitted[:, channel] = (
                first_weight * first_prediction +
                other_weight * other_prediction)
        return fitted.reshape(self.h, self.w, 3)

    def solve_coupled(
            self, multiscale=True, cartoon_softness=4.0,
            texture_softness=16.0):
        """Make the fitted fields agree with their partition of unity."""
        t0 = time.perf_counter()
        if multiscale:
            base = self._coupled_affine_field(
                self.base_lab, softness=cartoon_softness)
            detail = self._coupled_affine_field(
                self.detail_lab, softness=texture_softness)
        else:
            base = self._coupled_affine_field(
                self.base_lab, softness=self.cfg.softness)
            detail = self._coupled_affine_field(
                self.detail_lab, softness=self.cfg.softness)
        precision = float(np.clip(
            self.cfg.detail_precision, 0.0, 1.5))
        self.cartoon_reconstruction = base
        self.texture_reconstruction = detail
        self.reconstruction = base + precision * detail
        base_delta = self.base_lab - base
        detail_delta = self.detail_lab - detail
        delta = self.lab - self.reconstruction
        self.cartoon_error = np.sqrt(
            base_delta[..., 0] ** 2 +
            1.5 * np.sum(base_delta[..., 1:] ** 2, axis=2))
        self.texture_error = np.sqrt(
            detail_delta[..., 0] ** 2 +
            1.5 * np.sum(detail_delta[..., 1:] ** 2, axis=2))
        self.error = np.sqrt(
            delta[..., 0] ** 2 +
            1.5 * np.sum(delta[..., 1:] ** 2, axis=2))
        self._update_allocation_pressure()
        rgb = np.clip(lab_to_srgb(self.reconstruction), 0.0, 1.0)
        self.rgb_mse = float(np.mean((self.rgb - rgb) ** 2))
        self.psnr = -10.0 * math.log10(max(self.rgb_mse, 1e-12))
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.last_action = (
            "coupled multiscale solve" if multiscale else
            "coupled solve")
        return self.reconstruction

    def set_detail_precision(self, value):
        """Change the texture evidence gate without refitting geometry."""
        self.cfg.detail_precision = float(value)
        self._render()

    def legacy_recomposition(self):
        """Render the pre-transport colour-safe Meyer layer mixer."""
        if self.legacy_split is None:
            self.legacy_split = bfft.meyer_channels(
                self.rgb, space="oklab_lc", lam=self.cfg.lam,
                mu=self.cfg.mu, passes=self.cfg.passes, threads=4)
            self.legacy_shade = np.stack([
                bfft.shade(
                    self.legacy_split.cartoon[..., i],
                    c=self.cfg.shade_c, threads=4)
                for i in range(self.legacy_split.planes.shape[2])
            ], axis=-1)
        sp = self.legacy_split
        out = (sp.planes - sp.cartoon - sp.texture +
               self.cfg.legacy_cartoon_gain * sp.cartoon +
               self.cfg.legacy_texture_gain * sp.texture +
               self.cfg.legacy_shade_gain * self.legacy_shade)
        native = np.empty_like(out)
        for i in range(out.shape[2]):
            native[..., i] = sp.offset[i] + out[..., i] / sp.scale[i]
        rgb = _from_working(native, sp.space, sp.carried,
                            sp.carried.get("_was_gray", False))
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        return np.clip(rgb[..., :3], 0.0, 1.0)

    # ------------------------------------------------------------------
    # Coarse-to-fine update
    # ------------------------------------------------------------------

    def _lloyd_update(self):
        amount = float(np.clip(self.cfg.lloyd, 0.0, 1.0))
        if amount <= 0.0:
            return
        n = len(self.seeds)
        weights = self.density.ravel() * (0.25 + self.error.ravel())
        sw = np.bincount(self.owner, weights=weights, minlength=n)
        cx = np.bincount(self.owner, weights=weights * self.xf,
                         minlength=n) / np.maximum(sw, 1e-12)
        cy = np.bincount(self.owner, weights=weights * self.yf,
                         minlength=n) / np.maximum(sw, 1e-12)
        valid = sw > 0
        proposed = self.seeds.copy()
        proposed[valid, 0] = ((1.0 - amount) * self.seeds[valid, 0] +
                              amount * cx[valid])
        proposed[valid, 1] = ((1.0 - amount) * self.seeds[valid, 1] +
                              amount * cy[valid])
        px = np.clip(np.rint(proposed[:, 0]).astype(int), 0, self.w - 1)
        py = np.clip(np.rint(proposed[:, 1]).astype(int), 0, self.h - 1)
        stays_inside = self.territories[py, px] == self.site_regions
        accepted = valid & stays_inside
        self.seeds[accepted] = proposed[accepted]

    def _birth_mark(self, index):
        """The classic checkpoint has one global cell population."""
        return CARTOON

    def _subdivide(self):
        room = self.cfg.max_cells - len(self.seeds)
        budget = min(max(0, int(self.cfg.split_batch)), room)
        if budget <= 0:
            return self._rebalance()
        n = len(self.seeds)
        pressure = self.allocation_pressure.ravel()
        # Total reducible pressure, not average pixel error. Average error
        # gives tiny boundary cells an effectively permanent priority.
        score = np.bincount(self.owner, weights=pressure, minlength=n)
        cells = np.argsort(score)[::-1]
        best_pixel = _cell_pressure_argmax(self.owner, pressure, n)
        additions = []
        addition_marks = []
        addition_parents = []
        addition_generations = []
        for cell in cells:
            idx = int(best_pixel[cell])
            if idx < 0:
                continue
            # Pressure already contains robust residual, metric clearance,
            # and an edge-crowding penalty. Do not multiply by edge density
            # again here.
            x, y = self.xf[idx], self.yf[idx]
            if self.seeds.size:
                d2 = ((self.seeds[:, 0] - x) ** 2 +
                      (self.seeds[:, 1] - y) ** 2)
                if float(d2.min()) < 2.25:
                    continue
            if additions:
                added = np.asarray(additions)
                d2 = ((added[:, 0] - x) ** 2 +
                      (added[:, 1] - y) ** 2)
                if float(d2.min()) < 2.25:
                    continue
            additions.append((x, y))
            addition_marks.append(self._birth_mark(idx))
            parent = int(self.owner[idx])
            addition_parents.append(parent)
            addition_generations.append(int(self.generations[parent]) + 1)
            if len(additions) >= budget:
                break
        if additions:
            self.seeds = np.vstack([self.seeds, np.asarray(additions)])
            self.marks = np.concatenate([
                self.marks, np.asarray(addition_marks, dtype=np.uint8)])
            self.parents = np.concatenate([
                self.parents, np.asarray(addition_parents, dtype=np.int32)])
            self.generations = np.concatenate([
                self.generations,
                np.asarray(addition_generations, dtype=np.int16)])
            self.support_map_indices = np.concatenate([
                self.support_map_indices,
                np.full(len(additions), -1, dtype=np.int32)])
            return "add"
        return "none"

    def _rebalance(self):
        """Exchange low-value sites for uncovered residual pressure.

        Pure subdivision freezes every historical mistake. At a fixed budget
        this small birth/death step lets the representation recover from early
        edge crowding instead of declaring convergence merely because the
        array is full.
        """
        n = len(self.seeds)
        if n < 8:
            return "none"
        pressure = self.allocation_pressure.ravel()
        value = np.bincount(
            self.owner, weights=pressure, minlength=n)
        budget = min(max(1, int(self.cfg.split_batch) // 2),
                     max(1, n // 20))
        weak = np.argsort(value)[:budget]
        keep = np.ones(n, dtype=bool)
        keep[weak] = False
        kept_seeds = self.seeds[keep]

        candidates = pressure.copy()
        blocked = np.zeros(self.npix, dtype=bool)
        # Existing useful sites retain an exclusion disk.
        for x, y in kept_seeds:
            ix, iy = int(round(x)), int(round(y))
            x0, x1 = max(0, ix - 2), min(self.w, ix + 3)
            y0, y1 = max(0, iy - 2), min(self.h, iy + 3)
            blocked.reshape(self.h, self.w)[y0:y1, x0:x1] = True
        candidates[blocked] = -1.0

        replacements = []
        replacement_marks = []
        for _ in range(len(weak)):
            idx = int(np.argmax(candidates))
            if candidates[idx] <= 0:
                break
            y, x = divmod(idx, self.w)
            replacements.append((float(x), float(y)))
            replacement_marks.append(self._birth_mark(idx))
            radius = 4
            x0, x1 = max(0, x - radius), min(self.w, x + radius + 1)
            y0, y1 = max(0, y - radius), min(self.h, y + radius + 1)
            candidates.reshape(self.h, self.w)[y0:y1, x0:x1] = -1.0
        if not replacements:
            return "none"
        # If fewer valid births were found, retain the unmatched weak sites.
        moved = len(replacements)
        self.seeds[weak[:moved]] = np.asarray(replacements)
        self.marks[weak[:moved]] = np.asarray(
            replacement_marks, dtype=np.uint8)
        return "swap"

    def step(self, split=True):
        t0 = time.perf_counter()
        if split:
            # The preceding state already has a fitted assignment. Use its
            # residual to place children, then solve ownership only once.
            old_seeds = self.seeds.copy()
            old_marks = self.marks.copy()
            old_parents = self.parents.copy()
            old_generations = self.generations.copy()
            old_psnr = float(self.psnr)
            self._lloyd_update()
            action = self._subdivide()
            self._assign()
            self._fit_models()
            self._render()
            gain = self.psnr - old_psnr
            # At a fixed budget, reject exchanges that merely move the error
            # around. Re-solving the old state is slower but makes refinement
            # an actual descent rather than an animation.
            if action == "swap" and gain <= 1e-4:
                self.seeds = old_seeds
                self.marks = old_marks
                self.parents = old_parents
                self.generations = old_generations
                self._assign()
                self._fit_models()
                self._render()
                self.last_action = "rejected swap"
                self.last_gain = 0.0
                self.stagnation += 1
            else:
                self.last_action = action
                self.last_gain = gain
                self.stagnation = 0 if gain > 1e-4 else self.stagnation + 1
                self.iteration += 1
        else:
            self._assign()
            self._fit_models()
            self._render()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return self.reconstruction

    # ------------------------------------------------------------------
    # Viewer products
    # ------------------------------------------------------------------

    def view(self, mode):
        if mode == "Original":
            return self.rgb
        if mode == "Legacy BFFT recomposition":
            return self.legacy_recomposition()
        if mode == "Persistent cartoon":
            return np.clip(lab_to_srgb(self.base_lab), 0.0, 1.0)
        if mode == "Detail field":
            scale = max(float(np.percentile(np.abs(self.detail_lab), 99)), 1e-6)
            return np.clip(0.5 + self.detail_lab / (2.0 * scale), 0.0, 1.0)
        if mode == "Cartoon":
            g = np.clip(self.cartoon / 255.0, 0.0, 1.0)
            return np.repeat(g[..., None], 3, axis=2)
        if mode == "Texture":
            g = np.clip(0.5 + self.texture / 255.0, 0.0, 1.0)
            return np.repeat(g[..., None], 3, axis=2)
        if mode == "TV flow":
            scale = max(float(np.percentile(np.abs(self.flow), 99.0)), 1e-6)
            g = np.clip(0.5 + 0.5 * self.flow / scale, 0.0, 1.0)
            return np.repeat(g[..., None], 3, axis=2)
        if mode == "Flow magnitude":
            g = _normalize(np.abs(self.flow), 99.5)
            return np.stack([g, g * g, 0.2 * (1.0 - g)], axis=-1)
        if mode == "Flow territories":
            ids = self.territories.astype(np.uint32)
            r = ((ids * 1664525 + 1013904223) & 255) / 255.0
            g = ((ids * 22695477 + 1) & 255) / 255.0
            b = ((ids * 1103515245 + 12345) & 255) / 255.0
            color = np.stack([r, g, b], axis=-1)
            edge = ndi.morphological_gradient(
                self.territories, size=(3, 3)) != 0
            color[edge] = (1.0, 1.0, 1.0)
            return color
        if mode == "Texture activity":
            g = self.texture_activity
            return np.stack([g, 0.2 * g, 1.0 - g], axis=-1)
        if mode == "Texture entropy":
            g = self.texture_entropy
            return np.stack([0.15 * g, g, 1.0 - g], axis=-1)
        if mode == "Gradient consistency":
            g = self.gradient_consistency
            return np.stack([0.08 * g, 0.6 * g, g], axis=-1)
        if mode == "Texture likelihood":
            g = self.texture_likelihood
            return np.stack([g, 0.15 * g, 0.85 * (1.0 - g)], axis=-1)
        if mode == "Texture demand":
            g = _normalize(self.texture_demand, 99.0)
            return np.stack([g, g * g, 0.25 * (1.0 - g)], axis=-1)
        if mode == "Seed density":
            g = _normalize(self.density)
            return np.stack([g, 0.25 * g, 1.0 - g], axis=-1)
        if mode == "Error":
            g = _normalize(self.error, 99.5)
            return np.stack([g, g * g, 0.15 * (1.0 - g)], axis=-1)
        if mode == "Allocation pressure":
            g = _normalize(self.allocation_pressure, 99.0)
            return np.stack([0.15 * g, g, 1.0 - g], axis=-1)
        if mode == "Expected affine gain":
            g = _normalize(self.expected_gain, 99.0)
            return np.stack([g, 0.45 * g, 1.0 - g], axis=-1)
        if mode == "Cartoon pressure":
            g = _normalize(self.cartoon_pressure, 99.0)
            return np.stack([g, 0.72 * g, 0.08 * g], axis=-1)
        if mode == "Texture pressure":
            g = _normalize(self.texture_pressure, 99.0)
            return np.stack([0.85 * g, 0.1 * g, g], axis=-1)
        if mode == "Recursive residual memory":
            g = self.residual_memory
            return np.stack([g, g * g, 0.12 * (1.0 - g)], axis=-1)
        if mode == "Composition discrepancy":
            g = self.composition_discrepancy
            return np.stack([g, 0.2 * g, 1.0 - g], axis=-1)
        if mode == "Cartoon-cell reconstruction":
            return np.clip(
                lab_to_srgb(self.cartoon_reconstruction), 0.0, 1.0)
        if mode == "Texture-cell correction":
            scale = max(float(np.percentile(
                np.abs(self.texture_reconstruction), 99.0)), 1e-6)
            return np.clip(
                0.5 + self.texture_reconstruction / (2.0 * scale),
                0.0, 1.0)
        if mode == "Cells":
            ids = self.owner.reshape(self.h, self.w).astype(np.uint32)
            r = ((ids * 1664525 + 1013904223) & 255) / 255.0
            g = ((ids * 22695477 + 1) & 255) / 255.0
            b = ((ids * 1103515245 + 12345) & 255) / 255.0
            return np.stack([r, g, b], axis=-1)
        if mode in ("Marked cells", "Hierarchy cells"):
            ids = self.owner.reshape(self.h, self.w)
            marks = self.marks[ids]
            base = np.empty((self.h, self.w, 3), dtype=np.float64)
            territory_tint = (
                ((self.territories.astype(np.uint32) * 2654435761) & 255)
                / 255.0)
            base[..., 0] = np.where(
                marks == CARTOON, 0.95, 0.65 + 0.3 * territory_tint)
            base[..., 1] = np.where(
                marks == CARTOON, 0.58 + 0.25 * territory_tint, 0.08)
            base[..., 2] = np.where(
                marks == CARTOON, 0.08, 0.85)
            cell_edge = ndi.morphological_gradient(ids, size=(3, 3)) != 0
            base[cell_edge] *= 0.25
            return np.clip(base, 0.0, 1.0)
        out = np.clip(lab_to_srgb(self.reconstruction), 0.0, 1.0)
        if mode == "Reconstruction + sites":
            out = out.copy()
            angles, ratios = self._site_frames()
            for (x, y), a, r, mark in zip(
                    self.seeds, angles, ratios, self.marks):
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < self.w and 0 <= iy < self.h:
                    center = ((1.0, 0.72, 0.0) if mark == CARTOON
                              else (1.0, 0.0, 0.75))
                    out[max(0, iy - 1):min(self.h, iy + 2),
                        max(0, ix - 1):min(self.w, ix + 2)] = center
                    if mark == CARTOON:
                        continue
                    length = min(8.0, 2.0 + r)
                    for t in np.linspace(-length, length, 15):
                        px = int(round(x + math.cos(a) * t))
                        py = int(round(y + math.sin(a) * t))
                        if 0 <= px < self.w and 0 <= py < self.h:
                            out[py, px] = (0, 1, 1)
        return out
