"""ARCHIVED 2026-07-10: the intrinsic-FCT viewer mode.

The exact intrinsic phase-disk FCT remains a shipped library feature
(``src/fct.cpp``, ``src/detail/fct_intrinsic_kernel.hpp``, the ``iqw_fct_*``
C ABI, and ``iqwaterfall.FctWaterfall``) with its full test suite.  What is
archived here is only the *viewer mode* built on it: the asynchronous tile
stream and the endpoint-scatter rasterizer.  It was set aside because the
endpoint-timed image is not the display product we are pursuing; the viewer
now focuses on the streaming STFT and the two-aperture DIP-Zak
super-resolution.

To revive: import ``FctStream`` and ``scatter_endpoints`` from this module,
add a view mode that calls them the way ``iq_waterfall_app.py`` did (see the
git history of that file at the tag/commit that removed the mode), and tint
cells by ``support`` if the adaptive-aperture overlay is wanted.

Key semantics worth preserving on revival:

- An FCT frame owns one *origin* and a per-bin endpoint ``origin + tau[k]``;
  treating the row origin as an STFT center time is a category error.
- Cached origins live on the global hop lattice; place endpoints in sample
  coordinates before rasterizing so scrub positions between lattice points do
  not shift every cell by ``position % hop``.
- Displayed amplitude is ``|C|/sqrt(tau)`` (the optimized score itself);
  ``tau/N`` is a second measurement and is rendered as a tint, not intensity.
- The viewer declared ``t_min = N/32`` (one display hop): shorter IQ prefixes
  cannot carry frequency identity, and the walk stays exact over the declared
  domain.
"""
from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import iqwaterfall as iqw  # noqa: E402


class FctStream:
    """Asynchronous complex-IQ FCT rows on a fixed hop grid.

    Cached rows retain both normalized-correlation dB and tau/N.  The
    UI performs endpoint placement because tau is frequency-dependent; treating
    the row origin as an STFT center is the timing error this class avoids.
    """

    def __init__(self, src: iqw.IQSource, n_fft=1024, hop=256, workers=3,
                 rows_per_tile=24, prefetch=1, cache_tiles=96,
                 remove_dc=False, min_support=1, activity=0.0):
        self.src = src
        self.n_fft = int(n_fft)
        self.hop = max(1, int(hop))
        self.rows_per_tile = int(rows_per_tile)
        self.prefetch = int(prefetch)
        self.cache_tiles = int(cache_tiles)
        self.remove_dc = bool(remove_dc)
        self.min_support = int(min_support)
        self.activity = float(activity)
        self.ntiles = max(1, (src.num_samples + self.hop - 1) //
                          (self.rows_per_tile * self.hop) + 1)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray] | str | None] = {}
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._local = threading.local()
        self._closed = False

    def _engine(self):
        engine = getattr(self._local, "engine", None)
        if engine is None:
            engine = iqw.FctWaterfall(
                self.n_fft, self.min_support, self.activity)
            self._local.engine = engine
        return engine

    def _solve_tile(self, k):
        start = k * self.rows_per_tile * self.hop
        count = (self.rows_per_tile - 1) * self.hop + self.n_fft
        with self._io_lock:
            z = self.src.read(start, count)
        iq = np.empty(2 * count, dtype=np.float32)
        iq[0::2] = z.real
        iq[1::2] = z.imag
        return self._engine().render_mem(
            iq, 0, self.hop, self.rows_per_tile, remove_dc=self.remove_dc)

    def _store(self, k, fut):
        try:
            result = fut.result()
        except Exception:  # noqa: BLE001
            result = None
        with self._lock:
            if not self._closed:
                self._cache[k] = result

    def request(self, frame_lo, n_frames):
        rpt = self.rows_per_tile
        k0 = int(np.floor(int(frame_lo) / rpt))
        k1 = int(np.floor((int(frame_lo) + int(n_frames) - 1) / rpt))
        submit = []
        with self._lock:
            for k in range(k0, k1 + 1 + self.prefetch):
                if k < self.ntiles and k not in self._cache:
                    self._cache[k] = "pending"
                    submit.append(k)
            if len(self._cache) > self.cache_tiles:
                for k in list(self._cache):
                    if (k < k0 - self.prefetch or k > k1 + self.prefetch) and \
                            isinstance(self._cache[k], tuple):
                        del self._cache[k]
        for k in submit:
            fut = self._pool.submit(self._solve_tile, k)
            fut.add_done_callback(lambda f, kk=k: self._store(kk, f))

    def assemble(self, frame_lo, n_frames):
        frame_lo = int(frame_lo)
        n_frames = int(n_frames)
        rpt = self.rows_per_tile
        db = np.full((n_frames, self.n_fft), -240.0, dtype=np.float32)
        support = np.ones((n_frames, self.n_fft), dtype=np.float32)
        have = np.zeros(n_frames, dtype=bool)
        k0 = int(np.floor(frame_lo / rpt))
        k1 = int(np.floor((frame_lo + n_frames - 1) / rpt))
        with self._lock:
            ready = {k: self._cache.get(k) for k in range(k0, k1 + 1)}
        for k, data in ready.items():
            if not isinstance(data, tuple):
                continue
            g0 = k * rpt
            lo = max(g0, frame_lo)
            hi = min(g0 + rpt, frame_lo + n_frames)
            if lo < hi:
                dst = slice(lo - frame_lo, hi - frame_lo)
                src = slice(lo - g0, hi - g0)
                db[dst] = data[0][src]
                support[dst] = data[1][src]
                have[dst] = True
        return db, support, float(np.mean(have))

    def close(self):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)


def scatter_endpoints(db, support, raw_lo, hop, n_fft, position, n_rows):
    """Rasterize cached FCT rows at each bin's selected endpoint.

    ``db``/``support`` are ``(n_raw, n_fft)`` arrays whose row ``a`` has origin
    sample ``(raw_lo + a) * hop``.  Returns ``(block, chosen_support)`` shaped
    ``(n_rows, n_fft)`` where each cell keeps the strongest claimant whose
    endpoint ``origin + tau`` lands in that display row.
    """
    n_raw = db.shape[0]
    block = np.full((n_rows, n_fft), -240.0, dtype=np.float32)
    chosen_support = np.ones((n_rows, n_fft), dtype=np.float32)
    cols_all = np.arange(n_fft)
    for a in range(n_raw):
        origin_sample = (raw_lo + a) * hop
        endpoint_sample = origin_sample + support[a] * n_fft
        target = np.rint((endpoint_sample - position) / hop).astype(np.intp)
        valid = (target >= 0) & (target < n_rows)
        for dest in np.unique(target[valid]):
            cols = cols_all[valid & (target == dest)]
            better = db[a, cols] > block[dest, cols]
            take = cols[better]
            block[dest, take] = db[a, take]
            chosen_support[dest, take] = support[a, take]
    return block, chosen_support
