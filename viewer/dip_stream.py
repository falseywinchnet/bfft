"""Streaming DIP/PGHI reconstruction for the IQ waterfall.

Reconstruction runs on overlapping 8192-sample segments on a background thread
pool (ctypes releases the GIL during the C solve, so segments reconstruct in
parallel). Each segment is phase-aligned to its own samples and blended with a
Hann window (50% overlap = COLA), which removes the block-boundary striping
that independent, non-overlapping tiles produce. The UI thread asks for a
region each frame; the pool fills it in "under a little delay".

Complex IQ uses the native complex solver (`dip_run_complex`): full-spectrum
PGHI, no mirror ambiguity. The magnitude waterfall is invariant to the residual
global phase, but we still anchor each segment's phase to its samples so
adjacent segments overlap-add coherently.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import iqwaterfall as iqw

TILE = 8192          # matches the fixed DIP geometry (L = 8192)
STEP = TILE // 2     # 50% overlap -> Hann OLA is constant (COLA)

FRAME_HOP = 128      # long-aperture hop (HB): native time resolution of readout


class FctStream:
    """Asynchronous complex-IQ FCT rows on a fixed hop grid.

    Cached rows retain both energy-normalized correlation dB and tau/N.  The
    UI performs endpoint placement because tau is frequency-dependent; treating
    the row origin as an STFT center is the timing error this class avoids.
    """

    def __init__(self, src: iqw.IQSource, n_fft=1024, hop=256, workers=3,
                 rows_per_tile=24, prefetch=1, cache_tiles=96,
                 remove_dc=False):
        self.src = src
        self.n_fft = int(n_fft)
        self.hop = max(1, int(hop))
        self.rows_per_tile = int(rows_per_tile)
        self.prefetch = int(prefetch)
        self.cache_tiles = int(cache_tiles)
        self.remove_dc = bool(remove_dc)
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
            engine = iqw.FctWaterfall(self.n_fft)
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


class SuperResStream:
    """Streaming super-resolution waterfall via the active-delta endpoint readout.

    Each 8192-sample tile is solved independently (record-OLA PGHI seed + fixed
    L5 step) and the long-aperture endpoint magnitude is read DIRECTLY off the
    refined paused-DIP state -- |suffix_from_level(tap3(Zb))| -- never a record.
    The short-window constraints attach inside each long state through M_delta,
    so the long readout carries fine-time detail. Rows are magnitudes at
    FRAME_HOP (128) / 1024 bins; because they're per-frame magnitudes (phase-
    invariant) tiles are fully independent -> parallel, no OLA, no seams.
    """

    def __init__(self, src: iqw.IQSource, workers=4, prefetch=2, cache_tiles=96,
                 n_steps=1, alpha=1.0, mode=3, direct_seed=True,
                 nb=1024, ns=128):
        self.src = src
        self.nb = int(nb)                         # long aperture window (=cols)
        self.ns = int(ns)                         # short aperture window
        self.frame_hop = self.nb // 8             # HB = NB/8 (native time res)
        self.Ib = iqw.frames_long(TILE, self.nb)  # frames per tile
        self.stride = self.Ib * self.frame_hop    # contiguous-frame tile stride
        self.prefetch = prefetch
        self.cache_tiles = cache_tiles
        self.n_steps = n_steps
        self.alpha = alpha                        # projection step (0.5..1.0)
        self.mode = mode                          # 3 = long+short
        self.direct = direct_seed                 # direct tap-adjoint seed
        self.ntiles = max(1, src.num_samples // self.stride + 1)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache: dict[int, np.ndarray | str] = {}   # tile -> [Ib,1024] | "pending"
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._closed = False

    def _solve_tile(self, k):
        with self._io_lock:
            z = self.src.read(k * self.stride, TILE)
        m = iqw.dip_waterfall_proj(z, mode=self.mode, alpha=self.alpha,
                                   n_steps=self.n_steps, direct_seed=self.direct,
                                   nb=self.nb, ns=self.ns)    # [Ib, nb] magnitude
        return m.astype(np.float32)

    def _store(self, k, fut):
        try:
            r = fut.result()
        except Exception:  # noqa: BLE001
            r = None
        with self._lock:
            if not self._closed:
                self._cache[k] = r

    def request(self, frame_lo, n_frames):
        k0 = max(0, int(frame_lo) // self.Ib)
        k1 = min(self.ntiles - 1, (int(frame_lo) + int(n_frames) - 1) // self.Ib)
        submit = []
        with self._lock:
            for k in range(k0, k1 + 1 + self.prefetch):
                if 0 <= k < self.ntiles and k not in self._cache:
                    self._cache[k] = "pending"
                    submit.append(k)
            if len(self._cache) > self.cache_tiles:
                lo, hi = k0 - self.prefetch, k1 + self.prefetch
                for k in list(self._cache):
                    if (k < lo or k > hi) and isinstance(self._cache[k], np.ndarray):
                        del self._cache[k]
        for k in submit:
            fut = self._pool.submit(self._solve_tile, k)
            fut.add_done_callback(lambda f, kk=k: self._store(kk, f))

    def assemble(self, frame_lo, n_frames):
        """Return ((n_frames, nb) float32 magnitudes, coverage). Gaps are 0."""
        frame_lo = int(frame_lo); n_frames = int(n_frames)
        out = np.zeros((n_frames, self.nb), dtype=np.float32)
        have = np.zeros(n_frames, dtype=bool)
        k0 = frame_lo // self.Ib
        k1 = (frame_lo + n_frames - 1) // self.Ib
        with self._lock:
            ready = {k: self._cache.get(k) for k in range(k0, k1 + 1)}
        for k, data in ready.items():
            if not isinstance(data, np.ndarray):
                continue
            g0 = k * self.Ib               # global frame of this tile's row 0
            lo = max(g0, frame_lo)
            hi = min(g0 + self.Ib, frame_lo + n_frames)
            if lo < hi:
                out[lo - frame_lo:hi - frame_lo] = data[lo - g0:hi - g0]
                have[lo - frame_lo:hi - frame_lo] = True
        return out, float(np.mean(have))

    def close(self):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)


class FusionStream:
    """F1xT2 fusion super-resolution via calibrated gating.

    The active-delta inverse constrains the OWNED band of deltas (owned = HB/HS,
    centered on the long frame). The readout emits ONLY those constrained deltas
    -- every fine-time row is a measured, fitted short, not a model-implied one.
    Short-endpoint power partitions each long-bin power across the owned fine
    rows.  Thus sum_e |F[e,k]|^2 = |L[k]|^2 (apart from the optional uniform
    floor blend): no independent peak normalization and no fabricated energy.
    The coarse short weights are circularly interpolated onto the long grid so
    coarse bins are centered rather than left-edge repeated. Rows are at HS=32
    hop / nb bins."""

    def __init__(self, src, nb=4096, ns=512, workers=4, prefetch=1,
                 cache_tiles=48, n_steps=1, alpha=1.0, mode=3, direct_seed=True,
                 gate_floor=0.05, beta=1.0):
        self.src = src
        self.nb = int(nb); self.ns = int(ns)
        self.HB = self.nb // 8
        self.HS = 32                                  # HS = q* (fixed)
        self.Ib = iqw.frames_long(TILE, self.nb)
        self.owned = self.HB // self.HS               # fine rows per long frame
        self.lo = (self.nb // 2 - self.HB // 2) // self.HS   # first owned delta
        # Constrain AND read exactly the owned band (contiguous, measured gates).
        self.dsel = np.arange(self.lo, self.lo + self.owned, dtype=np.int32)
        self.parent = self.nb // self.ns              # exact grid ratio
        self.gate_floor = gate_floor; self.beta = beta
        self.frame_hop = self.HS                      # native time res (samples)
        self.rows_per_tile = self.Ib * self.owned
        self.stride = self.Ib * self.HB
        self.n_steps = n_steps; self.alpha = alpha
        self.mode = mode; self.direct = direct_seed
        self.prefetch = prefetch; self.cache_tiles = cache_tiles
        self.ntiles = max(1, src.num_samples // self.stride + 1)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache: dict[int, np.ndarray | str] = {}
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._closed = False

    def _solve_tile(self, k):
        with self._io_lock:
            z = self.src.read(k * self.stride, TILE)
        long_mag, short_mag = iqw.dip_fusion(       # short_mag [Ib, owned, ns]
            z, self.dsel, mode=self.mode, alpha=self.alpha, n_steps=self.n_steps,
            direct_seed=self.direct, nb=self.nb, ns=self.ns)
        out = np.empty((self.rows_per_tile, self.nb), dtype=np.float32)
        coord = np.arange(self.nb, dtype=np.float64) / self.parent
        q0 = np.floor(coord).astype(np.intp) % self.ns
        frac = coord - np.floor(coord)
        q1 = (q0 + 1) % self.ns
        for j in range(self.Ib):
            L = long_mag[j]                          # [nb] fine freq (true scale)
            S = short_mag[j]                         # [owned, ns] fine time
            P = S * S
            weights = P / (P.sum(axis=0, keepdims=True) + 1e-24)
            # gate_floor is now an honest uniform-power mixture, not a lower
            # clamp that duplicates long-bin energy into every fine row.
            weights = ((1.0 - self.gate_floor) * weights +
                       self.gate_floor / self.owned)
            for e in range(self.owned):
                w = weights[e]
                w_up = w[q0] * (1.0 - frac) + w[q1] * frac
                # beta=1 conserves power.  Other values are explicitly a
                # contrast rendering; renormalize so they still conserve it.
                if self.beta != 1.0:
                    w_up = np.power(w_up, self.beta)
                out[j * self.owned + e] = L * np.sqrt(w_up)
        if self.beta != 1.0:
            shaped = out.reshape(self.Ib, self.owned, self.nb)
            den = np.sqrt(np.sum(shaped * shaped, axis=1, keepdims=True)) + 1e-24
            shaped *= long_mag[:, None, :] / den
        return out

    def _store(self, k, fut):
        try:
            r = fut.result()
        except Exception:  # noqa: BLE001
            r = None
        with self._lock:
            if not self._closed:
                self._cache[k] = r

    def request(self, frame_lo, n_frames):
        rpt = self.rows_per_tile
        k0 = max(0, int(frame_lo) // rpt)
        k1 = min(self.ntiles - 1, (int(frame_lo) + int(n_frames) - 1) // rpt)
        submit = []
        with self._lock:
            for k in range(k0, k1 + 1 + self.prefetch):
                if 0 <= k < self.ntiles and k not in self._cache:
                    self._cache[k] = "pending"
                    submit.append(k)
            if len(self._cache) > self.cache_tiles:
                lo, hi = k0 - self.prefetch, k1 + self.prefetch
                for k in list(self._cache):
                    if (k < lo or k > hi) and isinstance(self._cache[k], np.ndarray):
                        del self._cache[k]
        for k in submit:
            fut = self._pool.submit(self._solve_tile, k)
            fut.add_done_callback(lambda f, kk=k: self._store(kk, f))

    def assemble(self, frame_lo, n_frames):
        frame_lo = int(frame_lo); n_frames = int(n_frames)
        rpt = self.rows_per_tile
        out = np.zeros((n_frames, self.nb), dtype=np.float32)
        have = np.zeros(n_frames, dtype=bool)
        k0 = frame_lo // rpt
        k1 = (frame_lo + n_frames - 1) // rpt
        with self._lock:
            ready = {k: self._cache.get(k) for k in range(k0, k1 + 1)}
        for k, data in ready.items():
            if not isinstance(data, np.ndarray):
                continue
            g0 = k * rpt
            lo = max(g0, frame_lo)
            hi = min(g0 + rpt, frame_lo + n_frames)
            if lo < hi:
                out[lo - frame_lo:hi - frame_lo] = data[lo - g0:hi - g0]
                have[lo - frame_lo:hi - frame_lo] = True
        return out, float(np.mean(have))

    def close(self):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)


class StreamReconstructor:
    def __init__(self, src: iqw.IQSource, workers=3, prefetch=4,
                 cache_segs=320, steepest_scale=2.5e-4, method="complex"):
        self.src = src
        self.prefetch = prefetch
        self.cache_segs = cache_segs
        self.steepest = steepest_scale
        # "complex": native complex reconstruction (default; mirror-free).
        # "quadrature": per-I/Q fallback (has per-frame mirror leakage).
        self.method = method
        self.nsegs = max(1, (src.num_samples) // STEP + 1)
        self._win = np.hanning(TILE).astype(np.float64)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache: dict[int, np.ndarray | str] = {}   # seg -> complex64 | "pending"
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._closed = False

    # -- reconstruction strategy (swap DIP variant here only) ---------------
    def _solve(self, z):
        if self.method == "complex" or self.src.is_complex:
            u, _ = iqw.dip_run_complex(np.ascontiguousarray(z),
                                       steepest_scale=self.steepest)
        else:
            uI, _ = iqw.dip_run(np.ascontiguousarray(z.real),
                                steepest_scale=self.steepest)
            u = uI.astype(np.complex128)
        # Anchor global phase to the known samples so overlapping segments
        # overlap-add coherently (kills block-boundary striping).
        th = np.angle(np.vdot(u, z))
        return u * np.exp(-1j * th)

    def _recon_seg(self, seg):
        start = seg * STEP
        with self._io_lock:
            z = self.src.read(start, TILE)
        return self._solve(z).astype(np.complex64)

    def _store(self, seg, fut):
        try:
            r = fut.result()
        except Exception:  # noqa: BLE001
            r = None
        with self._lock:
            if not self._closed:
                self._cache[seg] = r

    # -- UI-side API ---------------------------------------------------------
    def _seg_range(self, start, count):
        end = start + count
        lo = max(0, (start - TILE) // STEP + 1)
        hi = min(self.nsegs - 1, (end - 1) // STEP)
        return lo, hi

    def request(self, start, count):
        lo, hi = self._seg_range(int(start), int(count))
        submit = []
        with self._lock:
            for seg in range(lo, hi + 1 + self.prefetch):
                if 0 <= seg < self.nsegs and seg not in self._cache:
                    self._cache[seg] = "pending"
                    submit.append(seg)
            if len(self._cache) > self.cache_segs:
                keep_lo, keep_hi = lo - self.prefetch, hi + self.prefetch
                for seg in list(self._cache):
                    if (seg < keep_lo or seg > keep_hi) and \
                            isinstance(self._cache[seg], np.ndarray):
                        del self._cache[seg]
        for seg in submit:
            fut = self._pool.submit(self._recon_seg, seg)
            fut.add_done_callback(lambda f, s=seg: self._store(s, f))

    def assemble(self, start, count, raw_fill=True):
        """Return (iq_float32[2*count], coverage_fraction) for [start,start+count).

        Reconstructed segments are Hann-OLA blended; regions without coverage are
        filled with the raw signal (so the view is never blank while tiles load).
        """
        start = int(start); count = int(count)
        lo, hi = self._seg_range(start, count)
        acc = np.zeros(count, dtype=np.complex128)
        den = np.zeros(count, dtype=np.float64)
        with self._lock:
            ready = {s: self._cache.get(s) for s in range(lo, hi + 1)}
        w = self._win
        for seg, data in ready.items():
            if not isinstance(data, np.ndarray):
                continue
            s0 = seg * STEP
            a = max(s0, start); b = min(s0 + TILE, start + count)
            if a < b:
                acc[a - start:b - start] += data[a - s0:b - s0] * w[a - s0:b - s0]
                den[a - start:b - start] += w[a - s0:b - s0]
        covered = den > 1e-6
        out = np.zeros(count, dtype=np.complex128)
        out[covered] = acc[covered] / den[covered]
        cov = float(np.mean(covered))
        if raw_fill and cov < 0.999:
            with self._io_lock:
                raw = self.src.read(start, count)
            out[~covered] = raw[~covered]
        buf = np.empty(2 * count, dtype=np.float32)
        buf[0::2] = out.real
        buf[1::2] = out.imag
        return buf, cov

    def close(self):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
