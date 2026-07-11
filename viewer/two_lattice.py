"""Faithful two-lattice magnitude fusion from the original Python demo.

This module intentionally stays in NumPy.  It is the executable specification
for the viewer mode, not the eventual fast kernel.  The long and short STFT
families are independent measurements of one latent complex waveform.  They
are coupled only by alternating waveform-domain overlap/add projections.

There are no frequency-row claims, gates, active-delta parent assignments, or
image-space multiplication in this implementation.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def frame_offsets(length: int, n_fft: int, hop: int) -> np.ndarray:
    """Starts of every complete frame on one independent lattice."""
    return np.arange(0, length - n_fft + 1, hop, dtype=np.int64)


class MagnitudeFamily:
    """One symmetric-Hann STFT magnitude constraint family."""

    def __init__(self, observed: np.ndarray, n_fft: int, hop: int):
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.offsets = frame_offsets(len(observed), self.n_fft, self.hop)
        if not len(self.offsets):
            raise ValueError("segment is shorter than an analysis aperture")
        self.window = np.hanning(self.n_fft).astype(np.float64)
        self.indices = (self.offsets[:, None] +
                        np.arange(self.n_fft, dtype=np.int64)[None, :])
        frames = observed[self.indices] * self.window[None, :]
        self.target = np.abs(np.fft.fft(frames, axis=1))
        self.den = np.zeros(len(observed), dtype=np.float64)
        np.add.at(self.den, self.indices.ravel(),
                  np.broadcast_to(self.window ** 2, self.indices.shape).ravel())
        positive = self.den[self.den > 0.0]
        median = float(np.median(positive)) if len(positive) else 1.0
        self.support = self.den > 1e-2 * median

    def project(self, latent: np.ndarray, fallback: np.ndarray, *,
                real_signal: bool = False) -> np.ndarray:
        """Nearest endpoint-magnitude frames followed by Hann overlap/add."""
        frames = latent[self.indices] * self.window[None, :]
        spectrum = np.fft.fft(frames, axis=1)
        spectrum *= self.target / (np.abs(spectrum) + 1e-12)
        fitted = np.fft.ifft(spectrum, axis=1)
        if real_signal:
            fitted = fitted.real

        acc = np.zeros_like(latent, dtype=np.complex128)
        np.add.at(acc, self.indices.ravel(),
                  (fitted * self.window[None, :]).ravel())
        out = acc / np.maximum(self.den, 1e-8)
        # Symmetric Hann endpoints have no support.  Carry the previous iterate
        # there, exactly as the standalone solver carries its seed/fallback.
        return np.where(self.support, out, fallback)

    def relative_error(self, latent: np.ndarray) -> float:
        frames = latent[self.indices] * self.window[None, :]
        got = np.abs(np.fft.fft(frames, axis=1))
        num = np.sum((got - self.target) ** 2)
        den = np.sum(self.target ** 2) + 1e-30
        return float(np.sqrt(num / den))


def recover_two_lattice(observed: np.ndarray, n_long: int, n_short: int,
                        *, h_long: int | None = None, h_short: int = 32,
                        iterations: int = 24, beta: float = 0.88,
                        seed: int = 0, real_signal: bool = False,
                        initial: np.ndarray | None = None,
                        finish_mode: str = "legacy"
                        ) -> tuple[np.ndarray, dict[str, float]]:
    """Recover one complex latent waveform from two STFT magnitude families.

    This is the complex-valued translation of ``gl_align`` in
    ``zak_superresolution_wav_demo.py``.  The observation phase is not used by
    the iterations.  It is consulted once after convergence to choose the
    otherwise-free global phase, which makes neighboring reconstructed tiles
    overlap-add coherently in an IQ recording.
    """
    z = np.ascontiguousarray(observed, dtype=np.complex128)
    n_long = int(n_long)
    n_short = int(n_short)
    h_long = n_long // 8 if h_long is None else int(h_long)
    h_short = int(h_short)
    if n_long <= 0 or n_short <= 0 or n_short > n_long:
        raise ValueError("require 0 < short aperture <= long aperture")
    if h_long <= 0 or h_short <= 0:
        raise ValueError("analysis hops must be positive")

    long_family = MagnitudeFamily(z, n_long, h_long)
    short_family = MagnitudeFamily(z, n_short, h_short)

    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-12)
    if initial is not None:
        seed_value = np.asarray(initial).real if real_signal else initial
        latent = np.ascontiguousarray(
            seed_value, dtype=np.float64 if real_signal
            else np.complex128).copy()
        if latent.shape != z.shape:
            raise ValueError("initial latent waveform has the wrong shape")
    elif real_signal:
        # Same seed law as the original standalone Python solver.
        latent = 0.01 * rng.standard_normal(len(z))
    else:
        latent = (rng.standard_normal(len(z)) +
                  1j * rng.standard_normal(len(z))) * (0.01 * rms / np.sqrt(2.0))
    previous = latent.copy()

    for _ in range(max(0, int(iterations))):
        trial = latent + beta * (latent - previous)
        if (not np.all(np.isfinite(trial)) or
                np.max(np.abs(trial)) > 50.0 * max(rms, 1e-12)):
            trial = latent.copy()
        previous = latent
        if finish_mode == "palindromic":
            fitted = long_family.project(trial, trial,
                                         real_signal=real_signal)
            trial = trial + 0.5 * (fitted - trial)
            trial = short_family.project(trial, trial,
                                         real_signal=real_signal)
            fitted = long_family.project(trial, trial,
                                         real_signal=real_signal)
            latent = trial + 0.5 * (fitted - trial)
        else:
            fitted_long = long_family.project(trial, trial,
                                              real_signal=real_signal)
            latent = short_family.project(fitted_long, fitted_long,
                                          real_signal=real_signal)

    # Select only the global phase.  Magnitudes and the recovered structure are
    # unchanged.  vdot(latent, z) is the multiplier that maps latent toward z.
    cross = np.vdot(latent, z)
    if real_signal:
        if np.real(cross) < 0.0:
            latent *= -1.0
        latent = latent.real
    elif abs(cross) > 1e-30:
        latent *= cross / abs(cross)

    metrics = {
        "long_relative_error": long_family.relative_error(latent),
        "short_relative_error": short_family.relative_error(latent),
    }
    return latent, metrics


def recover_ladder(observed: np.ndarray, rungs, *, iterations: int = 1,
                   beta: float = 0.88, final_relax: float = 0.75,
                   initial: np.ndarray | None = None,
                   finish_mode: str = "legacy"
                   ) -> tuple[np.ndarray, dict[str, float]]:
    """Ladder generalization of the two-family loop (executable spec for
    ``iqw_dip_unified_ladder``).  ``rungs`` is a sequence of (n_fft, hop) in
    application order (largest first by convention); the final relaxed
    projection is applied on rungs[0].  Coverage rationale is recorded in
    notes/two_lattice_superresolution.md S5: an upward rung pins slow packet
    pairs the deployed pair cannot see, at nearly no cost.
    """
    z = np.ascontiguousarray(observed, dtype=np.complex128)
    families = [MagnitudeFamily(z, int(n), int(h)) for n, h in rungs]
    rms = float(np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-12)
    if initial is None:
        latent = z.copy()
    else:
        latent = np.ascontiguousarray(initial, dtype=np.complex128).copy()
    previous = latent.copy()
    for _ in range(max(0, int(iterations))):
        trial = latent + beta * (latent - previous)
        if (not np.all(np.isfinite(trial)) or
                np.max(np.abs(trial)) > 50.0 * max(rms, 1e-12)):
            trial = latent.copy()
        previous = latent
        if finish_mode == "palindromic" and families:
            for fam in families[:-1]:
                corrected = fam.project(trial, trial)
                trial += 0.5 * (corrected - trial)
            trial = families[-1].project(trial, trial)
            for fam in reversed(families[:-1]):
                corrected = fam.project(trial, trial)
                trial += 0.5 * (corrected - trial)
        else:
            for fam in families:
                trial = fam.project(trial, trial)
        latent = trial
    final_relax = min(1.0, max(0.0, float(final_relax)))
    if (finish_mode == "legacy" and iterations > 0 and
            final_relax > 0.0 and families):
        corrected = families[0].project(latent, latent)
        latent = latent + final_relax * (corrected - latent)
    cross = np.vdot(latent, z)
    if abs(cross) > 1e-30:
        latent = latent * (cross / abs(cross))
    metrics = {f"rung{n}_relative_error": fam.relative_error(latent)
               for (n, _h), fam in zip(rungs, families)}
    return latent, metrics


class TwoLatticeStream:
    """Asynchronous, overlap-added NumPy reference solver for an IQ source."""

    def __init__(self, src, n_long=1024, n_short=128, *, h_short=32,
                 iterations=24, workers=2, prefetch=1, cache_tiles=48,
                 rung_mult=0, finish_mode="legacy"):
        self.src = src
        self.n_long = int(n_long)
        self.n_short = int(n_short)
        self.h_long = self.n_long // 8
        self.h_short = int(h_short)
        self.iterations = int(iterations)
        self.finish_mode = str(finish_mode)
        self.tile = max(8192, 2 * self.n_long)
        # Optional upward rung (coverage law, notes S5): rung_mult in {0,2,4}
        # adds a rung_mult*n_long window family, capped at the tile length.
        # Upward rungs pin slow packet pairs (dt > n_long) the deployed pair
        # is blind to, at near-zero cost (few frames per tile).
        self.rung_mult = int(rung_mult)
        self.rungs = None
        if self.rung_mult > 1:
            up = min(self.rung_mult * self.n_long, self.tile)
            self.rungs = [(up, max(1, up // 8)),
                          (self.n_long, self.h_long),
                          (self.n_short, self.h_short)]
        self.step = self.tile // 2
        self.ntiles = max(1, (src.num_samples + self.step - 1) // self.step)
        self.prefetch = int(prefetch)
        self.cache_tiles = int(cache_tiles)
        self._ola_window = np.hanning(self.tile).astype(np.float64)
        # Unified PGHI phases are chained tile-to-tile. A serial executor keeps
        # ascending submissions deterministic; one tile is ~20 ms, so phase
        # continuity is worth more than speculative parallel cold seeds.
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._cache: dict[int, tuple[np.ndarray, dict[str, float]] | str | None] = {}
        self._warm_internal: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._closed = False

    def _solve_tile(self, tile_index: int):
        start = tile_index * self.step
        with self._io_lock:
            observed = self.src.read(start, self.tile)
        # One backend call now performs the validated shared fast1 seed and the
        # literal independent-family projection implemented above.  Keeping
        # the NumPy functions in this file makes the C++ semantics testable.
        import iqwaterfall as iqw
        warm = None
        previous = self._warm_internal.get(tile_index - 1)
        if previous is not None and self.step % self.h_long == 0:
            shift = self.step // self.h_long
            if shift < previous.shape[1]:
                warm = np.ascontiguousarray(previous[:, shift:])
        if self.rungs is not None:
            unified, internal, _loss0 = iqw.dip_unified_ladder(
                observed, self.rungs, steepest_scale=2.5e-4, shared_steps=1,
                nb=self.n_long, ns=self.n_short,
                unified_steps=self.iterations, beta=0.88,
                finish_mode=self.finish_mode,
                warm_internal=warm, return_internal=True)
        else:
            unified, internal, _loss0 = iqw.dip_unified(
                observed, steepest_scale=2.5e-4, shared_steps=1,
                nb=self.n_long, ns=self.n_short, h_short=self.h_short,
                unified_steps=self.iterations, beta=0.88,
                finish_mode=self.finish_mode,
                warm_internal=warm, return_internal=True)
        self._warm_internal[tile_index] = internal
        if not self.src.is_complex:
            unified = unified.real
        return unified, {}

    def _store(self, tile_index, future):
        try:
            result = future.result()
        except Exception:  # noqa: BLE001
            result = None
        with self._lock:
            if not self._closed:
                self._cache[tile_index] = result

    def _tile_range(self, start: int, count: int) -> tuple[int, int]:
        end = start + count
        lo = max(0, (start - self.tile) // self.step + 1)
        hi = min(self.ntiles - 1, (end - 1) // self.step)
        return lo, hi

    def request(self, start: int, count: int) -> None:
        lo, hi = self._tile_range(int(start), int(count))
        submit: list[int] = []
        futures = []
        # Selection and submission are one lifecycle transaction. Previously
        # a UI aperture callback could close the executor after the cache was
        # marked pending but before submit(), raising "cannot schedule new
        # futures after shutdown" on the next SR frame.
        with self._lifecycle_lock:
            if self._closed:
                return
            with self._lock:
                for k in range(lo, min(self.ntiles, hi + 1 + self.prefetch)):
                    if k not in self._cache:
                        self._cache[k] = "pending"
                        submit.append(k)
                if len(self._cache) > self.cache_tiles:
                    for k in list(self._cache):
                        if ((k < lo - self.prefetch or
                             k > hi + self.prefetch) and
                                isinstance(self._cache[k], tuple)):
                            del self._cache[k]
                            self._warm_internal.pop(k, None)
            for k in submit:
                futures.append((k, self._pool.submit(self._solve_tile, k)))
        for k, future in futures:
            future.add_done_callback(lambda f, kk=k: self._store(kk, f))

    def assemble(self, start: int, count: int, *, raw_fill=True):
        start = int(start)
        count = int(count)
        lo, hi = self._tile_range(start, count)
        acc = np.zeros(count, dtype=np.complex128)
        den = np.zeros(count, dtype=np.float64)
        errors: list[float] = []
        with self._lock:
            ready = {k: self._cache.get(k) for k in range(lo, hi + 1)}
        ready_tiles = sum(isinstance(item, tuple) for item in ready.values())
        expected_tiles = max(1, len(ready))
        for k, item in ready.items():
            if not isinstance(item, tuple):
                continue
            data, metrics = item
            tile_start = k * self.step
            a = max(start, tile_start)
            b = min(start + count, tile_start + self.tile)
            if a >= b:
                continue
            sl_data = slice(a - tile_start, b - tile_start)
            sl_out = slice(a - start, b - start)
            w = self._ola_window[sl_data]
            acc[sl_out] += data[sl_data] * w
            den[sl_out] += w
            if metrics:
                errors.append(max(metrics.values()))
        covered = den > 1e-8
        out = np.zeros(count, dtype=np.complex128)
        out[covered] = acc[covered] / den[covered]
        if raw_fill and not np.all(covered):
            with self._io_lock:
                raw = self.src.read(start, count)
            out[~covered] = raw[~covered]
        error = float(max(errors)) if errors else float("nan")
        # Readiness is a cache property, not a Hann endpoint property.  Using
        # sample coverage here made the GUI recompute forever at file edges,
        # where a legitimate tile window is exactly zero and raw_fill applies.
        return out, ready_tiles / expected_tiles, error

    def close(self):
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
            # A stream generation owns in-flight reads from the shared
            # IQSource. Returning before those reads finish lets a newly
            # selected rung start another worker on the same FILE*.
            self._pool.shutdown(wait=True, cancel_futures=True)
