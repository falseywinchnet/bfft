"""IQ waterfall viewer.

A streaming, zoomable spectrogram/waterfall for IQ recordings, backed by the
monolithic BFFT-based `libiqwaterfall` library.

Features
--------
  * Open WAV IQ files (auto header) or raw IQ (u8/s8/s16/s24/s32/f32, real or
    complex) with a user-set sample rate.
  * Play / Pause / Stop transport with a draggable position scrubber.
  * Transform-derived COLA time sampling and frequency zoom (band select).
  * Adjustable dynamic range (floor / ceiling dB) to stretch the colormap.
  * FFT size, analysis window, and colormap settings.

Run:  python iq_waterfall_app.py
"""
from __future__ import annotations

import os
import sys
import threading
import time

# Make this app runnable from any working directory: ensure its own folder is
# on sys.path so `import iqwaterfall` (the sibling wrapper) always resolves.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import dearpygui.dearpygui as dpg

import iqwaterfall as iqw
import two_lattice

# ---------------------------------------------------------------------------
# Colormaps: small anchor tables interpolated to 256-entry RGB LUTs.
# ---------------------------------------------------------------------------
_ANCHORS = {
    "Viridis": [(0.267, 0.005, 0.329), (0.229, 0.322, 0.545),
                (0.128, 0.567, 0.551), (0.369, 0.789, 0.383),
                (0.993, 0.906, 0.144)],
    "Inferno": [(0.001, 0.000, 0.014), (0.258, 0.039, 0.406),
                (0.578, 0.148, 0.404), (0.865, 0.316, 0.226),
                (0.988, 0.998, 0.645)],
    "Magma":   [(0.001, 0.000, 0.014), (0.232, 0.059, 0.437),
                (0.550, 0.161, 0.506), (0.887, 0.352, 0.383),
                (0.987, 0.991, 0.749)],
    "Turbo":   [(0.190, 0.072, 0.232), (0.098, 0.539, 0.925),
                (0.180, 0.933, 0.463), (0.976, 0.826, 0.118),
                (0.918, 0.190, 0.038), (0.480, 0.016, 0.011)],
    "Grayscale": [(0, 0, 0), (1, 1, 1)],
    "Jet":     [(0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0),
                (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
}


def _build_lut(anchors, n=256):
    a = np.asarray(anchors, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, len(a))
    xs = np.linspace(0.0, 1.0, n)
    lut = np.empty((n, 3), dtype=np.float32)
    for c in range(3):
        lut[:, c] = np.interp(xs, xp, a[:, c])
    return lut


_LUTS = {name: _build_lut(anc) for name, anc in _ANCHORS.items()}


def _anchored_zoom(lo, hi, anchor_fraction, wheel,
                   domain_lo=-0.5, domain_hi=0.5, min_span=1e-5):
    """Zoom an interval while keeping the point under the cursor fixed."""
    q = float(np.clip(anchor_fraction, 0.0, 1.0))
    old_span = max(min_span, float(hi) - float(lo))
    # One conventional wheel notch is one octave; fractional trackpad deltas
    # remain continuous instead of being quantized into barely visible steps.
    factor = 2.0 ** (-float(wheel))
    span = float(np.clip(old_span * factor, min_span,
                         domain_hi - domain_lo))
    anchor = float(lo) + q * old_span
    new_lo = anchor - q * span
    new_hi = new_lo + span
    if new_lo < domain_lo:
        new_hi += domain_lo - new_lo
        new_lo = domain_lo
    if new_hi > domain_hi:
        new_lo -= new_hi - domain_hi
        new_hi = domain_hi
    return max(domain_lo, new_lo), min(domain_hi, new_hi)


def _rung_gain_db(n_fft, base_fft):
    """Hann coherent-power gain relative to the base aperture, in dB."""
    # For the symmetric Hann used by both NumPy and the native renderer,
    # sum(w_N) == (N - 1) / 2.  Reassigned power therefore needs 20 log10
    # of this ratio removed before max-fusing different aperture lengths.
    return 20.0 * np.log10((int(n_fft) - 1) / (int(base_fft) - 1))


def _resample_axis(img, n_out, axis=1):
    """Peak-preserving display resample.

    Shrinking uses max-pooling so a one-bin tone or a one-row click can never
    fall between output samples (the old nearest-index pick dropped bins and
    made narrow features flicker with scroll position); growing uses linear
    interpolation.  Max in dB equals max in linear power, so pooling in the
    dB domain is exact.
    """
    n_in = img.shape[axis]
    if n_out == n_in:
        return img
    if n_out < n_in:
        starts = (np.arange(n_out) * n_in) // n_out
        return np.maximum.reduceat(img, starts, axis=axis)
    xi = np.linspace(0, n_in - 1, n_out)
    i0 = np.floor(xi).astype(np.intp)
    i1 = np.minimum(i0 + 1, n_in - 1)
    f = (xi - i0).astype(np.float32)
    shape = [1, 1]
    shape[axis] = n_out
    f = f.reshape(shape)
    # Interpolate in linear power, not dB: dB-domain lerp dims a one-bin
    # tone by tens of dB between samples; power-domain lerp caps at -3 dB.
    a = np.power(10.0, np.take(img, i0, axis=axis) / 10.0)
    b = np.power(10.0, np.take(img, i1, axis=axis) / 10.0)
    return 10.0 * np.log10(a * (1.0 - f) + b * f + 1e-30)

# Display texture dimensions (frequency columns x time rows).
DW = 1024
DH = 640

FFT_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


class App:
    def __init__(self):
        self.src: iqw.IQSource | None = None
        self.wf: iqw.Waterfall | None = None
        self.ra: iqw.Reassign | None = None
        self.sr_ra: iqw.Reassign | None = None
        self.sr_extra_ras: dict[int, iqw.Reassign] = {}
        self._sr_lock = threading.RLock()

        self.n_fft = 1024
        self.window = 0
        self.colormap = "Viridis"
        self.stft_hop_div = 2
        self.hop = self.n_fft // self.stft_hop_div
        self.position = 0             # top-row complex-sample index
        self.floor_db = -90.0
        self.ceil_db = -10.0
        # "Amplitude (auto)": linear amplitude scaled to the image's 99.5th
        # percentile -- the transfer function of the STFT reference figures.
        # "dB (auto range)": percentile ceiling, span from the two sliders.
        # "dB (manual)": the classic fixed floor/ceiling window.
        self.display_scale = "Amplitude (auto)"
        self.gamma = 1.0
        self.f_lo = -0.5              # frequency band, fraction of Fs
        self.f_hi = 0.5
        self.remove_dc = False
        self.flip_time = True         # newest row at top (SDR# style)

        self.playing = False
        self.play_rate = 1.0          # x realtime
        self.last_t = time.perf_counter()
        self.dirty = True
        self._wheel_target = None
        self._wheel_pending = 0.0
        self._wheel_anchor = 0.5

        self.recon_cov = 0.0
        # Faithful Python two-lattice magnitude inversion.  Each aperture owns
        # an independent STFT lattice; their only shared variable is a latent
        # waveform. Internally the short family uses its COLA hop short/2
        # (long/8). The displayed comparison lattice decimates that by two to
        # long/4, so SR N aligns with conventional STFT N/2.
        self.super_res = False
        self.sr_nb = 1024             # long aperture window (fine frequency)
        self.sr_ns = self.sr_nb // 4  # fixed unified geometry
        # Optional upward rung (x2/x4 the long window): pins slow structure
        # the {NB, NB/4} pair is provably blind to (coverage law, notes S5),
        # at near-zero cost (few frames per tile).
        self.sr_rung = 0              # 0 = off, else multiplier
        self.sr_finish = "legacy"     # legacy 75% or palindromic cycle
        self.sr_normalize_rungs = False
        self.sr_stage = "unified"     # raw, seed, or unified
        self.srs: two_lattice.TwoLatticeStream | None = None
        self.reassigned = False       # ordinary phase-reassigned STFT
        self._block = None            # reusable (DH, n_fft) render buffer
        self._rgba = np.zeros((DH, DW, 4), dtype=np.float32)
        self._rgba[..., 3] = 1.0

    # -- source management --------------------------------------------------
    def load(self, path, fmt, rate, is_complex):
        if self.srs:
            self.srs.close()
            self.srs = None
        if self.sr_ra:
            self.sr_ra.close()
            self.sr_ra = None
        for engine in self.sr_extra_ras.values():
            engine.close()
        self.sr_extra_ras.clear()
        if self.src:
            self.src.close()
            self.src = None
        try:
            self.src = iqw.IQSource(path, fmt=fmt, sample_rate=rate,
                                    is_complex=is_complex)
        except Exception as exc:  # noqa: BLE001
            # Nonstandard/raw captures with a .wav name fail header parsing;
            # fall back to int16 at the UI rate instead of making the user
            # flip the format combo by hand.
            if fmt == iqw.FMT_AUTO:
                try:
                    self.src = iqw.IQSource(path, fmt=iqw.FMT_S16,
                                            sample_rate=rate,
                                            is_complex=is_complex)
                    dpg.set_value("status",
                                  "Header parse failed; opened as raw int16 "
                                  f"@ {rate:,.0f} Hz")
                except Exception as exc2:  # noqa: BLE001
                    dpg.set_value("status", f"Open failed: {exc2}")
                    return
            else:
                dpg.set_value("status", f"Open failed: {exc}")
                return
        self._make_wf()
        self.position = 0
        self.playing = False
        self.dirty = True
        n = self.src.num_samples
        fs = self.src.sample_rate
        dur = n / fs if fs else 0
        dpg.set_value("status",
                      f"{path.split('/')[-1]}  |  {n:,} samples  |  "
                      f"{fs:,.0f} Hz  |  {self.src.bits}-bit  |  "
                      f"{'complex' if self.src.is_complex else 'real'}  |  "
                      f"{dur:.2f} s")
        dpg.configure_item("pos_slider", max_value=max(1, n - 1))
        self._sync_time_readout()

    def _make_wf(self):
        if self.wf:
            self.wf.close()
        if self.ra:
            self.ra.close()
        self.wf = iqw.Waterfall(self.n_fft, self.window)
        self.ra = iqw.Reassign(self.n_fft)
        self._block = np.empty((DH, self.n_fft), dtype=np.float32)

    # -- rendering ----------------------------------------------------------
    def _fusion_block(self):
        # Dear PyGui can dispatch callbacks concurrently unless manual callback
        # management is enabled.  Keep this barrier as defense in depth: a
        # solver/readout generation must remain alive for the complete frame.
        with self._sr_lock:
            return self._fusion_block_locked()

    def _fusion_block_locked(self):
        """Python two-lattice recovery followed by the reference readout."""
        needs_solver = self.sr_stage != "raw"
        if self.sr_ra is None or (needs_solver and self.srs is None):
            # Treat solver and readout engine as one generation.  The old
            # two-step mutation could leave srs alive after an aperture change
            # had closed sr_ra, causing a None.render_mem crash.
            if self.srs:
                self.srs.close()
            if self.sr_ra:
                self.sr_ra.close()
            for engine in self.sr_extra_ras.values():
                engine.close()
            self.sr_extra_ras.clear()
            stream = None
            if needs_solver:
                stream = two_lattice.TwoLatticeStream(
                    self.src, n_long=self.sr_nb, n_short=self.sr_ns,
                    h_short=self.sr_ns // 2,
                    iterations=0 if self.sr_stage == "seed" else 1,
                    rung_mult=self.sr_rung, finish_mode=self.sr_finish)
            readout = iqw.Reassign(self.sr_nb)
            readout.set_bilinear(True)
            self.srs, self.sr_ra = stream, readout
            readout_rungs = None
            if self.sr_rung > 1:
                tile = max(8192, 2 * self.sr_nb)
                up = min(self.sr_rung * self.sr_nb, tile)
                readout_rungs = [(up, max(1, up // 8)),
                                 (self.sr_nb, self.sr_nb // 8),
                                 (self.sr_ns, self.sr_ns // 2)]
            if readout_rungs is not None:
                for n, _h in readout_rungs:
                    if n == self.sr_nb or n in self.sr_extra_ras:
                        continue
                    engine = iqw.Reassign(n)
                    engine.set_bilinear(True)
                    self.sr_extra_ras[n] = engine
        # The solve uses its independent short-COLA lattice internally, but
        # reassignment is evaluated DIRECTLY at the external comparison
        # centers. Rendering at N/8 and selecting even rows discarded every
        # quantum assigned to an odd row, which was the main source of holes.
        readout_hop = self.hop
        n_native = DH
        # position is the first output center.  Both the reconstruction and
        # reassignment use center coordinates; no aperture-dependent shift is
        # hidden in a parent/crop index.
        readout_sizes = [self.sr_nb, *self.sr_extra_ras.keys()]
        display_n = max(readout_sizes)
        sample_start = self.position - display_n // 2
        need = (n_native - 1) * readout_hop + display_n
        stream = self.srs
        if self.sr_stage == "raw":
            latent = self.src.read(sample_start, need)
            cov, error = 1.0, float("nan")
        else:
            stream.request(sample_start, need)
            latent, cov, error = stream.assemble(sample_start, need)
        self.recon_cov = cov
        finish_text = ({"raw": "raw IQ", "seed": "shared seed"}.get(
            self.sr_stage) or
            ("75%" if self.sr_finish == "legacy" else "symmetric"))
        if self.sr_stage == "raw":
            err_text = "direct | raw IQ"
        else:
            err_text = (f"C++ shared+1 | {finish_text}"
                        if not np.isfinite(error)
                        else f"err {error:.3g} | {finish_text}")
        rung_text = ""
        if self.sr_rung > 1:
            tile = stream.tile if stream is not None else max(8192, 2*self.sr_nb)
            rung_text = f" | rung {min(self.sr_rung*self.sr_nb, tile)}"
        dpg.set_value("recon_status",
                      f"SR readout N={self.sr_nb}/{self.sr_ns}{rung_text}: "
                      f"{cov*100:.0f}% ready, {err_text}")
        if cov < 0.999:
            self._recon_pending = True
        if self.remove_dc:
            latent = latent - np.mean(latent)
        iq = np.empty(2 * need, dtype=np.float32)
        iq[0::2] = latent.real
        iq[1::2] = latent.imag
        readout = self.sr_ra
        if readout is None:  # generation changed between scheduling and readout
            self._recon_pending = True
            return np.full((DH, self.sr_nb), -120.0, dtype=np.float32)
        blocks = [(self.sr_nb, readout.render_mem(
            iq, display_n // 2, readout_hop, n_native))]
        for n, engine in self.sr_extra_ras.items():
            rung_block = engine.render_mem(
                iq, display_n // 2, readout_hop, n_native)
            if n != display_n:
                rung_block = _resample_axis(rung_block, display_n, axis=1)
            blocks.append((n, rung_block))
        if self.sr_nb != display_n:
            blocks[0] = (self.sr_nb, _resample_axis(
                blocks[0][1], display_n, axis=1))
        if self.sr_normalize_rungs:
            blocks = [(n, b - _rung_gain_db(n, self.sr_nb))
                      for n, b in blocks]
        block = np.maximum.reduce([b for _n, b in blocks])

        return block

    def _reassigned_block(self):
        """Ordinary phase-reassigned long-window STFT."""
        half = self.n_fft // 2
        need = (DH - 1) * self.hop + self.n_fft
        z = self.src.read(self.position - half, need)
        if self.remove_dc:
            z = z - np.mean(z)
        iq = np.empty(2 * need, dtype=np.float32)
        iq[0::2] = z.real
        iq[1::2] = z.imag
        block = self.ra.render_mem(iq, half, self.hop, DH)
        self.recon_cov = 1.0
        dpg.set_value("recon_status",
                      f"Reassigned STFT N={self.n_fft}, H={self.hop}")
        return block

    def recompute(self):
        if not (self.src and self.wf):
            return
        self._recon_pending = False

        if self.super_res:
            block = self._fusion_block()
        elif self.reassigned:
            block = self._reassigned_block()
        else:
            # Conventional STFT rows are center-timed.
            block = self.wf.render(self.src, self.position - self.n_fft // 2,
                                   self.hop, DH, remove_dc=self.remove_dc,
                                   out=self._block)

        # Frequency band -> fftshifted column slice (block may have any # cols).
        ncols = block.shape[1]
        c0 = int(round((self.f_lo + 0.5) * ncols))
        c1 = int(round((self.f_hi + 0.5) * ncols))
        c0 = max(0, min(ncols - 1, c0))
        c1 = max(c0 + 1, min(ncols, c1))
        band = block[:, c0:c1]

        # Resample the band to DW columns (max-pool down / lerp up).
        img = _resample_axis(band, DW, axis=1)

        # dB image -> [0,1] intensity.
        sub = img[::4, ::4]                     # percentile on a subsample
        if self.display_scale == "Amplitude (auto)":
            # Linear amplitude against the image's own 99.5th percentile:
            # background sits at zero, structure uses the whole palette.
            ref = float(np.percentile(sub, 99.5))
            u = np.clip(np.power(10.0, (img - ref) / 20.0), 0.0, 1.0)
        elif self.display_scale == "dB (auto range)":
            hi = float(np.percentile(sub, 99.8))
            span = max(10.0, self.ceil_db - self.floor_db)
            u = np.clip((img - (hi - span)) / span, 0.0, 1.0)
        else:                                   # dB (manual)
            lo, hi = self.floor_db, self.ceil_db
            if hi <= lo:
                hi = lo + 1.0
            u = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        if abs(self.gamma - 1.0) > 1e-3:
            u = np.power(u, self.gamma)

        lut = _LUTS[self.colormap]
        ui = (u * 255).astype(np.intp)
        rgb = lut[ui]                       # (DH, DW, 3)
        if self.flip_time:
            rgb = rgb[::-1]
        self._rgba[..., :3] = rgb
        dpg.set_value("wf_tex", self._rgba.ravel())

        self._update_axes(c0, c1, ncols)
        self.dirty = self._recon_pending

    def _update_axes(self, c0, c1, nf):
        if not self.src:
            return
        fs = self.src.sample_rate
        f0 = (c0 - nf / 2) / nf * fs
        f1 = (c1 - nf / 2) / nf * fs
        t_top = self.position / fs
        t_bot = (self.position + DH * self.hop) / fs
        if self.flip_time:
            ymin, ymax = t_top, t_bot          # newest (top) = t_top
        else:
            ymin, ymax = t_bot, t_top
        dpg.configure_item("wf_image", bounds_min=[f0, ymin],
                           bounds_max=[f1, ymax])
        dpg.set_axis_limits("wf_xaxis", f0, f1)
        dpg.set_axis_limits("wf_yaxis", min(ymin, ymax), max(ymin, ymax))

    def _sync_time_readout(self):
        if not self.src:
            return
        fs = self.src.sample_rate
        span = DH * self.hop / fs if fs else 0
        dpg.set_value("time_readout",
                      f"Row span: {span*1000:.1f} ms  |  hop {self.hop} smp  |  "
                      f"pos {self.position/fs:.3f} s" if fs else "")

    def set_derived_hop(self, new_hop):
        """Change analysis spacing without changing the position marker."""
        self.hop = max(1, int(new_hop))

    # -- transport ----------------------------------------------------------
    def tick(self):
        self._consume_wheel()
        now = time.perf_counter()
        dt = now - self.last_t
        self.last_t = now
        if self.playing and self.src:
            fs = self.src.sample_rate
            adv = int(self.play_rate * fs * dt)
            # advance in whole rows so motion is smooth in the hop grid
            adv = max(self.hop, (adv // self.hop) * self.hop) if adv > 0 else self.hop
            self.position += adv
            if self.position >= self.src.num_samples - self.n_fft:
                self.position = 0        # loop
            dpg.set_value("pos_slider", self.position)
            self.dirty = True
        if self.dirty:
            self.recompute()
            self._sync_time_readout()

    def _consume_wheel(self):
        if not self.src or abs(self._wheel_pending) < 0.5:
            return
        step = 1.0 if self._wheel_pending > 0 else -1.0
        self._wheel_pending -= step
        if self._wheel_target == "position":
            limit = max(0, self.src.num_samples - 1)
            self.position = int(np.clip(
                self.position + step * self.hop * 8, 0, limit))
            dpg.set_value("pos_slider", self.position)
            self._sync_time_readout()
            self.dirty = True
        elif self._wheel_target == "waterfall":
            self.f_lo, self.f_hi = _anchored_zoom(
                self.f_lo, self.f_hi, self._wheel_anchor, step,
                min_span=1.0 / 65536.0)
            dpg.set_value("freq_lo", self.f_lo)
            dpg.set_value("freq_hi", self.f_hi)
            self.dirty = True


app = App()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def cb_open(sender, data):
    dpg.show_item("file_dialog")


def cb_file_selected(sender, data):
    # `selections` is populated by a single click + Open, and is reliable
    # where `file_path_name` can hold directory+filter junk; prefer it.
    sels = data.get("selections") or {}
    path = next(iter(sels.values()), None) or data.get("file_path_name")
    if not path or not os.path.isfile(path):
        dpg.set_value("status", "No file selected (click the file, then Open)")
        return
    fmt = iqw.FORMAT_NAMES[dpg.get_value("fmt_combo")]
    rate = float(dpg.get_value("rate_input"))
    is_complex = dpg.get_value("complex_check")
    app.load(path, fmt, rate, is_complex)


def cb_play():
    app.playing = True
    app.last_t = time.perf_counter()


def cb_pause():
    app.playing = False


def cb_stop():
    app.playing = False
    app.position = 0
    dpg.set_value("pos_slider", 0)
    app.dirty = True


def cb_pos(sender, val):
    app.position = int(val)
    app.dirty = True


def cb_mouse_wheel(sender, wheel):
    """Queue a bounded wheel impulse; the render thread consumes it."""
    if not app.src or not wheel:
        return
    if dpg.is_item_hovered("pos_slider"):
        target, anchor = "position", 0.5
    elif dpg.is_item_hovered("wf_plot"):
        x_hz, _y = dpg.get_plot_mouse_pos()
        fs = float(app.src.sample_rate)
        if fs <= 0:
            return
        span = max(1e-12, app.f_hi - app.f_lo)
        target = "waterfall"
        anchor = float(np.clip((float(x_hz) / fs - app.f_lo) / span,
                               0.0, 1.0))
    else:
        return
    if app._wheel_target != target:
        app._wheel_pending = 0.0
    app._wheel_target = target
    app._wheel_anchor = anchor
    # macOS reports tiny deltas for slow movement and very large deltas for a
    # flick. Treat every callback as an impulse and cap the queued travel.
    impulse = 1.0 if float(wheel) > 0 else -1.0
    app._wheel_pending = float(np.clip(
        app._wheel_pending + impulse, -3.0, 3.0))


def cb_nfft(sender, val):
    app.n_fft = int(val)
    if not app.super_res:
        app.set_derived_hop(app.n_fft // app.stft_hop_div)
    if app.src:
        app._make_wf()
    app.dirty = True


def cb_stft_hop(sender, val):
    app.stft_hop_div = {"N/8": 8, "N/4": 4, "N/2": 2}[val]
    if not app.super_res:
        app.set_derived_hop(app.n_fft // app.stft_hop_div)
        app._sync_time_readout()
        cb_view_mode(None, dpg.get_value("view_mode"))
    app.dirty = True


def cb_window(sender, val):
    app.window = iqw.WINDOWS[val]
    if app.src:
        app._make_wf()
    app.dirty = True


def cb_colormap(sender, val):
    app.colormap = val
    app.dirty = True


def cb_display_scale(sender, val):
    app.display_scale = val
    manual = val != "Amplitude (auto)"
    dpg.configure_item("floor_slider", enabled=manual)
    dpg.configure_item("ceil_slider", enabled=manual)
    app.dirty = True


def cb_gamma(sender, val):
    app.gamma = float(val)
    app.dirty = True


def cb_floor(sender, val):
    app.floor_db = float(val)
    app.dirty = True


def cb_ceil(sender, val):
    app.ceil_db = float(val)
    app.dirty = True


def cb_freq(sender, val):
    lo, hi = dpg.get_value("freq_lo"), dpg.get_value("freq_hi")
    if lo >= hi:
        lo, hi = min(lo, hi - 0.01), max(hi, lo + 0.01)
    app.f_lo, app.f_hi = lo, hi
    app.dirty = True


def cb_freq_full():
    app.f_lo, app.f_hi = -0.5, 0.5
    dpg.set_value("freq_lo", -0.5)
    dpg.set_value("freq_hi", 0.5)
    app.dirty = True


def cb_dc(sender, val):
    app.remove_dc = bool(val)
    app.dirty = True


def cb_view_mode(sender, val):
    """Select one of three mathematically distinct display transforms."""
    app.super_res = val == "Super-resolution (two-aperture)"
    app.reassigned = val == "Reassigned STFT"
    is_streaming = val == "Streaming STFT"
    dpg.configure_item("win_combo", enabled=is_streaming)
    dpg.configure_item("stft_hop_combo", enabled=not app.super_res)
    dpg.configure_item("sr_nb_combo", enabled=app.super_res)
    dpg.configure_item("sr_rung_combo", enabled=app.super_res)
    dpg.configure_item("sr_finish_combo",
                       enabled=app.super_res and app.sr_stage == "unified")
    dpg.configure_item("sr_normalize_check", enabled=app.super_res)
    dpg.configure_item("sr_stage_combo", enabled=app.super_res)
    # Keep the solved SR tiles and its readout plan alive while comparing
    # modes. They are invalidated only by a source/aperture change. Closing
    # here made every A/B toggle restart PGHI and delayed the comparison.
    if app.super_res:
        profile = ({"raw": "raw IQ", "seed": "shared seed"}.get(
            app.sr_stage) or f"{app.sr_finish} finish")
        app.set_derived_hop(app.sr_nb // 4)
        if app.sr_rung > 1:
            rung = min(app.sr_rung * app.sr_nb, max(8192, 2 * app.sr_nb))
            note = (f"Super-resolution (ladder readout) | base {app.sr_nb} "
                    f"| rung {rung} | {profile} | per-cell max power")
        else:
            note = ("Two-lattice solve | independent magnitude families -> "
                    f"one latent waveform | {profile} | "
                    "short=N/4, internal H=N/8, display H=N/4")
        dpg.set_value("transform_note", note)
    elif app.reassigned:
        app.set_derived_hop(app.n_fft // app.stft_hop_div)
        dpg.set_value(
            "transform_note",
            f"Symmetric Hann | local phase reassignment | H=N/{app.stft_hop_div}")
    else:
        app.set_derived_hop(app.n_fft // app.stft_hop_div)
        dpg.set_value(
            "transform_note",
            f"Selected window | center time | H=N/{app.stft_hop_div}")
    dpg.set_value("recon_status", "")
    app.dirty = True


def _rebuild_srs():
    with app._sr_lock:
        if app.srs:
            app.srs.close()
            app.srs = None
        if app.sr_ra:
            app.sr_ra.close()
            app.sr_ra = None
        for engine in app.sr_extra_ras.values():
            engine.close()
        app.sr_extra_ras.clear()
    app.dirty = True


def cb_sr_nb(sender, val):
    nb = int(val)
    app.sr_nb = nb
    app.sr_ns = nb // 4
    app.set_derived_hop(nb // 4)
    rung = (min(app.sr_rung * nb, max(8192, 2 * nb))
            if app.sr_rung > 1 else 0)
    dpg.set_value(
        "sr_geometry",
        f"short {app.sr_ns} | internal hop {nb//8} | display hop {app.hop}"
        + (f" | rung {rung}" if rung else ""))
    if app.super_res:
        cb_view_mode(None, "Super-resolution (two-aperture)")
    _rebuild_srs()


def cb_sr_rung(sender, val):
    app.sr_rung = {"off": 0, "2x": 2, "4x": 4}[val]
    rung = (min(app.sr_rung * app.sr_nb, max(8192, 2 * app.sr_nb))
            if app.sr_rung > 1 else 0)
    dpg.set_value(
        "sr_geometry",
        f"short {app.sr_ns} | internal hop {app.sr_nb//8} | "
        f"display hop {app.hop}" + (f" | rung {rung}" if rung else ""))
    if app.super_res:
        cb_view_mode(None, "Super-resolution (two-aperture)")
    _rebuild_srs()


def cb_sr_finish(sender, val):
    app.sr_finish = {"Legacy 75% long": "legacy",
                     "Symmetric cycle": "palindromic"}[val]
    if app.super_res:
        cb_view_mode(None, "Super-resolution (two-aperture)")
    _rebuild_srs()


def cb_sr_normalize(sender, val):
    app.sr_normalize_rungs = bool(val)
    # Readout-only calibration: completed latent tiles remain valid.
    app.dirty = True


def cb_sr_stage(sender, val):
    app.sr_stage = {"Raw IQ through SR readout": "raw",
                    "Shared seed through SR readout": "seed",
                    "Unified SR": "unified"}[val]
    dpg.configure_item("sr_finish_combo",
                       enabled=app.super_res and app.sr_stage == "unified")
    if app.super_res:
        cb_view_mode(None, "Super-resolution (two-aperture)")
    _rebuild_srs()


def cb_flip(sender, val):
    app.flip_time = bool(val)
    app.dirty = True


def cb_speed(sender, val):
    app.play_rate = float(val)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    dpg.create_context()
    dpg.configure_app(manual_callback_management=True)

    with dpg.handler_registry():
        dpg.add_mouse_wheel_handler(callback=cb_mouse_wheel)

    with dpg.texture_registry():
        dpg.add_raw_texture(DW, DH, app._rgba.ravel(),
                            format=dpg.mvFormat_Float_rgba, tag="wf_tex")

    with dpg.file_dialog(directory_selector=False, show=False,
                         callback=cb_file_selected, tag="file_dialog",
                         width=700, height=460):
        # The first extension is the default filter.  A concrete filter set
        # makes files reliably clickable; ".*" stays available as a fallback.
        dpg.add_file_extension(
            "IQ/WAV files (*.wav *.iq *.raw *.cs16){.wav,.iq,.raw,.cs16}",
            color=(120, 200, 255, 255))
        dpg.add_file_extension(".wav", color=(120, 200, 255, 255))
        dpg.add_file_extension(".iq")
        dpg.add_file_extension(".raw")
        dpg.add_file_extension(".*")

    with dpg.window(tag="main"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Open IQ File", callback=cb_open)
            dpg.add_button(label="Play", callback=cb_play)
            dpg.add_button(label="Pause", callback=cb_pause)
            dpg.add_button(label="Stop", callback=cb_stop)
            dpg.add_text("", tag="status", color=(180, 220, 180))

        with dpg.group(horizontal=True):
            # ---- left settings panel ----
            with dpg.child_window(width=300, autosize_y=True):
                dpg.add_text("Source format")
                dpg.add_combo(list(iqw.FORMAT_NAMES.keys()),
                              default_value="Auto (WAV header)",
                              tag="fmt_combo", width=-1)
                with dpg.group(horizontal=True):
                    dpg.add_text("Rate (Hz)")
                    dpg.add_input_float(default_value=2400000.0, step=0,
                                        tag="rate_input", width=140,
                                        format="%.0f")
                dpg.add_checkbox(label="Complex IQ (raw only)",
                                 default_value=True, tag="complex_check")

                dpg.add_separator()
                dpg.add_text("Transform")
                dpg.add_combo(FFT_SIZES, default_value=1024, tag="nfft_combo",
                              width=-1, callback=cb_nfft)
                dpg.add_combo(["N/8", "N/4", "N/2"], default_value="N/2",
                              tag="stft_hop_combo", width=-1,
                              callback=cb_stft_hop)
                dpg.add_combo(list(iqw.WINDOWS.keys()), default_value="Hann",
                              tag="win_combo", width=-1, callback=cb_window)
                dpg.add_checkbox(label="Remove DC per frame", callback=cb_dc)
                dpg.add_radio_button(
                    ["Streaming STFT", "Super-resolution (two-aperture)",
                     "Reassigned STFT"], default_value="Streaming STFT",
                    tag="view_mode", callback=cb_view_mode)
                dpg.add_text(
                    "Selected window | center time | continuous complex STFT",
                    tag="transform_note", wrap=280,
                    color=(150, 190, 220))

                dpg.add_separator()
                dpg.add_text("Super-resolution apertures")
                with dpg.group(horizontal=True):
                    dpg.add_text("long win")
                    dpg.add_combo(["512", "1024", "2048", "4096","8192"],
                                  default_value="1024", width=80,
                                  tag="sr_nb_combo", enabled=False,
                                  callback=cb_sr_nb)
                    dpg.add_text(
                        "short 256 | internal hop 128 | display hop 256",
                        tag="sr_geometry")
                with dpg.group(horizontal=True):
                    dpg.add_text("slow rung")
                    dpg.add_combo(["off", "2x", "4x"], default_value="off",
                                  width=70, tag="sr_rung_combo",
                                  enabled=False, callback=cb_sr_rung)
                with dpg.group(horizontal=True):
                    dpg.add_text("finish")
                    dpg.add_combo(["Legacy 75% long", "Symmetric cycle"],
                                  default_value="Legacy 75% long", width=170,
                                  tag="sr_finish_combo", enabled=False,
                                  callback=cb_sr_finish)
                dpg.add_checkbox(label="Normalize rung gain",
                                 default_value=False,
                                 tag="sr_normalize_check", enabled=False,
                                 callback=cb_sr_normalize)
                dpg.add_combo(["Raw IQ through SR readout",
                               "Shared seed through SR readout",
                               "Unified SR"],
                              default_value="Unified SR", width=-1,
                              tag="sr_stage_combo", enabled=False,
                              callback=cb_sr_stage)
                dpg.add_text("", tag="recon_status", color=(200, 180, 120))

                dpg.add_separator()
                dpg.add_text("Display")
                dpg.add_combo(list(_LUTS.keys()), default_value="Viridis",
                              tag="cmap_combo", width=-1, callback=cb_colormap)
                dpg.add_combo(["Amplitude (auto)", "dB (auto range)",
                               "dB (manual)"],
                              default_value=app.display_scale,
                              tag="scale_combo", width=-1,
                              callback=cb_display_scale)
                dpg.add_slider_float(label="gamma", min_value=0.3,
                                     max_value=3.0, default_value=1.0,
                                     tag="gamma_slider", callback=cb_gamma)
                dpg.add_slider_float(label="Floor dB", min_value=-160,
                                     max_value=0, default_value=app.floor_db,
                                     tag="floor_slider", enabled=False,
                                     callback=cb_floor)
                dpg.add_slider_float(label="Ceiling dB", min_value=-160,
                                     max_value=40, default_value=app.ceil_db,
                                     tag="ceil_slider", enabled=False,
                                     callback=cb_ceil)
                dpg.add_checkbox(label="Newest on top", default_value=True,
                                 callback=cb_flip)

                dpg.add_separator()
                dpg.add_text("Frequency zoom (fraction of Fs)")
                dpg.add_slider_float(label="f low", min_value=-0.5,
                                     max_value=0.5, default_value=-0.5,
                                     tag="freq_lo", callback=cb_freq)
                dpg.add_slider_float(label="f high", min_value=-0.5,
                                     max_value=0.5, default_value=0.5,
                                     tag="freq_hi", callback=cb_freq)
                dpg.add_button(label="Full band", callback=cb_freq_full)

                dpg.add_separator()
                dpg.add_text("Time grid (transform-derived COLA)")
                dpg.add_text("", tag="time_readout")

                dpg.add_separator()
                dpg.add_text("Playback")
                dpg.add_slider_float(label="Speed x", min_value=0.1,
                                     max_value=50.0, default_value=1.0,
                                     callback=cb_speed)

            # ---- waterfall plot ----
            with dpg.child_window(autosize_x=True, autosize_y=True):
                dpg.add_slider_int(label="Position (samples)", min_value=0,
                                   max_value=1, default_value=0,
                                   tag="pos_slider", width=-1, callback=cb_pos)
                with dpg.plot(label="Waterfall", height=-1, width=-1,
                              tag="wf_plot"):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)",
                                      tag="wf_xaxis")
                    with dpg.plot_axis(dpg.mvYAxis, label="Time (s)",
                                       tag="wf_yaxis"):
                        dpg.add_image_series("wf_tex", [-0.5, 0.0], [0.5, 1.0],
                                             tag="wf_image")

    dpg.create_viewport(title="BFFT IQ Waterfall", width=1280, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)


def main():
    build_ui()
    while dpg.is_dearpygui_running():
        jobs = dpg.get_callback_queue()
        if jobs:
            dpg.run_callbacks(jobs)
        app.tick()
        dpg.render_dearpygui_frame()
    if app.srs:
        app.srs.close()
    if app.sr_ra:
        app.sr_ra.close()
    for engine in app.sr_extra_ras.values():
        engine.close()
    if app.ra:
        app.ra.close()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
