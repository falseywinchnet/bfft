"""IQ waterfall viewer.

A streaming, zoomable spectrogram/waterfall for IQ recordings, backed by the
monolithic BFFT-based `libiqwaterfall` library.

Features
--------
  * Open WAV IQ files (auto header) or raw IQ (u8/s8/s16/s24/s32/f32, real or
    complex) with a user-set sample rate.
  * Play / Pause / Stop transport with a draggable position scrubber.
  * Time zoom (samples-per-row) and frequency zoom (band select).
  * Adjustable dynamic range (floor / ceiling dB) to stretch the colormap.
  * FFT size, analysis window, and colormap settings.

Run:  python iq_waterfall_app.py
"""
from __future__ import annotations

import os
import sys
import time

# Make this app runnable from any working directory: ensure its own folder is
# on sys.path so `import iqwaterfall` (the sibling wrapper) always resolves.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import dearpygui.dearpygui as dpg

import iqwaterfall as iqw
import dip_stream

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

# Display texture dimensions (frequency columns x time rows).
DW = 1024
DH = 640

FFT_SIZES = [256, 512, 1024, 2048, 4096, 8192]


class App:
    def __init__(self):
        self.src: iqw.IQSource | None = None
        self.wf: iqw.Waterfall | None = None

        self.n_fft = 1024
        self.window = 0
        self.colormap = "Viridis"
        self.hop = 256                # samples per waterfall row (time zoom)
        self.position = 0             # top-row complex-sample index
        self.floor_db = -90.0
        self.ceil_db = -10.0
        self.f_lo = -0.5              # frequency band, fraction of Fs
        self.f_hi = 0.5
        self.remove_dc = False
        self.flip_time = True         # newest row at top (SDR# style)

        self.playing = False
        self.play_rate = 1.0          # x realtime
        self.last_t = time.perf_counter()
        self.dirty = True

        # DIP/PGHI streaming reconstruction
        self.reconstruct = False
        self.reconstructor: dip_stream.StreamReconstructor | None = None
        self.recon_workers = 3
        self.recon_cov = 0.0
        self.super_res = False        # active-delta endpoint-readout super-resolution
        self.sr_steps = 1             # up to 3 fixed L5 (projection) steps
        self.sr_alpha = 1.0           # projection step size (0.5..1.0)
        self.sr_nb = 1024             # long aperture window (fine freq)
        self.sr_ns = 128              # short aperture window (fine time)
        self.srs: dip_stream.FusionStream | None = None
        # Cap the reconstructed span (raw-fills beyond coverage). Sized to the
        # segment cache so zoomed-out views still reconstruct.
        self.max_recon_span = 1_200_000

        self._block = None            # reusable (DH, n_fft) render buffer
        self._rgba = np.zeros((DH, DW, 4), dtype=np.float32)
        self._rgba[..., 3] = 1.0

    # -- source management --------------------------------------------------
    def load(self, path, fmt, rate, is_complex):
        if self.srs:
            self.srs.close()
            self.srs = None
        if self.reconstructor:
            self.reconstructor.close()
            self.reconstructor = None
        if self.src:
            self.src.close()
            self.src = None
        try:
            self.src = iqw.IQSource(path, fmt=fmt, sample_rate=rate,
                                    is_complex=is_complex)
        except Exception as exc:  # noqa: BLE001
            dpg.set_value("status", f"Open failed: {exc}")
            return
        self._make_wf()
        if self.reconstruct:
            self.reconstructor = dip_stream.StreamReconstructor(
                self.src, workers=self.recon_workers)
        if self.super_res:
            self.srs = dip_stream.FusionStream(
                self.src, n_steps=self.sr_steps, alpha=self.sr_alpha,
                nb=self.sr_nb, ns=self.sr_ns)
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
        self.wf = iqw.Waterfall(self.n_fft, self.window)
        self._block = np.empty((DH, self.n_fft), dtype=np.float32)

    # -- rendering ----------------------------------------------------------
    def _recon_buffer(self, need, label):
        """Return a complex buffer for [position, position+need): reconstructed
        (DIP) if enabled and within the span cap, else raw."""
        if (self.reconstruct and self.reconstructor is not None
                and need <= self.max_recon_span):
            self.reconstructor.request(self.position, need)
            buf, cov = self.reconstructor.assemble(self.position, need)
            self.recon_cov = cov
            dpg.set_value("recon_status", f"{label} (DIP): {cov*100:.0f}% ready")
            if cov < 0.999:
                self._recon_pending = True
            return buf[0::2] + 1j * buf[1::2]
        if self.reconstruct and self.reconstructor is not None:
            dpg.set_value("recon_status", f"{label} (raw — zoom in for DIP)")
        return self.src.read(self.position, need)

    def _super_res_block(self):
        """F1xT2 fusion super-resolution: long endpoint (fine freq) MIN short
        endpoints (fine time), off one refined active-delta state. Native grid is
        HS=32-hop x nb bins; tiles solved in parallel and resampled to display."""
        if self.srs is None:
            self.srs = dip_stream.FusionStream(
                self.src, n_steps=self.sr_steps, alpha=self.sr_alpha,
                nb=self.sr_nb, ns=self.sr_ns)
        fh = self.srs.frame_hop                     # HS = 32 samples
        span = DH * self.hop
        # fine frame f is at sample (f + lo)*HS.
        frame_lo = max(0, self.position // fh - self.srs.lo)
        n_native = max(1, span // fh)
        self.srs.request(frame_lo, n_native)
        rows, cov = self.srs.assemble(frame_lo, n_native)     # [n_native, nb]
        self.recon_cov = cov
        dpg.set_value("recon_status",
                      f"Super-res (fusion F1xT2): {cov*100:.0f}% ready")
        if cov < 0.999:
            self._recon_pending = True
        block = 20.0 * np.log10(rows + 1e-9)                   # [n_native, nb]
        idx = np.linspace(0, n_native - 1, DH).astype(np.intp)  # time -> DH rows
        block = block[idx]
        return np.fft.fftshift(block, axes=1)                 # col 0 = -Fs/2

    def recompute(self):
        if not (self.src and self.wf):
            return
        self._recon_pending = False

        if self.super_res:
            block = self._super_res_block()
        elif self.reconstruct and self.reconstructor is not None \
                and (DH - 1) * self.hop + self.n_fft <= self.max_recon_span:
            need = (DH - 1) * self.hop + self.n_fft
            z = self._recon_buffer(need, "Reconstruction")
            buf = np.empty(2 * z.size, dtype=np.float32)
            buf[0::2] = z.real; buf[1::2] = z.imag
            block = self.wf.render_mem(buf, 0, self.hop, DH,
                                       remove_dc=self.remove_dc, out=self._block)
        else:
            if self.reconstruct and self.reconstructor is not None:
                dpg.set_value("recon_status",
                              "Zoom in (smaller hop) to reconstruct this span")
            block = self.wf.render(self.src, self.position, self.hop, DH,
                                   remove_dc=self.remove_dc, out=self._block)

        # Frequency band -> fftshifted column slice (block may have any # cols).
        ncols = block.shape[1]
        c0 = int(round((self.f_lo + 0.5) * ncols))
        c1 = int(round((self.f_hi + 0.5) * ncols))
        c0 = max(0, min(ncols - 1, c0))
        c1 = max(c0 + 1, min(ncols, c1))
        band = block[:, c0:c1]

        # Resample the band to DW columns (nearest).
        idx = np.linspace(0, band.shape[1] - 1, DW).astype(np.intp)
        img = band[:, idx]

        # Normalize dB -> [0,1] using the dynamic-range window.
        lo, hi = self.floor_db, self.ceil_db
        if hi <= lo:
            hi = lo + 1.0
        u = np.clip((img - lo) / (hi - lo), 0.0, 1.0)

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

    # -- transport ----------------------------------------------------------
    def tick(self):
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


app = App()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def cb_open(sender, data):
    dpg.show_item("file_dialog")


def cb_file_selected(sender, data):
    path = data.get("file_path_name")
    if not path:
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


def cb_nfft(sender, val):
    app.n_fft = int(val)
    if app.src:
        app._make_wf()
    app.dirty = True


def cb_window(sender, val):
    app.window = iqw.WINDOWS[val]
    if app.src:
        app._make_wf()
    app.dirty = True


def cb_colormap(sender, val):
    app.colormap = val
    app.dirty = True


def cb_hop(sender, val):
    app.hop = max(1, int(val))
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


def cb_reconstruct(sender, val):
    app.reconstruct = bool(val)
    if app.reconstruct:
        if app.src and app.reconstructor is None:
            app.reconstructor = dip_stream.StreamReconstructor(
                app.src, workers=app.recon_workers)
    else:
        dpg.set_value("recon_status", "")
    app.dirty = True


def cb_superres(sender, val):
    app.super_res = bool(val)
    if app.super_res and app.src and app.srs is None:
        app.srs = dip_stream.FusionStream(app.src, n_steps=app.sr_steps,
                                          alpha=app.sr_alpha, nb=app.sr_nb,
                                          ns=app.sr_ns)
    if not app.super_res:
        dpg.set_value("recon_status", "")
    app.dirty = True


def cb_sr_steps(sender, val):
    app.sr_steps = int(val)
    if app.srs:
        app.srs.close()
        app.srs = None            # rebuilt on next recompute with new step count
    app.dirty = True


def cb_sr_alpha(sender, val):
    app.sr_alpha = float(val)
    if app.srs:
        app.srs.close()
        app.srs = None
    app.dirty = True


def _rebuild_srs():
    if app.srs:
        app.srs.close()
        app.srs = None
    app.dirty = True


def _max_valid_ns(nb):
    """Largest short window: ES = power of two, ES <= EB-4 (EB=nb/32), ns=ES*32."""
    max_es = nb // 32 - 4
    es = 2
    while es * 2 <= max_es:
        es *= 2
    return es * 32


def cb_sr_nb(sender, val):
    nb = int(val)
    mx = _max_valid_ns(nb)
    if app.sr_ns > mx:                    # snap short window into valid range
        app.sr_ns = mx
        dpg.set_value("sr_ns_combo", mx)
    app.sr_nb = nb
    _rebuild_srs()


def cb_sr_ns(sender, val):
    ns = int(val)
    mx = _max_valid_ns(app.sr_nb)
    if ns > mx:
        ns = mx
        dpg.set_value("sr_ns_combo", ns)
    app.sr_ns = ns
    _rebuild_srs()


def cb_flip(sender, val):
    app.flip_time = bool(val)
    app.dirty = True


def cb_speed(sender, val):
    app.play_rate = float(val)


def cb_zoom_time(factor):
    app.hop = max(1, int(round(app.hop * factor)))
    dpg.set_value("hop_slider", app.hop)
    app.dirty = True


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    dpg.create_context()

    with dpg.texture_registry():
        dpg.add_raw_texture(DW, DH, app._rgba.ravel(),
                            format=dpg.mvFormat_Float_rgba, tag="wf_tex")

    with dpg.file_dialog(directory_selector=False, show=False,
                         callback=cb_file_selected, tag="file_dialog",
                         width=700, height=460):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".wav", color=(120, 200, 255, 255))
        dpg.add_file_extension(".iq")
        dpg.add_file_extension(".raw")

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
                dpg.add_combo(list(iqw.WINDOWS.keys()), default_value="Hann",
                              tag="win_combo", width=-1, callback=cb_window)
                dpg.add_checkbox(label="Remove DC per frame", callback=cb_dc)

                dpg.add_separator()
                dpg.add_text("DIP/PGHI reconstruction")
                dpg.add_checkbox(label="Reconstruct (streaming)",
                                 callback=cb_reconstruct)
                dpg.add_checkbox(label="Super-res (active-delta)",
                                 callback=cb_superres)
                with dpg.group(horizontal=True):
                    dpg.add_text("long win")
                    dpg.add_combo([512, 1024, 2048, 4096], default_value=1024,
                                  width=80, tag="sr_nb_combo", callback=cb_sr_nb)
                    dpg.add_text("short win")
                    dpg.add_combo([64, 128, 256, 512], default_value=128,
                                  width=70, tag="sr_ns_combo", callback=cb_sr_ns)
                with dpg.group(horizontal=True):
                    dpg.add_text("L5 steps")
                    dpg.add_combo([0, 1, 2, 3], default_value=1, width=70,
                                  callback=cb_sr_steps)
                dpg.add_slider_float(label="alpha", min_value=0.0, max_value=2.0,
                                     default_value=1.0, callback=cb_sr_alpha)
                dpg.add_text("", tag="recon_status", color=(200, 180, 120))

                dpg.add_separator()
                dpg.add_text("Display")
                dpg.add_combo(list(_LUTS.keys()), default_value="Viridis",
                              tag="cmap_combo", width=-1, callback=cb_colormap)
                dpg.add_slider_float(label="Floor dB", min_value=-160,
                                     max_value=0, default_value=app.floor_db,
                                     tag="floor_slider", callback=cb_floor)
                dpg.add_slider_float(label="Ceiling dB", min_value=-160,
                                     max_value=40, default_value=app.ceil_db,
                                     tag="ceil_slider", callback=cb_ceil)
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
                dpg.add_text("Time zoom")
                dpg.add_slider_int(label="hop (smp/row)", min_value=1,
                                   max_value=65536, default_value=app.hop,
                                   tag="hop_slider", callback=cb_hop)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Zoom in",
                                   callback=lambda: cb_zoom_time(0.5))
                    dpg.add_button(label="Zoom out",
                                   callback=lambda: cb_zoom_time(2.0))
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
        app.tick()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
