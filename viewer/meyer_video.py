#!/usr/bin/env python3
"""Meyer cartoon + texture video decomposer (DearPyGui).

Loads a video, decodes it to grayscale frames at a power-of-two working
size, runs bfft.meyer_split on every frame, and deposits the results in
place into two framebuffers (cartoon and texture).  Playback controls
scrub/play/loop the buffers at the source frame rate.

Decoding is done by piping raw frames from the ffmpeg CLI (which also
performs the scaling), so the only python dependencies are dearpygui,
numpy, and bfft itself.

Realtime: the 30 fps budget is 33.3 ms/frame.  Measured split cost on a
4-lane build (best of 5, ms):

    size      p=8   p=16  p=24  p=32  p=48  p=64
    256x256    2.8   5.6    8.7  11.9  21.5  23.9
    512x256    6.0  19.6   18.6  23.3  34.3  47.1
    512x512   13.9  27.1   41.2  64.4  81.3 105.0

The presets below are chosen to sit under the budget.  The processing
panel reports achieved ms/frame and flags whether the run was realtime.

Run:  .venv/bin/python viewer/meyer_video.py
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402
import dearpygui.dearpygui as dpg  # noqa: E402

# (label, (H, W), passes).  Rates are MEASURED SUSTAINED throughput --
# mean over 120 distinct frames at T=4, not best-of-N on a warm array.
# Frame-level and intra-frame parallelism both cap at the same rate here:
# the kernel is memory-bandwidth bound, so headroom comes from doing less
# work per frame, not from more concurrency.
PRESETS = [
    ("512x256 p=12  ~86 fps  REALTIME", (256, 512), 12),
    ("512x256 p=8   ~114 fps REALTIME", (256, 512), 8),
    ("256x256 p=24  ~88 fps  REALTIME", (256, 256), 24),
    ("512x256 p=16  ~48 fps  realtime, thin margin", (256, 512), 16),
    ("512x512 p=8   ~35 fps  marginal", (512, 512), 8),
    ("512x256 p=32  ~20 fps  OFFLINE", (256, 512), 32),
    ("512x512 p=16  ~22 fps  OFFLINE", (512, 512), 16),
    ("512x512 p=32  ~11 fps  OFFLINE", (512, 512), 32),
]
BUDGET_MS = 1000.0 / 30.0
MAX_FRAMES = 1200


class State:
    def __init__(self):
        self.path = None
        self.src_fps = 30.0
        self.n_frames = 0
        self.shape = (256, 512)
        self.cartoon = None      # uint8 (N, H, W)
        self.texture = None      # uint8 (N, H, W)
        self.gray = None         # uint8 (N, H, W) source
        self.ready = False
        self.busy = False
        self.cancel = False
        self.progress = 0.0
        self.status = "No video loaded."
        self.ms_per_frame = 0.0
        self.decode_ms = 0.0
        self.playing = False
        self.frame = 0
        self.t0 = 0.0
        self.f0 = 0
        self.loop = True
        self.tex_gain = 3.0
        self.n_done = 0          # frames decomposed so far (streaming)
        self.stream = True       # display live while decomposing
        self.stream_t0 = 0.0
        self.proc_fps = 0.0      # live EMA of decomposition throughput
        self.behind = 0          # frames the producer failed to deliver on time
        self.lock = threading.Lock()


S = State()


# ----------------------------------------------------------------------
# ffmpeg / ffprobe
# ----------------------------------------------------------------------

def probe(path):
    """Return (fps, n_frames_estimate, width, height)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,nb_frames,width,height,duration",
         "-of", "default=noprint_wrappers=1:nokey=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    d = {}
    for line in out.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    num, _, den = d.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    n = d.get("nb_frames", "N/A")
    if n.isdigit():
        n_frames = int(n)
    else:
        dur = d.get("duration", "")
        try:
            n_frames = int(float(dur) * fps)
        except ValueError:
            n_frames = 0
    return fps, n_frames, int(d.get("width", 0)), int(d.get("height", 0))


def decode_frames(path, H, W, max_frames):
    """Yield HxW uint8 grayscale frames; ffmpeg performs the scaling."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "gray",
           "-vf", f"scale={W}:{H}", "-"]
    nbytes = H * W
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nbytes * 8)
    try:
        count = 0
        while count < max_frames:
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(H, W)
            count += 1
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass
        p.terminate()
        p.wait(timeout=5)


# ----------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------

def calibrate(sample, target_fps, threads, headroom=1.25):
    """Pick the richest preset that actually sustains target_fps HERE.

    Machine state dominates: the same kernel measured 86 fps and 16 fps on
    the same host minutes apart, depending on background load and thermal
    state.  A baked-in table is therefore worthless as a realtime promise;
    we measure instead.  `sample` is one representative frame.
    """
    results = []
    # try richest first; stop at the first that clears target*headroom
    for idx, (label, shape, passes) in enumerate(PRESETS):
        H, W = shape
        fr = np.asarray(
            [[sample[int(y * sample.shape[0] / H)][int(x * sample.shape[1] / W)]
              for x in range(W)] for y in range(H)], dtype=np.float64) \
            if sample.shape != (H, W) else sample.astype(np.float64)
        pl = bfft.MeyerPlan((H, W), passes=passes, threads=threads)
        pl.split(fr)                                   # warm
        t0 = time.perf_counter()
        reps = 0
        while time.perf_counter() - t0 < 0.35:         # short, honest burst
            pl.split(fr)
            reps += 1
        fps = reps / (time.perf_counter() - t0)
        results.append((idx, label, fps))
        del pl
        if fps >= target_fps * headroom:
            return idx, results
    best = max(results, key=lambda r: r[2])
    return best[0], results


# ----------------------------------------------------------------------
# processing
# ----------------------------------------------------------------------

def process_video(path, shape, passes, threads):
    """Producer: decode -> decompose -> deposit into the framebuffers.

    Buffers are allocated up front and filled in place, so the consumer
    (the UI loop) can display frame i the moment S.n_done exceeds it --
    the decomposition streams rather than running as a batch pre-pass.
    """
    H, W = shape
    S.busy = True
    S.cancel = False
    S.ready = False
    S.n_done = 0
    S.behind = 0
    S.progress = 0.0
    S.proc_fps = 0.0
    S.status = "Probing..."
    try:
        fps, n_est, sw, sh = probe(path)
        S.src_fps = fps if fps > 0 else 30.0
        cap = min(n_est if n_est > 0 else MAX_FRAMES, MAX_FRAMES)

        plan = bfft.MeyerPlan((H, W), passes=passes, threads=threads)
        gray = np.zeros((cap, H, W), dtype=np.uint8)
        cart = np.zeros((cap, H, W), dtype=np.uint8)
        tex = np.zeros((cap, H, W), dtype=np.uint8)
        with S.lock:
            S.gray, S.cartoon, S.texture = gray, cart, tex
            S.n_frames = cap
            S.shape = (H, W)
            S.frame = 0
            S.ready = True          # consumer may start immediately
        S.stream_t0 = time.perf_counter()
        S.status = (f"Streaming {sw}x{sh} -> {W}x{H}, {cap} frames "
                    f"@ {S.src_fps:.2f} fps...")

        gain = S.tex_gain
        t_split = 0.0
        n = 0
        t_wall0 = time.perf_counter()
        for fr in decode_frames(path, H, W, cap):
            if S.cancel:
                break
            t_a = time.perf_counter()
            c, x = plan.split(fr.astype(np.float64))
            t_split += time.perf_counter() - t_a
            gray[n] = fr
            np.clip(c, 0, 255, out=c)
            cart[n] = c.astype(np.uint8)
            np.clip(128.0 + gain * x, 0, 255, out=x)
            tex[n] = x.astype(np.uint8)
            n += 1
            S.n_done = n            # publish AFTER the buffers are written
            S.progress = n / max(cap, 1)
            el = time.perf_counter() - t_wall0
            if el > 0:
                S.proc_fps = n / el
        wall = time.perf_counter() - t_wall0
        if n == 0:
            S.status = "No frames decoded (unsupported file?)."
            S.ready = False
            return
        with S.lock:
            S.gray, S.cartoon, S.texture = gray[:n], cart[:n], tex[:n]
            S.n_frames = n
        S.ms_per_frame = t_split / n * 1e3
        thru = n / wall if wall > 0 else 0.0
        rt = ("REALTIME (" + f"{thru / S.src_fps:.1f}x source)"
              if thru >= S.src_fps else
              f"NOT realtime ({thru / S.src_fps:.2f}x source)")
        S.status = (f"{n} frames @ {S.src_fps:.2f} fps source | "
                    f"pipeline {thru:.1f} fps sustained -- {rt} | "
                    f"split {S.ms_per_frame:.1f} ms/frame"
                    + (f" | {S.behind} late" if S.behind else ""))
    except Exception as exc:  # surfaced in the UI, not swallowed
        S.status = f"ERROR: {type(exc).__name__}: {exc}"
        S.ready = False
    finally:
        S.busy = False
        S.progress = 1.0


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

TEX_C, TEX_T, TEX_S = "tex_cartoon", "tex_texture", "tex_source"
BUF = {}


def alloc_textures(H, W):
    # RGBA, not RGB: 3-component float raw textures trip a Metal row-
    # alignment assertion on macOS (bytes_per_row >= used_bytes_per_row).
    for tag in (TEX_C, TEX_T, TEX_S):
        BUF[tag] = np.ones(H * W * 4, dtype=np.float32)
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        # delete_item frees the item but NOT its alias; re-registering the
        # same tag without this raises "Alias already exists"
        if dpg.does_alias_exist(tag):
            dpg.remove_alias(tag)
    with dpg.texture_registry():
        for tag in (TEX_C, TEX_T, TEX_S):
            dpg.add_raw_texture(W, H, BUF[tag], tag=tag,
                                format=dpg.mvFormat_Float_rgba)
    for img, tag in (("img_c", TEX_C), ("img_t", TEX_T), ("img_s", TEX_S)):
        if dpg.does_item_exist(img):
            dpg.configure_item(img, texture_tag=tag, width=W, height=H)


def push_frame(i):
    with S.lock:
        if not S.ready or S.cartoon is None:
            return
        i = max(0, min(i, S.n_frames - 1))
        c = S.cartoon[i]
        t = S.texture[i]
        g = S.gray[i]
    for arr, tag in ((c, TEX_C), (t, TEX_T), (g, TEX_S)):
        v = arr.astype(np.float32).ravel() * (1.0 / 255.0)
        b = BUF[tag]
        b[0::4] = v
        b[1::4] = v
        b[2::4] = v          # alpha stays 1.0 from allocation
        dpg.set_value(tag, b)
    dpg.set_value("frame_slider", i)
    dpg.set_value("frame_label", f"frame {i + 1} / {S.n_frames}")


def resolve_selection(app_data):
    """Extract a real file path from a DearPyGui file-dialog payload.

    `selections` is authoritative when present; `file_path_name` is a
    fallback that can arrive as a directory (or as a stale path from the
    dialog's current folder) depending on how the item was chosen.
    """
    sels = app_data.get("selections") or {}
    for cand in list(sels.values()):
        p = Path(cand)
        if p.is_file():
            return p
    cand = app_data.get("file_path_name") or ""
    if cand:
        p = Path(cand)
        if p.is_file():
            return p
        # some builds report folder + name separately
        alt = Path(app_data.get("current_path") or "") / \
            (app_data.get("file_name") or "")
        if alt.is_file():
            return alt
    return None


def cb_file(sender, app_data):
    p = resolve_selection(app_data)
    if p is None:
        S.path = None
        dpg.set_value("path_label", "(none)")
        S.status = ("Could not resolve that selection to a file. "
                    "Pick the video file itself, not a folder.")
        return
    S.path = str(p)
    dpg.set_value("path_label", p.name)
    try:
        fps, n, w, h = probe(p)
        S.src_fps = fps if fps > 0 else 30.0
        est = min(n if n > 0 else MAX_FRAMES, MAX_FRAMES)
        S.status = (f"Loaded {p.name}: {w}x{h}, {fps:.2f} fps, "
                    f"{n if n > 0 else '?'} frames "
                    f"({est} will be processed). Press Decompose.")
    except Exception as exc:
        S.status = (f"Loaded {p.name}, but ffprobe failed "
                    f"({type(exc).__name__}: {exc}). It may still decode; "
                    f"press Decompose to try.")


def cb_process():
    if S.busy:
        return
    if not S.path:
        S.status = "No video loaded -- press 'Load video...' and pick a file."
        return
    if not Path(S.path).is_file():
        S.status = f"File no longer exists: {S.path}"
        S.path = None
        dpg.set_value("path_label", "(none)")
        return
    idx = PRESETS_LABELS.index(dpg.get_value("preset"))
    _, shape, passes = PRESETS[idx]
    S.tex_gain = dpg.get_value("gain")
    threads = int(dpg.get_value("threads"))
    S.playing = False
    alloc_textures(*shape)
    threading.Thread(target=process_video,
                     args=(S.path, shape, passes, threads),
                     daemon=True).start()


def cb_play():
    if not S.ready:
        return
    S.playing = not S.playing
    if S.playing:
        S.t0 = time.perf_counter()
        S.f0 = S.frame
    dpg.set_item_label("btn_play", "Pause" if S.playing else "Play")


def cb_replay():
    if not S.ready:
        return
    S.frame = 0
    S.t0 = time.perf_counter()
    S.f0 = 0
    S.playing = True
    dpg.set_item_label("btn_play", "Pause")


def cb_slider(sender, val):
    if S.ready:
        S.frame = int(val)
        S.f0 = S.frame
        S.t0 = time.perf_counter()
        push_frame(S.frame)


def build_ui():
    global PRESETS_LABELS
    PRESETS_LABELS = [p[0] for p in PRESETS]

    with dpg.file_dialog(directory_selector=False, show=False,
                         callback=cb_file, tag="file_dlg",
                         default_path=str(Path.home()),
                         width=900, height=520):
        dpg.add_file_extension(
            "Video (*.mp4 *.mov *.avi *.mkv *.webm *.m4v){"
            ".mp4,.mov,.avi,.mkv,.webm,.m4v}")
        dpg.add_file_extension(".*")

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Load video...",
                           callback=lambda: dpg.show_item("file_dlg"))
            dpg.add_text("(none)", tag="path_label")
        with dpg.group(horizontal=True):
            dpg.add_text("Preset")
            dpg.add_combo(PRESETS_LABELS, default_value=PRESETS_LABELS[0],
                          tag="preset", width=280)
            dpg.add_text("Threads")
            dpg.add_input_int(default_value=4, min_value=1, max_value=16,
                              tag="threads", width=90, step=1)
            dpg.add_text("Texture gain")
            dpg.add_slider_float(default_value=3.0, min_value=1.0,
                                 max_value=8.0, tag="gain", width=140)
            dpg.add_button(label="Decompose", callback=cb_process,
                           tag="btn_go")
        dpg.add_progress_bar(tag="prog", width=-1, default_value=0.0)
        dpg.add_text("No video loaded.", tag="status", wrap=1180)
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_button(label="Play", callback=cb_play, tag="btn_play",
                           width=90)
            dpg.add_button(label="Replay", callback=cb_replay, width=90)
            dpg.add_checkbox(label="Loop", default_value=True, tag="loop")
            dpg.add_text("frame - / -", tag="frame_label")
            dpg.add_text("", tag="rate_label")
        dpg.add_slider_int(tag="frame_slider", width=-1, min_value=0,
                           max_value=0, callback=cb_slider,
                           default_value=0)
        dpg.add_separator()

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("source")
                dpg.add_image(TEX_S, tag="img_s")
            with dpg.group():
                dpg.add_text("cartoon  u")
                dpg.add_image(TEX_C, tag="img_c")
            with dpg.group():
                dpg.add_text("texture  v")
                dpg.add_image(TEX_T, tag="img_t")

    dpg.create_viewport(title="BFFT Meyer cartoon + texture video decomposer",
                        width=1720, height=760)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)


def main():
    dpg.create_context()
    alloc_textures(*PRESETS[0][1])   # textures must exist before add_image
    build_ui()
    last_ready = False
    last_ui = 0.0
    shown = -1
    while dpg.is_dearpygui_running():
        now = time.perf_counter()

        # Throttle UI writes to ~15 Hz.  Calling set_value on every render
        # frame competes with the producer for the GIL and measurably cuts
        # decomposition throughput.
        if now - last_ui > 0.066:
            dpg.set_value("prog", S.progress)
            dpg.set_value("status", S.status)
            dpg.configure_item("btn_go", enabled=not S.busy)
            if S.busy:
                dpg.set_value("rate_label",
                              f"decomposing {S.proc_fps:5.1f} fps  "
                              f"(source {S.src_fps:.1f})")
            last_ui = now

        if S.ready and not last_ready:
            dpg.configure_item("frame_slider", max_value=max(S.n_frames - 1, 0))
            S.stream_t0 = now
            S.playing = True
            dpg.set_item_label("btn_play", "Pause")
            last_ready = True
            shown = -1
        elif not S.ready:
            last_ready = False

        if S.playing and S.ready:
            avail = (S.n_done - 1) if S.busy else (S.n_frames - 1)
            if avail >= 0:
                target = S.f0 + int((now - S.t0) * S.src_fps) if not S.busy \
                    else int((now - S.stream_t0) * S.src_fps)
                if S.busy:
                    # streaming: follow the clock, but never past what the
                    # producer has published; count the shortfall honestly
                    if target > avail:
                        S.behind += 1
                        target = avail
                elif target >= S.n_frames:
                    if dpg.get_value("loop"):
                        S.t0 = now
                        S.f0 = 0
                        target = 0
                    else:
                        target = S.n_frames - 1
                        S.playing = False
                        dpg.set_item_label("btn_play", "Play")
                if target != shown:
                    S.frame = target
                    push_frame(target)
                    shown = target

        dpg.render_dearpygui_frame()
    S.cancel = True
    dpg.destroy_context()


if __name__ == "__main__":
    main()
