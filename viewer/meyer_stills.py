#!/usr/bin/env python3
"""Meyer decomposition and recomposition explorer (DearPyGui).

Picks an image from the built-in gallery -- texture segregation fields at
matched statistics, natural textures, colour photographs -- or one of your
own, decomposes it plane by plane in a chosen colour space, and lets you
drive the layer gains live.

The decomposition is the expensive part and runs once, in a worker thread.
Recomposition from cached layers is a handful of numpy passes, so the gain
sliders are interactive: cartoon gain, texture gain, and shading gain each
act on their own layer, and unit gains reproduce the source exactly.

The shading layer -- cartoon minus a plain ROF solve of the cartoon -- is
the illumination a flat cartoon discards.  Its ROF solve is cached and
recomputed in the background only when its own constant changes.

Run:  .venv/bin/python viewer/meyer_stills.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bfft  # noqa: E402
import dearpygui.dearpygui as dpg  # noqa: E402
import gallery  # noqa: E402
from bfft.effects import lab_to_srgb, meyer_channels, shade  # noqa: E402

SPACES = ["oklab_lc", "oklab", "rgb", "gray"]
SPACE_HELP = {
    "oklab_lc": "OKLab lightness + chroma; hue carried untouched",
    "oklab": "OKLab L, a, b independently",
    "rgb": "R, G, B independently",
    "gray": "luma only; original chroma carried through",
}
PANEL = 360
MAX_SIDE = 640          # images are downsampled past this for interactivity


class State:
    def __init__(self):
        self.img = None          # source, native form
        self.name = "(none)"
        self.split = None        # ChannelSplit
        self.shade = None        # (H, W, K) cached shading layers
        self.shade_c = 0.02      # constant the cache was built at
        self.shade_want = 0.02
        self.busy = False
        self.status = "Pick an image and press Decompose."
        self.split_ms = 0.0
        self.render_ms = 0.0
        self.dirty = False       # gains changed; redraw the panels
        self.shade_stale = False
        self.lock = threading.Lock()


S = State()


# ----------------------------------------------------------------------
# work
# ----------------------------------------------------------------------

def _fit(a):
    """Downsample by an integer stride so the long side is <= MAX_SIDE."""
    h, w = a.shape[:2]
    step = int(np.ceil(max(h, w) / MAX_SIDE))
    return a[::step, ::step] if step > 1 else a


def do_split(space, passes, mu):
    S.busy = True
    try:
        t0 = time.perf_counter()
        sp = meyer_channels(S.img, space=space, mu=mu, passes=passes)
        dt = time.perf_counter() - t0
        sh = np.stack([shade(sp.cartoon[..., i], c=S.shade_want)
                       for i in range(sp.planes.shape[2])], -1)
        with S.lock:
            S.split = sp
            S.shade = sh
            S.shade_c = S.shade_want
            S.split_ms = dt * 1e3
            S.dirty = True
        k = sp.planes.shape[2]
        S.status = (f"{S.name}  {S.img.shape[0]}x{S.img.shape[1]}  "
                    f"{space}: {k} plane(s) {sp.names} in {dt * 1e3:.0f} ms "
                    f"({dt * 1e3 / k:.0f} ms/plane, {passes} passes).  "
                    f"Drag the gains.")
    except Exception as exc:
        S.status = f"Decomposition failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def do_shade():
    """Rebuild the cached shading layers at the requested constant."""
    S.busy = True
    try:
        with S.lock:
            sp, c = S.split, S.shade_want
        if sp is None:
            return
        sh = np.stack([shade(sp.cartoon[..., i], c=c)
                       for i in range(sp.planes.shape[2])], -1)
        with S.lock:
            S.shade = sh
            S.shade_c = c
            S.dirty = True
    finally:
        S.busy = False


def _to_native(planes, sp):
    """Working-space planes -> a displayable RGB image in [0, 1]."""
    from bfft.effects import _from_working
    out = np.empty_like(planes)
    for i in range(planes.shape[2]):
        out[..., i] = sp.offset[i] + planes[..., i] / sp.scale[i]
    rgb = _from_working(out, sp.space, sp.carried,
                        sp.carried.get("_was_gray", False))
    if rgb.ndim == 2:
        rgb = np.stack([rgb / 255.0] * 3, -1)
    elif rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    return np.clip(rgb, 0.0, 1.0)


def panels(sp, sh, gc, gt, gs, only_luma=False):
    """The four display panels: source, cartoon, texture, recomposition.

    Pure: everything it needs is passed in, so it is testable without a
    UI."""
    k = sp.planes.shape[2]
    only = only_luma and sp.space in ("oklab", "oklab_lc")
    resid = sp.planes - sp.cartoon - sp.texture
    gcv = np.full(k, gc)
    gtv = np.full(k, gt)
    gsv = np.full(k, gs)
    if only:                       # act on the first plane (L) alone
        gcv[1:] = 1.0
        gtv[1:] = 1.0
        gsv[1:] = 0.0

    out = resid + sp.cartoon * gcv + sp.texture * gtv
    if sh is not None and np.any(gsv):
        out = out + sh * gsv

    src = _to_native(sp.planes, sp)
    cart = _to_native(resid + sp.cartoon, sp)
    tex = np.clip(0.5 + sp.texture.mean(axis=2) / 255.0, 0, 1)
    tex = np.stack([tex] * 3, -1)
    rec = _to_native(out, sp)
    return src, cart, tex, rec


def _source_display():
    """The source alone, before any decomposition, as RGB in [0, 1]."""
    a = S.img
    if a is None:
        return None
    if a.ndim == 2:
        a = np.stack([a / 255.0] * 3, -1)
    else:
        a = a[..., :3]
    grey = np.full_like(a, 0.12)
    return np.clip(a, 0, 1), grey, grey, grey


def render():
    """Read the current gains and build the panels."""
    with S.lock:
        sp, sh = S.split, S.shade
    if sp is None:
        return _source_display()
    t0 = time.perf_counter()
    out = panels(sp, sh, dpg.get_value("gc"), dpg.get_value("gt"),
                 dpg.get_value("gs"), dpg.get_value("only_luma"))
    S.render_ms = (time.perf_counter() - t0) * 1e3
    return out


# ----------------------------------------------------------------------
# textures
# ----------------------------------------------------------------------

TAGS = ("tex_src", "tex_cart", "tex_texd", "tex_rec")
IMGS = ("img_src", "img_cart", "img_texd", "img_rec")
BUF = {}
SHAPE = [8, 8]


def alloc_textures(H, W):
    # RGBA, not RGB: 3-component float raw textures trip a Metal row-
    # alignment assertion on macOS.
    SHAPE[0], SHAPE[1] = H, W
    for tag in TAGS:
        BUF[tag] = np.ones(H * W * 4, dtype=np.float32)
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        # delete_item frees the item but NOT its alias
        if dpg.does_alias_exist(tag):
            dpg.remove_alias(tag)
    with dpg.texture_registry():
        for tag in TAGS:
            dpg.add_raw_texture(W, H, BUF[tag], tag=tag,
                                format=dpg.mvFormat_Float_rgba)
    scale = PANEL / max(H, W)
    for img, tag in zip(IMGS, TAGS):
        if dpg.does_item_exist(img):
            dpg.configure_item(img, texture_tag=tag, width=int(W * scale),
                               height=int(H * scale))


def push(panels):
    for arr, tag in zip(panels, TAGS):
        b = BUF[tag]
        v = arr.astype(np.float32)
        b[0::4] = v[..., 0].ravel()
        b[1::4] = v[..., 1].ravel()
        b[2::4] = v[..., 2].ravel()      # alpha stays 1.0
        dpg.set_value(tag, b)


# ----------------------------------------------------------------------
# callbacks
# ----------------------------------------------------------------------

def _adopt(img, name):
    S.img = _fit(np.asarray(img, dtype=np.float64))
    S.name = name
    S.split = None
    S.shade = None
    h, w = S.img.shape[:2]
    alloc_textures(h, w)
    S.dirty = True
    S.status = f"{name}: {h}x{w}. Press Decompose."


def cb_gallery(sender, label):
    try:
        key = gallery.key_for_label(label)
        _adopt(gallery.load(key), gallery.describe(key)["label"])
    except Exception as exc:
        S.status = f"Could not load that entry: {type(exc).__name__}: {exc}"


def cb_file(sender, app_data):
    sels = app_data.get("selections") or {}
    path = None
    for cand in list(sels.values()):
        if Path(cand).is_file():
            path = Path(cand)
            break
    if path is None:
        cand = app_data.get("file_path_name") or ""
        if cand and Path(cand).is_file():
            path = Path(cand)
    if path is None:
        S.status = "Could not resolve that selection to a file."
        return
    try:
        import matplotlib.image as mpimg
        a = np.asarray(mpimg.imread(str(path)), dtype=np.float64)
        if a.ndim == 3 and a.shape[2] >= 3:
            a = a[..., :3]
            if a.max() > 1.5:
                a = a / 255.0
        elif a.ndim == 3:
            a = a[..., 0]
        if a.ndim == 2 and a.max() <= 1.5:
            a = a * 255.0
        _adopt(a, path.name)
    except Exception as exc:
        S.status = f"Could not read {path.name}: {type(exc).__name__}: {exc}"


def cb_decompose():
    if S.busy or S.img is None:
        if S.img is None:
            S.status = "No image selected."
        return
    space = dpg.get_value("space")
    passes = int(dpg.get_value("passes"))
    mu = float(dpg.get_value("mu"))
    S.status = f"Decomposing in {space}..."
    threading.Thread(target=do_split, args=(space, passes, mu),
                     daemon=True).start()


def cb_gain(sender, val):
    S.dirty = True


def cb_shade_c(sender, val):
    S.shade_want = float(val)
    S.shade_stale = True


def cb_reset():
    dpg.set_value("gc", 1.0)
    dpg.set_value("gt", 1.0)
    dpg.set_value("gs", 0.0)
    S.dirty = True


PRESETS = [
    ("identity", 1.0, 1.0, 0.0),
    ("cartoon only", 1.0, 0.0, 0.0),
    ("texture x3", 1.0, 3.0, 0.0),
    ("texture removed, shading up", 1.0, 0.0, 2.0),
    ("flatten (cartoon x0.7, texture x2)", 0.7, 2.0, 0.0),
    ("shading x3", 1.0, 1.0, 2.0),
    ("inverted texture", 1.0, -1.0, 0.0),
]


def cb_preset(sender, label):
    for name, gc, gt, gs in PRESETS:
        if name == label:
            dpg.set_value("gc", gc)
            dpg.set_value("gt", gt)
            dpg.set_value("gs", gs)
            S.dirty = True
            return


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

def build_ui(labels):
    with dpg.file_dialog(directory_selector=False, show=False,
                         callback=cb_file, tag="file_dlg",
                         default_path=str(Path.home()), width=900,
                         height=520):
        dpg.add_file_extension(
            "Image (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
            "{.png,.jpg,.jpeg,.tif,.tiff,.bmp}")
        dpg.add_file_extension(".*")

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            dpg.add_text("Gallery")
            dpg.add_combo(labels, default_value=labels[0], tag="pick",
                          width=340, callback=cb_gallery)
            dpg.add_button(label="Load image...",
                           callback=lambda: dpg.show_item("file_dlg"))
        with dpg.group(horizontal=True):
            dpg.add_text("Space")
            dpg.add_combo(SPACES, default_value=SPACES[0], tag="space",
                          width=110)
            dpg.add_text("passes")
            dpg.add_input_int(default_value=64, min_value=4, max_value=400,
                              step=8, tag="passes", width=110)
            dpg.add_text("mu")
            dpg.add_input_float(default_value=40.0, min_value=2.0,
                                max_value=400.0, step=5.0, tag="mu",
                                width=130, format="%.0f")
            dpg.add_button(label="Decompose", callback=cb_decompose,
                           tag="btn_go")
        dpg.add_text(SPACE_HELP[SPACES[0]], tag="space_help")
        dpg.add_text(S.status, tag="status", wrap=1500)
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_text("Preset")
            dpg.add_combo([p[0] for p in PRESETS], default_value="identity",
                          tag="preset", width=260, callback=cb_preset)
            dpg.add_button(label="Reset gains", callback=cb_reset)
            dpg.add_checkbox(label="luma only", tag="only_luma",
                             default_value=False, callback=cb_gain)
            dpg.add_text("", tag="timing")
        dpg.add_slider_float(label="cartoon gain", tag="gc", width=460,
                             default_value=1.0, min_value=0.0, max_value=2.0,
                             callback=cb_gain)
        dpg.add_slider_float(label="texture gain", tag="gt", width=460,
                             default_value=1.0, min_value=-2.0, max_value=6.0,
                             callback=cb_gain)
        dpg.add_slider_float(label="shading gain (added)", tag="gs", width=460,
                             default_value=0.0, min_value=-1.0, max_value=4.0,
                             callback=cb_gain)
        dpg.add_slider_float(label="shading ROF constant c", tag="sc",
                             width=460, default_value=0.02, min_value=0.004,
                             max_value=0.2, format="%.3f",
                             callback=cb_shade_c)
        dpg.add_separator()

        with dpg.group(horizontal=True):
            for title, img, tag in (("source", IMGS[0], TAGS[0]),
                                    ("cartoon", IMGS[1], TAGS[1]),
                                    ("texture (+0.5)", IMGS[2], TAGS[2]),
                                    ("recomposed", IMGS[3], TAGS[3])):
                with dpg.group():
                    dpg.add_text(title)
                    dpg.add_image(tag, tag=img)

    dpg.create_viewport(title="BFFT Meyer decomposition explorer",
                        width=1560, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)


def main():
    keys = gallery.available()
    if not keys:
        print("No gallery images available (is scikit-image installed?)")
        return 1
    labels = gallery.labels(keys)

    dpg.create_context()
    alloc_textures(8, 8)             # textures must exist before add_image
    build_ui(labels)
    cb_gallery(None, labels[0])

    last_ui = 0.0
    last_render = 0.0
    while dpg.is_dearpygui_running():
        now = time.perf_counter()

        if now - last_ui > 0.1:
            dpg.set_value("status", S.status)
            dpg.set_value("space_help", SPACE_HELP[dpg.get_value("space")])
            dpg.configure_item("btn_go", enabled=not S.busy)
            if S.split is not None:
                dpg.set_value("timing",
                              f"split {S.split_ms:.0f} ms | "
                              f"recompose {S.render_ms:.1f} ms | "
                              f"shading c={S.shade_c:.3f}")
            last_ui = now

        # A shading rebuild is a full ROF solve per plane; coalesce the
        # slider's stream of changes into one run once it settles.
        if S.shade_stale and not S.busy and S.split is not None and \
                abs(S.shade_want - S.shade_c) > 1e-9:
            S.shade_stale = False
            threading.Thread(target=do_shade, daemon=True).start()
        elif S.shade_stale and abs(S.shade_want - S.shade_c) <= 1e-9:
            S.shade_stale = False

        if S.dirty and now - last_render > 0.033:
            S.dirty = False
            panels = render()
            if panels is not None:
                if panels[0].shape[:2] != tuple(SHAPE):
                    alloc_textures(*panels[0].shape[:2])
                push(panels)
            last_render = now

        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    sys.exit(main())
