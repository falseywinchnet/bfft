#!/usr/bin/env python3
"""DearPyGui laboratory for recursive BFFT decomposition."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dearpygui.dearpygui as dpg  # noqa: E402
import gallery  # noqa: E402
from recursive_decomposition import (  # noqa: E402
    MODES, RecursiveConfig, RecursiveDecomposition, _fit_rgb,
)


VIEWS = [
    "Recursive image", "Final recursive image", "Removed shell",
    "Accumulated removal", "Cartoon at stage", "Texture at stage",
    "Carried residual", "Stage boundary", "Boundary accumulation",
    "Collective carried residual", "Collective residual support",
    "Nucleation density", "Nucleation seeds",
    "Detail germination density", "Detail germination order",
    "Residual scale map", "Residual directional consistency",
    "Detail anisotropy preview",
    "Diffuse coarse cells",
    "Coarse cell regions", "Original",
]
SPACES = ["oklab_lc", "gray", "oklab", "rgb"]
SOURCE_TEXTURE = "recursive_source"
RESULT_TEXTURE = "recursive_result"
PANEL = 680


class State:
    def __init__(self):
        self.image = None
        self.name = "(none)"
        self.model = None
        self.busy = False
        self.status = "Choose an image."
        self.lock = threading.Lock()
        self.shape = (8, 8)
        self.buffers = {}


S = State()


def alloc_textures(h, w):
    S.shape = (h, w)
    for tag in (SOURCE_TEXTURE, RESULT_TEXTURE):
        S.buffers[tag] = np.ones(h * w * 4, dtype=np.float32)
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        if dpg.does_alias_exist(tag):
            dpg.remove_alias(tag)
    with dpg.texture_registry():
        dpg.add_raw_texture(
            w, h, S.buffers[SOURCE_TEXTURE], tag=SOURCE_TEXTURE,
            format=dpg.mvFormat_Float_rgba)
        dpg.add_raw_texture(
            w, h, S.buffers[RESULT_TEXTURE], tag=RESULT_TEXTURE,
            format=dpg.mvFormat_Float_rgba)
    scale = PANEL / max(h, w)
    for item, tag in (("source_image", SOURCE_TEXTURE),
                      ("result_image", RESULT_TEXTURE)):
        if dpg.does_item_exist(item):
            dpg.configure_item(
                item, texture_tag=tag, width=max(1, int(w * scale)),
                height=max(1, int(h * scale)))


def push_texture(tag, rgb):
    image = np.asarray(rgb, dtype=np.float32)
    if image.shape[:2] != S.shape:
        alloc_textures(*image.shape[:2])
    buffer = S.buffers[tag]
    buffer[0::4] = image[..., 0].ravel()
    buffer[1::4] = image[..., 1].ravel()
    buffer[2::4] = image[..., 2].ravel()
    dpg.set_value(tag, buffer)


def config_from_ui():
    return RecursiveConfig(
        max_side=int(dpg.get_value("max_side")),
        iterations=int(dpg.get_value("iterations")),
        alpha=float(dpg.get_value("alpha")),
        mode=str(dpg.get_value("mode")),
        space=str(dpg.get_value("space")),
        passes=int(dpg.get_value("passes")),
        lam=float(dpg.get_value("lam")),
        mu=float(dpg.get_value("mu")),
        coarse_cells=int(dpg.get_value("coarse_cells")),
        detail_cells=int(dpg.get_value("detail_cells")),
        density_floor=float(dpg.get_value("density_floor")),
        detail_density_floor=float(dpg.get_value("detail_density_floor")),
        residual_scale=float(dpg.get_value("residual_scale")),
        detail_anisotropy=float(dpg.get_value("detail_anisotropy")),
        residual_gain=float(dpg.get_value("residual_gain")),
        boundary_gain=float(dpg.get_value("boundary_gain")),
        stage_decay=float(dpg.get_value("stage_decay")),
        cell_irregularity=float(dpg.get_value("cell_irregularity")),
        cell_softness=float(dpg.get_value("cell_softness")),
        descent_steps=int(dpg.get_value("descent_steps")),
        descent_rate=float(dpg.get_value("descent_rate")),
        blue_noise_repulsion=float(dpg.get_value("blue_noise_repulsion")),
    )


def refresh():
    with S.lock:
        model = S.model
    if model is None:
        return
    stage = int(np.clip(
        dpg.get_value("stage"), 0, len(model.splits) - 1))
    push_texture(SOURCE_TEXTURE, model.rgb)
    push_texture(
        RESULT_TEXTURE, model.view(dpg.get_value("view"), stage))
    dpg.configure_item(
        "stage", max_value=max(0, len(model.splits) - 1))
    dpg.set_value(
        "metrics",
        f"{model.w}x{model.h} | {len(model.splits)} recursive stages | "
        f"{len(model.seeds)} isotropic coarse germs | "
        f"{len(model.detail_seeds)} ordered detail germs | "
        f"{model.elapsed_ms:.0f} ms total"
        + (f" | decomposition loss "
           f"{model.descent_loss[0]:.3f} -> "
           f"{model.descent_loss[-1]:.3f}"
           if model.descent_loss else ""))


def compute_worker():
    if S.busy or S.image is None:
        return
    S.busy = True
    S.status = "Recursing BFFT decomposition and accumulating boundaries..."
    try:
        model = RecursiveDecomposition(S.image, config_from_ui())
        with S.lock:
            S.model = model
        S.status = (
            f"{S.name}: {len(model.splits)} stages computed. "
            "Move the stage slider and inspect the removed shells.")
    except Exception as exc:
        S.status = f"Compute failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_compute():
    if not S.busy:
        threading.Thread(target=compute_worker, daemon=True).start()


def adopt(image, name):
    S.image = np.asarray(image, dtype=np.float64)
    S.name = name
    with S.lock:
        S.model = None
    preview = _fit_rgb(S.image, int(dpg.get_value("max_side")))
    push_texture(SOURCE_TEXTURE, preview)
    push_texture(RESULT_TEXTURE, np.full_like(preview, 0.08))
    S.status = f"{name} loaded. Press Recompute."


def cb_gallery(sender, label):
    try:
        key = gallery.key_for_label(label)
        adopt(gallery.load(key), gallery.describe(key)["label"])
        cb_compute()
    except Exception as exc:
        S.status = f"Gallery load failed: {type(exc).__name__}: {exc}"


def cb_file(sender, app_data):
    selections = app_data.get("selections") or {}
    candidates = list(selections.values())
    if app_data.get("file_path_name"):
        candidates.append(app_data["file_path_name"])
    path = next((Path(p) for p in candidates if Path(p).is_file()), None)
    if path is None:
        S.status = "Could not resolve selected image."
        return
    try:
        import matplotlib.image as mpimg
        adopt(mpimg.imread(path), path.name)
        cb_compute()
    except Exception as exc:
        S.status = f"Image load failed: {type(exc).__name__}: {exc}"


def build_ui(labels):
    with dpg.file_dialog(
            directory_selector=False, show=False, callback=cb_file,
            tag="file_dialog", width=900, height=520):
        dpg.add_file_extension(
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
            "{.png,.jpg,.jpeg,.tif,.tiff,.bmp}")
        dpg.add_file_extension(".*")

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            dpg.add_text("Image")
            dpg.add_combo(
                labels, default_value=labels[0], tag="gallery",
                width=430, callback=cb_gallery)
            dpg.add_button(
                label="Load image...",
                callback=lambda: dpg.show_item("file_dialog"))
            dpg.add_button(
                label="Recompute", callback=cb_compute,
                tag="compute_button")
        dpg.add_text("", tag="status", wrap=1500)
        dpg.add_text("", tag="metrics")
        dpg.add_separator()

        with dpg.collapsing_header(
                label="Recursive decomposition", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    list(MODES), default_value="half both", tag="mode",
                    width=180)
                dpg.add_slider_float(
                    label="retained layer fraction", tag="alpha",
                    default_value=0.5, min_value=0.0, max_value=1.0,
                    width=320)
                dpg.add_slider_int(
                    label="recursive stages", tag="iterations",
                    default_value=6, min_value=1, max_value=16, width=280)
                dpg.add_combo(
                    SPACES, default_value="oklab_lc", tag="space",
                    width=150)
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="work max side", tag="max_side",
                    default_value=256, min_value=96, max_value=512,
                    width=280)
                dpg.add_slider_int(
                    label="TGFD passes per stage", tag="passes",
                    default_value=16, min_value=2, max_value=64, width=300)
                dpg.add_slider_float(
                    label="lambda", tag="lam", default_value=0.05,
                    min_value=0.01, max_value=0.12, format="%.3f",
                    width=280)
                dpg.add_slider_float(
                    label="mu", tag="mu", default_value=40.0,
                    min_value=4.0, max_value=120.0, width=280)

        with dpg.collapsing_header(
                label="Flow-boundary coarse underlayer",
                default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="isotropic coarse germs", tag="coarse_cells",
                    default_value=180, min_value=8, max_value=1000,
                    width=300)
                dpg.add_slider_int(
                    label="detail germ sequence", tag="detail_cells",
                    default_value=320, min_value=8, max_value=2000,
                    width=300)
                dpg.add_slider_float(
                    label="even coverage floor", tag="density_floor",
                    default_value=0.35, min_value=0.01, max_value=2.0,
                    width=300)
                dpg.add_slider_float(
                    label="collective residual attraction",
                    tag="residual_gain", default_value=8.0,
                    min_value=0.0, max_value=24.0, width=320)
                dpg.add_slider_float(
                    label="detail coverage floor",
                    tag="detail_density_floor", default_value=0.04,
                    min_value=0.0, max_value=1.0, width=300)
                dpg.add_slider_float(
                    label="coarse/fine residual scale",
                    tag="residual_scale", default_value=2.5,
                    min_value=0.35, max_value=10.0, width=300)
                dpg.add_slider_float(
                    label="fine-detail anisotropy",
                    tag="detail_anisotropy", default_value=8.0,
                    min_value=0.0, max_value=20.0, width=300)
                dpg.add_slider_float(
                    label="stage-boundary attraction", tag="boundary_gain",
                    default_value=3.0, min_value=0.0, max_value=20.0,
                    width=320)
                dpg.add_slider_float(
                    label="deeper-stage decay", tag="stage_decay",
                    default_value=0.72, min_value=0.1, max_value=1.0,
                    width=280)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="cell irregularity", tag="cell_irregularity",
                    default_value=0.28, min_value=0.0, max_value=1.5,
                    width=300)
                dpg.add_slider_float(
                    label="diffuse overlap", tag="cell_softness",
                    default_value=8.0, min_value=0.0, max_value=30.0,
                    width=300)
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="decomposition descent steps",
                    tag="descent_steps", default_value=8,
                    min_value=0, max_value=40, width=300)
                dpg.add_slider_float(
                    label="centroid descent rate", tag="descent_rate",
                    default_value=0.35, min_value=0.0, max_value=2.0,
                    width=300)
                dpg.add_slider_float(
                    label="blue-noise repulsion",
                    tag="blue_noise_repulsion", default_value=0.18,
                    min_value=0.0, max_value=1.0, width=300)

        with dpg.group(horizontal=True):
            dpg.add_text("Right panel")
            dpg.add_combo(
                VIEWS, default_value=VIEWS[0], tag="view",
                width=250, callback=lambda: refresh())
            dpg.add_slider_int(
                label="stage", tag="stage", default_value=0,
                min_value=0, max_value=5, width=340,
                callback=lambda: refresh())

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image(SOURCE_TEXTURE, tag="source_image")
            with dpg.group():
                dpg.add_text("Recursive decomposition study")
                dpg.add_image(RESULT_TEXTURE, tag="result_image")


def main():
    keys = gallery.available()
    labels = gallery.labels(keys)
    dpg.create_context()
    alloc_textures(8, 8)
    build_ui(labels)
    dpg.create_viewport(
        title="Recursive BFFT Adaptation Laboratory",
        width=1500, height=1020)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)
    cb_gallery(None, labels[0])
    last_refresh = 0.0
    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dpg.set_value("status", S.status)
        dpg.configure_item("compute_button", enabled=not S.busy)
        if now - last_refresh > 0.15:
            refresh()
            last_refresh = now
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
