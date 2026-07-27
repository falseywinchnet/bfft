#!/usr/bin/env python3
"""Viewer for one-shot recursively nucleated image cells."""

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
from recursive_decomposition import _fit_rgb  # noqa: E402
from seeded_decomposition import SeededConfig, SeededDecomposition  # noqa: E402


VIEWS = [
    "Reconstruction + sites", "Reconstruction", "Error",
    "Cell classes", "Cells", "Recursive boundary field",
    "Collective residual", "Detail germination order", "Original",
]
SOURCE_TEXTURE = "seeded_source"
RESULT_TEXTURE = "seeded_result"
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
        for tag in (SOURCE_TEXTURE, RESULT_TEXTURE):
            dpg.add_raw_texture(
                w, h, S.buffers[tag], tag=tag,
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
    return SeededConfig(
        max_side=int(dpg.get_value("max_side")),
        iterations=int(dpg.get_value("stages")),
        alpha=float(dpg.get_value("alpha")),
        mode=str(dpg.get_value("recurrence")),
        passes=int(dpg.get_value("passes")),
        lam=float(dpg.get_value("lam")),
        mu=float(dpg.get_value("mu")),
        coarse_cells=int(dpg.get_value("coarse_cells")),
        detail_cells=int(dpg.get_value("detail_cells")),
        density_floor=float(dpg.get_value("density_floor")),
        detail_density_floor=float(dpg.get_value("detail_floor")),
        boundary_gain=float(dpg.get_value("boundary_gain")),
        residual_gain=float(dpg.get_value("residual_gain")),
        stage_decay=float(dpg.get_value("stage_decay")),
        residual_scale=float(dpg.get_value("residual_scale")),
        detail_anisotropy=float(dpg.get_value("detail_anisotropy")),
        cell_softness=float(dpg.get_value("seed_softness")),
        descent_steps=int(dpg.get_value("seed_descent")),
        descent_rate=float(dpg.get_value("seed_rate")),
        blue_noise_repulsion=float(dpg.get_value("repulsion")),
        ownership_softness=float(dpg.get_value("ownership_softness")),
        detail_reach=float(dpg.get_value("detail_reach")),
        power_irregularity=float(dpg.get_value("power_irregularity")),
        bounded_gradients=bool(dpg.get_value("bounded_gradients")),
    )


def refresh():
    with S.lock:
        model = S.model
    if model is None:
        return
    push_texture(SOURCE_TEXTURE, model.rgb)
    push_texture(RESULT_TEXTURE, model.view(dpg.get_value("view")))
    coarse = int(np.count_nonzero(model.marks == 0))
    detail = len(model.marks) - coarse
    dpg.set_value(
        "metrics",
        f"cells {len(model.seeds)} (coarse {coarse}, detail {detail}) | "
        f"direct-fit PSNR {model.psnr:.2f} dB | "
        f"{model.elapsed_ms:.0f} ms")


def compute_worker():
    if S.busy or S.image is None:
        return
    S.busy = True
    S.status = (
        "Recursive fields -> mixed blue noise -> one direct OKLab fit...")
    try:
        model = SeededDecomposition(S.image, config_from_ui())
        with S.lock:
            S.model = model
        S.status = (
            f"{S.name}: one-shot fit complete at "
            f"{model.psnr:.2f} dB with {len(model.seeds)} cells.")
    except Exception as exc:
        S.status = f"Build failed: {type(exc).__name__}: {exc}"
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
    S.status = f"{name} loaded. Press Build seeded fit."


def cb_gallery(sender, label):
    try:
        key = gallery.key_for_label(label)
        adopt(gallery.load(key), gallery.describe(key)["label"])
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
    except Exception as exc:
        S.status = f"Image load failed: {type(exc).__name__}: {exc}"


def sf(label, tag, value, lo, hi, width=270, fmt="%.3f"):
    dpg.add_slider_float(
        label=label, tag=tag, default_value=value,
        min_value=lo, max_value=hi, width=width, format=fmt)


def build_ui(labels, default_label):
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
                labels, default_value=default_label, tag="gallery",
                callback=cb_gallery, width=430)
            dpg.add_button(
                label="Load image...",
                callback=lambda: dpg.show_item("file_dialog"))
            dpg.add_button(
                label="Build seeded fit", callback=cb_compute,
                tag="compute_button")
        dpg.add_text("", tag="status", wrap=1500)
        dpg.add_text("", tag="metrics")
        dpg.add_separator()

        with dpg.collapsing_header(
                label="Recursive nucleation fields", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="work max side", tag="max_side",
                    default_value=256, min_value=64, max_value=384,
                    width=260)
                dpg.add_slider_int(
                    label="recursive stages", tag="stages",
                    default_value=5, min_value=1, max_value=12, width=260)
                dpg.add_combo(
                    ["half both", "texture decay", "cartoon decay"],
                    default_value="half both", tag="recurrence", width=170)
                sf("retained fraction", "alpha", 0.5, 0.0, 1.0)
                dpg.add_slider_int(
                    label="TGFD passes", tag="passes", default_value=8,
                    min_value=2, max_value=32, width=240)
                sf("lambda", "lam", 0.05, 0.01, 0.12)
                sf("mu", "mu", 40.0, 4.0, 120.0, fmt="%.1f")
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="isotropic coarse cells", tag="coarse_cells",
                    default_value=180, min_value=8, max_value=1000,
                    width=280)
                dpg.add_slider_int(
                    label="residual detail cells", tag="detail_cells",
                    default_value=1020, min_value=8, max_value=4000,
                    width=280)
                sf("coarse floor", "density_floor", 0.35, 0.01, 2.0)
                sf("detail floor", "detail_floor", 0.04, 0.0, 1.0)
                sf("boundary attraction", "boundary_gain", 3.0, 0.0, 20.0)
                sf("residual attraction", "residual_gain", 8.0, 0.0, 24.0)
            with dpg.group(horizontal=True):
                sf("stage decay", "stage_decay", 0.72, 0.1, 1.0)
                sf("residual scale", "residual_scale", 2.5, 0.35, 10.0)
                sf(
                    "fine anisotropy", "detail_anisotropy",
                    8.0, 0.0, 20.0)
                sf("seed softness", "seed_softness", 8.0, 0.0, 30.0)
                dpg.add_slider_int(
                    label="seed descent", tag="seed_descent",
                    default_value=6, min_value=0, max_value=30, width=250)
                sf("seed rate", "seed_rate", 0.35, 0.0, 2.0)
                sf("blue-noise repulsion", "repulsion", 0.18, 0.0, 1.0)

        with dpg.collapsing_header(
                label="One direct global fit", default_open=True):
            with dpg.group(horizontal=True):
                sf(
                    "ownership softness", "ownership_softness",
                    9.0, 0.0, 30.0)
                sf("detail reach", "detail_reach", 0.8, 0.1, 2.0)
                sf(
                    "power irregularity", "power_irregularity",
                    0.16, 0.0, 1.5)
                dpg.add_checkbox(
                    label="bound affine gradients",
                    tag="bounded_gradients", default_value=True)

        with dpg.group(horizontal=True):
            dpg.add_text("Right panel")
            dpg.add_combo(
                VIEWS, default_value="Reconstruction", tag="view",
                width=250, callback=lambda: refresh())

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image(SOURCE_TEXTURE, tag="source_image")
            with dpg.group():
                dpg.add_text("Recursively nucleated direct fit")
                dpg.add_image(RESULT_TEXTURE, tag="result_image")


def main():
    keys = gallery.available()
    if "camera" in keys:
        keys.remove("camera")
        keys.insert(0, "camera")
    labels = gallery.labels(keys)
    default_label = (
        gallery.labels(["camera"])[0]
        if "camera" in keys else labels[0])
    dpg.create_context()
    alloc_textures(8, 8)
    build_ui(labels, default_label)
    dpg.create_viewport(
        title="BFFT Recursively Nucleated Decomposition",
        width=1580, height=1040)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)
    cb_gallery(None, default_label)
    cb_compute()
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
