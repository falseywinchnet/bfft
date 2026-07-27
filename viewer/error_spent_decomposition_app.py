#!/usr/bin/env python3
"""Viewer for blue-noise cells allocated by spending measured error."""

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
from error_spent_decomposition import (  # noqa: E402
    ErrorSpentConfig, ErrorSpentDecomposition,
)
from recursive_decomposition import _fit_rgb  # noqa: E402


VIEWS = [
    "Reconstruction", "Error spent", "Allocation order", "Cell classes",
    "Cells", "Recursive priority only", "Local cell character", "Original",
]
SOURCE_TEXTURE = "spend_source"
RESULT_TEXTURE = "spend_result"
PANEL = 690


class State:
    image = None
    name = "(none)"
    model = None
    busy = False
    status = "Choose an image."
    lock = threading.Lock()
    shape = (8, 8)
    buffers = {}
    rounds = 0


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
    for item, tag in (
            ("source_image", SOURCE_TEXTURE),
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
    return ErrorSpentConfig(
        max_side=int(dpg.get_value("max_side")),
        iterations=int(dpg.get_value("stages")),
        alpha=float(dpg.get_value("alpha")),
        passes=int(dpg.get_value("passes")),
        lam=float(dpg.get_value("lam")),
        mu=float(dpg.get_value("mu")),
        total_cells=int(dpg.get_value("foundation_cells")),
        foundation_cells=int(dpg.get_value("foundation_cells")),
        allocation_batch=int(dpg.get_value("allocation_batch")),
        error_blur=float(dpg.get_value("error_blur")),
        blue_noise_strength=float(dpg.get_value("blue_noise")),
        debit_radius=float(dpg.get_value("debit_radius")),
        detail_threshold=float(dpg.get_value("detail_threshold")),
        detail_anisotropy=float(dpg.get_value("anisotropy")),
        ownership_softness=float(dpg.get_value("softness")),
        detail_reach=float(dpg.get_value("detail_reach")))


def refresh():
    with S.lock:
        model = S.model
    if model is None:
        return
    push_texture(SOURCE_TEXTURE, model.rgb)
    push_texture(RESULT_TEXTURE, model.view(dpg.get_value("view")))
    coarse = int(np.count_nonzero(model.marks == 0))
    fine = len(model.marks) - coarse
    curve = model.allocation_psnr
    dpg.set_value(
        "metrics",
        f"{len(model.seeds)} cells: {coarse} broad + {fine} fine | "
        f"manual rounds {S.rounds} | "
        f"{model.psnr:.2f} dB | contrast retained "
        f"{100.0 * model.reconstruction.std() / max(model.rgb.std(), 1e-9):.1f}% | "
        f"{model.elapsed_ms:.0f} ms")
    if curve:
        dpg.set_value(
            "curve", "error-spending path: " +
            "  ".join(f"{value:.1f}" for value in curve))


def compute_worker():
    if S.busy or S.image is None:
        return
    S.busy = True
    S.status = "Building the broad foundation and measuring its error..."
    try:
        model = ErrorSpentDecomposition(
            S.image, config_from_ui(), initialize_only=True)
        with S.lock:
            S.model = model
            S.rounds = 0
        S.status = (
            f"{S.name}: foundation ready. Press Seed one round to add detail.")
    except Exception as exc:
        S.status = f"Build failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_compute():
    if not S.busy:
        threading.Thread(target=compute_worker, daemon=True).start()


def step_worker():
    if S.busy:
        return
    with S.lock:
        model = S.model
    if model is None:
        return
    S.busy = True
    S.status = "Measuring current error and seeding exactly one round..."
    try:
        updated = config_from_ui()
        # Recursive placement metadata stays frozen until Initialize is
        # pressed again; the manual allocation controls remain live.
        for name in (
                "allocation_batch", "error_blur", "blue_noise_strength",
                "debit_radius", "detail_threshold", "detail_anisotropy",
                "detail_reach", "ownership_softness"):
            setattr(model.cfg, name, getattr(updated, name))
        added = model.step()
        with S.lock:
            S.rounds += int(added > 0)
        S.status = (
            f"Round {S.rounds} complete: added {added} cells. "
            "Inspect it before pressing again.")
    except Exception as exc:
        S.status = f"Round failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_step():
    if not S.busy and S.model is not None:
        threading.Thread(target=step_worker, daemon=True).start()


def adopt(image, name):
    S.image = np.asarray(image, dtype=np.float64)
    S.name = name
    with S.lock:
        S.model = None
    preview = _fit_rgb(S.image, int(dpg.get_value("max_side")))
    push_texture(SOURCE_TEXTURE, preview)
    push_texture(RESULT_TEXTURE, np.full_like(preview, 0.08))
    S.status = f"{name} loaded."


def cb_gallery(sender, label):
    key = gallery.key_for_label(label)
    adopt(gallery.load(key), gallery.describe(key)["label"])


def cb_file(sender, app_data):
    selections = app_data.get("selections") or {}
    candidates = list(selections.values())
    if app_data.get("file_path_name"):
        candidates.append(app_data["file_path_name"])
    path = next((Path(p) for p in candidates if Path(p).is_file()), None)
    if path is None:
        S.status = "Could not resolve the selected image."
        return
    try:
        import matplotlib.image as mpimg
        adopt(mpimg.imread(path), path.name)
    except Exception as exc:
        S.status = f"Image load failed: {type(exc).__name__}: {exc}"


def sf(label, tag, value, lo, hi, width=245):
    dpg.add_slider_float(
        label=label, tag=tag, default_value=value,
        min_value=lo, max_value=hi, width=width, format="%.3f")


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
                callback=cb_gallery, width=410)
            dpg.add_button(
                label="Open image...",
                callback=lambda: dpg.show_item("file_dialog"))
            dpg.add_button(
                label="Initialize", callback=cb_compute,
                tag="compute_button")
            dpg.add_button(
                label="Seed one round", callback=cb_step,
                tag="step_button")
        dpg.add_text("", tag="status", wrap=1450)
        dpg.add_text("", tag="metrics")
        dpg.add_text("", tag="curve", wrap=1450)
        with dpg.collapsing_header(
                label="Blue-noise error spending", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="image size", tag="max_side",
                    default_value=192, min_value=64, max_value=384, width=245)
                dpg.add_slider_int(
                    label="even foundation", tag="foundation_cells",
                    default_value=48, min_value=4, max_value=300, width=245)
                dpg.add_slider_int(
                    label="cells this round", tag="allocation_batch",
                    default_value=64, min_value=8, max_value=256, width=245)
                sf("blue-noise spacing", "blue_noise", 1.0, 0.0, 2.0)
                sf("error debit radius", "debit_radius", 0.78, 0.2, 2.0)
            with dpg.group(horizontal=True):
                sf("error neighborhood", "error_blur", 1.1, 0.0, 5.0)
                sf("fine-cell threshold", "detail_threshold", 0.43, 0.0, 1.0)
                sf("fine anisotropy", "anisotropy", 8.0, 0.0, 20.0)
                sf("fine reach", "detail_reach", 0.82, 0.2, 2.0)
                sf("cell overlap", "softness", 9.0, 0.0, 30.0)
        with dpg.collapsing_header(
                label="Recursive placement metadata", default_open=False):
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="recursive stages", tag="stages",
                    default_value=4, min_value=1, max_value=10, width=245)
                sf("retained fraction", "alpha", 0.5, 0.0, 1.0)
                dpg.add_slider_int(
                    label="decomposition passes", tag="passes",
                    default_value=4, min_value=2, max_value=24, width=245)
                sf("lambda", "lam", 0.05, 0.01, 0.12)
                sf("mu", "mu", 40.0, 4.0, 120.0)
        with dpg.group(horizontal=True):
            dpg.add_text("Right panel")
            dpg.add_combo(
                VIEWS, default_value="Reconstruction", tag="view",
                callback=lambda: refresh(), width=240)
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image(SOURCE_TEXTURE, tag="source_image")
            with dpg.group():
                dpg.add_text("Blue noise, spending error")
                dpg.add_image(RESULT_TEXTURE, tag="result_image")


def main():
    keys = gallery.available()
    labels = gallery.labels(keys)
    default_label = (
        gallery.labels(["camera"])[0]
        if "camera" in keys else labels[0])
    dpg.create_context()
    alloc_textures(8, 8)
    build_ui(labels, default_label)
    dpg.create_viewport(
        title="BFFT — Blue Noise Spends Error",
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
        dpg.configure_item(
            "step_button", enabled=(not S.busy and S.model is not None))
        if now - last_refresh > 0.15:
            refresh()
            last_refresh = now
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
