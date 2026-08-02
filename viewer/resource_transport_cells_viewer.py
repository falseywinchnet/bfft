#!/usr/bin/env python3
"""Interactive copy of the viewer for owner-free resource transport cells.

Run:
    .venv/bin/python viewer/resource_transport_cells_viewer.py

This viewer exposes the validated experiment without changing the canonical
viewer.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import dearpygui.dearpygui as dpg  # noqa: E402
import gallery  # noqa: E402
from resource_transport_cells import (  # noqa: E402
    ResourceConfig, ResourceTransportCells,
)
from transport_voronoi import _fit_rgb  # noqa: E402


TEX_SOURCE = "rtc_source"
TEX_RESULT = "rtc_result"
PANEL = 660
VIEWS = [
    "Reconstruction + cells",
    "Reconstruction",
    "Remaining resource",
    "Smooth occupancy",
    "Germination memory",
]


class State:
    def __init__(self):
        self.image = None
        self.name = "(none)"
        self.model = None
        self.busy = False
        self.run = False
        self.status = "Choose an image and initialize."
        self.lock = threading.Lock()
        self.buffers = {}
        self.texture_shape = (8, 8)
        self.refresh_key = None
        self.decomposition_metrics = None


S = State()


def alloc_textures(h: int, w: int):
    S.texture_shape = (h, w)
    for tag in (TEX_SOURCE, TEX_RESULT):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        if dpg.does_alias_exist(tag):
            dpg.remove_alias(tag)
        S.buffers[tag] = np.ones(h * w * 4, dtype=np.float32)
    with dpg.texture_registry():
        dpg.add_raw_texture(
            w, h, S.buffers[TEX_SOURCE], tag=TEX_SOURCE,
            format=dpg.mvFormat_Float_rgba)
        dpg.add_raw_texture(
            w, h, S.buffers[TEX_RESULT], tag=TEX_RESULT,
            format=dpg.mvFormat_Float_rgba)
    scale = PANEL / max(h, w)
    for item, texture in (
        ("source_image", TEX_SOURCE), ("result_image", TEX_RESULT)
    ):
        if dpg.does_item_exist(item):
            dpg.configure_item(
                item, texture_tag=texture,
                width=max(1, int(w * scale)),
                height=max(1, int(h * scale)))


def push_texture(tag: str, rgb):
    image = np.asarray(rgb, dtype=np.float32)
    if max(image.shape[:2]) > PANEL:
        image = np.asarray(_fit_rgb(image, PANEL), dtype=np.float32)
    if image.shape[:2] != S.texture_shape:
        alloc_textures(*image.shape[:2])
    rgba = S.buffers[tag]
    rgba[0::4] = image[..., 0].ravel()
    rgba[1::4] = image[..., 1].ravel()
    rgba[2::4] = image[..., 2].ravel()
    rgba[3::4] = 1.0
    dpg.set_value(tag, rgba)


def heatmap(field):
    value = np.asarray(field, dtype=np.float64)
    scale = max(float(np.percentile(value, 99.0)), 1e-12)
    x = np.clip(value / scale, 0.0, 1.0)
    # Compact perceptual heat ramp: indigo -> magenta -> amber.
    red = np.clip(1.8 * x - 0.25, 0.0, 1.0)
    green = np.clip(2.2 * x - 1.15, 0.0, 1.0)
    blue = np.clip(0.28 + 1.8 * x - 2.0 * x * x, 0.0, 1.0)
    return np.stack([red, green, blue], axis=2)


def cell_overlay(model):
    image = model.rgb_reconstruction.copy()
    alpha = 1.0 / (1.0 + np.exp(-model.crystallinity_logit))
    radius = max(1, int(round(max(model.h, model.w) / 420.0)))
    for (cx, cy), crystal in zip(model.centers, alpha):
        x = int(np.clip(round(cx), 0, model.w - 1))
        y = int(np.clip(round(cy), 0, model.h - 1))
        y0, y1 = max(0, y - radius), min(model.h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(model.w, x + radius + 1)
        color = np.array(
            [1.0, 0.25 + 0.65 * (1.0 - crystal), 0.05]
            if crystal > 0.08 else
            [0.0, 0.85, 1.0])
        image[y0:y1, x0:x1] = (
            0.18 * image[y0:y1, x0:x1] + 0.82 * color)
    return image


def config_from_ui():
    return ResourceConfig(
        max_side=(
            0 if dpg.get_value("full_resolution")
            else int(dpg.get_value("max_side"))),
        cells=int(dpg.get_value("initial_cells")),
        passes=int(dpg.get_value("passes")),
        flow_sweeps=int(dpg.get_value("flow_sweeps")),
        lam=float(dpg.get_value("lam")),
        mu=float(dpg.get_value("mu")),
        initial_overlap=float(dpg.get_value("initial_overlap")),
        occupancy_floor=float(dpg.get_value("occupancy_floor")),
        kernel_family=str(dpg.get_value("kernel_family")),
        kernel_power=float(dpg.get_value("kernel_power")),
        adaptive_hardness=bool(dpg.get_value("adaptive_hardness")),
        hardness_rate=float(dpg.get_value("hardness_rate")),
        adaptive_crystallinity=bool(
            dpg.get_value("adaptive_crystallinity")),
        initial_crystallinity=float(
            dpg.get_value("initial_crystallinity")),
        crystallinity_rate=float(dpg.get_value("crystallinity_rate")),
        glass_mode=str(dpg.get_value("glass_mode")),
        glass_strength=0.0,
        glass_shape_strength=float(
            dpg.get_value("glass_shape_strength")),
        color_rate=float(dpg.get_value("color_rate")),
        center_rate=float(dpg.get_value("center_rate")),
        area_rate=float(dpg.get_value("area_rate")),
        shape_rate=float(dpg.get_value("shape_rate")),
        max_ratio=float(dpg.get_value("max_ratio")),
        min_area_fraction=float(dpg.get_value("min_area_fraction")),
        germination=bool(dpg.get_value("germination")),
        germination_threshold=float(
            dpg.get_value("germination_threshold")),
        germination_decay=float(dpg.get_value("germination_decay")),
        germination_separation=float(
            dpg.get_value("germination_separation")),
        germination_initial_scale=float(
            dpg.get_value("germination_initial_scale")),
        germination_inhibition=float(
            dpg.get_value("germination_inhibition")),
        shared_residual_credit=bool(
            dpg.get_value("shared_residual_credit")),
        shared_geometry_credit=bool(
            dpg.get_value("shared_geometry_credit")),
        conserved_cell_scale=bool(
            dpg.get_value("conserved_cell_scale")),
    )


def update_live_config(model):
    current = config_from_ui()
    names = (
        "kernel_family", "adaptive_hardness", "hardness_rate",
        "adaptive_crystallinity", "crystallinity_rate",
        "glass_mode", "glass_shape_strength",
        "color_rate", "center_rate", "area_rate", "shape_rate",
        "max_ratio", "min_area_fraction",
        "germination", "germination_threshold", "germination_decay",
        "germination_separation", "germination_inhibition",
        "shared_residual_credit", "shared_geometry_credit",
        "conserved_cell_scale",
    )
    for name in names:
        setattr(model.cfg, name, getattr(current, name))


def refresh(force=False):
    with S.lock:
        model = S.model
    if model is None or S.busy:
        return
    view = dpg.get_value("view")
    key = (id(model), model.iteration, view)
    if not force and key == S.refresh_key:
        return
    push_texture(TEX_SOURCE, model.rgb)
    if view == "Reconstruction + cells":
        result = cell_overlay(model)
    elif view == "Reconstruction":
        result = model.rgb_reconstruction
    elif view == "Remaining resource":
        result = heatmap(model.error)
    elif view == "Smooth occupancy":
        result = heatmap(
            model.occupancy - model.cfg.occupancy_floor)
    else:
        result = heatmap(model.germination_field)
    push_texture(TEX_RESULT, result)
    latest = model.trace[-1] if model.trace else {}
    alpha = 1.0 / (1.0 + np.exp(-model.crystallinity_logit))
    structural = ""
    if S.decomposition_metrics:
        metric = S.decomposition_metrics
        structural = (
            f" | C {metric['cartoon_mse']:.2e} "
            f"T {metric['texture_mse']:.2e} "
            f"objective {metric['objective']:.2e}")
    dpg.set_value(
        "metrics",
        f"{model.w}x{model.h} | cells {len(model.centers)} | "
        f"round {model.iteration} | PSNR {model.psnr:.2f} dB | "
        f"last {model.last_ms:.0f} ms | "
        f"visits/pixel {latest.get('visits_per_pixel', 0.0):.1f} | "
        f"births {latest.get('births', 0)} | "
        f"crystal {float(np.mean(alpha)):.3f}/"
        f"{float(np.max(alpha)):.3f}{structural}")
    S.refresh_key = key


def adopt(image, name):
    S.run = False
    S.image = image
    S.name = name
    S.model = None
    S.decomposition_metrics = None
    S.refresh_key = None
    preview = _fit_rgb(image, PANEL)
    push_texture(TEX_SOURCE, preview)
    push_texture(TEX_RESULT, np.full_like(preview, 0.06))
    S.status = f"{name}: ready. Press Initialize."


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
        S.status = "Could not resolve the selected image."
        return
    try:
        from skimage.io import imread
        adopt(imread(path), path.name)
    except Exception as exc:
        S.status = f"Image load failed: {type(exc).__name__}: {exc}"


def initialize_worker():
    if S.busy or S.image is None:
        return
    S.busy = True
    S.run = False
    S.status = "Initializing BFFT glass and diffuse/crystalline cells..."
    try:
        model = ResourceTransportCells(S.image, config_from_ui())
        with S.lock:
            S.model = model
        S.decomposition_metrics = None
        S.refresh_key = None
        S.status = (
            f"{S.name}: initialized {model.w}x{model.h} with "
            f"{len(model.centers)} owner-free cells.")
    except Exception as exc:
        S.status = f"Initialize failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_initialize():
    if S.image is None:
        S.status = "Choose an image first."
    elif not S.busy:
        threading.Thread(target=initialize_worker, daemon=True).start()


def step_worker():
    if S.busy:
        return
    with S.lock:
        model = S.model
    if model is None:
        S.status = "Initialize first."
        return
    S.busy = True
    try:
        update_live_config(model)
        report = model.step()
        S.decomposition_metrics = None
        S.refresh_key = None
        S.status = (
            f"Round {model.iteration}: {model.psnr:.2f} dB, "
            f"{len(model.centers)} cells (+{report['births']}), "
            f"{model.last_ms:.0f} ms.")
    except Exception as exc:
        S.status = f"Round failed: {type(exc).__name__}: {exc}"
        S.run = False
    finally:
        S.busy = False


def cb_step():
    if not S.busy:
        threading.Thread(target=step_worker, daemon=True).start()


def cb_run():
    if S.model is None:
        cb_initialize()
        return
    S.run = not S.run
    S.status = "Running..." if S.run else "Paused."


def score_worker():
    if S.busy:
        return
    with S.lock:
        model = S.model
    if model is None:
        S.status = "Initialize first."
        return
    S.busy = True
    S.run = False
    S.status = "Computing cached-target cartoon and texture objective..."
    try:
        started = time.perf_counter()
        S.decomposition_metrics = model.decomposition_metrics()
        elapsed = 1000.0 * (time.perf_counter() - started)
        S.refresh_key = None
        S.status = f"Structural objective measured in {elapsed:.0f} ms."
    except Exception as exc:
        S.status = f"Scoring failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_score():
    if not S.busy:
        threading.Thread(target=score_worker, daemon=True).start()


def cb_view():
    S.refresh_key = None
    refresh(force=True)


def build_ui(labels, default_label):
    with dpg.file_dialog(
        directory_selector=False, show=False, callback=cb_file,
        tag="file_dialog", width=900, height=520
    ):
        dpg.add_file_extension(
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
            "{.png,.jpg,.jpeg,.tif,.tiff,.bmp}")
        dpg.add_file_extension(".*")

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                labels, default_value=default_label, tag="gallery",
                width=380, callback=cb_gallery)
            dpg.add_button(
                label="Load image...",
                callback=lambda: dpg.show_item("file_dialog"))
            dpg.add_button(
                label="Initialize / reset", callback=cb_initialize,
                tag="initialize_button")
            dpg.add_button(
                label="One transport round", callback=cb_step,
                tag="step_button")
            dpg.add_button(
                label="Run continuously", callback=cb_run,
                tag="run_button")
            dpg.add_button(
                label="Measure C/T objective", callback=cb_score,
                tag="score_button")
        dpg.add_text("", tag="status", wrap=1500)
        dpg.add_text("", tag="metrics")
        dpg.add_separator()

        with dpg.collapsing_header(
            label="Image and initial chemistry", default_open=True
        ):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="full resolution", tag="full_resolution",
                    default_value=False)
                dpg.add_slider_int(
                    label="work max side", tag="max_side",
                    default_value=256, min_value=64, max_value=2160,
                    width=300)
                dpg.add_slider_int(
                    label="initial cells", tag="initial_cells",
                    default_value=180, min_value=20, max_value=5000,
                    width=280)
                dpg.add_slider_float(
                    label="initial overlap", tag="initial_overlap",
                    default_value=5.0, min_value=0.5, max_value=12.0,
                    width=280)
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="BFFT passes", tag="passes",
                    default_value=12, min_value=4, max_value=48, width=260)
                dpg.add_slider_int(
                    label="flow sweeps", tag="flow_sweeps",
                    default_value=32, min_value=8, max_value=160, width=260)
                dpg.add_slider_float(
                    label="lambda", tag="lam", default_value=0.054,
                    min_value=0.01, max_value=0.12, format="%.3f",
                    width=260)
                dpg.add_slider_float(
                    label="mu", tag="mu", default_value=0.02,
                    min_value=0.002, max_value=0.12, format="%.3f",
                    width=260)

        with dpg.collapsing_header(
            label="Local support state", default_open=True
        ):
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    ["mixture", "power", "logistic"],
                    default_value="mixture", label="support family",
                    tag="kernel_family", width=170)
                dpg.add_slider_float(
                    label="initial concentration", tag="kernel_power",
                    default_value=4.0, min_value=1.5, max_value=16.0,
                    width=280)
                dpg.add_checkbox(
                    label="learn concentration",
                    tag="adaptive_hardness", default_value=True)
                dpg.add_slider_float(
                    label="concentration rate", tag="hardness_rate",
                    default_value=0.10, min_value=0.0, max_value=0.5,
                    width=260)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="learn crystallinity",
                    tag="adaptive_crystallinity", default_value=True)
                dpg.add_slider_float(
                    label="initial crystal fraction",
                    tag="initial_crystallinity", default_value=0.05,
                    min_value=0.0, max_value=1.0, width=290)
                dpg.add_slider_float(
                    label="crystal learning rate",
                    tag="crystallinity_rate", default_value=0.18,
                    min_value=0.0, max_value=0.8, width=290)
                dpg.add_slider_float(
                    label="maximum axis ratio", tag="max_ratio",
                    default_value=14.0, min_value=1.0, max_value=40.0,
                    width=270)
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    ["fixed", "off", "discrepancy"],
                    default_value="fixed", label="BFFT glass",
                    tag="glass_mode", width=160)
                dpg.add_slider_float(
                    label="glass axis conductivity",
                    tag="glass_shape_strength", default_value=0.5,
                    min_value=0.0, max_value=3.0, width=320)
                dpg.add_slider_float(
                    label="minimum area fraction",
                    tag="min_area_fraction", default_value=0.005,
                    min_value=0.001, max_value=0.20, format="%.3f",
                    width=300)

        with dpg.collapsing_header(
            label="Residual germination", default_open=True
        ):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="germination", tag="germination",
                    default_value=True)
                dpg.add_slider_float(
                    label="activation threshold",
                    tag="germination_threshold", default_value=0.25,
                    min_value=0.05, max_value=4.0, width=300)
                dpg.add_slider_float(
                    label="memory decay", tag="germination_decay",
                    default_value=0.72, min_value=0.0, max_value=0.98,
                    width=280)
                dpg.add_slider_float(
                    label="lateral separation",
                    tag="germination_separation", default_value=0.34,
                    min_value=0.05, max_value=1.0, width=300)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="new germ radius",
                    tag="germination_initial_scale", default_value=0.15,
                    min_value=0.02, max_value=0.6, width=290)
                dpg.add_slider_float(
                    label="extra site inhibitor (normally zero)",
                    tag="germination_inhibition", default_value=0.0,
                    min_value=0.0, max_value=5.0, width=350)
                dpg.add_checkbox(
                    label="intrinsic cell scale",
                    tag="conserved_cell_scale", default_value=True)

        with dpg.collapsing_header(
            label="Receiver descent", default_open=True
        ):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="conserve shared color residual",
                    tag="shared_residual_credit", default_value=True)
                dpg.add_checkbox(
                    label="partition geometry too (rejected control)",
                    tag="shared_geometry_credit", default_value=False)
                dpg.add_slider_float(
                    label="occupancy floor", tag="occupancy_floor",
                    default_value=0.002, min_value=0.0001,
                    max_value=0.02, format="%.4f", width=270)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="color rate", tag="color_rate",
                    default_value=0.70, min_value=0.0, max_value=1.2,
                    width=260)
                dpg.add_slider_float(
                    label="center rate", tag="center_rate",
                    default_value=0.22, min_value=0.0, max_value=0.8,
                    width=260)
                dpg.add_slider_float(
                    label="area rate", tag="area_rate",
                    default_value=0.12, min_value=0.0, max_value=0.6,
                    width=260)
                dpg.add_slider_float(
                    label="shape rate", tag="shape_rate",
                    default_value=0.24, min_value=0.0, max_value=0.8,
                    width=260)

        with dpg.group(horizontal=True):
            dpg.add_text("Right panel")
            dpg.add_combo(
                VIEWS, default_value=VIEWS[0], tag="view",
                callback=cb_view, width=260)

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image(TEX_SOURCE, tag="source_image")
            with dpg.group():
                dpg.add_text("Owner-free resource transport cells")
                dpg.add_image(TEX_RESULT, tag="result_image")


def main():
    keys = gallery.available()
    labels = gallery.labels(keys)
    default_key = "pikachu" if "pikachu" in keys else keys[0]
    default_label = labels[keys.index(default_key)]
    dpg.create_context()
    alloc_textures(8, 8)
    build_ui(labels, default_label)
    dpg.create_viewport(
        title="BFFT Vision — Resource Transport Cell Experiment",
        width=1520, height=1040)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)
    cb_gallery(None, default_label)
    cb_initialize()

    last_refresh = 0.0
    while dpg.is_dearpygui_running():
        dpg.set_value("status", S.status)
        dpg.configure_item(
            "run_button",
            label="Pause" if S.run else "Run continuously")
        for item in (
            "initialize_button", "step_button", "score_button"
        ):
            dpg.configure_item(item, enabled=not S.busy)
        if S.run and not S.busy:
            threading.Thread(target=step_worker, daemon=True).start()
        now = time.perf_counter()
        if now - last_refresh > 0.12:
            refresh()
            last_refresh = now
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
