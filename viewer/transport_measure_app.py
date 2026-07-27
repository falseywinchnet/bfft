#!/usr/bin/env python3
"""DearPyGui viewer for BFFT flow-canopy and static transport cells."""

from __future__ import annotations

import colorsys
import math
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
from bfft.vision import SingleStageDecompositionObjective, vision_backend  # noqa: E402
from bfft_flow_stage_geometry import build_flow_volume  # noqa: E402
from flow_support_measure import infer_support_measure  # noqa: E402
from flow_volume_cells import fit_population  # noqa: E402
from transport_canopy_cells import (  # noqa: E402
    canopy_geometry_views,
    evolve_canopy_population,
    fit_canopy_population,
)
from transport_measure_cells import (  # noqa: E402
    build_streaming_population,
    fit_population_cg,
    population_geometry_views,
)
from transport_voronoi import _fit_rgb, srgb_to_lab  # noqa: E402


PANEL = 650
TEX_SOURCE = "transport_measure_source_texture"
TEX_RESULT = "transport_measure_result_texture"
VIEWS = (
    "Reconstruction",
    "Reconstruction + cell outlines",
    "Cell outlines",
    "Soft site IDs",
    "Dominant site IDs",
    "Effective contributors",
    "Support dominance",
    "Reconstruction + centroids",
    "Transport support measure",
    "Support scale",
    "Anisotropy",
    "RGB error",
    "Original",
)


class State:
    def __init__(self):
        self.image = None
        self.name = "(none)"
        self.busy = False
        self.status = "Choose an image, then build the representation."
        self.lock = threading.Lock()
        self.rgb = None
        self.volume = None
        self.support = None
        self.population = None
        self.record = None
        self.diagnostic = None
        self.geometry_views = None
        self.trace = None
        self.geometry_ms = 0.0
        self.solve_ms = 0.0
        self.buffers = {}
        self.display_shape = (8, 8)


S = State()


def _display_rgb(image):
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[2] > 3:
        array = array[..., :3]
    if array.max() > 1.5:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def alloc_textures(height, width):
    S.display_shape = (height, width)
    for tag in (TEX_SOURCE, TEX_RESULT):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        if dpg.does_alias_exist(tag):
            dpg.remove_alias(tag)
        S.buffers[tag] = np.ones(height * width * 4, dtype=np.float32)
    with dpg.texture_registry():
        for tag in (TEX_SOURCE, TEX_RESULT):
            dpg.add_raw_texture(
                width,
                height,
                S.buffers[tag],
                tag=tag,
                format=dpg.mvFormat_Float_rgba,
            )
    scale = PANEL / max(height, width)
    for item, tag in (
        ("transport_measure_source_image", TEX_SOURCE),
        ("transport_measure_result_image", TEX_RESULT),
    ):
        if dpg.does_item_exist(item):
            dpg.configure_item(
                item,
                texture_tag=tag,
                width=max(1, int(width * scale)),
                height=max(1, int(height * scale)),
            )


def push_texture(tag, image):
    rgb = _display_rgb(image).astype(np.float32)
    if max(rgb.shape[:2]) > PANEL:
        rgb = _fit_rgb(rgb, PANEL).astype(np.float32)
    if rgb.shape[:2] != S.display_shape:
        alloc_textures(*rgb.shape[:2])
    buffer = S.buffers[tag]
    buffer[0::4] = rgb[..., 0].ravel()
    buffer[1::4] = rgb[..., 1].ravel()
    buffer[2::4] = rgb[..., 2].ravel()
    buffer[3::4] = 1.0
    dpg.set_value(tag, buffer)


def _colour_map(field, cyclic=False):
    value = np.asarray(field, dtype=np.float64)
    if cyclic:
        normalized = np.mod(value, 1.0)
    else:
        lo, hi = np.percentile(value[np.isfinite(value)], (1.0, 99.5))
        normalized = np.clip(
            (value - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)
    # Compact perceptual-ish blue/cyan/yellow map without a plotting runtime.
    r = np.clip(1.7 * normalized - 0.35, 0.0, 1.0)
    g = np.clip(1.8 - 2.4 * np.abs(normalized - 0.58), 0.0, 1.0)
    b = np.clip(1.15 - 1.45 * normalized, 0.0, 1.0)
    return np.stack([r, g, b], axis=2)


def _centroid_overlay(rgb, population):
    result = np.asarray(rgb, dtype=np.float64).copy()
    stages = np.asarray(population.stage, dtype=np.float64)
    maximum = max(float(np.max(stages)), 1.0)
    for (x, y), stage in zip(population.centers, stages):
        hue = float(stage / maximum)
        colour = np.asarray(colorsys.hsv_to_rgb(hue, 0.82, 1.0))
        xi = int(np.clip(round(x), 0, result.shape[1] - 1))
        yi = int(np.clip(round(y), 0, result.shape[0] - 1))
        y0, y1 = max(0, yi - 1), min(result.shape[0], yi + 2)
        x0, x1 = max(0, xi - 1), min(result.shape[1], xi + 2)
        result[y0:y1, x0:x1] = (
            0.22 * result[y0:y1, x0:x1] + 0.78 * colour)
    return result


def _cell_outline_overlay(rgb, outlines):
    result = np.asarray(rgb, dtype=np.float64).copy()
    lines = np.asarray(outlines, dtype=np.float64)
    mask = np.max(lines, axis=2) > 0.0
    result[mask] = 0.16 * result[mask] + 0.84 * lines[mask]
    return result


def current_view():
    with S.lock:
        rgb = S.rgb
        volume = S.volume
        support = S.support
        population = S.population
        record = S.record
        geometry_views = S.geometry_views
    if rgb is None:
        return np.full((8, 8, 3), 0.08)
    view = dpg.get_value("transport_measure_view")
    if view == "Original" or record is None:
        return rgb
    reconstruction = record["rgb"]
    if view == "Reconstruction":
        return reconstruction
    if view == "Reconstruction + cell outlines":
        return _cell_outline_overlay(
            reconstruction, geometry_views["cell_outlines"])
    if view == "Cell outlines":
        return geometry_views["cell_outlines"]
    if view == "Soft site IDs":
        return geometry_views["soft_site_ids"]
    if view == "Dominant site IDs":
        return geometry_views["dominant_site_ids"]
    if view == "Effective contributors":
        return _colour_map(geometry_views["effective_contributors"])
    if view == "Support dominance":
        return _colour_map(geometry_views["dominance"])
    if view == "Reconstruction + centroids":
        return _centroid_overlay(reconstruction, population)
    if view == "Transport support measure":
        return _colour_map(support["envelope"][-1])
    if view == "Support scale":
        return _colour_map(support["scale_mean"])
    if view == "Anisotropy":
        xx = np.asarray(support["precision_xx"], dtype=np.float64)
        xy = np.asarray(support["precision_xy"], dtype=np.float64)
        yy = np.asarray(support["precision_yy"], dtype=np.float64)
        trace = xx + yy
        disc = np.hypot(xx - yy, 2.0 * xy)
        ratio = np.sqrt(
            np.maximum(trace + disc, 1e-20)
            / np.maximum(trace - disc, 1e-20))
        return _colour_map(np.log1p(ratio))
    if view == "RGB error":
        error = np.sqrt(np.mean((rgb - reconstruction) ** 2, axis=2))
        return _colour_map(error)
    return reconstruction


def refresh():
    with S.lock:
        rgb = S.rgb
        population = S.population
        record = S.record
        diagnostic = S.diagnostic
    if rgb is None:
        return
    push_texture(TEX_SOURCE, rgb)
    push_texture(TEX_RESULT, current_view())
    if record is None:
        dpg.set_value("transport_measure_metrics", "Not built yet.")
        return
    dpg.set_value(
        "transport_measure_metrics",
        f"{len(population.centers)} inferred supports | "
        f"PSNR {record['psnr']:.2f} dB | "
        f"cartoon MSE {record['cartoon_mse']:.3e} | "
        f"texture MSE {record['texture_mse']:.3e} | "
        f"objective {record['objective']:.3e} | "
        f"coverage {100*(1-diagnostic.get('uncovered_fraction', 0)):.3f}% | "
        f"geometry {S.geometry_ms:.0f} ms | solve {S.solve_ms:.0f} ms | "
        f"{vision_backend()}",
    )


def _work_side():
    return (
        0
        if dpg.get_value("transport_measure_full_resolution")
        else int(dpg.get_value("transport_measure_side"))
    )


def build_worker():
    if S.busy or S.image is None:
        return
    S.busy = True
    S.status = "Reading the complete BFFT transport stack..."
    try:
        method = dpg.get_value("transport_measure_method")
        work_side = _work_side()
        if (
            method == "Flow canopy diagram"
            and max(S.image.shape[:2]) > 512
            and (work_side == 0 or work_side > 512)
        ):
            raise ValueError(
                "flow canopy is currently a research-resolution trace; "
                "choose work max side <= 512, or use Static overlap control "
                "for the streaming HD path")
        rgb = _fit_rgb(S.image, work_side)
        passes = int(dpg.get_value("transport_measure_passes"))
        flow_sweeps = int(dpg.get_value("transport_measure_flow_sweeps"))
        overlap = float(dpg.get_value("transport_measure_overlap"))
        horizon = float(dpg.get_value("transport_measure_horizon"))
        temperature = float(
            dpg.get_value("transport_measure_temperature"))
        started = time.perf_counter()
        trace = None
        volume = None
        if method == "Flow canopy diagram":
            S.status = "Building pass geometry for fixed-population descent..."
            volume = build_flow_volume(
                rgb, passes=passes, flow_sweeps=flow_sweeps)
            support = infer_support_measure(
                volume, max_support_fraction=horizon)
            population, canopy_weight, canopy_samples, trace = (
                evolve_canopy_population(
                    volume,
                    support,
                    overlap=overlap,
                    sharpness_start=1.5,
                    sharpness_end=temperature,
                    initialization="density",
                ))
            geometry_ms = (time.perf_counter() - started) * 1000.0
            S.status = (
                f"Fitting {len(population.centers)} evolved canopies...")
            objective = SingleStageDecompositionObjective(
                rgb, passes=passes)
            target_lab = srgb_to_lab(rgb)
            solve_started = time.perf_counter()
            record, _, diagnostic = fit_canopy_population(
                population,
                canopy_samples,
                canopy_weight,
                target_lab,
                objective,
            )
            solve_ms = (time.perf_counter() - solve_started) * 1000.0
            geometry_views = canopy_geometry_views(
                population,
                canopy_samples,
                canopy_weight,
                rgb.shape[0],
                rgb.shape[1],
            )
        else:
            population, support, _ = build_streaming_population(
                rgb,
                passes=passes,
                flow_sweeps=flow_sweeps,
                max_support_fraction=horizon,
                overlap=overlap,
            )
            geometry_ms = (time.perf_counter() - started) * 1000.0
            S.status = (
                f"Fitting {len(population.centers)} inferred supports...")
            objective = SingleStageDecompositionObjective(
                rgb, passes=passes)
            target_lab = srgb_to_lab(rgb)
            solver = dpg.get_value("transport_measure_solver")
            solve_started = time.perf_counter()
            if solver == "Exact sparse":
                record, _, diagnostic = fit_population(
                    population, target_lab, objective,
                    temperature=temperature)
            else:
                iterations = 40 if solver == "Fast HD (40)" else 80
                record, _, diagnostic = fit_population_cg(
                    population,
                    target_lab,
                    objective,
                    iterations=iterations,
                    temperature=temperature,
                )
            solve_ms = (time.perf_counter() - solve_started) * 1000.0
            S.status = "Rasterizing the exact support partition..."
            geometry_views = population_geometry_views(
                population, rgb.shape[0], rgb.shape[1],
                temperature=temperature)
        with S.lock:
            S.rgb = rgb
            S.volume = volume
            S.support = support
            S.population = population
            S.record = record
            S.diagnostic = diagnostic
            S.geometry_views = geometry_views
            S.trace = trace
            S.geometry_ms = geometry_ms
            S.solve_ms = solve_ms
        S.status = (
            f"{S.name}: {method} complete. "
            f"Count {support['transported_count']:.1f} → "
            f"{len(population.centers)} fixed sites.")
    except Exception as exc:
        S.status = f"Build failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_build():
    if not S.busy:
        threading.Thread(target=build_worker, daemon=True).start()


def adopt(image, name):
    S.image = np.asarray(image, dtype=np.float64)
    S.name = name
    with S.lock:
        S.rgb = _fit_rgb(S.image, _work_side())
        S.volume = None
        S.support = None
        S.population = None
        S.record = None
        S.diagnostic = None
        S.geometry_views = None
        S.trace = None
    S.status = f"{name} loaded. Press Build representation."
    push_texture(TEX_SOURCE, S.rgb)
    push_texture(TEX_RESULT, np.full_like(S.rgb, 0.08))


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
    path = next((Path(value) for value in candidates
                 if Path(value).is_file()), None)
    if path is None:
        S.status = "Could not resolve the selected image."
        return
    try:
        import matplotlib.image as mpimg

        adopt(mpimg.imread(path), path.name)
    except Exception as exc:
        S.status = f"Image load failed: {type(exc).__name__}: {exc}"


def build_ui(labels, default_label):
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=cb_file,
        tag="transport_measure_file_dialog",
        width=900,
        height=520,
    ):
        dpg.add_file_extension(
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
            "{.png,.jpg,.jpeg,.tif,.tiff,.bmp}")
        dpg.add_file_extension(".*")

    with dpg.window(tag="transport_measure_root"):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                labels,
                default_value=default_label,
                tag="transport_measure_gallery",
                width=390,
                callback=cb_gallery,
            )
            dpg.add_button(
                label="Load image...",
                callback=lambda: dpg.show_item(
                    "transport_measure_file_dialog"),
            )
            dpg.add_button(
                label="Build representation",
                callback=cb_build,
                tag="transport_measure_build",
            )
        dpg.add_text("", tag="transport_measure_status", wrap=1500)
        dpg.add_text("", tag="transport_measure_metrics", wrap=1500)
        dpg.add_separator()

        with dpg.collapsing_header(
            label="Transport geometry", default_open=True
        ):
            dpg.add_combo(
                ("Flow canopy diagram", "Static overlap control"),
                default_value="Flow canopy diagram",
                label="representation",
                tag="transport_measure_method",
                width=330,
            )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="process every source pixel (HD/full resolution)",
                    tag="transport_measure_full_resolution",
                    default_value=False,
                )
                dpg.add_slider_int(
                    label="work max side",
                    tag="transport_measure_side",
                    default_value=128,
                    min_value=96,
                    max_value=2160,
                    width=340,
                )
                dpg.add_slider_int(
                    label="BFFT passes",
                    tag="transport_measure_passes",
                    default_value=24,
                    min_value=4,
                    max_value=64,
                    width=260,
                )
                dpg.add_slider_int(
                    label="guide projection sweeps",
                    tag="transport_measure_flow_sweeps",
                    default_value=4,
                    min_value=1,
                    max_value=32,
                    width=280,
                )
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="broad support horizon",
                    tag="transport_measure_horizon",
                    default_value=0.18,
                    min_value=0.08,
                    max_value=0.4,
                    width=320,
                    format="%.3f",
                )
                dpg.add_slider_float(
                    label="smooth aperture coverage",
                    tag="transport_measure_overlap",
                    default_value=8.0,
                    min_value=2.0,
                    max_value=10.0,
                    width=320,
                )

        with dpg.collapsing_header(
            label="Reconstruction finish", default_open=True
        ):
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    ("Fast HD (40)", "High quality (80)", "Exact sparse"),
                    default_value="Fast HD (40)",
                    label="solver",
                    tag="transport_measure_solver",
                    width=260,
                )
                dpg.add_slider_float(
                    label="final canopy sharpness",
                    tag="transport_measure_temperature",
                    default_value=10.0,
                    min_value=0.5,
                    max_value=16.0,
                    width=300,
                    format="%.2f",
                )
                dpg.add_text(
                    "Canopy mode keeps one population through the BFFT flow. "
                    "The solver selector applies to the static control.")

        with dpg.group(horizontal=True):
            dpg.add_text("Right panel")
            dpg.add_combo(
                VIEWS,
                default_value=VIEWS[0],
                tag="transport_measure_view",
                callback=lambda: refresh(),
                width=300,
            )

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image(
                    TEX_SOURCE, tag="transport_measure_source_image")
            with dpg.group():
                dpg.add_text("BFFT transport-cell representation")
                dpg.add_image(
                    TEX_RESULT, tag="transport_measure_result_image")


def main():
    keys = gallery.available()
    labels = gallery.labels(keys)
    default_key = "pikachu" if "pikachu" in keys else keys[0]
    default_label = labels[keys.index(default_key)]
    dpg.create_context()
    alloc_textures(8, 8)
    build_ui(labels, default_label)
    dpg.create_viewport(
        title="BFFT Vision — Flow Canopy Transport Cells",
        width=1500,
        height=980,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("transport_measure_root", True)
    cb_gallery(None, default_label)
    cb_build()
    last_refresh = 0.0
    while dpg.is_dearpygui_running():
        dpg.set_value("transport_measure_status", S.status)
        dpg.configure_item("transport_measure_build", enabled=not S.busy)
        now = time.perf_counter()
        if now - last_refresh > 0.15:
            refresh()
            last_refresh = now
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
