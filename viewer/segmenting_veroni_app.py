#!/usr/bin/env python3
"""Canonical viewer for one-decomposition transport-cell segmentation."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "viewer", ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import dearpygui.dearpygui as dpg  # noqa: E402
import gallery  # noqa: E402
from port_needed import SegmentingConfig, build_segmenting_representation  # noqa: E402
from transport_voronoi import _fit_rgb  # noqa: E402

PANEL = 650
SOURCE = "segmenting_source_texture"
RESULT = "segmenting_result_texture"
VIEWS = (
    "Reconstruction",
    "Reconstruction + cell boundaries",
    "Site IDs",
    "Site IDs + boundaries",
    "Cell boundaries",
    "Reconstruction + sites",
    "Transport support measure",
    "Cartoon",
    "Texture",
    "Transport glass",
    "Metric anisotropy",
    "Residual energy",
    "Reverse residual flow",
    "Refinement demand",
    "Residual pressure density",
    "Residual metric pressure",
    "Power reach",
    "Site motion",
    "Characteristic force",
    "Topology clearance",
    "Trust-limited step",
    "RGB error",
    "Original",
)


class State:
    def __init__(self):
        self.image = None
        self.name = "(none)"
        self.rgb = None
        self.result = None
        self.busy = False
        self.status = "Choose an image, then build the representation."
        self.lock = threading.Lock()
        self.buffers = {}
        self.display_shape = (8, 8)


S = State()


def _rgb(image):
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    a = a[..., :3]
    if a.max(initial=0.0) > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def alloc_textures(height, width):
    S.display_shape = (height, width)
    for tag in (SOURCE, RESULT):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        if dpg.does_alias_exist(tag):
            dpg.remove_alias(tag)
        S.buffers[tag] = np.ones(height * width * 4, dtype=np.float32)
    with dpg.texture_registry():
        for tag in (SOURCE, RESULT):
            dpg.add_raw_texture(
                width, height, S.buffers[tag], tag=tag,
                format=dpg.mvFormat_Float_rgba)
    scale = PANEL / max(height, width)
    for item, tag in (
        ("segmenting_source_image", SOURCE),
        ("segmenting_result_image", RESULT),
    ):
        if dpg.does_item_exist(item):
            dpg.configure_item(
                item, texture_tag=tag,
                width=max(1, int(width * scale)),
                height=max(1, int(height * scale)))


def push_texture(tag, image):
    image = _rgb(image).astype(np.float32)
    if max(image.shape[:2]) > PANEL:
        image = _fit_rgb(image, PANEL).astype(np.float32)
    if image.shape[:2] != S.display_shape:
        alloc_textures(*image.shape[:2])
    buffer = S.buffers[tag]
    buffer[0::4], buffer[1::4], buffer[2::4] = (
        image[..., 0].ravel(), image[..., 1].ravel(), image[..., 2].ravel())
    buffer[3::4] = 1.0
    dpg.set_value(tag, buffer)


def colour_map(field):
    value = np.asarray(field, dtype=np.float64)
    finite = value[np.isfinite(value)]
    if not finite.size:
        return np.zeros(value.shape + (3,))
    low, high = np.percentile(finite, (1.0, 99.5))
    z = np.clip((value - low) / max(float(high - low), 1e-12), 0.0, 1.0)
    return np.stack((
        np.clip(1.7 * z - 0.35, 0.0, 1.0),
        np.clip(1.8 - 2.4 * np.abs(z - 0.58), 0.0, 1.0),
        np.clip(1.15 - 1.45 * z, 0.0, 1.0)), axis=2)


def boundaries(labels):
    labels = np.asarray(labels)
    mask = np.zeros(labels.shape, dtype=bool)
    mask[1:] |= labels[1:] != labels[:-1]
    mask[:-1] |= labels[:-1] != labels[1:]
    mask[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return mask


def overlay_boundaries(image, labels):
    out = np.asarray(image, dtype=np.float64).copy()
    mask = boundaries(labels)
    out[mask] = 0.12 * out[mask] + 0.88 * np.array([0.05, 1.0, 0.95])
    return out


def site_ids(labels):
    count = int(np.max(labels)) + 1
    index = np.arange(count, dtype=np.uint32)
    value = index * np.uint32(747796405) + np.uint32(2891336453)
    value = ((value >> ((value >> 28) + 4)) ^ value) * np.uint32(277803737)
    value = (value >> 22) ^ value
    colours = np.column_stack((
        value & 255, (value >> 8) & 255, (value >> 16) & 255,
    )).astype(np.float64) / 255.0
    return (0.12 + 0.86 * colours)[labels]


def overlay_sites(image, centers):
    out = np.asarray(image, dtype=np.float64).copy()
    height, width = out.shape[:2]
    for x, y in np.asarray(centers):
        xi = int(np.clip(round(x * width - 0.5), 0, width - 1))
        yi = int(np.clip(round(y * height - 0.5), 0, height - 1))
        out[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3] = (1.0, 0.1, 0.08)
    return out


def overlay_motion(image, initial_centers, final_centers):
    out = np.asarray(image, dtype=np.float64).copy()
    height, width = out.shape[:2]
    for start, stop in zip(initial_centers, final_centers):
        x0, y0 = start[0] * width - 0.5, start[1] * height - 0.5
        x1, y1 = stop[0] * width - 0.5, stop[1] * height - 0.5
        samples = max(int(np.ceil(np.hypot(x1 - x0, y1 - y0))), 1)
        xs = np.clip(
            np.rint(np.linspace(x0, x1, samples + 1)).astype(np.int32),
            0, width - 1)
        ys = np.clip(
            np.rint(np.linspace(y0, y1, samples + 1)).astype(np.int32),
            0, height - 1)
        out[ys, xs] = (1.0, 0.15, 0.05)
        out[int(np.clip(round(y1), 0, height - 1)),
            int(np.clip(round(x1), 0, width - 1))] = (0.05, 1.0, 0.95)
    return out


def current_view():
    with S.lock:
        rgb, result = S.rgb, S.result
    if rgb is None:
        return np.full((8, 8, 3), 0.08)
    view = dpg.get_value("segmenting_view")
    if view == "Original" or result is None:
        return rgb
    labels, geometry = result["labels"], result["geometry"]
    reconstruction = result["record"]["rgb"]
    if view == "Reconstruction":
        return reconstruction
    if view == "Reconstruction + cell boundaries":
        return overlay_boundaries(reconstruction, labels)
    ids = site_ids(labels)
    if view == "Site IDs":
        return ids
    if view == "Site IDs + boundaries":
        return overlay_boundaries(ids, labels)
    if view == "Cell boundaries":
        return overlay_boundaries(np.zeros_like(reconstruction), labels)
    if view == "Reconstruction + sites":
        return overlay_sites(reconstruction, result["centers"])
    if view == "Transport support measure":
        return colour_map(geometry["measure"])
    if view == "Cartoon":
        return colour_map(geometry["cartoon"])
    if view == "Texture":
        return colour_map(np.abs(geometry["texture"]))
    if view == "Transport glass":
        return colour_map(np.abs(geometry["glass"]))
    if view == "Metric anisotropy":
        xx, xy, yy = (
            np.asarray(geometry["precision_xx"]),
            np.asarray(geometry["precision_xy"]),
            np.asarray(geometry["precision_yy"]))
        trace, disc = xx + yy, np.hypot(xx - yy, 2.0 * xy)
        return colour_map(np.log1p(np.sqrt(
            np.maximum(trace + disc, 1e-20)
            / np.maximum(trace - disc, 1e-20))))
    if view == "Residual energy":
        return colour_map(result["residual_energy"])
    if view == "Reverse residual flow":
        if not result["refinements"]:
            return np.zeros_like(reconstruction)
        return colour_map(np.log1p(result["refinements"][-1]["flux"]))
    if view == "Refinement demand":
        if not result["refinements"]:
            return np.zeros_like(reconstruction)
        refinement = result["refinements"][-1]
        return colour_map(
            refinement["error_ratio_map"]
            * refinement["return_extent_map"])
    if view == "Residual pressure density":
        if result["pressure"] is None:
            return np.zeros_like(reconstruction)
        return colour_map(result["pressure"]["density"])
    if view == "Residual metric pressure":
        if result["pressure"] is None:
            return np.zeros_like(reconstruction)
        return colour_map(result["pressure"]["metric_pressure"])
    if view == "Power reach":
        if result["pressure"] is None:
            return np.zeros_like(reconstruction)
        return colour_map(result["pressure"]["reach"][labels])
    if view == "Site motion":
        if result["characteristic"] is not None:
            return overlay_motion(
                reconstruction,
                result["characteristic"]["initial_centers"],
                result["centers"],
            )
        if result["pressure"] is not None:
            return overlay_motion(
                reconstruction,
                result["pressure"]["initial_centers"],
                result["centers"],
            )
        return np.zeros_like(reconstruction)
    if view in (
        "Characteristic force",
        "Topology clearance",
        "Trust-limited step",
    ):
        characteristic = result["characteristic"]
        if characteristic is None or not characteristic["trace"]:
            return np.zeros_like(reconstruction)
        last = characteristic["trace"][-1]
        if view == "Characteristic force":
            field = last["force"]["force_per_mass"]
        elif view == "Topology clearance":
            field = last["inradius_px"]
        else:
            field = last["limited_step_px"]
        return colour_map(np.asarray(field)[labels])
    if view == "RGB error":
        return colour_map(np.sqrt(np.mean((rgb - reconstruction) ** 2, axis=2)))
    return reconstruction


def refresh():
    with S.lock:
        rgb, result = S.rgb, S.result
    if rgb is None:
        return
    push_texture(SOURCE, rgb)
    push_texture(RESULT, current_view())
    if result is None:
        dpg.set_value("segmenting_metrics", "Not built yet.")
        return
    record, timing = result["record"], result["timing"]
    ah, aw = result["allocation_geometry"]["measure"].shape
    refinement_text = ""
    if result["refinements"]:
        counts = [str(item["split_count"]) for item in result["refinements"]]
        refinement_text = f" | residual splits {' → '.join(counts)}"
    if result["pressure"] is not None:
        pressure = result["pressure"]
        refinement_text += (
            f" | organic pressure {len(pressure['trace'])} passes, "
            f"{pressure['initial_cells']} → {pressure['final_cells']} sites")
    if result["characteristic"] is not None:
        characteristic = result["characteristic"]
        accepted = sum(
            bool(item["accepted"]) for item in characteristic["trace"])
        last = (
            characteristic["trace"][-1]
            if characteristic["trace"] else None)
        refinement_text += (
            f" | causal characteristic {accepted}/"
            f"{len(characteristic['trace'])} accepted")
        if last is not None:
            refinement_text += (
                f", transport action "
                f"{100.0 * last['relative_action_change']:+.2f}%")
    dpg.set_value(
        "segmenting_metrics",
        f"{len(result['centers'])} cells | PSNR {record['psnr']:.2f} dB | "
        f"cartoon MSE {record['cartoon_mse']:.3e} | "
        f"texture MSE {record['texture_mse']:.3e} | "
        f"allocation {aw}×{ah} → readout {rgb.shape[1]}×{rgb.shape[0]} | "
        f"Meyer {timing['geometry_ms']:.0f} ms | "
        f"transport {timing['allocation_ms']:.0f} ms | "
        f"fit/refine/score {timing['fit_ms']:.0f} ms"
        f"{refinement_text}")


def work_side():
    return 0 if dpg.get_value("segmenting_full") else int(
        dpg.get_value("segmenting_work_side"))


def build_worker():
    if S.busy or S.image is None:
        return
    S.busy = True
    try:
        rgb = _fit_rgb(S.image, work_side())
        S.status = "Measuring one frozen optimized Meyer geometry..."
        evolution = dpg.get_value("segmenting_evolution")
        config = SegmentingConfig(
            allocation_method=(
                "causal_density"
                if evolution == "Causal characteristic relaxation"
                else "legacy_bifurcation"
            ),
            allocation_max_side=int(dpg.get_value("segmenting_allocation_side")),
            tgfd_sweeps=int(dpg.get_value("segmenting_tgfd_sweeps")),
            flow_sweeps=int(dpg.get_value("segmenting_flow_sweeps")),
            threshold=float(dpg.get_value("segmenting_transport_threshold")),
            metric_extent_threshold=float(dpg.get_value("segmenting_metric_threshold")),
            metric_strength=float(dpg.get_value("segmenting_metric_strength")),
            minimum_region_pixels=int(dpg.get_value("segmenting_min_pixels")),
            maximum_rounds=int(dpg.get_value("segmenting_rounds")),
            safety_cells=int(dpg.get_value("segmenting_safety")),
            branch_bins=int(dpg.get_value("segmenting_bins")),
            characteristic_passes=(
                int(dpg.get_value("segmenting_characteristic_passes"))
                if evolution == "Causal characteristic relaxation" else 0),
            characteristic_trust_fraction=float(
                dpg.get_value("segmenting_characteristic_trust")),
            characteristic_core_radius=float(
                dpg.get_value("segmenting_characteristic_core")),
            ridge_count=int(dpg.get_value("segmenting_ridges")),
            refinement_iterations=(
                int(dpg.get_value("segmenting_refinement_iterations"))
                if evolution == "Hierarchical split control" else 0),
            refinement_error_ratio=float(dpg.get_value("segmenting_refinement_error")),
            refinement_return_distance=float(dpg.get_value("segmenting_refinement_extent")),
            refinement_detail_gain=float(dpg.get_value("segmenting_refinement_gain")),
            pressure_passes=(
                int(dpg.get_value("segmenting_pressure_passes"))
                if evolution == "Organic residual pressure" else 0),
            pressure_strength=float(dpg.get_value("segmenting_pressure_strength")),
            pressure_temperature=float(dpg.get_value("segmenting_pressure_temperature")),
            pressure_position_relaxation=float(dpg.get_value("segmenting_pressure_position")),
            pressure_capacity_relaxation=float(dpg.get_value("segmenting_pressure_capacity")),
            pressure_metric_gain=float(dpg.get_value("segmenting_pressure_metric")),
            queue="bucket" if dpg.get_value("segmenting_queue") == "Exact monotone bucket" else "heap")
        result = build_segmenting_representation(rgb, config)
        with S.lock:
            S.rgb, S.result = rgb, result
        S.status = (
            f"{S.name}: complete at {len(result['centers'])} cells. "
            "Use Site IDs + boundaries to inspect the literal partition.")
    except Exception as exc:
        S.status = f"Build failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def adopt(image, name):
    S.image, S.name = np.asarray(image, dtype=np.float64), name
    with S.lock:
        S.rgb, S.result = _fit_rgb(S.image, work_side()), None
    S.status = f"{name} loaded. Press Build representation."
    push_texture(SOURCE, S.rgb)
    push_texture(RESULT, np.full_like(S.rgb, 0.08))


def cb_gallery(sender, label):
    try:
        key = gallery.key_for_label(label)
        adopt(gallery.load(key), gallery.describe(key)["label"])
    except Exception as exc:
        S.status = f"Gallery load failed: {type(exc).__name__}: {exc}"


def cb_file(sender, app_data):
    candidates = list((app_data.get("selections") or {}).values())
    candidates.append(app_data.get("file_path_name", ""))
    path = next((Path(p) for p in candidates if p and Path(p).is_file()), None)
    if path is None:
        S.status = "Could not resolve the selected image."
        return
    try:
        import matplotlib.image as mpimg
        adopt(mpimg.imread(path), path.name)
    except Exception as exc:
        S.status = f"Image load failed: {type(exc).__name__}: {exc}"


def slider(tag, label, default, low, high, *, floating=False, width=280):
    function = dpg.add_slider_float if floating else dpg.add_slider_int
    function(
        label=label, tag=tag, default_value=default,
        min_value=low, max_value=high, width=width)


def build_ui(labels, default_label):
    with dpg.file_dialog(
        directory_selector=False, show=False, callback=cb_file,
        tag="segmenting_file_dialog", width=900, height=520):
        dpg.add_file_extension(
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
            "{.png,.jpg,.jpeg,.tif,.tiff,.bmp}")
        dpg.add_file_extension(".*")
    with dpg.window(tag="segmenting_root"):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                labels, default_value=default_label, width=390,
                tag="segmenting_gallery", callback=cb_gallery)
            dpg.add_button(
                label="Load image...",
                callback=lambda: dpg.show_item("segmenting_file_dialog"))
            dpg.add_button(
                label="Build representation",
                callback=lambda: threading.Thread(
                    target=build_worker, daemon=True).start(),
                tag="segmenting_build")
        dpg.add_text("", tag="segmenting_status", wrap=1500)
        dpg.add_text("", tag="segmenting_metrics", wrap=1500)
        with dpg.collapsing_header(label="Image and Meyer geometry", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="process every source pixel (full-resolution output)",
                    tag="segmenting_full", default_value=False)
                slider("segmenting_work_side", "otherwise longest side", 768, 128, 3840, width=330)
                slider("segmenting_allocation_side", "allocation grid side", 512, 128, 1536, width=320)
            with dpg.group(horizontal=True):
                slider("segmenting_tgfd_sweeps", "single Meyer sweeps", 24, 4, 64)
                slider("segmenting_flow_sweeps", "fixed glass sweeps", 24, 4, 64)
                dpg.add_text("Optimized one-axis Meyer is always used.")
        with dpg.collapsing_header(label="Simultaneous transport allocation", default_open=True):
            with dpg.group(horizontal=True):
                slider("segmenting_transport_threshold", "transport extent", 2.5, 1.0, 12.0, floating=True)
                slider("segmenting_metric_threshold", "metric extent", 8.0, 1.0, 24.0, floating=True)
                slider("segmenting_metric_strength", "metric strength", 1.5, 0.0, 8.0, floating=True)
            with dpg.group(horizontal=True):
                slider("segmenting_min_pixels", "minimum region pixels", 12, 2, 128)
                slider("segmenting_rounds", "refill rounds", 24, 2, 32)
                slider("segmenting_safety", "safety ceiling (not target)", 32768, 256, 65536, width=310)
            with dpg.group(horizontal=True):
                slider("segmenting_bins", "local balance bins", 64, 16, 256)
                dpg.add_combo(
                    ("Exact monotone bucket", "Exact heap control"),
                    default_value="Exact monotone bucket",
                    label="topology refresh", tag="segmenting_queue", width=260)
                slider("segmenting_ridges", "measured ridge finish", 1, 0, 2)
            dpg.add_text(
                "Every unstable support refills at once. No top-k, candidate "
                "scan, deletion, or population target.")
        with dpg.collapsing_header(label="Geometry evolution", default_open=True):
            dpg.add_combo(
                (
                    "Causal characteristic relaxation",
                    "Organic residual pressure",
                    "Hierarchical split control",
                    "None",
                ),
                default_value="Causal characteristic relaxation",
                label="method",
                tag="segmenting_evolution",
                width=300)
            dpg.add_text(
                "Causal mode emits the complete population directly from the "
                "frozen BFFT tensor, then moves each germ only within its "
                "topology-safe first-arrival clearance.")
        with dpg.collapsing_header(
            label="Causal characteristic relaxation",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                slider(
                    "segmenting_characteristic_passes",
                    "exact front passes", 1, 0, 4)
                slider(
                    "segmenting_characteristic_trust",
                    "interface safety", 0.5, 0.05, 0.5,
                    floating=True)
                slider(
                    "segmenting_characteristic_core",
                    "germ shell radius", 3.0, 2.0, 8.0,
                    floating=True)
            dpg.add_text(
                "No runner-up field or centroid motion. Resource is carried "
                "back through each achieving front; half-inradius trust "
                "regions and exact action decrease keep every germ alive.")
        with dpg.collapsing_header(label="Organic residual pressure", default_open=True):
            with dpg.group(horizontal=True):
                slider("segmenting_pressure_passes", "equilibrium passes", 4, 1, 16)
                slider(
                    "segmenting_pressure_strength", "residual density", 0.5,
                    0.0, 4.0, floating=True)
                slider(
                    "segmenting_pressure_metric", "local metric pressure", 4.0,
                    0.0, 16.0, floating=True)
            with dpg.group(horizontal=True):
                slider(
                    "segmenting_pressure_position", "site mobility", 0.25,
                    0.0, 1.0, floating=True)
                slider(
                    "segmenting_pressure_capacity", "capacity equalization", 0.0,
                    0.0, 1.0, floating=True)
                slider(
                    "segmenting_pressure_temperature", "soft occupancy", 2.0,
                    0.25, 8.0, floating=True)
            dpg.add_text(
                "Residual strengthens the pointwise BFFT metric and softly "
                "moves existing sites. Count is conserved exactly.")
        with dpg.collapsing_header(label="Hierarchical split control", default_open=False):
            with dpg.group(horizontal=True):
                slider("segmenting_refinement_iterations", "iterations", 1, 0, 4)
                slider(
                    "segmenting_refinement_error", "local error ratio", 1.5,
                    1.0, 12.0, floating=True)
                slider(
                    "segmenting_refinement_extent", "return-flow extent", 8.0,
                    0.0, 24.0, floating=True)
                slider(
                    "segmenting_refinement_gain", "detail pull", 1.0,
                    0.0, 4.0, floating=True)
            dpg.add_text(
                "Single-stage residual energy flows backward along each exact "
                "transport tree. Deserving cells refill forward together.")
        with dpg.group(horizontal=True):
            dpg.add_text("Right panel")
            dpg.add_combo(
                VIEWS, default_value=VIEWS[0], tag="segmenting_view",
                callback=lambda: refresh(), width=330)
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image(SOURCE, tag="segmenting_source_image")
            with dpg.group():
                dpg.add_text("BFFT transport-cell representation")
                dpg.add_image(RESULT, tag="segmenting_result_image")


def main():
    keys = gallery.available()
    labels = gallery.labels(keys)
    key = "pikachu" if "pikachu" in keys else keys[0]
    label = labels[keys.index(key)]
    dpg.create_context()
    alloc_textures(8, 8)
    build_ui(labels, label)
    dpg.create_viewport(
        title="BFFT Vision — Segmenting Transport Cells",
        width=1500, height=980)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("segmenting_root", True)
    cb_gallery(None, label)
    last = 0.0
    while dpg.is_dearpygui_running():
        dpg.set_value("segmenting_status", S.status)
        dpg.configure_item("segmenting_build", enabled=not S.busy)
        now = time.perf_counter()
        if now - last > 0.15:
            refresh()
            last = now
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
