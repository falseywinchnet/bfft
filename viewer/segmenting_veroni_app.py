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
from port_needed.ownership_diagnostics import (  # noqa: E402
    residual_ownership_diagnostics,
)
from port_needed.soft_support_diffusion import (  # noqa: E402
    conductance_field,
    diffuse_soft_support,
)
from transport_voronoi import _fit_rgb  # noqa: E402

PANEL = 650
SOURCE = "segmenting_source_texture"
RESULT = "segmenting_result_texture"
VIEWS = (
    "Reconstruction",
    "Hard reconstruction",
    "Reconstruction + cell boundaries",
    "Site IDs",
    "Soft Site IDs",
    "Soft Site IDs + hard boundaries",
    "Site IDs + boundaries",
    "Cell boundaries",
    "Reconstruction + sites",
    "Transport support measure",
    "Null evidence confidence",
    "Boundary jump confidence",
    "Interface coverage",
    "Curvature population factor",
    "Soft support conductance",
    "Cartoon",
    "Texture",
    "Transport glass",
    "Metric anisotropy",
    "Residual energy",
    "Residual energy + cell boundaries",
    "Same-owner source jump",
    "Interface-aligned source jump",
    "Germ injection source jump",
    "Cell mean residual energy",
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
        self.prepared_target = None
        self.busy = False
        self.status = "Choose an image, then build the representation."
        self.lock = threading.Lock()
        self.buffers = {}
        self.texture_shapes = {}


S = State()


def _rgb(image):
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    a = a[..., :3]
    if a.max(initial=0.0) > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def alloc_texture(tag, height, width):
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    if dpg.does_alias_exist(tag):
        dpg.remove_alias(tag)
    S.buffers[tag] = np.ones(height * width * 4, dtype=np.float32)
    S.texture_shapes[tag] = (height, width)
    with dpg.texture_registry():
        dpg.add_raw_texture(
            width, height, S.buffers[tag], tag=tag,
            format=dpg.mvFormat_Float_rgba)
    scale = PANEL / max(height, width)
    item = (
        "segmenting_source_image"
        if tag == SOURCE
        else "segmenting_result_image"
    )
    if dpg.does_item_exist(item):
        dpg.configure_item(
            item, texture_tag=tag,
            width=max(1, int(width * scale)),
            height=max(1, int(height * scale)))


def alloc_textures(height, width):
    for tag in (SOURCE, RESULT):
        alloc_texture(tag, height, width)


def rgba_buffer(image, buffer=None):
    """Pack an RGB display image, resizing stale storage when necessary."""
    height, width = image.shape[:2]
    required = height * width * 4
    if (
        buffer is None
        or buffer.dtype != np.float32
        or buffer.size != required
    ):
        buffer = np.empty(required, dtype=np.float32)
    pixels = buffer.reshape(height, width, 4)
    pixels[..., :3] = image
    pixels[..., 3] = 1.0
    return buffer


def push_texture(tag, image):
    image = _rgb(image).astype(np.float32)
    if max(image.shape[:2]) > PANEL:
        image = _fit_rgb(image, PANEL).astype(np.float32)
    shape = image.shape[:2]
    expected = shape[0] * shape[1] * 4
    if (
        S.texture_shapes.get(tag) != shape
        or tag not in S.buffers
        or S.buffers[tag].size != expected
    ):
        alloc_texture(tag, *shape)
    buffer = rgba_buffer(image, S.buffers.get(tag))
    S.buffers[tag] = buffer
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


_CURRENT = object()


def current_view(rgb=_CURRENT, result=_CURRENT):
    if rgb is _CURRENT or result is _CURRENT:
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
    if view == "Hard reconstruction":
        interface = result.get("interface_coverage")
        if interface is not None:
            return interface["hard_record"]["rgb"]
        soft = result["soft_support"]
        return (
            reconstruction
            if soft is None
            else soft["hard_record"]["rgb"]
        )
    if view == "Reconstruction + cell boundaries":
        return overlay_boundaries(reconstruction, labels)
    ids = site_ids(labels)
    if view == "Site IDs":
        return ids
    if view in ("Soft Site IDs", "Soft Site IDs + hard boundaries"):
        soft = result["soft_support"]
        if soft is None:
            return ids
        if "site_ids" not in soft:
            soft["site_ids"] = np.clip(diffuse_soft_support(
                ids,
                soft["conductance"],
                passes=soft["passes"],
                coupling=soft["coupling"],
            ), 0.0, 1.0)
        softened = soft["site_ids"]
        return (
            overlay_boundaries(softened, labels)
            if view == "Soft Site IDs + hard boundaries"
            else softened
        )
    if view == "Site IDs + boundaries":
        return overlay_boundaries(ids, labels)
    if view == "Cell boundaries":
        return overlay_boundaries(np.zeros_like(reconstruction), labels)
    if view == "Reconstruction + sites":
        return overlay_sites(reconstruction, result["centers"])
    if view == "Transport support measure":
        return colour_map(geometry["measure"])
    if view == "Null evidence confidence":
        return colour_map(geometry["null_confidence"])
    if view == "Boundary jump confidence":
        return colour_map(geometry["boundary_confidence"])
    if view == "Interface coverage":
        interface = result.get("interface_coverage")
        if interface is None:
            return np.zeros_like(reconstruction)
        return colour_map(interface["coverage"])
    if view == "Curvature population factor":
        return colour_map(geometry.get(
            "curvature_population_factor",
            np.ones(labels.shape, dtype=np.float64),
        ))
    if view == "Soft support conductance":
        soft = result["soft_support"]
        if soft is None:
            return np.zeros_like(reconstruction)
        return colour_map(conductance_field(soft["conductance"]))
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
    if view == "Residual energy + cell boundaries":
        return overlay_boundaries(
            colour_map(result["residual_energy"]), labels)
    ownership = result.get("ownership_diagnostics")
    if (
        ownership is None
        and view in (
            "Same-owner source jump",
            "Interface-aligned source jump",
            "Germ injection source jump",
            "Cell mean residual energy",
        )
    ):
        ownership = residual_ownership_diagnostics(
            rgb,
            labels,
            result["residual_energy"],
            centers=result["centers"],
        )
        result["ownership_diagnostics"] = ownership
    if view == "Same-owner source jump":
        return colour_map(ownership["same_owner_source_jump"])
    if view == "Interface-aligned source jump":
        return colour_map(ownership["interface_source_jump"])
    if view == "Germ injection source jump":
        return colour_map(ownership["germ_source_jump_map"])
    if view == "Cell mean residual energy":
        return colour_map(ownership["cell_mean_residual_energy"][labels])
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
    push_texture(RESULT, current_view(rgb, result))
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
            pixels = ah * aw
            refinement_text += (
                f", transport action "
                f"{100.0 * last['relative_action_change']:+.2f}%, "
                f"front {last['front_updates_after'] / pixels:.2f} "
                f"updates/px")
    if result["soft_support"] is not None:
        soft = result["soft_support"]
        hard = soft["hard_record"]
        refinement_text += (
            f" | soft support {'accepted' if soft['accepted'] else 'rejected'}"
            f", objective {hard['objective']:.3e} → "
            f"{soft['proposal_record']['objective']:.3e}"
        )
    if result.get("interface_coverage") is not None:
        interface = result["interface_coverage"]
        hard = interface["hard_record"]
        refinement_text += (
            f" | interface "
            f"{'accepted' if interface['accepted'] else 'rejected'}, "
            f"{interface['covered_pixels']} covered px, objective "
            f"{hard['objective']:.3e} → "
            f"{interface['proposal_record']['objective']:.3e}"
        )
    refinement_text += (
        f" | null retention "
        f"{100.0 * float(np.mean(result['geometry']['null_attenuation'])):.1f}%"
    )
    ownership = result.get("ownership_diagnostics")
    if ownership is not None:
        refinement_text += (
            f" | ownership {ownership['unowned_pixels']} unowned px, "
            f"{100.0 * ownership['same_owner_jump_fraction']:.1f}% "
            "source-jump mass inside cells"
        )
    if "straight_implied_cells" in result["geometry"]:
        refinement_text += (
            f" | curvature population "
            f"{result['geometry']['straight_implied_cells']:.0f} → "
            f"{result['geometry']['implied_cells']:.0f}"
        )
    fit_detail = timing.get("fit_detail", {})
    allocation_detail = timing.get("allocation_detail", {})
    transport_breakdown = ""
    if allocation_detail:
        transport_breakdown = (
            "transport detail: population "
            f"{float(allocation_detail.get('population_ms', 0.0)):.0f} ms, "
            "metric "
            f"{float(allocation_detail.get('full_metric_ms', 0.0)):.0f} ms, "
            "front "
            f"{float(allocation_detail.get('full_front_ms', 0.0)):.0f} ms | "
        )
    fit_breakdown = ""
    if fit_detail:
        interface_ms = (
            float(fit_detail.get("interface_proposal_ms", 0.0))
            + float(fit_detail.get("interface_score_ms", 0.0))
        )
        soft_ms = (
            float(fit_detail.get("soft_conductance_ms", 0.0))
            + float(fit_detail.get("soft_diffusion_ms", 0.0))
            + float(fit_detail.get("soft_score_ms", 0.0))
        )
        fit_breakdown = (
            " | fit detail: "
            + (
                "target cached, "
                if int(fit_detail.get("prepared_target_reused", 0))
                else ""
            )
            + "target split "
            f"{float(fit_detail.get('target_objective_ms', 0.0)):.0f} ms, "
            "region/ridge "
            f"{float(fit_detail.get('initial_region_fit_ms', 0.0)):.0f} ms, "
            "mechanics "
            f"{float(fit_detail.get('region_mechanics_ms', 0.0)):.0f} ms, "
            "candidate scores "
            f"{float(fit_detail.get('region_candidate_score_ms', 0.0)):.0f} ms, "
            f"interface {interface_ms:.0f} ms, "
            f"soft {soft_ms:.0f} ms, "
            f"{int(fit_detail.get('objective_evaluations', 0))} scores / "
            f"{int(fit_detail.get('objective_state_restores', 0))} "
            "cached restores"
        )
    dpg.set_value(
        "segmenting_metrics",
        f"{len(result['centers'])} cells | PSNR {record['psnr']:.2f} dB | "
        f"cartoon MSE {record['cartoon_mse']:.3e} | "
        f"texture MSE {record['texture_mse']:.3e} | "
        f"allocation {aw}×{ah} → readout {rgb.shape[1]}×{rgb.shape[0]} | "
        f"Meyer {timing['geometry_ms']:.0f} ms | "
        f"transport {timing['allocation_ms']:.0f} ms | "
        f"{transport_breakdown}"
        f"fit/refine/score {timing['fit_ms']:.0f} ms"
        f"{fit_breakdown}"
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
            curvature_limited_density=bool(
                dpg.get_value("segmenting_curvature_density")),
            null_evidence_strength=float(
                dpg.get_value("segmenting_null_evidence")),
            boundary_jump_strength=float(
                dpg.get_value("segmenting_boundary_jump")),
            interface_coverage_strength=float(
                dpg.get_value("segmenting_interface_coverage")),
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
            soft_support_passes=int(
                dpg.get_value("segmenting_soft_passes")),
            soft_support_coupling=float(
                dpg.get_value("segmenting_soft_coupling")),
            soft_support_colour_percentile=float(
                dpg.get_value("segmenting_soft_colour")),
            queue="bucket" if dpg.get_value("segmenting_queue") == "Exact monotone bucket" else "heap")
        with S.lock:
            prepared_target = S.prepared_target
        result = build_segmenting_representation(
            rgb, config, prepared_target=prepared_target)
        with S.lock:
            S.rgb, S.result = rgb, result
            S.prepared_target = result["prepared_target"]
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
        S.prepared_target = None
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
                slider("segmenting_tgfd_sweeps", "single Meyer sweeps", 1, 1, 64)
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
                dpg.add_checkbox(
                    label="curvature-limited anisotropic population",
                    tag="segmenting_curvature_density",
                    default_value=True)
            with dpg.group(horizontal=True):
                slider(
                    "segmenting_null_evidence",
                    "weak-detail null confidence", 0.5, 0.0, 1.0,
                    floating=True)
                slider(
                    "segmenting_boundary_jump",
                    "true-edge crossing action", 24.0, 0.0, 48.0,
                    floating=True)
            dpg.add_text(
                "Cross-scale agreement suppresses unsupported weak detail; "
                "decisive target jumps add finite action only across their "
                "normal. No top-k, candidate scan, deletion, or population "
                "target.")
        with dpg.collapsing_header(
            label="Owner-free soft support",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                slider(
                    "segmenting_interface_coverage",
                    "fractional interface coverage", 0.4, 0.0, 1.0,
                    floating=True)
                slider(
                    "segmenting_soft_passes",
                    "support diffusion passes", 16, 0, 64)
                slider(
                    "segmenting_soft_coupling",
                    "boundary sharing", 0.8, 0.0, 2.0,
                    floating=True)
                slider(
                    "segmenting_soft_colour",
                    "target agreement percentile", 60.0, 0.0, 100.0,
                    floating=True)
            dpg.add_text(
                "Hard arrival initializes an anisotropic heat cover. BFFT "
                "transport and unchanged-target agreement gate each exchange; "
                "constants are preserved, so the implicit site weights remain "
                "a normalized partition of unity. Unsupported boundaries fade "
                "without deleting a site.")
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
                    "exact front passes", 0, 0, 4)
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
