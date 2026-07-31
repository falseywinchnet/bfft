#!/usr/bin/env python3
"""Interactive viewer for the strict two-scale segmenting 3.0 experiment."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "viewer", ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import dearpygui.dearpygui as dpg  # noqa: E402
import gallery  # noqa: E402
from experiments.segmenting_v3 import (  # noqa: E402
    SegmentingV3Config,
    build_segmenting_v3,
)
from port_needed.fast_image_ops import resize  # noqa: E402
from port_needed.eikonal_lanczos import (  # noqa: E402
    eikonal_lanczos_resize,
)
from port_needed.soft_support_diffusion import (  # noqa: E402
    build_soft_support_conductance,
    diffuse_soft_support,
)


PANEL = 720
SOURCE = "v3_source_texture"
RESULT = "v3_result_texture"
VIEWS = (
    "Reconstruction",
    "Structural soft IDs",
    "Texture micro IDs",
    "Residual error",
)


class State:
    def __init__(self):
        self.image = None
        self.name = "(none)"
        self.rgb = None
        self.result = None
        self.busy = False
        self.status = "Choose an image, then build version 3.0."
        self.metrics = ""
        self.lock = threading.Lock()
        self.buffers = {}
        self.texture_shapes = {}
        self.display_key = None
        self.source_display_key = None
        self.resampled_display = False


S = State()


def _rgb(image):
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    value = value[..., :3]
    peak = float(np.max(value, initial=0.0))
    if peak > 1.5:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def _work_rgb(image):
    value = _rgb(image)
    if dpg.get_value("v3_full"):
        return value
    return _fit_to_side(value, int(dpg.get_value("v3_work_side")))


def _fit_to_side(image, maximum_side):
    value = _rgb(image)
    height, width = value.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    output_shape = (
        max(16, round(height * scale)),
        max(16, round(width * scale)),
    )
    if output_shape != (height, width):
        value = resize(value, output_shape, order=1, anti_aliasing=True)
    return np.clip(value, 0.0, 1.0)


def _display_metric(result, shape):
    if result is None:
        return None
    labels = np.asarray(result["labels"], dtype=np.int32)
    if labels.shape != tuple(shape):
        return None
    geometry = result.get("texture_geometry")
    if geometry is None:
        geometry = result.get("cartoon_geometry")
    if geometry is None:
        return None
    tensor = tuple(
        np.asarray(geometry[name], dtype=np.float64)
        for name in ("boundary_xx", "boundary_xy", "boundary_yy")
    )
    if any(component.shape != labels.shape for component in tensor):
        return None
    return labels, tensor


def _fit_display_to_side(image, maximum_side, result=None):
    """Point proxy or owner-masked eikonal Lanczos display resampling."""
    value = _rgb(image)
    height, width = value.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    output_height = max(1, round(height * scale))
    output_width = max(1, round(width * scale))
    if (output_height, output_width) == (height, width):
        return value
    if S.resampled_display:
        metric = _display_metric(result, (height, width))
        if metric is not None:
            labels, tensor = metric
            return eikonal_lanczos_resize(
                value,
                (output_height, output_width),
                labels,
                tensor,
                anisotropy=0.5,
                clamp_range=True,
            )
        pixels = np.clip(np.rint(value * 255.0), 0.0, 255.0).astype(np.uint8)
        resized = Image.fromarray(pixels, mode="RGB").resize(
            (output_width, output_height),
            resample=Image.Resampling.LANCZOS,
        )
        return np.asarray(resized, dtype=np.float64) / 255.0
    y = np.minimum(
        ((np.arange(output_height) + 0.5) * height / output_height).astype(
            np.intp),
        height - 1,
    )
    x = np.minimum(
        ((np.arange(output_width) + 0.5) * width / output_width).astype(
            np.intp),
        width - 1,
    )
    return value[y[:, None], x[None, :]]


def _alloc_texture(tag, height, width):
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    if dpg.does_alias_exist(tag):
        dpg.remove_alias(tag)
    S.buffers[tag] = np.ones(height * width * 4, dtype=np.float32)
    S.texture_shapes[tag] = (height, width)
    with dpg.texture_registry():
        dpg.add_raw_texture(
            width,
            height,
            S.buffers[tag],
            tag=tag,
            format=dpg.mvFormat_Float_rgba,
        )
    scale = PANEL / max(height, width)
    item = "v3_source_image" if tag == SOURCE else "v3_result_image"
    if dpg.does_item_exist(item):
        dpg.configure_item(
            item,
            texture_tag=tag,
            width=max(1, int(width * scale)),
            height=max(1, int(height * scale)),
        )


def _push_texture(tag, image, result=None):
    value = _rgb(image).astype(np.float32)
    if max(value.shape[:2]) > PANEL:
        value = _fit_display_to_side(
            value, PANEL, result=result).astype(np.float32)
    height, width = value.shape[:2]
    shape = (height, width)
    if S.texture_shapes.get(tag) != shape:
        _alloc_texture(tag, height, width)
    buffer = S.buffers[tag].reshape(height, width, 4)
    buffer[..., :3] = value
    buffer[..., 3] = 1.0
    dpg.set_value(tag, S.buffers[tag])


def _boundaries(labels):
    labels = np.asarray(labels)
    mask = np.zeros(labels.shape, dtype=bool)
    mask[1:] |= labels[1:] != labels[:-1]
    mask[:-1] |= labels[:-1] != labels[1:]
    mask[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return mask


def _overlay(image, labels):
    value = np.asarray(image, dtype=np.float64).copy()
    value[_boundaries(labels)] = (0.0, 1.0, 0.85)
    return value


def _owner_colours(labels):
    count = int(np.max(labels)) + 1
    index = np.arange(count, dtype=np.uint32)
    value = index * np.uint32(747796405) + np.uint32(2891336453)
    value = ((value >> ((value >> 28) + 4)) ^ value) * np.uint32(277803737)
    value = (value >> 22) ^ value
    colours = np.column_stack((
        value & 255,
        (value >> 8) & 255,
        (value >> 16) & 255,
    )).astype(np.float64) / 255.0
    return 0.12 + 0.86 * colours[labels]


def _scalar_map(field):
    value = np.asarray(field, dtype=np.float64)
    finite = value[np.isfinite(value)]
    if not finite.size:
        return np.zeros(value.shape + (3,), dtype=np.float64)
    low, high = np.percentile(finite, (1.0, 99.5))
    z = np.clip((value - low) / max(float(high - low), 1e-12), 0.0, 1.0)
    return np.stack((
        np.clip(1.7 * z - 0.35, 0.0, 1.0),
        np.clip(1.8 - 2.4 * np.abs(z - 0.58), 0.0, 1.0),
        np.clip(1.15 - 1.45 * z, 0.0, 1.0),
    ), axis=2)


def _signed_map(field):
    value = np.asarray(field, dtype=np.float64)
    finite = np.abs(value[np.isfinite(value)])
    scale = max(float(np.percentile(finite, 98.0)), 1e-12)
    z = np.clip(value / scale, -1.0, 1.0)
    white = 1.0 - np.abs(z)
    return np.stack((
        white + np.maximum(z, 0.0),
        white,
        white + np.maximum(-z, 0.0),
    ), axis=2)


def _view(result, name):
    if name == "Reconstruction":
        return result["reconstruction_rgb"]
    if name == "Texture micro IDs":
        return _owner_colours(result["texture_labels"])
    if name == "Structural soft IDs":
        if "structural_soft_ids" not in result:
            ids = _owner_colours(result["labels"])
            geometry = result.get("texture_geometry")
            if (
                geometry is not None
                and np.asarray(geometry["measure"]).shape
                == result["labels"].shape
            ):
                conductance = build_soft_support_conductance(
                    geometry,
                    result["source_rgb"],
                    metric_strength=1.5,
                )
                ids = np.clip(diffuse_soft_support(
                    ids,
                    conductance,
                    passes=16,
                    coupling=0.8,
                    threads=4,
                ), 0.0, 1.0)
            result["structural_soft_ids"] = ids
        return result["structural_soft_ids"]
    if name == "Residual error":
        return _scalar_map(result["residual_energy"])
    raise ValueError(f"unknown v3 view {name!r}")


def refresh():
    if S.busy:
        return
    with S.lock:
        result = S.result
        rgb = S.rgb
    if result is None:
        return
    source_key = (id(result), S.resampled_display)
    if source_key != S.source_display_key and rgb is not None:
        _push_texture(SOURCE, rgb, result=result)
        S.source_display_key = source_key
    name = dpg.get_value("v3_view")
    key = (id(result), name, S.resampled_display)
    if key == S.display_key:
        return
    _push_texture(RESULT, _view(result, name), result=result)
    S.display_key = key


def toggle_display_sampling():
    S.resampled_display = not S.resampled_display
    label = (
        "Display: eikonal Lanczos view"
        if S.resampled_display
        else "Display: full-res point view"
    )
    dpg.configure_item("v3_display_sampling", label=label)
    if S.busy:
        # The build worker owns Numba's parallel workqueue. The main-loop
        # refresh will apply this display state as soon as the build returns.
        S.display_key = None
        S.source_display_key = None
        return
    with S.lock:
        rgb = S.rgb
        result = S.result
    if rgb is not None:
        _push_texture(SOURCE, rgb, result=result)
        S.source_display_key = (
            (id(result), S.resampled_display)
            if result is not None else None
        )
    if result is not None:
        S.display_key = None
        refresh()


def _config():
    geometry = (
        "owner_eikonal"
        if dpg.get_value("v3_coordinate_geometry") == "Owner-masked eikonal"
        else "straight"
    )
    axes = (
        "four_axes"
        if dpg.get_value("v3_coordinate_axes") == "Four tensor axes"
        else "paired"
    )
    upgrade_mode = (
        "full_map"
        if dpg.get_value("v3_owner_mode") == "Full-map germ refresh"
        else "boundary_band"
    )
    texture_model = (
        "nested_population"
        if dpg.get_value("v3_texture_model")
        == "Nested full-resolution supports"
        else "parent_ridges"
    )
    split_transport = (
        "local_eikonal"
        if dpg.get_value("v3_texture_split_transport")
        == "Local eikonal remarch (control)"
        else "paired_metric"
    )
    structural_topology = (
        "canonical_v2"
        if dpg.get_value("v3_structural_topology")
        == "Canonical v2 structural quotient"
        else "half_cartoon"
    )
    structural_transport = {
        "Automatic: coarse continuous / full bucket": "auto",
        "Continuous control": "continuous",
        "Full bucket graph": "bucket_graph",
    }[dpg.get_value("v3_structural_transport")]
    return SegmentingV3Config(
        structural_topology=structural_topology,
        structural_full_transport=structural_transport,
        structural_allocation_side=int(
            dpg.get_value("v3_structural_allocation_side")),
        structural_safety_cells=int(dpg.get_value("v3_safety")),
        structural_population_scale=float(
            dpg.get_value("v3_structural_population_scale")),
        structural_characteristic_passes=int(
            dpg.get_value("v3_structural_characteristic_passes")),
        cartoon_scale=float(dpg.get_value("v3_cartoon_scale")),
        meyer_sweeps=int(dpg.get_value("v3_meyer_sweeps")),
        metric_strength=float(dpg.get_value("v3_metric_strength")),
        boundary_jump_strength=float(dpg.get_value("v3_boundary_jump")),
        safety_cells=int(dpg.get_value("v3_safety")),
        owner_upgrade=bool(dpg.get_value("v3_owner_upgrade")),
        owner_upgrade_mode=upgrade_mode,
        owner_upgrade_radius=int(dpg.get_value("v3_owner_radius")),
        owner_upgrade_sweeps=int(dpg.get_value("v3_owner_sweeps")),
        owner_upgrade_strength=float(dpg.get_value("v3_owner_strength")),
        owner_upgrade_cartoon_strength=float(
            dpg.get_value("v3_owner_cartoon_strength")),
        cartoon_full_refit=bool(dpg.get_value("v3_cartoon_refit")),
        cartoon_refit_strength=float(
            dpg.get_value("v3_cartoon_refit_strength")),
        texture_model=texture_model,
        nested_texture_ridges=int(dpg.get_value("v3_nested_ridges")),
        texture_graph_phase=bool(
            dpg.get_value("v3_texture_graph_phase")),
        texture_dirichlet_envelope=bool(
            dpg.get_value("v3_texture_dirichlet_envelope")),
        texture_support_weight=float(
            dpg.get_value("v3_texture_support_weight")),
        texture_population_phase=float(
            dpg.get_value("v3_texture_population_phase")),
        texture_curvature_population=bool(
            dpg.get_value("v3_texture_curvature")),
        texture_safety_cells=int(dpg.get_value("v3_texture_safety")),
        texture_cleanup=bool(dpg.get_value("v3_texture_cleanup")),
        texture_split_error_ratio=float(
            dpg.get_value("v3_texture_split_ratio")),
        texture_split_return_extent=float(
            dpg.get_value("v3_texture_split_extent")),
        texture_split_minimum_pixels=int(
            dpg.get_value("v3_texture_split_pixels")),
        texture_split_transport=split_transport,
        texture_split_metric_strength=float(
            dpg.get_value("v3_texture_split_metric_strength")),
        texture_merge_penalty=float(
            dpg.get_value("v3_texture_merge_penalty")),
        texture_cross_structural_merges=bool(
            dpg.get_value("v3_texture_cross_structural")),
        texture_interface_refresh=bool(
            dpg.get_value("v3_texture_interface_refresh")),
        texture_interface_confidence=float(
            dpg.get_value("v3_texture_interface_confidence")),
        texture_interface_error_ratio=float(
            dpg.get_value("v3_texture_interface_error")),
        texture_coordinates=int(dpg.get_value("v3_texture_coordinates")),
        texture_tensor_sigma=float(dpg.get_value("v3_tensor_sigma")),
        coordinate_axes=axes,
        coordinate_geometry=geometry,
        eikonal_sweeps=int(dpg.get_value("v3_eikonal_sweeps")),
        eikonal_metric_strength=float(dpg.get_value("v3_eikonal_strength")),
        offset_bins=int(dpg.get_value("v3_offset_bins")),
        ridge_kappa=float(dpg.get_value("v3_ridge_kappa")),
        threads=int(dpg.get_value("v3_threads")),
    )


def build_worker(rgb, config):
    try:
        result = build_segmenting_v3(rgb, config)
        timing = result["timing"]
        geometry = result["coordinate_geometry"]
        coordinate_ms = (
            timing["coordinate_geometry_ms"]
            + timing["texture_coordinate_ms"]
        )
        texture_population = result["texture_population"]
        characteristic = result["structural_characteristic"]["trace"]
        accepted_characteristic = sum(
            bool(item["accepted"]) for item in characteristic)
        phase_graph = result["texture_phase_graph"]
        texture_envelope = result["texture_dirichlet_envelope"]
        characteristic_state = (
            f"{accepted_characteristic}/{len(characteristic)} accepted"
            if result["structural_characteristic"]["resolved_core"]
            else "skipped: core under-resolved"
        )
        S.metrics = (
            f"{len(result['centers']):,} preserved cartoon owner IDs | "
            f"{int(texture_population.get('surplus_sites', 0)):,} "
            "surplus texture germs | "
            f"{len(result['texture_centers']):,} texture microcells | "
            f"{result['record']['psnr']:.2f} dB | "
            f"transport {result['structural_transport_model']} | "
            f"characteristic {characteristic_state} | "
            f"cartoon geometry {timing['cartoon_geometry_ms']:.0f} ms | "
            f"transport {timing['cartoon_transport_ms']:.0f} ms | "
            f"cartoon fit {timing['cartoon_fit_ms']:.0f} ms | "
            f"owner upgrade {timing['owner_upgrade_ms']:.0f} ms | "
            f"cartoon refit {timing['cartoon_refit_ms']:.0f} ms | "
            f"tensor {timing['texture_tensor_ms']:.0f} ms | "
            f"texture population "
            f"{timing['texture_population_geometry_ms']:.0f} + "
            f"{timing['texture_population_transport_ms']:.0f} ms | "
            f"texture affine {timing['texture_affine_ms']:.0f} ms | "
            f"flat cleanup {timing['texture_cleanup_ms']:.0f} ms | "
            f"phase graph "
            f"{phase_graph['graph_edges']:,}→"
            f"{phase_graph['tree_edges']:,} edges in "
            f"{timing['texture_phase_graph_ms']:.0f} ms | "
            f"energy envelope "
            f"{texture_envelope['contracted_cells']:,} cells in "
            f"{timing['texture_dirichlet_envelope_ms']:.0f} ms | "
            f"coordinates {coordinate_ms:.0f} ms | "
            f"total {timing['total_ms']:.0f} ms"
        )
        fallback_fields = [
            value for key, value in geometry.items()
            if key.endswith("_fallback_pixels")]
        fallback = sum(
            value for key, value in geometry.items()
            if key.endswith("_fallback_pixels"))
        upgrade = result["owner_upgrade"]
        cleanup = result["texture_cleanup"]
        crossing_description = (
            "texture-model merges shared across retained structural IDs"
            if result["structural_topology"] == "canonical_v2"
            else "merges crossed the expired cartoon scaffold"
        )
        S.status = (
            f"{S.name}: {result['model']}. "
            f"Flat texture cleanup: {cleanup['initial_cells']:,} + "
            f"{cleanup['split_count']:,} splits - "
            f"{cleanup['merge_count']:,} mutual merges = "
            f"{cleanup['final_cells']:,}; "
            f"{cleanup['cross_parent_merge_count']:,} "
            f"{crossing_description}. "
            f"Strong hot-interface refresh moved "
            f"{cleanup.get('interface_changed_pixels', 0):,} pixels. "
            f"Owner upgrade moved {upgrade['changed_pixels']:,} / "
            f"{upgrade['band_pixels']:,} boundary-band pixels; "
            f"Raster-disconnected coordinate fallbacks: {fallback:,} "
            f"across {len(fallback_fields)} "
            f"{geometry['total_pixels']:,}-pixel fields.")
        with S.lock:
            S.rgb = rgb
            S.result = result
            S.display_key = None
            S.source_display_key = None
    except Exception as exc:
        S.status = f"Build failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def start_build():
    if S.busy or S.image is None:
        return
    # Read DearPyGui state and perform any display/input resize on the UI
    # thread before the worker starts. Numba's workqueue backend is not
    # re-entrant across simultaneous Python threads.
    rgb = _work_rgb(S.image)
    config = _config()
    S.busy = True
    S.status = f"{S.name}: building strict version 3.0..."
    threading.Thread(
        target=build_worker,
        args=(rgb, config),
        daemon=True,
    ).start()


def adopt(image, name):
    if S.busy:
        raise RuntimeError("wait for the current build to finish")
    S.image = np.asarray(image)
    S.name = name
    rgb = _work_rgb(S.image)
    with S.lock:
        S.rgb = rgb
        S.result = None
        S.display_key = None
        S.source_display_key = None
    S.metrics = ""
    S.status = f"{name} loaded. Press Build version 3.0."
    _push_texture(SOURCE, rgb)
    _push_texture(RESULT, np.full_like(rgb, 0.08))


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
        from PIL import Image
        adopt(np.asarray(Image.open(path).convert("RGB")), path.name)
    except Exception as exc:
        S.status = f"Image load failed: {type(exc).__name__}: {exc}"


def slider(tag, label, default, low, high, *, floating=False, width=270):
    function = dpg.add_slider_float if floating else dpg.add_slider_int
    function(
        label=label,
        tag=tag,
        default_value=default,
        min_value=low,
        max_value=high,
        width=width,
    )


def build_ui(labels, default_label):
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=cb_file,
        tag="v3_file_dialog",
        width=900,
        height=520,
    ):
        dpg.add_file_extension(
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
            "{.png,.jpg,.jpeg,.tif,.tiff,.bmp}")
        dpg.add_file_extension(".*")
    with dpg.window(tag="v3_root"):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                labels,
                default_value=default_label,
                width=410,
                tag="v3_gallery",
                callback=cb_gallery,
            )
            dpg.add_button(
                label="Load image...",
                callback=lambda: dpg.show_item("v3_file_dialog"),
            )
            dpg.add_button(
                label="Build version 3.0",
                callback=lambda: start_build(),
                tag="v3_build",
            )
        dpg.add_text("", tag="v3_status", wrap=1550)
        dpg.add_text("", tag="v3_metrics", wrap=1550)

        with dpg.collapsing_header(
            label="1. Structural quotient and one transport",
            default_open=True,
        ):
            dpg.add_combo(
                (
                    "Canonical v2 structural quotient",
                    "Half-scale cartoon control",
                ),
                default_value="Canonical v2 structural quotient",
                label="structural topology",
                tag="v3_structural_topology",
                width=360,
            )
            dpg.add_combo(
                (
                    "Automatic: coarse continuous / full bucket",
                    "Continuous control",
                    "Full bucket graph",
                ),
                default_value=(
                    "Automatic: coarse continuous / full bucket"),
                label="structural transport",
                tag="v3_structural_transport",
                width=360,
            )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="process every source pixel",
                    tag="v3_full",
                    default_value=True,
                )
                slider("v3_work_side", "otherwise longest side", 768, 128, 3840)
                slider(
                    "v3_cartoon_scale",
                    "cartoon scale",
                    0.5,
                    0.2,
                    1.0,
                    floating=True,
                )
            with dpg.group(horizontal=True):
                slider("v3_meyer_sweeps", "Meyer sweeps", 1, 1, 16)
                slider("v3_safety", "cell safety ceiling", 32768, 256, 65536)
                slider(
                    "v3_structural_allocation_side",
                    "v2 allocation side",
                    512,
                    128,
                    1536,
                )
                slider(
                    "v3_structural_characteristic_passes",
                    "characteristic passes",
                    1,
                    0,
                    2,
                )
                slider(
                    "v3_structural_population_scale",
                    "structural mass",
                    0.8,
                    0.1,
                    1.0,
                    floating=True,
                )
            with dpg.group(horizontal=True):
                slider(
                    "v3_metric_strength",
                    "transport metric",
                    1.5,
                    0.0,
                    8.0,
                    floating=True,
                )
                slider(
                    "v3_boundary_jump",
                    "boundary jump",
                    24.0,
                    0.0,
                    48.0,
                    floating=True,
                )
            dpg.add_text(
                "The hybrid retains the canonical sparse v2 partition as a "
                "structural quotient. Dense texture cells refine residuals "
                "under those IDs. The half-scale v3 scaffold remains as the "
                "A/B control.")

        with dpg.collapsing_header(
            label="2. Upgrade existing owners on the full-resolution map",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="edge-upgrade lifted owner boundaries",
                    tag="v3_owner_upgrade",
                    default_value=True,
                )
                dpg.add_combo(
                    ("Full-map germ refresh", "Boundary-band polish"),
                    default_value="Boundary-band polish",
                    label="upgrade topology",
                    tag="v3_owner_mode",
                    width=270,
                )
                slider("v3_owner_radius", "boundary radius", 8, 1, 24)
                slider("v3_owner_sweeps", "fixed causal sweeps", 2, 1, 8)
                slider(
                    "v3_owner_strength",
                    "full-resolution edge strength",
                    64.0,
                    0.0,
                    128.0,
                    floating=True,
                )
                slider(
                    "v3_owner_cartoon_strength",
                    "straight-cell BFFT direction",
                    0.0,
                    0.0,
                    64.0,
                    floating=True,
                )
            dpg.add_text(
                "Full-map mode seeds only the original cartoon germs, allowing "
                "long anisotropic cells to form again. Boundary mode retains "
                "coarse interiors as an A/B control. Every owner survives.")
            dpg.add_checkbox(
                label="refit upgraded cartoon cells at full resolution",
                tag="v3_cartoon_refit",
                default_value=True,
            )
            slider(
                "v3_cartoon_refit_strength",
                "cartoon refit blend",
                0.5,
                0.0,
                1.0,
                floating=True,
            )

        with dpg.collapsing_header(
            label="3. Full-resolution texture inside upgraded owners",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    (
                        "Nested full-resolution supports",
                        "Parent-cell coordinate control",
                    ),
                    default_value="Nested full-resolution supports",
                    label="texture topology",
                    tag="v3_texture_model",
                    width=320,
                )
                slider("v3_nested_ridges", "ridges per microcell", 3, 0, 4)
                dpg.add_checkbox(
                    label="graph-unrolled paired phase",
                    tag="v3_texture_graph_phase",
                    default_value=True,
                )
                dpg.add_checkbox(
                    label="nonexpansive texture-gradient envelope",
                    tag="v3_texture_dirichlet_envelope",
                    default_value=True,
                )
                slider(
                    "v3_texture_support_weight",
                    "texture support weight",
                    0.65,
                    0.0,
                    1.5,
                    floating=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="curvature-limited texture population",
                    tag="v3_texture_curvature",
                    default_value=True,
                )
                slider(
                    "v3_texture_safety",
                    "texture cell safety ceiling",
                    131072,
                    4096,
                    131072,
                )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="flat residual split / MDL merge cleanup",
                    tag="v3_texture_cleanup",
                    default_value=True,
                )
                slider(
                    "v3_texture_split_ratio",
                    "split error ratio",
                    2.5,
                    1.0,
                    8.0,
                    floating=True,
                )
                slider(
                    "v3_texture_split_extent",
                    "split return extent",
                    2.0,
                    0.0,
                    8.0,
                    floating=True,
                )
            with dpg.group(horizontal=True):
                slider(
                    "v3_texture_split_pixels",
                    "minimum split pixels",
                    12,
                    2,
                    128,
                )
                slider(
                    "v3_texture_merge_penalty",
                    "merge model allowance",
                    4.0,
                    0.0,
                    8.0,
                    floating=True,
                )
                dpg.add_checkbox(
                    label="share texture models across structural IDs",
                    tag="v3_texture_cross_structural",
                    default_value=False,
                )
                slider(
                    "v3_texture_population_phase",
                    "surplus phase",
                    0.125,
                    0.0,
                    1.0,
                    floating=True,
                )
                slider(
                    "v3_texture_split_metric_strength",
                    "paired metric strength",
                    0.25,
                    0.0,
                    2.0,
                    floating=True,
                )
                dpg.add_combo(
                    (
                        "Closed-form paired metric (fast)",
                        "Local eikonal remarch (control)",
                    ),
                    default_value="Closed-form paired metric (fast)",
                    label="split transport",
                    tag="v3_texture_split_transport",
                    width=320,
                )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="refresh hot strong texture interfaces",
                    tag="v3_texture_interface_refresh",
                    default_value=False,
                )
                slider(
                    "v3_texture_interface_confidence",
                    "minimum edge confidence",
                    0.15,
                    0.0,
                    1.0,
                    floating=True,
                )
                slider(
                    "v3_texture_interface_error",
                    "minimum incident error ratio",
                    2.0,
                    0.0,
                    8.0,
                    floating=True,
                )
            with dpg.group(horizontal=True):
                slider(
                    "v3_texture_coordinates",
                    "coordinate slots",
                    8,
                    0,
                    12,
                )
                dpg.add_combo(
                    ("Paired normal/tangent", "Four tensor axes"),
                    default_value="Paired normal/tangent",
                    label="axis schedule",
                    tag="v3_coordinate_axes",
                    width=260,
                )
                slider(
                    "v3_tensor_sigma",
                    "texture tensor sigma",
                    1.0,
                    0.0,
                    5.0,
                    floating=True,
                )
                slider("v3_offset_bins", "paired offset bins", 161, 17, 321)
            with dpg.group(horizontal=True):
                slider(
                    "v3_ridge_kappa",
                    "half-angle ridge gain",
                    16.0,
                    1.0,
                    48.0,
                    floating=True,
                )
                slider("v3_threads", "native threads", 4, 0, 16)
            dpg.add_text(
                "Nested mode emits full-resolution supports, assigns each to "
                "one cartoon parent, and transports only among siblings. The "
                "parent is then discarded: one residual backflow pass splits "
                "hot cells and reciprocal local model-cost exchange merges "
                "compatible flat texture IDs across former parent edges. "
                "Parent-coordinate mode retains the former eight-cut control.")

        with dpg.collapsing_header(
            label="4. Experimental coordinate geometry",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    ("Straight tensor frame", "Owner-masked eikonal"),
                    default_value="Straight tensor frame",
                    label="coordinate geometry",
                    tag="v3_coordinate_geometry",
                    width=280,
                )
                slider("v3_eikonal_sweeps", "causal raster sweeps", 2, 1, 8)
                slider(
                    "v3_eikonal_strength",
                    "eikonal tensor strength",
                    2.0,
                    0.0,
                    16.0,
                    floating=True,
                )
            dpg.add_text(
                "Eikonal mode bends the same paired normal/tangent basis "
                "inside each immutable owner using fixed causal sweeps. It "
                "never launches a per-cell heap and never crosses an owner.")

        with dpg.group(horizontal=True):
            dpg.add_text("Right panel")
            dpg.add_combo(
                VIEWS,
                default_value=VIEWS[0],
                tag="v3_view",
                callback=lambda: refresh(),
                width=360,
            )
            dpg.add_button(
                label="Display: full-res point view",
                tag="v3_display_sampling",
                callback=lambda: toggle_display_sampling(),
            )
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image(SOURCE, tag="v3_source_image")
            with dpg.group():
                dpg.add_text("Version 3.0 experiment")
                dpg.add_image(RESULT, tag="v3_result_image")


def main():
    keys = gallery.available()
    labels = gallery.labels(keys)
    key = "golden_gate" if "golden_gate" in keys else (
        "pikachu" if "pikachu" in keys else (
        "coffee" if "coffee" in keys else keys[0])
    )
    label = labels[keys.index(key)]
    dpg.create_context()
    _alloc_texture(SOURCE, 8, 8)
    _alloc_texture(RESULT, 8, 8)
    build_ui(labels, label)
    dpg.create_viewport(
        title="BFFT Vision — Segmenting Version 3.0",
        width=1600,
        height=1040,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("v3_root", True)
    cb_gallery(None, label)
    last = 0.0
    while dpg.is_dearpygui_running():
        dpg.set_value("v3_status", S.status)
        dpg.set_value("v3_metrics", S.metrics)
        dpg.configure_item("v3_build", enabled=not S.busy)
        now = time.perf_counter()
        if now - last > 0.15:
            refresh()
            last = now
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
