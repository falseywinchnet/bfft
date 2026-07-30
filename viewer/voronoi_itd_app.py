#!/usr/bin/env python3
"""Interactive image explorer for the 2-D Voronoi ITD operator.

Run:

    .venv/bin/python viewer/voronoi_itd_viewer.py

The expensive decomposition runs in a worker thread.  The five panels expose
the source, selected baseline, selected intrinsic rotation, amplitude-lifted
Voronoi support and a live detail-gain recomposition.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
from skimage.color import hsv2rgb, lab2rgb, rgb2lab

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dearpygui.dearpygui as dpg  # noqa: E402
import gallery  # noqa: E402
from voronoi_itd import (  # noqa: E402
    VoronoiITDConfig,
    level_statistics,
    support_boundary,
    voronoi_itd,
)

PANEL = 280
MAX_SIDE = 512
TAGS = (
    "vitd_source_texture",
    "vitd_baseline_texture",
    "vitd_rotation_texture",
    "vitd_support_texture",
    "vitd_recompose_texture",
)
IMAGES = (
    "vitd_source_image",
    "vitd_baseline_image",
    "vitd_rotation_image",
    "vitd_support_image",
    "vitd_recompose_image",
)
BUFFERS: dict[str, np.ndarray] = {}
TEXTURE_SHAPE = [8, 8]


class State:
    def __init__(self):
        self.rgb: np.ndarray | None = None
        self.lab: np.ndarray | None = None
        self.lightness: np.ndarray | None = None
        self.guidance: np.ndarray | None = None
        self.name = "(none)"
        self.result = None
        self.busy = False
        self.dirty = False
        self.elapsed_ms = 0.0
        self.status = "Choose an image and press Decompose."
        self.lock = threading.Lock()


S = State()


def _fit_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        if array.max(initial=0.0) > 1.5:
            array = array / 255.0
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3:
        array = array[..., :3]
        if array.max(initial=0.0) > 1.5:
            array = array / 255.0
    else:
        raise ValueError("expected a grayscale or RGB image")
    height, width = array.shape[:2]
    step = max(int(np.ceil(max(height, width) / MAX_SIDE)), 1)
    return np.ascontiguousarray(np.clip(array[::step, ::step], 0.0, 1.0))


def _with_lightness(lightness: np.ndarray) -> np.ndarray:
    if S.lab is None:
        raise RuntimeError("no image loaded")
    lab = S.lab.copy()
    lab[..., 0] = np.clip(lightness, 0.0, 1.0) * 100.0
    return np.clip(lab2rgb(lab), 0.0, 1.0)


def _rotation_rgb(rotation: np.ndarray) -> np.ndarray:
    scale = max(float(np.percentile(np.abs(rotation), 99.0)), 1e-12)
    value = np.clip(0.5 + 0.48 * rotation / scale, 0.0, 1.0)
    positive = np.clip(2.0 * (value - 0.5), 0.0, 1.0)
    negative = np.clip(2.0 * (0.5 - value), 0.0, 1.0)
    neutral = 1.0 - np.maximum(positive, negative)
    return np.stack((
        neutral + positive,
        neutral,
        neutral + negative,
    ), axis=2)


def _support_rgb(level) -> np.ndarray:
    owner = level.owner
    hue = np.mod(owner.astype(np.float64) * 0.6180339887498949, 1.0)
    confidence = np.clip(level.support_confidence, 0.0, 1.0)
    hsv = np.stack((
        hue,
        0.42 + 0.45 * confidence,
        0.72 + 0.25 * confidence,
    ), axis=2)
    rgb = hsv2rgb(hsv)
    rgb[support_boundary(owner)] *= 0.18
    for (y, x), polarity in zip(level.sites_yx, level.polarity):
        y0, y1 = max(int(y) - 2, 0), min(int(y) + 3, owner.shape[0])
        x0, x1 = max(int(x) - 2, 0), min(int(x) + 3, owner.shape[1])
        if polarity > 0:
            rgb[y0:y1, x0:x1] = (
                (1.0, 0.55, 0.05)
                if abs(int(polarity)) > 1
                else (1.0, 0.12, 0.05)
            )
        elif polarity < 0:
            rgb[y0:y1, x0:x1] = (
                (0.15, 0.75, 1.0)
                if abs(int(polarity)) > 1
                else (0.05, 0.25, 1.0)
            )
        else:
            rgb[y0:y1, x0:x1] = (1.0, 1.0, 1.0)
    return rgb


def render_panels():
    with S.lock:
        result = S.result
        source = None if S.rgb is None else S.rgb.copy()
    if source is None:
        return None
    if result is None or not result.levels:
        blank = np.full_like(source, 0.12)
        return source, blank, blank, blank, source
    index = int(np.clip(
        dpg.get_value("vitd_level") - 1, 0, len(result.levels) - 1))
    level = result.levels[index]
    gain = float(dpg.get_value("vitd_gain"))
    recomposed = result.residual.copy()
    for rotation in result.rotations:
        recomposed += gain * rotation
    return (
        source,
        _with_lightness(level.baseline),
        _rotation_rgb(level.rotation),
        _support_rgb(level),
        _with_lightness(recomposed),
    )


def allocate_textures(height: int, width: int):
    TEXTURE_SHAPE[:] = (height, width)
    for tag in TAGS:
        BUFFERS[tag] = np.ones(height * width * 4, dtype=np.float32)
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        if dpg.does_alias_exist(tag):
            dpg.remove_alias(tag)
    with dpg.texture_registry():
        for tag in TAGS:
            dpg.add_raw_texture(
                width,
                height,
                BUFFERS[tag],
                tag=tag,
                format=dpg.mvFormat_Float_rgba,
            )
    scale = PANEL / max(height, width)
    for image, texture in zip(IMAGES, TAGS):
        if dpg.does_item_exist(image):
            dpg.configure_item(
                image,
                texture_tag=texture,
                width=int(width * scale),
                height=int(height * scale),
            )


def push_panels(panels):
    for array, tag in zip(panels, TAGS):
        rgba = BUFFERS[tag]
        rgb = np.asarray(array, dtype=np.float32)
        rgba[0::4] = rgb[..., 0].ravel()
        rgba[1::4] = rgb[..., 1].ravel()
        rgba[2::4] = rgb[..., 2].ravel()
        dpg.set_value(tag, rgba)


def _adopt(image: np.ndarray, name: str):
    rgb = _fit_rgb(image)
    lab = rgb2lab(rgb)
    with S.lock:
        S.rgb = rgb
        S.lab = lab
        S.lightness = np.ascontiguousarray(lab[..., 0] / 100.0)
        # The scalar baseline remains intrinsic to lightness.  RGB supplies a
        # vector-valued pullback metric so chromatic edges can also obstruct
        # Eikonal travel without becoming extra extrema.
        S.guidance = rgb
        S.name = name
        S.result = None
        S.dirty = True
    height, width = rgb.shape[:2]
    allocate_textures(height, width)
    S.status = f"{name}: {height}x{width}. Press Decompose."


def gallery_callback(sender, label):
    del sender
    try:
        key = gallery.key_for_label(label)
        _adopt(gallery.load(key), gallery.describe(key)["label"])
    except Exception as exc:
        S.status = f"Could not load image: {type(exc).__name__}: {exc}"


def file_callback(sender, app_data):
    del sender
    selections = app_data.get("selections") or {}
    candidates = list(selections.values())
    if app_data.get("file_path_name"):
        candidates.append(app_data["file_path_name"])
    path = next((Path(item) for item in candidates
                 if item and Path(item).is_file()), None)
    if path is None:
        S.status = "Could not resolve the selected file."
        return
    try:
        import matplotlib.image as mpimg
        _adopt(mpimg.imread(path), path.name)
    except Exception as exc:
        S.status = f"Could not read {path.name}: {type(exc).__name__}: {exc}"


def _config_from_ui() -> VoronoiITDConfig:
    return VoronoiITDConfig(
        levels=1,
        alpha=float(dpg.get_value("vitd_alpha")),
        tgfd_sweeps=int(dpg.get_value("vitd_tgfd")),
        flow_sweeps=int(dpg.get_value("vitd_flow")),
        null_evidence_strength=float(dpg.get_value("vitd_null")),
        coherent_tangent_fraction=float(
            dpg.get_value("vitd_tangent")),
        texture_support_weight=float(
            dpg.get_value("vitd_texture_weight")),
        curvature_limited_density=bool(dpg.get_value("vitd_curve")),
        metric_strength=float(dpg.get_value("vitd_metric")),
        boundary_jump_strength=float(dpg.get_value("vitd_jump")),
        allocation_max_side=int(dpg.get_value("vitd_allocation_side")),
    )


def decompose_worker(config: VoronoiITDConfig):
    try:
        if S.lightness is None:
            return
        started = time.perf_counter()
        result = voronoi_itd(S.lightness, config, S.guidance)
        elapsed = 1000.0 * (time.perf_counter() - started)
        error = float(np.max(np.abs(
            result.reconstruction - S.lightness)))
        with S.lock:
            S.result = result
            S.elapsed_ms = elapsed
            S.dirty = True
        if result.levels:
            first = level_statistics(result.levels[0])
            S.status = (
                f"{S.name}: one frozen measurement, "
                f"{first['sites']} simultaneous sites, "
                f"{first['delaunay_edges']} Eikonal adjacencies, "
                f"Meyer {first['geometry_ms']:.0f} ms + "
                f"population/march {first['allocation_ms']:.0f} ms, "
                f"{elapsed:.0f} ms total. Reconstruction max error "
                f"{error:.2e}.")
            dpg.configure_item(
                "vitd_level", max_value=len(result.levels))
            dpg.set_value("vitd_level", 1)
        else:
            S.status = "The frozen support measure emitted no population."
    except Exception as exc:
        S.status = f"Decomposition failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def decompose_callback():
    if S.busy or S.lightness is None:
        return
    S.busy = True
    S.status = (
        "Frozen Meyer measure → simultaneous density population "
        "→ one Eikonal march...")
    threading.Thread(
        target=decompose_worker,
        args=(_config_from_ui(),),
        daemon=True,
    ).start()


def dirty_callback(sender=None, app_data=None):
    del sender, app_data
    S.dirty = True


def build_ui(labels: list[str]):
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=file_callback,
        tag="vitd_file_dialog",
        default_path=str(Path.home()),
        width=900,
        height=520,
    ):
        dpg.add_file_extension(
            "Image (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
            "{.png,.jpg,.jpeg,.tif,.tiff,.bmp}")
        dpg.add_file_extension(".*")

    with dpg.window(tag="vitd_root"):
        with dpg.group(horizontal=True):
            dpg.add_text("Gallery")
            dpg.add_combo(
                labels,
                default_value=labels[0],
                width=340,
                callback=gallery_callback,
            )
            dpg.add_button(
                label="Load image...",
                callback=lambda: dpg.show_item("vitd_file_dialog"),
            )
            dpg.add_button(
                label="Decompose",
                callback=decompose_callback,
                tag="vitd_decompose",
            )
        dpg.add_text(S.status, tag="vitd_status", wrap=1500)

        with dpg.collapsing_header(
            label="Intrinsic Voronoi baseline",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("Frozen Meyer density Voronoi")
                dpg.add_slider_int(
                    label="Meyer stages",
                    tag="vitd_tgfd",
                    default_value=2,
                    min_value=1,
                    max_value=24,
                    width=220,
                )
                dpg.add_slider_int(
                    label="outer-defect stages",
                    tag="vitd_flow",
                    default_value=2,
                    min_value=1,
                    max_value=24,
                    width=250,
                )
                dpg.add_slider_float(
                    label="knot blend alpha",
                    tag="vitd_alpha",
                    default_value=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    width=280,
                )
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="metric strength",
                    tag="vitd_metric",
                    default_value=1.5,
                    min_value=0.0,
                    max_value=6.0,
                    width=280,
                )
                dpg.add_slider_float(
                    label="boundary jump action",
                    tag="vitd_jump",
                    default_value=48.0,
                    min_value=0.0,
                    max_value=96.0,
                    width=280,
                )
                dpg.add_slider_float(
                    label="null-evidence suppression",
                    tag="vitd_null",
                    default_value=1.0,
                    min_value=0.0,
                    max_value=1.0,
                    width=300,
                )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="analytic curvature population",
                    tag="vitd_curve",
                    default_value=True,
                )
                dpg.add_slider_float(
                    label="coherent tangent floor",
                    tag="vitd_tangent",
                    default_value=0.08,
                    min_value=0.0,
                    max_value=0.3,
                    width=280,
                )
                dpg.add_slider_float(
                    label="texture geometry weight",
                    tag="vitd_texture_weight",
                    default_value=0.65,
                    min_value=0.0,
                    max_value=1.0,
                    width=300,
                )
                dpg.add_slider_int(
                    label="allocation max side",
                    tag="vitd_allocation_side",
                    default_value=256,
                    min_value=128,
                    max_value=512,
                    width=260,
                )
            dpg.add_text(
                "The C++ Meyer split is measured once. Its tensor determinant "
                "emits every gold germ simultaneously; there is no extrema "
                "spacing, candidate search, birth loop, Lloyd motion, or "
                "support diffusion. One reduced-basis first-arrival march "
                "forms the cells. Cell means and the actual interface graph "
                "produce the bounded intrinsic-amplitude baseline.")

        with dpg.group(horizontal=True):
            dpg.add_slider_int(
                label="display level",
                tag="vitd_level",
                default_value=1,
                min_value=1,
                max_value=1,
                width=300,
                callback=dirty_callback,
            )
            dpg.add_slider_float(
                label="rotation/detail gain",
                tag="vitd_gain",
                default_value=1.0,
                min_value=-1.0,
                max_value=4.0,
                width=340,
                callback=dirty_callback,
            )
            dpg.add_text("", tag="vitd_timing")

        dpg.add_separator()
        titles = (
            "source",
            "selected baseline",
            "selected rotation",
            "intrinsic Voronoi support",
            "all-level recomposition",
        )
        with dpg.group(horizontal=True):
            for title, image, texture in zip(titles, IMAGES, TAGS):
                with dpg.group():
                    dpg.add_text(title)
                    dpg.add_image(texture, tag=image)

    dpg.create_viewport(
        title="BFFT 2-D Voronoi intrinsic time-scale decomposition",
        width=1510,
        height=850,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("vitd_root", True)


def main() -> int:
    keys = gallery.available()
    if not keys:
        print("No gallery images available.")
        return 1
    labels = gallery.labels(keys)
    dpg.create_context()
    allocate_textures(8, 8)
    build_ui(labels)
    gallery_callback(None, labels[0])
    last_update = 0.0
    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        if now - last_update >= 0.05:
            dpg.set_value("vitd_status", S.status)
            dpg.configure_item("vitd_decompose", enabled=not S.busy)
            if S.result is not None:
                dpg.set_value("vitd_timing", f"{S.elapsed_ms:.0f} ms")
            last_update = now
        if S.dirty:
            S.dirty = False
            panels = render_panels()
            if panels is not None:
                if panels[0].shape[:2] != tuple(TEXTURE_SHAPE):
                    allocate_textures(*panels[0].shape[:2])
                push_panels(panels)
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
