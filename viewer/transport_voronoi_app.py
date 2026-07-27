#!/usr/bin/env python3
"""DearPyGui laboratory for BFFT-guided anisotropic Voronoi reconstruction.

Run:
    .venv/bin/python viewer/transport_voronoi_app.py
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments"))

import dearpygui.dearpygui as dpg  # noqa: E402
import gallery  # noqa: E402
from transport_voronoi import Config, TransportVoronoi  # noqa: E402
from claude_trial_sigma import SigmaVoronoi  # noqa: E402
from bfft.vision import vision_backend  # noqa: E402
from receiver_guided_graph import ReceiverGuidedVoronoi  # noqa: E402
from marching_fusion import marching_fusion  # noqa: E402


VIEWS = [
    "Reconstruction + sites", "Reconstruction",
    "Cartoon-cell reconstruction", "Texture-cell correction",
    "Legacy BFFT recomposition", "Cells", "Soft fusion groups", "Error",
    "Allocation pressure", "Expected affine gain",
    "Recursive residual memory",
    "Composition discrepancy", "Texture demand", "Texture activity",
    "Gradient consistency", "Seed density", "Persistent cartoon",
    "Detail field", "Cartoon", "Texture", "TV flow",
    "Flow magnitude", "Original",
]
TEX_SOURCE = "tv_source"
TEX_RESULT = "tv_result"
PANEL = 620


class State:
    def __init__(self):
        self.image = None
        self.name = "(none)"
        self.model = None
        self.busy = False
        self.run = False
        self.status = "Choose an image and initialize."
        self.precision_target = 1.0
        self.lock = threading.Lock()
        self.buffers = {}
        self.shape = (8, 8)


S = State()


def alloc_textures(h, w):
    S.shape = (h, w)
    for tag in (TEX_SOURCE, TEX_RESULT):
        S.buffers[tag] = np.ones(h * w * 4, dtype=np.float32)
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        if dpg.does_alias_exist(tag):
            dpg.remove_alias(tag)
    with dpg.texture_registry():
        dpg.add_raw_texture(w, h, S.buffers[TEX_SOURCE], tag=TEX_SOURCE,
                            format=dpg.mvFormat_Float_rgba)
        dpg.add_raw_texture(w, h, S.buffers[TEX_RESULT], tag=TEX_RESULT,
                            format=dpg.mvFormat_Float_rgba)
    scale = PANEL / max(h, w)
    for item, tag in (("source_image", TEX_SOURCE),
                      ("result_image", TEX_RESULT)):
        if dpg.does_item_exist(item):
            dpg.configure_item(item, texture_tag=tag,
                               width=max(1, int(w * scale)),
                               height=max(1, int(h * scale)))


def push_texture(tag, rgb):
    a = np.asarray(rgb, dtype=np.float32)
    if max(a.shape[:2]) > PANEL:
        from transport_voronoi import _fit_rgb
        a = np.asarray(_fit_rgb(a, PANEL), dtype=np.float32)
    if a.shape[:2] != S.shape:
        alloc_textures(*a.shape[:2])
    b = S.buffers[tag]
    b[0::4] = a[..., 0].ravel()
    b[1::4] = a[..., 1].ravel()
    b[2::4] = a[..., 2].ravel()
    dpg.set_value(tag, b)


def config_from_ui():
    return Config(
        max_side=(
            0 if bool(dpg.get_value("full_resolution"))
            else int(dpg.get_value("max_side"))),
        passes=int(dpg.get_value("passes")),
        lam=float(dpg.get_value("lam")),
        mu=float(dpg.get_value("mu")),
        flow_sweeps=int(dpg.get_value("flow_sweeps")),
        marked_cells=False,
        territory_count=1,
        initial_cells=int(dpg.get_value("initial_cells")),
        max_cells=int(dpg.get_value("max_cells")),
        split_batch=int(dpg.get_value("split_batch")),
        edge_density=float(dpg.get_value("edge_density")),
        texture_density=float(dpg.get_value("texture_density")),
        seed_bias=float(dpg.get_value("seed_bias")),
        anisotropy=float(dpg.get_value("anisotropy")),
        density_anisotropy=0.0,
        entropy_threshold=0.0,
        entropy_gain=1.0,
        consistency_weight=0.0,
        texture_reach_scale=1.0,
        edge_barrier=float(dpg.get_value("edge_barrier")),
        site_reach=float(dpg.get_value("site_reach")),
        softness=float(dpg.get_value("softness")),
        lloyd=float(dpg.get_value("lloyd")),
        shade_c=float(dpg.get_value("shade_c")),
        detail_precision=float(dpg.get_value("detail_precision")),
        bounded_gradients=bool(dpg.get_value("bounded_gradients")),
        recursive_memory_stages=int(dpg.get_value("memory_stages")),
        residual_memory_weight=float(dpg.get_value("memory_weight")),
        composition_discrepancy_weight=float(
            dpg.get_value("discrepancy_weight")),
        allocation_mode=str(dpg.get_value("allocation_mode")),
    )


def refresh():
    with S.lock:
        model = S.model
    if model is None:
        return
    model.cfg.legacy_cartoon_gain = float(
        dpg.get_value("legacy_cartoon_gain"))
    model.cfg.legacy_texture_gain = float(
        dpg.get_value("legacy_texture_gain"))
    model.cfg.legacy_shade_gain = float(
        dpg.get_value("legacy_shade_gain"))
    push_texture(TEX_SOURCE, model.rgb)
    view = dpg.get_value("view")
    if (view == "Soft fusion groups" and
            hasattr(model, "fusion_labels") and
            len(model.fusion_labels) == len(model.seeds)):
        group = model.fusion_labels[model.owner].reshape(model.h, model.w)
        phase = group.astype(np.float64)
        product = np.stack([
            0.5 + 0.5 * np.sin(phase * 2.399 + 0.0),
            0.5 + 0.5 * np.sin(phase * 2.399 + 2.094),
            0.5 + 0.5 * np.sin(phase * 2.399 + 4.189),
        ], axis=2)
        result = 0.72 * product + 0.28 * model.rgb
    else:
        result = model.view(view)
    push_texture(TEX_RESULT, result)
    if model.decomp_metrics_fresh:
        decomp_metrics = (
            f"cartoon MSE {model.cartoon_decomp_mse:.3e} | "
            f"texture MSE {model.texture_decomp_mse:.3e}")
    else:
        decomp_metrics = "decomposition MSE deferred"
    dpg.set_value(
        "metrics",
        f"cells {len(model.seeds)} | iteration {model.iteration} | "
        f"PSNR {model.psnr:.2f} dB | MSE {model.rgb_mse:.3e} | "
        f"{decomp_metrics} | "
        f"last {model.last_ms:.0f} ms | "
        f"BFFT geometry {model.split_ms:.0f} ms | "
        f"{model.w}x{model.h} {vision_backend()}")


def initialize_worker():
    if S.busy or S.image is None:
        return
    S.busy = True
    S.run = False
    S.status = "BFFT decomposition -> transport tensor -> coarse cells..."
    try:
        model = ReceiverGuidedVoronoi(S.image, config_from_ui())
        with S.lock:
            S.model = model
        S.status = (
            f"{S.name}: initialized {model.w}x{model.h} with "
            f"{len(model.seeds)} cartoon-guided cells.")
    except Exception as exc:
        S.status = f"Initialize failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_initialize():
    if S.image is None:
        S.status = "Choose an image first."
        return
    if not S.busy:
        threading.Thread(target=initialize_worker, daemon=True).start()


def step_worker(count):
    if S.busy:
        return
    with S.lock:
        model = S.model
    if model is None:
        S.status = "Initialize first."
        return
    S.busy = True
    try:
        # Live controls alter rendering/topology policy without throwing away
        # the current hierarchy.
        cfg = config_from_ui()
        model.cfg.max_cells = cfg.max_cells
        model.cfg.split_batch = cfg.split_batch
        model.cfg.site_reach = cfg.site_reach
        model.cfg.softness = cfg.softness
        model.cfg.lloyd = cfg.lloyd
        model.cfg.bounded_gradients = cfg.bounded_gradients
        model.cfg.residual_memory_weight = cfg.residual_memory_weight
        model.cfg.composition_discrepancy_weight = (
            cfg.composition_discrepancy_weight)
        if model.cfg.allocation_mode != cfg.allocation_mode:
            model.cfg.allocation_mode = cfg.allocation_mode
            model._update_allocation_pressure()
        for _ in range(count):
            if (isinstance(model, SigmaVoronoi) and
                    bool(dpg.get_value("exact_each_step"))):
                model.step_direct(
                    split=True,
                    cartoon_softness=float(
                        dpg.get_value("coupled_cartoon_softness")),
                    texture_softness=float(
                        dpg.get_value("coupled_texture_softness")))
            else:
                model.step(split=True)
            if model.stagnation >= 3:
                S.run = False
                break
        if len(model.seeds) >= model.cfg.max_cells and model.stagnation >= 3:
            S.status = (
                f"Budget stabilized at {model.cfg.max_cells} cells and "
                f"{model.psnr:.2f} dB after three rejected exchanges.")
            return
        S.status = (
            f"{model.last_action}: {len(model.seeds)} cells, "
            f"{model.psnr:.2f} dB ({model.last_gain:+.3f} dB).")
    except Exception as exc:
        S.status = f"Step failed: {type(exc).__name__}: {exc}"
        S.run = False
    finally:
        S.busy = False


def cb_step():
    threading.Thread(target=step_worker, args=(1,), daemon=True).start()


def coupled_worker():
    if S.busy:
        return
    with S.lock:
        model = S.model
    if model is None:
        S.status = "Initialize first."
        return
    S.busy = True
    S.run = False
    S.status = "Factoring the exact cell interaction graph..."
    try:
        before = model.psnr
        model.solve_direct_coupled(
            cartoon_softness=float(
                dpg.get_value("coupled_cartoon_softness")),
            texture_softness=float(
                dpg.get_value("coupled_texture_softness")))
        S.status = (
            f"Exact coupled solve: {before:.2f} → "
            f"{model.psnr:.2f} dB in {model.last_ms:.0f} ms.")
    except Exception as exc:
        S.status = f"Coupled solve failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_coupled():
    threading.Thread(target=coupled_worker, daemon=True).start()


def decomposition_metrics_worker():
    if S.busy:
        return
    with S.lock:
        model = S.model
    if model is None:
        S.status = "Initialize first."
        return
    S.busy = True
    S.run = False
    S.status = "Measuring the reconstruction's cartoon/texture mismatch..."
    try:
        started = time.perf_counter()
        cartoon_mse, texture_mse = model.refresh_decomposition_metrics()
        elapsed = (time.perf_counter() - started) * 1000.0
        S.status = (
            f"Decomposition metrics: cartoon {cartoon_mse:.3e}, "
            f"texture {texture_mse:.3e} ({elapsed:.0f} ms).")
    except Exception as exc:
        S.status = (
            f"Decomposition measurement failed: "
            f"{type(exc).__name__}: {exc}")
    finally:
        S.busy = False


def cb_decomposition_metrics():
    threading.Thread(
        target=decomposition_metrics_worker, daemon=True).start()


def reach_worker(enriched=False):
    if S.busy:
        return
    with S.lock:
        model = S.model
    if model is None:
        S.status = "Initialize first."
        return
    if not isinstance(model, SigmaVoronoi):
        S.status = "Re-initialize to use the merged exact backend."
        return
    S.busy = True
    S.run = False
    before = float(model.psnr)
    steps = int(dpg.get_value("reach_steps"))
    S.status = (
        "Optimizing cell reach on the renderer's measured graph"
        + (" and measuring bounded ridges..." if enriched else "..."))
    try:
        report = model.optimize_site_reach(
            steps=steps,
            learning_rate=float(dpg.get_value("reach_rate")),
            cartoon_softness=float(
                dpg.get_value("coupled_cartoon_softness")),
            texture_softness=float(
                dpg.get_value("coupled_texture_softness")),
            enriched=enriched)
        lo, hi = report["offset_span"]
        S.status = (
            f"{model.last_action}: {before:.2f} → {model.psnr:.2f} dB; "
            f"{report['steps']} accepted/evaluated reach steps; "
            f"offset {lo:+.2f}…{hi:+.2f}; {model.last_ms:.0f} ms.")
    except Exception as exc:
        S.status = f"Reach optimization failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_reach():
    threading.Thread(target=reach_worker, args=(False,), daemon=True).start()


def cb_reach_ridge():
    threading.Thread(target=reach_worker, args=(True,), daemon=True).start()


def stronger_round_worker():
    if S.busy:
        return
    with S.lock:
        model = S.model
    if model is None or not isinstance(model, ReceiverGuidedVoronoi):
        S.status = "Initialize the current high-performance model first."
        return
    S.busy = True
    S.run = False
    before = float(model.psnr)
    S.status = "Taking joint reach + overlap trust rounds..."
    try:
        report = model.receiver_trust(
            outer_steps=int(dpg.get_value("trust_rounds")),
            cartoon_softness=float(
                dpg.get_value("coupled_cartoon_softness")),
            texture_softness=float(
                dpg.get_value("coupled_texture_softness")),
            damping=float(dpg.get_value("trust_damping")),
            trust=float(dpg.get_value("trust_radius")),
            accept_objective=(
                "composite"
                if dpg.get_value("trust_acceptance") == "Composite"
                else "rgb"),
            joint_softness=True)
        dpg.set_value(
            "coupled_cartoon_softness", report["cartoon_softness"])
        dpg.set_value(
            "coupled_texture_softness", report["texture_softness"])
        S.status = (
            f"Stronger round: {before:.2f} → {model.psnr:.2f} dB; "
            f"{report['outer_steps']} rounds, "
            f"{report['evaluations']} measured proposals; "
            f"overlap {report['cartoon_softness']:.2f} / "
            f"{report['texture_softness']:.2f}.")
    except Exception as exc:
        S.status = f"Stronger round failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_stronger_round():
    threading.Thread(target=stronger_round_worker, daemon=True).start()


def fusion_worker():
    if S.busy:
        return
    with S.lock:
        model = S.model
    if model is None or not isinstance(model, SigmaVoronoi):
        S.status = "Initialize the current high-performance model first."
        return
    S.busy = True
    S.run = False
    before = float(model.psnr)
    S.status = "Marching compatible support fusions..."
    try:
        report = marching_fusion(
            model, rounds=int(dpg.get_value("fusion_rounds")),
            fraction=float(dpg.get_value("fusion_fraction")),
            quantile=float(dpg.get_value("fusion_quantile")),
            objective_tolerance=float(
                dpg.get_value("fusion_objective_tolerance")),
            psnr_tolerance=float(dpg.get_value("fusion_psnr_tolerance")),
            cartoon_softness=float(
                dpg.get_value("coupled_cartoon_softness")),
            texture_softness=float(
                dpg.get_value("coupled_texture_softness")))
        S.status = (
            f"Soft fusion: {report['cells']} cells → "
            f"{report['groups']} affine groups; "
            f"{report['fused']} internal boundaries tied; "
            f"{before:.2f} → {model.psnr:.2f} dB.")
        dpg.set_value("view", "Soft fusion groups")
    except Exception as exc:
        S.status = f"Fusion failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def cb_fusion():
    threading.Thread(target=fusion_worker, daemon=True).start()


def cb_run():
    S.run = not S.run
    dpg.configure_item("run_button",
                       label="Pause" if S.run else "Run continuously")


def cb_view():
    refresh()


def cb_precision(sender=None, value=None):
    S.precision_target = float(
        dpg.get_value("detail_precision") if value is None else value)


def cb_fade():
    S.precision_target = 0.0


def cb_refresh_detail():
    S.precision_target = 1.0


def cb_full_resolution(sender=None, value=None):
    enabled = bool(
        dpg.get_value("full_resolution") if value is None else value)
    dpg.configure_item("max_side", enabled=not enabled)
    if S.image is not None:
        h, w = S.image.shape[:2]
        mode = (
            f"every source pixel ({w}x{h})" if enabled
            else f"maximum side {int(dpg.get_value('max_side'))}")
        S.status = f"Work resolution set to {mode}. Press Initialize."


def adopt(image, name):
    S.run = False
    S.image = np.asarray(image, dtype=np.float64)
    S.name = name
    S.precision_target = 1.0
    with S.lock:
        S.model = None
    source_h, source_w = S.image.shape[:2]
    S.status = (
        f"{name} loaded at {source_w}x{source_h}. "
        "Press Initialize.")
    a = S.image
    if a.ndim == 2:
        g = a / 255.0 if a.max() > 1.5 else a
        rgb = np.stack([g, g, g], axis=-1)
    else:
        rgb = a[..., :3] / 255.0 if a.max() > 1.5 else a[..., :3]
    # Initial preview may be large; the model will replace it with its fitted
    # work resolution on initialization.
    from transport_voronoi import _fit_rgb
    work_side = (
        0 if bool(dpg.get_value("full_resolution"))
        else int(dpg.get_value("max_side")))
    rgb = _fit_rgb(rgb, work_side)
    push_texture(TEX_SOURCE, rgb)
    push_texture(TEX_RESULT, np.full_like(rgb, 0.08))


def cb_gallery(sender, label):
    try:
        key = gallery.key_for_label(label)
        adopt(gallery.load(key), gallery.describe(key)["label"])
    except Exception as exc:
        S.status = f"Gallery load failed: {type(exc).__name__}: {exc}"


def cb_file(sender, app_data):
    sels = app_data.get("selections") or {}
    candidates = list(sels.values())
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


def build_ui(labels, default_label):
    with dpg.file_dialog(directory_selector=False, show=False,
                         callback=cb_file, tag="file_dialog", width=900,
                         height=520):
        dpg.add_file_extension(
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
            "{.png,.jpg,.jpeg,.tif,.tiff,.bmp}")
        dpg.add_file_extension(".*")

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            dpg.add_text("Image")
            dpg.add_combo(labels, default_value=default_label, tag="gallery",
                          width=390, callback=cb_gallery)
            dpg.add_button(label="Load image...",
                           callback=lambda: dpg.show_item("file_dialog"))
            dpg.add_button(label="Initialize", callback=cb_initialize,
                           tag="initialize_button")
            dpg.add_button(label="One subdivision", callback=cb_step,
                           tag="step_button")
            dpg.add_button(label="Run continuously", callback=cb_run,
                           tag="run_button")
            dpg.add_button(label="Exact couple", callback=cb_coupled,
                           tag="coupled_button")
        dpg.add_text("", tag="status", wrap=1500)
        dpg.add_text("", tag="metrics")
        dpg.add_separator()

        with dpg.collapsing_header(label="BFFT geometry", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="process every source pixel (HD/full resolution)",
                    tag="full_resolution", default_value=False,
                    callback=cb_full_resolution)
                dpg.add_slider_int(label="work max side", tag="max_side",
                                   default_value=256, min_value=96,
                                   max_value=2160, width=360)
                dpg.add_slider_int(label="TGFD passes", tag="passes",
                                   default_value=24, min_value=4,
                                   max_value=64, width=280)
                dpg.add_slider_float(label="mu", tag="mu",
                                     default_value=40, min_value=4,
                                     max_value=120, width=280)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(label="cartoon lambda", tag="lam",
                                     default_value=0.05, min_value=0.01,
                                     max_value=0.12, width=280,
                                     format="%.3f")
                dpg.add_slider_int(label="flow TV sweeps",
                                   tag="flow_sweeps", default_value=64,
                                   min_value=8, max_value=240, width=280)
                dpg.add_slider_float(label="cartoon edge density",
                                     tag="edge_density", default_value=4,
                                     min_value=0, max_value=12, width=320)
                dpg.add_slider_float(label="texture density",
                                     tag="texture_density", default_value=3,
                                     min_value=0, max_value=12, width=320)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(label="initial content bias",
                                     tag="seed_bias", default_value=0,
                                     min_value=0, max_value=0.5, width=320)
                dpg.add_slider_float(label="cartoon crossing barrier",
                                     tag="edge_barrier", default_value=12,
                                     min_value=0, max_value=40, width=320)
                dpg.add_slider_float(label="legacy shading c",
                                     tag="shade_c", default_value=0.02,
                                     min_value=0.004, max_value=0.12,
                                     width=280, format="%.3f")
                dpg.add_text(
                    "Geometry controls above apply when Initialize is pressed.")

        with dpg.collapsing_header(label="Cells and reconstruction",
                                   default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_slider_int(label="initial cells",
                                   tag="initial_cells", default_value=180,
                                   min_value=20, max_value=5000, width=280)
                dpg.add_slider_int(label="maximum cells", tag="max_cells",
                                   default_value=2400, min_value=100,
                                   max_value=20000, width=320)
                dpg.add_slider_int(label="split batch", tag="split_batch",
                                   default_value=48, min_value=1,
                                   max_value=200, width=280)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(label="anisotropy",
                                     tag="anisotropy", default_value=5,
                                     min_value=0, max_value=12, width=320)
                dpg.add_slider_float(label="site reach",
                                     tag="site_reach", default_value=1.5,
                                     min_value=0, max_value=8, width=320)
                dpg.add_slider_float(label="ownership softness",
                                     tag="softness", default_value=10,
                                     min_value=0, max_value=40, width=320)
                dpg.add_slider_float(label="centroid yield",
                                     tag="lloyd", default_value=0,
                                     min_value=0, max_value=0.8, width=320)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="bound local gradients",
                                 tag="bounded_gradients", default_value=True)
                dpg.add_checkbox(
                    label="exact fused fit after each subdivision",
                    tag="exact_each_step", default_value=True)
                dpg.add_combo(
                    ["Expected affine gain", "RGB + decomposition gain",
                     "Robust integrated error"],
                    default_value="Expected affine gain",
                    label="allocation currency", tag="allocation_mode",
                    width=300)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="coupled cartoon overlap",
                    tag="coupled_cartoon_softness", default_value=4.0,
                    min_value=1.0, max_value=20.0, width=320)
                dpg.add_slider_float(
                    label="coupled texture sharpness",
                    tag="coupled_texture_softness", default_value=16.0,
                    min_value=1.0, max_value=30.0, width=320)
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="reach descent steps", tag="reach_steps",
                    default_value=18, min_value=1, max_value=40, width=280)
                dpg.add_slider_float(
                    label="reach step size", tag="reach_rate",
                    default_value=0.35, min_value=0.05, max_value=0.8,
                    width=280, format="%.2f")
                dpg.add_button(
                    label="Optimize site reach", callback=cb_reach,
                    tag="reach_button")
                dpg.add_button(
                    label="Reach + measured ridge",
                    callback=cb_reach_ridge, tag="reach_ridge_button")
                dpg.add_text(
                    "Best after growing to the chosen maximum cell budget.")

        with dpg.collapsing_header(
                label="Stronger rounds and marching fusion",
                default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="joint trust rounds", tag="trust_rounds",
                    default_value=3, min_value=1, max_value=12, width=260)
                dpg.add_slider_float(
                    label="trust damping", tag="trust_damping",
                    default_value=0.05, min_value=0.005, max_value=0.5,
                    width=280, format="%.3f")
                dpg.add_slider_float(
                    label="trust radius", tag="trust_radius",
                    default_value=1.5, min_value=0.25, max_value=4.0,
                    width=280)
                dpg.add_button(
                    label="Joint reach + overlap round",
                    callback=cb_stronger_round, tag="stronger_round_button")
                dpg.add_combo(
                    ["Composite", "RGB only"],
                    default_value="Composite",
                    label="acceptance", tag="trust_acceptance", width=150)
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="fusion fronts", tag="fusion_rounds",
                    default_value=6, min_value=1, max_value=20, width=240)
                dpg.add_slider_float(
                    label="front fraction", tag="fusion_fraction",
                    default_value=0.08, min_value=0.01, max_value=0.25,
                    width=260, format="%.2f")
                dpg.add_slider_float(
                    label="compatibility quantile", tag="fusion_quantile",
                    default_value=0.35, min_value=0.05, max_value=0.8,
                    width=280, format="%.2f")
                dpg.add_button(
                    label="March soft fusion", callback=cb_fusion,
                    tag="fusion_button")
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="objective loss budget",
                    tag="fusion_objective_tolerance",
                    default_value=0.0025, min_value=0.0, max_value=0.02,
                    width=300, format="%.4f")
                dpg.add_slider_float(
                    label="PSNR loss budget", tag="fusion_psnr_tolerance",
                    default_value=0.03, min_value=0.0, max_value=0.25,
                    width=300, format="%.3f")
                dpg.add_text(
                    "Fusion ties affine jets first; it does not delete sites.")

        with dpg.collapsing_header(label="Perceptual detail gate",
                                   default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="detail precision", tag="detail_precision",
                    default_value=1.0, min_value=0, max_value=1.25,
                    width=420, callback=cb_precision)
                dpg.add_button(label="Fade detail", callback=cb_fade)
                dpg.add_button(label="Refresh detail",
                               callback=cb_refresh_detail)

        with dpg.collapsing_header(
                label="Refinement guidance (placement only)",
                default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    label="residual memory stages", tag="memory_stages",
                    default_value=1, min_value=1, max_value=1, width=300)
                dpg.add_slider_float(
                    label="recursive residual guidance",
                    tag="memory_weight", default_value=0.0,
                    min_value=0.0, max_value=3.0, width=320)
                dpg.add_slider_float(
                    label="composition discrepancy guidance",
                    tag="discrepancy_weight", default_value=0.0,
                    min_value=0.0, max_value=3.0, width=340)
                dpg.add_text(
                    "Optional focus guidance; allocation currency is chosen "
                    "above. Residual memory remains one stage.")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Measure cartoon + texture MSE",
                    callback=cb_decomposition_metrics,
                    tag="decomposition_metrics_button")
                dpg.add_text(
                    "Explicit on HD: this runs a full single-stage "
                    "decomposition of the current result.")

        with dpg.collapsing_header(label="Legacy BFFT recomposition",
                                   default_open=False):
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="cartoon gain", tag="legacy_cartoon_gain",
                    default_value=1.0, min_value=0.0, max_value=2.0,
                    width=280)
                dpg.add_slider_float(
                    label="texture gain", tag="legacy_texture_gain",
                    default_value=1.0, min_value=-2.0, max_value=6.0,
                    width=280)
                dpg.add_slider_float(
                    label="shading gain", tag="legacy_shade_gain",
                    default_value=0.0, min_value=-1.0, max_value=4.0,
                    width=280)

        with dpg.group(horizontal=True):
            dpg.add_text("Right panel")
            dpg.add_combo(VIEWS, default_value=VIEWS[0], tag="view",
                          callback=cb_view, width=250)

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image(TEX_SOURCE, tag="source_image")
            with dpg.group():
                dpg.add_text("BFFT transport cells")
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
        title="BFFT Vision — Native Transport Cells",
        width=1450, height=980)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)
    cb_gallery(None, default_label)
    cb_initialize()

    last_refresh = 0.0
    last_frame = time.perf_counter()
    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dt = min(now - last_frame, 0.1)
        last_frame = now
        dpg.set_value("status", S.status)
        dpg.configure_item(
            "run_button", label="Pause" if S.run else "Run continuously")
        dpg.configure_item("initialize_button", enabled=not S.busy)
        dpg.configure_item("step_button", enabled=not S.busy)
        dpg.configure_item("coupled_button", enabled=not S.busy)
        dpg.configure_item(
            "decomposition_metrics_button", enabled=not S.busy)
        dpg.configure_item("reach_button", enabled=not S.busy)
        dpg.configure_item("reach_ridge_button", enabled=not S.busy)
        dpg.configure_item("stronger_round_button", enabled=not S.busy)
        dpg.configure_item("fusion_button", enabled=not S.busy)
        if S.run and not S.busy:
            threading.Thread(target=step_worker, args=(1,), daemon=True).start()
        with S.lock:
            model = S.model
        if model is not None and not S.busy:
            current = float(model.cfg.detail_precision)
            tau = 0.7 if S.precision_target < current else 0.22
            blend = 1.0 - math.exp(-dt / tau)
            updated = current + blend * (S.precision_target - current)
            if abs(updated - current) > 1e-5:
                model.set_detail_precision(updated)
                dpg.set_value("detail_precision", updated)
        if now - last_refresh > 0.12:
            refresh()
            last_refresh = now
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
