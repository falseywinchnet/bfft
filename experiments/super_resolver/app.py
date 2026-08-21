"""Dear PyGui scaffold for measured single-image super-resolution."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import numpy as np

from . import datasets
from .core import (
    DECIMATION_MODES,
    SUPPORT_MODES,
    SuperResolutionConfig,
    as_rgb,
    focus_crop,
    prepare_observation,
    run_eikonal_upscale,
)


PANEL_SIDE = 460
VIEW_TAGS = ("sr_left_view", "sr_middle_view", "sr_right_view")
IMAGE_TAGS = ("sr_left_image", "sr_middle_image", "sr_right_image")


class SuperResolverApp:
    def __init__(self, dpg):
        self.dpg = dpg
        self.source: np.ndarray | None = None
        self.source_name = "(none)"
        self.prepared = None
        self.result = None
        self.worker: threading.Thread | None = None
        self.worker_result = None
        self.worker_error: str | None = None
        self.busy = False
        self.worker_kind: str | None = None
        self.worker_config: SuperResolutionConfig | None = None
        self.worker_source_name = "(none)"
        self.lock = threading.Lock()
        # Keep three fixed-size raw textures alive for the full GUI lifetime.
        # Dear PyGui's Metal backend can retain a texture for a rendered frame;
        # deleting/replacing it in a callback leaves the command encoder with a
        # dangling GPU object and causes SIGBUS in setFragmentTexture.
        self.texture_buffers = [
            np.zeros(PANEL_SIDE * PANEL_SIDE * 4, dtype=np.float32)
            for _ in range(3)
        ]

    def config(self) -> SuperResolutionConfig:
        dpg = self.dpg
        scale = int(str(dpg.get_value("sr_scale"))[0])
        return SuperResolutionConfig(
            scale=scale,
            decimation=dpg.get_value("sr_decimation"),
            support=dpg.get_value("sr_support"),
            anisotropy=float(dpg.get_value("sr_anisotropy")),
            tensor_sigma=float(dpg.get_value("sr_tensor_sigma")),
            clamp_range=bool(dpg.get_value("sr_clamp")),
            maximum_side=int(dpg.get_value("sr_work_side")),
        )

    def adopt(self, image: np.ndarray, name: str):
        self.source = as_rgb(image)
        self.source_name = name
        self.prepared = None
        self.result = None
        self.dpg.set_value(
            "sr_status",
            f"Loaded {name}: {self.source.shape[1]} × {self.source.shape[0]}. Generate a decimated observation.",
        )
        self.dpg.set_value("sr_metrics", "")

    def choose_gallery(self, _sender=None, label=None):
        try:
            key = datasets.key_for_label(label)
            entry = next(x for x in datasets.available_entries() if x["key"] == key)
            self.adopt(datasets.load_gallery(key), entry["label"])
        except Exception:
            self.dpg.set_value("sr_status", traceback.format_exc())

    def choose_file(self, _sender=None, app_data=None):
        candidates = list((app_data.get("selections") or {}).values())
        candidates.append(app_data.get("file_path_name", ""))
        path = next((Path(item) for item in candidates if item and Path(item).is_file()), None)
        if path is None:
            self.dpg.set_value("sr_status", "Could not resolve the selected image.")
            return
        try:
            self.adopt(datasets.load_file(path), path.name)
        except Exception:
            self.dpg.set_value("sr_status", traceback.format_exc())

    def generate_observation(self):
        if self.source is None:
            self.dpg.set_value("sr_status", "Choose a V3 gallery image or local file first.")
            return
        if self.busy:
            return
        try:
            config = self.config()
            if config.decimation == "Eikonal prefilter":
                source = self.source.copy()
                self.prepared = None
                self.result = None
                self.worker_result = None
                self.worker_error = None
                self.worker_kind = "observation"
                self.worker_config = config
                self.worker_source_name = self.source_name
                self.busy = True
                self.dpg.configure_item("sr_generate", enabled=False)
                self.dpg.configure_item("sr_run", enabled=False)
                self.dpg.set_value("sr_metrics", "")
                self.dpg.set_value(
                    "sr_status",
                    f"Measuring the source-side tensor and generating a {config.scale}× "
                    "Eikonal-prefiltered observation…",
                )

                def work():
                    try:
                        value = prepare_observation(source, config)
                        with self.lock:
                            self.worker_result = value
                    except Exception:
                        with self.lock:
                            self.worker_error = traceback.format_exc()

                worker = threading.Thread(
                    target=work,
                    name="super-resolver-observation",
                    daemon=True,
                )
                # Dear PyGui executes callbacks off the render loop. Publish
                # only after start so poll_worker cannot join an unstarted
                # thread in the assignment/start interval.
                worker.start()
                self.worker = worker
            else:
                prepared = prepare_observation(self.source, config)
                self._publish_observation(prepared, config, self.source_name)
        except Exception:
            self.busy = False
            self.dpg.set_value("sr_status", traceback.format_exc())

    def _publish_observation(self, prepared, config, source_name):
        self.prepared = prepared
        self.result = None
        reference = prepared.reference
        observed = prepared.observed
        qualification = (
            " Source-side Eikonal geometry was used by the forward model, so "
            "these LR pixels are richer aggregates than point-decimated pixels."
            if config.decimation == "Eikonal prefilter"
            else ""
        )
        self.dpg.set_value(
            "sr_status",
            f"{source_name}: retained truth {reference.shape[1]} × {reference.shape[0]}; "
            f"{config.decimation.lower()} produced {observed.shape[1]} × {observed.shape[0]}. "
            f"The truth is now reserved for scoring.{qualification}",
        )
        self.dpg.set_value("sr_metrics", "")
        self.render()

    def start_upscale(self):
        if self.prepared is None:
            self.generate_observation()
        if self.prepared is None or self.worker is not None:
            return
        config = self.config()
        if config.scale != self.prepared.scale or config.decimation != self.prepared.decimation:
            self.dpg.set_value("sr_status", "Scale or decimation changed; generate the observation again.")
            return
        if self.prepared.decimation == "Eikonal prefilter" and (
            config.anisotropy != self.prepared.forward_anisotropy
            or config.tensor_sigma != self.prepared.forward_tensor_sigma
            or config.clamp_range != self.prepared.forward_clamp_range
        ):
            self.dpg.set_value(
                "sr_status",
                "An Eikonal forward-model control changed; generate the observation again.",
            )
            return
        self.worker_result = None
        self.worker_error = None
        self.worker_kind = "upscale"
        self.worker_config = config
        self.busy = True
        self.dpg.configure_item("sr_generate", enabled=False)
        self.dpg.configure_item("sr_run", enabled=False)
        self.dpg.set_value(
            "sr_status", f"Running {config.scale}× upscale with {config.support.lower()}…"
        )

        prepared = self.prepared

        def work():
            try:
                value = run_eikonal_upscale(prepared, config)
                with self.lock:
                    self.worker_result = value
            except Exception:
                with self.lock:
                    self.worker_error = traceback.format_exc()

        worker = threading.Thread(
            target=work, name="super-resolver", daemon=True
        )
        worker.start()
        self.worker = worker

    def poll_worker(self):
        if self.worker is None or self.worker.is_alive():
            return
        self.worker.join()
        self.worker = None
        self.busy = False
        self.dpg.configure_item("sr_generate", enabled=True)
        self.dpg.configure_item("sr_run", enabled=True)
        with self.lock:
            result = self.worker_result
            error = self.worker_error
            self.worker_result = None
            self.worker_error = None
        worker_kind = self.worker_kind
        worker_config = self.worker_config
        worker_source_name = self.worker_source_name
        self.worker_kind = None
        self.worker_config = None
        if error is not None:
            self.dpg.set_value("sr_status", error)
            return
        if worker_kind == "observation":
            self._publish_observation(
                result, worker_config, worker_source_name
            )
            return
        self.result = result
        for tag, value in zip(
            VIEW_TAGS,
            ("Reference HR", "Lanczos baseline", "Eikonal upscale"),
        ):
            self.dpg.set_value(tag, value)
        metric = result.metrics
        delta = metric["difference"]
        self.dpg.set_value(
            "sr_metrics",
            "Lanczos  "
            f"MSE {metric['Lanczos']['mse']:.7f}   SSIM {metric['Lanczos']['ssim']:.6f}   "
            f"fine MSE {metric['Lanczos']['fine_mse']:.7f}\n"
            "Eikonal  "
            f"MSE {metric['Eikonal']['mse']:.7f}   SSIM {metric['Eikonal']['ssim']:.6f}   "
            f"fine MSE {metric['Eikonal']['fine_mse']:.7f}\n"
            "Eikonal improvement  "
            f"ΔMSE {delta['mse']:+.7f}   ΔSSIM {delta['ssim']:+.6f}   "
            f"Δfine MSE {delta['fine_mse']:+.7f}",
        )
        self.dpg.set_value(
            "sr_status",
            f"Completed {result.prepared.scale}× reconstruction with {result.support_description}. "
            "Positive differences mean the Eikonal result improved on Lanczos.",
        )
        self.render()

    def _views(self):
        if self.result is not None:
            return self.result.views
        if self.prepared is not None:
            from .core import _pillow_resize
            from PIL import Image

            return {
                "Reference HR": self.prepared.reference,
                "Observed LR (nearest)": _pillow_resize(
                    self.prepared.observed,
                    self.prepared.reference.shape[:2],
                    Image.Resampling.NEAREST,
                ),
            }
        if self.source is not None:
            return {"Reference HR": self.source}
        return {}

    def _display_image(self, image):
        mode = self.dpg.get_value("sr_display_mode")
        if mode == "Focus crop":
            image = focus_crop(
                image,
                self.dpg.get_value("sr_focus_x") / 100.0,
                self.dpg.get_value("sr_focus_y") / 100.0,
                self.dpg.get_value("sr_focus_side"),
            )
        return as_rgb(image)

    def _replace_texture(self, slot: int, image: np.ndarray):
        dpg = self.dpg
        value = self._display_image(image).astype(np.float32)
        height, width = value.shape[:2]
        if max(height, width) > PANEL_SIDE:
            from .core import _pillow_resize
            from PIL import Image

            scale = PANEL_SIDE / max(height, width)
            value = _pillow_resize(
                value,
                (max(1, round(height * scale)), max(1, round(width * scale))),
                Image.Resampling.LANCZOS,
            ).astype(np.float32)
            height, width = value.shape[:2]
        buffer = self.texture_buffers[slot].reshape(PANEL_SIDE, PANEL_SIDE, 4)
        buffer.fill(0.0)
        buffer[:height, :width, :3] = value
        buffer[:height, :width, 3] = 1.0
        texture = f"sr_texture_{slot}"
        dpg.set_value(texture, self.texture_buffers[slot])
        display_scale = PANEL_SIDE / max(height, width)
        dpg.configure_item(
            IMAGE_TAGS[slot],
            texture_tag=texture,
            width=max(1, int(width * display_scale)),
            height=max(1, int(height * display_scale)),
            uv_min=(0.0, 0.0),
            uv_max=(width / PANEL_SIDE, height / PANEL_SIDE),
            show=True,
        )

    def render(self):
        views = self._views()
        if not views:
            return
        available = tuple(views)
        for slot, tag in enumerate(VIEW_TAGS):
            current = self.dpg.get_value(tag)
            if current not in views:
                current = available[min(slot, len(available) - 1)]
                self.dpg.set_value(tag, current)
            self.dpg.configure_item(tag, items=available)
            self._replace_texture(slot, views[current])


def build_ui(dpg, app: SuperResolverApp, labels: list[str], default_label: str):
    with dpg.texture_registry(show=False):
        for slot in range(3):
            dpg.add_raw_texture(
                PANEL_SIDE,
                PANEL_SIDE,
                app.texture_buffers[slot],
                tag=f"sr_texture_{slot}",
                format=dpg.mvFormat_Float_rgba,
            )
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=app.choose_file,
        tag="sr_file_dialog",
        width=900,
        height=520,
    ):
        dpg.add_file_extension(
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp){.png,.jpg,.jpeg,.tif,.tiff,.bmp}"
        )
        dpg.add_file_extension(".*")

    with dpg.window(tag="sr_root"):
        dpg.add_text("Single-observation Eikonal Super Resolver", color=(90, 220, 210))
        dpg.add_text(
            "The retained HR image is scoring truth only. This stage performs measured interpolation, not generative detail recovery.",
            wrap=1500,
        )
        with dpg.group(horizontal=True):
            dpg.add_combo(
                labels,
                default_value=default_label,
                width=440,
                tag="sr_gallery",
                callback=app.choose_gallery,
            )
            dpg.add_button(label="Load image…", callback=lambda: dpg.show_item("sr_file_dialog"))
            dpg.add_button(
                label="1. Generate decimated observation",
                callback=app.generate_observation,
                tag="sr_generate",
            )
            dpg.add_button(
                label="2. Run Eikonal upscale", callback=app.start_upscale, tag="sr_run"
            )
        with dpg.group(horizontal=True):
            dpg.add_combo(("2×", "4×"), default_value="2×", tag="sr_scale", label="scale", width=90)
            dpg.add_combo(
                DECIMATION_MODES,
                default_value=DECIMATION_MODES[0],
                tag="sr_decimation",
                label="observation",
                width=210,
            )
            with dpg.tooltip("sr_decimation"):
                dpg.add_text(
                    "Eikonal prefilter measures geometry on the HR source and "
                    "integrates it into fewer pixels. It is a richer forward "
                    "observation, not information-equivalent to point decimation.",
                    wrap=420,
                )
            dpg.add_combo(
                SUPPORT_MODES,
                default_value=SUPPORT_MODES[0],
                tag="sr_support",
                label="support",
                width=205,
            )
            dpg.add_slider_int(
                default_value=256,
                min_value=64,
                max_value=768,
                tag="sr_work_side",
                label="maximum working side",
                width=190,
            )
            dpg.add_slider_float(
                default_value=0.75,
                min_value=0.0,
                max_value=2.0,
                tag="sr_anisotropy",
                label="Eikonal anisotropy",
                width=180,
            )
            dpg.add_slider_float(
                default_value=1.0,
                min_value=0.0,
                max_value=3.0,
                tag="sr_tensor_sigma",
                label="tensor smoothing",
                width=160,
            )
            dpg.add_checkbox(default_value=True, tag="sr_clamp", label="same-support clamp")
        dpg.add_text("Choose an image, then generate its controlled low-resolution observation.", tag="sr_status", wrap=1500)
        dpg.add_text("", tag="sr_metrics", wrap=1500)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_combo(
                ("Full image", "Focus crop"),
                default_value="Full image",
                tag="sr_display_mode",
                label="display",
                width=130,
                callback=lambda: app.render(),
            )
            dpg.add_slider_float(
                default_value=50.0,
                min_value=0.0,
                max_value=100.0,
                tag="sr_focus_x",
                label="focus x %",
                width=180,
                callback=lambda: app.render(),
            )
            dpg.add_slider_float(
                default_value=50.0,
                min_value=0.0,
                max_value=100.0,
                tag="sr_focus_y",
                label="focus y %",
                width=180,
                callback=lambda: app.render(),
            )
            dpg.add_slider_int(
                default_value=96,
                min_value=16,
                max_value=256,
                tag="sr_focus_side",
                label="focus side (HR px)",
                width=190,
                callback=lambda: app.render(),
            )
        with dpg.group(horizontal=True):
            defaults = ("Reference HR", "Lanczos baseline", "Eikonal upscale")
            for slot, (view_tag, image_tag, default) in enumerate(zip(VIEW_TAGS, IMAGE_TAGS, defaults)):
                with dpg.child_window(width=PANEL_SIDE + 24, height=PANEL_SIDE + 78):
                    dpg.add_combo(
                        (default,),
                        default_value=default,
                        tag=view_tag,
                        width=PANEL_SIDE,
                        callback=lambda: app.render(),
                    )
                    dpg.add_image(
                        f"sr_texture_{slot}", tag=image_tag, show=False
                    )


def main():
    try:
        import dearpygui.dearpygui as dpg
    except ImportError as exc:  # pragma: no cover - GUI dependency
        raise SystemExit(
            "Dear PyGui is required. Install the vision-viewer optional dependencies."
        ) from exc

    entries = datasets.available_entries()
    if not entries:
        raise SystemExit("No V3 gallery entries are available on this machine.")
    labels = [datasets.label_for(entry) for entry in entries]
    default_entry = next((entry for entry in entries if entry["key"] == "astronaut"), entries[0])
    default_label = datasets.label_for(default_entry)

    dpg.create_context()
    app = SuperResolverApp(dpg)
    build_ui(dpg, app, labels, default_label)
    dpg.create_viewport(title="Eikonal Super Resolver", width=1510, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("sr_root", True)
    app.adopt(datasets.load_gallery(default_entry["key"]), default_entry["label"])
    app.render()
    while dpg.is_dearpygui_running():
        app.poll_worker()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
