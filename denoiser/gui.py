"""Dear PyGui archive for rejected FMMT and pre-FMMT denoising controls."""

from __future__ import annotations

import json
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from .cross_predictive_transport import (
        action_contracting_connection_readout_forms,
        denoise_cross_predictive_transport,
    )
    from .fmmt_certified import denoise_fmmt
    from .fabada_oracle import denoise_oracle_fabada_from_corruption_1d
    from .probes import hair_edge_scene
    from .sample_series import (
        COMPONENTS,
        CORRUPTIONS,
        DEFAULT_PARAMETERS,
        PRESETS,
        compose_series,
        corrupt,
    )
    from .transport_support import (
        TransportResolution,
        denoise_1d,
        denoise_2d_fmmt,
        support_density,
        transport_support_birth,
    )
except ImportError:
    from cross_predictive_transport import (
        action_contracting_connection_readout_forms,
        denoise_cross_predictive_transport,
    )
    from fmmt_certified import denoise_fmmt
    from fabada_oracle import denoise_oracle_fabada_from_corruption_1d
    from probes import hair_edge_scene
    from sample_series import (
        COMPONENTS,
        CORRUPTIONS,
        DEFAULT_PARAMETERS,
        PRESETS,
        compose_series,
        corrupt,
    )
    from transport_support import (
        TransportResolution,
        denoise_1d,
        denoise_2d_fmmt,
        support_density,
        transport_support_birth,
    )


SKIMAGE_SOURCES = {
    "camera — Cameraman": "camera",
    "coins": "coins",
    "moon": "moon",
    "page — printed page": "page",
    "text — handwriting": "text",
    "clock": "clock",
    "cell": "cell",
    "brick": "brick",
    "grass": "grass",
    "gravel": "gravel",
    "checkerboard": "checkerboard",
    "astronaut (grayscale)": "astronaut",
    "coffee (grayscale)": "coffee",
    "chelsea cat (grayscale)": "chelsea",
    "rocket (grayscale)": "rocket",
    "Hubble deep field (grayscale)": "hubble_deep_field",
}


LINE_PARAMETER_TAGS = {
    key: f"line_{key}" for key in DEFAULT_PARAMETERS
}


def _texture_data(image: np.ndarray) -> tuple[int, int, list[float]]:
    gray = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgba = np.concatenate((rgb, np.ones((*gray.shape, 1), np.float32)), axis=-1)
    return gray.shape[1], gray.shape[0], rgba.ravel().tolist()


def _gray(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value, dtype=np.float64)
    if image.ndim == 3:
        image = image[..., :3] @ np.array([0.2125, 0.7154, 0.0721])
    if float(np.max(image)) > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _fit_gray(value: np.ndarray, side: int) -> np.ndarray:
    image = _gray(value)
    height, width = image.shape
    scale = min(float(side) / max(height, width), 1.0)
    output = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
    pixels = np.uint8(np.round(image * 255.0))
    return np.asarray(
        Image.fromarray(pixels).resize(output, Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0


class DenoiserLab:
    def __init__(self, dpg):
        self.dpg = dpg
        self.line_x: np.ndarray | None = None
        self.line_truth: np.ndarray | None = None
        self.line_observed: np.ndarray | None = None
        self.line_output: np.ndarray | None = None
        self.clean_image: np.ndarray | None = None
        self.image: np.ndarray | None = None
        self.image_output: np.ndarray | None = None
        self.source_path: Path | None = None
        self.source_name = "(none)"
        self.texture_tags: list[str] = []

    def resolution(self) -> TransportResolution:
        return TransportResolution(
            scale_samples=int(self.dpg.get_value("scale_samples")),
            histogram_bins=int(self.dpg.get_value("histogram_bins")),
            maximum_steps=int(self.dpg.get_value("maximum_steps")),
        )

    # ------------------------------------------------------------------ 1-D

    def line_parameters(self) -> dict[str, float]:
        return {
            key: float(self.dpg.get_value(tag))
            for key, tag in LINE_PARAMETER_TAGS.items()
        }

    def active_components(self) -> set[str]:
        return {
            name for name in COMPONENTS
            if bool(self.dpg.get_value(f"line_component_{name}"))
        }

    def apply_line_preset(self, _sender=None, app_data=None):
        preset = str(app_data or self.dpg.get_value("line_preset"))
        active = set(PRESETS[preset])
        for component in COMPONENTS:
            self.dpg.set_value(
                f"line_component_{component}", component in active)
        self.compose_1d()

    def _push_lines(self):
        if self.line_x is None or self.line_truth is None:
            return
        empty = np.full_like(self.line_truth, np.nan)
        for tag, value in (
            ("line_truth", self.line_truth),
            ("line_observed", self.line_observed if self.line_observed is not None else empty),
            ("line_transport", self.line_output if self.line_output is not None else empty),
        ):
            self.dpg.set_value(tag, [self.line_x.tolist(), value.tolist()])
        self.dpg.fit_axis_data("line_x")
        self.dpg.fit_axis_data("line_y")

    def compose_1d(self):
        try:
            self.line_x, self.line_truth, fields = compose_series(
                int(self.dpg.get_value("line_samples")),
                self.active_components(),
                self.line_parameters(),
            )
            self.line_observed = self.line_truth.copy()
            self.line_output = None
            self._push_lines()
            components = ", ".join(fields) if fields else "flat negative control"
            self.dpg.set_value("line_status", f"Composed: {components}")
            self.dpg.set_value("line_diagnostics", "")
        except Exception:
            self.dpg.set_value("line_diagnostics", traceback.format_exc())

    def corrupt_1d(self):
        try:
            if self.line_truth is None:
                self.compose_1d()
            assert self.line_truth is not None
            self.line_observed = corrupt(
                self.line_truth,
                self.dpg.get_value("line_corruption"),
                amount=float(self.dpg.get_value("line_noise_amount")),
                density=float(self.dpg.get_value("line_noise_density")),
                seed=int(self.dpg.get_value("line_seed")),
            )
            self.line_output = None
            self._push_lines()
            self.dpg.set_value(
                "line_status", f"Corrupted with {self.dpg.get_value('line_corruption')}")
        except Exception:
            self.dpg.set_value("line_diagnostics", traceback.format_exc())

    def run_1d(self):
        try:
            if self.line_observed is None:
                self.corrupt_1d()
            assert self.line_observed is not None
            method = self.dpg.get_value("line_method")
            self.dpg.set_value("line_status", f"Running {method}…")
            if method == "full-scale cross-predictive equilibrium":
                self.line_output, diagnostics = denoise_cross_predictive_transport(
                    self.line_observed)
            elif method == "action-contracting connection (research)":
                forms, diagnostics = (
                    action_contracting_connection_readout_forms(
                        self.line_observed,
                        fuse_population_phase_odds=True,
                        phase_coherent_connection_posterior=True,
                    )
                )
                self.line_output = forms["collision_mean"]
                diagnostics["readout"] = "local joint collision mean"
            elif method == "PFABADA-Cesaro oracle risk":
                if self.line_truth is None:
                    raise ValueError(
                        "oracle PFABADA requires the composed clean reference "
                        "to supply generating-noise moments")
                forms, diagnostics = (
                    denoise_oracle_fabada_from_corruption_1d(
                        self.line_observed,
                        self.line_truth,
                        self.dpg.get_value("line_corruption"),
                        amount=float(
                            self.dpg.get_value("line_noise_amount")),
                        density=float(
                            self.dpg.get_value("line_noise_density")),
                    )
                )
                self.line_output = forms["global"]
                diagnostics["readout"] = (
                    "global known-covariance affine-risk aggregate")
                diagnostics["point_adaptive_control_mse"] = float(np.mean(
                    (forms["local"] - self.line_truth) ** 2))
            elif method == "legacy Gaussian+support flow":
                self.line_output, diagnostics = denoise_1d(
                    self.line_observed,
                    self.resolution(),
                    provisional_sigma=float(
                        self.dpg.get_value("line_provisional_sigma")),
                    action_budget_multiplier=float(
                        self.dpg.get_value("line_action_multiplier")),
                    continuation_rounds=int(
                        self.dpg.get_value("line_continuation_rounds")),
                )
            else:
                raise ValueError(f"unknown 1-D method: {method}")
            if self.line_truth is not None:
                diagnostics["observed_mse"] = float(np.mean(
                    (self.line_observed - self.line_truth) ** 2))
                diagnostics["denoised_mse"] = float(np.mean(
                    (self.line_output - self.line_truth) ** 2))
            diagnostics["active_components"] = sorted(self.active_components())
            diagnostics["corruption"] = self.dpg.get_value("line_corruption")
            diagnostics["method"] = method
            self._push_lines()
            self.dpg.set_value("line_diagnostics", json.dumps(diagnostics, indent=2))
            self.dpg.set_value("line_status", f"Finished {method}")
        except Exception:
            self.dpg.set_value("line_status", "1-D run failed; see diagnostics")
            self.dpg.set_value("line_diagnostics", traceback.format_exc())

    def run_1d_pipeline(self):
        self.compose_1d()
        self.corrupt_1d()
        self.run_1d()

    # ------------------------------------------------------------------ 2-D

    def choose_image(self, _sender=None, app_data=None):
        if app_data and app_data.get("file_path_name"):
            self.load_image(Path(app_data["file_path_name"]))

    def _accept_clean_image(self, image: np.ndarray, name: str):
        self.clean_image = _fit_gray(
            image, int(self.dpg.get_value("image_size")))
        self.image = self.clean_image.copy()
        self.image_output = None
        self.source_name = name
        self.dpg.set_value(
            "image_status",
            f"Loaded clean source {name}: "
            f"{self.clean_image.shape[1]} x {self.clean_image.shape[0]}",
        )
        self.render_images({"clean source": self.clean_image, "observation": self.image})

    def load_image(self, path: Path):
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        self.source_path = path
        self.dpg.set_value("image_path", str(path))
        self._accept_clean_image(image, path.name)

    def load_text_path(self):
        value = self.dpg.get_value("image_path").strip()
        if value:
            self.load_image(Path(value).expanduser())

    def load_skimage(self):
        try:
            from skimage import data
            label = self.dpg.get_value("skimage_source")
            key = SKIMAGE_SOURCES[label]
            self.source_path = None
            self._accept_clean_image(getattr(data, key)(), f"skimage.data.{key}")
        except Exception:
            self.dpg.set_value(
                "image_status",
                "Could not load skimage source; install the repository's "
                "vision-viewer extra. See diagnostics.",
            )
            self.dpg.set_value("image_diagnostics", traceback.format_exc())

    def synthetic_image(self):
        truth, _observed = hair_edge_scene(
            int(self.dpg.get_value("image_size")),
            int(self.dpg.get_value("image_noise_seed")),
        )
        self.source_path = None
        self._accept_clean_image(truth, "tapered hair-edge control")

    def corrupt_2d(self):
        try:
            if self.clean_image is None:
                self.load_skimage()
            if self.clean_image is None:
                return
            self.image = corrupt(
                self.clean_image,
                self.dpg.get_value("image_corruption"),
                amount=float(self.dpg.get_value("image_noise_amount")),
                density=float(self.dpg.get_value("image_noise_density")),
                seed=int(self.dpg.get_value("image_noise_seed")),
            )
            self.image_output = None
            self.render_images({"clean source": self.clean_image, "corrupted": self.image})
            self.dpg.set_value(
                "image_status",
                f"Corrupted {self.source_name} with "
                f"{self.dpg.get_value('image_corruption')}",
            )
        except Exception:
            self.dpg.set_value("image_status", "Corruption failed; see diagnostics")
            self.dpg.set_value("image_diagnostics", traceback.format_exc())

    def render_images(self, images: dict[str, np.ndarray]):
        dpg = self.dpg
        for tag in self.texture_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.texture_tags.clear()
        dpg.delete_item("image_row", children_only=True)
        with dpg.texture_registry(show=False):
            for index, image in enumerate(images.values()):
                width, height, data = _texture_data(image)
                tag = f"denoiser_texture_{index}"
                dpg.add_static_texture(width, height, data, tag=tag)
                self.texture_tags.append(tag)
        for index, (name, image) in enumerate(images.items()):
            height, width = image.shape
            with dpg.group(parent="image_row"):
                dpg.add_text(name.replace("_", " ").title())
                scale = min(300.0 / width, 300.0 / height, 1.5)
                dpg.add_image(
                    f"denoiser_texture_{index}",
                    width=int(width * scale),
                    height=int(height * scale),
                )

    def run_2d(self):
        if self.image is None:
            self.synthetic_image()
        assert self.image is not None
        dpg = self.dpg
        mode = dpg.get_value("image_method")
        dpg.set_value("image_status", f"Running {mode}…")
        try:
            if mode == "support-birth diagnostic":
                support, support_diag = support_density(
                    self.image, self.resolution())
                provisional = ndimage.gaussian_filter(self.image, 1.0, mode="reflect")
                output, barrier, diagnostics = transport_support_birth(
                    self.image,
                    provisional,
                    self.resolution(),
                    support_field=support,
                    support_diagnostics=support_diag,
                )
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "transported support state": output,
                    "support density": support,
                    "barrier admission": barrier,
                }
            elif mode == "continuous-support FMMT":
                support, support_diag = support_density(
                    self.image, self.resolution())
                output, diagnostics = denoise_2d_fmmt(
                    self.image,
                    resolution=self.resolution(),
                    precomputed_support=(support, support_diag),
                )
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "continuous FMMT": output,
                    "support density": support,
                }
            elif mode == "integrated FMMT checkpoint":
                output, diagnostics = denoise_fmmt(self.image)
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "integrated FMMT": output,
                }
            elif mode == "plain FMMT":
                output, diagnostics = denoise_fmmt(
                    self.image, certify_support=False)
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "plain FMMT": output,
                }
            else:
                raise ValueError(f"unknown method: {mode}")
            self.image_output = output
            if self.clean_image is not None:
                diagnostics["corrupted_mse"] = float(np.mean(
                    (self.image - self.clean_image) ** 2))
                diagnostics["denoised_mse"] = float(np.mean(
                    (output - self.clean_image) ** 2))
            diagnostics["source"] = self.source_name
            diagnostics["corruption"] = dpg.get_value("image_corruption")
            self.render_images({key: value for key, value in views.items() if value is not None})
            dpg.set_value("image_diagnostics", json.dumps(diagnostics, indent=2))
            dpg.set_value("image_status", f"Finished {mode}")
        except Exception:
            dpg.set_value("image_status", "Denoising failed; see diagnostics")
            dpg.set_value("image_diagnostics", traceback.format_exc())

    def run_2d_pipeline(self):
        self.corrupt_2d()
        self.run_2d()


def _add_float(dpg, label: str, tag: str, default: float, low: float, high: float):
    dpg.add_slider_float(
        label=label, tag=tag, default_value=default,
        min_value=low, max_value=high, width=260)


def _build_line_controls(dpg, app: DenoiserLab):
    with dpg.group(horizontal=True):
        dpg.add_combo(
            tuple(PRESETS), tag="line_preset",
            default_value="mixed transport stress", label="composite preset",
            width=300, callback=app.apply_line_preset)
        dpg.add_input_int(
            label="samples", tag="line_samples", default_value=128,
            min_value=64, min_clamped=True, width=180)
    with dpg.collapsing_header(label="1. Compose clean series", default_open=True):
        with dpg.group(horizontal=True):
            for component in COMPONENTS:
                dpg.add_checkbox(
                    label=component,
                    tag=f"line_component_{component}",
                    default_value=component in PRESETS["mixed transport stress"],
                )
        with dpg.group(horizontal=True):
            _add_float(dpg, "baseline", "line_baseline", DEFAULT_PARAMETERS["baseline"], 0.0, 0.8)
            _add_float(dpg, "trend amplitude", "line_trend_amplitude", DEFAULT_PARAMETERS["trend_amplitude"], -0.5, 0.5)
            _add_float(dpg, "bump amplitude", "line_bump_amplitude", DEFAULT_PARAMETERS["bump_amplitude"], -0.5, 0.5)
        with dpg.group(horizontal=True):
            _add_float(dpg, "bump center", "line_bump_center", DEFAULT_PARAMETERS["bump_center"], 0.0, 1.0)
            _add_float(dpg, "bump width", "line_bump_width", DEFAULT_PARAMETERS["bump_width"], 0.005, 0.35)
            _add_float(dpg, "step amplitude", "line_step_amplitude", DEFAULT_PARAMETERS["step_amplitude"], -0.5, 0.5)
        with dpg.group(horizontal=True):
            _add_float(dpg, "step center", "line_step_center", DEFAULT_PARAMETERS["step_center"], 0.0, 1.0)
            _add_float(dpg, "step width", "line_step_width", DEFAULT_PARAMETERS["step_width"], 0.001, 0.1)
            _add_float(dpg, "tone amplitude", "line_tone_amplitude", DEFAULT_PARAMETERS["tone_amplitude"], 0.0, 0.3)
        with dpg.group(horizontal=True):
            _add_float(dpg, "tone cycles", "line_tone_cycles", DEFAULT_PARAMETERS["tone_cycles"], 1.0, 80.0)
            _add_float(dpg, "chirp amplitude", "line_chirp_amplitude", DEFAULT_PARAMETERS["chirp_amplitude"], 0.0, 0.3)
            _add_float(dpg, "chirp start", "line_chirp_start", DEFAULT_PARAMETERS["chirp_start"], 0.0, 1.0)
        with dpg.group(horizontal=True):
            _add_float(dpg, "chirp cycles", "line_chirp_cycles", DEFAULT_PARAMETERS["chirp_cycles"], 1.0, 80.0)
            _add_float(dpg, "chirp sweep", "line_chirp_sweep", DEFAULT_PARAMETERS["chirp_sweep"], -60.0, 60.0)
            _add_float(dpg, "ripple amplitude", "line_ripple_amplitude", DEFAULT_PARAMETERS["ripple_amplitude"], 0.0, 0.3)
        with dpg.group(horizontal=True):
            _add_float(dpg, "ripple cycles", "line_ripple_cycles", DEFAULT_PARAMETERS["ripple_cycles"], 1.0, 100.0)
            _add_float(dpg, "ripple start", "line_ripple_start", DEFAULT_PARAMETERS["ripple_start"], 0.0, 1.0)
            _add_float(dpg, "ripple decay", "line_ripple_decay", DEFAULT_PARAMETERS["ripple_decay"], 0.0, 30.0)
        with dpg.group(horizontal=True):
            _add_float(dpg, "pulse amplitude", "line_pulse_amplitude", DEFAULT_PARAMETERS["pulse_amplitude"], 0.0, 0.5)
            _add_float(dpg, "pulse width", "line_pulse_width", DEFAULT_PARAMETERS["pulse_width"], 0.003, 0.15)
            dpg.add_button(label="Compose clean series", callback=app.compose_1d)
    with dpg.collapsing_header(label="2. Corrupt observation", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                CORRUPTIONS, tag="line_corruption",
                default_value="uniform additive", label="corruption", width=280)
            _add_float(dpg, "amount", "line_noise_amount", 0.24, 0.0, 0.6)
            _add_float(dpg, "replacement density", "line_noise_density", 0.08, 0.0, 1.0)
            dpg.add_input_int(label="seed", tag="line_seed", default_value=4701, width=150)
            dpg.add_button(label="Corrupt clean series", callback=app.corrupt_1d)
    with dpg.collapsing_header(label="3. Denoise", default_open=True):
        dpg.add_combo(
            (
                "full-scale cross-predictive equilibrium",
                "action-contracting connection (research)",
                "PFABADA-Cesaro oracle risk",
                "legacy Gaussian+support flow",
            ),
            tag="line_method",
            default_value="full-scale cross-predictive equilibrium",
            label="1-D method",
            width=380,
        )
        with dpg.group(horizontal=True):
            _add_float(
                dpg, "provisional smoothing scale", "line_provisional_sigma",
                2.0, 0.0, 8.0)
            _add_float(
                dpg, "transport action budget x", "line_action_multiplier",
                8.0, 0.0, 32.0)
            dpg.add_slider_int(
                label="continuation rounds",
                tag="line_continuation_rounds",
                default_value=4,
                min_value=1,
                max_value=16,
                width=260,
            )
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Denoise current 1-D observation",
                callback=app.run_1d)
            dpg.add_button(
                label="Compose -> corrupt -> denoise",
                callback=app.run_1d_pipeline)
            dpg.add_text("Ready", tag="line_status")
        dpg.add_text(
            "The full-scale and action-contracting candidates use every "
            "topological lag and ignore the three legacy controls above. "
            "The research connection form transports Gaussian connection "
            "laws through exact ancestry. Spherical phase collision evolves "
            "the uncertainty between analytic Newton and continuous action-"
            "posterior connections before harmonic contraction, without a "
            "run duration. PFABADA-Cesaro is an explicitly unfair comparison: "
            "it receives the selected corruption law and its exact generating "
            "moments, replaces PFABADA's invalid chi-square machinery with "
            "affine-risk aggregation, and ignores the legacy controls. "
            "The validated research size is 128; "
            "the current exact oracle scales steeply above it. For the "
            "legacy method, "
            "smoothing scale sets its "
            "provisional chart, action budget sets travel, and rounds restart "
            "that flow. The global maximum-step setting is a safety guard.")


def _build_image_controls(dpg, app: DenoiserLab):
    with dpg.collapsing_header(label="1. Choose clean source", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                tuple(SKIMAGE_SOURCES), tag="skimage_source",
                default_value="camera — Cameraman", label="skimage.data",
                width=340)
            dpg.add_input_int(
                label="longest side", tag="image_size",
                default_value=128, min_value=64, max_value=256, width=190)
            dpg.add_button(label="Load skimage source", callback=app.load_skimage)
            dpg.add_button(label="Hair-edge control", callback=app.synthetic_image)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="image_path", width=520, hint="PNG/JPEG path")
            dpg.add_button(label="Open path", callback=app.load_text_path)
            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("image_dialog"))
    with dpg.collapsing_header(label="2. Corrupt observation", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                CORRUPTIONS, tag="image_corruption",
                default_value="uniform additive", label="corruption", width=280)
            _add_float(dpg, "amount", "image_noise_amount", 0.24, 0.0, 0.6)
            _add_float(dpg, "replacement density", "image_noise_density", 0.08, 0.0, 1.0)
            dpg.add_input_int(
                label="seed", tag="image_noise_seed", default_value=719, width=150)
            dpg.add_button(label="Corrupt clean image", callback=app.corrupt_2d)
        dpg.add_text(
            "Amount controls additive/multiplicative scale; density controls "
            "salt, pepper, and replacement mass.")
    with dpg.collapsing_header(label="3. Denoise", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                (
                    "continuous-support FMMT",
                    "integrated FMMT checkpoint",
                    "plain FMMT",
                    "support-birth diagnostic",
                ),
                tag="image_method",
                default_value="continuous-support FMMT",
                label="archived denoiser form",
                width=350,
            )
            dpg.add_button(label="Denoise current corrupted image", callback=app.run_2d)
            dpg.add_button(label="Corrupt -> denoise", callback=app.run_2d_pipeline)
        dpg.add_text("Ready", tag="image_status")


def main() -> None:
    try:
        import dearpygui.dearpygui as dpg
    except ImportError as exc:
        raise SystemExit(
            "Dear PyGui is required. Install the repository's vision-viewer "
            "extra before launching this interface."
        ) from exc

    dpg.create_context()
    app = DenoiserLab(dpg)
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=app.choose_image,
        tag="image_dialog",
        width=760,
        height=480,
    ):
        dpg.add_file_extension("Images (*.png *.jpg *.jpeg){.png,.jpg,.jpeg}")
        dpg.add_file_extension(".*")

    with dpg.window(tag="main", label="Rejected FMMT Denoiser Archive"):
        dpg.add_text(
            "FMMT is rejected and retained only for reproducible falsification. "
            "The post-FMMT typical-orbit set estimator is not yet promoted here.",
            color=(208, 96, 72),
        )
        with dpg.collapsing_header(label="Numerical equation resolution", default_open=False):
            dpg.add_slider_int(
                label="Scale quadrature samples", tag="scale_samples",
                min_value=3, max_value=13, default_value=7)
            dpg.add_slider_int(
                label="Empirical histogram bins", tag="histogram_bins",
                min_value=16, max_value=128, default_value=64)
            dpg.add_input_int(
                label="Maximum flux steps (guard)", tag="maximum_steps",
                default_value=4096, min_value=64, min_clamped=True)
            dpg.add_text("These controls resolve equations; they are not quality presets.")
        with dpg.tab_bar(tag="laboratory_tabs"):
            with dpg.tab(label="Archived 1-D experiments", tag="line_tab"):
                _build_line_controls(dpg, app)
                with dpg.plot(label="Composited 1-D denoising", height=390, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="domain", tag="line_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="state", tag="line_y"):
                        dpg.add_line_series([], [], label="clean composite", tag="line_truth")
                        dpg.add_line_series([], [], label="corrupted", tag="line_observed")
                        dpg.add_line_series([], [], label="denoised", tag="line_transport")
                dpg.add_input_text(
                    tag="line_diagnostics", multiline=True,
                    readonly=True, height=200, width=-1)
            with dpg.tab(label="Rejected 2-D FMMT controls", tag="image_tab"):
                _build_image_controls(dpg, app)
                with dpg.group(horizontal=True, tag="image_row"):
                    pass
                dpg.add_input_text(
                    tag="image_diagnostics", multiline=True,
                    readonly=True, height=250, width=-1)

    dpg.create_viewport(
        title="Rejected FMMT Denoiser Archive", width=1540, height=1040)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)
    dpg.set_value("laboratory_tabs", "image_tab")
    app.compose_1d()
    app.corrupt_1d()
    try:
        import skimage  # noqa: F401
    except ImportError:
        app.synthetic_image()
    else:
        app.load_skimage()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
