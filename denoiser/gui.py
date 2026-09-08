"""Dear PyGui evaluation lab for the active transport denoiser and controls."""

from __future__ import annotations

import json
from pathlib import Path
import re
from time import perf_counter
import traceback

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy import sparse

try:
    from .cross_predictive_transport import (
        action_contracting_connection_readout_forms,
        denoise_cross_predictive_transport,
    )
    from .dcnt import denoise_dcnt, tv_chambolle_reference
    from .cross_validated_transport_stopping import (
        denoise_cross_validated_transport_chambolle,
    )
    from .transport_chambolle import denoise_transport_chambolle
    from .selling_chambolle import (
        denoise_cross_chart_phase_chambolle,
        denoise_phase_action_chambolle,
        denoise_population_phase_chambolle,
    )
    from .fmmt_certified import denoise_fmmt
    from .fmmt_rebuilt import denoise_rebuilt_fmmt, rebuild_fmmt_posterior
    from .lifted_endpoint_action_transport_2d import (
        denoise_lifted_endpoint_action_transport_2d,
    )
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
    from .run_2d_denoiser_battery import metrics as image_metrics
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
    from dcnt import denoise_dcnt, tv_chambolle_reference
    from cross_validated_transport_stopping import (
        denoise_cross_validated_transport_chambolle,
    )
    from transport_chambolle import denoise_transport_chambolle
    from selling_chambolle import (
        denoise_cross_chart_phase_chambolle,
        denoise_phase_action_chambolle,
        denoise_population_phase_chambolle,
    )
    from fmmt_certified import denoise_fmmt
    from fmmt_rebuilt import denoise_rebuilt_fmmt, rebuild_fmmt_posterior
    from lifted_endpoint_action_transport_2d import (
        denoise_lifted_endpoint_action_transport_2d,
    )
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
    from run_2d_denoiser_battery import metrics as image_metrics
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


IMAGE_EVALUATION_CASES = {
    "clean identity": ("none", 0.0, 0.0),
    "Gaussian additive 0.10": ("Gaussian additive", 0.10, 0.0),
    "uniform additive 0.10": ("uniform additive", 0.10, 0.0),
    "salt and pepper 0.20": ("salt and pepper", 0.0, 0.20),
    "mixed replacement + uniform 0.25": (
        "mixed replacement + uniform", 0.10, 0.25),
}


IMAGE_METHODS = (
    "Best — rebuilt FMMT transport support",
    "Last canon freeze — causal population Chambolle",
    "Canon compare — OEM Chambolle / TV / mean / FMMT",
)


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


def _diagnostic_summary(value):
    """Make nested research diagnostics readable without serializing images."""
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        finite = array[np.isfinite(array)] if array.size else array
        result = {
            "kind": "array summary",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if finite.size:
            result.update({
                "minimum": float(np.min(finite)),
                "maximum": float(np.max(finite)),
                "mean": float(np.mean(finite)),
            })
        return result
    if sparse.issparse(value):
        return {
            "kind": "sparse matrix summary",
            "shape": list(value.shape),
            "nonzero_count": int(value.nnz),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _diagnostic_summary(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_summary(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _evaluation_table(
    truth: np.ndarray | None,
    outputs: dict[str, np.ndarray],
    runtimes: dict[str, float] | None = None,
) -> tuple[str, dict[str, dict[str, float]]]:
    if truth is None:
        return "No clean reference is loaded; visual evaluation only.", {}
    scores = {
        name: image_metrics(np.asarray(output), truth)
        for name, output in outputs.items()
    }
    lines = [
        "method                         MSE        SSIM    edge    variance  seconds",
        "----------------------------  ---------  ------  ------  --------  -------",
    ]
    for name, score in scores.items():
        seconds = (runtimes or {}).get(name)
        runtime = "   —   " if seconds is None else f"{seconds:7.3f}"
        lines.append(
            f"{name[:28]:28s}  {score['mse']:9.6f}  "
            f"{score['ssim']:6.4f}  {score['edge_retention']:6.3f}  "
            f"{score['variance_ratio']:8.3f}  {runtime}"
        )
    return "\n".join(lines), scores


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return name or "image"


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
        self.last_evaluation_images: dict[str, np.ndarray] = {}
        self.last_evaluation_report: dict | None = None
        self.last_evaluation_table = ""

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
        self.last_evaluation_images.clear()
        self.last_evaluation_report = None
        self.dpg.set_value(
            "image_status",
            f"Loaded clean source {name}: "
            f"{self.clean_image.shape[1]} x {self.clean_image.shape[0]}",
        )
        self.render_images({"clean source": self.clean_image, "observation": self.image})
        table, _scores = _evaluation_table(
            self.clean_image, {"clean source": self.clean_image})
        self.dpg.set_value("image_scoreboard", table)

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

    def apply_image_case(self, name: str):
        corruption, amount, density = IMAGE_EVALUATION_CASES[name]
        self.dpg.set_value("image_corruption", corruption)
        self.dpg.set_value("image_noise_amount", amount)
        self.dpg.set_value("image_noise_density", density)
        self.corrupt_2d()

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
            self.last_evaluation_images.clear()
            self.last_evaluation_report = None
            self.render_images({"clean source": self.clean_image, "corrupted": self.image})
            table, _scores = _evaluation_table(
                self.clean_image, {"observation": self.image})
            self.dpg.set_value("image_scoreboard", table)
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
        registry = "denoiser_texture_registry"
        if not dpg.does_item_exist(registry):
            with dpg.texture_registry(show=False, tag=registry):
                pass
        for index, image in enumerate(images.values()):
            width, height, data = _texture_data(image)
            tag = f"denoiser_texture_{index}"
            dpg.add_static_texture(
                width, height, data, tag=tag, parent=registry)
            self.texture_tags.append(tag)
        maximum_side = min(300.0, 1420.0 / max(len(images), 1))
        for index, (name, image) in enumerate(images.items()):
            height, width = image.shape
            with dpg.group(parent="image_row"):
                dpg.add_text(name.replace("_", " ").title())
                scale = min(
                    maximum_side / width, maximum_side / height, 1.5)
                dpg.add_image(
                    f"denoiser_texture_{index}",
                    width=int(width * scale),
                    height=int(height * scale),
                )

    def save_evaluation_bundle(self):
        try:
            if not self.last_evaluation_images or self.last_evaluation_report is None:
                raise ValueError("run a 2-D evaluation before exporting")
            output = Path(
                self.dpg.get_value("image_evaluation_out")).expanduser()
            output.mkdir(parents=True, exist_ok=True)
            for name, image in self.last_evaluation_images.items():
                pixels = np.uint8(np.round(
                    np.clip(np.asarray(image), 0.0, 1.0) * 255.0))
                Image.fromarray(pixels).save(
                    output / f"{_safe_filename(name)}.png")
            (output / "evaluation.json").write_text(
                json.dumps(self.last_evaluation_report, indent=2) + "\n")
            (output / "scoreboard.txt").write_text(
                self.last_evaluation_table + "\n")
            self.dpg.set_value(
                "image_status", f"Saved evaluation bundle to {output}")
        except Exception:
            self.dpg.set_value(
                "image_status", "Evaluation export failed; see diagnostics")
            self.dpg.set_value("image_diagnostics", traceback.format_exc())

    def run_2d(self):
        if self.image is None:
            self.synthetic_image()
        assert self.image is not None
        dpg = self.dpg
        mode = dpg.get_value("image_method")
        dpg.set_value("image_status", f"Running {mode}…")
        try:
            runtimes: dict[str, float] = {}
            evaluation_outputs: dict[str, np.ndarray] = {
                "observation": self.image,
            }
            if mode == "support-birth diagnostic archive":
                started = perf_counter()
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
                runtimes["support-birth archive"] = perf_counter() - started
                evaluation_outputs["support-birth archive"] = output
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "transported support state": output,
                    "support density": support,
                    "barrier admission": barrier,
                }
            elif mode == "continuous-support FMMT archive":
                started = perf_counter()
                support, support_diag = support_density(
                    self.image, self.resolution())
                output, diagnostics = denoise_2d_fmmt(
                    self.image,
                    resolution=self.resolution(),
                    precomputed_support=(support, support_diag),
                )
                runtimes["continuous FMMT archive"] = perf_counter() - started
                evaluation_outputs["continuous FMMT archive"] = output
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "continuous FMMT": output,
                    "support density": support,
                }
            elif mode == "integrated FMMT checkpoint":
                started = perf_counter()
                output, diagnostics = denoise_fmmt(self.image)
                runtimes["integrated FMMT"] = perf_counter() - started
                evaluation_outputs["integrated FMMT"] = output
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "integrated FMMT": output,
                }
            elif mode == "plain FMMT":
                started = perf_counter()
                output, diagnostics = denoise_fmmt(
                    self.image, certify_support=False)
                runtimes["plain FMMT"] = perf_counter() - started
                evaluation_outputs["plain FMMT"] = output
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "plain FMMT": output,
                }
            elif mode in (
                    "Best — rebuilt FMMT transport support",
                    "Canon compare — OEM Chambolle / TV / mean / FMMT"):
                started = perf_counter()
                if mode == (
                        "Canon compare — OEM Chambolle / TV / mean / FMMT"):
                    fmmt_output, fmmt_diagnostic = denoise_fmmt(self.image)
                    fmmt_runtime = perf_counter() - started
                    started = perf_counter()
                    output, best_diagnostic = rebuild_fmmt_posterior(
                        self.image, fmmt_output, fmmt_diagnostic)
                else:
                    output, best_diagnostic = denoise_rebuilt_fmmt(self.image)
                runtimes["BEST — rebuilt FMMT transport support"] = (
                    perf_counter() - started)
                evaluation_outputs[
                    "BEST — rebuilt FMMT transport support"] = output
                diagnostics = {
                    "best_rebuilt_FMMT": best_diagnostic,
                }
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "BEST — rebuilt FMMT transport support": output,
                }
                if mode == (
                        "Canon compare — OEM Chambolle / TV / mean / FMMT"):
                    from skimage.restoration import denoise_tv_chambolle

                    started = perf_counter()
                    oem_output = denoise_tv_chambolle(
                        self.image, weight=0.10, channel_axis=None)
                    runtimes["CANON — OEM Chambolle 0.10"] = (
                        perf_counter() - started)
                    evaluation_outputs[
                        "CANON — OEM Chambolle 0.10"] = oem_output
                    started = perf_counter()
                    tv_output = tv_chambolle_reference(
                        self.image, weight=0.10)
                    runtimes["CANON — reconstructed TV 0.10"] = (
                        perf_counter() - started)
                    evaluation_outputs[
                        "CANON — reconstructed TV 0.10"] = tv_output
                    started = perf_counter()
                    mean_output = ndimage.uniform_filter(
                        self.image, size=3, mode="reflect")
                    runtimes["CANON — mean 3x3"] = perf_counter() - started
                    evaluation_outputs["CANON — mean 3x3"] = mean_output
                    runtimes["CANON — integrated FMMT"] = fmmt_runtime
                    evaluation_outputs[
                        "CANON — integrated FMMT"] = fmmt_output
                    diagnostics.update({
                        "OEM_Chambolle": {
                            "implementation": "skimage.restoration",
                            "weight": 0.10,
                        },
                        "reconstructed_TV": {
                            "implementation": "literal scikit recurrence",
                            "weight": 0.10,
                        },
                        "mean": {"window": "3x3 reflected"},
                        "integrated_FMMT": fmmt_diagnostic,
                    })
                    views.update({
                        "CANON — OEM Chambolle 0.10": oem_output,
                        "CANON — reconstructed TV 0.10": tv_output,
                        "CANON — mean 3x3": mean_output,
                        "CANON — integrated FMMT": fmmt_output,
                    })
            elif mode == "Last canon freeze — causal population Chambolle":
                started = perf_counter()
                output, diagnostics = (
                    denoise_cross_validated_transport_chambolle(self.image))
                runtimes[
                    "LAST CANON FREEZE — causal population Chambolle"] = (
                    perf_counter() - started)
                evaluation_outputs[
                    "LAST CANON FREEZE — causal population Chambolle"] = output
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "LAST CANON FREEZE — causal population Chambolle": output,
                }
            elif mode in (
                    "Phase-action Chambolle suite — transport ablations",
                    "Phase-action Chambolle",
                    "Cross-chart phase Chambolle",
                    "Population-phase Chambolle suite — transport ablations",
                    "Population-phase Chambolle"):
                phase_action = mode.startswith("Phase-action")
                cross_chart = mode == "Cross-chart phase Chambolle"
                started = perf_counter()
                if cross_chart:
                    output, primary_diagnostic = (
                        denoise_cross_chart_phase_chambolle(self.image))
                    primary_name = "cross-chart phase Chambolle"
                    diagnostic_name = "cross_chart_phase_Chambolle"
                elif phase_action:
                    output, primary_diagnostic = (
                        denoise_phase_action_chambolle(self.image))
                    primary_name = "phase-action Chambolle"
                    diagnostic_name = "phase_action_Chambolle"
                else:
                    output, primary_diagnostic = (
                        denoise_population_phase_chambolle(self.image))
                    primary_name = "population-phase Chambolle"
                    diagnostic_name = "population_phase_Chambolle"
                runtimes[primary_name] = perf_counter() - started
                evaluation_outputs[primary_name] = output
                diagnostics = {diagnostic_name: primary_diagnostic}
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    primary_name: output,
                }
                if mode.endswith("suite — transport ablations"):
                    if phase_action:
                        started = perf_counter()
                        cross_output, cross_diagnostic = (
                            denoise_cross_chart_phase_chambolle(self.image))
                        runtimes["cross-chart phase Chambolle"] = (
                            perf_counter() - started)
                        evaluation_outputs[
                            "cross-chart phase Chambolle"] = cross_output
                        diagnostics[
                            "cross_chart_phase_Chambolle"] = cross_diagnostic
                        views[
                            "cross-chart phase Chambolle"] = cross_output
                        started = perf_counter()
                        population_output, population_diagnostic = (
                            denoise_population_phase_chambolle(self.image))
                        runtimes["population-phase Chambolle"] = (
                            perf_counter() - started)
                        evaluation_outputs[
                            "population-phase Chambolle"] = population_output
                        diagnostics[
                            "population_phase_Chambolle"] = population_diagnostic
                        views[
                            "population-phase Chambolle"] = population_output
                    started = perf_counter()
                    cartesian_output, cartesian_diagnostic = (
                        denoise_transport_chambolle(self.image))
                    runtimes["Cartesian transport Chambolle"] = (
                        perf_counter() - started)
                    evaluation_outputs[
                        "Cartesian transport Chambolle"] = cartesian_output
                    started = perf_counter()
                    tv_output = tv_chambolle_reference(
                        self.image, weight=0.10)
                    runtimes["classical Chambolle 0.10"] = (
                        perf_counter() - started)
                    evaluation_outputs["classical Chambolle 0.10"] = tv_output
                    started = perf_counter()
                    dcnt_output, dcnt_diagnostic = denoise_dcnt(
                        self.image, mode="uncertainty")
                    runtimes["DCNT uncertainty"] = perf_counter() - started
                    evaluation_outputs["DCNT uncertainty"] = dcnt_output
                    started = perf_counter()
                    fmmt_output, fmmt_diagnostic = denoise_fmmt(self.image)
                    runtimes["integrated FMMT"] = perf_counter() - started
                    evaluation_outputs["integrated FMMT"] = fmmt_output
                    diagnostics.update({
                        "Cartesian_transport_Chambolle": cartesian_diagnostic,
                        "classical_Chambolle": {
                            "weight": 0.10,
                            "role": "literal scikit recurrence control",
                        },
                        "DCNT_uncertainty": dcnt_diagnostic,
                        "integrated_FMMT": fmmt_diagnostic,
                    })
                    views.update({
                        "Cartesian transport Chambolle": cartesian_output,
                        "classical Chambolle": tv_output,
                        "DCNT uncertainty": dcnt_output,
                        "integrated FMMT": fmmt_output,
                    })
            elif mode in (
                    "Transport Chambolle suite — rebuilt vs classical",
                    "Transport-admissible Chambolle"):
                started = perf_counter()
                output, transport_diagnostic = denoise_transport_chambolle(
                    self.image)
                runtimes["transport Chambolle"] = perf_counter() - started
                evaluation_outputs["transport Chambolle"] = output
                diagnostics = {
                    "transport_Chambolle": transport_diagnostic,
                }
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "transport Chambolle": output,
                }
                if mode == "Transport Chambolle suite — rebuilt vs classical":
                    started = perf_counter()
                    tv_output = tv_chambolle_reference(
                        self.image, weight=0.10)
                    runtimes["classical Chambolle 0.10"] = (
                        perf_counter() - started)
                    evaluation_outputs["classical Chambolle 0.10"] = tv_output
                    started = perf_counter()
                    dcnt_output, dcnt_diagnostic = denoise_dcnt(
                        self.image, mode="uncertainty")
                    runtimes["DCNT uncertainty"] = perf_counter() - started
                    evaluation_outputs["DCNT uncertainty"] = dcnt_output
                    started = perf_counter()
                    fmmt_output, fmmt_diagnostic = denoise_fmmt(self.image)
                    runtimes["integrated FMMT"] = perf_counter() - started
                    evaluation_outputs["integrated FMMT"] = fmmt_output
                    diagnostics.update({
                        "classical_Chambolle": {
                            "weight": 0.10,
                            "role": "literal scikit recurrence control",
                        },
                        "DCNT_uncertainty": dcnt_diagnostic,
                        "integrated_FMMT": fmmt_diagnostic,
                    })
                    views.update({
                        "classical Chambolle": tv_output,
                        "DCNT uncertainty": dcnt_output,
                        "integrated FMMT": fmmt_output,
                    })
            elif mode in (
                    "DCNT research suite — uncertainty vs descent",
                    "DCNT uncertainty contractor",
                    "DCNT transport-descent control"):
                run_uncertainty = mode != "DCNT transport-descent control"
                if run_uncertainty:
                    started = perf_counter()
                    uncertainty_output, uncertainty_diagnostic = denoise_dcnt(
                        self.image, mode="uncertainty")
                    runtimes["DCNT uncertainty"] = perf_counter() - started
                    evaluation_outputs["DCNT uncertainty"] = uncertainty_output
                run_descent = mode != "DCNT uncertainty contractor"
                if run_descent:
                    started = perf_counter()
                    descent_output, descent_diagnostic = denoise_dcnt(
                        self.image, mode="transport_descent")
                    runtimes["DCNT transport descent"] = perf_counter() - started
                    evaluation_outputs["DCNT transport descent"] = descent_output
                if mode == "DCNT uncertainty contractor":
                    output = uncertainty_output
                    diagnostics = {"DCNT_uncertainty": uncertainty_diagnostic}
                    views = {
                        "clean source": self.clean_image,
                        "corrupted": self.image,
                        "DCNT uncertainty": uncertainty_output,
                    }
                elif mode == "DCNT transport-descent control":
                    output = descent_output
                    diagnostics = {"DCNT_transport_descent": descent_diagnostic}
                    views = {
                        "clean source": self.clean_image,
                        "corrupted": self.image,
                        "DCNT transport descent": descent_output,
                    }
                else:
                    started = perf_counter()
                    tv_output = tv_chambolle_reference(
                        self.image, weight=0.10)
                    runtimes["reconstructed TV"] = perf_counter() - started
                    evaluation_outputs["reconstructed TV"] = tv_output
                    started = perf_counter()
                    fmmt_output, fmmt_diagnostic = denoise_fmmt(self.image)
                    runtimes["integrated FMMT"] = perf_counter() - started
                    evaluation_outputs["integrated FMMT"] = fmmt_output
                    output = uncertainty_output
                    diagnostics = {
                        "DCNT_uncertainty": uncertainty_diagnostic,
                        "DCNT_transport_descent": descent_diagnostic,
                        "integrated_FMMT": fmmt_diagnostic,
                    }
                    views = {
                        "clean source": self.clean_image,
                        "corrupted": self.image,
                        "DCNT uncertainty": uncertainty_output,
                        "DCNT transport descent": descent_output,
                        "reconstructed TV": tv_output,
                        "integrated FMMT": fmmt_output,
                    }
            elif mode in (
                    "stopped causal endpoint transport",
                    "evaluation suite — stopped causal vs FMMT"):
                started = perf_counter()
                output, causal_diagnostic = (
                    denoise_lifted_endpoint_action_transport_2d(
                        self.image, return_temporal_ablation=True))
                runtimes["stopped causal"] = perf_counter() - started
                first_cycle = np.asarray(
                    causal_diagnostic["first_cycle_estimate"])
                unstopped = np.asarray(
                    causal_diagnostic["unstopped_second_cycle_estimate"])
                continuation = np.asarray(
                    causal_diagnostic["temporal_continuation_authority"])
                evaluation_outputs.update({
                    "one-cycle causal": first_cycle,
                    "unstopped causal": unstopped,
                    "stopped causal": output,
                })
                diagnostics = {"stopped_causal": causal_diagnostic}
                views = {
                    "clean source": self.clean_image,
                    "corrupted": self.image,
                    "one-cycle causal": first_cycle,
                    "stopped causal": output,
                    "continuation authority": continuation,
                }
                if mode == "evaluation suite — stopped causal vs FMMT":
                    started = perf_counter()
                    fmmt_output, fmmt_diagnostic = denoise_fmmt(self.image)
                    runtimes["integrated FMMT"] = perf_counter() - started
                    evaluation_outputs["integrated FMMT"] = fmmt_output
                    diagnostics["integrated_FMMT"] = fmmt_diagnostic
                    views = {
                        "clean source": self.clean_image,
                        "corrupted": self.image,
                        "one-cycle causal": first_cycle,
                        "unstopped causal": unstopped,
                        "stopped causal": output,
                        "integrated FMMT": fmmt_output,
                    }
            else:
                raise ValueError(f"unknown method: {mode}")
            self.image_output = output
            table, scores = _evaluation_table(
                self.clean_image, evaluation_outputs, runtimes)
            report = {
                "source": self.source_name,
                "corruption": dpg.get_value("image_corruption"),
                "corruption_amount": float(
                    dpg.get_value("image_noise_amount")),
                "corruption_density": float(
                    dpg.get_value("image_noise_density")),
                "method": mode,
                "runtime_seconds": runtimes,
                "reference_metrics": scores,
                "diagnostics": diagnostics,
            }
            compact_report = _diagnostic_summary(report)
            bundle_images = {
                key: np.asarray(value)
                for key, value in views.items() if value is not None
            }
            if mode in (
                    "stopped causal endpoint transport",
                    "evaluation suite — stopped causal vs FMMT"):
                bundle_images["continuation authority"] = continuation
            self.last_evaluation_images = bundle_images
            self.last_evaluation_report = compact_report
            self.last_evaluation_table = table
            self.render_images({key: value for key, value in views.items() if value is not None})
            dpg.set_value("image_scoreboard", table)
            dpg.set_value(
                "image_diagnostics",
                json.dumps(compact_report, indent=2),
            )
            elapsed = sum(runtimes.values())
            dpg.set_value(
                "image_status", f"Finished {mode} in {elapsed:.3f} seconds")
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
                default_value=96, min_value=64, max_value=256, width=190)
            dpg.add_button(label="Load skimage source", callback=app.load_skimage)
            dpg.add_button(label="Hair-edge control", callback=app.synthetic_image)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="image_path", width=520, hint="PNG/JPEG path")
            dpg.add_button(label="Open path", callback=app.load_text_path)
            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("image_dialog"))
    with dpg.collapsing_header(label="2. Corrupt observation", default_open=True):
        with dpg.group(horizontal=True):
            for case_name in IMAGE_EVALUATION_CASES:
                dpg.add_button(
                    label=case_name,
                    user_data=case_name,
                    callback=lambda _s, _a, case: app.apply_image_case(case),
                )
        with dpg.group(horizontal=True):
            dpg.add_combo(
                CORRUPTIONS, tag="image_corruption",
                default_value="mixed replacement + uniform",
                label="corruption", width=280)
            _add_float(dpg, "amount", "image_noise_amount", 0.10, 0.0, 0.6)
            _add_float(
                dpg, "replacement density", "image_noise_density",
                0.25, 0.0, 1.0)
            dpg.add_input_int(
                label="seed", tag="image_noise_seed", default_value=719, width=150)
            dpg.add_button(label="Corrupt clean image", callback=app.corrupt_2d)
        dpg.add_text(
            "Amount controls additive/multiplicative scale; density controls "
            "salt, pepper, and replacement mass.")
    with dpg.collapsing_header(label="3. Denoise", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                IMAGE_METHODS,
                tag="image_method",
                default_value="Best — rebuilt FMMT transport support",
                label="2-D evaluation method",
                width=430,
            )
            dpg.add_button(label="Denoise current corrupted image", callback=app.run_2d)
            dpg.add_button(label="Corrupt -> denoise", callback=app.run_2d_pipeline)
        dpg.add_text("Ready", tag="image_status")
        dpg.add_text(
            "Best runs only rebuilt FMMT transport support. Last canon freeze "
            "is the previous causal-population method. Canon compare evaluates "
            "the best only against OEM scikit Chambolle, the literal rebuilt "
            "TV recurrence, a reflected 3x3 mean, and integrated FMMT. The "
            "clean source is used only for scoring.")
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="image_evaluation_out",
                default_value="/tmp/denoiser_evaluation",
                label="evaluation bundle",
                width=520,
            )
            dpg.add_button(
                label="Save images + report",
                callback=app.save_evaluation_bundle,
            )


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

    with dpg.window(tag="main", label="Transport Denoiser Evaluation Lab"):
        dpg.add_text(
            "Active experiment: population-phase transport Chambolle. Selling "
            "flow and exact CONV detail meet without a noise-class setting.",
            color=(90, 190, 150),
        )
        with dpg.collapsing_header(
                label="Archived numerical controls", default_open=False):
            dpg.add_slider_int(
                label="Scale quadrature samples", tag="scale_samples",
                min_value=3, max_value=13, default_value=7)
            dpg.add_slider_int(
                label="Empirical histogram bins", tag="histogram_bins",
                min_value=16, max_value=128, default_value=64)
            dpg.add_input_int(
                label="Maximum flux steps (guard)", tag="maximum_steps",
                default_value=4096, min_value=64, min_clamped=True)
            dpg.add_text(
                "Only archived support/FMMT and legacy 1-D forms read these. "
                "DCNT and the stopped causal estimator ignore them.")
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
            with dpg.tab(label="2-D evaluation", tag="image_tab"):
                _build_image_controls(dpg, app)
                dpg.add_input_text(
                    tag="image_scoreboard", multiline=True,
                    readonly=True, height=145, width=-1)
                with dpg.group(horizontal=True, tag="image_row"):
                    pass
                dpg.add_input_text(
                    tag="image_diagnostics", multiline=True,
                    readonly=True, height=250, width=-1)

    dpg.create_viewport(
        title="Transport Denoiser Evaluation Lab", width=1540, height=1100)
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
