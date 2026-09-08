"""On-demand Dear PyGui comparison of 2x interpolation on ``skimage.data``.

The GUI calls the experimental operators directly.  It does not load
precomputed reconstructions.  Two acquisition protocols match the catalog
benchmark: exact even/even native samples, and a common positive-antialiased
2x reduction followed by reconstruction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.easu_reference import easu_nested_2x
from experiments.run_skimage_interpolation_benchmark import (
    SKIMAGE_DATA_CATALOG,
    _array_rasters,
    _global_overshoot,
    _held_out_mask,
    _mse,
    _patch_side,
    _unit_float_image,
)
from experiments.self_geometric_harmonic_interpolation import (
    bilinear_resample_to_shape,
    cosine_hermite_characteristic_resize,
    harmonic_resize,
    lanczos_resize,
    lanczos_resample_to_shape,
    positive_restrict,
    riemannian_lifting_restrict,
)


Array = np.ndarray


PROTOCOL_LABELS = {
    "Exact native sublattice": "exact_sublattice",
    "Positive 2x down, then up": "positive_antialiased",
    "Riemannian lifting down, then up": "riemannian_lifting",
}


def compute_comparison(
    image: Array,
    *,
    protocol: str,
    maximum_truth_side: int = 65,
    origin_fraction_yx: tuple[float, float] = (0.5, 0.5),
) -> dict[str, object]:
    """Compute one native-pixel comparison without any GUI state."""

    raster, normalization = _unit_float_image(image)
    side = _patch_side(raster.shape[:2], maximum_truth_side, 2)
    limit_y = raster.shape[0] - side
    limit_x = raster.shape[1] - side
    origin_y = int(round(np.clip(origin_fraction_yx[0], 0.0, 1.0) * limit_y))
    origin_x = int(round(np.clip(origin_fraction_yx[1], 0.0, 1.0) * limit_x))
    truth = raster[origin_y : origin_y + side, origin_x : origin_x + side]
    if protocol == "exact_sublattice":
        source = truth[::2, ::2]
    elif protocol == "positive_antialiased":
        source = positive_restrict(truth, 2)
    elif protocol == "riemannian_lifting":
        source = riemannian_lifting_restrict(truth, 2)
    else:
        raise ValueError(f"unknown acquisition protocol {protocol!r}")
    reconstructions = {
        "Bilinear": harmonic_resize(source, 2, adaptive=False),
        "Lanczos-3 sinc": lanczos_resize(source, 2, radius=3),
        "AMD FSR 1.0 EASU": easu_nested_2x(source),
        "Cosine-Hermite Riemannian transport": (
            cosine_hermite_characteristic_resize(source, 2)
        ),
    }
    held_out = _held_out_mask(truth.shape[:2], 2)
    metrics = {
        name: {
            "held_out_mse": _mse(value, truth, held_out),
            "all_pixel_mse": _mse(value, truth),
            "common_positive_return_mse": _mse(
                positive_restrict(value, 2), source
            ),
            "global_source_range_overshoot": _global_overshoot(value, source),
        }
        for name, value in reconstructions.items()
    }
    coarse_shape = source.shape[:2]
    matched_cycles = {
        "Bilinear": bilinear_resample_to_shape(
            bilinear_resample_to_shape(truth, coarse_shape), truth.shape[:2]
        ),
        "Lanczos-3 sinc": lanczos_resample_to_shape(
            lanczos_resample_to_shape(truth, coarse_shape, radius=3),
            truth.shape[:2],
            radius=3,
        ),
        "Cosine-Hermite Riemannian transport": (
            cosine_hermite_characteristic_resize(
                riemannian_lifting_restrict(truth, 2), 2
            )
        ),
    }
    for name, cycle in matched_cycles.items():
        metrics[name]["matched_family_round_trip_mse"] = _mse(cycle, truth)
    metrics["AMD FSR 1.0 EASU"]["matched_family_round_trip_mse"] = None
    return {
        "truth": truth,
        "source": source,
        "reconstructions": reconstructions,
        "metrics": metrics,
        "origin_yx": (origin_y, origin_x),
        "truth_side": side,
        "normalization": normalization,
        "protocol": protocol,
    }


def _rgb(value: Array) -> Array:
    image = np.asarray(value, dtype=np.float64)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return np.clip(image[..., :3], 0.0, 1.0)


def _rgba_flat(value: Array) -> list[float]:
    rgb = _rgb(value).astype(np.float32)
    alpha = np.ones(rgb.shape[:2] + (1,), dtype=np.float32)
    return np.concatenate((rgb, alpha), axis=2).ravel().tolist()


def _residual_rgb(estimate: Array, truth: Array, gain: float) -> Array:
    residual = np.asarray(estimate) - np.asarray(truth)
    if residual.ndim == 3:
        residual = np.mean(residual, axis=2)
    shown = np.zeros(residual.shape + (3,), dtype=np.float64)
    shown[..., 0] = np.clip(gain * residual, 0.0, 1.0)
    shown[..., 2] = np.clip(-gain * residual, 0.0, 1.0)
    return shown


@dataclass
class DemoState:
    rasters: dict[str, Array] = field(default_factory=dict)
    texture_serial: int = 0


def run_gui(initial_provider: str = "astronaut") -> None:
    import dearpygui.dearpygui as dpg
    from skimage import data

    state = DemoState()
    dpg.create_context()

    def set_status(message: str) -> None:
        dpg.set_value("status", message)

    def load_provider(provider: str) -> None:
        set_status(f"Loading skimage.data.{provider}() ...")
        try:
            value = getattr(data, provider)()
            state.rasters = {
                name: array for name, array in _array_rasters(provider, value)
            }
            names = list(state.rasters)
            dpg.configure_item("raster", items=names)
            dpg.set_value("raster", names[0])
            set_status(
                f"Loaded {provider}: {len(names)} raster(s). Press Compute."
            )
        except Exception as error:
            state.rasters = {}
            set_status(f"Load failed: {type(error).__name__}: {error}")

    def provider_changed(_sender: object, app_data: object) -> None:
        load_provider(str(app_data))

    def add_texture(value: Array) -> str:
        state.texture_serial += 1
        tag = f"result_texture_{state.texture_serial}"
        height, width = value.shape[:2]
        dpg.add_static_texture(
            width, height, _rgba_flat(value), tag=tag, parent="textures"
        )
        return tag

    def compute(_sender: object = None, _app_data: object = None) -> None:
        raster_name = dpg.get_value("raster")
        if raster_name not in state.rasters:
            set_status("Select a loaded raster first.")
            return
        set_status("Computing all four operators on demand ...")
        dpg.delete_item("results", children_only=True)
        dpg.delete_item("textures", children_only=True)
        try:
            protocol = PROTOCOL_LABELS[dpg.get_value("protocol")]
            result = compute_comparison(
                state.rasters[raster_name],
                protocol=protocol,
                maximum_truth_side=int(dpg.get_value("truth_side")),
                origin_fraction_yx=(
                    float(dpg.get_value("origin_y")),
                    float(dpg.get_value("origin_x")),
                ),
            )
            gain = float(dpg.get_value("residual_gain"))
            panels = [("Native truth", result["truth"], None)] + [
                (name, value, result["metrics"][name])
                for name, value in result["reconstructions"].items()
            ]
            with dpg.group(horizontal=True, parent="results"):
                for name, value, metrics in panels:
                    with dpg.child_window(width=292, height=650):
                        dpg.add_text(name)
                        texture = add_texture(value)
                        dpg.add_image(texture, width=260, height=260)
                        if metrics is None:
                            dpg.add_text("Original native pixels")
                            residual = np.zeros_like(result["truth"])
                        else:
                            dpg.add_text(
                                f"held-out MSE  {metrics['held_out_mse']:.9g}"
                            )
                            dpg.add_text(
                                f"all-pixel MSE {metrics['all_pixel_mse']:.9g}"
                            )
                            dpg.add_text(
                                "common positive return MSE "
                                f"{metrics['common_positive_return_mse']:.9g}"
                            )
                            matched = metrics["matched_family_round_trip_mse"]
                            dpg.add_text(
                                "matched family cycle MSE "
                                + ("n/a (upscaler only)" if matched is None
                                   else f"{matched:.9g}")
                            )
                            dpg.add_text(
                                "range excess   "
                                f"{metrics['global_source_range_overshoot']:.9g}"
                            )
                            residual = _residual_rgb(
                                value, result["truth"], gain
                            )
                        dpg.add_text(
                            f"Signed residual (red +, blue -, {gain:g}x)"
                        )
                        residual_texture = add_texture(residual)
                        dpg.add_image(
                            residual_texture, width=260, height=260
                        )
            set_status(
                f"Computed {raster_name}, native patch {result['truth_side']}x"
                f"{result['truth_side']} at {result['origin_yx']}; "
                f"normalization: {result['normalization']}."
            )
        except Exception as error:
            set_status(f"Compute failed: {type(error).__name__}: {error}")

    with dpg.texture_registry(tag="textures"):
        pass
    with dpg.window(tag="primary", label="Cosine-Hermite Riemannian Interpolation"):
        dpg.add_text(
            "Native-pixel 2x comparison: bilinear, Lanczos-3 sinc, AMD "
            "FSR 1.0 EASU, and shape-constrained cosine-Hermite transport"
        )
        with dpg.group(horizontal=True):
            dpg.add_combo(
                list(SKIMAGE_DATA_CATALOG),
                default_value=initial_provider,
                label="skimage.data provider",
                tag="provider",
                width=220,
                callback=provider_changed,
            )
            dpg.add_combo([], label="raster/frame", tag="raster", width=250)
            dpg.add_combo(
                list(PROTOCOL_LABELS),
                default_value="Exact native sublattice",
                label="acquisition",
                tag="protocol",
                width=230,
            )
        with dpg.group(horizontal=True):
            dpg.add_slider_float(
                label="crop y", tag="origin_y", min_value=0.0,
                max_value=1.0, default_value=0.5, width=180,
            )
            dpg.add_slider_float(
                label="crop x", tag="origin_x", min_value=0.0,
                max_value=1.0, default_value=0.5, width=180,
            )
            dpg.add_combo(
                [17, 33, 65, 129], default_value=65,
                label="maximum native patch side", tag="truth_side", width=90,
            )
            dpg.add_slider_float(
                label="residual gain", tag="residual_gain", min_value=1.0,
                max_value=100.0, default_value=20.0, width=180,
            )
            dpg.add_button(label="Compute", callback=compute)
        dpg.add_text("", tag="status", wrap=1150)
        dpg.add_separator()
        dpg.add_group(tag="results")

    dpg.create_viewport(
        title="2x Interpolation: Cosine-Hermite vs EASU vs Lanczos vs Bilinear",
        width=1540,
        height=850,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    load_provider(initial_provider)
    dpg.start_dearpygui()
    dpg.destroy_context()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="astronaut")
    args = parser.parse_args()
    run_gui(args.provider)


if __name__ == "__main__":
    main()
