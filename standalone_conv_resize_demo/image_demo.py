"""On-demand arbitrary image resize comparison using Dear PyGui."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from backend import (
    backend_description,
    conv_transport_resize,
    lanczos3_resize,
    linear_resize,
    polyphase_fir_resize,
)
from gui_common import mse, range_excess, rgba_preview, unit_image


PROVIDERS = (
    "astronaut", "camera", "text", "logo", "page", "checkerboard",
    "coffee", "coins", "moon", "chelsea", "rocket", "horse",
    "brick", "grass", "gravel", "clock", "cat", "colorwheel",
)

METHODS = {
    "CONV": conv_transport_resize,
    "Polyphase FIR-8": lambda value, shape: polyphase_fir_resize(
        value, shape, radius=8
    ),
    "Lanczos-sinc (3-lobe)": lanczos3_resize,
    "Bilinear": linear_resize,
}


@dataclass
class State:
    source: np.ndarray | None = None
    source_name: str = ""
    texture_serial: int = 0
    textures: list[str] = field(default_factory=list)


def _raster(value: object) -> np.ndarray:
    if isinstance(value, (tuple, list)):
        value = value[0]
    array = np.asarray(value)
    while array.ndim > 3:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] not in (1,3,4):
        array = array[0]
    if array.ndim not in (2,3):
        raise ValueError(f"provider did not yield a raster: {array.shape}")
    return unit_image(array)


def evaluate_methods(
    source: np.ndarray, target_shape: tuple[int, int], names: list[str],
    methods: dict[str, object] | None = None,
    *, matched_cycle: bool = True,
) -> dict[str, dict[str, object]]:
    # Build/load the shared native library and initialize its persistent worker
    # pool before any method clock begins. Compilation is setup, not filtering.
    backend_description()
    records: dict[str, dict[str, object]] = {}
    available = METHODS if methods is None else methods
    for name in names:
        operation = available[name]
        started = time.perf_counter()
        value = operation(source, target_shape)
        forward = time.perf_counter()-started
        record: dict[str, object] = {
            "value": value,
            "forward_ms": 1000.0*forward,
            "forward_range_excess": range_excess(value, source),
        }
        if matched_cycle:
            started = time.perf_counter()
            returned = operation(value, source.shape[:2])
            cycle = forward+time.perf_counter()-started
            record.update({
                "returned": returned,
                "cycle_ms": 1000.0*cycle,
                "matched_cycle_mse": mse(returned, source),
                "returned_range_excess": range_excess(returned, source),
            })
        records[name] = record
    return records


def run_gui(
    initial_provider: str = "astronaut", *, initial_cycle: bool = False,
    initial_cycle_factor: int = 2,
    methods: dict[str, object] | None = None,
) -> None:
    import dearpygui.dearpygui as dpg
    from PIL import Image
    from skimage import data

    state = State()
    methods = dict(METHODS if methods is None else methods)
    dpg.create_context()

    def status(message: str) -> None:
        dpg.set_value("status", message)

    def apply_source(value: object, name: str) -> None:
        state.source = _raster(value)
        state.source_name = name
        height, width = state.source.shape[:2]
        dpg.set_value("target_width", width*2)
        dpg.set_value("target_height", height*2)
        status(f"Loaded {name}: {width} x {height}.")

    def load_provider(_sender: object = None, app_data: object = None) -> None:
        name = str(app_data if app_data is not None else dpg.get_value("provider"))
        try:
            status(f"Loading skimage.data.{name}() ...")
            apply_source(getattr(data, name)(), f"skimage.data.{name}")
        except Exception as error:
            status(f"Load failed: {type(error).__name__}: {error}")

    def file_selected(_sender: object, app_data: dict[str, object]) -> None:
        try:
            path = Path(str(app_data["file_path_name"]))
            apply_source(np.asarray(Image.open(path)), path.name)
        except Exception as error:
            status(f"File load failed: {type(error).__name__}: {error}")

    def populate_factor(factor: int, shrink: bool) -> None:
        if state.source is None:
            return
        h, w = state.source.shape[:2]
        target_w = round(w/factor) if shrink else w*factor
        target_h = round(h/factor) if shrink else h*factor
        dpg.set_value("target_width", max(5, int(target_w)))
        dpg.set_value("target_height", max(5, int(target_h)))

    def factor_callback(factor: int, shrink: bool):
        return lambda *_: populate_factor(factor, shrink)

    def add_texture(image: np.ndarray) -> tuple[str,int,int]:
        state.texture_serial += 1
        tag = f"texture_{state.texture_serial}"
        rgba, width, height = rgba_preview(image, int(dpg.get_value("preview_side")))
        # Dear PyGui accepts the contiguous float32 buffer directly.  Avoid
        # allocating one Python float object per texture component.
        dpg.add_static_texture(width, height, rgba, tag=tag, parent="textures")
        state.textures.append(tag)
        return tag, width, height

    def add_panel(
        name: str, image: np.ndarray, record: dict[str, object] | None,
        *, cycle_view: bool = False, target: tuple[int,int] | None = None,
    ) -> None:
        with dpg.child_window(width=390, height=650):
            dpg.add_text(name, wrap=360)
            h, w = image.shape[:2]
            dpg.add_text(f"{w} x {h}")
            tag, pw, ph = add_texture(image)
            dpg.add_image(tag, width=pw, height=ph)
            if record is None:
                dpg.add_text("Input raster")
            else:
                if cycle_view and target is not None:
                    dpg.add_text(
                        f"displayed: shrink to {target[1]} x {target[0]}, "
                        "then enlarge to source"
                    )
                else:
                    dpg.add_text("displayed: one-pass target raster")
                dpg.add_text(f"forward              {record['forward_ms']:.3f} ms")
                if cycle_view:
                    dpg.add_text(f"matched cycle        {record['cycle_ms']:.3f} ms")
                    dpg.add_text(f"matched cycle MSE    {record['matched_cycle_mse']:.9g}")
                excess = (
                    record["returned_range_excess"] if cycle_view
                    else record["forward_range_excess"]
                )
                dpg.add_text(f"display range excess {excess:.9g}")

    def compute(_sender: object = None, _app_data: object = None) -> None:
        if state.source is None:
            status("Load a raster first.")
            return
        try:
            target = (
                max(5, int(dpg.get_value("target_height"))),
                max(5, int(dpg.get_value("target_width"))),
            )
            cycle_view=bool(dpg.get_value("cycle_view"))
            if cycle_view and not (
                target[0] < state.source.shape[0]
                and target[1] < state.source.shape[1]
            ):
                status(
                    "Matched shrink→enlarge view requires both target "
                    "dimensions to be smaller than the source."
                )
                return
            selected = [name for index,name in enumerate(methods) if dpg.get_value(f"method_{index}")]
            if not selected:
                status("Select at least one method.")
                return
            status(f"Computing {len(selected)} matched resize families on demand ...")
            dpg.delete_item("results", children_only=True)
            dpg.delete_item("textures", children_only=True)
            state.textures.clear()
            records = evaluate_methods(
                state.source, target, selected, methods,
                matched_cycle=cycle_view,
            )
            with dpg.group(horizontal=True, parent="results"):
                add_panel(
                    f"Source: {state.source_name}",state.source,None,
                    cycle_view=cycle_view,target=target,
                )
                for name in selected:
                    shown=(
                        records[name]["returned"] if cycle_view
                        else records[name]["value"]
                    )
                    title=(
                        f"{name} — matched shrink→enlarge" if cycle_view
                        else name
                    )
                    add_panel(
                        title,shown,records[name],cycle_view=cycle_view,
                        target=target,
                    )
            status(
                (
                    f"Complete matched double pass: {state.source.shape[1]} x "
                    f"{state.source.shape[0]} → {target[1]} x {target[0]} → "
                    f"{state.source.shape[1]} x {state.source.shape[0]}. "
                    if cycle_view else
                    f"Complete: {state.source.shape[1]} x {state.source.shape[0]} "
                    f"→ {target[1]} x {target[0]}. "
                ) + backend_description()
            )
        except Exception as error:
            status(f"Compute failed: {type(error).__name__}: {error}")

    with dpg.texture_registry(tag="textures"):
        pass
    with dpg.file_dialog(
        directory_selector=False, show=False, callback=file_selected,
        tag="file_dialog", width=800, height=500,
    ):
        dpg.add_file_extension("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp){.png,.jpg,.jpeg,.tif,.tiff,.bmp}")
        dpg.add_file_extension(".*")
    with dpg.window(tag="primary", label="CONV arbitrary image resize"):
        dpg.add_text("Four matched arbitrary-resize operators; every result is computed when requested.")
        with dpg.group(horizontal=True):
            dpg.add_combo(PROVIDERS, default_value=initial_provider, tag="provider", label="skimage.data", width=190, callback=load_provider)
            dpg.add_button(label="Load provider", callback=load_provider)
            dpg.add_button(label="Open image...", callback=lambda *_: dpg.show_item("file_dialog"))
            dpg.add_input_int(default_value=1024, min_value=128, max_value=2048, min_clamped=True, max_clamped=True, tag="preview_side", label="nearest preview max side", width=130)
        with dpg.group(horizontal=True):
            dpg.add_input_int(default_value=1024, min_value=5, min_clamped=True, tag="target_width", label="new width", width=140)
            dpg.add_input_int(default_value=1024, min_value=5, min_clamped=True, tag="target_height", label="new height", width=140)
            dpg.add_text("Populate new size:")
            for factor in (2,4,8):
                dpg.add_button(label=f"/{factor}", callback=factor_callback(factor, True))
            for factor in (2,4,8):
                dpg.add_button(label=f"x{factor}", callback=factor_callback(factor, False))
            dpg.add_checkbox(
                label="Display matched shrink→enlarge",
                default_value=False,tag="cycle_view",
            )
            dpg.add_button(label="Compute", callback=compute)
        with dpg.group(horizontal=True):
            for index, name in enumerate(methods):
                dpg.add_checkbox(
                    label=name,
                    default_value=True,
                    tag=f"method_{index}",
                )
        dpg.add_text("", tag="status", wrap=1450)
        dpg.add_separator()
        dpg.add_group(tag="results")

    dpg.create_viewport(title="CONV matched resize comparison", width=1580, height=900)
    dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window("primary", True)
    load_provider(app_data=initial_provider)
    if initial_cycle:
        populate_factor(initial_cycle_factor,True)
        dpg.set_value("cycle_view",True)
        compute()
    dpg.start_dearpygui(); dpg.destroy_context()


def main(methods: dict[str, object] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="astronaut", choices=PROVIDERS)
    parser.add_argument(
        "--cycle",action="store_true",
        help="start in matched shrink-then-enlarge display mode",
    )
    parser.add_argument(
        "--cycle-factor", type=int, choices=(2, 4, 8), default=2,
        help="dyadic reduction used with --cycle",
    )
    args = parser.parse_args()
    run_gui(
        args.provider, initial_cycle=args.cycle,
        initial_cycle_factor=args.cycle_factor,
        methods=methods,
    )


if __name__ == "__main__":
    main()
