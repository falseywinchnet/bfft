"""Dear PyGui laboratory for arbitrary one-dimensional CONV* resizing."""

from __future__ import annotations

import argparse
import time

import numpy as np

from backend import (
    backend_description,
    conv_transport_resize,
    lanczos3_resize,
    linear_resize,
    polyphase_fir_resize,
)
from gui_common import mse
from signals import SIGNALS, evaluate, samples


METHODS = {
    "CONV": conv_transport_resize,
    "Polyphase FIR-8": lambda value, length: polyphase_fir_resize(
        value, length, radius=8
    ),
    "Lanczos-sinc (3-lobe)": lanczos3_resize,
    "Linear": linear_resize,
}


def evaluate_resize(signal: str, source_length: int, target_length: int) -> dict[str, object]:
    backend_description()
    x_source, source = samples(signal, source_length)
    x_target = np.linspace(0.0, 1.0, target_length)
    truth = evaluate(signal, x_target)
    records: dict[str, object] = {}
    for name, operation in METHODS.items():
        started = time.perf_counter()
        result = operation(source, target_length)
        forward = time.perf_counter()-started
        started = time.perf_counter()
        returned = operation(result, source_length)
        cycle = time.perf_counter()-started+forward
        records[name] = {
            "value": np.asarray(result),
            "error": np.asarray(result)-truth,
            "analytic_mse": mse(result, truth),
            "maximum_error": float(np.max(np.abs(result-truth))),
            "matched_cycle_mse": mse(returned, source),
            "forward_ms": 1000.0*forward,
            "cycle_ms": 1000.0*cycle,
        }
    return {
        "x_source": x_source, "source": source,
        "x_target": x_target, "truth": truth, "records": records,
    }


def _spectrum(
    value: np.ndarray, *, hann: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return a positive, coherently normalized one-sided RFFT amplitude."""

    signal = np.asarray(value, dtype=np.float64)
    window = np.hanning(signal.size) if hann else np.ones(signal.size)
    coherent_sum = float(np.sum(window))
    coefficient = np.fft.rfft(signal*window)
    amplitude = np.abs(coefficient)/coherent_sum
    if amplitude.size > 1:
        stop = -1 if signal.size % 2 == 0 else None
        amplitude[1:stop] *= 2.0
    # Samples span [0,1] inclusively, hence d=1/(N-1). The ordinate remains
    # an amplitude (never dB); the positive floor exists only for log display.
    frequency = np.fft.rfftfreq(signal.size,d=1.0/(signal.size-1))
    amplitude = np.maximum(amplitude,1.0e-10)
    return frequency,amplitude


def run_gui(initial_source: int = 4097, initial_target: int = 8193) -> None:
    import dearpygui.dearpygui as dpg

    dpg.create_context()
    colors = {
        "Analytic truth": (245,245,245,255),
        "Source anchors": (250,185,30,255),
        "CONV": (15,190,230,255),
        "Polyphase FIR-8": (235,80,80,255),
        "Lanczos-sinc (3-lobe)": (155,105,235,255),
        "Linear": (100,210,115,255),
    }

    def set_target_factor(factor: int, shrink: bool) -> None:
        source = max(5, int(dpg.get_value("source_length")))
        target = max(2, int(round(source/factor if shrink else source*factor)))
        dpg.set_value("target_length", target)

    def factor_callback(factor: int, shrink: bool):
        return lambda *_: set_target_factor(factor, shrink)

    def themed_series(x: np.ndarray, y: np.ndarray, name: str, parent: str) -> None:
        tag = dpg.add_line_series(x.tolist(), y.tolist(), label=name, parent=parent)
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, colors[name])
        dpg.bind_item_theme(tag, theme)

    def compute(_sender: object = None, _app_data: object = None) -> None:
        try:
            signal = str(dpg.get_value("signal"))
            n = max(5, int(dpg.get_value("source_length")))
            m = max(2, int(dpg.get_value("target_length")))
            dpg.set_value("status", f"Computing {n} -> {m} ...")
            result = evaluate_resize(signal, n, m)
            use_hann = bool(dpg.get_value("hann_window"))
            for axis in ("wave_y", "error_y", "fft_y"):
                dpg.delete_item(axis, children_only=True)
            dpg.delete_item("metrics", children_only=True)
            stride = max(1, n//2048)
            themed_series(result["x_target"], result["truth"], "Analytic truth", "wave_y")
            source_tag = dpg.add_scatter_series(
                result["x_source"][::stride].tolist(), result["source"][::stride].tolist(),
                label="Source anchors", parent="wave_y",
            )
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, colors["Source anchors"])
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, colors["Source anchors"])
            dpg.bind_item_theme(source_tag, theme)
            f, magnitude = _spectrum(result["truth"],hann=use_hann)
            themed_series(f, magnitude, "Analytic truth", "fft_y")
            f, magnitude = _spectrum(result["source"],hann=use_hann)
            themed_series(f,magnitude,"Source anchors","fft_y")
            marker=dpg.add_inf_line_series(
                [0.5*(n-1)],label="Source Nyquist",horizontal=False,parent="fft_y"
            )
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvInfLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line,(170,170,170,180))
            dpg.bind_item_theme(marker,theme)
            with dpg.table(header_row=True, parent="metrics", policy=dpg.mvTable_SizingStretchProp):
                for label in ("Method", "Analytic point MSE", "max |error|", "matched cycle MSE", "forward ms", "cycle ms"):
                    dpg.add_table_column(label=label)
                for name, record in result["records"].items():
                    themed_series(result["x_target"], record["value"], name, "wave_y")
                    themed_series(result["x_target"], record["error"], name, "error_y")
                    f, magnitude = _spectrum(record["value"],hann=use_hann)
                    themed_series(f, magnitude, name, "fft_y")
                    with dpg.table_row():
                        dpg.add_text(name)
                        dpg.add_text(f"{record['analytic_mse']:.9g}")
                        dpg.add_text(f"{record['maximum_error']:.9g}")
                        dpg.add_text(f"{record['matched_cycle_mse']:.9g}")
                        dpg.add_text(f"{record['forward_ms']:.3f}")
                        dpg.add_text(f"{record['cycle_ms']:.3f}")
            dpg.fit_axis_data("wave_x"); dpg.fit_axis_data("wave_y")
            dpg.fit_axis_data("error_x"); dpg.fit_axis_data("error_y")
            dpg.fit_axis_data("fft_x"); dpg.set_axis_limits("fft_y",1.0e-10,2.0)
            dpg.set_value("status", f"Complete. {backend_description()}")
        except Exception as error:
            dpg.set_value("status", f"Failed: {type(error).__name__}: {error}")

    with dpg.window(tag="primary", label="CONV one-dimensional resize laboratory"):
        dpg.add_text("Analytic truth, pointwise error, matched-family round trip, and physical-domain FFT")
        with dpg.group(horizontal=True):
            dpg.add_combo(SIGNALS, default_value=SIGNALS[0], tag="signal", label="signal", width=230)
            dpg.add_input_int(default_value=initial_source, min_value=5, min_clamped=True, tag="source_length", label="source anchors", width=150)
            dpg.add_input_int(default_value=initial_target, min_value=2, min_clamped=True, tag="target_length", label="new length", width=150)
            dpg.add_checkbox(label="Hann-window RFFT",default_value=False,tag="hann_window")
            dpg.add_button(label="Compute", callback=compute)
        with dpg.group(horizontal=True):
            dpg.add_text("Populate new length:")
            for factor in (2,4,8):
                dpg.add_button(label=f"/{factor}", callback=factor_callback(factor, True))
            for factor in (2,4,8):
                dpg.add_button(label=f"x{factor}", callback=factor_callback(factor, False))
        dpg.add_text("", tag="status")
        with dpg.plot(label="Signal", height=300, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="normalized coordinate", tag="wave_x")
            dpg.add_plot_axis(dpg.mvYAxis, label="amplitude", tag="wave_y")
        with dpg.plot(label="Error against analytic signal", height=250, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="normalized coordinate", tag="error_x")
            dpg.add_plot_axis(dpg.mvYAxis, label="signed error", tag="error_y")
        with dpg.plot(label="Positive one-sided RFFT amplitude", height=320, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="cycles per normalized domain", tag="fft_x")
            dpg.add_plot_axis(
                dpg.mvYAxis,label="one-sided amplitude",tag="fft_y",
                scale=dpg.mvPlotScale_Log10,
            )
        dpg.add_group(tag="metrics")

    dpg.create_viewport(title="CONV* 1-D arbitrary resampling", width=1500, height=1050)
    dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window("primary", True)
    compute()
    dpg.start_dearpygui(); dpg.destroy_context()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=4097)
    parser.add_argument("--target", type=int, default=8193)
    args = parser.parse_args()
    run_gui(args.source, args.target)


if __name__ == "__main__":
    main()
