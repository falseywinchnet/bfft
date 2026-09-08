"""Evidence for the compiled convolutional/polyphase CONV* realization."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.conv_synthetic_evidence import (
    _pchip_resize,
    _raw_quintic_resize,
    _scene,
    _sign_changes,
    _timing_source,
    _two_dimensional_cases,
)
from experiments.convstar import (
    convstar_resize,
    ordered_sign_ledger,
    project_signed_fibres,
    raw_current_fused_bank,
    raw_current_jet_bank,
    refine_lines,
)
from experiments.easu_reference import easu_nested_2x
from experiments.self_geometric_harmonic_interpolation import (
    _quintic_variation_lineage_profile,
    lanczos_resize,
    tensor_product_quintic_variation_resize,
)


Array = np.ndarray
COLORS = {
    "formal CONV": "#0072B2",
    "CONV* direct": "#009E73",
    "CONV* fused": "#56B4E9",
    "raw two-jet": "#E69F00",
    "Lanczos-3": "#D55E00",
    "PCHIP": "#999999",
    "EASU": "#882255",
}


def _hardware() -> dict[str, str]:
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        text = subprocess.run(
            ("system_profiler", "SPHardwareDataType"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key in ("Model Name", "Model Identifier", "Chip", "Memory"):
                result[key.lower().replace(" ", "_")] = value
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _median_time(operation: object, source: Array, repeats: int) -> float:
    operation(source)
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        result = operation(source)
        elapsed.append(perf_counter() - start)
    finite = (
        all(np.all(np.isfinite(value)) for value in result)
        if isinstance(result, tuple)
        else np.all(np.isfinite(result))
    )
    if not finite:
        raise AssertionError("timed operation returned nonfinite output")
    return float(np.median(elapsed))


def _line_equivalence() -> dict[str, object]:
    rng = np.random.default_rng(20260827)
    records = []
    maximum_current_error = 0.0
    maximum_output_error = 0.0
    maximum_cardinal_error = 0.0
    maximum_mass_error = 0.0
    maximum_sign_surplus = 0
    for side in (5, 9, 17, 33, 65, 129):
        for trial in range(12):
            source = rng.normal(size=(side, 3))
            raw, delta = raw_current_jet_bank(source)
            signs = ordered_sign_ledger(raw, delta)
            current = project_signed_fibres(raw, signs, delta)
            _, formal_current = _quintic_variation_lineage_profile(source)
            current_error = float(np.max(np.abs(current - formal_current)))
            compiled = refine_lines(source, 2)
            formal = np.empty_like(compiled)
            formal[::2] = source
            for interval in range(side - 1):
                control = np.concatenate((
                    source[interval][None],
                    source[interval][None]
                    + np.cumsum(formal_current[interval], axis=0),
                ))
                value = control.copy()
                for _ in range(5):
                    value = 0.5 * value[:-1] + 0.5 * value[1:]
                formal[2 * interval + 1] = value[0]
            output_error = float(np.max(np.abs(compiled - formal)))
            cardinal_error = float(np.max(np.abs(compiled[::2] - source)))
            mass_error = float(np.max(np.abs(np.sum(current, axis=1) - delta)))
            sign_surplus = max(
                _sign_changes(current.transpose(0, 1, 2).reshape(-1, 3)[:, 0])
                - _sign_changes(np.diff(source[:, 0])),
                0,
            )
            maximum_current_error = max(maximum_current_error, current_error)
            maximum_output_error = max(maximum_output_error, output_error)
            maximum_cardinal_error = max(maximum_cardinal_error, cardinal_error)
            maximum_mass_error = max(maximum_mass_error, mass_error)
            maximum_sign_surplus = max(maximum_sign_surplus, sign_surplus)
            records.append({
                "side": side,
                "trial": trial,
                "current_linf": current_error,
                "output_linf": output_error,
                "cardinal_linf": cardinal_error,
                "mass_linf": mass_error,
            })
    return {
        "trial_count": len(records),
        "maximum_current_linf": maximum_current_error,
        "maximum_output_linf": maximum_output_error,
        "maximum_cardinal_linf": maximum_cardinal_error,
        "maximum_mass_linf": maximum_mass_error,
        "maximum_sign_surplus": maximum_sign_surplus,
        "records": records,
    }


def _field_equivalence() -> dict[str, object]:
    records = []
    for source_side in (9, 17, 33):
        for case_index, case in enumerate(_two_dimensional_cases(source_side)):
            source = _scene(
                str(case["kind"]), source_side,
                angle=float(case["angle"]),
                phase=float(case["phase"]),
                offset=float(case["offset"]),
            )
            formal = tensor_product_quintic_variation_resize(source, 2)
            compiled = convstar_resize(source, 2)
            records.append({
                "source_side": source_side,
                "case_index": case_index,
                "kind": case["kind"],
                "maximum_absolute_difference": float(
                    np.max(np.abs(compiled - formal))
                ),
                "mean_square_difference": float(np.mean((compiled - formal) ** 2)),
                "cardinal_error": float(np.max(np.abs(compiled[::2, ::2] - source))),
            })
    return {
        "case_count": len(records),
        "maximum_absolute_difference": max(
            row["maximum_absolute_difference"] for row in records
        ),
        "maximum_mean_square_difference": max(
            row["mean_square_difference"] for row in records
        ),
        "maximum_cardinal_error": max(row["cardinal_error"] for row in records),
        "records": records,
    }


def _resize_timing(repeats: int) -> list[dict[str, object]]:
    operations = {
        "formal CONV": lambda value: tensor_product_quintic_variation_resize(value, 2),
        "CONV* direct": lambda value: convstar_resize(value, 2, front_end="jet"),
        "CONV* fused": lambda value: convstar_resize(value, 2, front_end="fused"),
        "raw two-jet": lambda value: _raw_quintic_resize(value, 2),
        "Lanczos-3": lambda value: lanczos_resize(value, 2, radius=3),
        "PCHIP": lambda value: _pchip_resize(value, 2),
        "EASU": easu_nested_2x,
    }
    records = []
    for side in (17, 33, 65):
        source = _timing_source(side)
        target_side = 2 * side - 1
        for method, operation in operations.items():
            seconds = _median_time(operation, source, repeats)
            records.append({
                "source_side": side,
                "target_side": target_side,
                "channels": 3,
                "method": method,
                "median_seconds": seconds,
                "microseconds_per_output_pixel": (
                    1.0e6 * seconds / (target_side * target_side)
                ),
            })
    return records


def _front_end_timing(repeats: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(73)
    operations = {
        "jet FIR": raw_current_jet_bank,
        "fused 5x6 FIR": raw_current_fused_bank,
    }
    records = []
    for length in (257, 1025, 4097, 16385, 65537):
        source = rng.normal(size=(length, 16))
        for method, operation in operations.items():
            seconds = _median_time(operation, source, repeats)
            records.append({
                "length": length,
                "components": 16,
                "method": method,
                "median_seconds": seconds,
                "nanoseconds_per_input": 1.0e9 * seconds / source.size,
            })
    return records


def _plot_resize_timing(records: list[dict[str, object]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.08, 2.75))
    methods = tuple(
        method
        for method in dict.fromkeys(str(row["method"]) for row in records)
        if method != "raw two-jet"
    )
    for method in methods:
        rows = [row for row in records if row["method"] == method]
        axis.semilogy(
            [row["source_side"] for row in rows],
            [1000.0 * row["median_seconds"] for row in rows],
            marker="o", linewidth=1.35, markersize=3.2,
            label=method, color=COLORS[method],
        )
    axis.set_xlabel("source nodes per axis")
    axis.set_ylabel("median RGB 2x wall time (ms)")
    axis.set_xticks((17, 33, 65))
    axis.grid(True, which="both", linewidth=0.35, alpha=0.45)
    axis.legend(frameon=False, ncol=3, fontsize=7)
    figure.tight_layout(pad=0.5)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_front_end(records: list[dict[str, object]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.08, 2.75))
    for method, color in (
        ("jet FIR", "#009E73"),
        ("fused 5x6 FIR", "#56B4E9"),
    ):
        rows = [row for row in records if row["method"] == method]
        axis.loglog(
            [row["length"] for row in rows],
            [1.0e6 * row["median_seconds"] for row in rows],
            marker="o", linewidth=1.35, markersize=3.2,
            label=method, color=color,
        )
    axis.set_xlabel("line length (16 components)")
    axis.set_ylabel("front-end wall time (microseconds)")
    axis.grid(True, which="both", linewidth=0.35, alpha=0.45)
    axis.legend(frameon=False)
    figure.tight_layout(pad=0.5)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def run(output: Path, repeats: int = 3) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    line = _line_equivalence()
    fields = _field_equivalence()
    resize = _resize_timing(repeats)
    front_end = _front_end_timing(max(repeats, 5))
    result = {
        "method": "CONV* compiled convolutional/polyphase realization",
        "hardware": _hardware(),
        "timing_protocol": (
            "single-process reference CPU; one warmup; median wall time; "
            "no affinity or explicit thread control"
        ),
        "line_equivalence": line,
        "field_equivalence": fields,
        "resize_timing": resize,
        "front_end_timing": front_end,
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    _plot_resize_timing(resize, output / "resize_timing.pdf")
    _plot_front_end(front_end, output / "front_end_timing.pdf")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("output/pdf/convstar_evidence")
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = run(args.out, args.repeats)
    print(json.dumps({
        "hardware": result["hardware"],
        "line_equivalence": {
            key: value for key, value in result["line_equivalence"].items()
            if key != "records"
        },
        "field_equivalence": {
            key: value for key, value in result["field_equivalence"].items()
            if key != "records"
        },
        "resize_timing": result["resize_timing"],
        "front_end_timing": result["front_end_timing"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
