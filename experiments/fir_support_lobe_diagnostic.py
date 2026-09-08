"""Separate long FIR phase support from the effect of signed lobes.

The signed arms reproduce the demo's scale-widened Lanczos formula.  Their
positive controls keep the identical tap interval and set only negative raw
weights to zero before DC normalization.  This is an ablation, not a proposed
interpolator: it asks how much diagonal coherence survives when the FIR can no
longer leave the convex hull of its support.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
for directory in (ROOT, DEMO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend import (  # noqa: E402
    conv_transport_resize,
    linear_resize,
    polyphase_fir_resize,
)
from experiments.conv_phase_transport_diagnostic import (  # noqa: E402
    _central_mse,
    analytic_feature,
    tangent_ripple,
)


Array = np.ndarray


def fir_matrix(n: int, m: int, radius: int, *, positive: bool) -> Array:
    """Return the demo FIR's complete endpoint-clamped line matrix."""

    n, m, radius = int(n), int(m), int(radius)
    rate = (m - 1) / (n - 1) if n > 1 and m > 1 else 1.0
    cutoff = min(rate, 1.0)
    support = radius / cutoff
    matrix = np.zeros((m, n), dtype=np.float64)
    for target in range(m):
        t = 0.0 if m == 1 else target * (n - 1) / (m - 1)
        begin = math.ceil(t - support)
        end = math.floor(t + support)
        taps = np.arange(begin, end + 1, dtype=np.int64)
        distance = t - taps
        raw = cutoff * np.sinc(cutoff * distance) * np.sinc(
            distance / support
        )
        if positive:
            raw = np.maximum(raw, 0.0)
        normalizer = float(np.sum(raw))
        if abs(normalizer) < 1e-18:
            normalizer = 1.0
        weights = raw / normalizer
        np.add.at(matrix[target], np.clip(taps, 0, n - 1), weights)
    return matrix


def matrix_resize(value: Array, target: tuple[int, int], radius: int, positive: bool) -> Array:
    source = np.asarray(value, dtype=np.float64)
    wy = fir_matrix(source.shape[0], target[0], radius, positive=positive)
    wx = fir_matrix(source.shape[1], target[1], radius, positive=positive)
    if source.ndim == 2:
        return wy @ source @ wx.T
    return np.einsum("ai,ijc,bj->abc", wy, source, wx, optimize=True)


def matched(value: Array, radius: int, positive: bool) -> Array:
    coarse = matrix_resize(value, (33, 33), radius, positive)
    return matrix_resize(coarse, value.shape[:2], radius, positive)


def infinite_phase_kernel(radius: int, phase: float, *, positive: bool) -> tuple[Array, Array]:
    begin = math.ceil(phase - radius)
    end = math.floor(phase + radius)
    taps = np.arange(begin, end + 1, dtype=np.float64)
    distance = phase - taps
    raw = np.sinc(distance) * np.sinc(distance / radius)
    if positive:
        raw = np.maximum(raw, 0.0)
    weights = raw / np.sum(raw)
    return taps, weights


def kernel_diagnostic(radius: int, positive: bool) -> dict[str, object]:
    phases = (0.0, 0.25, 0.5, 0.75)
    omega = np.linspace(0.0, np.pi, 2049)
    responses = []
    negative_mass = []
    gains = []
    for phase in phases:
        taps, weights = infinite_phase_kernel(radius, phase, positive=positive)
        response = np.exp(
            1j * omega[:, None] * (taps[None] - phase)
        ) @ weights
        responses.append(response)
        negative_mass.append(float(-np.sum(np.minimum(weights, 0.0))))
        gains.append(float(np.sum(np.abs(weights))))
    response = np.stack(responses, axis=0)
    amplitude = np.abs(response)
    phase_error = np.abs(np.angle(response))
    phase_spread = np.max(amplitude, axis=0) - np.min(amplitude, axis=0)
    passband = omega <= 0.75 * np.pi
    return {
        "radius": radius,
        "positive": positive,
        "phases": list(phases),
        "negative_mass_by_phase": negative_mass,
        "maximum_negative_mass": float(np.max(negative_mass)),
        "maximum_1d_linf_gain": float(np.max(gains)),
        "maximum_2d_separable_linf_gain_bound": float(np.max(gains) ** 2),
        "mean_passband_amplitude_error": float(np.mean(
            np.abs(amplitude[:, passband] - 1.0)
        )),
        "maximum_passband_phase_amplitude_spread": float(np.max(
            phase_spread[passband]
        )),
        "mean_passband_phase_amplitude_spread": float(np.mean(
            phase_spread[passband]
        )),
        "maximum_passband_phase_error_radians": float(np.max(
            phase_error[:, passband]
        )),
        "omega": omega,
        "amplitude": amplitude,
    }


def run_census() -> tuple[list[dict[str, object]], dict[str, object]]:
    methods = {
        "CONV basin": lambda value: conv_transport_resize(
            conv_transport_resize(value, (33, 33)), value.shape[:2]
        ),
        "FIR-8 signed": lambda value: matched(value, 8, False),
        "FIR-8 positive": lambda value: matched(value, 8, True),
        "FIR-3 signed": lambda value: matched(value, 3, False),
        "FIR-3 positive": lambda value: matched(value, 3, True),
        "Bilinear": lambda value: linear_resize(
            linear_resize(value, (33, 33)), value.shape[:2]
        ),
    }
    records: list[dict[str, object]] = []
    for family, width in (("ridge", 1.15), ("edge", 0.85)):
        for angle in (15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0):
            for phase in np.linspace(-1.5, 1.5, 7):
                truth = analytic_feature(
                    129, angle, float(phase), family, width
                ).astype(np.float64)
                low, high = float(np.min(truth)), float(np.max(truth))
                for name, operation in methods.items():
                    estimate = operation(truth)
                    ripple = tangent_ripple(
                        estimate, truth, angle, float(phase), family
                    )
                    records.append({
                        "method": name,
                        "family": family,
                        "angle_degrees": angle,
                        "phase": float(phase),
                        "mse": _central_mse(estimate, truth),
                        "tangent_ripple_rms": ripple["ripple_rms"],
                        "range_excess": max(
                            0.0,
                            low - float(np.min(estimate)),
                            float(np.max(estimate)) - high,
                        ),
                    })
    summary: dict[str, object] = {}
    for name in methods:
        group = [record for record in records if record["method"] == name]
        mse = np.array([record["mse"] for record in group], dtype=np.float64)
        ripple = np.array([
            record["tangent_ripple_rms"] for record in group
        ], dtype=np.float64)
        excess = np.array([
            record["range_excess"] for record in group
        ], dtype=np.float64)
        summary[name] = {
            "case_count": len(group),
            "geometric_mean_mse": float(np.exp(np.mean(np.log(mse)))),
            "mean_tangent_ripple_rms": float(np.mean(ripple)),
            "maximum_tangent_ripple_rms": float(np.max(ripple)),
            "mean_range_excess": float(np.mean(excess)),
            "maximum_range_excess": float(np.max(excess)),
            "range_excess_cases": int(np.count_nonzero(excess > 1e-8)),
        }
    return records, summary


def plot_kernel_diagnostics(kernel: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    styles = {
        "FIR-8 signed": (8, False, "tab:blue"),
        "FIR-8 positive": (8, True, "tab:cyan"),
        "FIR-3 signed": (3, False, "tab:red"),
        "FIR-3 positive": (3, True, "tab:orange"),
    }
    for name, (radius, positive, colour) in styles.items():
        taps, weights = infinite_phase_kernel(radius, 0.5, positive=positive)
        axes[0, 0].plot(
            taps, weights, "o-", color=colour, markersize=2.5,
            linewidth=0.9, label=name,
        )
        item = kernel[name]
        omega = np.asarray(item.pop("omega"))
        amplitude = np.asarray(item.pop("amplitude"))
        axes[0, 1].plot(
            omega / np.pi, np.mean(amplitude, axis=0), color=colour,
            label=name,
        )
        axes[1, 0].plot(
            omega / np.pi,
            np.max(amplitude, axis=0) - np.min(amplitude, axis=0),
            color=colour,
            label=name,
        )
    axes[0, 0].set_title("Half-phase synthesis weights")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.6)
    axes[0, 0].set_xlabel("source tap")
    axes[0, 0].set_ylabel("normalized weight")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Mean fractional-phase amplitude response")
    axes[0, 1].set_xlabel(r"normalized frequency $\omega/\pi$")
    axes[0, 1].set_ylabel("amplitude")
    axes[0, 1].set_ylim(0.0, 1.35)
    axes[1, 0].set_title("Amplitude spread across quarter phases")
    axes[1, 0].set_xlabel(r"normalized frequency $\omega/\pi$")
    axes[1, 0].set_ylabel("max phase - min phase")
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 1].axis("off")
    text = [
        "Exact structural facts:",
        "signed sinc lobes imply negative coefficients;",
        "DC normalization gives ||W||∞ = 1 + 2 negative mass;",
        "positive projection gives ||W||∞ = 1 and convex-hull output.",
    ]
    axes[1, 1].text(0.02, 0.95, "\n".join(text), va="top", fontsize=10)
    for axis in axes.flat[:3]:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_behavior(summary: dict[str, object], output: Path) -> None:
    names = list(summary)
    ripple = [summary[name]["mean_tangent_ripple_rms"] for name in names]
    mse = [summary[name]["geometric_mean_mse"] for name in names]
    excess = [summary[name]["mean_range_excess"] for name in names]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, values, title in zip(
        axes,
        (ripple, mse, excess),
        ("Mean tangential ripple RMS", "Geometric-mean MSE", "Mean range excess"),
    ):
        axis.bar(np.arange(len(names)), values)
        axis.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    records, summary = run_census()
    kernel = {
        name: kernel_diagnostic(radius, positive)
        for name, radius, positive in (
            ("FIR-8 signed", 8, False),
            ("FIR-8 positive", 8, True),
            ("FIR-3 signed", 3, False),
            ("FIR-3 positive", 3, True),
        )
    }
    plot_kernel_diagnostics(kernel, output / "kernel_phase_response.png")
    plot_behavior(summary, output / "behavior_ablation.png")
    generator = np.random.default_rng(1907)
    validation_source = generator.normal(size=(33, 29)).astype(np.float32)
    native_validation = {}
    for radius in (3, 8):
        reference = matrix_resize(validation_source, (129, 113), radius, False)
        native = polyphase_fir_resize(
            validation_source, (129, 113), radius=radius
        )
        native_validation[f"radius_{radius}_maximum_absolute_difference"] = (
            float(np.max(np.abs(reference - native)))
        )
    result = {
        "design": {
            "cycle": "129 -> 33 -> 129",
            "cases": 126,
            "positive_control": (
                "same tap interval; negative raw weights set to zero; "
                "then DC renormalized"
            ),
            "interpretation": "paired ablation, not additive attribution",
        },
        "summary": summary,
        "kernel": kernel,
        "native_validation": native_validation,
        "records": records,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/support_geometry/fir_support_lobe"),
    )
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({
        "design": result["design"],
        "summary": result["summary"],
        "kernel": result["kernel"],
        "native_validation": result["native_validation"],
    }, indent=2))


if __name__ == "__main__":
    main()
