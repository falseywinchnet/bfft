"""ITD ablation: sampled extrema versus admitted-CONV current extrema."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial import Polynomial


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conv_itd_comparison import (  # noqa: E402
    cases,
    conv_irregular,
    extrema,
    natural_cubic,
    turning_count,
)
from experiments.convstar import (  # noqa: E402
    ordered_sign_ledger,
    project_signed_fibres,
    raw_current_jet_bank,
)


OUT = ROOT / "experiments" / "out" / "conv_itd_extrema_ablation"


def admitted_controls(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(signal, dtype=np.float64)[:, None]
    raw, delta = raw_current_jet_bank(source)
    signs = ordered_sign_ledger(raw, delta)
    current = project_signed_fibres(raw, signs, delta)[..., 0]
    control = np.concatenate(
        (source[:-1], source[:-1] + np.cumsum(current, axis=1)), axis=1
    )
    return current, control


def bernstein_value(control: np.ndarray, u: float) -> float:
    return float(sum(
        math.comb(5, k)*u**k*(1-u)**(5-k)*control[k]
        for k in range(6)
    ))


def conv_current_extrema(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sign-changing roots of the admitted quintic derivative."""

    current, control = admitted_controls(signal)
    unit = Polynomial((0.0, 1.0)); complement = Polynomial((1.0, -1.0))
    position = [0.0]; value = [float(signal[0])]
    derivatives: list[Polynomial] = []
    for cell in range(current.shape[0]):
        derivative = Polynomial((0.0,))
        for k in range(5):
            derivative += (
                5.0*current[cell, k]*math.comb(4, k)
                * unit**k * complement**(4-k)
            )
        derivatives.append(derivative)
        for root in derivative.roots():
            if abs(root.imag) > 1e-9:
                continue
            u = float(root.real)
            # A root numerically coincident with a source knot is handled once
            # below from the two one-sided admitted derivatives.
            if not 1e-6 < u < 1-1e-6:
                continue
            epsilon = min(1e-6, 0.25*min(u, 1-u))
            if derivative(u-epsilon)*derivative(u+epsilon) >= 0:
                continue
            position.append(cell+u)
            value.append(bernstein_value(control[cell], u))
    for knot in range(1, signal.size-1):
        left = derivatives[knot-1](1.0-1e-7)
        right = derivatives[knot](1e-7)
        if left*right < 0.0:
            position.append(float(knot))
            value.append(float(signal[knot]))
    position.append(float(signal.size-1)); value.append(float(signal[-1]))
    order = np.argsort(position)
    position = np.asarray(position)[order]
    value = np.asarray(value)[order]
    keep = np.r_[True, np.diff(position) > 1e-6]
    position, value = position[keep], value[keep]

    # A cell root can converge immediately beside a nonsmooth source knot and
    # evaluate to the bit-identical profile value.  Consecutive extrema with
    # no value current between them are one zero-persistence extremum, not two
    # anchors.  Prefer the exact source knot when one representative is an
    # integer; this introduces no amplitude or distance threshold.
    reduced_position = [position[0]]
    reduced_value = [value[0]]
    for candidate_position, candidate_value in zip(position[1:], value[1:]):
        if candidate_value == reduced_value[-1]:
            if candidate_position == round(candidate_position):
                reduced_position[-1] = candidate_position
                reduced_value[-1] = candidate_value
            continue
        reduced_position.append(candidate_position)
        reduced_value.append(candidate_value)
    return np.asarray(reduced_position), np.asarray(reduced_value)


def itd_knots_at(ext_x: np.ndarray, ext_y: np.ndarray) -> np.ndarray:
    knot = np.empty(ext_x.size, dtype=np.float64)
    knot[0], knot[-1] = ext_y[0], ext_y[-1]
    for k in range(1, ext_x.size-1):
        weight = (ext_x[k]-ext_x[k-1])/(ext_x[k+1]-ext_x[k-1])
        chord = ext_y[k-1] + weight*(ext_y[k+1]-ext_y[k-1])
        knot[k] = 0.5*(chord+ext_y[k])
    return knot


def evaluate_variant(
    ext_x: np.ndarray,
    ext_y: np.ndarray,
    query: np.ndarray,
    baseline_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    knot = itd_knots_at(ext_x, ext_y)
    baseline = (
        natural_cubic(ext_x, knot, query)
        if baseline_kind == "cubic"
        else conv_irregular(ext_x, knot, query)
    )
    return baseline, knot


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/conv_itd_mpl")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    examples = cases()
    variants = (
        ("sample+cubic", "sample", "cubic", "#d97706"),
        ("sample+CONV", "sample", "conv", "#0369a1"),
        ("CONV-extrema+cubic", "conv", "cubic", "#a21caf"),
        ("CONV-extrema+CONV", "conv", "conv", "#15803d"),
    )
    records: dict[str, dict[str, dict[str, float|int]]] = {}
    fig, axes = plt.subplots(len(examples), 2, figsize=(14, 2.65*len(examples)), constrained_layout=True)

    for row, (name, (signal, truth)) in enumerate(examples.items()):
        sample_i = extrema(signal).astype(float)
        sample_y = signal[sample_i.astype(int)]
        conv_x, conv_y = conv_current_extrema(signal)
        query = np.arange(signal.size, dtype=float)
        records[name] = {}
        ax0, ax1 = axes[row]
        ax0.plot(signal, color="0.82", lw=.7, label="signal")
        ax0.plot(truth, "k--", lw=1, label="known trend")
        for label, clock, baseline_kind, color in variants:
            ex, ey = (sample_i, sample_y) if clock == "sample" else (conv_x, conv_y)
            baseline, knot = evaluate_variant(ex, ey, query, baseline_kind)
            ax0.plot(baseline, color=color, lw=1, label=label)
            ax1.plot(signal-baseline, color=color, lw=.85, label=label)
            records[name][label] = {
                "source_extrema": int(ex.size-2),
                "baseline_knot_extrema": int(turning_count(knot)),
                "baseline_extrema": int(turning_count(baseline)),
                "truth_mse": float(np.mean((baseline-truth)**2)),
                "rotation_range": float(np.ptp(signal-baseline)),
            }
        records[name]["clock_comparison"] = {
            "sample_extrema": int(sample_i.size-2),
            "conv_current_extrema": int(conv_x.size-2),
            "mean_location_shift": float(np.mean(np.abs(sample_i-conv_x)))
                if sample_i.size == conv_x.size else float("nan"),
            "max_location_shift": float(np.max(np.abs(sample_i-conv_x)))
                if sample_i.size == conv_x.size else float("nan"),
        }
        ax0.set_title(name); ax0.set_xticks([]); ax1.set_xticks([])
        ax1.axhline(0, color="0.6", lw=.5)
        if row == 0:
            ax0.legend(ncol=3, fontsize=7); ax1.legend(ncol=2, fontsize=7)
    fig.savefig(OUT / "itd_extrema_ablation.png", dpi=180)
    plt.close(fig)
    (OUT / "metrics.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
