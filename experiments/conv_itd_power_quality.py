"""Power-quality decomposition under deterministic ITD knot-law substitutes.

Every arm uses the same continuous CONV extrema clock and the same irregular
CONV reconstruction. Only the map assigning baseline values at consecutive
extrema is changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conv_itd_comparison import conv_irregular  # noqa: E402
from experiments.conv_itd_extrema_ablation import (  # noqa: E402
    admitted_controls,
    conv_current_extrema,
)

OUT = ROOT / "experiments" / "out" / "conv_itd_power_quality"


def make_power_quality_record(n: int = 2048, sample_rate: float = 2048.0):
    """One second of mains voltage with known distortions and a transient."""
    t = np.arange(n, dtype=np.float64) / sample_rate
    trend = 0.025 * (t - 0.45) + 0.018 * np.cos(2 * np.pi * 0.7 * t)
    sag = 1.0 - 0.22 * np.exp(-((t - 0.66) / 0.085) ** 4)
    mains = sag * np.sin(2 * np.pi * 60.0 * t + 0.23)
    fifth = 0.145 * np.sin(2 * np.pi * 300.0 * t - 0.61)
    interharmonic = 0.18 * np.sin(
        2 * np.pi * (105.0 * t + 0.5 * 100.0 * t * t) + 0.9
    )
    event_start = 0.565
    event_time = np.maximum(t - event_start, 0.0)
    switching = (
        0.34 * (t >= event_start) * np.exp(-event_time / 0.032)
        * np.sin(2 * np.pi * 690.0 * event_time + 0.35)
    )
    rng = np.random.default_rng(20260830)
    noise = 0.028 * rng.standard_normal(n)
    parts = {
        "60 Hz sagged mains": mains,
        "300 Hz fifth harmonic": fifth,
        "105–205 Hz interharmonic": interharmonic,
        "690 Hz switching transient": switching,
        "measurement noise": noise,
    }
    return t, trend + sum(parts.values()), {"slow drift": trend, **parts}, (
        event_start, event_start + 0.12
    )


def _finish(ext_y: np.ndarray, knot: np.ndarray) -> np.ndarray:
    knot[0], knot[-1] = ext_y[0], ext_y[-1]
    return knot


def itd_knots(signal: np.ndarray, ext_x: np.ndarray, ext_y: np.ndarray) -> np.ndarray:
    """Exact ITD: half central extremum, half opposite chord."""
    del signal
    knot = np.empty_like(ext_y)
    weight = (ext_x[1:-1] - ext_x[:-2]) / (ext_x[2:] - ext_x[:-2])
    chord = ext_y[:-2] + weight * (ext_y[2:] - ext_y[:-2])
    knot[1:-1] = 0.5 * (ext_y[1:-1] + chord)
    return _finish(ext_y, knot)


def chord_knots(signal: np.ndarray, ext_x: np.ndarray, ext_y: np.ndarray) -> np.ndarray:
    """Full local affine prediction from the neighboring extrema."""
    del signal
    knot = np.empty_like(ext_y)
    weight = (ext_x[1:-1] - ext_x[:-2]) / (ext_x[2:] - ext_x[:-2])
    knot[1:-1] = ext_y[:-2] + weight * (ext_y[2:] - ext_y[:-2])
    return _finish(ext_y, knot)


def binomial_knots(signal: np.ndarray, ext_x: np.ndarray, ext_y: np.ndarray) -> np.ndarray:
    """Equal-extremum-clock 1:2:1 smoothing."""
    del signal, ext_x
    knot = np.empty_like(ext_y)
    knot[1:-1] = 0.25 * ext_y[:-2] + 0.5 * ext_y[1:-1] + 0.25 * ext_y[2:]
    return _finish(ext_y, knot)


def voronoi_knots(signal: np.ndarray, ext_x: np.ndarray, ext_y: np.ndarray) -> np.ndarray:
    """Mean of the polygonal extrema profile over each Voronoi basin."""
    del signal
    knot = np.empty_like(ext_y)
    left_h = ext_x[1:-1] - ext_x[:-2]
    right_h = ext_x[2:] - ext_x[1:-1]
    knot[1:-1] = (
        left_h * ext_y[:-2]
        + 3.0 * (left_h + right_h) * ext_y[1:-1]
        + right_h * ext_y[2:]
    ) / (4.0 * (left_h + right_h))
    return _finish(ext_y, knot)


def affine_ls_knots(signal: np.ndarray, ext_x: np.ndarray, ext_y: np.ndarray) -> np.ndarray:
    """Central value of the least-squares affine fit to three extrema."""
    del signal
    knot = np.empty_like(ext_y)
    for k in range(1, ext_x.size - 1):
        dx = ext_x[k - 1:k + 2] - ext_x[k]
        design = np.column_stack((np.ones(3), dx))
        knot[k] = np.linalg.lstsq(
            design, ext_y[k - 1:k + 2], rcond=None
        )[0][0]
    return _finish(ext_y, knot)


def _bernstein_quintic(control: np.ndarray, u: np.ndarray) -> np.ndarray:
    coefficients = np.array((1.0, 5.0, 10.0, 10.0, 5.0, 1.0))
    power = np.arange(6)
    basis = coefficients * u[:, None] ** power * (1.0 - u[:, None]) ** (5 - power)
    return basis @ control


def _integrate_admitted_profile(
    control: np.ndarray,
    left: float,
    right: float,
) -> float:
    """Exact degree-five quadrature after splitting at source knots."""
    if right <= left:
        return 0.0
    nodes = np.array((-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)))
    weights = np.array((5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0))
    total = 0.0
    cursor = left
    while cursor < right:
        cell = min(int(np.floor(cursor)), control.shape[0] - 1)
        boundary = min(right, float(cell + 1))
        # Avoid a zero segment when cursor is an integer at the right boundary.
        if boundary <= cursor:
            cell = min(cell + 1, control.shape[0] - 1)
            boundary = min(right, float(cell + 1))
        midpoint = 0.5 * (cursor + boundary)
        half = 0.5 * (boundary - cursor)
        query = midpoint + half * nodes
        values = _bernstein_quintic(control[cell], query - cell)
        total += half * float(weights @ values)
        cursor = boundary
    return total


def conv_basin_knots(
    signal: np.ndarray,
    ext_x: np.ndarray,
    ext_y: np.ndarray,
) -> np.ndarray:
    """Conservative mean of the admitted CONV profile on each extrema basin."""
    _, control = admitted_controls(signal)
    knot = np.empty_like(ext_y)
    for k in range(1, ext_x.size - 1):
        left = 0.5 * (ext_x[k - 1] + ext_x[k])
        right = 0.5 * (ext_x[k] + ext_x[k + 1])
        knot[k] = _integrate_admitted_profile(control, left, right) / (right - left)
    return _finish(ext_y, knot)


def chord_blend_knots(alpha: float):
    """Return a universal centre/chord blend with the stated coefficient."""
    def law(signal: np.ndarray, ext_x: np.ndarray, ext_y: np.ndarray) -> np.ndarray:
        del signal
        knot = np.empty_like(ext_y)
        weight = (ext_x[1:-1] - ext_x[:-2]) / (ext_x[2:] - ext_x[:-2])
        chord = ext_y[:-2] + weight * (ext_y[2:] - ext_y[:-2])
        knot[1:-1] = (1.0 - alpha) * ext_y[1:-1] + alpha * chord
        return _finish(ext_y, knot)
    return law


def basin_itd_knots(
    signal: np.ndarray,
    ext_x: np.ndarray,
    ext_y: np.ndarray,
) -> np.ndarray:
    """Conserve each basin first, then apply ITD cancellation to those states."""
    basin = conv_basin_knots(signal, ext_x, ext_y)
    return itd_knots(signal, ext_x, basin)


def centered_basin_states(
    signal: np.ndarray,
    ext_x: np.ndarray,
    ext_y: np.ndarray,
) -> np.ndarray:
    """Move each conservative basin mean from its centroid back to tau_k."""
    basin = conv_basin_knots(signal, ext_x, ext_y)
    state = basin.copy()
    left = 0.5 * (ext_x[:-2] + ext_x[1:-1])
    right = 0.5 * (ext_x[1:-1] + ext_x[2:])
    centroid = 0.5 * (left + right)
    slope = (ext_y[2:] - ext_y[:-2]) / (ext_x[2:] - ext_x[:-2])
    state[1:-1] -= slope * (centroid - ext_x[1:-1])
    return _finish(ext_y, state)


def centered_basin_itd_knots(
    signal: np.ndarray,
    ext_x: np.ndarray,
    ext_y: np.ndarray,
) -> np.ndarray:
    state = centered_basin_states(signal, ext_x, ext_y)
    return itd_knots(signal, ext_x, state)


def half_point_centered_basin_knots(
    signal: np.ndarray,
    ext_x: np.ndarray,
    ext_y: np.ndarray,
) -> np.ndarray:
    """ITD on the equal point/centered-basin zeroth-moment state."""
    basin = centered_basin_states(signal, ext_x, ext_y)
    state = 0.5 * (ext_y + basin)
    return itd_knots(signal, ext_x, state)


def _carrier_regression_knots(
    signal: np.ndarray,
    ext_x: np.ndarray,
    ext_y: np.ndarray,
    varying_envelope: bool,
) -> np.ndarray:
    """Separate a slow affine baseline from the alternating extrema carrier."""
    fallback = itd_knots(signal, ext_x, ext_y)
    knot = fallback.copy()
    parity = np.array((1.0, -1.0, 1.0, -1.0, 1.0))
    for k in range(2, ext_x.size - 2):
        dx = ext_x[k - 2:k + 3] - ext_x[k]
        scale = np.max(np.abs(dx))
        z = dx / scale if scale > 0.0 else dx
        columns = [np.ones(5), z, parity]
        if varying_envelope:
            columns.append(parity * z)
        design = np.column_stack(columns)
        knot[k] = np.linalg.lstsq(
            design, ext_y[k - 2:k + 3], rcond=None
        )[0][0]
    return _finish(ext_y, knot)


def carrier_knots(signal: np.ndarray, ext_x: np.ndarray, ext_y: np.ndarray) -> np.ndarray:
    return _carrier_regression_knots(signal, ext_x, ext_y, False)


def carrier_envelope_knots(
    signal: np.ndarray,
    ext_x: np.ndarray,
    ext_y: np.ndarray,
) -> np.ndarray:
    return _carrier_regression_knots(signal, ext_x, ext_y, True)


KNOT_LAWS = {
    "3/8 centre-chord": chord_blend_knots(3.0 / 8.0),
    "exact ITD": itd_knots,
    "5/8 centre-chord": chord_blend_knots(5.0 / 8.0),
    "basin-state ITD": basin_itd_knots,
    "centered basin-state ITD": centered_basin_itd_knots,
    "half point + centered basin ITD": half_point_centered_basin_knots,
    "five-extrema carrier regression": carrier_knots,
    "five-extrema carrier-envelope regression": carrier_envelope_knots,
}


def decompose(signal: np.ndarray, knot_law: str, max_levels: int = 12):
    query = np.arange(signal.size, dtype=np.float64)
    state = np.asarray(signal, dtype=np.float64).copy()
    rotations, counts = [], []
    start = time.perf_counter()
    for _ in range(max_levels):
        ext_x, ext_y = conv_current_extrema(state)
        count = int(ext_x.size - 2)
        counts.append(count)
        if count < 2 or ext_x.size < 5:
            break
        knot = KNOT_LAWS[knot_law](state, ext_x, ext_y)
        next_state = conv_irregular(ext_x, knot, query)
        rotations.append(state - next_state)
        state = next_state
    return rotations, state, counts, time.perf_counter() - start


def component_metrics(rotations, baseline, truth, t, event_window):
    names = [name for name in truth if name != "slow drift"]
    corr = np.zeros((len(rotations), len(names)), dtype=np.float64)
    for i, rotation in enumerate(rotations):
        for j, name in enumerate(names):
            corr[i, j] = np.corrcoef(rotation, truth[name])[0, 1]
    absolute = np.abs(corr)
    best_index = np.argmax(absolute, axis=0)
    best = absolute[best_index, np.arange(len(names))]
    concentration = best / np.maximum(np.sum(absolute, axis=0), 1e-15)
    transient_column = names.index("690 Hz switching transient")
    transient_rotation = rotations[int(best_index[transient_column])]
    inside = (t >= event_window[0]) & (t <= event_window[1])
    return {
        "component_names": names,
        "best_rotation": (best_index + 1).tolist(),
        "best_abs_correlation": best.tolist(),
        "single_rotation_unexplained_fraction": (1.0 - best * best).tolist(),
        "correlation_concentration": concentration.tolist(),
        "mean_best_abs_correlation": float(np.mean(best)),
        "mean_correlation_concentration": float(np.mean(concentration)),
        "switching_event_energy_localization": float(
            np.sum(transient_rotation[inside] ** 2)
            / np.maximum(np.sum(transient_rotation ** 2), 1e-30)
        ),
        "slow_drift_mse": float(np.mean((baseline - truth["slow drift"]) ** 2)),
        "slow_drift_correlation": float(
            np.corrcoef(baseline, truth["slow drift"])[0, 1]
        ),
    }, corr


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/conv_itd_mpl")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    t, signal, truth, event_window = make_power_quality_record()
    records = {
        "signal": {
            "sample_count": signal.size,
            "sample_rate_hz": 2048.0,
            "event_window_seconds": list(event_window),
        },
        "invariants": {
            "extrema_clock": "continuous admitted-CONV current roots",
            "baseline_reconstruction": "irregular CONV",
            "changed_quantity": "baseline knot law only",
        },
        "laws": {},
    }
    outputs = {}
    for label in KNOT_LAWS:
        rotations, baseline, counts, elapsed = decompose(signal, label)
        reconstruction = baseline + np.sum(rotations, axis=0)
        metric, corr = component_metrics(rotations, baseline, truth, t, event_window)
        metric.update({
            "rotation_count": len(rotations),
            "extrema_descent": counts,
            "closure_linf": float(np.max(np.abs(signal - reconstruction))),
            "elapsed_seconds": elapsed,
        })
        records["laws"][label] = metric
        outputs[label] = (rotations, baseline, corr)

    (OUT / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    np.savez_compressed(
        OUT / "power_quality_record.npz",
        t=t.astype(np.float32), signal=signal.astype(np.float32),
        **{f"truth_{name.replace(' ', '_').replace('–', '_')}": value.astype(np.float32)
           for name, value in truth.items()},
    )

    fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t, signal, color="0.2", lw=0.55)
    axes[0].set_title("synthetic power-quality record")
    for ax, (name, component) in zip(
        axes[1:], ((k, v) for k, v in truth.items() if k != "slow drift")
    ):
        ax.plot(t, component, lw=0.75)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.savefig(OUT / "source_and_components.png", dpi=180)
    plt.close(fig)

    component_names = list(next(iter(records["laws"].values()))["component_names"])
    fig, axes = plt.subplots(len(KNOT_LAWS), len(component_names), figsize=(17, 10),
                             sharex=True, constrained_layout=True)
    for row, label in enumerate(KNOT_LAWS):
        rotations, _, corr = outputs[label]
        for column, name in enumerate(component_names):
            ax = axes[row, column]
            truth_component = truth[name]
            index = int(np.argmax(np.abs(corr[:, column])))
            rotation = rotations[index]
            a = truth_component - np.mean(truth_component)
            b = rotation - np.mean(rotation)
            scale = float((a @ b) / np.maximum(b @ b, 1e-30))
            ax.plot(t, a, color="0.75", lw=0.65)
            ax.plot(t, scale * b, color="#0369a1", lw=0.55)
            ax.text(0.01, 0.91, f"R{index + 1}, |r|={abs(corr[index, column]):.3f}",
                    transform=ax.transAxes, fontsize=7, va="top")
            if row == 0:
                ax.set_title(name, fontsize=9)
            if column == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
            ax.set_xticks([])
    fig.savefig(OUT / "best_rotation_extractions.png", dpi=180)
    plt.close(fig)

    best = np.asarray([
        records["laws"][label]["best_abs_correlation"] for label in KNOT_LAWS
    ])
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    image = ax.imshow(best, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(component_names)), component_names, rotation=24, ha="right")
    ax.set_yticks(range(len(KNOT_LAWS)), list(KNOT_LAWS))
    for row in range(best.shape[0]):
        for column in range(best.shape[1]):
            ax.text(column, row, f"{best[row, column]:.3f}", ha="center", va="center",
                    color="white" if best[row, column] < 0.72 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="best absolute component correlation")
    ax.set_title("known-component extraction by baseline knot law")
    fig.savefig(OUT / "knot_law_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
