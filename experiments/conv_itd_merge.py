"""Conservative extrema-basin merging versus the exact ITD split."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conv_itd_comparison import conv_irregular  # noqa: E402
from experiments.conv_itd_extrema_ablation import (  # noqa: E402
    admitted_controls,
    conv_current_extrema,
)
from experiments.conv_itd_power_quality import (  # noqa: E402
    _integrate_admitted_profile,
    component_metrics,
    decompose,
    itd_knots,
    make_power_quality_record,
)

OUT = ROOT / "experiments" / "out" / "conv_itd_merge"


def clipped_basins(owner_x: np.ndarray, end: float):
    left = np.empty_like(owner_x)
    right = np.empty_like(owner_x)
    left[0] = 0.0
    right[-1] = end
    boundary = 0.5 * (owner_x[:-1] + owner_x[1:])
    right[:-1] = boundary
    left[1:] = boundary
    return left, right


def merge_step(state: np.ndarray, placement: str):
    """Discover coarse geometry, conserve its basin masses, then synthesize."""
    query = np.arange(state.size, dtype=np.float64)
    fine_x, fine_y = conv_current_extrema(state)
    if fine_x.size < 5:
        return None

    # The exact ITD split supplies only the coarse geometry.
    proposal_knot = itd_knots(state, fine_x, fine_y)
    proposal = conv_irregular(fine_x, proposal_knot, query)
    owner_x, _ = conv_current_extrema(proposal)
    if owner_x.size >= fine_x.size or owner_x.size < 3:
        return None

    left, right = clipped_basins(owner_x, float(state.size - 1))
    length = right - left
    centroid = 0.5 * (left + right)
    _, control = admitted_controls(state)
    mass = np.asarray([
        _integrate_admitted_profile(control, float(a), float(b))
        for a, b in zip(left, right)
    ])
    mean = mass / length
    if placement == "owner":
        nodes, values = owner_x, mean
    elif placement == "centroid":
        nodes, values = centroid, mean
    elif placement == "centroid+boundary":
        nodes = np.r_[0.0, centroid, float(state.size - 1)]
        values = np.r_[state[0], mean, state[-1]]
    else:
        raise ValueError(placement)
    coarse = conv_irregular(nodes, values, query)

    source_mass = _integrate_admitted_profile(
        control, 0.0, float(state.size - 1)
    )
    _, coarse_control = admitted_controls(coarse)
    uncorrected_mass = _integrate_admitted_profile(
        coarse_control, 0.0, float(state.size - 1)
    )
    # Constant transport is the unique scalar correction of the zeroth moment;
    # it leaves all admitted currents and extrema unchanged.
    mass_correction = (source_mass - uncorrected_mass) / float(state.size - 1)
    coarse = coarse + mass_correction
    _, coarse_control = admitted_controls(coarse)
    synthesis_mass = _integrate_admitted_profile(
        coarse_control, 0.0, float(state.size - 1)
    )
    return coarse, {
        "fine_extrema": int(fine_x.size - 2),
        "coarse_owners": int(owner_x.size - 2),
        "merged_mass_error": float(abs(np.sum(mass) - source_mass)),
        "synthesis_mass_error": float(abs(synthesis_mass - source_mass)),
        "constant_mass_correction": float(mass_correction),
        "coarse_range": [float(np.min(coarse)), float(np.max(coarse))],
    }


def merge_decompose(state: np.ndarray, placement: str, max_levels: int = 12):
    current = np.asarray(state, dtype=np.float64).copy()
    details = []
    records = []
    for level in range(max_levels):
        result = merge_step(current, placement)
        if result is None:
            break
        coarse, record = result
        detail = current - coarse
        record.update({
            "level": level + 1,
            "detail_rms": float(np.sqrt(np.mean(detail * detail))),
        })
        details.append(detail)
        records.append(record)
        current = coarse
    return details, current, records


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/conv_itd_mpl")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    t, signal, truth, event_window = make_power_quality_record()
    exact_rotations, exact_baseline, exact_counts, _ = decompose(signal, "exact ITD")
    methods = {
        "exact ITD split": (exact_rotations, exact_baseline, [
            {"fine_extrema": a, "coarse_owners": b}
            for a, b in zip(exact_counts[:-1], exact_counts[1:])
        ]),
    }
    for placement in ("owner", "centroid", "centroid+boundary"):
        methods[f"conservative merge at {placement}"] = merge_decompose(
            signal, placement
        )

    summary = {"methods": {}}
    correlations = {}
    for label, (rotations, baseline, records) in methods.items():
        metric, corr = component_metrics(
            rotations, baseline, truth, t, event_window
        )
        reconstruction = baseline + np.sum(rotations, axis=0)
        metric.update({
            "levels": len(rotations),
            "closure_linf": float(np.max(np.abs(reconstruction - signal))),
            "hierarchy": records,
        })
        summary["methods"][label] = metric
        correlations[label] = corr
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    component_names = next(iter(summary["methods"].values()))["component_names"]
    best = np.asarray([
        summary["methods"][label]["best_abs_correlation"] for label in methods
    ])
    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    image = ax.imshow(best, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(component_names)), component_names, rotation=24, ha="right")
    ax.set_yticks(range(len(methods)), list(methods))
    for row in range(best.shape[0]):
        for column in range(best.shape[1]):
            ax.text(column, row, f"{best[row, column]:.3f}", ha="center", va="center",
                    color="white" if best[row, column] < 0.72 else "black")
    fig.colorbar(image, ax=ax, label="best absolute component correlation")
    ax.set_title("splitting versus conservative basin merging")
    fig.savefig(OUT / "merge_comparison.png", dpi=180)
    plt.close(fig)

    rows = max(len(value[0]) for value in methods.values()) + 2
    fig, axes = plt.subplots(rows, len(methods), figsize=(15, 1.45 * rows),
                             sharex=True, constrained_layout=True)
    for column, (label, (details, baseline, _)) in enumerate(methods.items()):
        axes[0, column].plot(t, signal, color="0.35", lw=0.5)
        axes[0, column].set_title(label)
        for row, detail in enumerate(details, start=1):
            axes[row, column].plot(t, detail, lw=0.55)
            axes[row, column].set_ylabel(f"D{row}", rotation=0, ha="right", fontsize=7)
        axes[len(details) + 1, column].plot(t, baseline, color="#15803d", lw=0.8)
        axes[len(details) + 1, column].plot(t, truth["slow drift"], "k--", lw=0.7)
        axes[len(details) + 1, column].set_ylabel("coarse", rotation=0, ha="right", fontsize=7)
        for row in range(len(details) + 2, rows):
            axes[row, column].axis("off")
    fig.savefig(OUT / "merge_decompositions.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
