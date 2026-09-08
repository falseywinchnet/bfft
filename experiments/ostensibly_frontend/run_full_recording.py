"""Run and render the Ostensibly front end over an entire WAV recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.ostensibly_frontend.full_recording import (
    build_full_recording,
    serializable_intervals,
)


def relative_db(values: np.ndarray) -> np.ndarray:
    peak = max(float(np.max(np.abs(values))), 1e-30)
    return 20.0 * np.log10(np.maximum(np.abs(values), peak * 1e-6) / peak)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detail-seconds", type=float, default=12.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    result = build_full_recording(args.wav)
    sample_rate = int(result["sample_rate"])
    hop = int(result["diagnostics"]["hop_length"])
    cartoon = result["cartoon"]
    frames = cartoon.shape[1]
    duration = float(result["diagnostics"]["duration_seconds"])
    times = np.arange(frames) * hop / sample_rate

    np.savez_compressed(
        args.out / "full_recording_frontend.npz",
        texture=result["texture"],
        cartoon=cartoon,
        meyer_texture=result["meyer_texture"],
        activity_score=result["activity_score"],
        active=result["active"],
    )
    report = {
        "wav": str(args.wav),
        "sample_rate": sample_rate,
        "spectrum_shape": list(result["spectrum_shape"]),
        "diagnostics": result["diagnostics"],
        "speech": serializable_intervals(result["speech"]),
        "phones": serializable_intervals(result["phones"]),
    }
    (args.out / "full_recording_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(20, 8), constrained_layout=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    axes[0].imshow(
        np.flipud(relative_db(cartoon)),
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, cartoon.shape[0]],
        cmap="magma",
        vmin=-60,
        vmax=0,
        interpolation="nearest",
    )
    axes[0].set_title("Full N=2048 Meyer speech cartoon (row zero at top)")
    axes[0].set_ylabel("IRODFT row")
    axes[1].plot(times, result["activity_score"], color="#2864dc", lw=1.0)
    axes[1].fill_between(
        times,
        0,
        result["activity_score"],
        where=result["active"],
        color="#ff7a18",
        alpha=0.55,
    )
    axes[1].set_xlim(0, duration)
    axes[1].set_title("Persistent-ridge voice activity")
    axes[1].set_xlabel("time (seconds)")
    axes[1].set_ylabel("log score")
    full_path = args.out / "full_recording_activity.png"
    fig.savefig(full_path, dpi=160)
    plt.close(fig)

    detail_frames = min(
        frames,
        int(round(args.detail_seconds * sample_rate / hop)),
    )
    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    ax.imshow(
        np.flipud(relative_db(cartoon[:, :detail_frames])),
        origin="lower",
        aspect="auto",
        extent=[0, detail_frames * hop / sample_rate, 0, cartoon.shape[0]],
        cmap="magma",
        vmin=-60,
        vmax=0,
        interpolation="nearest",
    )
    for phone in result["phones"]:
        if phone.seconds0 > args.detail_seconds:
            break
        ax.axvline(phone.seconds0, color="#43ffb4", lw=0.55, alpha=0.75)
    ax.set_title(
        f"First {args.detail_seconds:g} seconds with phone-boundary proposals"
    )
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("IRODFT row (low at top)")
    detail_path = args.out / "phone_boundary_detail.png"
    fig.savefig(detail_path, dpi=160)
    plt.close(fig)

    print(json.dumps({
        "diagnostics": result["diagnostics"],
        "full_figure": str(full_path),
        "detail_figure": str(detail_path),
        "report": str(args.out / "full_recording_report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
