#!/usr/bin/env python3
"""Probe recovery from Fuji HDMI frames censored at video black.

This is an oracle-scored experiment, not a production enhancer.  The oracle is
used only to align the unavoidable output gain and to report metrics.  Every
candidate reconstruction is computed from the low-exposure burst alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle


def probe_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def read_luma(path: Path) -> np.ndarray:
    """Decode limited-range Y as full-range gray; legal black becomes zero."""
    width, height = probe_size(path)
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-pix_fmt", "gray", "-f", "rawvideo", "-",
        ],
        check=True,
        capture_output=True,
    )
    frame_bytes = width * height
    if len(result.stdout) % frame_bytes:
        raise RuntimeError("decoded byte count is not a whole number of frames")
    return np.frombuffer(result.stdout, np.uint8).reshape(
        -1, height, width).astype(np.float32) / 255.0


def gauge_match(candidate: np.ndarray, oracle: np.ndarray) -> np.ndarray:
    """Fit only global black and gain; do not import spatial oracle structure."""
    design = np.column_stack(
        [candidate.ravel(), np.ones(candidate.size, dtype=np.float32)])
    gain, offset = np.linalg.lstsq(
        design, oracle.ravel(), rcond=None)[0]
    return np.clip(gain * candidate + offset, 0.0, 1.0)


def score(candidate: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    matched = gauge_match(candidate, oracle)
    return {
        "psnr_db": float(peak_signal_noise_ratio(
            oracle, matched, data_range=1.0)),
        "ssim": float(structural_similarity(
            oracle, matched, data_range=1.0)),
    }


def distribution_calibration_diagnostic(
        samples: np.ndarray,
        oracle: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Test whether temporal variance resolves a mode-dependent mean transfer.

    This deliberately uses the oracle as a camera calibration target. Training
    uses one checkerboard parity and metrics include the held-out parity, so it
    cannot memorize an independent correction at every pixel. The diagnostic
    is not included in the low-only candidate ranking.
    """
    mean = samples.mean(axis=0)
    variance = samples.var(axis=0)
    deviation = np.sqrt(variance + 1.0e-9)
    feature_sets = {
        "mean_polynomial": [mean, mean * mean, mean ** 3,
                            np.sqrt(np.clip(mean, 0.0, None))],
        "mean_and_variance": [
            mean, mean * mean, mean ** 3, deviation, variance,
            mean * deviation, mean * variance,
        ],
    }
    yy, xx = np.indices(mean.shape)
    train_mask = ((xx + yy) & 1) == 0
    test_mask = ~train_mask
    train_indices = np.flatnonzero(train_mask)
    rng = np.random.default_rng(4)
    train_indices = rng.choice(
        train_indices, min(300_000, train_indices.size), replace=False)

    results: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, fields in feature_sets.items():
        design = np.column_stack(
            [np.ones(mean.size, dtype=np.float32)]
            + [field.ravel() for field in fields])
        coefficients = np.linalg.lstsq(
            design[train_indices],
            oracle.ravel()[train_indices],
            rcond=None,
        )[0]
        prediction = np.clip(
            (design @ coefficients).reshape(mean.shape), 0.0, 1.0)
        predictions[name] = prediction
        results[name] = {
            "full": score(prediction, oracle),
            "held_out_checkerboard": score(
                prediction[test_mask], oracle[test_mask]),
            "coefficients": [float(value) for value in coefficients],
        }
    return predictions["mean_and_variance"], results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--crop", default="150,0,1520,1080",
        help="x,y,width,height; excludes HDMI pillarbox and camera UI")
    args = parser.parse_args()

    reference = read_luma(args.reference)
    low = read_luma(args.low)
    oracle_full = reference.mean(axis=0)

    x, y, width, height = map(int, args.crop.split(","))
    region = np.s_[y:y + height, x:x + width]
    oracle = oracle_full[region]
    samples = low[:, region[0], region[1]]

    # The direct temporal mean is the sufficient statistic only if HDMI code
    # value remains proportional to photoelectrons.  The occupancy statistic
    # asks whether black clipping converted the stream into a one-bit detector.
    mean_excess = samples.mean(axis=0)
    occupancy = (samples > 0.0).mean(axis=0)
    occupancy_rate = -np.log(np.clip(1.0 - occupancy, 1.0e-6, 1.0))

    candidates: dict[str, np.ndarray] = {
        "single": samples[0],
        "mean": mean_excess,
        "median": np.median(samples, axis=0),
        "occupancy": occupancy,
        "bernoulli_rate": occupancy_rate,
    }
    for sigma in (0.6, 1.0, 1.6, 2.4):
        # Smooth accumulated evidence, not display output.  At this stage the
        # sweep establishes whether local support exists at any sane scale.
        candidates[f"event_blur_{sigma:g}"] = gaussian_filter(
            occupancy_rate, sigma=sigma)
    for weight in (0.002, 0.005, 0.01, 0.02):
        candidates[f"mean_tv_{weight:g}"] = denoise_tv_chambolle(
            mean_excess, weight=weight, channel_axis=None)

    metrics = {name: score(value, oracle)
               for name, value in candidates.items()}
    calibrated_distribution, calibration_diagnostic = (
        distribution_calibration_diagnostic(samples, oracle))
    dwell_counts = sorted(set(
        [1, 2, 4, 8, 16, 32, 64, int(samples.shape[0])]))
    dwell_counts = [count for count in dwell_counts
                    if count <= samples.shape[0]]
    dwell = {
        str(count): score(samples[:count].mean(axis=0), oracle)
        for count in dwell_counts
    }
    ranking = sorted(
        metrics, key=lambda name: metrics[name]["psnr_db"], reverse=True)

    best_name = ranking[0]
    best = gauge_match(candidates[best_name], oracle)
    mean_matched = gauge_match(mean_excess, oracle)
    event_matched = gauge_match(occupancy_rate, oracle)
    single_matched = gauge_match(samples[0], oracle)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ledger = {
        "reference": str(args.reference),
        "low": str(args.low),
        "reference_frames": int(reference.shape[0]),
        "low_frames": int(low.shape[0]),
        "crop_xywh": [x, y, width, height],
        "zero_fraction": float(np.mean(samples == 0.0)),
        "ever_above_black_fraction": float(np.mean(np.any(
            samples > 0.0, axis=0))),
        "metrics": metrics,
        "mean_dwell": dwell,
        "oracle_distribution_calibration": calibration_diagnostic,
        "ranking": ranking,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(ledger, indent=2) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [
        ("Bright oracle", oracle),
        ("One low frame", single_matched),
        (f"{samples.shape[0]}-frame mean", mean_matched),
        ("Oracle-calibrated mean + variance", calibrated_distribution),
        (f"Best: {best_name}", best),
        ("Absolute best error", np.abs(best - oracle)),
    ]
    for axis, (title, image) in zip(axes.flat, panels):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
