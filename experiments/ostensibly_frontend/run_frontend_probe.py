"""Exact double-IRFFT speech texture and fast Meyer cartoon probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np

import bfft


def exact_double_irfft_texture(
    samples: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    crop_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(stft, signed_field, abs_low_rows)` using NumPy's defaults."""
    spectrum = librosa.stft(
        samples,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
    )
    recovered_frames = np.fft.irfft(spectrum, axis=0)
    signed_field = np.fft.irfft(recovered_frames, axis=0)
    if signed_field.shape[0] != 2 * (n_fft - 1):
        raise RuntimeError("unexpected default irfft output length")
    if not 0 < crop_rows <= signed_field.shape[0]:
        raise ValueError("crop_rows must lie inside the second-irfft field")
    return spectrum, signed_field, np.abs(signed_field[:crop_rows])


def scale_for_meyer(texture: np.ndarray) -> tuple[np.ndarray, float]:
    """Linearly scale a nonnegative texture to Meyer's `[0,255]` contract."""
    peak = float(np.max(texture))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(texture, dtype=np.float64), peak
    return np.ascontiguousarray(texture * (255.0 / peak), dtype=np.float64), peak


def relative_db(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(np.asarray(values, dtype=np.float64))
    peak = max(float(np.max(magnitude)), 1e-30)
    return 20.0 * np.log10(np.maximum(magnitude, peak * 1e-6) / peak)


def total_variation(values: np.ndarray) -> float:
    a = np.asarray(values, dtype=np.float64)
    return float(
        np.sum(np.abs(np.diff(a, axis=0)))
        + np.sum(np.abs(np.diff(a, axis=1)))
    )


def run_probe(
    wav_path: Path,
    out_dir: Path,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    crop_rows: int = 256,
    duration: float = 1.0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, sample_rate = librosa.load(
        wav_path, sr=None, mono=True, duration=duration
    )
    spectrum, signed_field, texture = exact_double_irfft_texture(
        samples,
        n_fft=n_fft,
        hop_length=hop_length,
        crop_rows=crop_rows,
    )
    meyer_input, linear_peak = scale_for_meyer(texture)

    started = time.perf_counter()
    cartoon, meyer_texture = bfft.meyer_split(meyer_input, threads=4)
    meyer_ms = 1000.0 * (time.perf_counter() - started)

    complement_error = float(
        np.max(np.abs(meyer_input - cartoon - meyer_texture))
    )
    input_energy = max(float(np.sum(meyer_input * meyer_input)), 1e-30)
    input_tv = max(total_variation(meyer_input), 1e-30)
    metrics = {
        "wav": str(wav_path),
        "sample_rate": int(sample_rate),
        "samples": int(samples.size),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "stft_shape": list(spectrum.shape),
        "second_irfft_shape": list(signed_field.shape),
        "retained_shape": list(texture.shape),
        "linear_abs_peak_before_scaling": linear_peak,
        "meyer_ms": meyer_ms,
        "exact_complement_max_error": complement_error,
        "cartoon_energy_fraction": float(
            np.sum(cartoon * cartoon) / input_energy
        ),
        "texture_energy_fraction": float(
            np.sum(meyer_texture * meyer_texture) / input_energy
        ),
        "cartoon_tv_fraction": total_variation(cartoon) / input_tv,
        "texture_tv_fraction": total_variation(meyer_texture) / input_tv,
    }

    np.savez_compressed(
        out_dir / "n2048_first_second_arrays.npz",
        abs_double_irfft=texture,
        meyer_input=meyer_input,
        cartoon=cartoon,
        meyer_texture=meyer_texture,
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    times = librosa.frames_to_time(
        np.arange(spectrum.shape[1]),
        sr=sample_rate,
        hop_length=hop_length,
    )
    extent = [float(times[0]), float(times[-1]), 0, crop_rows]
    panels = (
        ("abs(double irfft), first 256 rows", relative_db(meyer_input)),
        ("fast Meyer cartoon only", relative_db(cartoon)),
        ("removed Meyer texture", relative_db(meyer_texture)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    image = None
    for ax, (title, values) in zip(axes, panels):
        image = ax.imshow(
            np.flipud(values),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=-60.0,
            vmax=0.0,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("time (seconds)")
        ax.set_yticks([0, crop_rows])
        ax.set_yticklabels([str(crop_rows - 1), "0"])
        ax.set_ylabel("original row (low at top)")
    fig.suptitle(
        "Ostensibly oracle — N=2048, hop=512, exact 4094-row second irfft",
        fontsize=16,
    )
    fig.colorbar(image, ax=axes, label="relative magnitude (dB)", shrink=0.9)
    figure_path = out_dir / "n2048_meyer_cartoon.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    metrics["figure"] = str(figure_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_probe(args.wav, args.out, duration=args.duration)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
