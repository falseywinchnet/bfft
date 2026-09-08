"""Measure registration stability of the N=2048 speech-trace fingerprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.optimize import linear_sum_assignment

import bfft
from experiments.ostensibly_frontend.run_frontend_probe import (
    exact_double_irfft_texture,
    scale_for_meyer,
)
from experiments.ostensibly_frontend.trace_geometry import (
    TraceComponent,
    descriptor_distance,
    extract_trace_components,
)


def fingerprint(samples: np.ndarray) -> tuple[np.ndarray, tuple[TraceComponent, ...]]:
    _, _, texture = exact_double_irfft_texture(
        samples,
        n_fft=2048,
        hop_length=512,
        crop_rows=256,
    )
    meyer_input, _ = scale_for_meyer(texture)
    cartoon, _ = bfft.meyer_split(meyer_input, threads=4)
    _, _, components = extract_trace_components(cartoon)
    return cartoon, components


def match_components(
    reference: tuple[TraceComponent, ...],
    candidate: tuple[TraceComponent, ...],
) -> dict:
    if not reference or not candidate:
        return {
            "reference_components": len(reference),
            "candidate_components": len(candidate),
            "matches": 0,
            "median_distance": None,
            "maximum_distance": None,
        }
    costs = np.array(
        [[descriptor_distance(left, right) for right in candidate] for left in reference],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(costs)
    distances = costs[rows, columns]
    return {
        "reference_components": len(reference),
        "candidate_components": len(candidate),
        "matches": int(distances.size),
        "median_distance": float(np.median(distances)),
        "maximum_distance": float(np.max(distances)),
        "pairs": [
            {
                "reference": int(row + 1),
                "candidate": int(column + 1),
                "distance": float(costs[row, column]),
            }
            for row, column in zip(rows, columns)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    samples, sample_rate = librosa.load(
        args.wav, sr=None, mono=True, duration=1.0
    )
    rng = np.random.default_rng(20260825)
    signal_power = max(float(np.mean(samples * samples)), 1e-30)
    noise = rng.normal(size=samples.shape)
    noise *= np.sqrt(signal_power / (100.0 * float(np.mean(noise * noise))))
    phase_shift = 256
    variants = {
        "reference": samples,
        "gain_0.25": 0.25 * samples,
        "polarity_inverted": -samples,
        "window_phase_shift_256_samples": np.concatenate(
            [np.zeros(phase_shift, dtype=samples.dtype), samples[:-phase_shift]]
        ),
        "additive_white_noise_20db": samples + noise,
    }
    cartoons: dict[str, np.ndarray] = {}
    components: dict[str, tuple[TraceComponent, ...]] = {}
    for name, signal in variants.items():
        cartoons[name], components[name] = fingerprint(signal)

    reference = components["reference"]
    report = {
        "sample_rate": int(sample_rate),
        "frontend": {
            "n_fft": 2048,
            "hop_length": 512,
            "crop_rows": 256,
            "meyer": "fixed-cost jump-measure",
            "descriptor": "signed-distance/Fourier-circle plus spatial/aspect",
        },
        "variants": {
            name: {
                **match_components(reference, values),
                "cartoon_relative_l2": float(
                    np.linalg.norm(cartoons[name] - cartoons["reference"])
                    / max(np.linalg.norm(cartoons["reference"]), 1e-30)
                ),
            }
            for name, values in components.items()
        },
    }
    path = args.out / "registration_probe.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
