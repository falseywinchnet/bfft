#!/usr/bin/env python3
"""Score retained Köhler scene-1 pair checkpoints against its web reference.

This is deliberately not the official Köhler score.  The official protocol
compares against roughly 200 trajectory samples; the public page embeds one
representative compressed JPEG for each of its four distinct base images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from denoiser.run_2d_denoiser_battery import metrics


def _load_luminance(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return rgb @ np.asarray((0.2126, 0.7152, 0.0722))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score(image: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    measured = metrics(
        image[32:-32, 32:-32], reference[32:-32, 32:-32])
    return {
        "web_reference_psnr": float(-10.0 * math.log10(max(
            float(measured["mse"]), np.finfo(float).tiny))),
        "web_reference_ssim": float(measured["ssim"]),
    }


def run(data: Path, results: Path, output: Path) -> dict[str, object]:
    paths = {
        "blur_1": data / "blurry_1_1.jpg",
        "blur_2": data / "blurry_1_2.jpg",
        "center_only": results / "koehler_pair_center_only" / "deblurred.png",
        "relative_mixing": (
            results / "personal_deblurrer_koehler_relative_mixing"
            / "deblurred.png"),
    }
    reference = _load_luminance(data / "ground_truth_1.jpg")
    loaded = {name: _load_luminance(path) for name, path in paths.items()}
    loaded["unregistered_average"] = 0.5 * (
        loaded["blur_1"] + loaded["blur_2"])
    scores = {name: _score(image, reference) for name, image in loaded.items()}
    audits = {}
    for name, directory in (
        ("center_only", "koehler_pair_center_only"),
        ("relative_mixing", "personal_deblurrer_koehler_relative_mixing"),
    ):
        audit = json.loads((results / directory / "evaluation.json").read_text())
        audits[name] = {
            key: audit[key] for key in (
                "all_sources_unchanged",
                "forward_closure_rms",
                "forward_closure_over_pair_disagreement",
                "edge_concentration_ratio",
                "local_observation_envelope_excursion",
                "fourier_circle_amplification",
                "uncertainty_rms",
            )
        }
    report = {
        "experiment": "koehler_scene1_two_capture_checkpoint_v1",
        "status": "failed_real_capture_acceptance_gate",
        "reference_scope": (
            "one compressed scene-1 web JPEG; not the official roughly-200-"
            "sample Koehler trajectory evaluation"),
        "crop_pixels": 32,
        "scores": scores,
        "audits": audits,
        "source_sha256": {
            path.name: _sha256(path)
            for path in sorted(data.glob("*.jpg"))
        },
        "decision": (
            "relative mixing reduces ringing and slightly improves the center-"
            "only result, but remains below the immutable observation average; "
            "do not promote as a solved real-capture method"),
        "next_identifiability_step": (
            "multi-observation consensus or independent trajectory/inertial "
            "evidence is required to constrain blur common to both captures"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("personal_deblurrer/real_capture_data/koehler_scene1_web_jpeg"))
    parser.add_argument(
        "--results", type=Path,
        default=Path("personal_deblurrer/real_capture_results"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/real_capture_results/koehler_checkpoint.json"))
    args = parser.parse_args()
    report = run(args.data, args.results, args.out)
    for name, score in report["scores"].items():
        print(f"{name:24s} {score['web_reference_psnr']:.3f} dB  "
              f"SSIM {score['web_reference_ssim']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
