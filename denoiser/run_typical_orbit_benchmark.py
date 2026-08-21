"""Falsification benchmark for the first post-FMMT typical-orbit checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal import savgol_filter

from .fmmt_certified import denoise_fmmt
from .sample_series import corrupt
from .typical_orbit_set import (
    LocalOrbitResolution,
    TypicalOrbitResolution,
    denoise_local_orbit_survival,
    denoise_typical_orbit_set,
    fmmt_feasible_afterpass,
    fmmt_local_feasible_afterpass,
    fmmt_local_redundancy_afterpass,
    fmmt_local_coherence_veto,
)


CASES = (
    ("clean", "none", 0.0, 0.0),
    ("salt_pepper_25", "salt and pepper", 0.0, 0.25),
    ("uniform_24", "uniform additive", 0.24, 0.0),
    ("mixed_08", "mixed replacement + uniform", 0.24, 0.08),
    ("mixed_25", "mixed replacement + uniform", 0.24, 0.25),
)

ROIS = {
    "hair": (0.08, 0.36, 0.28, 0.58),
    "camera_hands": (0.27, 0.58, 0.39, 0.72),
    "tripod": (0.43, 1.00, 0.43, 0.88),
}


def _camera(size: int) -> np.ndarray:
    source_path = (
        Path(__file__).resolve().parent.parent
        / "svg_converter" / "examples" / "input" / "cameraman_source.png"
    )
    source = Image.open(source_path).convert("L").resize(
        (int(size), int(size)), Image.Resampling.LANCZOS)
    return np.asarray(source, np.float64) / 255.0


def _roi(image: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    y0, y1, x0, x1 = bounds
    height, width = image.shape
    return image[
        int(round(y0 * height)):int(round(y1 * height)),
        int(round(x0 * width)):int(round(x1 * width)),
    ]


def _edge_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    truth_y = ndimage.sobel(truth, axis=0, mode="reflect") / 8.0
    truth_x = ndimage.sobel(truth, axis=1, mode="reflect") / 8.0
    estimate_y = ndimage.sobel(estimate, axis=0, mode="reflect") / 8.0
    estimate_x = ndimage.sobel(estimate, axis=1, mode="reflect") / 8.0
    magnitude = np.hypot(truth_x, truth_y)
    estimate_magnitude = np.hypot(estimate_x, estimate_y)
    threshold = float(np.quantile(magnitude, 0.8))
    strong = magnitude >= threshold
    denominator = max(float(np.sum(magnitude[strong])), np.finfo(float).tiny)
    projected = (
        estimate_x[strong] * truth_x[strong]
        + estimate_y[strong] * truth_y[strong]
    ) / np.maximum(magnitude[strong], np.finfo(float).tiny)
    correlation = np.corrcoef(
        np.concatenate((estimate_x[strong], estimate_y[strong])),
        np.concatenate((truth_x[strong], truth_y[strong])),
    )[0, 1]
    return {
        "strong_edge_magnitude_ratio": float(
            np.sum(estimate_magnitude[strong]) / denominator),
        "strong_edge_projected_retention": float(
            np.sum(projected) / denominator),
        "strong_edge_vector_correlation": float(correlation),
        "gradient_mse": float(np.mean(
            (estimate_x - truth_x) ** 2 + (estimate_y - truth_y) ** 2)),
    }


def _metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "mse": float(np.mean((estimate - truth) ** 2)),
        "ssim": _ssim(estimate, truth),
        **_edge_metrics(estimate, truth),
        "rois": {},
    }
    for name, bounds in ROIS.items():
        estimate_roi = _roi(estimate, bounds)
        truth_roi = _roi(truth, bounds)
        result["rois"][name] = {
            "mse": float(np.mean((estimate_roi - truth_roi) ** 2)),
            **_edge_metrics(estimate_roi, truth_roi),
        }
    return result


def _ssim(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Windowed grayscale SSIM without a scikit-image runtime dependency."""
    sigma = 1.5
    mean_e = ndimage.gaussian_filter(estimate, sigma, mode="reflect")
    mean_t = ndimage.gaussian_filter(truth, sigma, mode="reflect")
    variance_e = np.maximum(
        ndimage.gaussian_filter(estimate * estimate, sigma, mode="reflect")
        - mean_e * mean_e,
        0.0,
    )
    variance_t = np.maximum(
        ndimage.gaussian_filter(truth * truth, sigma, mode="reflect")
        - mean_t * mean_t,
        0.0,
    )
    covariance = (
        ndimage.gaussian_filter(estimate * truth, sigma, mode="reflect")
        - mean_e * mean_t
    )
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (
        (2.0 * mean_e * mean_t + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_e * mean_e + mean_t * mean_t + c1)
            * (variance_e + variance_t + c2)
        )
    )
    return float(np.mean(ssim))


def _save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.uint8(np.round(np.clip(image, 0.0, 1.0) * 255.0))).save(path)


def _save_panel(path: Path, images: dict[str, np.ndarray]) -> None:
    names = tuple(images)
    height, width = next(iter(images.values())).shape
    label_height = 22
    panel = Image.new("L", (width * len(names), height + label_height), 255)
    draw = ImageDraw.Draw(panel)
    for index, name in enumerate(names):
        tile = Image.fromarray(np.uint8(np.round(np.clip(images[name], 0.0, 1.0) * 255.0)))
        panel.paste(tile, (index * width, label_height))
        draw.text((index * width + 3, 4), name, fill=0)
    panel.save(path)


def run_benchmark(size: int, seed: int, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    truth = _camera(size)
    resolution = TypicalOrbitResolution()
    local_resolution = LocalOrbitResolution()
    report: dict[str, object] = {
        "status": "falsifiable first checkpoint; not a promoted denoiser",
        "size": int(size),
        "seed": int(seed),
        "resolution": {
            "ring_radii": list(resolution.ring_radii),
            "candidate_count": resolution.candidate_count,
        },
        "cases": {},
    }
    for case_name, corruption_name, amount, density in CASES:
        observed = corrupt(
            truth, corruption_name, amount=amount, density=density, seed=seed)
        methods = {
            "observed": lambda: (observed.copy(), {}),
            "mean3": lambda: (
                ndimage.uniform_filter(observed, size=3, mode="reflect"), {}),
            "median3": lambda: (
                ndimage.median_filter(observed, size=3, mode="reflect"), {}),
            "savgol5": lambda: (
                savgol_filter(
                    savgol_filter(observed, 5, 2, axis=0, mode="mirror"),
                    5, 2, axis=1, mode="mirror"),
                {},
            ),
            "rejected_fmmt": lambda: denoise_fmmt(observed),
            "typical_orbit": lambda: denoise_typical_orbit_set(
                observed, resolution),
            "typical_orbit_fmmt_afterpass": lambda: fmmt_feasible_afterpass(
                observed, resolution),
            "local_orbit_survival": lambda: denoise_local_orbit_survival(
                observed, local_resolution),
            "local_orbit_fmmt_afterpass": lambda: fmmt_local_feasible_afterpass(
                observed, local_resolution),
            "local_orbit_redundant_fmmt": lambda: fmmt_local_redundancy_afterpass(
                observed, local_resolution),
            "local_orbit_coherence_veto": lambda: fmmt_local_coherence_veto(
                observed, local_resolution),
        }
        images: dict[str, np.ndarray] = {"truth": truth, "observed": observed}
        case_record: dict[str, object] = {
            "corruption": corruption_name,
            "amount": float(amount),
            "density": float(density),
            "methods": {},
        }
        for method_name, run in methods.items():
            started = time.perf_counter()
            estimate, diagnostics = run()
            elapsed = time.perf_counter() - started
            images[method_name] = estimate
            case_record["methods"][method_name] = {
                "seconds": float(elapsed),
                "metrics": _metrics(estimate, truth),
                "diagnostics": diagnostics,
            }
            _save_gray(output / f"{case_name}_{method_name}.png", estimate)
        _save_panel(output / f"{case_name}_panel.png", images)
        focus = {
            "truth": images["truth"],
            "observed": images["observed"],
            "FMMT": images["rejected_fmmt"],
            "structure": images["local_orbit_survival"],
            "FMMT+orbit": images["local_orbit_coherence_veto"],
        }
        _save_panel(output / f"{case_name}_focus.png", focus)
        tripod_focus = {
            name: _roi(image, ROIS["tripod"])
            for name, image in focus.items()
        }
        tripod_path = output / f"{case_name}_tripod_focus.png"
        _save_panel(tripod_path, tripod_focus)
        tripod_panel = Image.open(tripod_path)
        tripod_panel.resize(
            (tripod_panel.width * 3, tripod_panel.height * 3),
            Image.Resampling.NEAREST,
        ).save(tripod_path)
        report["cases"][case_name] = case_record
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(args.size, args.seed, args.out)
    summary = {}
    for case, record in report["cases"].items():
        summary[case] = {
            method: {
                "mse": values["metrics"]["mse"],
                "ssim": values["metrics"]["ssim"],
                "edge": values["metrics"]["strong_edge_projected_retention"],
                "tripod_edge": values["metrics"]["rois"]["tripod"]["strong_edge_projected_retention"],
                "seconds": values["seconds"],
            }
            for method, values in record["methods"].items()
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
