"""Reproducible 1-D and 2-D probes, including a tapered hair-edge control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

try:
    from .transport_support import (
        TransportResolution,
        denoise_1d,
        support_density,
        transport_support_birth,
    )
except ImportError:
    from transport_support import (
        TransportResolution,
        denoise_1d,
        support_density,
        transport_support_birth,
    )


def one_dimensional_scene(size: int = 512, seed: int = 4701):
    x = np.linspace(0.0, 1.0, size, endpoint=False)
    truth = 0.28 + 0.13 * x + 0.17 * np.exp(-((x - 0.24) / 0.09) ** 2)
    truth += 0.29 / (1.0 + np.exp(-(x - 0.52) * size / 3.0))
    truth += 0.055 * np.sin(2.0 * np.pi * (16.0 * x + 18.0 * x * x)) * (x > 0.66)
    rng = np.random.default_rng(seed)
    observed = np.clip(truth + rng.uniform(-0.24, 0.24, size), 0.0, 1.0)
    return x, truth, observed


def hair_edge_scene(size: int = 128, seed: int = 719):
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    truth = 0.70 + 0.055 * xx + 0.035 * yy
    head = ((xx + 0.15) / 0.34) ** 2 + ((yy + 0.03) / 0.43) ** 2 < 1.0
    hair_cap = head & (yy < 0.02 + 0.25 * xx)
    truth[head] = 0.49 + 0.06 * xx[head]
    truth[hair_cap] = 0.10
    # Smoothly tapered strands are the falsification target: a useful support
    # law must not buy a clean sky by erasing their oblique continuation.
    for offset in (-0.10, -0.02, 0.07):
        center = 0.15 + 0.72 * (xx - offset)
        width = 0.012 + 0.012 * np.maximum(yy + 0.15, 0.0)
        strand = (xx > -0.10) & (xx < 0.36) & (np.abs(yy - center) < width)
        truth[strand] = 0.12
    truth = ndimage.gaussian_filter(truth, 0.45, mode="reflect")
    rng = np.random.default_rng(seed)
    observed = np.clip(truth + rng.uniform(-0.26, 0.26, truth.shape), 0.0, 1.0)
    return truth, observed


def mse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((np.asarray(estimate) - np.asarray(truth)) ** 2))


def edge_retention(estimate: np.ndarray, truth: np.ndarray) -> float:
    gy, gx = np.gradient(truth)
    magnitude = np.hypot(gx, gy)
    weight = magnitude / max(float(np.sum(magnitude)), np.finfo(float).tiny)
    ey, ex = np.gradient(estimate)
    projected = (ex * gx + ey * gy) / np.maximum(magnitude, np.finfo(float).tiny)
    reference = float(np.sum(weight * magnitude))
    return float(np.sum(weight * projected) / max(reference, np.finfo(float).tiny))


def run_probes(
    output: Path,
    resolution: TransportResolution = TransportResolution(),
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    x, truth_1d, observed_1d = one_dimensional_scene()
    result_1d, diagnostic_1d = denoise_1d(observed_1d, resolution)

    truth_2d, observed_2d = hair_edge_scene()
    provisional_2d = ndimage.gaussian_filter(observed_2d, 1.0, mode="reflect")
    result_2d, barrier, diagnostic_2d = transport_support_birth(
        observed_2d, provisional_2d, resolution)
    support, _ = support_density(observed_2d, resolution)

    report = {
        "one_dimensional": {
            "observed_mse": mse(observed_1d, truth_1d),
            "transport_mse": mse(result_1d, truth_1d),
            "diagnostics": diagnostic_1d,
        },
        "two_dimensional_hair_edge": {
            "observed_mse": mse(observed_2d, truth_2d),
            "provisional_mse": mse(provisional_2d, truth_2d),
            "transport_mse": mse(result_2d, truth_2d),
            "provisional_edge_retention": edge_retention(provisional_2d, truth_2d),
            "transport_edge_retention": edge_retention(result_2d, truth_2d),
            "diagnostics": diagnostic_2d,
        },
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    arrays = {
        "truth": truth_2d,
        "observed": observed_2d,
        "provisional": provisional_2d,
        "transport": result_2d,
        "support": support,
        "barrier": barrier,
    }
    for name, value in arrays.items():
        scaled = np.uint8(np.round(np.clip(value, 0.0, 1.0) * 255.0))
        Image.fromarray(scaled).save(output / f"hair_{name}.png")

    tile_size = truth_2d.shape[0]
    comparison = Image.new("L", (tile_size * len(arrays), tile_size + 22), 255)
    draw = ImageDraw.Draw(comparison)
    for index, (name, value) in enumerate(arrays.items()):
        tile = Image.fromarray(np.uint8(np.round(np.clip(value, 0.0, 1.0) * 255.0)))
        comparison.paste(tile, (index * tile_size, 22))
        draw.text((index * tile_size + 4, 4), name, fill=0)
    comparison.save(output / "hair_comparison.png")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
        axes[0].plot(x, truth_1d, color="#111827", linewidth=2, label="truth")
        axes[0].plot(x, observed_1d, color="#94a3b8", linewidth=0.7, label="observed")
        axes[0].plot(x, result_1d, color="#0f766e", linewidth=1.5, label="transport")
        axes[0].legend(frameon=False, ncol=3)
        axes[0].set_title("One law on a line: broad shape, jump, and chirped detail")
        montage = np.concatenate(tuple(arrays.values()), axis=1)
        axes[1].imshow(montage, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title("truth | observed | provisional | transport | support | barrier")
        axes[1].axis("off")
        figure.savefig(output / "probes.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--scale-samples", type=int, default=7)
    args = parser.parse_args()
    report = run_probes(
        args.out,
        TransportResolution(scale_samples=args.scale_samples),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
