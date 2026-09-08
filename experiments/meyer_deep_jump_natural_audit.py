#!/usr/bin/env python3
"""Natural-image audit of the scalar boundary term in the Meyer deep jump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from skimage import data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402
from experiments.meyer_eikonal_natural_probe import normalized  # noqa: E402
from experiments.meyer_transverse_route_research import (  # noqa: E402
    jump_texture_components,
    native_structural_gate,
    tangent_reservoir_route,
)

OUT = ROOT / "experiments" / "out" / "meyer_deep_jump_natural_audit"


def corrected_split(
    source: np.ndarray, *, virtual_passes: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    gate = native_structural_gate(source)
    seed, jump, _ = jump_texture_components(
        source, gate, lam=0.05, virtual_passes=virtual_passes
    )
    texture, route = tangent_reservoir_route(seed, gate, radius=40.0)
    return source - texture, texture, {
        "jump_rms": float(np.sqrt(np.mean(jump * jump))),
        "texture_rms": float(np.sqrt(np.mean(texture * texture))),
        "route_alpha": float(route["alpha"]),
    }


def gradient_magnitude(value: np.ndarray) -> np.ndarray:
    return np.hypot(
        np.roll(value, -1, axis=0) - value,
        np.roll(value, -1, axis=1) - value,
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left).ravel() - float(np.mean(left))
    b = np.asarray(right).ravel() - float(np.mean(right))
    return float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30))


def signed(value: np.ndarray, scale: float) -> Image.Image:
    shown = np.uint8(np.clip(127.5 + 127.5 * value / max(scale, 1e-30), 0, 255))
    return Image.fromarray(shown, mode="L").convert("RGB")


def plain(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(value, 0, 255)), mode="L").convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {
        "camera": normalized(data.camera()),
        "coins": normalized(data.coins()),
        "page": normalized(data.page()),
        "moon": normalized(data.moon()),
    }
    sources = {
        name: np.asarray(Image.fromarray(np.uint8(value)).resize(
            (args.size, args.size), Image.Resampling.LANCZOS
        ), dtype=np.float64)
        for name, value in sources.items()
    }
    titles = (
        "source", "old cartoon", "old texture", "no-boundary cartoon K8",
        "no-boundary texture K8", "no-boundary cartoon K12",
        "no-boundary texture K12",
    )
    cell, header, label = args.size, 30, 110
    canvas = Image.new(
        "RGB", (label + len(titles) * cell, header + len(sources) * cell), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for j, title in enumerate(titles):
        draw.text((label + j * cell + 3, 8), title, fill="black")
    report = {}
    arrays = {}
    for i, (name, source) in enumerate(sources.items()):
        plan = bfft.MeyerPlan(source.shape, lam=0.05, mu=40, passes=1, threads=1)
        start = time.perf_counter()
        old_u, old_v = plan.split_jump_measure(source)
        old_ms = 1000.0 * (time.perf_counter() - start)
        start = time.perf_counter()
        u8, v8, d8 = corrected_split(source, virtual_passes=8)
        k8_ms = 1000.0 * (time.perf_counter() - start)
        start = time.perf_counter()
        u12, v12, d12 = corrected_split(source, virtual_passes=12)
        k12_ms = 1000.0 * (time.perf_counter() - start)
        edge = gradient_magnitude(source)
        report[name] = {
            "old_milliseconds_python_call": old_ms,
            "research_k8_milliseconds": k8_ms,
            "research_k12_milliseconds": k12_ms,
            "old_texture_rms": float(np.sqrt(np.mean(old_v * old_v))),
            "old_absolute_texture_edge_correlation": correlation(np.abs(old_v), edge),
            "k8_absolute_texture_edge_correlation": correlation(np.abs(v8), edge),
            "k12_absolute_texture_edge_correlation": correlation(np.abs(v12), edge),
            "k8": d8,
            "k12": d12,
        }
        scale = max(float(np.percentile(np.abs(old_v), 99.5)), 1.0)
        panels = (
            plain(source), plain(old_u), signed(old_v, scale), plain(u8),
            signed(v8, scale), plain(u12), signed(v12, scale),
        )
        y0 = header + i * cell
        draw.text((4, y0 + 8), name, fill="black")
        for j, panel in enumerate(panels):
            canvas.paste(panel, (label + j * cell, y0))
        for key, value in (("source", source), ("old_cartoon", old_u),
                           ("old_texture", old_v), ("k8_cartoon", u8),
                           ("k8_texture", v8), ("k12_cartoon", u12),
                           ("k12_texture", v12)):
            arrays[f"{name}_{key}"] = value.astype(np.float32)
    canvas.save(args.out / "comparison.png")
    np.savez_compressed(args.out / "arrays.npz", **arrays)
    (args.out / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
