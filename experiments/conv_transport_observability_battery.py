"""Synthetic battery for CONV transport-visible/null decomposition."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conv_5x5_transport_erosion import (  # noqa: E402
    cycle_orders,
    synthetic_cartoon,
    transport_explained_component,
)


OUT = ROOT / "experiments" / "out" / "conv_transport_observability_battery"


def rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(gray, dtype=np.float64)[..., None], 3, axis=2)


def sources(size: int = 256) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    x, y = (xx + 0.5) / size, (yy + 0.5) / size
    rng = np.random.default_rng(2701)

    white = np.clip(0.5 + 0.22 * rng.standard_normal((size, size)), 0, 1)
    localized = np.full((size, size), 0.5)
    mask = ((x - 0.52) / 0.34) ** 2 + ((y - 0.50) / 0.27) ** 2 < 1
    localized[mask] += 0.38 * np.sin(2 * np.pi * (19 * x[mask] + 7 * y[mask]))
    chirp = 0.5 + 0.45 * np.sin(2 * np.pi * (2.0 * x + 24.0 * x * x))
    checker = 0.15 + 0.7 * (((np.floor(18 * x) + np.floor(18 * y)) % 2) > 0)
    impulses = np.full((size, size), 0.2)
    impulses[30:33, 25:220] = 0.9
    impulses[40:220, 175:178] = 0.75
    impulses[np.abs(y - (0.88 - 0.71 * x)) < 0.006] = 1.0
    for cx, cy in ((0.22, 0.62), (0.70, 0.28), (0.79, 0.76)):
        impulses[(x-cx)**2 + (y-cy)**2 < 0.012**2] = 1.0
    smooth = 0.15 + 0.62 * x + 0.15 * np.sin(np.pi * y)
    interface = np.where(y < 0.77 - 0.58 * x, 0.88, 0.12)
    interface += 0.10 * np.sin(2 * np.pi * 3 * (x + y))

    color_osc = np.empty((size, size, 3), dtype=np.float64)
    color_osc[..., 0] = 0.5 + 0.38 * np.sin(2*np.pi*14*x)
    color_osc[..., 1] = 0.5 + 0.38 * np.sin(2*np.pi*11*y)
    color_osc[..., 2] = 0.5 + 0.38 * np.sin(2*np.pi*9*(x+y))
    return {
        "cartoon geometry": synthetic_cartoon(size).astype(np.float64),
        "white noise": rgb(white),
        "localized oscillation": rgb(np.clip(localized, 0, 1)),
        "quadratic chirp": rgb(np.clip(chirp, 0, 1)),
        "checkerboard": rgb(checker),
        "thin lines and impulses": rgb(impulses),
        "smooth field": rgb(np.clip(smooth, 0, 1)),
        "diagonal interface": rgb(np.clip(interface, 0, 1)),
        "multichannel oscillation": np.clip(color_osc, 0, 1),
    }


def decompose(source: np.ndarray, levels: int) -> dict[str, np.ndarray | float]:
    residual = source.copy()
    visible = np.zeros_like(source)
    for _ in range(levels):
        layer = np.asarray(cycle_orders(residual.astype(np.float32))[-1], dtype=np.float64)
        visible += layer
        residual -= layer
    null_cycle = np.asarray(cycle_orders(residual.astype(np.float32))[-1], dtype=np.float64)
    _potential, _dc, flow, closure = transport_explained_component(residual, null_cycle)
    geometry = np.zeros_like(source)
    geometry[:-1, :-1] = np.linalg.norm(flow, axis=-1)
    return {
        "visible": visible,
        "detail": residual,
        "geometry": geometry,
        "null_cycle": null_cycle,
        "closure": closure,
    }


def ranged(values: np.ndarray) -> np.ndarray:
    lo = np.min(values, axis=(0, 1), keepdims=True)
    hi = np.max(values, axis=(0, 1), keepdims=True)
    return np.clip((values-lo)/np.maximum(hi-lo, 1e-30), 0, 1)


def signed(values: np.ndarray) -> np.ndarray:
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    return np.clip(0.5 + 0.5 * values / scale, 0, 1)


def main(levels: int = 50) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = sources()
    results = {name: decompose(source, levels) for name, source in cases.items()}
    metrics: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {}

    cell, label_width, header = 192, 150, 34
    canvas = Image.new("RGB", (label_width + 5*cell, header + len(cases)*cell), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("source", "visible G", "null detail H", "geometry trace", "P(H)")):
        draw.text((label_width + column*cell + 8, 9), title, fill="black")

    for row, (name, source) in enumerate(cases.items()):
        result = results[name]
        visible = np.asarray(result["visible"])
        detail = np.asarray(result["detail"])
        geometry = np.asarray(result["geometry"])
        null_cycle = np.asarray(result["null_cycle"])
        ratio = float(np.linalg.norm(null_cycle)/max(np.linalg.norm(detail), 1e-30))
        source_ac = source - np.mean(source, axis=(0, 1), keepdims=True)
        visible_ac = visible - np.mean(visible, axis=(0, 1), keepdims=True)
        detail_ac = detail - np.mean(detail, axis=(0, 1), keepdims=True)
        source_ac_l2 = float(np.linalg.norm(source_ac))
        metrics[name] = {
            "source_l2": float(np.linalg.norm(source)),
            "visible_l2": float(np.linalg.norm(visible)),
            "null_detail_l2": float(np.linalg.norm(detail)),
            "null_retransport_l2": float(np.linalg.norm(null_cycle)),
            "null_ratio": ratio,
            "source_ac_l2": source_ac_l2,
            "visible_ac_l2": float(np.linalg.norm(visible_ac)),
            "null_detail_ac_l2": float(np.linalg.norm(detail_ac)),
            "null_to_source_ac_ratio": float(
                np.linalg.norm(detail_ac) / max(source_ac_l2, 1e-30)
            ),
            "additive_closure_linf": float(np.max(np.abs(source-visible-detail))),
            "geometry_path_closure_linf": float(result["closure"]),
        }
        key = name.replace(" ", "_")
        arrays[f"{key}_source"] = source.astype(np.float32)
        arrays[f"{key}_visible"] = visible.astype(np.float32)
        arrays[f"{key}_detail"] = detail.astype(np.float32)
        arrays[f"{key}_geometry"] = geometry.astype(np.float32)
        y0 = header + row*cell
        draw.text((6, y0+12), name, fill="black")
        draw.text((6, y0+30), f"||P(H)||/||H||", fill="black")
        draw.text((6, y0+47), f"{ratio:.3e}", fill="black")
        panels = (
            np.clip(source, 0, 1), ranged(visible), signed(detail),
            signed(geometry), signed(null_cycle),
        )
        for column, panel in enumerate(panels):
            image = Image.fromarray(np.uint8(panel*255)).resize((cell, cell), Image.Resampling.NEAREST)
            canvas.paste(image, (label_width + column*cell, y0))
    canvas.save(OUT / "observability_battery.png")
    np.savez_compressed(OUT / "observability_battery.npz", **arrays)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
