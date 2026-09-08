"""CONV transport-visible / transport-null decomposition and geometry trace."""

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


OUT = ROOT / "experiments" / "out" / "conv_transport_observability"


def signed(values: np.ndarray) -> np.ndarray:
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    return np.clip(0.5 + 0.5 * values / scale, 0.0, 1.0)


def range_rgb(values: np.ndarray) -> np.ndarray:
    lo = np.min(values, axis=(0, 1), keepdims=True)
    hi = np.max(values, axis=(0, 1), keepdims=True)
    return np.clip((values - lo) / np.maximum(hi - lo, 1e-30), 0.0, 1.0)


def main(levels: int = 50) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = synthetic_cartoon(256).astype(np.float64)
    residual = source.copy()
    visible_layers: list[np.ndarray] = []
    for _ in range(levels):
        layer = np.asarray(cycle_orders(residual.astype(np.float32))[-1], dtype=np.float64)
        visible_layers.append(layer)
        residual = residual - layer

    visible = np.sum(visible_layers, axis=0)
    null_detail = residual
    null_cycle = np.asarray(cycle_orders(null_detail.astype(np.float32))[-1], dtype=np.float64)
    _potential, _dc, flow, closure = transport_explained_component(
        null_detail, null_cycle
    )
    geometry = np.zeros_like(source)
    geometry[:-1, :-1] = np.linalg.norm(flow, axis=-1)
    reconstruction = visible + null_detail

    metrics = {
        "levels": levels,
        "source_l2": float(np.linalg.norm(source)),
        "transport_visible_l2": float(np.linalg.norm(visible)),
        "transport_null_detail_l2": float(np.linalg.norm(null_detail)),
        "null_detail_retransport_l2": float(np.linalg.norm(null_cycle)),
        "null_ratio": float(np.linalg.norm(null_cycle) / np.linalg.norm(null_detail)),
        "additive_closure_linf": float(np.max(np.abs(source - reconstruction))),
        "geometry_path_closure_linf": closure,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        OUT / "transport_observability.npz",
        source=source.astype(np.float32),
        visible_layers=np.stack(visible_layers).astype(np.float32),
        transport_visible=visible.astype(np.float32),
        transport_null_detail=null_detail.astype(np.float32),
        null_retransport=null_cycle.astype(np.float32),
        detail_geometry_current=geometry.astype(np.float32),
    )

    cell, header = 256, 34
    canvas = Image.new("RGB", (5 * cell, cell + header), "white")
    draw = ImageDraw.Draw(canvas)
    panels = (
        (np.clip(source, 0, 1), "source A"),
        (range_rgb(visible), "transport-visible structure G"),
        (signed(null_detail), "transport-null detail H"),
        (signed(geometry), "geometry trace ||J(H)-J(PH)||"),
        (signed(null_cycle), "retransport P(H), nearly zero"),
    )
    for column, (image, title) in enumerate(panels):
        canvas.paste(Image.fromarray(np.uint8(image * 255)), (column * cell, header))
        draw.text((column * cell + 8, 9), title, fill="black")
    canvas.save(OUT / "transport_observability.png")


if __name__ == "__main__":
    main()
