"""One CONV flow-outline extraction followed by two image decompositions."""

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


OUT = ROOT / "experiments" / "out" / "conv_5x5_single_flow_branches"


def ranged(values: np.ndarray) -> np.ndarray:
    lo = np.min(values, axis=(0, 1), keepdims=True)
    hi = np.max(values, axis=(0, 1), keepdims=True)
    return np.clip((values - lo) / np.maximum(hi - lo, 1e-30), 0.0, 1.0)


def flow_image(source: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    _potential, _dc, flow, _closure = transport_explained_component(source, cycle)
    image = np.zeros_like(source)
    image[:-1, :-1] = np.linalg.norm(flow, axis=-1)
    return image


def branch(source: np.ndarray, kind: str, levels: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    state = source.copy()
    extracted: list[np.ndarray] = []
    states: list[np.ndarray] = [state.copy()]
    for _ in range(levels):
        cycle = cycle_orders(state.astype(np.float32))[-1]
        residual = state - cycle
        if kind == "coarse":
            extracted.append(cycle)
            state = residual
        elif kind == "residual":
            extracted.append(residual)
            state = cycle
        else:
            raise ValueError(kind)
        states.append(state.copy())
    return extracted, states


def render(
    source: np.ndarray,
    first_flow: np.ndarray,
    coarse: list[np.ndarray],
    residual: list[np.ndarray],
    path: Path,
) -> None:
    chosen = (0, 1, 2, 3, 4, 7, 11, 19, 31, 48)
    cell, header = 256, 34
    canvas = Image.new("RGB", (3 * cell, (len(chosen) + 1) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(np.uint8(np.clip(source, 0, 1) * 255)), (0, header))
    canvas.paste(Image.fromarray(np.uint8(ranged(first_flow) * 255)), (cell, header))
    draw.text((8, 9), "source A", fill="black")
    draw.text((cell + 8, 9), "the single flow-outline layer F0", fill="black")
    draw.text((2 * cell + 8, 9), "subsequent image-domain layers", fill="black")
    for row, index in enumerate(chosen, 1):
        y0 = row * (cell + header)
        left = ranged(coarse[index])
        right = ranged(residual[index])
        difference = ranged(coarse[index] - residual[index])
        for column, image in enumerate((left, right, difference)):
            canvas.paste(Image.fromarray(np.uint8(image * 255)),
                         (column * cell, y0 + header))
        draw.text((8, y0 + 9), f"level {index+1}: extract coarse B", fill="black")
        draw.text((cell + 8, y0 + 9), f"level {index+1}: extract residual A-B", fill="black")
        draw.text((2 * cell + 8, y0 + 9), "coarse layer - residual layer", fill="black")
    canvas.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = synthetic_cartoon(256).astype(np.float64)
    first_cycle = cycle_orders(source.astype(np.float32))[-1]
    first_flow = flow_image(source, first_cycle)
    byproduct = source - first_flow
    coarse_layers, coarse_states = branch(byproduct, "coarse", 49)
    residual_layers, residual_states = branch(byproduct, "residual", 49)
    np.savez_compressed(
        OUT / "single_flow_two_branches.npz",
        source=source.astype(np.float32),
        first_flow_outline=first_flow.astype(np.float32),
        initial_byproduct=byproduct.astype(np.float32),
        coarse_extracted=np.stack(coarse_layers).astype(np.float32),
        coarse_remainders=np.stack(coarse_states).astype(np.float32),
        residual_extracted=np.stack(residual_layers).astype(np.float32),
        residual_remainders=np.stack(residual_states).astype(np.float32),
    )
    metrics = {
        "first_flow_l2": float(np.linalg.norm(first_flow)),
        "coarse_layer_l2": [float(np.linalg.norm(x)) for x in coarse_layers],
        "residual_layer_l2": [float(np.linalg.norm(x)) for x in residual_layers],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    Image.fromarray(np.uint8(ranged(first_flow) * 255)).save(OUT / "first_flow_outline.png")
    render(source, first_flow, coarse_layers, residual_layers, OUT / "branch_comparison.png")


if __name__ == "__main__":
    main()
