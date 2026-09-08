"""Recursive CONV* discard ledger: save UDA, recurse on A-UDA."""

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
)


OUT = ROOT / "experiments" / "out" / "conv_5x5_recursive_discard_ledger"


def ranged(values: np.ndarray) -> np.ndarray:
    lo = np.min(values, axis=(0, 1), keepdims=True)
    hi = np.max(values, axis=(0, 1), keepdims=True)
    return np.clip((values - lo) / np.maximum(hi - lo, 1e-30), 0.0, 1.0)


def render(
    source: np.ndarray,
    products: list[np.ndarray],
    residuals: list[np.ndarray],
    path: Path,
) -> None:
    chosen = (0, 1, 2, 3, 4, 7, 11, 19, 31, 49)
    cell, header = 256, 34
    canvas = Image.new("RGB", (3 * cell, (len(chosen) + 1) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(np.uint8(np.clip(source, 0, 1) * 255)), (0, header))
    draw.text((8, 9), "source A", fill="black")
    draw.text((cell + 8, 9), "saved product S_k = U D A_k", fill="black")
    draw.text((2 * cell + 8, 9), "thrown remainder A_(k+1)", fill="black")
    for row, index in enumerate(chosen, 1):
        y0 = row * (cell + header)
        product = ranged(products[index])
        remainder = ranged(residuals[index + 1])
        cumulative = ranged(np.sum(products[: index + 1], axis=0))
        for column, image in enumerate((product, remainder, cumulative)):
            canvas.paste(Image.fromarray(np.uint8(image * 255)),
                         (column * cell, y0 + header))
        draw.text((8, y0 + 9),
                  f"S_{index}, L2={np.linalg.norm(products[index]):.6g}", fill="black")
        draw.text((cell + 8, y0 + 9),
                  f"A_{index+1}, L2={np.linalg.norm(residuals[index+1]):.6g}", fill="black")
        draw.text((2 * cell + 8, y0 + 9), f"sum S_0...S_{index}", fill="black")
    canvas.save(path)


def main(levels: int = 50) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = synthetic_cartoon(256).astype(np.float64)
    residual = source.copy()
    products: list[np.ndarray] = []
    residuals: list[np.ndarray] = [residual.copy()]
    metrics: list[dict[str, float | int]] = []

    for level in range(levels):
        saved = np.asarray(cycle_orders(residual.astype(np.float32))[-1], dtype=np.float64)
        residual = residual - saved
        products.append(saved)
        residuals.append(residual.copy())
        reconstruction = np.sum(products, axis=0) + residual
        metrics.append({
            "level": level + 1,
            "saved_l2": float(np.linalg.norm(saved)),
            "remainder_l2": float(np.linalg.norm(residual)),
            "telescoping_closure_linf": float(np.max(np.abs(source - reconstruction))),
        })

    np.savez_compressed(
        OUT / "recursive_discard_ledger.npz",
        source=source.astype(np.float32),
        saved_products=np.stack(products).astype(np.float32),
        remainders=np.stack(residuals).astype(np.float32),
        final_reconstruction=(np.sum(products, axis=0) + residual).astype(np.float32),
    )
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    render(source, products, residuals, OUT / "recursive_discard_ledger.png")


if __name__ == "__main__":
    main()
