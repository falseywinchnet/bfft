"""Exact value ledger for one 256 -> 5 -> 256 CONV* transport."""

from __future__ import annotations

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


OUT = ROOT / "experiments" / "out" / "conv_5x5_discard_ledger"


def signed_display(values: np.ndarray) -> np.ndarray:
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    return np.clip(0.5 + 0.5 * values / scale, 0.0, 1.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = synthetic_cartoon(256).astype(np.float64)
    coarse, _xy, _yx, _eta, retained = cycle_orders(source.astype(np.float32))
    discarded = source - retained
    saved = source - discarded
    closure = float(np.max(np.abs(saved - retained)))

    np.savez_compressed(
        OUT / "discard_ledger.npz",
        source=source.astype(np.float32),
        coarse_5x5=coarse.astype(np.float32),
        discarded=discarded.astype(np.float32),
        saved= saved.astype(np.float32),
        reconstruction=retained.astype(np.float32),
    )
    Image.fromarray(np.uint8(np.clip(saved, 0, 1) * 255)).save(OUT / "saved.png")
    Image.fromarray(np.uint8(signed_display(discarded) * 255)).save(OUT / "discarded.png")

    cell, header = 256, 34
    canvas = Image.new("RGB", (3 * cell, cell + header), "white")
    draw = ImageDraw.Draw(canvas)
    panels = (
        (np.clip(source, 0, 1), "original A"),
        (signed_display(discarded), "thrown away D = A - UDA"),
        (np.clip(saved, 0, 1), "saved A - D = UDA"),
    )
    for column, (array, title) in enumerate(panels):
        canvas.paste(Image.fromarray(np.uint8(array * 255)), (column * cell, header))
        draw.text((column * cell + 8, 9), title, fill="black")
    canvas.save(OUT / "discard_ledger.png")
    print(f"algebraic closure linf {closure:.17g}")


if __name__ == "__main__":
    main()
