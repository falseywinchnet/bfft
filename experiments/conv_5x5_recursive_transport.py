"""Recursively peel CONV* compression/inversion flow images from a source."""

from __future__ import annotations

import argparse
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


def normalized_flow_image(values: np.ndarray) -> np.ndarray:
    """Display a nonnegative RGB flow image on the original gray ground."""

    scale = max(float(np.max(values)), np.finfo(np.float64).tiny)
    return np.clip(0.5 + 0.5 * values / scale, 0.0, 1.0)


def render(
    initial: np.ndarray,
    inputs: list[np.ndarray],
    flow_images: list[np.ndarray],
    cycles: list[np.ndarray],
    records: list[dict[str, float | int]],
    path: Path,
) -> None:
    chosen = (0, 1, 2, 3, 4, 7, 11, 19, 31, 49)
    cell, header = 256, 34
    canvas = Image.new("RGB", (3 * cell, (len(chosen) + 1) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(np.uint8(np.clip(initial, 0, 1) * 255)), (0, header))
    draw.text((8, 9), "source A", fill="black")
    draw.text((cell + 8, 9), "5x5 round trip of current input", fill="black")
    draw.text((2 * cell + 8, 9), "extracted flow layer", fill="black")

    for row, index in enumerate(chosen, 1):
        y0 = row * (cell + header)
        input_image = inputs[index]
        # Each row is independently ranged only for inspection; raw states
        # and layer amplitudes remain untouched in the archive.
        lo = np.min(input_image, axis=(0, 1), keepdims=True)
        hi = np.max(input_image, axis=(0, 1), keepdims=True)
        shown_input = np.clip((input_image - lo) / np.maximum(hi - lo, 1e-30), 0, 1)
        shown_cycle = normalized_flow_image(np.abs(cycles[index]))
        shown_flow = normalized_flow_image(flow_images[index])
        canvas.paste(Image.fromarray(np.uint8(shown_input * 255)), (0, y0 + header))
        canvas.paste(Image.fromarray(np.uint8(shown_cycle * 255)), (cell, y0 + header))
        canvas.paste(Image.fromarray(np.uint8(shown_flow * 255)), (2 * cell, y0 + header))
        energy = records[index]["flow_image_l2"]
        draw.text((8, y0 + 9), f"level {index + 1} input", fill="black")
        draw.text((cell + 8, y0 + 9), "compressed then inverted", fill="black")
        draw.text((2 * cell + 8, y0 + 9),
                  f"flow image, L2={energy:.6g}", fill="black")
    canvas.save(path)


def run(out: Path, passes: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    initial = synthetic_cartoon(256).astype(np.float64)
    source = initial.copy()
    inputs: list[np.ndarray] = []
    flow_images: list[np.ndarray] = []
    vector_flows: list[np.ndarray] = []
    cycles: list[np.ndarray] = []
    coarse_states: list[np.ndarray] = []
    summaries: list[dict[str, float | int]] = []

    for index in range(passes):
        inputs.append(source.copy())
        coarse, _xy, _yx, _eta, cycle = cycle_orders(source.astype(np.float32))
        _explained, _dc, flow, closure = transport_explained_component(source, cycle)
        flow_image = np.zeros_like(source)
        flow_image[:-1, :-1] = np.linalg.norm(flow, axis=-1)
        flow_images.append(flow_image)
        vector_flows.append(flow)
        cycles.append(cycle)
        coarse_states.append(coarse)
        summaries.append({
            "level": index + 1,
            "input_l2": float(np.linalg.norm(source)),
            "cycle_l2": float(np.linalg.norm(cycle)),
            "flow_image_l2": float(np.linalg.norm(flow_image)),
            "flow_image_linf": float(np.max(flow_image)),
            "flow_mismatch_l2": float(np.linalg.norm(flow)),
            "path_integrability_linf": closure,
        })
        # Recurse on the unresolved cycle residual after removing the flow
        # layer.  The large-scale cycle image is not carried into the next
        # level, so it cannot regenerate the same outline merely by surviving.
        source = (source - cycle) - flow_image

    np.savez_compressed(
        out / "recursive_transport.npz",
        source_initial=initial.astype(np.float32),
        recursive_inputs=np.stack(inputs).astype(np.float32),
        flow_images=np.stack(flow_images).astype(np.float32),
        vector_flow_mismatches=np.stack(vector_flows).astype(np.float32),
        first_flow_image=flow_images[0].astype(np.float32),
        first_vector_flow_mismatch=vector_flows[0].astype(np.float32),
        final_remainder=source.astype(np.float32),
        coarse_states=np.stack(coarse_states).astype(np.float32),
    )
    np.save(out / "first_flow_mismatch.npy", flow_images[0].astype(np.float32))
    Image.fromarray(
        np.uint8(normalized_flow_image(flow_images[0]) * 255.0)
    ).save(out / "first_flow_mismatch.png")
    (out / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    render(initial, inputs, flow_images, cycles, summaries,
           out / "recursive_byproducts.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "out" / "conv_5x5_recursive_transport")
    args = parser.parse_args()
    run(args.out, args.passes)


if __name__ == "__main__":
    main()
