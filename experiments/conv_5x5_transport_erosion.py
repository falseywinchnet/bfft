"""Iterated 256 -> 5 -> 256 CONV* compression/inversion flow erosion.

At each pass, the five admitted Bernstein currents of the source and its
round-trip reconstruction are compared on every horizontal and vertical
cell.  Their difference is integrated exactly on the Cartesian cell complex.
Currents determine every nonconstant component; the channel-wise spatial DC
is retained because no transport flow can explain it.
"""

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

from experiments.conv_distilled_core import (  # noqa: E402
    nodal_current_geometry,
    reverse_conv_resize,
)
from experiments.convstar import (  # noqa: E402
    ordered_sign_ledger,
    project_signed_fibres,
    raw_current_jet_bank,
)
from standalone_conv_resize_demo.backend import (  # noqa: E402
    conv_basin_average,
    conv_resize,
    q1_order_blend,
)


def synthetic_cartoon(size: int = 256) -> np.ndarray:
    """Deterministic piecewise-smooth RGB cartoon with curved/oblique geometry."""

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    x = (xx + 0.5) / size
    y = (yy + 0.5) / size
    image = np.empty((size, size, 3), dtype=np.float64)
    image[...] = (0.82, 0.88, 0.91)
    image += (0.035 * x - 0.025 * y)[..., None]

    # A slanted ground plane and three constant/affine regions.
    ground = y > 0.73 - 0.18 * x
    image[ground] = np.stack(
        (0.23 + 0.10 * x[ground], 0.31 + 0.06 * x[ground],
         0.25 + 0.04 * y[ground]), axis=-1
    )
    body = ((x - 0.48) / 0.25) ** 2 + ((y - 0.57) / 0.31) ** 2 < 1.0
    image[body] = np.stack(
        (0.78 - 0.11 * y[body], 0.25 + 0.05 * x[body],
         0.13 + 0.04 * y[body]), axis=-1
    )
    head = (x - 0.48) ** 2 + (y - 0.25) ** 2 < 0.115 ** 2
    image[head] = (0.94, 0.73, 0.52)
    hat = ((x - 0.47) / 0.145) ** 2 + ((y - 0.19) / 0.072) ** 2 < 1.0
    image[hat] = (0.16, 0.20, 0.28)

    # Narrow diagonal and curved structures expose factor-order transport.
    line = np.abs(y - (0.84 - 0.63 * x)) < 0.010
    image[line] = (0.96, 0.91, 0.37)
    radius = np.sqrt((x - 0.76) ** 2 + (y - 0.43) ** 2)
    arc = (np.abs(radius - 0.165) < 0.010) & (x > 0.69)
    image[arc] = (0.08, 0.10, 0.12)
    eye = (x - 0.445) ** 2 + (y - 0.245) ** 2 < 0.010 ** 2
    image[eye] = (0.04, 0.04, 0.04)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def cycle_orders(source: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return 5x5 state, x-y/y-x synthesis, blend coordinate, and cycle."""

    coarse = conv_basin_average(source, (5, 5)).astype(np.float32)
    xy = np.asarray(conv_resize(coarse, source.shape[:2]), dtype=np.float64)
    yx = np.asarray(reverse_conv_resize(coarse, source.shape[:2]), dtype=np.float64)
    eta = nodal_current_geometry(coarse)[3].astype(np.float32)
    cycle = np.asarray(q1_order_blend(eta, xy, yx), dtype=np.float64)
    return coarse, xy, yx, eta, cycle


def admitted_axis_flow(values: np.ndarray, axis: int) -> np.ndarray:
    """Return the integrated canonical five-current fibres on one axis."""

    field = np.asarray(values, dtype=np.float64)
    moved = np.moveaxis(field, axis, 0)
    lines = moved.reshape(moved.shape[0], -1)
    raw, delta = raw_current_jet_bank(lines)
    signs = ordered_sign_ledger(raw, delta)
    current = project_signed_fibres(raw, signs, delta)
    fibres = current.reshape((moved.shape[0] - 1, 5) + moved.shape[1:])
    return np.moveaxis(np.sum(fibres, axis=1), 0, axis)


def transport_explained_component(
    source: np.ndarray, cycle: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate the compression/inversion current mismatch.

    Endpoint conservation makes the sum of each five-current fibre equal its
    cell secant.  Consequently the difference of the admitted source and
    reconstructed currents is an integrable discrete one-form.  A fixed
    horizontal path followed by a vertical path reconstructs its potential;
    equality with the reverse path is the exact integrability diagnostic.
    The additive spatial constant is not carried by a current and is therefore
    the sole unexplained component.
    """

    flow_x = admitted_axis_flow(source, 1) - admitted_axis_flow(cycle, 1)
    flow_y = admitted_axis_flow(source, 0) - admitted_axis_flow(cycle, 0)

    # x then y, anchored at the top-left vertex.
    path_xy = np.zeros_like(source, dtype=np.float64)
    path_xy[:, 1:] = np.cumsum(flow_x, axis=1)
    left_edge = np.zeros((source.shape[0], source.shape[2]), dtype=np.float64)
    left_edge[1:] = np.cumsum(flow_y[:, 0], axis=0)
    path_xy += left_edge[:, None, :]

    # y then x supplies a certificate that the measured current is integrable.
    path_yx = np.zeros_like(source, dtype=np.float64)
    path_yx[1:] = np.cumsum(flow_y, axis=0)
    top_edge = np.zeros((source.shape[1], source.shape[2]), dtype=np.float64)
    top_edge[1:] = np.cumsum(flow_x[0], axis=0)
    path_yx += top_edge[None, :, :]
    closure = float(np.max(np.abs(path_xy - path_yx)))

    explained = 0.5 * (path_xy + path_yx)
    explained -= np.mean(explained, axis=(0, 1), keepdims=True)
    error = source - cycle
    unexplained = error - explained
    flow = np.stack((flow_x[:-1], flow_y[:, :-1]), axis=-1)
    return explained, unexplained, flow, closure


def signed_rgb(values: np.ndarray, common_scale: float) -> np.ndarray:
    """White-zero signed display: negative cyan, positive magenta/red."""

    normalized = np.clip(values / common_scale, -1.0, 1.0)
    return np.clip(0.5 + 0.5 * normalized, 0.0, 1.0)


def panel_grid(
    source0: np.ndarray,
    records: list[dict[str, object]],
    output: Path,
) -> None:
    chosen = (0, 1, 4, 9, 19, 49)
    def local_signed(item: np.ndarray) -> np.ndarray:
        scale = max(float(np.max(np.abs(item))), np.finfo(np.float64).tiny)
        return signed_rgb(item, scale)

    def flow_gray(item: np.ndarray) -> np.ndarray:
        scale = max(float(np.max(item)), np.finfo(np.float64).tiny)
        gray = np.clip(item / scale, 0.0, 1.0)
        return np.repeat(gray[..., :1], 3, axis=2)
    cell = 256
    header = 34
    canvas = Image.new("RGB", (4 * cell, (len(chosen) + 1) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)

    initial = Image.fromarray(np.uint8(np.clip(source0, 0, 1) * 255.0))
    canvas.paste(initial, (0, header))
    draw.text((8, 9), "source A (pass 0)", fill="black")
    first = records[0]
    initial_panels = (
        (np.asarray(first["cycle"]), "target B: 5x5 -> 256"),
        (flow_gray(np.asarray(first["flow_display"])), "compression - inversion flow"),
        (local_signed(np.asarray(first["error"])), "cycle error A - B"),
    )
    for column, (array, title) in enumerate(initial_panels, 1):
        shown = array if column == 1 else np.asarray(array)
        canvas.paste(Image.fromarray(np.uint8(np.clip(shown, 0, 1) * 255.0)),
                     (column * cell, header))
        draw.text((column * cell + 8, 9), title, fill="black")

    for row, index in enumerate(chosen, 1):
        record = records[index]
        y0 = row * (cell + header)
        entries = (
            (local_signed(np.asarray(record["explained"])),
             f"byproduct {index + 1}: integrated flow difference"),
            (np.clip(np.asarray(record["updated"]), 0, 1),
             f"updated source after {index + 1}"),
            (local_signed(np.asarray(record["unexplained"])),
             "unexplained error"),
            (flow_gray(np.asarray(record["flow_display"])),
             "transport-flow mismatch"),
        )
        for column, (array, title) in enumerate(entries):
            canvas.paste(Image.fromarray(np.uint8(array * 255.0)),
                         (column * cell, y0 + header))
            draw.text((column * cell + 8, y0 + 9), title, fill="black")
    canvas.save(output)


def run(out: Path, passes: int = 50) -> None:
    out.mkdir(parents=True, exist_ok=True)
    initial = synthetic_cartoon(256).astype(np.float64)
    source = initial.copy()
    records: list[dict[str, object]] = []
    summary: list[dict[str, float | int]] = []
    cumulative = np.zeros_like(source)

    for iteration in range(passes):
        coarse, xy, yx, eta, cycle = cycle_orders(source.astype(np.float32))
        error = source - cycle
        explained, unexplained, flow, closure = transport_explained_component(
            source, cycle
        )
        flow_display = np.zeros_like(source)
        flow_display[:-1, :-1] = np.linalg.norm(flow, axis=-1)
        # The requested erosion: remove only what the factor-order difference
        # explains.  No clipping is applied to the mathematical state.
        source = source - explained
        cumulative += explained
        record = {
            "coarse": coarse,
            "eta": eta,
            "cycle": cycle,
            "error": error,
            "flow": flow,
            "flow_display": flow_display,
            "explained": explained,
            "unexplained": unexplained,
            "updated": source.copy(),
        }
        records.append(record)
        error_energy = float(np.vdot(error, error).real)
        explained_energy = float(np.vdot(explained, explained).real)
        summary.append({
            "pass": iteration + 1,
            "error_l2": float(np.sqrt(error_energy)),
            "flow_mismatch_l2": float(np.linalg.norm(flow)),
            "explained_l2": float(np.sqrt(explained_energy)),
            "explained_energy_fraction": (
                explained_energy / error_energy if error_energy > 0 else 0.0
            ),
            "unexplained_spatial_variation_linf": float(np.max(
                np.abs(unexplained - np.mean(unexplained, axis=(0, 1), keepdims=True))
            )),
            "path_integrability_linf": closure,
        })

    panel_grid(initial, records, out / "byproducts_montage.png")
    np.savez_compressed(
        out / "transport_byproducts.npz",
        source_initial=initial.astype(np.float32),
        byproducts=np.stack([r["explained"] for r in records]).astype(np.float32),
        updated_sources=np.stack([r["updated"] for r in records]).astype(np.float32),
        flow_mismatches=np.stack([r["flow"] for r in records]).astype(np.float32),
        cumulative_byproduct=cumulative.astype(np.float32),
    )
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "out" / "conv_5x5_transport_erosion")
    parser.add_argument("--passes", type=int, default=50)
    args = parser.parse_args()
    run(args.out, args.passes)


if __name__ == "__main__":
    main()
