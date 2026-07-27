#!/usr/bin/env python3
"""Repeatable phase profiler for the canonical segmentation viewer.

Examples:

    python viewer/profile_segmenting_veroni.py --gallery astronaut
    python viewer/profile_segmenting_veroni.py photo.png --full-resolution
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from receiver_guided_graph import ReceiverGuidedVoronoi  # noqa: E402
from transport_voronoi import Config  # noqa: E402


def _load(path: str | None, gallery_key: str):
    if path is None:
        return gallery.load(gallery_key), f"gallery:{gallery_key}"
    from skimage.io import imread

    resolved = Path(path).expanduser().resolve()
    return imread(resolved), str(resolved)


def _peak_rss_gib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the BSD-derived Python docs report KiB.
    if sys.platform == "darwin":
        return value / (1024.0 ** 3)
    return value / (1024.0 ** 2)


def _time(label, operation):
    started = time.perf_counter()
    operation()
    elapsed = (time.perf_counter() - started) * 1000.0
    print(f"{label:28s} {elapsed:10.1f} ms")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Profile BFFT segmentation from initialization onward.")
    parser.add_argument("image", nargs="?", help="optional image file")
    parser.add_argument("--gallery", default="astronaut",
                        help="bundled gallery key when no file is supplied")
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--full-resolution", action="store_true")
    parser.add_argument("--passes", type=int, default=24)
    parser.add_argument("--flow-sweeps", type=int, default=64)
    parser.add_argument("--cells", type=int, default=180)
    parser.add_argument("--max-cells", type=int, default=1200)
    parser.add_argument("--split-batch", type=int, default=24)
    parser.add_argument(
        "--measure-decomposition", action="store_true",
        help="also time the deliberately deferred cartoon/texture metric")
    args = parser.parse_args()

    image, source = _load(args.image, args.gallery)
    max_side = 0 if args.full_resolution else args.max_side
    config = Config(
        max_side=max_side, passes=args.passes,
        flow_sweeps=args.flow_sweeps, initial_cells=args.cells,
        max_cells=args.max_cells, split_batch=args.split_batch,
        marked_cells=False, territory_count=1,
        allocation_mode="Expected affine gain")

    print(f"source                      {source}")
    started = time.perf_counter()
    model = ReceiverGuidedVoronoi(image, config)
    total = (time.perf_counter() - started) * 1000.0
    print(f"working image               {model.w} x {model.h} "
          f"({model.npix:,} pixels)")
    for phase, elapsed in model.init_timings.items():
        print(f"init: {phase:22s} {elapsed:10.1f} ms")
    print(f"initialization total         {total:10.1f} ms")
    print(f"initial PSNR                 {model.psnr:10.3f} dB")

    _time("repeat: geodesic assign", model._assign)
    _time("repeat: local affine fit", model._fit_models)
    _time("repeat: render + pressure", model._render)
    _time("round: choose new sites", model._subdivide)

    if args.measure_decomposition:
        _time("explicit decomp metrics",
              model.refresh_decomposition_metrics)
        print(f"cartoon / texture MSE        "
              f"{model.cartoon_decomp_mse:.4e} / "
              f"{model.texture_decomp_mse:.4e}")

    print(f"peak resident memory         {_peak_rss_gib():10.3f} GiB")
    print(f"packed edge graph            "
          f"{model._edge_cost_volume.nbytes / (1024 ** 2):10.1f} MiB "
          f"({model._edge_cost_volume.dtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
