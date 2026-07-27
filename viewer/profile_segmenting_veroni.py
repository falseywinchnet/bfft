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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from port_needed.pipeline import (  # noqa: E402
    SegmentingConfig,
    build_segmenting_representation,
)


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
    parser.add_argument("--allocation-side", type=int, default=512)
    parser.add_argument("--safety-cells", type=int, default=32768)
    parser.add_argument("--characteristic-passes", type=int, default=1)
    parser.add_argument("--interface-safety", type=float, default=0.5)
    parser.add_argument("--germ-shell-radius", type=float, default=3.0)
    args = parser.parse_args()

    image, source = _load(args.image, args.gallery)
    max_side = 0 if args.full_resolution else args.max_side
    from transport_voronoi import _fit_rgb

    rgb = _fit_rgb(image, max_side)
    config = SegmentingConfig(
        allocation_method="causal_density",
        allocation_max_side=args.allocation_side,
        tgfd_sweeps=args.passes,
        flow_sweeps=args.flow_sweeps,
        safety_cells=args.safety_cells,
        characteristic_passes=args.characteristic_passes,
        characteristic_trust_fraction=args.interface_safety,
        characteristic_core_radius=args.germ_shell_radius,
    )

    print(f"source                      {source}")
    started = time.perf_counter()
    result = build_segmenting_representation(rgb, config)
    total = (time.perf_counter() - started) * 1000.0
    record = result["record"]
    timing = result["timing"]
    print(f"working image               {rgb.shape[1]} x {rgb.shape[0]} "
          f"({rgb.shape[0] * rgb.shape[1]:,} pixels)")
    print(f"cells                       {len(result['centers']):10,d}")
    print(f"Meyer geometry              {timing['geometry_ms']:10.1f} ms")
    print(f"population/front/step       {timing['allocation_ms']:10.1f} ms")
    print(f"fit/ridge/score             {timing['fit_ms']:10.1f} ms")
    print(f"wall total                  {total:10.1f} ms")
    print(f"PSNR                        {record['psnr']:10.3f} dB")
    print(f"cartoon / texture MSE       "
          f"{record['cartoon_mse']:.4e} / {record['texture_mse']:.4e}")
    characteristic = result["characteristic"]
    if characteristic and characteristic["trace"]:
        for item in characteristic["trace"]:
            print(
                f"front pass {item['iteration']:2d}             "
                f"scale {item['accepted_scale']:.3f}, "
                f"trials {item['trials']}, "
                f"action {100.0 * item['relative_action_change']:+.3f}%, "
                f"accepted {item['accepted']}")

    print(f"peak resident memory         {_peak_rss_gib():10.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
