"""Render the nested region-balanced palette stack on V3 controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.compound_segment_benchmark import _fit_side
from experiments.region_posterization import (
    build_region_posterization,
    multiscale_region_affinity,
    region_adjacency,
    render_posterization_level,
)
from experiments.segmenting_v3 import SegmentingV3Config, build_segmenting_v3
from viewer import gallery


def _panel(image: np.ndarray, label: str) -> Image.Image:
    rgb = np.clip(np.asarray(image) * 255.0, 0.0, 255.0).astype(np.uint8)
    panel = Image.fromarray(rgb, mode="RGB")
    header = Image.new("RGB", (panel.width, 26), (31, 31, 34))
    ImageDraw.Draw(header).text((7, 6), label, fill=(235, 235, 235))
    output = Image.new("RGB", (panel.width, panel.height + 26))
    output.paste(header, (0, 0))
    output.paste(panel, (0, 26))
    return output


def _sheet(panels: list[Image.Image]) -> Image.Image:
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    output = Image.new("RGB", (width, height), (24, 24, 24))
    left = 0
    for panel in panels:
        output.paste(panel, (left, 0))
        left += panel.width
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", nargs="+", default=("pikachu", "coffee", "coins"))
    parser.add_argument("--side", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for name in args.images:
        source = _fit_side(gallery.load(name), args.side)
        result = build_segmenting_v3(
            source,
            SegmentingV3Config(
                structural_topology="canonical_v2",
                structural_allocation_side=min(args.side, 512),
                structural_flow_sweeps=1,
                compound_segmentation=True,
                threads=4,
            ),
        )
        labels = result["compound_segmentation"]["labels"]
        poster = build_region_posterization(
            source,
            labels,
            max_depth=args.depth,
            labels_are_compact=True,
        )
        pairs = region_adjacency(labels)
        affinity = multiscale_region_affinity(poster, pairs)
        selected_depths = sorted(set((2, 3, 4, args.depth)))
        panels = [_panel(source, "source")]
        for depth in selected_depths:
            level = poster["levels"][min(depth, len(poster["levels"]) - 1)]
            panels.append(_panel(
                render_posterization_level(poster, level["depth"]),
                f"region-average palette depth {level['depth']} "
                f"({level['family_count']} families)",
            ))
        path = args.output / f"posterization_{name}.png"
        _sheet(panels).save(path)
        print({
            "name": name,
            "shape": source.shape[:2],
            "regions": poster["region_count"],
            "occupied_bins": poster["occupied_bins"],
            "families": [level["family_count"] for level in poster["levels"]],
            "poster_ms": poster["milliseconds"],
            "adjacency": len(pairs),
            "affinity_p50": float(np.percentile(affinity, 50)),
            "affinity_p90": float(np.percentile(affinity, 90)),
            "affinity_p99": float(np.percentile(affinity, 99)),
            "output": str(path),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
