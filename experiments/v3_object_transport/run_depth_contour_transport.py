#!/usr/bin/env python3
"""Measure persistent T caps and focus ownership on V3 contour runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.transport_focus_forensics import (
    transport_focus_forensics,
    transport_focus_interfaces,
)
from experiments.v3_object_transport.contour_transport import (
    build_contour_transport,
)
from experiments.v3_object_transport.depth_contour_transport import (
    build_depth_contour_transport,
    summarize_depth_contour_transport,
)
from experiments.v3_object_transport.junction_depth import (
    build_junction_depth,
    summarize_junction_depth,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_RESULTS,
    _load_bundle,
    _load_complex,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "depth_contour_transport")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": (
            "raw T-cap persistence and contrast-normalized focus ownership "
            "on exact one-sided contours; no depth/object partition"
        ),
        "images": {},
    }
    for name in CONTROLS:
        image_dir = args.results / name
        complex_ = _load_complex(image_dir / "compound_region_complex.npz")
        bundle = _load_bundle(image_dir / "compound_incidence_bundle.npz")
        contour = build_contour_transport(complex_, bundle)
        junction = build_junction_depth(complex_)
        source = np.asarray(Image.open(image_dir / "source.png").convert("RGB"))
        focus = transport_focus_forensics(source, complex_["labels"])
        interfaces = transport_focus_interfaces(
            focus, complex_["labels"], complex_["topology"])
        depth = build_depth_contour_transport(
            complex_, bundle, contour, junction, interfaces)
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "depth_contour_transport.npz",
            **junction,
            **depth,
            component_owner=contour["component_owner"],
            component_length=contour["component_length"],
            component_arcs=contour["component_arcs"],
            pair_component=contour["pair_component"],
            pair_region=contour["pair_region"],
            pair_fraction=contour["pair_fraction"],
        )
        report["images"][name] = {
            "junction": summarize_junction_depth(junction),
            "depth_contour": summarize_depth_contour_transport(depth, contour),
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
