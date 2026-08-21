#!/usr/bin/env python3
"""Materialize the V3 skimage portfolio as method-agnostic source assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .workbench import V3_SKIMAGE_PORTFOLIO, load_v3_skimage_source


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in V3_SKIMAGE_PORTFOLIO:
        image = load_v3_skimage_source(name)
        pixels = np.uint8(np.round(np.clip(image, 0.0, 1.0) * 255.0))
        path = output / f"{name}.png"
        Image.fromarray(pixels).save(path, optimize=True)
        records.append({
            "name": name,
            "file": path.name,
            "shape": list(image.shape),
            "sha256": _sha256(path),
        })
    manifest = {
        "portfolio": "segmenter_v3_skimage_source_data_only",
        "method_inheritance": "none",
        "source_api": "skimage.data",
        "count": len(records),
        "records": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("personal_deblurrer/source_assets/v3_skimage"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
