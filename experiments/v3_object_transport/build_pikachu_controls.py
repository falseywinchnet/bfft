#!/usr/bin/env python3
"""Freeze the user-supplied easy Pikachu and derive its hard frame exactly.

The character and the complete original black panel are byte-for-byte
unchanged.  Only the exterior margin is replaced: black canvas plus an
eight-pixel white wall immediately outside the original panel.  The top wall's
inner edge therefore remains at y=35, only three black pixels above the first
ear-tip pixel at y=38.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/Users/ultimussecundai/Downloads/025.png")
DEFAULT_ASSETS = ROOT / "experiments/v3_object_transport/assets"
PANEL_BOUNDS = (22, 35, 453, 440)  # left, top, exclusive right/bottom
WALL_WIDTH = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, assets: Path) -> tuple[Path, Path]:
    assets.mkdir(parents=True, exist_ok=True)
    easy_path = assets / "pikachu_easy.png"
    hard_path = assets / "pikachu_hard.png"
    shutil.copyfile(source, easy_path)

    with Image.open(source) as image:
        easy = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if easy.shape != (475, 475, 3):
        raise ValueError(f"unexpected source shape {easy.shape}")
    left, top, right, bottom = PANEL_BOUNDS
    panel = easy[top:bottom, left:right].copy()
    hard = np.zeros_like(easy)
    outer_left = left - WALL_WIDTH
    outer_top = top - WALL_WIDTH
    outer_right = right + WALL_WIDTH
    outer_bottom = bottom + WALL_WIDTH
    hard[outer_top:outer_bottom, outer_left:outer_right] = 255
    hard[top:bottom, left:right] = panel
    if not np.array_equal(hard[top:bottom, left:right], panel):
        raise AssertionError("hard-control construction changed panel pixels")
    Image.fromarray(hard, mode="RGB").save(hard_path)
    return easy_path, hard_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args()
    easy, hard = build(args.source, args.assets)
    print(f"easy {sha256(easy)} {easy}")
    print(f"hard {sha256(hard)} {hard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
