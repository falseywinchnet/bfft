#!/usr/bin/env python3
"""Embed the verified Matplotlib 3-D figure in the report fragment."""
from __future__ import annotations

import base64
from pathlib import Path


HERE = Path(__file__).resolve().parent
IMAGE = HERE / "complex_spiral_3d_inline.png"
TEMPLATE = HERE / "complex_spiral_3d.template.html"
FRAGMENT = HERE / "complex_spiral_3d.fragment.html"


def main():
    encoded = base64.b64encode(IMAGE.read_bytes()).decode("ascii")
    FRAGMENT.write_text(TEMPLATE.read_text().replace("__IMAGE__", encoded))
    print(FRAGMENT)
    print(f"{FRAGMENT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
