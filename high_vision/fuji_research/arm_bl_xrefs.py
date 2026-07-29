#!/usr/bin/env python3
"""Find direct ARM-state BL references in a flat firmware image."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def direct_bl_xrefs(image: bytes, base: int, target: int) -> list[int]:
    hits: list[int] = []
    for offset in range(0, len(image) - 3, 4):
        instruction = struct.unpack_from("<I", image, offset)[0]
        if (instruction >> 24) & 0xF != 0xB:
            continue
        displacement = instruction & 0xFFFFFF
        if displacement & 0x800000:
            displacement -= 1 << 24
        destination = (base + offset + 8 + (displacement << 2)) & 0xFFFFFFFF
        if destination == target:
            hits.append(base + offset)
    return hits


def integer(text: str) -> int:
    return int(text, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("targets", nargs="+", type=integer)
    parser.add_argument("--base", type=integer, default=0xC0000000)
    args = parser.parse_args()

    image = args.image.read_bytes()
    for target in args.targets:
        hits = direct_bl_xrefs(image, args.base, target)
        rendered = " ".join(f"0x{address:08x}" for address in hits) or "(none)"
        print(f"0x{target:08x}: {rendered}")


if __name__ == "__main__":
    main()
