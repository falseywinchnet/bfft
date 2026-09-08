#!/usr/bin/env python3
"""List pages whose JBIG2 image representation differs between two PDFs."""

from __future__ import annotations

import argparse

import pikepdf


def _masks(page):
    masks = []
    for name, image in page.images.items():
        candidates = [image]
        soft_mask = image.get("/SMask")
        if soft_mask is not None:
            candidates.append(soft_mask)
        for candidate in candidates:
            if str(candidate.get("/Filter", "")) == "/JBIG2Decode":
                masks.append((name, candidate.read_raw_bytes()))
    return sorted(masks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    with pikepdf.open(args.left) as left, pikepdf.open(args.right) as right:
        if len(left.pages) != len(right.pages):
            raise ValueError("page count differs")
        changed = [
            index + 1 for index, (a, b) in enumerate(zip(left.pages, right.pages))
            if _masks(a) != _masks(b)
        ]
    print(",".join(map(str, changed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
