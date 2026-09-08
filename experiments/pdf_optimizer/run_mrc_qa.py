#!/usr/bin/env python3
"""Render selected pages and measure an opt-in MRC rewrite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from scipy.ndimage import sobel


def _render(document: fitz.Document, page_number: int, dpi: int) -> np.ndarray:
    page = document[page_number - 1]
    pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)


def _psnr(left: np.ndarray, right: np.ndarray) -> float:
    difference = left.astype(np.float64) - right.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return 20.0 * math.log10(255.0 / max(rmse, 1e-12))


def _edge(image: np.ndarray) -> np.ndarray:
    luminance = image.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    return np.hypot(sobel(luminance, axis=0), sobel(luminance, axis=1))


def _dark_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_luma = left.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    right_luma = right.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    threshold = min(float(np.percentile(left_luma, 25)), 150.0)
    a = left_luma < threshold
    b = right_luma < threshold
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--pages", required=True, help="comma-separated one-based pages")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--renders", type=Path)
    args = parser.parse_args()
    pages = [int(value) for value in args.pages.split(",")]
    if args.renders:
        args.renders.mkdir(parents=True, exist_ok=True)

    records = []
    with fitz.open(args.source) as source, fitz.open(args.candidate) as candidate:
        if source.page_count != candidate.page_count:
            raise RuntimeError("page count differs")
        for page_number in pages:
            left = _render(source, page_number, args.dpi)
            right = _render(candidate, page_number, args.dpi)
            if left.shape != right.shape:
                raise RuntimeError(f"render dimensions differ on page {page_number}")
            left_edge = _edge(left)
            right_edge = _edge(right)
            edge_rmse = float(np.sqrt(np.mean((left_edge - right_edge) ** 2)))
            edge_peak = max(float(np.max(left_edge)), 1e-12)
            record = {
                "page": page_number,
                "pixel_differences": int(np.count_nonzero(left != right)),
                "pixel_identical": bool(np.array_equal(left, right)),
                "psnr": _psnr(left, right),
                "edge_psnr": 20.0 * math.log10(edge_peak / max(edge_rmse, 1e-12)),
                "dark_mask_iou": _dark_iou(left, right),
                "text_identical": source[page_number - 1].get_text() == candidate[page_number - 1].get_text(),
            }
            records.append(record)
            if args.renders:
                Image.fromarray(right, "RGB").save(args.renders / f"page-{page_number:04d}.png")

    payload = {"dpi": args.dpi, "pages": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
