"""Measure flattened high-resolution JPEGs against a scanned PDF's MRC pages.

This is deliberately a page-representation experiment, not a PDF optimizer
pass. It renders at the requested DPI, inventories the source page's JPX/JBIG2
and content-stream bytes, traces the best ordinary JPEG at several multiples of
that real budget, and records image and optional OCR degradation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np
from PIL import Image
import pikepdf

from experiments.manual_jpeg_optimizer.core import image_metrics


@dataclass(frozen=True)
class PageStorage:
    page: int
    image_bytes: int
    content_bytes: int
    total_stream_bytes: int
    filters: dict[str, int]


def _streams(value: Any) -> list[pikepdf.Stream]:
    if value is None:
        return []
    if isinstance(value, pikepdf.Array):
        return list(value)
    return [value]


def page_storage(pdf: pikepdf.Pdf, page_number: int) -> PageStorage:
    page = pdf.pages[page_number - 1]
    seen: set[tuple[int, int]] = set()
    filters: Counter[str] = Counter()
    image_bytes = 0

    def add_image(stream: Any) -> None:
        nonlocal image_bytes
        if not isinstance(stream, pikepdf.Stream) or stream.objgen in seen:
            return
        seen.add(stream.objgen)
        size = len(stream.read_raw_bytes())
        image_bytes += size
        filters[str(stream.get("/Filter", "none"))] += size
        add_image(stream.get("/SMask"))
        mask = stream.get("/Mask")
        if isinstance(mask, pikepdf.Stream):
            add_image(mask)

    resources = page.get("/Resources") or {}
    for xobject in (resources.get("/XObject", {}) or {}).values():
        if str(xobject.get("/Subtype", "")) == "/Image":
            add_image(xobject)
    content_bytes = sum(len(stream.read_raw_bytes()) for stream in _streams(page.get("/Contents")))
    return PageStorage(
        page=page_number,
        image_bytes=image_bytes,
        content_bytes=content_bytes,
        total_stream_bytes=image_bytes + content_bytes,
        filters=dict(sorted(filters.items())),
    )


def _render_page(pdftoppm: str, source: Path, page: int, dpi: int, output: Path) -> None:
    prefix = output.with_suffix("")
    subprocess.run(
        [
            pdftoppm,
            "-f", str(page),
            "-l", str(page),
            "-singlefile",
            "-r", str(dpi),
            "-png",
            str(source),
            str(prefix),
        ],
        check=True,
    )
    rendered = prefix.with_suffix(".png")
    if rendered != output:
        rendered.replace(output)


def _best_jpeg_candidates(image: Image.Image) -> list[tuple[int, int, int, bool, bytes]]:
    candidates = []
    for progressive in (False, True):
        for subsampling in (2, 1, 0):
            for quality in range(1, 96):
                buffer = BytesIO()
                image.save(
                    buffer,
                    format="JPEG",
                    quality=quality,
                    subsampling=subsampling,
                    optimize=True,
                    progressive=progressive,
                )
                candidates.append(
                    (len(buffer.getvalue()), quality, subsampling, progressive, buffer.getvalue())
                )
    return candidates


def _normalize_ocr(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tesseract(path: Path) -> str | None:
    executable = shutil.which("tesseract")
    if not executable:
        return None
    completed = subprocess.run(
        [executable, str(path), "stdout", "--psm", "6"],
        check=True,
        capture_output=True,
        text=True,
    )
    return _normalize_ocr(completed.stdout)


def run_study(
    source: Path,
    output: Path,
    *,
    pages: list[int],
    dpi: int,
    multiples: list[int],
    pdftoppm: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"source": str(source), "dpi": dpi, "pages": {}}
    with pikepdf.open(source) as pdf:
        for page_number in pages:
            storage = page_storage(pdf, page_number)
            source_png = output / f"page-{page_number}-source.png"
            _render_page(pdftoppm, source, page_number, dpi, source_png)
            source_image = Image.open(source_png).convert("RGB")
            reference = np.asarray(source_image, dtype=np.float64)
            centered_power = float(np.mean((reference - reference.mean(axis=(0, 1), keepdims=True)) ** 2))
            signal_power = float(np.mean(reference**2))
            source_ocr = _tesseract(source_png)
            candidates = _best_jpeg_candidates(source_image)
            frontier = []
            for multiple in multiples:
                target = storage.total_stream_bytes * multiple
                under = [candidate for candidate in candidates if candidate[0] <= target]
                selected = (
                    max(under, key=lambda item: (item[1], -item[2], item[3], item[0]))
                    if under
                    else min(candidates, key=lambda item: item[0])
                )
                size, quality, subsampling, progressive, data = selected
                candidate_path = output / f"page-{page_number}-jpeg-{multiple}x.jpg"
                candidate_path.write_bytes(data)
                candidate = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.float64)
                mse = float(np.mean((reference - candidate) ** 2))
                ssim, psnr, edge_psnr = image_metrics(reference, candidate)
                candidate_ocr = _tesseract(candidate_path)
                frontier.append(
                    {
                        "multiple": multiple,
                        "target_bytes": target,
                        "bytes": size,
                        "target_met": size <= target,
                        "quality": quality,
                        "subsampling": subsampling,
                        "progressive": progressive,
                        "ssim": ssim,
                        "psnr_db": psnr,
                        "edge_psnr_db": edge_psnr,
                        "snr_db": 10.0 * math.log10(signal_power / mse),
                        "ac_snr_db": 10.0 * math.log10(centered_power / mse),
                        "ocr_similarity_to_source_render": (
                            SequenceMatcher(None, source_ocr, candidate_ocr).ratio()
                            if source_ocr is not None and candidate_ocr is not None
                            else None
                        ),
                        "path": str(candidate_path),
                    }
                )
            report["pages"][str(page_number)] = {
                "storage": asdict(storage),
                "render_pixels": list(source_image.size),
                "source_ocr_characters": len(source_ocr) if source_ocr is not None else None,
                "jpeg_frontier": frontier,
            }
    report_path = output / "scan-jpeg-study.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--pages", default="30,100")
    parser.add_argument("--dpi", type=int, default=360)
    parser.add_argument("--multiples", default="1,2,4,8")
    parser.add_argument("--out", type=Path, default=Path("tmp/pdfs/scan-jpeg-study"))
    parser.add_argument("--pdftoppm", default=shutil.which("pdftoppm") or "pdftoppm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_study(
        args.source,
        args.out,
        pages=[int(value) for value in args.pages.split(",")],
        dpi=args.dpi,
        multiples=[int(value) for value in args.multiples.split(",")],
        pdftoppm=args.pdftoppm,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

