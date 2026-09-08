"""Measured agglomeration of lossless per-page JBIG2 symbol dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from .jbig2 import append_cropped_xor_region


@dataclass(frozen=True)
class PageProfile:
    page: int
    entropy: float
    signature: np.ndarray
    generic_bytes: int


@dataclass(frozen=True)
class GroupMeasurement:
    pages: tuple[int, ...]
    total_bytes: int
    global_bytes: int
    page_bytes: tuple[int, ...]
    residual_pixels: tuple[int, ...]
    pixel_exact: bool


def bitmap_profile(page: int, bitmap: Path, generic_bytes: int) -> PageProfile:
    """Measure binary entropy and a coarse layout/ink signature."""

    with Image.open(bitmap) as image:
        ink = np.logical_not(np.asarray(image.convert("1"), dtype=bool))
    probability = float(ink.mean())
    entropy = 0.0
    for value in (probability, 1.0 - probability):
        if value > 0:
            entropy -= value * np.log2(value)

    height, width = ink.shape
    y_edges = np.linspace(0, height, 17, dtype=int)
    x_edges = np.linspace(0, width, 17, dtype=int)
    blocks = np.empty((16, 16), dtype=np.float64)
    for y in range(16):
        for x in range(16):
            block = ink[y_edges[y]:y_edges[y + 1], x_edges[x]:x_edges[x + 1]]
            blocks[y, x] = block.mean() if block.size else 0.0
    rows = np.array([
        ink[y_edges[y]:y_edges[y + 1]].mean() for y in range(16)
    ])
    columns = np.array([
        ink[:, x_edges[x]:x_edges[x + 1]].mean() for x in range(16)
    ])
    signature = np.concatenate((blocks.ravel(), rows, columns, [probability]))
    norm = float(np.linalg.norm(signature))
    if norm:
        signature /= norm
    return PageProfile(page, entropy, signature, generic_bytes)


def signature_match(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def encode_verified_group(
    pages: tuple[int, ...],
    *,
    bitmap_dir: Path,
    encoder: Path,
    decoder: Path,
    threshold: float,
    work: Path,
) -> GroupMeasurement:
    """Encode one shared dictionary and require exact stock-decoder output."""

    label = f"{abs(hash((pages, threshold))):x}"
    base = work / f"group-{label}"
    inputs = [bitmap_dir / f"{page:04d}.pbm" for page in pages]
    subprocess.run(
        [str(encoder), "-p", "-s", "-r", "-t", str(threshold),
         "-b", str(base), *(str(path) for path in inputs)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    globals_path = base.with_suffix(".sym")
    global_bytes = globals_path.stat().st_size
    page_sizes = []
    residual_counts = []
    all_exact = True
    for index, (page, source_path) in enumerate(zip(pages, inputs)):
        page_path = Path(f"{base}.{index:04d}")
        decoded_path = work / f"decoded-{label}-{index}.pbm"
        subprocess.run(
            [str(decoder), "-q", "-e", "-t", "pbm", "-o", str(decoded_path),
             str(globals_path), str(page_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with Image.open(source_path) as source_image, Image.open(decoded_path) as decoded_image:
            source = np.asarray(source_image.convert("1"), dtype=bool)
            decoded = np.asarray(decoded_image.convert("1"), dtype=bool)
        difference = source != decoded
        residual_count = int(difference.sum())
        page_data = page_path.read_bytes()
        if residual_count:
            ys, xs = np.nonzero(difference)
            box = (
                max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4),
                min(source.shape[1], int(xs.max()) + 5),
                min(source.shape[0], int(ys.max()) + 5),
            )
            crop = difference[box[1]:box[3], box[0]:box[2]]
            residual_bitmap = work / f"residual-{label}-{index}.pbm"
            Image.fromarray(np.where(crop, 0, 255).astype(np.uint8), "L").convert("1").save(residual_bitmap)
            residual = subprocess.run(
                [str(encoder), "-p", str(residual_bitmap)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
            page_data = append_cropped_xor_region(page_data, residual, x=box[0], y=box[1])
            page_path = work / f"hybrid-{label}-{index}.jb2"
            page_path.write_bytes(page_data)
            verified = work / f"verified-{label}-{index}.pbm"
            subprocess.run(
                [str(decoder), "-q", "-e", "-t", "pbm", "-o", str(verified),
                 str(globals_path), str(page_path)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            with Image.open(source_path) as source_image, Image.open(verified) as verified_image:
                exact = np.array_equal(np.asarray(source_image), np.asarray(verified_image))
        else:
            exact = True
        all_exact &= exact
        page_sizes.append(len(page_data))
        residual_counts.append(residual_count)
    return GroupMeasurement(
        pages=pages,
        total_bytes=global_bytes + sum(page_sizes),
        global_bytes=global_bytes,
        page_bytes=tuple(page_sizes),
        residual_pixels=tuple(residual_counts),
        pixel_exact=all_exact,
    )
