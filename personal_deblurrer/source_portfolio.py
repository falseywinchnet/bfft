"""Method-agnostic image sources for deblurring generalization batteries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from denoiser.run_2d_denoiser_battery import sources as denoiser_sources

from .workbench import V3_SKIMAGE_PORTFOLIO


def _square_resize(image: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    height, width = value.shape[:2]
    extent = min(height, width)
    top = (height - extent) // 2
    left = (width - extent) // 2
    cropped = value[top:top + extent, left:left + extent]
    pixels = np.uint8(np.round(np.clip(cropped, 0.0, 1.0) * 255.0))
    return np.asarray(Image.fromarray(pixels).resize(
        (int(size), int(size)), Image.Resampling.LANCZOS),
        dtype=np.float64) / 255.0


def research_source_portfolio(
    size: int,
    *,
    asset_root: Path | None = None,
) -> dict[str, np.ndarray]:
    """Return denoiser sources plus all V3 skimage files as source data only."""
    extent = max(int(size), 16)
    root = (
        Path(__file__).resolve().parent / "source_assets" / "v3_skimage"
        if asset_root is None else Path(asset_root))
    output = {
        f"denoiser/{name}": image
        for name, image in denoiser_sources(extent).items()
    }
    missing = []
    for name in V3_SKIMAGE_PORTFOLIO:
        path = root / f"{name}.png"
        if not path.is_file():
            missing.append(path.name)
            continue
        with Image.open(path) as opened:
            mode = "L" if opened.mode in ("1", "L", "I", "F") else "RGB"
            image = np.asarray(opened.convert(mode), dtype=np.float64) / 255.0
        output[f"v3_skimage/{name}"] = _square_resize(image, extent)
    if missing:
        raise FileNotFoundError(
            "materialized V3 skimage assets are missing: " + ", ".join(missing))
    return output
