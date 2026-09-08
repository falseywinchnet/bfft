"""Conservative foreground/background optimization for MRC scanned PDFs.

This module is intentionally separate from the lossless optimizer.  It keeps
the one-bit foreground geometry byte-for-byte intact, but may replace the
color planes when measured gates pass:

* a high-resolution ink-color plane can become a tiny interpolated RGB field;
* an anomalous high-resolution background can be reduced to the document's
  normal background resolution after mask-guided removal of leaked glyphs.

The implementation is page-streaming and rejects every proposal that is not
smaller or that misses its representation-specific fidelity gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable
import zlib

import numpy as np
from PIL import Image
import pikepdf
from pikepdf import Dictionary, Name, ObjectStreamMode, Stream
from scipy.ndimage import binary_dilation, gaussian_filter
import zopfli.zlib


@dataclass(frozen=True)
class MrcConfig:
    """Controls for the opt-in MRC color-plane pass."""

    jbig2_decoder: str | Path
    mask_cache: str | Path | None = None
    foreground_rmse_limit: float = 8.0
    foreground_grid_widths: tuple[int, ...] = (1, 8, 16, 32)
    foreground_max_mask_fraction: float = 0.18
    background_scale: int = 3
    background_min_source_bytes: int = 12_000
    background_min_mask_fraction: float = 0.05
    background_max_mask_fraction: float = 0.18
    background_min_ghost_rmse: float = 2.0
    background_max_ghost_rmse: float = 12.0
    background_outside_psnr: float = 44.0
    background_jpx_rates: tuple[int, ...] = (200, 400, 800, 1200)
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.foreground_rmse_limit <= 0:
            raise ValueError("foreground_rmse_limit must be positive")
        if not self.foreground_grid_widths or min(self.foreground_grid_widths) < 1:
            raise ValueError("foreground_grid_widths must contain positive integers")
        if not 0 < self.foreground_max_mask_fraction < 1:
            raise ValueError("foreground_max_mask_fraction must be between zero and one")
        if self.background_scale < 2:
            raise ValueError("background_scale must be at least two")
        if not 0 < self.background_min_mask_fraction < self.background_max_mask_fraction < 1:
            raise ValueError("background mask-fraction bounds are inconsistent")
        if self.background_outside_psnr <= 0:
            raise ValueError("background_outside_psnr must be positive")
        if self.background_max_ghost_rmse <= self.background_min_ghost_rmse:
            raise ValueError("background ghost RMSE bounds are inconsistent")
        if not self.background_jpx_rates or min(self.background_jpx_rates) <= 0:
            raise ValueError("background_jpx_rates must contain positive values")


@dataclass(frozen=True)
class MrcResult:
    source: Path
    output: Path
    source_bytes: int
    output_bytes: int
    pages_examined: int
    foreground_planes_replaced: int
    foreground_bytes_before: int
    foreground_bytes_after: int
    backgrounds_replaced: int
    background_bytes_before: int
    background_bytes_after: int
    skipped: dict[str, int]
    pages: list[dict[str, Any]]

    @property
    def saved_bytes(self) -> int:
        return self.source_bytes - self.output_bytes

    @property
    def saved_fraction(self) -> float:
        return self.saved_bytes / self.source_bytes if self.source_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source"] = str(self.source)
        value["output"] = str(self.output)
        value["saved_bytes"] = self.saved_bytes
        value["saved_fraction"] = self.saved_fraction
        return value


def _image_xobjects(page: pikepdf.Page) -> list[tuple[str, Stream]]:
    resources = page.get("/Resources") or {}
    xobjects = resources.get("/XObject") or {}
    return [
        (str(name), obj)
        for name, obj in xobjects.items()
        if isinstance(obj, Stream) and str(obj.get("/Subtype", "")) == "/Image"
    ]


def _mrc_pair(page: pikepdf.Page) -> tuple[Stream, Stream, Stream] | None:
    """Return background, foreground, and mask for a simple full-page MRC page."""

    images = _image_xobjects(page)
    foregrounds = [obj for _, obj in images if isinstance(obj.get("/SMask"), Stream)]
    backgrounds = [obj for _, obj in images if obj.get("/SMask") is None]
    if len(foregrounds) != 1 or len(backgrounds) != 1:
        return None
    foreground = foregrounds[0]
    mask = foreground.get("/SMask")
    if str(foreground.get("/ColorSpace", "")) != "/DeviceRGB":
        return None
    if str(backgrounds[0].get("/ColorSpace", "")) != "/DeviceRGB":
        return None
    if str(mask.get("/Filter", "")) != "/JBIG2Decode":
        return None
    return backgrounds[0], foreground, mask


def _decode_jpx(stream: Stream) -> Image.Image:
    if str(stream.get("/Filter", "")) != "/JPXDecode":
        raise ValueError("color plane is not JPEG 2000")
    with Image.open(BytesIO(stream.read_raw_bytes())) as image:
        return image.convert("RGB").copy()


def _decode_mask(
    stream: Stream,
    *,
    page_number: int,
    decoder: str | Path,
    mask_cache: str | Path | None,
    timeout: float,
    work: Path,
) -> Image.Image:
    width = int(stream.get("/Width", 0))
    height = int(stream.get("/Height", 0))
    if mask_cache is not None:
        cached = Path(mask_cache) / f"{page_number:04d}.pbm"
        if cached.exists():
            with Image.open(cached) as image:
                result = image.convert("1").copy()
            if result.size == (width, height):
                return result

    raw = stream.read_raw_bytes()
    decode_parms = stream.get("/DecodeParms")
    globals_stream = (
        decode_parms.get("/JBIG2Globals")
        if isinstance(decode_parms, Dictionary)
        else None
    )
    encoded = work / f"mask-{page_number:04d}.jb2"
    bitmap = work / f"mask-{page_number:04d}.pbm"
    prefix = globals_stream.read_raw_bytes() if isinstance(globals_stream, Stream) else b""
    encoded.write_bytes(prefix + raw)
    subprocess.run(
        [str(decoder), "-q", "-e", "-t", "pbm", "-o", str(bitmap), str(encoded)],
        check=True,
        capture_output=True,
        timeout=timeout,
    )
    with Image.open(bitmap) as image:
        result = image.convert("1").copy()
    if result.size != (width, height):
        raise ValueError("decoded mask dimensions do not match the PDF image")
    return result


def _mask_for_size(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask.convert("L")) > 0


def _field_grid(
    image: np.ndarray,
    mask: np.ndarray,
    grid_width: int,
) -> tuple[np.ndarray, float]:
    """Fit a smooth masked RGB field and return its visible-pixel RMSE."""

    height, width = image.shape[:2]
    grid_height = max(1, round(grid_width * height / width))
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("empty foreground mask")
    if len(x) > 300_000:
        selected = np.linspace(0, len(x) - 1, 300_000, dtype=np.int64)
        x = x[selected]
        y = y[selected]
    samples = image[y, x].astype(np.float64)
    if grid_width == 1:
        grid = np.rint(samples.mean(axis=0)).astype(np.uint8).reshape(1, 1, 3)
    else:
        bx = np.minimum(grid_width - 1, x * grid_width // width)
        by = np.minimum(grid_height - 1, y * grid_height // height)
        index = by * grid_width + bx
        count = np.bincount(index, minlength=grid_width * grid_height).reshape(
            grid_height, grid_width
        ).astype(np.float64)
        channels: list[np.ndarray] = []
        for channel in range(3):
            total = np.bincount(
                index,
                weights=samples[:, channel],
                minlength=grid_width * grid_height,
            ).reshape(grid_height, grid_width)
            numerator = gaussian_filter(total, 1.0, mode="nearest")
            denominator = gaussian_filter(count, 1.0, mode="nearest")
            values = numerator / np.maximum(denominator, 1e-9)
            values[denominator < 1e-6] = samples[:, channel].mean()
            channels.append(values)
        grid = np.clip(np.rint(np.stack(channels, axis=2)), 0, 255).astype(np.uint8)

    # Evaluate only a deterministic masked sample.  Constructing a full-size
    # reconstruction for every grid proposal is wasteful on 360-dpi books.
    if len(x) > 200_000:
        selected = np.linspace(0, len(x) - 1, 200_000, dtype=np.int64)
        sample_x = x[selected]
        sample_y = y[selected]
    else:
        sample_x = x
        sample_y = y
    gx = np.clip((sample_x + 0.5) * grid.shape[1] / width - 0.5, 0, grid.shape[1] - 1)
    gy = np.clip((sample_y + 0.5) * grid.shape[0] / height - 0.5, 0, grid.shape[0] - 1)
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    x1 = np.minimum(x0 + 1, grid.shape[1] - 1)
    y1 = np.minimum(y0 + 1, grid.shape[0] - 1)
    wx = (gx - x0)[:, None]
    wy = (gy - y0)[:, None]
    field = grid.astype(np.float64)
    top = field[y0, x0] * (1.0 - wx) + field[y0, x1] * wx
    bottom = field[y1, x0] * (1.0 - wx) + field[y1, x1] * wx
    reconstructed = top * (1.0 - wy) + bottom * wy
    difference = image[sample_y, sample_x].astype(np.float64) - reconstructed
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return grid, rmse


def _smallest_foreground(
    image: Image.Image,
    mask: Image.Image,
    *,
    grid_widths: Iterable[int],
    rmse_limit: float,
) -> tuple[bytes, tuple[int, int], float, int] | None:
    pixels = np.asarray(image.convert("RGB"))
    visible = _mask_for_size(mask, image.size)
    candidates: list[tuple[int, bytes, tuple[int, int], float, int]] = []
    for width in grid_widths:
        grid, rmse = _field_grid(pixels, visible, width)
        if rmse > rmse_limit:
            continue
        encoded = zopfli.zlib.compress(grid.tobytes(), numiterations=8)
        candidates.append((len(encoded), encoded, (grid.shape[1], grid.shape[0]), rmse, width))
    if not candidates:
        return None
    _, encoded, size, rmse, width = min(candidates, key=lambda item: (item[0], item[4]))
    return encoded, size, rmse, width


def _write_flate_rgb(stream: Stream, data: bytes, size: tuple[int, int]) -> None:
    stream.write(data, filter=Name.FlateDecode)
    stream.Width = size[0]
    stream.Height = size[1]
    stream.ColorSpace = Name.DeviceRGB
    stream.BitsPerComponent = 8
    stream.Interpolate = True
    for key in ("/DecodeParms", "/Decode", "/Intent"):
        if key in stream:
            del stream[key]


def _background_repair(
    background: Image.Image,
    mask: Image.Image,
    *,
    scale: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Downsample and remove mask-correlated glyph leakage from a background."""

    width, height = background.size
    target_width = max(1, round(mask.width / scale))
    target_height = max(1, round(mask.height / scale))
    source = np.asarray(background.convert("RGB"), dtype=np.float64)
    reduced = np.asarray(
        background.resize((target_width, target_height), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    coverage = np.asarray(
        mask.convert("L").resize((target_width, target_height), Image.Resampling.BOX),
        dtype=np.float64,
    ) / 255.0
    contaminated = binary_dilation(coverage > 0.01, iterations=2)
    # A broad low-pass paper estimate is deliberately used instead of
    # normalized filling from the irregular valid set.  The latter can turn
    # dense text lines into horizontal color bands on sparse title pages.
    estimate = np.empty_like(reduced)
    for channel in range(3):
        estimate[:, :, channel] = gaussian_filter(
            reduced[:, :, channel], 6.0, mode="nearest"
        )
    ghost_rmse = float(
        np.sqrt(np.mean((reduced[contaminated] - estimate[contaminated]) ** 2))
    )
    alpha = gaussian_filter(contaminated.astype(np.float64), 0.7, mode="nearest")[:, :, None]
    repaired = np.clip(np.rint(reduced * (1.0 - alpha) + estimate * alpha), 0, 255).astype(np.uint8)

    # This measures the resolution change away from all mask-influenced pixels.
    enlarged = np.asarray(
        Image.fromarray(repaired, "RGB").resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    full_mask = _mask_for_size(mask, (width, height))
    protected = binary_dilation(full_mask, iterations=max(3, scale * 3))
    outside = source[~protected] - enlarged[~protected]
    outside_rmse = float(np.sqrt(np.mean(outside * outside))) if outside.size else math.inf
    outside_psnr = 20.0 * math.log10(255.0 / max(outside_rmse, 1e-12))
    return repaired, contaminated, ghost_rmse, outside_psnr


def _encode_jpx(image: np.ndarray, rate: int) -> bytes:
    output = BytesIO()
    Image.fromarray(image, "RGB").save(
        output,
        format="JPEG2000",
        irreversible=True,
        quality_mode="rates",
        quality_layers=[rate],
    )
    return output.getvalue()


def _smallest_background(
    repaired: np.ndarray,
    contaminated: np.ndarray,
    *,
    rates: Iterable[int],
    outside_psnr_limit: float,
) -> tuple[bytes, float, int] | None:
    candidates: list[tuple[int, bytes, float, int]] = []
    for rate in rates:
        encoded = _encode_jpx(repaired, rate)
        with Image.open(BytesIO(encoded)) as decoded_image:
            decoded = np.asarray(decoded_image.convert("RGB"), dtype=np.float64)
        outside = repaired[~contaminated].astype(np.float64) - decoded[~contaminated]
        rmse = float(np.sqrt(np.mean(outside * outside))) if outside.size else math.inf
        psnr = 20.0 * math.log10(255.0 / max(rmse, 1e-12))
        if psnr >= outside_psnr_limit:
            candidates.append((len(encoded), encoded, psnr, rate))
    if not candidates:
        return None
    _, encoded, psnr, rate = min(candidates, key=lambda item: (item[0], -item[3]))
    return encoded, psnr, rate


def _write_jpx_rgb(stream: Stream, data: bytes, size: tuple[int, int]) -> None:
    stream.write(data, filter=Name.JPXDecode)
    stream.Width = size[0]
    stream.Height = size[1]
    stream.ColorSpace = Name.DeviceRGB
    stream.BitsPerComponent = 8
    stream.Interpolate = True
    for key in ("/DecodeParms", "/Decode", "/Intent"):
        if key in stream:
            del stream[key]


def optimize_mrc_pdf(
    source: str | Path,
    output: str | Path,
    *,
    config: MrcConfig,
    password: str = "",
) -> MrcResult:
    """Optimize compatible MRC color planes while retaining exact mask streams."""

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError("source and output must be different paths")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped: Counter[str] = Counter()
    page_records: list[dict[str, Any]] = []
    foreground_before = foreground_after = foreground_replaced = 0
    background_before = background_after = background_replaced = 0

    with tempfile.TemporaryDirectory(prefix=".pdf-mrc-", dir=output_path.parent) as directory:
        work = Path(directory)
        with pikepdf.open(source_path, password=password) as pdf:
            for page_number, page in enumerate(pdf.pages, 1):
                pair = _mrc_pair(page)
                if pair is None:
                    skipped["not_simple_mrc"] += 1
                    continue
                background, foreground, mask_stream = pair
                record: dict[str, Any] = {"page": page_number}
                try:
                    mask = _decode_mask(
                        mask_stream,
                        page_number=page_number,
                        decoder=config.jbig2_decoder,
                        mask_cache=config.mask_cache,
                        timeout=config.timeout_seconds,
                        work=work,
                    )
                    foreground_image = _decode_jpx(foreground)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    skipped["decode_failed"] += 1
                    record["skip"] = f"decode_failed:{type(exc).__name__}"
                    page_records.append(record)
                    continue

                mask_fraction = float(np.mean(_mask_for_size(mask, mask.size)))
                record["mask_fraction"] = mask_fraction
                source_foreground_bytes = len(foreground.read_raw_bytes())
                foreground_before += source_foreground_bytes
                field = (
                    _smallest_foreground(
                        foreground_image,
                        mask,
                        grid_widths=config.foreground_grid_widths,
                        rmse_limit=config.foreground_rmse_limit,
                    )
                    if mask_fraction <= config.foreground_max_mask_fraction
                    else None
                )
                if field is not None and len(field[0]) < source_foreground_bytes:
                    encoded, size, rmse, grid_width = field
                    _write_flate_rgb(foreground, encoded, size)
                    foreground_replaced += 1
                    foreground_after += len(encoded)
                    record["foreground"] = {
                        "source_bytes": source_foreground_bytes,
                        "candidate_bytes": len(encoded),
                        "visible_rmse": rmse,
                        "grid": [size[0], size[1]],
                        "grid_width": grid_width,
                    }
                else:
                    foreground_after += source_foreground_bytes
                    skipped["foreground_gate"] += 1

                source_background_bytes = len(background.read_raw_bytes())
                background_before += source_background_bytes
                high_resolution_background = (
                    int(background.get("/Width", 0)) >= mask.width * 0.8
                    and int(background.get("/Height", 0)) >= mask.height * 0.8
                )
                eligible_background = (
                    high_resolution_background
                    and source_background_bytes >= config.background_min_source_bytes
                    and mask_fraction >= config.background_min_mask_fraction
                    and mask_fraction <= config.background_max_mask_fraction
                )
                if not eligible_background:
                    background_after += source_background_bytes
                    skipped["background_gate"] += 1
                    page_records.append(record)
                    continue
                try:
                    background_image = _decode_jpx(background)
                    repaired, contaminated, ghost_rmse, scale_psnr = _background_repair(
                        background_image,
                        mask,
                        scale=config.background_scale,
                    )
                    candidate = _smallest_background(
                        repaired,
                        contaminated,
                        rates=config.background_jpx_rates,
                        outside_psnr_limit=config.background_outside_psnr,
                    )
                except (OSError, ValueError):
                    candidate = None
                    ghost_rmse = 0.0
                    scale_psnr = 0.0
                if (
                    candidate is not None
                    and ghost_rmse >= config.background_min_ghost_rmse
                    and ghost_rmse <= config.background_max_ghost_rmse
                    and scale_psnr >= config.background_outside_psnr
                    and len(candidate[0]) < source_background_bytes
                ):
                    encoded, codec_psnr, rate = candidate
                    size = (repaired.shape[1], repaired.shape[0])
                    _write_jpx_rgb(background, encoded, size)
                    background_replaced += 1
                    background_after += len(encoded)
                    record["background"] = {
                        "source_bytes": source_background_bytes,
                        "candidate_bytes": len(encoded),
                        "source_size": [background_image.width, background_image.height],
                        "candidate_size": [size[0], size[1]],
                        "ghost_rmse": ghost_rmse,
                        "outside_scale_psnr": scale_psnr,
                        "outside_codec_psnr": codec_psnr,
                        "jpx_rate": rate,
                    }
                else:
                    background_after += source_background_bytes
                    skipped["background_quality_gate"] += 1
                page_records.append(record)

            candidate = work / "candidate.pdf"
            pdf.remove_unreferenced_resources()
            pdf.save(
                candidate,
                compress_streams=True,
                recompress_flate=False,
                object_stream_mode=ObjectStreamMode.preserve,
                linearize=False,
                preserve_pdfa=True,
                encryption=True if pdf.is_encrypted else None,
            )
        candidate.replace(output_path)

    with pikepdf.open(output_path, password=password) as checked:
        if len(checked.pages) != len(page_records) + skipped["not_simple_mrc"]:
            raise RuntimeError("page count changed during MRC optimization")
        problems = checked.check_pdf_syntax()
        if problems:
            raise RuntimeError("PDF syntax check failed: " + "; ".join(problems))

    return MrcResult(
        source=source_path,
        output=output_path,
        source_bytes=source_path.stat().st_size,
        output_bytes=output_path.stat().st_size,
        pages_examined=len(page_records),
        foreground_planes_replaced=foreground_replaced,
        foreground_bytes_before=foreground_before,
        foreground_bytes_after=foreground_after,
        backgrounds_replaced=background_replaced,
        background_bytes_before=background_before,
        background_bytes_after=background_after,
        skipped=dict(sorted(skipped.items())),
        pages=page_records,
    )


def write_mrc_json(path: str | Path, result: MrcResult) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
