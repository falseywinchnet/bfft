"""PNG-native palette ownership transport and measured rate/distortion search.

Unlike the JPEG experiment, this module never applies a DCT.  Its lossless
transport acts on palette identities: a palette permutation and the inverse
pixel-label permutation conserve every displayed RGBA constituent exactly,
while changing the byte field seen by PNG filters and DEFLATE.  The optional
lossy pass propagates palette ownership only across edge-gated neighboring
pixels and accepts candidates by actual PNG bytes and decoded image metrics.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from io import BytesIO
import json
from pathlib import Path
import struct
from time import perf_counter
from typing import Callable, Iterable
import zlib

import numpy as np
from PIL import Image

from experiments.manual_jpeg_optimizer.core import image_metrics

try:
    from bfft.vision import (
        nearest_code_native as _native_nearest_code,
        weighted_kmeanspp_native as _native_weighted_kmeanspp,
    )
except (ImportError, OSError):  # Portable/source-only installations.
    _native_nearest_code = None
    _native_weighted_kmeanspp = None


Progress = Callable[[int, int, str], None]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _nearest_labels(observations: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Assign low-dimensional palette owners using the fastest exact backend."""
    if _native_nearest_code is not None:
        labels = _native_nearest_code(observations, codes, threads=0)
        if labels is not None:
            return labels
    from scipy.cluster.vq import vq
    return np.asarray(vq(observations, codes, check_finite=False)[0], np.int32)


def _weighted_kmeanspp(
    sample: np.ndarray,
    sample_weight: np.ndarray,
    colors: int,
    rng: np.random.Generator,
    *,
    pad_degenerate: bool = False,
) -> np.ndarray:
    """Seed palette nodes while keeping deterministic RNG ownership in Python."""
    if _native_weighted_kmeanspp is not None:
        centers = _native_weighted_kmeanspp(
            sample, sample_weight, colors, rng.random(colors)
        )
        if centers is not None:
            if pad_degenerate and len(centers) < colors:
                centers = np.concatenate((
                    centers,
                    np.repeat(centers[-1:], colors - len(centers), axis=0),
                ))
            return centers
    centers = np.empty((colors, sample.shape[1]), dtype=np.float32)
    first = rng.choice(len(sample), p=sample_weight / sample_weight.sum())
    centers[0] = sample[first]
    closest = np.sum((sample - centers[0]) ** 2, axis=1)
    live = 1
    for index in range(1, colors):
        probability = sample_weight * closest
        total = float(probability.sum())
        if total <= 1e-12:
            break
        choice = rng.choice(len(sample), p=probability / total)
        centers[index] = sample[choice]
        closest = np.minimum(
            closest, np.sum((sample - centers[index]) ** 2, axis=1)
        )
        live = index + 1
    if pad_degenerate and live < colors:
        centers[live:] = centers[live - 1]
        live = colors
    return centers[:live]


@dataclass(frozen=True)
class PNGConfig:
    """Search and encoding controls.

    ``colors=0`` selects a short rate-guided palette search.  ``lossless``
    disables palette quantization and ownership flow.  Ownership strength is
    measured in squared Oklab units; zero is pure palette-index transport.
    """

    target_bytes: int = 0
    minimum_ssim: float = 0.0
    colors: int = 0
    minimum_colors: int = 8
    dither: str = "none"  # none, floyd, auto
    quantizer: str = "auto"  # auto adds full-image Lloyd refinement at the selected rate
    lloyd_iterations: int = 10
    palette_edge_weight: float = 1.5
    palette_sample_limit: int = 131072
    palette_seed: int = 508030340
    diffusion_strength: float = 0.9
    diffusion_edge_barrier: float = 3.0
    ownership_strength: float = -1.0  # negative means a short automatic ladder
    ownership_iterations: int = 2
    edge_protection: float = 8.0
    palette_transport: bool = True
    filter_search: str = "fast"  # fast or thorough
    compression_level: int = 9
    lossless: bool = False
    preserve_color_profile: bool = True


@dataclass
class PNGCandidate:
    colors: int
    quantizer: str
    palette_edge_weight: float
    dither: str
    diffusion_strength: float
    ownership_strength: float
    palette_order: str
    filter_policy: str
    zlib_strategy: str
    size: int
    ssim: float
    psnr_db: float
    edge_psnr_db: float
    smooth_transition_coverage: float
    elapsed_seconds: float
    data: bytes = field(repr=False)
    rgba: np.ndarray = field(repr=False)

    def report(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("data")
        result.pop("rgba")
        for key, value in tuple(result.items()):
            if isinstance(value, float) and not np.isfinite(value):
                result[key] = None
        return result


@dataclass
class PNGOptimizationResult:
    source: Path
    source_bytes: int
    source_mode: str
    shape: tuple[int, int, int]
    config: PNGConfig
    winner: PNGCandidate
    candidates: list[PNGCandidate]
    elapsed_seconds: float

    def report(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "source_bytes": self.source_bytes,
            "source_mode": self.source_mode,
            "shape": list(self.shape),
            "config": asdict(self.config),
            "palette_compute_backend": (
                "native_cpp"
                if _native_nearest_code is not None and
                _native_weighted_kmeanspp is not None
                else "scipy_numpy"
            ),
            "terminal_deflate_workers": 4,
            "winner": self.winner.report(),
            "candidates": [candidate.report() for candidate in self.candidates],
            "elapsed_seconds": self.elapsed_seconds,
        }

    def save(self, output: str | Path, report: str | Path | None = None) -> None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.winner.data)
        if report is not None:
            report_path = Path(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(self.report(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _paeth(left: np.ndarray, above: np.ndarray, upper_left: np.ndarray) -> np.ndarray:
    left16 = left.astype(np.int16)
    above16 = above.astype(np.int16)
    corner16 = upper_left.astype(np.int16)
    estimate = left16 + above16 - corner16
    left_distance = np.abs(estimate - left16)
    above_distance = np.abs(estimate - above16)
    corner_distance = np.abs(estimate - corner16)
    return np.where(
        (left_distance <= above_distance) & (left_distance <= corner_distance),
        left,
        np.where(above_distance <= corner_distance, above, upper_left),
    ).astype(np.uint8)


def _filtered_rows(rows: np.ndarray, bytes_per_pixel: int) -> tuple[list[np.ndarray], np.ndarray]:
    raw = np.ascontiguousarray(rows, dtype=np.uint8)
    above = np.zeros_like(raw)
    above[1:] = raw[:-1]
    left = np.zeros_like(raw)
    left[:, bytes_per_pixel:] = raw[:, :-bytes_per_pixel]
    upper_left = np.zeros_like(raw)
    upper_left[1:, bytes_per_pixel:] = raw[:-1, :-bytes_per_pixel]

    filters = [
        raw,
        np.subtract(raw, left, dtype=np.uint8),
        np.subtract(raw, above, dtype=np.uint8),
        np.subtract(
            raw,
            ((left.astype(np.uint16) + above.astype(np.uint16)) // 2).astype(np.uint8),
            dtype=np.uint8,
        ),
        np.subtract(raw, _paeth(left, above, upper_left), dtype=np.uint8),
    ]
    scores = np.empty((5, raw.shape[0]), dtype=np.int64)
    for index, values in enumerate(filters):
        signed = values.astype(np.int16)
        signed[signed >= 128] -= 256
        scores[index] = np.abs(signed).sum(axis=1, dtype=np.int64)
    return filters, scores


def _filter_stream(filters: list[np.ndarray], choices: np.ndarray) -> bytes:
    height, row_bytes = filters[0].shape
    output = np.empty((height, row_bytes + 1), dtype=np.uint8)
    output[:, 0] = choices
    stack = np.stack(filters, axis=0)
    output[:, 1:] = stack[choices, np.arange(height)]
    return output.tobytes()


_STRATEGIES = {
    "default": zlib.Z_DEFAULT_STRATEGY,
    "filtered": zlib.Z_FILTERED,
    "rle": zlib.Z_RLE,
}

# Terminal DEFLATE trials are independent and Python's zlib releases the GIL.
# One shared bounded pool avoids paying thread construction once per palette.
_ZLIB_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="png-zlib")


def _compress(stream: bytes, level: int, strategy: str, memory_level: int = 8) -> bytes:
    compressor = zlib.compressobj(
        level=max(0, min(9, int(level))),
        method=zlib.DEFLATED,
        wbits=15,
        memLevel=max(1, min(9, int(memory_level))),
        strategy=_STRATEGIES[strategy],
    )
    return compressor.compress(stream) + compressor.flush()


def _best_idat(
    rows: np.ndarray,
    bytes_per_pixel: int,
    level: int,
    search: str,
    indexed: bool = False,
) -> tuple[bytes, str, str]:
    filters, scores = _filtered_rows(rows, bytes_per_pixel)
    height = rows.shape[0]
    adaptive = np.argmin(scores, axis=0).astype(np.uint8)
    best_fixed = int(np.argmin(scores.sum(axis=1)))
    if indexed:
        # Palette indices are identities, not sample magnitudes.  Sub/Paeth
        # residuals usually destroy repeated index runs; libpng therefore uses
        # filter 0 for indexed images.  Palette transport simplifies that raw
        # index stream directly.
        policies: dict[str, np.ndarray] = {
            "fixed-0": np.zeros(height, dtype=np.uint8),
        }
    else:
        policies = {
            "adaptive": adaptive,
            f"fixed-{best_fixed}": np.full(height, best_fixed, dtype=np.uint8),
        }
        if best_fixed != 4:
            policies["fixed-4"] = np.full(height, 4, dtype=np.uint8)
    strategies = ("default", "filtered")
    if search == "quick":
        strategies = ("default",)
    if search == "thorough":
        policies.update({
            f"fixed-{kind}": np.full(height, kind, dtype=np.uint8)
            for kind in range(5)
        })
        strategies = ("default", "filtered", "rle")
    memory_levels = (8,) if search == "quick" else (4, 5, 8, 9)

    streams = {name: _filter_stream(filters, choice) for name, choice in policies.items()}
    jobs = [
        (policy, stream, strategy, memory_level)
        for policy, stream in streams.items()
        for strategy in strategies
        for memory_level in memory_levels
    ]

    def compress_job(job: tuple[str, bytes, str, int]) -> tuple[int, bytes, str, str]:
        policy, stream, strategy, memory_level = job
        compressed = _compress(stream, level, strategy, memory_level)
        return (
            len(compressed), compressed, policy,
            f"{strategy}-m{memory_level}",
        )

    if len(jobs) > 1 and rows.nbytes >= 32_768:
        encoded = _ZLIB_EXECUTOR.map(compress_job, jobs)
    else:
        encoded = map(compress_job, jobs)
    best: tuple[int, bytes, str, str] | None = None
    for candidate in encoded:
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def _profile_chunks(info: dict[str, object], preserve: bool) -> bytes:
    if not preserve:
        return b""
    profile = info.get("icc_profile")
    if isinstance(profile, bytes) and profile:
        payload = b"ICC Profile\x00\x00" + zlib.compress(profile, 9)
        return _chunk(b"iCCP", payload)
    srgb = info.get("srgb")
    if isinstance(srgb, int):
        return _chunk(b"sRGB", bytes((max(0, min(3, srgb)),)))
    gamma = info.get("gamma")
    if isinstance(gamma, (float, int)):
        return _chunk(b"gAMA", struct.pack(">I", round(float(gamma) * 100000)))
    return b""


def _pack_indices(labels: np.ndarray, bit_depth: int) -> np.ndarray:
    labels8 = np.asarray(labels, dtype=np.uint8)
    if bit_depth == 8:
        return labels8
    height, width = labels8.shape
    per_byte = 8 // bit_depth
    padded_width = ((width + per_byte - 1) // per_byte) * per_byte
    padded = np.zeros((height, padded_width), dtype=np.uint8)
    padded[:, :width] = labels8
    packed = np.zeros((height, padded_width // per_byte), dtype=np.uint8)
    mask = (1 << bit_depth) - 1
    for offset in range(per_byte):
        shift = 8 - bit_depth * (offset + 1)
        packed |= (padded[:, offset::per_byte] & mask) << shift
    return packed


def _encode_indexed(
    labels: np.ndarray,
    palette_rgba: np.ndarray,
    info: dict[str, object],
    config: PNGConfig,
    search: str | None = None,
) -> tuple[bytes, str, str]:
    colors = len(palette_rgba)
    bit_depth = 1 if colors <= 2 else 2 if colors <= 4 else 4 if colors <= 16 else 8
    rows = _pack_indices(labels, bit_depth)
    selected_search = search or config.filter_search
    level = 1 if selected_search == "quick" else config.compression_level
    idat, policy, strategy = _best_idat(
        rows, 1, level, selected_search, indexed=True
    )
    height, width = labels.shape
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, 3, 0, 0, 0)
    palette = np.asarray(palette_rgba, dtype=np.uint8)
    body = _chunk(b"IHDR", ihdr) + _profile_chunks(info, config.preserve_color_profile)
    body += _chunk(b"PLTE", palette[:, :3].tobytes())
    alpha = palette[:, 3]
    nonopaque = np.flatnonzero(alpha != 255)
    if len(nonopaque):
        body += _chunk(b"tRNS", alpha[: int(nonopaque[-1]) + 1].tobytes())
    body += _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    return PNG_SIGNATURE + body, policy, strategy


def _truecolor_rows(rgba: np.ndarray) -> tuple[np.ndarray, int, int]:
    opaque = bool(np.all(rgba[..., 3] == 255))
    gray = bool(np.array_equal(rgba[..., 0], rgba[..., 1])) and bool(
        np.array_equal(rgba[..., 1], rgba[..., 2])
    )
    if gray and opaque:
        return rgba[..., 0], 0, 1
    if gray:
        return rgba[..., (0, 3)].reshape(rgba.shape[0], -1), 4, 2
    if opaque:
        return rgba[..., :3].reshape(rgba.shape[0], -1), 2, 3
    return rgba.reshape(rgba.shape[0], -1), 6, 4


def _encode_truecolor(
    rgba: np.ndarray,
    info: dict[str, object],
    config: PNGConfig,
) -> tuple[bytes, str, str]:
    rows, color_type, bpp = _truecolor_rows(rgba)
    idat, policy, strategy = _best_idat(
        rows, bpp, config.compression_level, config.filter_search
    )
    height, width = rgba.shape[:2]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    body = _chunk(b"IHDR", ihdr) + _profile_chunks(info, config.preserve_color_profile)
    body += _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    return PNG_SIGNATURE + body, policy, strategy


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    red, green, blue = np.moveaxis(linear, -1, 0)
    l = np.cbrt(0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue)
    m = np.cbrt(0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue)
    s = np.cbrt(0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue)
    return np.stack((
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ), axis=-1)


def _rgba_transport_features(rgba: np.ndarray) -> np.ndarray:
    """Return display color plus alpha, without charging for invisible RGB.

    Composite color and alpha together retain the premultiplied constituent:
    ``premul = composite - (1-alpha)``.  The coordinates therefore determine
    appearance on every background, while raw RGB under zero alpha is correctly
    free.  Values remain in byte scale for stable SciPy VQ behavior.
    """
    values = np.asarray(rgba, dtype=np.float32)
    alpha = values[..., 3:4] / 255.0
    composite = values[..., :3] * alpha + 255.0 * (1.0 - alpha)
    return np.concatenate((composite, values[..., 3:4]), axis=-1)


def _features_to_rgba(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    alpha_byte = np.clip(values[..., 3:4], 0.0, 255.0)
    alpha = alpha_byte / 255.0
    premultiplied = values[..., :3] - 255.0 * (1.0 - alpha)
    rgb = np.divide(
        premultiplied,
        alpha,
        out=np.zeros_like(premultiplied),
        where=alpha > (0.5 / 255.0),
    )
    return np.concatenate((
        np.clip(np.rint(rgb), 0, 255),
        np.rint(alpha_byte),
    ), axis=-1).astype(np.uint8)


def _alpha_classes(alpha: np.ndarray) -> np.ndarray:
    """Split alpha into transparent, fractional, and opaque ownership strata."""
    values = np.asarray(alpha)
    return np.where(values == 0, 0, np.where(values == 255, 2, 1)).astype(np.int8)


def _alpha_stratified_refine(
    source_rgba: np.ndarray,
    labels: np.ndarray,
    palette_rgba: np.ndarray,
    iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine an RGBA palette without moving pixels across alpha topology.

    A single four-dimensional centroid can otherwise average an opaque fill
    with its antialiased boundary.  That saves a palette entry, but turns a
    large opaque region slightly translucent and makes subsequent spatial
    transport propagate the error.  Keep zero, fractional, and full alpha as
    separate constituent classes while still optimizing color within each
    class.
    """
    if np.all(source_rgba[..., 3] == 255):
        return labels.copy(), palette_rgba.copy()

    shape = labels.shape
    flat_labels = labels.reshape(-1).astype(np.int32, copy=True)
    features = _rgba_transport_features(source_rgba).reshape(-1, 4).astype(np.float32)
    source_class = _alpha_classes(source_rgba[..., 3]).reshape(-1)
    centers_rgba = np.asarray(palette_rgba, dtype=np.uint8).copy()
    colors = len(centers_rgba)

    counts = np.column_stack(tuple(
        np.bincount(
            flat_labels[source_class == category], minlength=colors
        )
        for category in range(3)
    ))
    center_class = np.argmax(counts, axis=1).astype(np.int8)

    # Reserve a distinct node for every class in the source.  Rarest first
    # keeps the antialiased boundary from being swallowed by either endpoint.
    present = [
        category for category in range(3)
        if np.any(source_class == category)
    ]
    reserved: set[int] = set()
    for category in sorted(present, key=lambda item: np.count_nonzero(source_class == item)):
        choices = np.argsort(-counts[:, category], kind="stable")
        chosen = next((int(item) for item in choices if int(item) not in reserved), None)
        if chosen is None:
            continue
        center_class[chosen] = category
        reserved.add(chosen)

    chunk = 131072

    def assign(centers_feature: np.ndarray) -> None:
        for category in present:
            pixels = np.flatnonzero(source_class == category)
            owners = np.flatnonzero(center_class == category)
            if not len(owners):
                continue
            for start in range(0, len(pixels), chunk):
                selected = pixels[start:start + chunk]
                local = _nearest_labels(features[selected], centers_feature[owners])
                flat_labels[selected] = owners[local]

    centers_feature = _rgba_transport_features(centers_rgba).astype(np.float32)
    for _ in range(max(1, int(iterations))):
        assign(centers_feature)
        population = np.bincount(flat_labels, minlength=colors).astype(np.float64)
        sums = np.column_stack(tuple(
            np.bincount(
                flat_labels, weights=features[:, channel], minlength=colors
            )
            for channel in range(4)
        ))
        live = population > 0
        centers_feature[live] = sums[live] / population[live, None]
        centers_rgba = _features_to_rgba(centers_feature)
        centers_rgba[center_class == 0] = np.array((0, 0, 0, 0), np.uint8)
        fractional = center_class == 1
        centers_rgba[fractional, 3] = np.clip(
            centers_rgba[fractional, 3], 1, 254
        )
        centers_rgba[center_class == 2, 3] = 255
        centers_feature = _rgba_transport_features(centers_rgba).astype(np.float32)

    assign(centers_feature)
    active = np.unique(flat_labels)
    remap = np.full(colors, -1, dtype=np.int32)
    remap[active] = np.arange(len(active), dtype=np.int32)
    dense_labels = remap[flat_labels].reshape(shape)
    dense_palette = centers_rgba[active]
    unique_palette, inverse = np.unique(dense_palette, axis=0, return_inverse=True)
    return inverse[dense_labels].astype(np.int32), unique_palette


def _dense_palette(labels: np.ndarray, rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.unique(labels)
    remap = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
    remap[active] = np.arange(len(active), dtype=np.int32)
    dense_labels = remap[labels]
    flat_labels = labels.ravel()
    flat_rgba = rgba.reshape(-1, 4)
    first = np.full(int(labels.max()) + 1, len(flat_labels), dtype=np.int64)
    np.minimum.at(first, flat_labels, np.arange(len(flat_labels), dtype=np.int64))
    palette = flat_rgba[first[active]]
    # A fixed Pillow palette always has 256 slots.  When fewer colors were
    # allocated, error diffusion can select duplicate padded entries; merge
    # those identities before ownership transport and bit-depth selection.
    unique_palette, palette_inverse = np.unique(
        palette, axis=0, return_inverse=True
    )
    return palette_inverse[dense_labels].astype(np.int32), unique_palette


def _quantize(
    source_rgba: np.ndarray,
    colors: int,
    dither: str,
    quantizer: str,
    lloyd_iterations: int = 10,
    palette_edge_weight: float = 1.5,
    palette_sample_limit: int = 131072,
    palette_seed: int = 508030340,
) -> tuple[np.ndarray, np.ndarray]:
    opaque = bool(np.all(source_rgba[..., 3] == 255))
    image = Image.fromarray(source_rgba[..., :3] if opaque else source_rgba, "RGB" if opaque else "RGBA")
    methods = {
        "median-cut": Image.Quantize.MEDIANCUT,
        "maximum-coverage": Image.Quantize.MAXCOVERAGE,
        "fast-octree": Image.Quantize.FASTOCTREE,
    }
    allocation_method = (
        "median-cut" if quantizer in {"lloyd-rgb", "edge-lloyd"} else quantizer
    )
    method = methods[allocation_method]
    if not opaque and method != Image.Quantize.FASTOCTREE:
        method = Image.Quantize.FASTOCTREE
    # Palette construction and error diffusion are separate operations in
    # Pillow.  Passing ``dither`` while constructing the palette is silently
    # ignored, so first allocate deterministically and then remap through that
    # fixed palette when diffusion was requested.
    quantized = image.quantize(
        colors=max(2, min(256, colors)), method=method, dither=Image.Dither.NONE
    )
    if dither == "floyd" and opaque:
        quantized = image.quantize(
            palette=quantized, dither=Image.Dither.FLOYDSTEINBERG
        )
    labels = np.asarray(quantized, dtype=np.int32)
    displayed = np.asarray(quantized.convert("RGBA"), dtype=np.uint8)
    labels, palette = _dense_palette(labels, displayed)
    if quantizer == "lloyd-rgb":
        if opaque:
            return _lloyd_refine_rgb(source_rgba, palette, lloyd_iterations)
        labels, palette = _edge_lloyd_rgba(
            source_rgba, max(2, min(256, colors)), lloyd_iterations,
            0.0, palette_sample_limit, palette_seed,
        )
    elif quantizer == "edge-lloyd":
        if opaque:
            return _edge_lloyd_rgb(
                source_rgba,
                max(2, min(256, colors)),
                lloyd_iterations,
                palette_edge_weight,
                palette_sample_limit,
                palette_seed,
            )
        labels, palette = _edge_lloyd_rgba(
            source_rgba, max(2, min(256, colors)), lloyd_iterations,
            palette_edge_weight, palette_sample_limit, palette_seed,
        )
    if not opaque:
        return _alpha_stratified_refine(source_rgba, labels, palette)
    return labels, palette


def _lloyd_refine_rgb(
    source_rgba: np.ndarray,
    initial_palette: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine an allocated palette against every source pixel.

    The native vector-quantization kernel (or SciPy fallback) performs the
    heavy nearest-centroid step in compiled code. Centroids are updated from
    the complete image, rather than a random sample, so the result is
    deterministic and avoids the rare-color drift seen in sampled palettes.
    """
    flat = source_rgba[..., :3].reshape(-1, 3).astype(np.float32)
    centers = initial_palette[:, :3].astype(np.float32)
    labels = np.empty(len(flat), dtype=np.int32)
    chunk = 131072
    for _ in range(max(0, int(iterations))):
        for start in range(0, len(flat), chunk):
            stop = min(start + chunk, len(flat))
            labels[start:stop] = _nearest_labels(flat[start:stop], centers)
        population = np.bincount(labels, minlength=len(centers)).astype(np.float64)
        sums = np.column_stack((
            np.bincount(labels, weights=flat[:, 0], minlength=len(centers)),
            np.bincount(labels, weights=flat[:, 1], minlength=len(centers)),
            np.bincount(labels, weights=flat[:, 2], minlength=len(centers)),
        ))
        live = population > 0
        centers[live] = sums[live] / population[live, None]
    for start in range(0, len(flat), chunk):
        stop = min(start + chunk, len(flat))
        labels[start:stop] = _nearest_labels(flat[start:stop], centers)
    palette = np.column_stack((
        np.clip(np.rint(centers), 0, 255).astype(np.uint8),
        np.full(len(centers), 255, dtype=np.uint8),
    ))
    return labels.reshape(source_rgba.shape[:2]), palette


def _edge_detail(source_rgba: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "edge-aware palette allocation requires SciPy"
        ) from error
    rgb = source_rgba[..., :3].astype(np.float32) / 255.0
    edge2 = np.zeros(source_rgba.shape[:2], dtype=np.float32)
    for channel in range(3):
        gx = ndimage.sobel(rgb[..., channel], axis=1, mode="reflect")
        gy = ndimage.sobel(rgb[..., channel], axis=0, mode="reflect")
        edge2 += gx * gx + gy * gy
    edge = np.sqrt(edge2)
    scale = max(float(np.quantile(edge, 0.9)), 1e-6)
    return np.clip(edge / scale, 0.0, 4.0)


def _edge_lloyd_rgb(
    source_rgba: np.ndarray,
    colors: int,
    iterations: int,
    edge_weight: float,
    sample_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate palette capacity with weighted k-means++ and full Lloyd steps."""
    flat = source_rgba[..., :3].reshape(-1, 3).astype(np.float32)
    detail = _edge_detail(source_rgba).ravel()
    weights = (1.0 + max(0.0, float(edge_weight)) * detail).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    sample_count = min(len(flat), max(colors, int(sample_limit)))
    sample_index = rng.choice(len(flat), size=sample_count, replace=False)
    sample = flat[sample_index]
    sample_weight = weights[sample_index]

    centers = _weighted_kmeanspp(
        sample, sample_weight, colors, rng, pad_degenerate=True
    )

    labels = np.empty(len(flat), dtype=np.int32)
    chunk = 131072
    for _ in range(max(0, int(iterations))):
        for start in range(0, len(flat), chunk):
            stop = min(start + chunk, len(flat))
            labels[start:stop] = _nearest_labels(flat[start:stop], centers)
        population = np.bincount(
            labels, weights=weights, minlength=colors
        ).astype(np.float64)
        sums = np.column_stack((
            np.bincount(labels, weights=weights * flat[:, 0], minlength=colors),
            np.bincount(labels, weights=weights * flat[:, 1], minlength=colors),
            np.bincount(labels, weights=weights * flat[:, 2], minlength=colors),
        ))
        live = population > 0
        centers[live] = sums[live] / population[live, None]
    for start in range(0, len(flat), chunk):
        stop = min(start + chunk, len(flat))
        labels[start:stop] = _nearest_labels(flat[start:stop], centers)
    palette = np.column_stack((
        np.clip(np.rint(centers), 0, 255).astype(np.uint8),
        np.full(colors, 255, dtype=np.uint8),
    ))
    active = np.unique(labels)
    remap = np.full(len(palette), -1, dtype=np.int32)
    remap[active] = np.arange(len(active), dtype=np.int32)
    return remap[labels].reshape(source_rgba.shape[:2]), palette[active]


def _edge_lloyd_rgba(
    source_rgba: np.ndarray,
    colors: int,
    iterations: int,
    edge_weight: float,
    sample_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate RGBA nodes in displayed/premultiplied color and alpha space."""
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("RGBA edge Lloyd requires SciPy") from error
    feature_image = _rgba_transport_features(source_rgba)
    flat = feature_image.reshape(-1, 4).astype(np.float32)
    normalized = feature_image / 255.0
    edge2 = np.zeros(source_rgba.shape[:2], dtype=np.float32)
    for channel in range(4):
        gx = ndimage.sobel(normalized[..., channel], axis=1, mode="reflect")
        gy = ndimage.sobel(normalized[..., channel], axis=0, mode="reflect")
        edge2 += gx * gx + gy * gy
    detail = np.sqrt(edge2)
    scale = max(float(np.quantile(detail, 0.9)), 1e-6)
    weights = (
        1.0 + max(0.0, float(edge_weight)) * np.clip(detail / scale, 0.0, 4.0)
    ).ravel().astype(np.float64)

    rng = np.random.default_rng(int(seed))
    sample_count = min(len(flat), max(colors, int(sample_limit)))
    sample_index = rng.choice(len(flat), size=sample_count, replace=False)
    sample, sample_weight = flat[sample_index], weights[sample_index]
    centers = _weighted_kmeanspp(sample, sample_weight, colors, rng)

    labels = np.empty(len(flat), dtype=np.int32)
    chunk = 131072
    for _ in range(max(0, int(iterations))):
        for start in range(0, len(flat), chunk):
            stop = min(start + chunk, len(flat))
            labels[start:stop] = _nearest_labels(flat[start:stop], centers)
        population = np.bincount(labels, weights=weights, minlength=len(centers)).astype(np.float64)
        sums = np.column_stack(tuple(
            np.bincount(labels, weights=weights * flat[:, channel], minlength=len(centers))
            for channel in range(4)
        ))
        live = population > 0
        centers[live] = sums[live] / population[live, None]
    palette = _features_to_rgba(centers)
    palette_features = _rgba_transport_features(palette).astype(np.float32)
    for start in range(0, len(flat), chunk):
        stop = min(start + chunk, len(flat))
        labels[start:stop] = _nearest_labels(flat[start:stop], palette_features)
    active = np.unique(labels)
    remap = np.full(len(palette), -1, dtype=np.int32)
    remap[active] = np.arange(len(active), dtype=np.int32)
    return remap[labels].reshape(source_rgba.shape[:2]), palette[active]


def _neighbor_labels(labels: np.ndarray) -> np.ndarray:
    return np.stack((
        labels,
        np.pad(labels[:, :-1], ((0, 0), (1, 0)), mode="edge"),
        np.pad(labels[:, 1:], ((0, 0), (0, 1)), mode="edge"),
        np.pad(labels[:-1], ((1, 0), (0, 0)), mode="edge"),
        np.pad(labels[1:], ((0, 1), (0, 0)), mode="edge"),
    ), axis=0)


def _ownership_flow(
    source_rgba: np.ndarray,
    labels: np.ndarray,
    palette_rgba: np.ndarray,
    strength: float,
    iterations: int,
    edge_protection: float,
) -> np.ndarray:
    if strength <= 0.0 or iterations <= 0:
        return labels.copy()
    opaque = bool(np.all(source_rgba[..., 3] == 255))
    if opaque:
        source_space = _srgb_to_oklab(source_rgba[..., :3].astype(np.float32) / 255.0)
        palette_space = _srgb_to_oklab(palette_rgba[:, :3].astype(np.float32) / 255.0)
    else:
        source_space = _rgba_transport_features(source_rgba) / 255.0
        palette_space = _rgba_transport_features(palette_rgba) / 255.0
        source_class = _alpha_classes(source_rgba[..., 3])
        palette_class = _alpha_classes(palette_rgba[:, 3])
    gx = np.linalg.norm(source_space[:, 1:] - source_space[:, :-1], axis=2)
    gy = np.linalg.norm(source_space[1:] - source_space[:-1], axis=2)
    edge = np.zeros(labels.shape, dtype=np.float32)
    edge[:, 1:] = np.maximum(edge[:, 1:], gx)
    edge[:, :-1] = np.maximum(edge[:, :-1], gx)
    edge[1:] = np.maximum(edge[1:], gy)
    edge[:-1] = np.maximum(edge[:-1], gy)
    local_strength = float(strength) * np.exp(-max(0.0, edge_protection) * edge)

    origin = labels.copy()
    origin_value = palette_space[origin]
    origin_cost = np.sum((origin_value - source_space) ** 2, axis=2)
    current = origin.copy()
    for _ in range(iterations):
        proposals = _neighbor_labels(current)
        proposal_value = palette_space[proposals]
        color_cost = np.sum((proposal_value - source_space[None]) ** 2, axis=3)
        if not opaque:
            compatible = palette_class[proposals] == source_class[None]
            color_cost = np.where(compatible, color_cost, np.inf)
        # Charge only perceptual regret relative to the quantizer's original
        # owner.  This makes zero transport the exact identity and prevents a
        # metric-basis change from masquerading as spatial compression.
        color_cost = np.maximum(color_cost - origin_cost[None], 0.0)
        neighbors = proposals[1:]
        boundary_cost = np.sum(proposals[:, None] != neighbors[None], axis=1).astype(np.float32)
        anchor_cost = 2.0 * local_strength[None] * (proposals != origin[None])
        scores = color_cost + local_strength[None] * boundary_cost + anchor_cost
        current = np.take_along_axis(
            proposals, np.argmin(scores, axis=0)[None], axis=0
        )[0]
    return current


def _bayer_lattice(size: int = 8) -> np.ndarray:
    lattice = np.array(((0, 2), (3, 1)), dtype=np.int32)
    while lattice.shape[0] < size:
        lattice = np.block([
            [4 * lattice, 4 * lattice + 2],
            [4 * lattice + 3, 4 * lattice + 1],
        ])
    return (lattice.astype(np.float32) + 0.5) / float(lattice.size)


def _selective_ordered_diffusion(
    source_rgba: np.ndarray,
    labels: np.ndarray,
    palette_rgba: np.ndarray,
    strength: float,
    edge_barrier: float,
) -> np.ndarray:
    """Transport quantization residual on a compressible phase lattice.

    Each pixel may transfer ownership to the palette color nearest the point
    reflected across its current quantization residual.  The mixing fraction
    is the projection that reconstructs the source color in expectation.
    A Bayer phase lattice realizes that fraction, while an exponential edge
    gate prevents the residual from crossing structural boundaries.
    """
    if strength <= 0.0:
        return labels.copy()
    source = source_rgba[..., :3].astype(np.float32)
    palette = palette_rgba[:, :3].astype(np.float32)
    first = palette[labels]
    opaque = bool(np.all(source_rgba[..., 3] == 255))
    if opaque:
        eligible = np.ones(labels.shape, dtype=bool)
        palette_owners = np.arange(len(palette), dtype=np.int32)
    else:
        palette_owners = np.flatnonzero(palette_rgba[:, 3] == 255).astype(np.int32)
        eligible = (
            (source_rgba[..., 3] == 255)
            & (palette_rgba[labels, 3] == 255)
        )
    if len(palette_owners) < 2 or not np.any(eligible):
        return labels.copy()

    eligible_flat = np.flatnonzero(eligible.reshape(-1))
    reflected = np.clip(
        2.0 * source.reshape(-1, 3)[eligible_flat]
        - first.reshape(-1, 3)[eligible_flat],
        0.0,
        255.0,
    )
    second_labels = labels.reshape(-1).copy()
    chunk = 131072
    for start in range(0, len(reflected), chunk):
        stop = min(start + chunk, len(reflected))
        local = _nearest_labels(reflected[start:stop], palette[palette_owners])
        second_labels[eligible_flat[start:stop]] = palette_owners[local]
    second_labels = second_labels.reshape(labels.shape)
    second = palette[second_labels]
    direction = second - first
    residual = source - first
    denominator = np.sum(direction * direction, axis=2)
    mixture = np.zeros(labels.shape, dtype=np.float32)
    valid = (denominator > 0.0) & eligible
    mixture[valid] = np.clip(
        np.sum(residual[valid] * direction[valid], axis=1) / denominator[valid],
        0.0,
        0.5,
    )
    detail = _edge_detail(source_rgba)
    gate = np.exp(-max(0.0, float(edge_barrier)) * detail)
    probability = np.clip(float(strength) * mixture * gate, 0.0, 1.0)
    lattice = _bayer_lattice(8)
    tiled = np.tile(
        lattice,
        (
            int(np.ceil(labels.shape[0] / lattice.shape[0])),
            int(np.ceil(labels.shape[1] / lattice.shape[1])),
        ),
    )[: labels.shape[0], : labels.shape[1]]
    return np.where(eligible & (tiled < probability), second_labels, labels).astype(np.int32)


def _smooth_transition_coverage(
    reference_rgba: np.ndarray, candidate_rgba: np.ndarray
) -> float:
    """Fraction of subtle source transitions retained as non-flat output steps."""
    weights = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    reference = reference_rgba[..., :3].astype(np.float32) @ weights
    candidate = candidate_rgba[..., :3].astype(np.float32) @ weights
    source_delta = np.concatenate((
        np.abs(reference[:, 1:] - reference[:, :-1]).ravel(),
        np.abs(reference[1:] - reference[:-1]).ravel(),
    ))
    output_delta = np.concatenate((
        np.abs(candidate[:, 1:] - candidate[:, :-1]).ravel(),
        np.abs(candidate[1:] - candidate[:-1]).ravel(),
    ))
    nonzero = source_delta > (1.0 / 255.0)
    if not np.any(nonzero):
        return 1.0
    ceiling = float(np.quantile(source_delta[nonzero], 0.75))
    subtle = nonzero & (source_delta <= ceiling)
    return float(np.mean(output_delta[subtle] > (1.0 / 255.0)))


def _adjacency_matrix(labels: np.ndarray, colors: int) -> np.ndarray:
    left = np.concatenate((labels[:, :-1].ravel(), labels[:-1].ravel()))
    right = np.concatenate((labels[:, 1:].ravel(), labels[1:].ravel()))
    changed = left != right
    low = np.minimum(left[changed], right[changed]).astype(np.int64)
    high = np.maximum(left[changed], right[changed]).astype(np.int64)
    keys = low * np.int64(colors) + high
    counts = np.bincount(keys, minlength=colors * colors).reshape(colors, colors)
    adjacency = counts + counts.T
    return adjacency.astype(np.float64)


def _palette_orders(labels: np.ndarray, colors: int, enabled: bool) -> dict[str, np.ndarray]:
    identity = np.arange(colors, dtype=np.int32)
    if not enabled or colors < 3:
        return {"identity": identity}
    population = np.bincount(labels.ravel(), minlength=colors)
    adjacency = _adjacency_matrix(labels, colors)
    degree = adjacency.sum(axis=1)
    inverse_root = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    normalized = np.eye(colors) - inverse_root[:, None] * adjacency * inverse_root[None, :]
    _, vectors = np.linalg.eigh(normalized)
    fiedler = vectors[:, 1]
    spectral = np.lexsort((-population, fiedler)).astype(np.int32)
    frequency = np.argsort(-population, kind="stable").astype(np.int32)
    return {"identity": identity, "spectral": spectral, "frequency": frequency}


def _apply_order(
    labels: np.ndarray, palette: np.ndarray, order: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    return inverse[labels], palette[order]


def _encode_best_palette_order(
    labels: np.ndarray,
    palette: np.ndarray,
    info: dict[str, object],
    config: PNGConfig,
) -> tuple[bytes, np.ndarray, np.ndarray, str, str, str]:
    orders = _palette_orders(labels, len(palette), config.palette_transport)
    quick: list[tuple[int, str, bytes, np.ndarray, np.ndarray, str, str]] = []
    for name, order in orders.items():
        ordered_labels, ordered_palette = _apply_order(labels, palette, order)
        data, policy, strategy = _encode_indexed(
            ordered_labels, ordered_palette, info, config, search="quick"
        )
        quick.append((
            len(data), name, data, ordered_labels, ordered_palette, policy, strategy
        ))
    quick.sort(key=lambda item: item[0])
    final: list[tuple[int, str, bytes, np.ndarray, np.ndarray, str, str]] = []
    for _, candidate_name, _, candidate_labels, candidate_palette, _, _ in quick[:2]:
        candidate_data, candidate_policy, candidate_strategy = _encode_indexed(
            candidate_labels, candidate_palette, info, config,
            search=config.filter_search,
        )
        final.append((
            len(candidate_data), candidate_name, candidate_data,
            candidate_labels, candidate_palette,
            candidate_policy, candidate_strategy,
        ))
    _, name, data, ordered_labels, ordered_palette, policy, strategy = min(
        final, key=lambda item: item[0]
    )

    # Pillow/libpng is a useful terminal-codec lower bound.  The ownership
    # search remains ours; this candidate merely ensures that a custom DEFLATE
    # choice never makes the transported pixels larger than the stock encoder.
    pillow_image = Image.fromarray(ordered_labels.astype(np.uint8), mode="P")
    rgb_palette = ordered_palette[:, :3].astype(np.uint8).ravel().tolist()
    pillow_image.putpalette(rgb_palette + [0] * (768 - len(rgb_palette)))
    alpha = ordered_palette[:, 3].astype(np.uint8)
    nonopaque = np.flatnonzero(alpha != 255)
    save_options: dict[str, object] = {
        "format": "PNG",
        "optimize": True,
        "compress_level": config.compression_level,
    }
    if len(nonopaque):
        save_options["transparency"] = alpha[: int(nonopaque[-1]) + 1].tobytes()
    if config.preserve_color_profile and isinstance(info.get("icc_profile"), bytes):
        save_options["icc_profile"] = info["icc_profile"]
    buffer = BytesIO()
    pillow_image.save(buffer, **save_options)
    pillow_data = buffer.getvalue()
    if len(pillow_data) < len(data):
        data, policy, strategy = pillow_data, "pillow-indexed", "default"
    return data, ordered_labels, ordered_palette, name, policy, strategy


def _metric_rgba(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    if np.all(reference[..., 3] == 255) and np.all(candidate[..., 3] == 255):
        return image_metrics(reference[..., :3], candidate[..., :3])
    alpha_ref = reference[..., 3:4].astype(np.float64) / 255.0
    alpha_out = candidate[..., 3:4].astype(np.float64) / 255.0
    ref = reference[..., :3] * alpha_ref + 255.0 * (1.0 - alpha_ref)
    out = candidate[..., :3] * alpha_out + 255.0 * (1.0 - alpha_out)
    return image_metrics(np.rint(ref).astype(np.uint8), np.rint(out).astype(np.uint8))


def _candidate(
    reference: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    info: dict[str, object],
    config: PNGConfig,
    colors: int,
    quantizer: str,
    palette_edge_weight: float,
    dither: str,
    diffusion_strength: float,
    strength: float,
) -> PNGCandidate:
    started = perf_counter()
    data, labels, palette, order, policy, strategy = _encode_best_palette_order(
        labels, palette, info, config
    )
    displayed = palette[labels]
    ssim, psnr, edge_psnr = _metric_rgba(reference, displayed)
    return PNGCandidate(
        colors=len(palette),
        quantizer=quantizer,
        palette_edge_weight=palette_edge_weight,
        dither=dither,
        diffusion_strength=diffusion_strength,
        ownership_strength=strength,
        palette_order=order,
        filter_policy=policy,
        zlib_strategy=strategy,
        size=len(data),
        ssim=ssim,
        psnr_db=psnr,
        edge_psnr_db=edge_psnr,
        smooth_transition_coverage=_smooth_transition_coverage(reference, displayed),
        elapsed_seconds=perf_counter() - started,
        data=data,
        rgba=displayed,
    )


def _better(candidate: PNGCandidate, incumbent: PNGCandidate, config: PNGConfig) -> bool:
    target = max(0, config.target_bytes)
    minimum = max(0.0, config.minimum_ssim)
    candidate_quality = candidate.ssim >= minimum
    incumbent_quality = incumbent.ssim >= minimum
    if candidate_quality != incumbent_quality:
        return candidate_quality
    if target:
        candidate_rate = candidate.size <= target
        incumbent_rate = incumbent.size <= target
        if candidate_rate != incumbent_rate:
            return candidate_rate
        if candidate_rate:
            # Keep SSIM dominant, but do not make it lexicographic.  Inside a
            # tightly bounded SSIM band, measured color and edge fidelity can
            # identify a cleaner structural representation (notably thin-line
            # and texture palettes) that pure SSIM would discard.
            if candidate.ssim < incumbent.ssim - 2.5e-4:
                return False
            if candidate.ssim > incumbent.ssim + 2.5e-4:
                return candidate.ssim > incumbent.ssim
            candidate_fidelity = (
                1000.0 * candidate.ssim
                + 0.2 * candidate.psnr_db
                + 0.5 * candidate.edge_psnr_db
            )
            incumbent_fidelity = (
                1000.0 * incumbent.ssim
                + 0.2 * incumbent.psnr_db
                + 0.5 * incumbent.edge_psnr_db
            )
            if abs(candidate_fidelity - incumbent_fidelity) > 1e-9:
                return candidate_fidelity > incumbent_fidelity
            return candidate.size < incumbent.size
        return candidate.size < incumbent.size
    if minimum and candidate_quality:
        return candidate.size < incumbent.size
    if abs(candidate.ssim - incumbent.ssim) > 1e-9:
        return candidate.ssim > incumbent.ssim
    return candidate.size < incumbent.size


def _winner(candidates: Iterable[PNGCandidate], config: PNGConfig) -> PNGCandidate:
    iterator = iter(candidates)
    winner = next(iterator)
    for candidate in iterator:
        if _better(candidate, winner, config):
            winner = candidate
    return winner


def _color_schedule(config: PNGConfig) -> list[int]:
    if config.colors:
        return [max(2, min(256, int(config.colors)))]
    floor = max(2, min(256, int(config.minimum_colors)))
    coarse = [256, 128, 64, 32, 16, 8, 4, 2]
    return [count for count in coarse if count >= floor] or [floor]


def _lossless_candidate(
    source: Path,
    image: Image.Image,
    rgba: np.ndarray,
    info: dict[str, object],
    config: PNGConfig,
) -> PNGCandidate:
    started = perf_counter()
    choices: list[tuple[bytes, str, str]] = []
    if config.filter_search == "thorough":
        encoded, policy, strategy = _encode_truecolor(rgba, info, config)
        choices.append((encoded, policy, strategy))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=config.compression_level)
    choices.append((buffer.getvalue(), "pillow", "default"))
    if source.suffix.lower() == ".png":
        original = source.read_bytes()
        choices.append((original, "source", "source"))

    colors = image.getcolors(maxcolors=257)
    order = "truecolor"
    if colors is not None:
        flat = rgba.reshape(-1, 4)
        palette, inverse = np.unique(flat, axis=0, return_inverse=True)
        labels = inverse.reshape(rgba.shape[:2]).astype(np.int32)
        indexed, _, _, palette_order, indexed_policy, indexed_strategy = _encode_best_palette_order(
            labels, palette.astype(np.uint8), info, config
        )
        choices.append((indexed, indexed_policy, indexed_strategy))
        order = palette_order if len(indexed) <= min(len(item[0]) for item in choices[:-1]) else order
    data, policy, strategy = min(choices, key=lambda item: len(item[0]))
    return PNGCandidate(
        colors=len(colors) if colors is not None else 0,
        quantizer="lossless",
        palette_edge_weight=0.0,
        dither="none",
        diffusion_strength=0.0,
        ownership_strength=0.0,
        palette_order=order,
        filter_policy=policy,
        zlib_strategy=strategy,
        size=len(data),
        ssim=1.0,
        psnr_db=float("inf"),
        edge_psnr_db=float("inf"),
        smooth_transition_coverage=1.0,
        elapsed_seconds=perf_counter() - started,
        data=data,
        rgba=rgba.copy(),
    )


def optimize_png(
    source: str | Path,
    output: str | Path | None = None,
    config: PNGConfig | None = None,
    report: str | Path | None = None,
    progress: Progress | None = None,
) -> PNGOptimizationResult:
    """Optimize ``source`` and optionally write a standards-compatible PNG."""
    started = perf_counter()
    source_path = Path(source)
    config = config or PNGConfig()
    with Image.open(source_path) as opened:
        source_mode = opened.mode
        info = dict(opened.info)
        image = opened.copy()
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    candidates: list[PNGCandidate] = []
    schedule = _color_schedule(config)
    # The guided path may add palette-rate refinement, Lloyd, ownership, and
    # diffusion probes after the coarse schedule.  Keep GUI progress monotone
    # instead of letting those probes run beyond the advertised budget.
    total = 1 if config.lossless else len(schedule) + 38
    completed = 0

    def notify(message: str) -> None:
        if progress is not None:
            progress(completed, total, message)

    if config.lossless:
        notify("optimizing lossless scanlines")
        candidates.append(_lossless_candidate(source_path, image, rgba, info, config))
        completed = 1
    else:
        quantized_cache: dict[tuple[int, str, str, float], tuple[np.ndarray, np.ndarray]] = {}
        base_quantizer = "median-cut" if config.quantizer == "auto" else config.quantizer

        def evaluate(
            colors: int,
            dither: str,
            strength: float,
            quantizer: str | None = None,
            diffusion_strength: float = 0.0,
            edge_weight: float | None = None,
        ) -> PNGCandidate:
            nonlocal completed
            selected_quantizer = quantizer or base_quantizer
            selected_edge_weight = (
                config.palette_edge_weight if edge_weight is None else float(edge_weight)
            )
            palette_dither = "floyd" if dither == "floyd" else "none"
            key = (colors, palette_dither, selected_quantizer, selected_edge_weight)
            if key not in quantized_cache:
                quantized_cache[key] = _quantize(
                    rgba,
                    colors,
                    palette_dither,
                    selected_quantizer,
                    config.lloyd_iterations,
                    selected_edge_weight,
                    config.palette_sample_limit,
                    config.palette_seed,
                )
            labels, palette = quantized_cache[key]
            flowed = _ownership_flow(
                rgba, labels, palette, strength,
                config.ownership_iterations, config.edge_protection,
            )
            if dither == "selective":
                flowed = _selective_ordered_diffusion(
                    rgba,
                    flowed,
                    palette,
                    diffusion_strength,
                    config.diffusion_edge_barrier,
                )
            notify(
                f"{colors} colors, ownership {strength:g}, "
                f"{dither} diffusion {diffusion_strength:g}, edge {selected_edge_weight:g}"
            )
            candidate = _candidate(
                rgba, flowed, palette, info, config, colors,
                selected_quantizer, selected_edge_weight, dither,
                diffusion_strength, strength
            )
            candidates.append(candidate)
            completed += 1
            return candidate

        base: list[PNGCandidate] = []
        for colors in schedule:
            candidate = evaluate(colors, "none", 0.0)
            base.append(candidate)
            if config.target_bytes and candidate.size <= config.target_bytes:
                break
            if config.minimum_ssim and candidate.ssim < config.minimum_ssim:
                break

        current = _winner(base, config)
        refinement_colors = {current.colors}
        ordered_counts = [candidate.colors for candidate in base]
        current_index = ordered_counts.index(current.colors)
        if current_index > 0:
            high, low = ordered_counts[current_index - 1], current.colors
            midpoint = max(low + 1, int(round(np.sqrt(high * low))))
            if midpoint < high:
                evaluate(midpoint, "none", 0.0)
                refinement_colors.add(midpoint)

        current = _winner(candidates, config)
        if config.quantizer == "auto":
            evaluate(current.colors, "none", 0.0, "edge-lloyd")
            current = _winner(candidates, config)
            if config.target_bytes:
                # Palette cardinality is itself a rate coordinate.  A single
                # geometric midpoint left large holes (often 20-40%) below a
                # requested byte budget, making matched-rate comparisons
                # inconclusive.  Bracket the target in the final Lloyd family
                # and resolve the integer crossing with a bounded binary trace.
                tested = {
                    candidate.colors for candidate in candidates
                    if candidate.quantizer == "edge-lloyd"
                    and candidate.palette_edge_weight == config.palette_edge_weight
                    and candidate.dither == "none"
                    and candidate.ownership_strength == 0.0
                }
                for _ in range(6):
                    family = [
                        candidate for candidate in candidates
                        if candidate.quantizer == "edge-lloyd"
                        and candidate.palette_edge_weight == config.palette_edge_weight
                        and candidate.dither == "none"
                        and candidate.ownership_strength == 0.0
                    ]
                    under = [
                        candidate for candidate in family
                        if candidate.size <= config.target_bytes
                    ]
                    if not under:
                        high = min(family, key=lambda candidate: candidate.colors).colors
                        proposal = max(2, high // 2)
                    else:
                        low = max(under, key=lambda candidate: candidate.colors).colors
                        if low >= 256:
                            break
                        over = [
                            candidate for candidate in family
                            if candidate.colors > low
                            and candidate.size > config.target_bytes
                        ]
                        high = (
                            min(over, key=lambda candidate: candidate.colors).colors
                            if over else min(256, max(low + 1, low * 2))
                        )
                        if high not in tested:
                            proposal = high
                        elif high - low > 1:
                            proposal = (low + high) // 2
                        else:
                            break
                    if proposal in tested:
                        break
                    evaluate(proposal, "none", 0.0, "edge-lloyd")
                    tested.add(proposal)
                current = _winner(candidates, config)
                if np.all(rgba[..., 3] == 255):
                    # The correct edge prior is content-dependent.  Probe a
                    # small fixed atlas at the resolved rate rather than
                    # pretending one global weight serves lines, texture, and
                    # flat graphics equally.  Also admit unweighted full Lloyd,
                    # which is often the smoother phase/frequency endpoint.
                    resolved_colors = current.colors
                    atlas: list[tuple[str, float]] = []
                    for edge_weight in (0.0, 0.5, 1.0, 2.5):
                        if edge_weight != config.palette_edge_weight:
                            probe = evaluate(
                                resolved_colors, "none", 0.0, "edge-lloyd",
                                edge_weight=edge_weight,
                            )
                            atlas.append(("edge-lloyd", edge_weight))
                            predicted = int(np.clip(round(
                                probe.colors * config.target_bytes / max(probe.size, 1)
                            ), 2, 256))
                            if predicted != probe.colors:
                                evaluate(
                                    predicted, "none", 0.0, "edge-lloyd",
                                    edge_weight=edge_weight,
                                )
                    probe = evaluate(
                        resolved_colors, "none", 0.0, "lloyd-rgb",
                        edge_weight=0.0,
                    )
                    atlas.append(("lloyd-rgb", 0.0))
                    predicted = int(np.clip(round(
                        probe.colors * config.target_bytes / max(probe.size, 1)
                    ), 2, 256))
                    if predicted != probe.colors:
                        evaluate(
                            predicted, "none", 0.0, "lloyd-rgb",
                            edge_weight=0.0,
                        )
                    # One proportional step can miss when index entropy, not
                    # PLTE bytes, controls rate (thin lines are the extreme
                    # case).  Use the measured bytes/color secant for at most
                    # two corrections per family; once an under/over bracket
                    # exists, integer bisection resolves it.
                    for quantizer_name, edge_weight in atlas:
                        for _ in range(2):
                            family = [
                                candidate for candidate in candidates
                                if candidate.quantizer == quantizer_name
                                and candidate.palette_edge_weight == edge_weight
                                and candidate.dither == "none"
                                and candidate.ownership_strength == 0.0
                            ]
                            tested_counts = {candidate.colors for candidate in family}
                            under = [candidate for candidate in family if candidate.size <= config.target_bytes]
                            over = [candidate for candidate in family if candidate.size > config.target_bytes]
                            proposal: int | None = None
                            if under and over:
                                low = max(under, key=lambda candidate: candidate.colors)
                                above = [candidate for candidate in over if candidate.colors > low.colors]
                                if above:
                                    high = min(above, key=lambda candidate: candidate.colors)
                                    if high.colors - low.colors > 1:
                                        proposal = (low.colors + high.colors) // 2
                            elif under:
                                low = max(under, key=lambda candidate: candidate.colors)
                                if low.colors < 256 and low.size < config.target_bytes:
                                    proposal = min(256, max(
                                        low.colors + 1,
                                        round(low.colors * config.target_bytes / max(low.size, 1)),
                                    ))
                            elif len(over) >= 2:
                                ordered = sorted(over, key=lambda candidate: candidate.colors)
                                first, second = ordered[0], ordered[1]
                                slope = (second.size - first.size) / max(second.colors - first.colors, 1)
                                if slope > 0.5:
                                    proposal = round(
                                        first.colors - (first.size - config.target_bytes) / slope
                                    )
                                else:
                                    proposal = first.colors - 1
                            elif over:
                                first = min(over, key=lambda candidate: candidate.colors)
                                proposal = round(
                                    first.colors * config.target_bytes / max(first.size, 1)
                                )
                            if proposal is None:
                                break
                            proposal = int(np.clip(proposal, 2, 256))
                            if proposal in tested_counts:
                                break
                            evaluate(
                                proposal, "none", 0.0, quantizer_name,
                                edge_weight=edge_weight,
                            )
                    current = _winner(candidates, config)
        strengths = (
            [max(0.0, config.ownership_strength)]
            if config.ownership_strength >= 0.0
            else [0.0005]
        )
        for strength in strengths:
            if strength > 0.0:
                probe = evaluate(
                    current.colors, "none", strength, current.quantizer,
                    edge_weight=current.palette_edge_weight,
                )
                if config.target_bytes:
                    for _ in range(2):
                        if probe.size <= config.target_bytes and probe.colors < 256:
                            proposal = min(256, max(
                                probe.colors + 1,
                                round(
                                    probe.colors * config.target_bytes
                                    / max(probe.size, 1)
                                ),
                            ))
                        elif probe.size > config.target_bytes and probe.colors > 2:
                            proposal = max(2, min(
                                probe.colors - 1,
                                round(
                                    probe.colors * config.target_bytes
                                    / max(probe.size, 1)
                                ),
                            ))
                        else:
                            break
                        if proposal == probe.colors:
                            break
                        probe = evaluate(
                            proposal, "none", strength, current.quantizer,
                            edge_weight=current.palette_edge_weight,
                        )

        current = _winner(candidates, config)
        if config.dither == "floyd":
            evaluate(
                current.colors, "floyd", current.ownership_strength,
                current.quantizer, edge_weight=current.palette_edge_weight,
            )
        elif config.dither == "selective":
            evaluate(
                current.colors, "selective", current.ownership_strength,
                current.quantizer, config.diffusion_strength,
                edge_weight=current.palette_edge_weight,
            )
        elif config.dither == "auto":
            for diffusion_strength in (0.25, 0.5, 0.75, 1.0):
                evaluate(
                    current.colors, "selective", current.ownership_strength,
                    current.quantizer, diffusion_strength,
                    edge_weight=current.palette_edge_weight,
                )

    winner_pool = candidates
    if config.dither in {"floyd", "selective"}:
        requested = [
            candidate for candidate in candidates
            if candidate.dither == config.dither
        ]
        if requested:
            winner_pool = requested
    winner = _winner(winner_pool, config)
    result = PNGOptimizationResult(
        source=source_path,
        source_bytes=source_path.stat().st_size,
        source_mode=source_mode,
        shape=rgba.shape,
        config=config,
        winner=winner,
        candidates=candidates,
        elapsed_seconds=perf_counter() - started,
    )
    if output is not None:
        result.save(output, report)
    if progress is not None:
        progress(total, total, f"selected {winner.size:,} bytes")
    return result


def compare_pngs(reference: str | Path, candidate: str | Path) -> dict[str, object]:
    reference_path = Path(reference)
    candidate_path = Path(candidate)
    with Image.open(reference_path) as image:
        first = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    with Image.open(candidate_path) as image:
        second = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if first.shape != second.shape:
        raise ValueError(f"image shapes differ: {first.shape} != {second.shape}")
    ssim, psnr, edge_psnr = _metric_rgba(first, second)
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_bytes": reference_path.stat().st_size,
        "candidate_bytes": candidate_path.stat().st_size,
        "ssim": ssim,
        "psnr_db": psnr,
        "edge_psnr_db": edge_psnr,
        "smooth_transition_coverage": _smooth_transition_coverage(first, second),
    }
