"""Shared analysis and optimization engine for the manual JPEG laboratory.

The encoder is deliberately ordinary JPEG (Pillow/libjpeg).  The experimental
part is a reversible-basis *analysis* followed by a distortion-controlled
projection in that basis.  No decoder side information is hidden in the file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label as connected_components, sobel

try:
    from bfft.vision import (
        block_dct8_native as _native_block_dct,
        image_metrics_native as _native_image_metrics,
        inverse_block_dct8_native as _native_inverse_block_dct,
    )
except (ImportError, OSError):  # Portable/source-only installations.
    _native_block_dct = None
    _native_image_metrics = None
    _native_inverse_block_dct = None


LUMA_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

CHROMA_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)

ZIGZAG = sorted(
    ((u, v) for u in range(8) for v in range(8)),
    key=lambda uv: (sum(uv), -uv[0] if sum(uv) % 2 else uv[0]),
)


def _dct_matrix() -> np.ndarray:
    x = np.arange(8, dtype=np.float64)
    u = x[:, None]
    matrix = np.cos((2.0 * x[None, :] + 1.0) * u * np.pi / 16.0)
    matrix[0] *= 1.0 / math.sqrt(2.0)
    return matrix * 0.5


DCT8 = _dct_matrix()


@dataclass(frozen=True)
class JPEGConfig:
    quality: int = 80
    subsampling: int = 2
    progressive: bool = True
    optimize: bool = True
    cartoon_sigma: float = 1.2
    chroma_projection: float = 0.0
    luma_texture_shrink: float = 0.0
    phase_degrees: float = 0.0
    region_threshold: float = 0.58


@dataclass
class Candidate:
    config: JPEGConfig
    size_bytes: int
    ssim: float
    psnr_db: float
    edge_psnr_db: float
    objective: float
    data: bytes | None = None

    def report(self) -> dict:
        value = asdict(self)
        value.pop("data", None)
        return value


@dataclass
class OptimizationResult:
    source: Path
    output: Path
    source_bytes: int
    source_quality: int
    target_bytes: int
    best: Candidate
    frontier: list[Candidate]
    regions: int
    evaluations: int = 0
    search_strategy: str = "frontier_trace"
    metric_backend: str = "scipy"

    def report(self) -> dict:
        return {
            "source": str(self.source),
            "output": str(self.output),
            "source_bytes": self.source_bytes,
            "source_quality": self.source_quality,
            "target_bytes": self.target_bytes,
            "target_met": self.best.size_bytes <= self.target_bytes,
            "best": self.best.report(),
            "frontier": [candidate.report() for candidate in self.frontier],
            "regions": self.regions,
            "evaluations": self.evaluations,
            "search_strategy": self.search_strategy,
            "metric_backend": self.metric_backend,
            "analysis_dct_backend": (
                "native_cpp" if _native_block_dct is not None else "numpy"
            ),
            "region_signature_backend": "numpy_exact",
        }


@dataclass
class PreprocessResult:
    rgb: np.ndarray
    labels: np.ndarray
    cartoon: np.ndarray
    texture: np.ndarray
    preferred_angle: np.ndarray


@dataclass
class PreprocessBasis:
    ycc: np.ndarray
    cartoon: np.ndarray
    texture: np.ndarray
    block_labels: np.ndarray
    pixel_labels: np.ndarray
    preferred_angle: np.ndarray
    flat_luma_weight: np.ndarray
    smooth_chroma_texture: np.ndarray


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def infer_source_quality(path: str | Path) -> int:
    """Return the closest standard libjpeg luma table quality."""
    with Image.open(path) as image:
        tables = getattr(image, "quantization", None) or {}
    if 0 not in tables:
        return 75
    observed = np.asarray(tables[0], dtype=np.float64).reshape(8, 8)
    errors = [float(np.mean(np.abs(observed - quality_table(q)))) for q in range(1, 101)]
    return int(np.argmin(errors)) + 1


def rgb_to_ycc(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    r, g, b = np.moveaxis(value, -1, 0)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack((y, cb, cr), axis=-1)


def ycc_to_rgb(ycc: np.ndarray) -> np.ndarray:
    y, cb, cr = np.moveaxis(np.asarray(ycc, dtype=np.float64), -1, 0)
    cb = cb - 128.0
    cr = cr - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack((r, g, b), axis=-1), 0.0, 255.0)


def _pad8(channel: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = channel.shape
    padded = np.pad(
        channel,
        ((0, (-height) % 8), (0, (-width) % 8)),
        mode="edge",
    )
    return padded, (height, width)


def _block_dct_numpy(channel: np.ndarray) -> np.ndarray:
    padded, _ = _pad8(channel)
    bh, bw = padded.shape[0] // 8, padded.shape[1] // 8
    blocks = padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    return np.einsum("ui,abij,vj->abuv", DCT8, blocks - 128.0, DCT8)


def block_dct(channel: np.ndarray) -> np.ndarray:
    if _native_block_dct is not None:
        native = _native_block_dct(channel, DCT8, threads=0)
        if native is not None:
            return native
    return _block_dct_numpy(channel)


def inverse_block_dct(coefficients: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if _native_inverse_block_dct is not None:
        native = _native_inverse_block_dct(coefficients, shape, DCT8, threads=0)
        if native is not None:
            return native
    blocks = np.einsum("ui,abuv,vj->abij", DCT8, coefficients, DCT8)
    image = blocks.transpose(0, 2, 1, 3).reshape(
        coefficients.shape[0] * 8, coefficients.shape[1] * 8
    ) + 128.0
    return image[: shape[0], : shape[1]]


def quality_table(quality: int, chroma: bool = False) -> np.ndarray:
    q = min(100, max(1, int(quality)))
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    base = CHROMA_Q if chroma else LUMA_Q
    return np.clip(np.floor((base * scale + 50.0) / 100.0), 1.0, 255.0)


def _expand_blocks(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    expanded = np.repeat(np.repeat(values, 8, axis=0), 8, axis=1)
    return expanded[: shape[0], : shape[1]]


def _region_labels(ycc: np.ndarray, sigma: float, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Connected block regions from v3-like cartoon ownership and texture phase."""
    cartoon = gaussian_filter(ycc, sigma=(max(sigma, 0.05), max(sigma, 0.05), 0.0))
    texture = ycc - cartoon
    # Region signatures are a discrete bifurcation: a coefficient differing
    # by one final floating-point bit can move a phase angle across a bin and
    # relabel a component.  Keep the historical NumPy accumulation here for
    # byte-identical optimization; interactive/explicit DCT routes use native.
    y_dct = _block_dct_numpy(ycc[..., 0])
    cb_dct = _block_dct_numpy(texture[..., 1])
    cr_dct = _block_dct_numpy(texture[..., 2])
    low = np.sum(y_dct[..., :3, :3] ** 2, axis=(-2, -1))
    high = np.sum(y_dct[..., 3:, 3:] ** 2, axis=(-2, -1))
    detail = np.log1p(high) - np.log1p(low)
    detail_cut = np.quantile(detail, np.clip(threshold, 0.05, 0.95))
    texture_class = detail > detail_cut

    cross = np.sum(cb_dct[..., 1:, 1:] * cr_dct[..., 1:, 1:], axis=(-2, -1))
    delta = np.sum(cb_dct[..., 1:, 1:] ** 2 - cr_dct[..., 1:, 1:] ** 2, axis=(-2, -1))
    angle = 0.5 * np.arctan2(2.0 * cross, delta + 1e-12)
    angle_bin = np.mod(np.floor((angle + np.pi / 2) * 4 / np.pi), 4).astype(np.int32)
    signature = angle_bin + 4 * texture_class.astype(np.int32)

    labels = np.full(signature.shape, -1, dtype=np.int32)
    next_label = 0
    for value in range(8):
        components, count = connected_components(signature == value)
        mask = components > 0
        labels[mask] = components[mask] + next_label - 1
        next_label += int(count)
    return labels, angle


def _prepare_preprocess_basis(rgb: np.ndarray, config: JPEGConfig) -> PreprocessBasis:
    """Compute the representation-independent ownership atlas once."""
    ycc = rgb_to_ycc(rgb)
    sigma = max(float(config.cartoon_sigma), 0.05)
    cartoon = gaussian_filter(ycc, sigma=(sigma, sigma, 0.0))
    texture = ycc - cartoon
    block_labels, preferred = _region_labels(ycc, sigma, config.region_threshold)
    pixel_labels = _expand_blocks(block_labels, ycc.shape[:2]).astype(np.int32)
    gradient = np.hypot(
        sobel(cartoon[..., 0], axis=0), sobel(cartoon[..., 0], axis=1)
    )
    scale = np.quantile(gradient, 0.7) + 1e-9
    return PreprocessBasis(
        ycc=ycc,
        cartoon=cartoon,
        texture=texture,
        block_labels=block_labels,
        pixel_labels=pixel_labels,
        preferred_angle=preferred,
        flat_luma_weight=np.exp(-((gradient / scale) ** 2)),
        smooth_chroma_texture=gaussian_filter(
            texture[..., 1:3], sigma=(0.75, 0.75, 0.0)
        ),
    )


def _project_preprocess_ycc(
    basis: PreprocessBasis, config: JPEGConfig
) -> np.ndarray:
    """Apply projection controls without materializing diagnostic views."""
    ycc = basis.ycc
    cartoon = basis.cartoon
    texture = basis.texture
    block_labels = basis.block_labels
    pixel_labels = basis.pixel_labels

    result = ycc.copy()
    luma_amount = np.clip(config.luma_texture_shrink, 0.0, 1.0)
    if luma_amount > 0.0:
        # Preserve edge texture; simplify only the low-gradient ownership interiors.
        result[..., 0] -= luma_amount * basis.flat_luma_weight * texture[..., 0]

    amount = np.clip(config.chroma_projection, 0.0, 1.0)
    if amount > 0.0:
        chroma_texture = texture[..., 1:3]
        smooth_texture = basis.smooth_chroma_texture
        projected = chroma_texture.copy()
        global_phase = math.radians(float(config.phase_degrees))
        # Accumulate every region covariance in one linear pass.  The former
        # region loop built a full-image Boolean mask for every owner, making
        # this O(pixels * regions) on detailed images.
        owner = pixel_labels.ravel()
        regions = int(block_labels.max()) + 1
        cb = chroma_texture[..., 0].ravel()
        cr = chroma_texture[..., 1].ravel()
        counts = np.bincount(owner, minlength=regions)
        covariance_cb = np.bincount(owner, weights=cb * cb, minlength=regions)
        covariance_cross = np.bincount(owner, weights=cb * cr, minlength=regions)
        covariance_cr = np.bincount(owner, weights=cr * cr, minlength=regions)
        theta = 0.5 * np.arctan2(
            2.0 * covariance_cross,
            covariance_cb - covariance_cr + 1e-12,
        ) + global_phase
        minor_cb = -np.sin(theta)
        minor_cr = np.cos(theta)
        valid = counts >= 8
        minor_cb[~valid] = 0.0
        minor_cr[~valid] = 0.0
        local_cb = minor_cb[owner]
        local_cr = minor_cr[owner]
        raw_minor = cb * local_cb + cr * local_cr
        smooth_cb = smooth_texture[..., 0].ravel()
        smooth_cr = smooth_texture[..., 1].ravel()
        smooth_minor = smooth_cb * local_cb + smooth_cr * local_cr
        displacement = amount * (raw_minor - smooth_minor)
        projected[..., 0] -= (displacement * local_cb).reshape(pixel_labels.shape)
        projected[..., 1] -= (displacement * local_cr).reshape(pixel_labels.shape)
        result[..., 1:3] = cartoon[..., 1:3] + projected

    return result


def _apply_preprocess_basis(
    basis: PreprocessBasis, config: JPEGConfig
) -> PreprocessResult:
    """Apply projection controls and materialize the interactive diagnostics."""
    return PreprocessResult(
        rgb=ycc_to_rgb(_project_preprocess_ycc(basis, config)),
        labels=basis.pixel_labels,
        cartoon=ycc_to_rgb(basis.cartoon),
        texture=basis.texture,
        preferred_angle=_expand_blocks(basis.preferred_angle, basis.ycc.shape[:2]),
    )


def preprocess(rgb: np.ndarray, config: JPEGConfig) -> PreprocessResult:
    """Project texture in locally aligned YCbCr bases and return standard RGB."""
    return _apply_preprocess_basis(_prepare_preprocess_basis(rgb, config), config)


def _subsample_preview(ycc: np.ndarray, subsampling: int) -> np.ndarray:
    result = ycc.copy()
    if subsampling == 0:
        return result
    sy, sx = (1, 2) if subsampling == 1 else (2, 2)
    for channel in (1, 2):
        plane = ycc[..., channel]
        height = plane.shape[0] - plane.shape[0] % sy
        width = plane.shape[1] - plane.shape[1] % sx
        core = plane[:height, :width]
        small = core.reshape(height // sy, sy, width // sx, sx).mean(axis=(1, 3))
        result[:height, :width, channel] = np.repeat(np.repeat(small, sy, 0), sx, 1)
    return result


def _coefficient_preview(coefficients: np.ndarray, quantized: bool = False) -> np.ndarray:
    value = np.asarray(coefficients, dtype=np.float64)
    if quantized:
        visible = np.sign(value) * np.log1p(np.abs(value))
    else:
        visible = np.log1p(np.abs(value))
    scale = np.quantile(np.abs(visible), 0.995) + 1e-9
    visible = np.clip(visible / scale, -1.0, 1.0)
    if quantized:
        visible = 127.5 + 127.5 * visible
    else:
        visible = 255.0 * np.maximum(visible, 0.0)
    return visible.transpose(0, 2, 1, 3).reshape(
        value.shape[0] * 8, value.shape[1] * 8
    )


def _entropy_preview(quantized: np.ndarray) -> np.ndarray:
    bh, bw = quantized.shape[:2]
    run_cost = np.zeros((bh, bw), dtype=np.float64)
    nonzero = np.zeros((bh, bw), dtype=np.float64)
    for rank, (u, v) in enumerate(ZIGZAG):
        active = quantized[..., u, v] != 0
        nonzero += active
        run_cost += active * (1.0 + np.log2(1.0 + np.abs(quantized[..., u, v])))
        if rank:
            run_cost += active * 0.08 * rank
    composite = np.stack((run_cost, nonzero, np.log1p(np.abs(quantized[..., 0, 0]))), axis=-1)
    composite /= np.quantile(composite, 0.99, axis=(0, 1), keepdims=True) + 1e-9
    return np.uint8(255.0 * np.clip(_expand_blocks(composite, (bh * 8, bw * 8)), 0.0, 1.0))


def analyze_five_stages(rgb: np.ndarray, config: JPEGConfig) -> dict[str, np.ndarray]:
    """Return the five forward-JPEG stages plus learned structure overlays."""
    prep = preprocess(rgb, config)
    ycc = rgb_to_ycc(prep.rgb)
    sampled = _subsample_preview(ycc, config.subsampling)
    coefficients = [block_dct(sampled[..., channel]) for channel in range(3)]
    quantized = [
        np.rint(coefficients[channel] / quality_table(config.quality, channel > 0))
        for channel in range(3)
    ]
    dct_views = [_coefficient_preview(value) for value in coefficients]
    quant_views = [_coefficient_preview(value, True) for value in quantized]
    dct_rgb = np.stack(dct_views, axis=-1)
    quant_rgb = np.stack(quant_views, axis=-1)
    entropy_rgb = _entropy_preview(quantized[0])
    labels = prep.labels
    label_rgb = np.stack((labels * 73, labels * 151, labels * 199), axis=-1) % 255
    texture_visible = np.clip(128.0 + prep.texture * 4.0, 0.0, 255.0)
    return {
        "1_ycbcr": np.uint8(np.clip(ycc, 0.0, 255.0)),
        "2_chroma_sampling": np.uint8(ycc_to_rgb(sampled)),
        "3_dct_cascade": np.uint8(np.clip(dct_rgb, 0.0, 255.0)),
        "4_quantization": np.uint8(np.clip(quant_rgb, 0.0, 255.0)),
        "5_zigzag_entropy": entropy_rgb,
        "cartoon": np.uint8(np.clip(prep.cartoon, 0.0, 255.0)),
        "texture_x4": np.uint8(texture_visible),
        "aligned_regions": np.uint8(label_rgb),
        "preprocessed": np.uint8(np.clip(prep.rgb, 0.0, 255.0)),
    }


def encode(rgb: np.ndarray, config: JPEGConfig) -> bytes:
    output = BytesIO()
    Image.fromarray(np.uint8(np.clip(np.rint(rgb), 0, 255)), "RGB").save(
        output,
        "JPEG",
        quality=int(config.quality),
        subsampling=int(config.subsampling),
        progressive=bool(config.progressive),
        optimize=bool(config.optimize),
        exif=b"",
    )
    return output.getvalue()


def decode(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def _box_filter(image: np.ndarray, radius: int = 3) -> np.ndarray:
    # A Gaussian local statistic is stable, cheap, and avoids a skimage dependency.
    return gaussian_filter(image, sigma=(radius / 2.0, radius / 2.0, 0.0))


@dataclass
class MetricReference:
    rgb: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    edge: np.ndarray
    gaussian_kernel: np.ndarray


def _prepare_metric_reference(reference: np.ndarray) -> MetricReference:
    rgb = np.asarray(reference, dtype=np.float64)
    mean = _box_filter(rgb)
    variance = np.maximum(_box_filter(rgb * rgb) - mean * mean, 0.0)
    luma = rgb_to_ycc(rgb)[..., 0]
    edge = np.stack((sobel(luma, 0), sobel(luma, 1)), axis=-1)
    coordinate = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinate / 1.5) ** 2)
    kernel /= np.sum(kernel)
    return MetricReference(
        rgb=rgb, mean=mean, variance=variance, edge=edge,
        gaussian_kernel=kernel,
    )


def _image_metrics_prepared(
    prepared: MetricReference, candidate: np.ndarray
) -> tuple[float, float, float]:
    reference = prepared.rgb
    candidate = np.asarray(candidate, dtype=np.float64)
    if _native_image_metrics is not None:
        native = _native_image_metrics(
            reference,
            candidate,
            prepared.mean,
            prepared.variance,
            prepared.edge,
            prepared.gaussian_kernel,
            threads=0,
        )
        if native is not None:
            mse, ssim, edge_mse = native
            psnr = 99.0 if mse <= 1e-20 else 10.0 * math.log10(255.0 ** 2 / mse)
            edge_psnr = (
                99.0 if edge_mse <= 1e-20
                else 10.0 * math.log10(255.0 ** 2 / edge_mse)
            )
            return ssim, psnr, edge_psnr
    mse = float(np.mean((reference - candidate) ** 2))
    psnr = 99.0 if mse <= 1e-20 else 10.0 * math.log10(255.0 ** 2 / mse)

    # Report a true color SSIM mean. A luma-only SSIM can conceal the exact
    # failure mode under study: overly aggressive chroma redistribution.
    mu_x, mu_y = prepared.mean, _box_filter(candidate)
    var_x = prepared.variance
    var_y = np.maximum(_box_filter(candidate * candidate) - mu_y * mu_y, 0.0)
    covariance = _box_filter(reference * candidate) - mu_x * mu_y
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2) + 1e-20
    )
    ssim = float(np.mean(ssim_map))

    cand_y = rgb_to_ycc(candidate)[..., 0]
    cand_edge = np.stack((sobel(cand_y, 0), sobel(cand_y, 1)), axis=-1)
    edge_mse = float(np.mean((prepared.edge - cand_edge) ** 2))
    edge_psnr = 99.0 if edge_mse <= 1e-20 else 10.0 * math.log10(255.0 ** 2 / edge_mse)
    return ssim, psnr, edge_psnr


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    return _image_metrics_prepared(_prepare_metric_reference(reference), candidate)


def _objective(size: int, target: int, ssim: float, psnr: float, edge_psnr: float) -> float:
    overshoot = max(0.0, (size - target) / max(target, 1))
    # Once under budget, fidelity dominates. Oversize candidates remain useful
    # on the Pareto frontier but cannot win a close under-budget comparison.
    return 1000.0 * ssim + 0.2 * psnr + 0.1 * edge_psnr - 500.0 * overshoot


def _pareto(candidates: Iterable[Candidate]) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda value: (value.size_bytes, -value.ssim))
    result: list[Candidate] = []
    best_quality = -math.inf
    for candidate in ordered:
        if candidate.ssim > best_quality + 1e-9:
            result.append(candidate)
            best_quality = candidate.ssim
    return result


def optimize_jpeg(
    source: str | Path,
    output: str | Path,
    *,
    target_bytes: int = 29_200,
    exhaustive: bool = False,
    progress: Callable[[int, int, Candidate], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> OptimizationResult:
    """Trace the measured rate/distortion boundary over representation controls.

    The default search does not enumerate the quality axis.  For each ownership
    representation it follows the monotone JPEG rate field to the target-size
    crossing with safeguarded log-rate interpolation.  Exhaustive mode retains
    the old Cartesian grid as a diagnostic oracle.
    """
    source_path, output_path = Path(source), Path(output)
    reference = load_rgb(source_path)
    source_quality = infer_source_quality(source_path)
    if exhaustive:
        qualities = range(max(1, source_quality - 10), min(100, source_quality + 20) + 1)
    else:
        qualities = sorted(set(
            range(max(1, source_quality - 8), min(100, source_quality + 16) + 1, 2)
        ) | {source_quality})
    subsamplings = (0, 1, 2)
    # The zero controls prove how much comes from libjpeg alone. Later levels
    # progressively introduce aligned chroma and flat-interior luma projection.
    projections = (0.0, 0.18, 0.35, 0.55) if exhaustive else (0.0, 0.25, 0.5)
    luma_shrinks = (0.0, 0.08, 0.16) if exhaustive else (0.0, 0.12)
    phases = (-22.5, 0.0, 22.5) if exhaustive else (0.0,)
    structural = [
        JPEGConfig(
            quality=source_quality,
            subsampling=subsampling,
            chroma_projection=projection,
            luma_texture_shrink=luma,
            phase_degrees=phase,
        )
        for projection in projections
        for luma in luma_shrinks
        for phase in phases
        for subsampling in subsamplings
    ]

    preprocess_basis = _prepare_preprocess_basis(reference, structural[0])
    metric_reference = _prepare_metric_reference(reference)
    # The optimizer only needs encoded RGB.  Diagnostic cartoon/angle views
    # are invariant and formerly got materialized once per representation,
    # wasting both bandwidth and hundreds of megabytes on large inputs.
    preprocess_cache: dict[tuple[float, float, float], np.ndarray] = {}
    encoded_cache: dict[JPEGConfig, bytes] = {}
    candidates: list[Candidate] = []
    best: Candidate | None = None
    probe_count = 0
    maximum_evaluations = len(structural) * (len(qualities) if exhaustive else 10)

    def probe(config: JPEGConfig) -> bytes:
        nonlocal probe_count
        if cancelled is not None and cancelled():
            raise InterruptedError("JPEG optimization cancelled")
        cached = encoded_cache.get(config)
        if cached is not None:
            return cached
        key = (config.chroma_projection, config.luma_texture_shrink, config.phase_degrees)
        if key not in preprocess_cache:
            preprocess_cache[key] = ycc_to_rgb(
                _project_preprocess_ycc(preprocess_basis, config)
            )
        data = encode(preprocess_cache[key], config)
        encoded_cache[config] = data
        probe_count += 1
        if progress is not None:
            progress(probe_count, maximum_evaluations, Candidate(
                config=config,
                size_bytes=len(data),
                ssim=math.nan,
                psnr_db=math.nan,
                edge_psnr_db=math.nan,
                objective=math.nan,
            ))
        return data

    def measure(config: JPEGConfig) -> Candidate:
        data = probe(config)
        reconstructed = decode(data)
        ssim, psnr, edge_psnr = _image_metrics_prepared(
            metric_reference, reconstructed
        )
        candidate = Candidate(
            config=config,
            size_bytes=len(data),
            ssim=ssim,
            psnr_db=psnr,
            edge_psnr_db=edge_psnr,
            objective=_objective(len(data), target_bytes, ssim, psnr, edge_psnr),
            data=data,
        )
        return candidate

    def accept(candidate: Candidate) -> Candidate:
        nonlocal best
        candidates.append(candidate)
        eligible = candidate.size_bytes <= target_bytes
        if best is None or (
            eligible and best.size_bytes > target_bytes
        ) or (
            eligible == (best.size_bytes <= target_bytes) and candidate.objective > best.objective
        ):
            best = candidate
        return candidate

    def evaluate(config: JPEGConfig) -> Candidate:
        return accept(measure(config))

    if exhaustive:
        for base in structural:
            for quality in qualities:
                evaluate(replace(base, quality=quality))
    else:
        # The byte target can be far below a high-quality source.  The old
        # source-local [quality-8, quality+16] bracket made the rate tracer stop
        # before it had an under-target sample, so a requested 2.5 KiB result
        # could silently return 10 KiB.  Keep the source quality as the warm
        # start, but let the monotone characteristic reach the JPEG endpoints.
        q_min, q_max = 1, 100
        quality_seed_by_subsampling: dict[int, int] = {}
        transported_quality_seed = int(np.clip(source_quality, q_min, q_max))
        resolved_configs: list[JPEGConfig] = []
        # Each branch is a one-dimensional rate field.  Follow its target
        # characteristic instead of sampling every integer quality.
        for base in structural:
            samples: dict[int, Candidate] = {}
            q = int(np.clip(
                quality_seed_by_subsampling.get(
                    base.subsampling, transported_quality_seed
                ),
                q_min,
                q_max,
            ))
            for _step in range(10):
                if q not in samples:
                    config = replace(base, quality=q)
                    data = probe(config)
                    samples[q] = Candidate(
                        config=config, size_bytes=len(data), ssim=math.nan,
                        psnr_db=math.nan, edge_psnr_db=math.nan,
                        objective=math.nan,
                    )
                under = sorted(
                    (value for value in samples.values() if value.size_bytes <= target_bytes),
                    key=lambda value: value.config.quality,
                )
                over = sorted(
                    (value for value in samples.values() if value.size_bytes > target_bytes),
                    key=lambda value: value.config.quality,
                )
                if under and over:
                    lo = under[-1]
                    hi = over[0]
                    if hi.config.quality - lo.config.quality <= 1:
                        break
                    log_lo = math.log(max(lo.size_bytes, 1))
                    log_hi = math.log(max(hi.size_bytes, 1))
                    fraction = (
                        (math.log(max(target_bytes, 1)) - log_lo)
                        / max(log_hi - log_lo, 1e-12)
                    )
                    predicted = lo.config.quality + fraction * (
                        hi.config.quality - lo.config.quality
                    )
                    q = int(np.clip(round(predicted), lo.config.quality + 1, hi.config.quality - 1))
                else:
                    current = samples[q]
                    # JPEG log-rate is locally close to affine in quality.
                    # The estimate only proposes a step; a bracketed refinement
                    # certifies the final integer crossing.
                    delta = int(np.clip(round(
                        math.log(max(target_bytes, 1) / max(current.size_bytes, 1)) / 0.055
                    ), -12, 12))
                    if delta == 0:
                        delta = 1 if current.size_bytes <= target_bytes else -1
                    proposed = int(np.clip(q + delta, q_min, q_max))
                    if proposed == q:
                        break
                    q = proposed
            eligible_samples = [
                value for value in samples.values()
                if value.size_bytes <= target_bytes
            ]
            resolved = (
                max(eligible_samples, key=lambda value: value.config.quality)
                if eligible_samples
                else min(samples.values(), key=lambda value: value.size_bytes)
            )
            quality_seed_by_subsampling[base.subsampling] = resolved.config.quality
            transported_quality_seed = resolved.config.quality
            resolved_configs.append(resolved.config)
        # SciPy's filters release the GIL. Two bounded workers overlap the
        # independent terminal measurements without the memory spike of one
        # worker per branch; map preserves deterministic acceptance order.
        metric_workers = 1 if _native_image_metrics is not None else 2
        with ThreadPoolExecutor(max_workers=metric_workers) as executor:
            for candidate in executor.map(measure, resolved_configs):
                accept(candidate)

    assert best is not None and best.data is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(best.data)
    frontier = _pareto(candidates)
    # Keep bytes only for the winning artifact; reports remain compact.
    for candidate in candidates:
        if candidate is not best:
            candidate.data = None
    regions = int(preprocess_basis.block_labels.max()) + 1
    return OptimizationResult(
        source=source_path,
        output=output_path,
        source_bytes=source_path.stat().st_size,
        source_quality=source_quality,
        target_bytes=int(target_bytes),
        best=best,
        frontier=frontier,
        regions=regions,
        evaluations=probe_count,
        search_strategy="cartesian" if exhaustive else "frontier_trace",
        metric_backend="native_fused_cpp" if _native_image_metrics is not None else "scipy",
    )


def save_stage_images(stages: dict[str, np.ndarray], directory: str | Path) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name, image in stages.items():
        Image.fromarray(np.uint8(image), "RGB").save(destination / f"{name}.png")


def save_report(result: OptimizationResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.report(), indent=2, sort_keys=True) + "\n")
