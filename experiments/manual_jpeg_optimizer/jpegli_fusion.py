"""Dogfood runner for the Jpegli ownership-trellis backend."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile

from PIL import Image

from .balanced_regions import BalancedRegionConfig, balanced_bifurcation_regions
from .certified_relaxation import _coefficients
from .core import image_metrics, load_rgb, rgb_to_ycc
from .jpegli_bridge import ownership_dead_zone_field, write_jldz
from .spatial_dct_transport import SpatialDCTTransportConfig, transport_spatial_dct


def encode_jpegli_fusion(
    source: Path,
    output: Path,
    *,
    quality: int = 72,
    target_bytes: int = 0,
    regions: int = 256,
    minimum_region_blocks: int = 24,
    field_strength: float = 0.25,
    edge_protection: float = 2.0,
    transport_lambda: float = 0.002,
    frequency_weight: float = 0.1,
    cross_region_weight: float = 0.05,
    trellis_lambda: float = 0.0695,
    ownership_weight: float = 0.05,
    trellis_edge_weight: float = 1.0,
    trellis_luma_weight: float = 1.0,
    trellis_chroma_weight: float = 1.0,
    quant_luma_tilt: float = -0.5,
    quant_chroma_tilt: float = 0.0,
) -> dict:
    root = Path(__file__).resolve().parents[2]
    encoder = root / "third_party/jpeg_fusion/build/jpegli/tools/cjpegli"
    if not encoder.exists():
        raise FileNotFoundError(
            f"Missing {encoder}; run third_party/jpeg_fusion/build_backends.sh"
        )
    reference = load_rgb(source)
    ycc = rgb_to_ycc(reference)
    quotient = balanced_bifurcation_regions(
        ycc,
        quality,
        BalancedRegionConfig(
            target_regions=regions, minimum_blocks=minimum_region_blocks
        ),
    )
    transported = None
    transport_report = None
    if transport_lambda > 0.0:
        transport = transport_spatial_dct(
            _coefficients(ycc),
            quotient.labels,
            quality,
            SpatialDCTTransportConfig(
                transport_lambda=transport_lambda,
                frequency_weight=frequency_weight,
                cross_region_weight=cross_region_weight,
            ),
        )
        transported = transport.coefficients
        transport_report = transport.report()
    field = ownership_dead_zone_field(
        ycc,
        quotient.labels,
        quality,
        strength=field_strength,
        edge_protection=edge_protection,
        transported_coefficients=transported,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jpegli-fusion-") as directory:
        temporary = Path(directory)
        png = temporary / "input.png"
        atlas = temporary / "ownership.jldz"
        Image.fromarray(reference.astype("uint8"), "RGB").save(png)
        write_jldz(atlas, field)
        command = [
            str(encoder), str(png), str(output),
            "--chroma_subsampling=420", "-p", "2",
            f"--dead_zone_field={atlas}",
            f"--trellis_lambda={trellis_lambda}",
            f"--ownership_weight={ownership_weight}",
            f"--trellis_edge_weight={trellis_edge_weight}",
            f"--trellis_luma_weight={trellis_luma_weight}",
            f"--trellis_chroma_weight={trellis_chroma_weight}",
            f"--quant_luma_tilt={quant_luma_tilt}",
            f"--quant_chroma_tilt={quant_chroma_tilt}",
        ]
        if target_bytes > 0:
            command.append(f"--target_size={target_bytes}")
        else:
            command.extend(("-q", str(quality)))
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True
        )
    candidate = load_rgb(output)
    ssim, psnr, edge_psnr = image_metrics(reference, candidate)
    match = re.search(
        r"Trellis: (\d+) blocks, nonzeros (\d+) -> (\d+), "
        r"estimated bits ([0-9]+(?:\.[0-9]+)?) -> "
        r"([0-9]+(?:\.[0-9]+)?), objective ([0-9]+(?:\.[0-9]+)?)\.",
        completed.stderr,
    )
    trellis = None
    if match:
        trellis = {
            "blocks": int(match.group(1)),
            "nonzeros_before": int(match.group(2)),
            "nonzeros_after": int(match.group(3)),
            "estimated_bits_before": float(match.group(4)),
            "estimated_bits_after": float(match.group(5)),
            "terminal_objective": float(match.group(6)),
        }
    report = {
        "source": str(source),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "ssim": ssim,
        "psnr_db": psnr,
        "edge_psnr_db": edge_psnr,
        "quality": quality,
        "target_bytes": target_bytes,
        "regions": quotient.report(),
        "spatial_frequency_transport": transport_report,
        "field": {
            "strength": field_strength,
            "edge_protection": edge_protection,
            "mean_ac_threshold": float(field[..., 1:].mean()),
            "maximum_threshold": float(field.max()),
        },
        "trellis": trellis,
        "trellis_config": {
            "rate_lambda": trellis_lambda,
            "ownership_weight": ownership_weight,
            "edge_weight": trellis_edge_weight,
            "luma_weight": trellis_luma_weight,
            "chroma_weight": trellis_chroma_weight,
        },
        "quantization_tilt": {
            "luma": quant_luma_tilt,
            "chroma": quant_chroma_tilt,
        },
        "encoder_stderr": completed.stderr,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
