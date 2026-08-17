"""Verify the fused city result strictly dominates the TinyPNG control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import image_metrics, load_rgb


def measure(reference: Path, candidate: Path) -> dict:
    ssim, psnr, edge_psnr = image_metrics(load_rgb(reference), load_rgb(candidate))
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "ssim": ssim,
        "psnr_db": psnr,
        "edge_psnr_db": edge_psnr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("city_image.jpg"))
    parser.add_argument("--control", type=Path, default=Path("tinypng.jpg"))
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(
            "experiments/manual_jpeg_optimizer/results/city_full_transport_win.jpg"
        ),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    control = measure(args.reference, args.control)
    candidate = measure(args.reference, args.candidate)
    checks = {
        "smaller": candidate["bytes"] < control["bytes"],
        "higher_ssim": candidate["ssim"] > control["ssim"],
        "higher_psnr": candidate["psnr_db"] > control["psnr_db"],
        "higher_edge_psnr": candidate["edge_psnr_db"] > control["edge_psnr_db"],
    }
    report = {
        "reference": str(args.reference),
        "control": control,
        "candidate": candidate,
        "margin": {
            "bytes": control["bytes"] - candidate["bytes"],
            "ssim": candidate["ssim"] - control["ssim"],
            "psnr_db": candidate["psnr_db"] - control["psnr_db"],
            "edge_psnr_db": candidate["edge_psnr_db"] - control["edge_psnr_db"],
        },
        "checks": checks,
        "strictly_dominates": all(checks.values()),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    return 0 if report["strictly_dominates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

