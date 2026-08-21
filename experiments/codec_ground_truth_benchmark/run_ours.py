"""Run the local optimizers at byte targets established by TinyPNG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from experiments.manual_jpeg_optimizer.core import optimize_jpeg
from experiments.manual_png_optimizer.core import PNGConfig, optimize_png

from .evaluate import _find_candidate


Progress = Callable[[str], None]


def run_ours(
    root: str | Path,
    *,
    codecs: tuple[str, ...] = ("png", "jpeg"),
    output_dir: str | Path | None = None,
    case_names: set[str] | None = None,
    progress: Progress | None = None,
) -> list[dict[str, object]]:
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    tinypng = root/"candidates"/"tinypng"
    ours = Path(output_dir) if output_dir is not None else root/"candidates"/"ours"
    ours.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for case in manifest["cases"]:
        name = case["name"]
        if case_names is not None and name not in case_names:
            continue
        for codec in codecs:
            target = _find_candidate(tinypng, codec, name)
            if target is None:
                continue
            target_bytes = target.stat().st_size
            source = root/case[f"upload_{codec}"]
            output = ours/f"{codec}__{name}.{'png' if codec == 'png' else 'jpg'}"
            if progress:
                progress(f"{codec}/{name}: matching TinyPNG's {target_bytes:,} bytes")
            if codec == "png":
                result = optimize_png(
                    source, output,
                    config=PNGConfig(
                        target_bytes=target_bytes,
                        colors=0,
                        dither="auto",
                        quantizer="auto",
                        ownership_strength=-1.0,
                    ),
                )
                report = result.report()
            else:
                report = optimize_jpeg(source, output, target_bytes=target_bytes).report()
            report["benchmark_case"] = name
            report["benchmark_codec"] = codec
            report["matched_tinypng_bytes"] = target_bytes
            reports.append(report)
    report_path = ours/"optimizer_reports.json"
    previous: list[dict[str, object]] = []
    if report_path.is_file():
        previous = json.loads(report_path.read_text())
    merged = {
        (str(report.get("benchmark_codec", "")), str(report.get("benchmark_case", ""))): report
        for report in previous
    }
    merged.update({
        (str(report["benchmark_codec"]), str(report["benchmark_case"])): report
        for report in reports
    })
    report_path.write_text(json.dumps(list(merged.values()), indent=2)+"\n")
    return reports
