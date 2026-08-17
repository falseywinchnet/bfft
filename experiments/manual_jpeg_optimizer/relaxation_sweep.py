"""Measured Pareto sweep over globally optimal fixed connection relaxations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .certified_relaxation import (
    RelaxationConfig,
    _coefficients,
    coefficients_to_rgb,
    solve_coefficients,
)
from .core import (
    JPEGConfig,
    _objective,
    _region_labels,
    decode,
    encode,
    image_metrics,
    infer_source_quality,
    load_rgb,
    rgb_to_ycc,
)


@dataclass
class RelaxationCandidate:
    config: RelaxationConfig
    size_bytes: int
    ssim: float
    psnr_db: float
    edge_psnr_db: float
    objective: float
    certificate: dict[str, float | int | bool]
    data: bytes | None = None

    def report(self) -> dict:
        return {
            "config": self.config.__dict__,
            "size_bytes": self.size_bytes,
            "ssim": self.ssim,
            "psnr_db": self.psnr_db,
            "edge_psnr_db": self.edge_psnr_db,
            "objective": self.objective,
            "certificate": self.certificate,
        }


def _pareto(candidates: Iterable[RelaxationCandidate]) -> list[RelaxationCandidate]:
    result = []
    best_ssim = -np.inf
    for candidate in sorted(candidates, key=lambda item: (item.size_bytes, -item.ssim)):
        if candidate.ssim > best_ssim + 1e-10:
            result.append(candidate)
            best_ssim = candidate.ssim
    return result


def search_relaxations(
    source: str | Path,
    output: str | Path,
    *,
    target_bytes: int = 29_200,
    rate_lambdas: Iterable[float] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
    connection_lambdas: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
    frame_modes: Iterable[str] = ("identity", "chroma", "full"),
    iterations: int = 500,
    gap_tolerance: float = 1e-5,
    progress: Callable[[int, int, RelaxationCandidate], None] | None = None,
) -> dict:
    source_path, output_path = Path(source), Path(output)
    reference = load_rgb(source_path)
    ycc = rgb_to_ycc(reference)
    labels, _ = _region_labels(ycc, 1.2, 0.58)
    source_coefficients = _coefficients(ycc)
    quality = infer_source_quality(source_path)
    jpeg_config = JPEGConfig(quality=quality, subsampling=1)

    configs = []
    seen = set()
    for rate in rate_lambdas:
        for connection in connection_lambdas:
            for frame in frame_modes:
                # With no connection term, frame choice is a pure gauge and
                # the unique global coefficient solution is identical.
                key = (float(rate), float(connection), frame if connection else "identity")
                if key in seen:
                    continue
                seen.add(key)
                configs.append(RelaxationConfig(
                    rate_lambda=float(rate),
                    connection_lambda=float(connection),
                    frame_mode=frame if connection else "identity",
                    iterations=int(iterations),
                    relative_gap_tolerance=float(gap_tolerance),
                ))

    candidates: list[RelaxationCandidate] = []
    best: RelaxationCandidate | None = None
    for index, config in enumerate(configs, 1):
        coefficients, certificate = solve_coefficients(
            source_coefficients, labels, config
        )
        rgb = coefficients_to_rgb(coefficients, labels.shape, reference.shape[:2])
        data = encode(rgb, jpeg_config)
        ssim, psnr, edge_psnr = image_metrics(reference, decode(data))
        candidate = RelaxationCandidate(
            config=config,
            size_bytes=len(data),
            ssim=ssim,
            psnr_db=psnr,
            edge_psnr_db=edge_psnr,
            objective=_objective(len(data), target_bytes, ssim, psnr, edge_psnr),
            certificate=certificate,
            data=data,
        )
        candidates.append(candidate)
        eligible = candidate.size_bytes <= target_bytes
        if best is None or (eligible and best.size_bytes > target_bytes) or (
            eligible == (best.size_bytes <= target_bytes)
            and candidate.objective > best.objective
        ):
            best = candidate
        if progress:
            progress(index, len(configs), candidate)

    assert best is not None and best.data is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(best.data)
    report = {
        "source": str(source_path),
        "output": str(output_path),
        "source_bytes": source_path.stat().st_size,
        "target_bytes": int(target_bytes),
        "source_quality": quality,
        "best": best.report(),
        "frontier": [candidate.report() for candidate in _pareto(candidates)],
        "candidate_count": len(candidates),
        "proof_scope": (
            "Each candidate is the certified global optimum of its fixed-frame "
            "convex relaxation. Selection across candidates uses actual libjpeg "
            "bytes and decoded metrics and is finite/exhaustive over this grid."
        ),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
