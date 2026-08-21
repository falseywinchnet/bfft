"""Reference-grounded structural metrics for synthetic codec candidates."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, laplace, sobel, uniform_filter


def _load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.float64)


def _composite(rgba: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
    alpha = rgba[..., 3:4]/255.0
    return rgba[..., :3]*alpha + np.asarray(background, dtype=np.float64)*(1.0-alpha)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.asarray((.2126, .7152, .0722))


def _psnr(error: np.ndarray) -> float:
    mse = float(np.mean(np.square(error)))
    return 99.0 if mse <= 1e-20 else 10*math.log10(255.0**2/mse)


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    mu_x = uniform_filter(reference, size=(7, 7, 1), mode="reflect")
    mu_y = uniform_filter(candidate, size=(7, 7, 1), mode="reflect")
    var_x = np.maximum(uniform_filter(reference*reference, size=(7, 7, 1), mode="reflect")-mu_x*mu_x, 0)
    var_y = np.maximum(uniform_filter(candidate*candidate, size=(7, 7, 1), mode="reflect")-mu_y*mu_y, 0)
    covariance = uniform_filter(reference*candidate, size=(7, 7, 1), mode="reflect")-mu_x*mu_y
    c1, c2 = (2.55)**2, (7.65)**2
    value = ((2*mu_x*mu_y+c1)*(2*covariance+c2))/((mu_x*mu_x+mu_y*mu_y+c1)*(var_x+var_y+c2)+1e-20)
    return float(np.mean(value))


def _gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = sobel(image, axis=1, mode="reflect")/8.0
    gy = sobel(image, axis=0, mode="reflect")/8.0
    return gx, gy, np.hypot(gx, gy)


def _rect(shape: tuple[int, int], normalized: Iterable[float]) -> tuple[slice, slice]:
    x0, y0, x1, y1 = normalized
    height, width = shape
    return (
        slice(max(0, round(y0*height)), min(height, round(y1*height))),
        slice(max(0, round(x0*width)), min(width, round(x1*width))),
    )


def _probe_metrics(reference: np.ndarray, candidate: np.ndarray, probes: list[dict[str, object]]) -> dict[str, float]:
    ref_y, cand_y = _luma(reference), _luma(candidate)
    result: dict[str, list[float]] = {}
    for probe in probes:
        region = _rect(ref_y.shape, probe["rect"])
        r, c = ref_y[region], cand_y[region]
        kind = str(probe["kind"])
        if kind == "gradient":
            axis = 1 if probe.get("axis", "x") == "x" else 0
            ref_delta, cand_delta = np.diff(r, axis=axis), np.diff(c, axis=axis)
            active = np.abs(ref_delta) > .05
            plateau = float(np.mean((np.abs(cand_delta) < .05) & active))
            curvature = float(np.mean(np.abs(np.diff(cand_delta-ref_delta, axis=axis))))
            result.setdefault("gradient_false_plateau_fraction", []).append(plateau)
            result.setdefault("gradient_curvature_error", []).append(curvature)
        elif kind == "texture":
            ref_hp = r-uniform_filter(r, 3, mode="reflect")
            cand_hp = c-uniform_filter(c, 3, mode="reflect")
            ref_energy = float(np.mean(ref_hp*ref_hp))
            cand_energy = float(np.mean(cand_hp*cand_hp))
            correlation = float(np.sum(ref_hp*cand_hp)/math.sqrt(max(np.sum(ref_hp*ref_hp)*np.sum(cand_hp*cand_hp), 1e-20)))
            result.setdefault("texture_energy_ratio", []).append(cand_energy/max(ref_energy, 1e-20))
            result.setdefault("texture_correlation", []).append(correlation)
        elif kind == "flat":
            residual = (c-r)-uniform_filter(c-r, 9, mode="reflect")
            result.setdefault("flat_spurious_texture_rms", []).append(float(np.sqrt(np.mean(residual*residual))))
    return {key: float(np.mean(values)) for key, values in result.items()}


def candidate_metrics(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    probes: list[dict[str, object]] | None = None,
    jpeg_composite: tuple[int, int, int] = (239, 241, 244),
) -> dict[str, float | int | str]:
    reference_path, candidate_path = Path(reference_path), Path(candidate_path)
    reference_rgba, candidate_rgba = _load_rgba(reference_path), _load_rgba(candidate_path)
    if reference_rgba.shape != candidate_rgba.shape:
        raise ValueError(f"shape mismatch: {reference_rgba.shape} != {candidate_rgba.shape}")
    reference = _composite(reference_rgba, jpeg_composite)
    candidate = _composite(candidate_rgba, jpeg_composite)
    ref_y, cand_y = _luma(reference), _luma(candidate)
    ref_gx, ref_gy, ref_magnitude = _gradient(ref_y)
    cand_gx, cand_gy, cand_magnitude = _gradient(cand_y)
    edge_mask = ref_magnitude > max(2.0, float(np.percentile(ref_magnitude, 82)))
    edge_band = binary_dilation(edge_mask, iterations=3)
    flat_mask = ref_magnitude < .75
    color_error = candidate-reference
    edge_error = np.stack((cand_gx-ref_gx, cand_gy-ref_gy), axis=-1)
    chroma_matrix = np.asarray(((-.1146, -.3854, .5), (.5, -.4542, -.0458)))
    ref_chroma, cand_chroma = reference @ chroma_matrix.T, candidate @ chroma_matrix.T
    chroma_edge = np.stack([
        sobel(ref_chroma, axis=1, mode="reflect")/8,
        sobel(ref_chroma, axis=0, mode="reflect")/8,
    ], axis=-1)
    cand_chroma_edge = np.stack([
        sobel(cand_chroma, axis=1, mode="reflect")/8,
        sobel(cand_chroma, axis=0, mode="reflect")/8,
    ], axis=-1)
    edge_luma_mae = float(np.mean(np.abs(cand_y[edge_band]-ref_y[edge_band]))) if np.any(edge_band) else 0.0
    flat_luma_mae = float(np.mean(np.abs(cand_y[flat_mask]-ref_y[flat_mask]))) if np.any(flat_mask) else 0.0
    ringing = np.abs(laplace(cand_y-ref_y, mode="reflect"))[edge_band]
    alpha_error = candidate_rgba[..., 3]-reference_rgba[..., 3]
    metrics: dict[str, float | int | str] = {
        "candidate": str(candidate_path),
        "bytes": candidate_path.stat().st_size,
        "ssim": _ssim(reference, candidate),
        "psnr_db": _psnr(color_error),
        "edge_psnr_db": _psnr(edge_error),
        "chroma_edge_psnr_db": _psnr(cand_chroma_edge-chroma_edge),
        "edge_luma_mae": edge_luma_mae,
        "flat_luma_mae": flat_luma_mae,
        "edge_magnitude_bias": float(np.mean(cand_magnitude[edge_mask]-ref_magnitude[edge_mask])) if np.any(edge_mask) else 0.0,
        "ringing_laplacian_mae": float(np.mean(ringing)) if ringing.size else 0.0,
        "alpha_psnr_db": _psnr(alpha_error),
    }
    metrics.update(_probe_metrics(reference, candidate, probes or []))
    return metrics


def _find_candidate(directory: Path, codec: str, case: str) -> Path | None:
    names = (
        f"{codec}__{case}.{'png' if codec == 'png' else 'jpg'}",
        f"{case}.{'png' if codec == 'png' else 'jpg'}",
        f"jpeg__{case}.jpeg" if codec == "jpeg" else "",
    )
    for name in names:
        if name and (directory/name).is_file():
            return directory/name
    return None


def evaluate_suite(root: str | Path, candidates: dict[str, str | Path]) -> dict[str, object]:
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    background = tuple(manifest["jpeg_composite"])
    rows: list[dict[str, object]] = []
    for label, directory_value in candidates.items():
        directory = Path(directory_value)
        for case in manifest["cases"]:
            reference = root/case["reference"]
            for codec in ("png", "jpeg"):
                candidate = _find_candidate(directory, codec, case["name"])
                if candidate is None:
                    continue
                row: dict[str, object] = {
                    "label": label,
                    "case": case["name"],
                    "codec": codec,
                    "tags": case["tags"],
                }
                row.update(candidate_metrics(reference, candidate, probes=case["probes"], jpeg_composite=background))
                rows.append(row)
    summary: list[dict[str, object]] = []
    for label in sorted({str(row["label"]) for row in rows}):
        for codec in ("png", "jpeg"):
            group = [row for row in rows if row["label"] == label and row["codec"] == codec]
            if not group:
                continue
            numeric = ("bytes", "ssim", "psnr_db", "edge_psnr_db", "chroma_edge_psnr_db", "edge_luma_mae", "ringing_laplacian_mae")
            summary.append({
                "label": label,
                "codec": codec,
                "cases": len(group),
                **{key: float(np.mean([float(row[key]) for row in group])) for key in numeric},
            })
    report = {"schema": 1, "root": str(root), "rows": rows, "summary": summary}
    (root/"evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
    keys = sorted({key for row in rows for key in row if key != "tags"})
    with (root/"evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return report
