#!/usr/bin/env python3
"""Measure the BFFT pass sequence as a scale-time support geometry.

The usual pipeline keeps only the final cartoon, texture, and TV-defect
fields.  This experiment keeps every intermediate pass and asks a different
question: where, at what scale, and in which direction did structure become
specific?

For pass event ``s_k = (dc_k, dt_k, dg_k)`` we form

    C = G_sigma sum_c s_c^2
    J = G_sigma sum_c grad(s_c) grad(s_c)^T
    Q = J / (C + eps).

``Q`` has units of inverse pixels squared.  Its eigenvectors are the local
normal/tangent frame and the inverse square roots of its eigenvalues are
local correlation lengths.  Thus an elongated support is not requested by
an anisotropy score: it falls out when an event varies rapidly across one
axis and persists along the other.

No cells are allocated here.  The output is an audit of whether the cheap
vanilla decomposition already contains the geometry needed to emit them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage as ndi
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402
import gallery  # noqa: E402
from transport_voronoi import _fit_rgb, srgb_to_lab  # noqa: E402


def _load_image(path: str | None, gallery_key: str) -> tuple[np.ndarray, str]:
    if path:
        from skimage.io import imread

        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float,
) -> float:
    """Weighted quantile used only for reporting, never for allocation."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return float("nan")
    v = v[valid]
    w = w[valid]
    order = np.argsort(v)
    cumulative = np.cumsum(w[order])
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    return float(v[order[np.searchsorted(cumulative, target, side="left")]])


def _event_geometry(
    dc: np.ndarray,
    dt: np.ndarray,
    dg: np.ndarray,
    sigma: float,
    max_length: float,
) -> dict[str, np.ndarray]:
    """Closed-form correlation ellipse of one pass-to-pass event."""
    channels = (
        np.asarray(dc, dtype=np.float64),
        np.asarray(dt, dtype=np.float64),
        math.sqrt(0.5) * np.asarray(dg, dtype=np.float64),
    )
    raw_energy = sum(channel * channel for channel in channels)
    energy = ndi.gaussian_filter(raw_energy, sigma, mode="reflect")

    jxx = np.zeros_like(energy)
    jxy = np.zeros_like(energy)
    jyy = np.zeros_like(energy)
    for channel in channels:
        gx = ndi.sobel(channel, axis=1, mode="reflect") / 8.0
        gy = ndi.sobel(channel, axis=0, mode="reflect") / 8.0
        jxx += gx * gx
        jxy += gx * gy
        jyy += gy * gy
    jxx = ndi.gaussian_filter(jxx, sigma, mode="reflect")
    jxy = ndi.gaussian_filter(jxy, sigma, mode="reflect")
    jyy = ndi.gaussian_filter(jyy, sigma, mode="reflect")

    # Dividing by event energy removes amplitude.  What remains is a local
    # squared spatial frequency tensor.
    scale = max(float(np.percentile(energy, 99.5)), 1e-20)
    denominator = energy + 1e-5 * scale
    qxx = jxx / denominator
    qxy = jxy / denominator
    qyy = jyy / denominator

    trace = qxx + qyy
    disc = np.hypot(qxx - qyy, 2.0 * qxy)
    high = np.maximum(0.5 * (trace + disc), 0.0)
    low = np.maximum(0.5 * (trace - disc), 0.0)
    floor = 1.0 / max(max_length * max_length, 1.0)

    # High spatial frequency is the normal direction and therefore the short
    # support axis.  The low-frequency eigenvector is the support tangent.
    minor = np.clip(1.0 / np.sqrt(high + floor), 0.75, max_length)
    major = np.clip(1.0 / np.sqrt(low + floor), minor, max_length)
    coherence = disc / np.maximum(trace + 2.0 * floor, 1e-20)
    normal_angle = 0.5 * np.arctan2(2.0 * qxy, qxx - qyy)
    tangent_angle = normal_angle + 0.5 * np.pi

    # A straight ellipse cannot purchase an arbitrarily long curved
    # boundary.  Estimate tangent curvature without unwrapping the axial
    # angle by differentiating its doubled-angle representation.  For local
    # curvature k, a tangent departs from the curve by about k*L^2/2.
    # Requiring that departure to remain inside the minor radius gives the
    # closed-form horizon L <= sqrt(2*minor/k).
    cos2 = np.cos(2.0 * tangent_angle)
    sin2 = np.sin(2.0 * tangent_angle)
    dcx = ndi.sobel(cos2, axis=1, mode="reflect") / 8.0
    dcy = ndi.sobel(cos2, axis=0, mode="reflect") / 8.0
    dsx = ndi.sobel(sin2, axis=1, mode="reflect") / 8.0
    dsy = ndi.sobel(sin2, axis=0, mode="reflect") / 8.0
    dtheta_x = 0.5 * (cos2 * dsx - sin2 * dcx)
    dtheta_y = 0.5 * (cos2 * dsy - sin2 * dcy)
    curvature = np.abs(
        np.cos(tangent_angle) * dtheta_x
        + np.sin(tangent_angle) * dtheta_y)
    trusted_curvature = curvature * np.clip(coherence, 0.0, 1.0)
    curvature_horizon = np.sqrt(
        2.0 * minor / np.maximum(trusted_curvature, 1.0 / max_length**2))
    curvature_horizon = np.clip(curvature_horizon, minor, max_length)
    major_curvature = np.minimum(major, curvature_horizon)
    ratio = major / np.maximum(minor, 1e-12)
    return {
        "energy": energy,
        "high_frequency": high,
        "low_frequency": low,
        "minor": minor,
        "major": major,
        "major_curvature": major_curvature,
        "ratio": ratio,
        "ratio_curvature": major_curvature / np.maximum(minor, 1e-12),
        "curvature": curvature,
        "coherence": np.clip(coherence, 0.0, 1.0),
        "angle": tangent_angle,
    }


def _normal_transport(
    previous: np.ndarray,
    current: np.ndarray,
    sigma: float = 1.5,
    max_step: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form local motion between two decomposition states."""
    midpoint = 0.5 * (
        np.asarray(previous, dtype=np.float64)
        + np.asarray(current, dtype=np.float64))
    temporal = np.asarray(current, dtype=np.float64) - previous
    gx = ndi.sobel(midpoint, axis=1, mode="reflect") / 8.0
    gy = ndi.sobel(midpoint, axis=0, mode="reflect") / 8.0
    a = ndi.gaussian_filter(gx * gx, sigma, mode="reflect")
    b = ndi.gaussian_filter(gx * gy, sigma, mode="reflect")
    c = ndi.gaussian_filter(gy * gy, sigma, mode="reflect")
    rhs_x = -ndi.gaussian_filter(gx * temporal, sigma, mode="reflect")
    rhs_y = -ndi.gaussian_filter(gy * temporal, sigma, mode="reflect")
    trace = a + c
    ridge = 0.04 * trace + 1e-6 * max(
        float(np.percentile(trace, 95.0)), 1e-12)
    determinant = (a + ridge) * (c + ridge) - b * b
    vx = ((c + ridge) * rhs_x - b * rhs_y) / np.maximum(
        determinant, 1e-30)
    vy = ((a + ridge) * rhs_y - b * rhs_x) / np.maximum(
        determinant, 1e-30)
    magnitude = np.hypot(vx, vy)
    limiter = np.minimum(1.0, max_step / np.maximum(magnitude, 1e-30))
    vx *= limiter
    vy *= limiter
    confidence = (
        np.maximum(a * c - b * b, 0.0) /
        np.maximum(trace * trace, 1e-30)
    )
    return vx, vy, np.clip(4.0 * confidence, 0.0, 1.0)


def build_flow_volume(
    rgb: np.ndarray,
    passes: int = 24,
    lam: float = 0.05,
    mu: float = 40.0,
    flow_sweeps: int = 32,
    tensor_sigma: float = 1.0,
    threads: int = 4,
) -> dict[str, np.ndarray | list[dict[str, float]]]:
    """Return every vanilla pass and its derived support geometry."""
    lab = srgb_to_lab(rgb)
    light = lab[..., 0] * 255.0
    h, w = light.shape
    max_length = 0.18 * max(h, w)

    cartoons = []
    textures = []
    defects = []
    geometries = []
    transport_x = []
    transport_y = []
    transport_confidence = []
    stage_stats: list[dict[str, float]] = []

    previous_cartoon = light / 255.0
    previous_texture = np.zeros_like(previous_cartoon)
    previous_defect = np.zeros_like(previous_cartoon)
    started = time.perf_counter()
    cartoon_trace, texture_trace = bfft.meyer_trace(
        light, lam=lam, mu=mu, passes=passes, threads=threads)

    for stage in range(1, passes + 1):
        cartoon = cartoon_trace[stage - 1]
        texture = texture_trace[stage - 1]
        projected = bfft.rof(
            light - texture,
            c=lam,
            eta=2.0 * lam,
            sweeps=flow_sweeps,
            tol=0.0,
            threads=threads,
        )
        cartoon = cartoon / 255.0
        texture = texture / 255.0
        defect = (cartoon * 255.0 - projected) / 255.0
        vx, vy, vconfidence = _normal_transport(
            previous_cartoon, cartoon)
        geometry = _event_geometry(
            cartoon - previous_cartoon,
            texture - previous_texture,
            defect - previous_defect,
            tensor_sigma,
            max_length,
        )
        weight = geometry["energy"]
        total = max(float(np.sum(weight)), 1e-30)
        stage_stats.append({
            "stage": stage,
            "event_mass": total,
            "event_rms": float(np.sqrt(np.mean(weight))),
            "minor_q50_px": _weighted_quantile(
                geometry["minor"], weight, 0.50),
            "minor_q20_px": _weighted_quantile(
                geometry["minor"], weight, 0.20),
            "major_q50_px": _weighted_quantile(
                geometry["major"], weight, 0.50),
            "ratio_q50": _weighted_quantile(
                geometry["ratio"], weight, 0.50),
            "ratio_q80": _weighted_quantile(
                geometry["ratio"], weight, 0.80),
            "coherence_mean": float(
                np.sum(weight * geometry["coherence"]) / total),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        })
        cartoons.append(cartoon.astype(np.float32))
        textures.append(texture.astype(np.float32))
        defects.append(defect.astype(np.float32))
        geometries.append(geometry)
        transport_x.append(vx.astype(np.float32))
        transport_y.append(vy.astype(np.float32))
        transport_confidence.append(vconfidence.astype(np.float32))
        previous_cartoon = cartoon
        previous_texture = texture
        previous_defect = defect

    keys = tuple(geometries[0])
    result: dict[str, np.ndarray | list[dict[str, float]]] = {
        "cartoon": np.stack(cartoons),
        "texture": np.stack(textures),
        "defect": np.stack(defects),
        "stage_stats": stage_stats,
        "transport_x": np.stack(transport_x),
        "transport_y": np.stack(transport_y),
        "transport_confidence": np.stack(transport_confidence),
    }
    for key in keys:
        result[key] = np.stack(
            [geometry[key] for geometry in geometries]).astype(np.float32)

    tx = np.asarray(result["transport_x"], dtype=np.float64)
    ty = np.asarray(result["transport_y"], dtype=np.float64)
    tc = np.asarray(result["transport_confidence"], dtype=np.float64)
    signed_x = np.sum(tc * tx, axis=0)
    signed_y = np.sum(tc * ty, axis=0)
    path_length = np.sum(tc * np.hypot(tx, ty), axis=0)
    result["transport_persistence"] = (
        np.hypot(signed_x, signed_y) /
        np.maximum(path_length, 1e-30)
    ).astype(np.float32)

    energy = np.asarray(result["energy"], dtype=np.float64)
    stage_axis = np.arange(1, passes + 1, dtype=np.float64)[:, None, None]
    integrated = np.sum(energy, axis=0)
    result["integrated_energy"] = integrated.astype(np.float32)
    result["birth_stage"] = (
        np.sum(energy * stage_axis, axis=0) /
        np.maximum(integrated, 1e-30)
    ).astype(np.float32)
    result["peak_stage"] = (
        np.argmax(energy, axis=0) + 1
    ).astype(np.int16)
    result["total_ms"] = np.asarray(
        (time.perf_counter() - started) * 1000.0)
    return result


def _robust_view(field: np.ndarray, symmetric: bool = False) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if symmetric:
        bound = max(float(np.percentile(np.abs(field), 99.5)), 1e-12)
        return np.clip(0.5 + 0.5 * field / bound, 0.0, 1.0)
    lo, hi = np.percentile(field, [1.0, 99.5])
    return np.clip((field - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)


def _orientation_rgb(
    angle: np.ndarray, coherence: np.ndarray, energy: np.ndarray,
) -> np.ndarray:
    hsv = np.empty((*angle.shape, 3), dtype=np.float64)
    hsv[..., 0] = np.mod(angle, np.pi) / np.pi
    hsv[..., 1] = 0.25 + 0.75 * coherence
    hsv[..., 2] = _robust_view(np.sqrt(np.maximum(energy, 0.0)))
    return hsv_to_rgb(hsv)


def _selected_stages(count: int) -> list[int]:
    candidates = [1, 2, 4, 8, 12, 16, 24, count]
    return sorted({min(max(stage, 1), count) for stage in candidates})


def save_panel(
    rgb: np.ndarray,
    volume: dict[str, np.ndarray | list[dict[str, float]]],
    output: Path,
) -> None:
    selected = _selected_stages(int(np.asarray(volume["cartoon"]).shape[0]))
    columns = len(selected)
    fig, axes = plt.subplots(
        5, columns, figsize=(3.0 * columns, 13.5), squeeze=False)
    cartoons = np.asarray(volume["cartoon"])
    textures = np.asarray(volume["texture"])
    defects = np.asarray(volume["defect"])
    energy = np.asarray(volume["energy"])
    coherence = np.asarray(volume["coherence"])
    angle = np.asarray(volume["angle"])
    minor = np.asarray(volume["minor"])
    major = np.asarray(volume["major"])

    for column, stage in enumerate(selected):
        index = stage - 1
        views = (
            (cartoons[index], "gray", "cartoon"),
            (_robust_view(textures[index], True), "coolwarm", "texture"),
            (_robust_view(defects[index], True), "coolwarm", "TV defect"),
            (_robust_view(np.sqrt(energy[index])), "magma", "event mass"),
        )
        for row, (view, cmap, label) in enumerate(views):
            axes[row, column].imshow(view, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[row, column].set_title(f"p={stage}  {label}")
        axes[4, column].imshow(
            _orientation_rgb(
                angle[index], coherence[index], energy[index]))
        step = max(5, min(rgb.shape[:2]) // 18)
        yy, xx = np.mgrid[
            step // 2:rgb.shape[0]:step,
            step // 2:rgb.shape[1]:step,
        ]
        aa = angle[index, yy, xx]
        rr = np.clip(
            major[index, yy, xx] / np.maximum(
                minor[index, yy, xx], 1e-9),
            1.0,
            8.0,
        )
        strength = coherence[index, yy, xx] * (
            _robust_view(np.sqrt(energy[index]))[yy, xx] > 0.12)
        axes[4, column].quiver(
            xx,
            yy,
            np.cos(aa) * rr * strength,
            np.sin(aa) * rr * strength,
            color="white",
            angles="xy",
            scale_units="xy",
            scale=0.55,
            width=0.004,
            alpha=0.75,
        )
        axes[4, column].set_title(f"p={stage}  inferred support")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "BFFT pass volume — structure appears as mass, scale, and direction",
        fontsize=15,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    rgb: np.ndarray,
    volume: dict[str, np.ndarray | list[dict[str, float]]],
    output: Path,
) -> None:
    energy = np.asarray(volume["energy"], dtype=np.float64)
    integrated = np.asarray(volume["integrated_energy"])
    birth_stage = np.asarray(volume["birth_stage"])
    peak_stage = np.asarray(volume["peak_stage"])
    ratio = np.asarray(volume["ratio"], dtype=np.float64)
    coherence = np.asarray(volume["coherence"], dtype=np.float64)
    weighted_ratio = np.sum(energy * ratio, axis=0) / np.maximum(
        integrated, 1e-30)
    weighted_coherence = np.sum(energy * coherence, axis=0) / np.maximum(
        integrated, 1e-30)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("input")
    axes[0, 1].imshow(_robust_view(np.sqrt(integrated)), cmap="magma")
    axes[0, 1].set_title("integrated event mass")
    axes[0, 2].imshow(birth_stage, cmap="turbo", vmin=1, vmax=energy.shape[0])
    axes[0, 2].set_title("mass-weighted emergence pass")
    axes[1, 0].imshow(peak_stage, cmap="turbo", vmin=1, vmax=energy.shape[0])
    axes[1, 0].set_title("peak event pass")
    axes[1, 1].imshow(
        np.clip(weighted_ratio, 1.0, 8.0), cmap="viridis", vmin=1, vmax=8)
    axes[1, 1].set_title("flow-weighted aspect ratio")
    axes[1, 2].imshow(
        weighted_coherence, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("flow-weighted directional confidence")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--passes", type=int, default=24)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=40.0)
    parser.add_argument("--flow-sweeps", type=int, default=32)
    parser.add_argument("--tensor-sigma", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/out/flow_stage_geometry.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "experiments/out/flow_stage_summary.png",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "experiments/out/flow_stage_geometry.json",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=ROOT / "experiments/out/flow_stage_geometry.npz",
    )
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    rgb = _fit_rgb(image, args.side)
    volume = build_flow_volume(
        rgb,
        passes=args.passes,
        lam=args.lam,
        mu=args.mu,
        flow_sweeps=args.flow_sweeps,
        tensor_sigma=args.tensor_sigma,
    )
    save_panel(rgb, volume, args.output)
    save_summary(rgb, volume, args.summary)

    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.npz,
        **{
            key: value
            for key, value in volume.items()
            if isinstance(value, np.ndarray)
        },
    )
    report = {
        "source": source,
        "shape": list(rgb.shape),
        "passes": args.passes,
        "lam": args.lam,
        "mu": args.mu,
        "flow_sweeps": args.flow_sweeps,
        "tensor_sigma": args.tensor_sigma,
        "total_ms": float(np.asarray(volume["total_ms"])),
        "stages": volume["stage_stats"],
        "outputs": {
            "panel": str(args.output.resolve()),
            "summary": str(args.summary.resolve()),
            "volume": str(args.npz.resolve()),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
