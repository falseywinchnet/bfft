#!/usr/bin/env python3
"""Infer a final support measure directly from the BFFT transport stack.

The decomposition-pass axis is not biological time.  There is no persistent
population, parent, child, birth, death, or requested cell budget.  Each pass
states a local support requirement.  Consecutive passes also state how an
earlier requirement maps into the next state.

For normalized event tensor Q, the density of locally admissible supports is

    rho* = sqrt(det(R Q + l_max^-2 I)) / pi,

where R suppresses undefined tensor directions when the event amplitude is
numerically negligible.  The final measure is the transported upper envelope

    rho_p = max(rho*_p, T_p rho_{p-1}).

T is a conservative push-forward through the measured inter-pass flow.  The
maximum is a union of support obligations, not population accumulation.  Its
integral gives the implied count in one shot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from bfft_flow_stage_geometry import build_flow_volume  # noqa: E402
from transport_voronoi import _fit_rgb  # noqa: E402


def _load_image(path: str | None, gallery_key: str) -> tuple[np.ndarray, str]:
    if path:
        from skimage.io import imread

        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def push_measure(
    measure: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> np.ndarray:
    """Push a scalar measure forward with exactly conserved integral."""
    height, width = measure.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    destination_x = np.clip(xx + vx, 0.0, width - 1.0)
    destination_y = np.clip(yy + vy, 0.0, height - 1.0)
    x0 = np.floor(destination_x).astype(np.intp)
    y0 = np.floor(destination_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = destination_x - x0
    fy = destination_y - y0
    pushed = np.zeros_like(measure, dtype=np.float64)
    for yi, xi, weight in (
        (y0, x0, (1.0 - fx) * (1.0 - fy)),
        (y0, x1, fx * (1.0 - fy)),
        (y1, x0, (1.0 - fx) * fy),
        (y1, x1, fx * fy),
    ):
        np.add.at(
            pushed,
            (yi.ravel(), xi.ravel()),
            (measure * weight).ravel(),
        )
    return pushed


def infer_support_measure(
    volume: dict,
    max_support_fraction: float = 0.18,
    transport_strength: float = 1.0,
) -> dict:
    """Return the transported union of all pass-local support obligations."""
    energy = np.asarray(volume["energy"], dtype=np.float64)
    high = np.asarray(volume["high_frequency"], dtype=np.float64)
    low = np.asarray(volume["low_frequency"], dtype=np.float64)
    angle = np.asarray(volume["angle"], dtype=np.float64)
    tx = np.asarray(volume["transport_x"], dtype=np.float64)
    ty = np.asarray(volume["transport_y"], dtype=np.float64)
    confidence = np.asarray(
        volume["transport_confidence"], dtype=np.float64)
    persistence = np.asarray(
        volume["transport_persistence"], dtype=np.float64)
    stages, height, width = energy.shape

    max_length = max(
        float(max_support_fraction) * max(height, width), 1.0)
    frequency_floor = 1.0 / (max_length * max_length)

    # Q is normalized by event energy and is therefore unreliable where the
    # event is numerically absent.  Reliability gates Q itself.  The isotropic
    # horizon remains, giving a finite broad-support measure without an
    # arbitrary initial cell count.
    amplitude = np.sqrt(np.maximum(energy, 0.0))
    stage_scale = np.maximum(
        np.percentile(amplitude, 99.5, axis=(1, 2), keepdims=True),
        1e-30,
    )
    reliability = amplitude / (amplitude + 1e-5 * stage_scale)
    local_high = reliability * high + frequency_floor
    local_low = reliability * low + frequency_floor
    required = np.sqrt(local_high * local_low) / math.pi

    # Reconstruct the local precision tensor.  ``angle`` is its low-frequency
    # tangent eigenvector; the orthogonal direction has eigenvalue local_high.
    tangent_x = np.cos(angle)
    tangent_y = np.sin(angle)
    normal_x = -tangent_y
    normal_y = tangent_x
    local_qxx = (
        local_low * tangent_x * tangent_x
        + local_high * normal_x * normal_x)
    local_qxy = (
        local_low * tangent_x * tangent_y
        + local_high * normal_x * normal_y)
    local_qyy = (
        local_low * tangent_y * tangent_y
        + local_high * normal_y * normal_y)

    envelope = required[0].copy()
    qxx = local_qxx[0].copy()
    qxy = local_qxy[0].copy()
    qyy = local_qyy[0].copy()
    envelopes = [envelope.astype(np.float32)]
    pushed_fields = [np.zeros_like(envelope, dtype=np.float32)]
    binding_fields = [required[0].astype(np.float32)]
    stage_stats = [{
        "stage": 1,
        "local_requirement": float(np.sum(required[0])),
        "transported_prior_requirement": 0.0,
        "transported_prior_mass_error": 0.0,
        "newly_binding_measure": float(np.sum(required[0])),
        "envelope_measure": float(np.sum(envelope)),
    }]

    for stage in range(1, stages):
        gate = confidence[stage] * persistence
        vx = float(transport_strength) * gate * tx[stage]
        vy = float(transport_strength) * gate * ty[stage]
        previous_mass = float(np.sum(envelope))
        pushed = push_measure(envelope, vx, vy)
        pushed_qxx = push_measure(envelope * qxx, vx, vy)
        pushed_qxy = push_measure(envelope * qxy, vx, vy)
        pushed_qyy = push_measure(envelope * qyy, vx, vy)
        safe_pushed = np.maximum(pushed, 1e-30)
        pushed_qxx /= safe_pushed
        pushed_qxy /= safe_pushed
        pushed_qyy /= safe_pushed
        binding = np.maximum(required[stage] - pushed, 0.0)
        local_binds = required[stage] >= pushed
        envelope = np.where(local_binds, required[stage], pushed)
        qxx = np.where(local_binds, local_qxx[stage], pushed_qxx)
        qxy = np.where(local_binds, local_qxy[stage], pushed_qxy)
        qyy = np.where(local_binds, local_qyy[stage], pushed_qyy)
        stage_stats.append({
            "stage": stage + 1,
            "local_requirement": float(np.sum(required[stage])),
            "transported_prior_requirement": float(np.sum(pushed)),
            "transported_prior_mass_error": float(
                np.sum(pushed) - previous_mass),
            "newly_binding_measure": float(np.sum(binding)),
            "envelope_measure": float(np.sum(envelope)),
        })
        pushed_fields.append(pushed.astype(np.float32))
        binding_fields.append(binding.astype(np.float32))
        envelopes.append(envelope.astype(np.float32))

    static_envelope = np.max(required, axis=0)
    binding_stack = np.stack(binding_fields).astype(np.float32)
    stage_axis = np.arange(
        1, stages + 1, dtype=np.float64)[:, None, None]
    binding_mass = np.sum(binding_stack, axis=0)
    return {
        "local_requirement": required.astype(np.float32),
        "transported_prior": np.stack(pushed_fields),
        "newly_binding": binding_stack,
        "envelope": np.stack(envelopes),
        "static_envelope": static_envelope.astype(np.float32),
        "precision_xx": qxx.astype(np.float32),
        "precision_xy": qxy.astype(np.float32),
        "precision_yy": qyy.astype(np.float32),
        "scale_mean": (
            np.sum(binding_stack * stage_axis, axis=0)
            / np.maximum(binding_mass, 1e-30)
        ).astype(np.float32),
        "local_precision_xx": local_qxx.astype(np.float32),
        "local_precision_xy": local_qxy.astype(np.float32),
        "local_precision_yy": local_qyy.astype(np.float32),
        "stage_stats": stage_stats,
        "static_count": float(np.sum(static_envelope)),
        "transported_count": float(np.sum(envelope)),
        "broad_horizon_count": float(
            height * width * frequency_floor / math.pi),
        "max_support_px": max_length,
    }


def _normalize(field: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    scale = max(float(np.percentile(field, percentile)), 1e-30)
    return np.clip(field / scale, 0.0, 1.0)


def save_summary(
    rgb: np.ndarray,
    result: dict,
    output: Path,
) -> None:
    stats = result["stage_stats"]
    passes = [record["stage"] for record in stats]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("input")
    axes[0, 1].imshow(
        _normalize(result["static_envelope"]), cmap="viridis")
    axes[0, 1].set_title(
        "untransported support union\n"
        f"{result['static_count']:.0f} implied cells")
    axes[0, 2].imshow(
        _normalize(result["envelope"][-1]), cmap="viridis")
    axes[0, 2].set_title(
        "transported support union\n"
        f"{result['transported_count']:.0f} implied cells")

    axes[1, 0].plot(
        passes,
        [record["local_requirement"] for record in stats],
        label="pass-local requirement",
    )
    axes[1, 0].plot(
        passes,
        [record["envelope_measure"] for record in stats],
        label="transported union",
    )
    axes[1, 0].set_title("count is an integral of the support measure")
    axes[1, 0].set_xlabel("BFFT pass")
    axes[1, 0].set_ylabel("implied cells")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(
        passes, [record["newly_binding_measure"] for record in stats])
    axes[1, 1].set_title("newly binding support—not births")
    axes[1, 1].set_xlabel("BFFT pass")
    axes[1, 1].set_ylabel("measure")

    mass_errors = np.asarray([
        record["transported_prior_mass_error"] for record in stats])
    axes[1, 2].plot(passes, mass_errors)
    axes[1, 2].set_title("transport conservation audit")
    axes[1, 2].set_xlabel("BFFT pass")
    axes[1, 2].set_ylabel("mass error")
    for axis in axes[0]:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "One-shot support measure from the BFFT transport stack",
        fontsize=15,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--passes", type=int, default=24)
    parser.add_argument("--max-support-fraction", type=float, default=0.18)
    parser.add_argument("--transport-strength", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/out/flow_support_measure.png",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "experiments/out/flow_support_measure.json",
    )
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    rgb = _fit_rgb(image, args.side)
    volume = build_flow_volume(rgb, passes=args.passes)
    result = infer_support_measure(
        volume,
        max_support_fraction=args.max_support_fraction,
        transport_strength=args.transport_strength,
    )
    save_summary(rgb, result, args.output)
    report = {
        "source": source,
        "side": args.side,
        "passes": args.passes,
        "max_support_fraction": args.max_support_fraction,
        "max_support_px": result["max_support_px"],
        "broad_horizon_count": result["broad_horizon_count"],
        "static_count": result["static_count"],
        "transported_count": result["transported_count"],
        "stage_stats": result["stage_stats"],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
