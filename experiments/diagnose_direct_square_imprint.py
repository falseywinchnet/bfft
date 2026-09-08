"""Locate the cellular residual in direct all-source chart fusion.

This is a mechanism diagnostic, not an interpolator benchmark.  It compares
the current squared source-response fusion with the ordinary barycentric
fusion induced by the same Eikonal actions, frames, chart proposals, and
CONV synthesis.  Consequently any difference is caused only by the support
product law.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
for directory in (ROOT, DEMO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend import conv_resize  # noqa: E402
from experiments.conv_fermi_owner_demo import (  # noqa: E402
    _distance_response,
    _distance_scales,
    _target_action,
    build_direct_distance_atlas,
    synthesize_distance_charts,
)


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(-1.0, 1.0, shape[0], dtype=np.float64)
    x = np.linspace(-1.0, 1.0, shape[1], dtype=np.float64)
    return np.meshgrid(x, y)


def _interface45(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = (x + y) / math.sqrt(2.0)
    return 0.5 + 0.4 * np.tanh((q - 0.035) / 0.055)


def _support_fields(atlas, target_shape: tuple[int, int]):
    scales = _distance_scales(atlas)
    raw = np.stack([
        _distance_response(
            _target_action(distance, target_shape), scales[index],
            "inverse_power_2",
        )
        for index, distance in enumerate(atlas.distances)
    ])
    weight = raw / np.sum(raw, axis=0, keepdims=True)
    compatibility = np.asarray([
        frame.transport_compatibility for frame in atlas.frames
    ], dtype=np.float64)[:, None, None]
    return {
        "herfindahl": np.sum(weight * weight, axis=0),
        "linear_mass": np.sum(compatibility * weight, axis=0),
        "squared_mass": np.sum(compatibility * weight * weight, axis=0),
        "winner_weight": np.max(weight, axis=0),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    aa -= np.mean(aa)
    bb -= np.mean(bb)
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator else 0.0


def run(output: Path) -> None:
    coarse_shape = (25, 27)
    fine_shape = (97, 105)
    coarse = _interface45(*_grid(coarse_shape))
    truth = _interface45(*_grid(fine_shape))
    atlas = build_direct_distance_atlas(coarse)
    squared = np.asarray(synthesize_distance_charts(
        coarse, fine_shape, atlas, support_power=2.0
    ), dtype=np.float64)
    linear = np.asarray(synthesize_distance_charts(
        coarse, fine_shape, atlas, support_power=1.0
    ), dtype=np.float64)
    baseline = np.asarray(conv_resize(coarse, fine_shape), dtype=np.float64)
    fields = _support_fields(atlas, fine_shape)

    residual_squared = squared - truth
    residual_linear = linear - truth
    fusion_delta = squared - linear
    mass_deficit = fields["linear_mass"] - fields["squared_mass"]
    report = {
        "coarse_shape": coarse_shape,
        "fine_shape": fine_shape,
        "source_count": len(atlas.frames),
        "mse": {
            "cartesian": float(np.mean((baseline - truth) ** 2)),
            "squared_support": float(np.mean(residual_squared ** 2)),
            "linear_support": float(np.mean(residual_linear ** 2)),
        },
        "maximum_absolute_error": {
            "squared_support": float(np.max(np.abs(residual_squared))),
            "linear_support": float(np.max(np.abs(residual_linear))),
        },
        "support": {
            "herfindahl_min": float(np.min(fields["herfindahl"])),
            "herfindahl_max": float(np.max(fields["herfindahl"])),
            "linear_mass_min": float(np.min(fields["linear_mass"])),
            "linear_mass_max": float(np.max(fields["linear_mass"])),
            "squared_mass_min": float(np.min(fields["squared_mass"])),
            "squared_mass_max": float(np.max(fields["squared_mass"])),
        },
        "correlation": {
            "absolute_fusion_delta_vs_mass_deficit": _corr(
                np.abs(fusion_delta), mass_deficit
            ),
            "squared_residual_energy_vs_mass_deficit": _corr(
                residual_squared ** 2, mass_deficit
            ),
            "linear_residual_energy_vs_mass_deficit": _corr(
                residual_linear ** 2, mass_deficit
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    panels = (
        (truth, "analytic truth", "gray", 0.1, 0.9),
        (squared, "direct: squared support", "gray", 0.1, 0.9),
        (linear, "direct: barycentric support", "gray", 0.1, 0.9),
        (fields["herfindahl"], r"$\sum_a w_a^2$", "viridis", None, None),
        (fields["squared_mass"], r"$\sum_a\kappa_a w_a^2$", "viridis", None, None),
        (fields["linear_mass"], r"$\sum_a\kappa_a w_a$", "viridis", None, None),
        (residual_squared, "squared-support residual", "coolwarm", None, None),
        (residual_linear, "barycentric residual", "coolwarm", None, None),
        (fusion_delta, "squared minus barycentric", "coolwarm", None, None),
    )
    residual_limit = max(
        float(np.max(np.abs(residual_squared))),
        float(np.max(np.abs(residual_linear))),
        float(np.max(np.abs(fusion_delta))),
    )
    figure, axes = plt.subplots(3, 3, figsize=(12.0, 10.0), constrained_layout=True)
    for index, (image, title, cmap, vmin, vmax) in enumerate(panels):
        axis = axes.flat[index]
        if index >= 6:
            vmin, vmax = -residual_limit, residual_limit
        rendered = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax,
                               interpolation="nearest")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if index in (3, 4, 5):
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.02)
    figure.suptitle(
        "Direct action-frame cellular imprint: only the support exponent changes\n"
        f"MSE squared={report['mse']['squared_support']:.6e}, "
        f"barycentric={report['mse']['linear_support']:.6e}"
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    run(ROOT / "output/support_geometry/direct_square_imprint.png")
