"""Localized Eikonal-path correlogram for interpolation residuals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import rotate


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
for directory in (ROOT, DEMO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend import conv_basin_average, conv_resize, lanczos3_resize  # noqa: E402
from experiments.conv_fermi_owner_demo import (  # noqa: E402
    eikonal_basin_conv_resize,
)
from experiments.eikonal_current_pushforward_diagnostic import (  # noqa: E402
    analytic_feature,
    eikonal_fiber_resize,
)


Array = np.ndarray


@dataclass(frozen=True)
class PathOscillation:
    maximum_negative_product_moment: float
    pearson_at_maximum: float
    orientation_degrees: float
    rotated_row: int
    window_start: int
    window_stop: int
    lag: int
    residual_derivative_sign_changes: int
    reference_derivative_sign_changes: int
    estimate_derivative_sign_changes: int
    derivative_sign_surplus: int
    examined_windows: int
    qualifying_windows: int
    maximum_ringing_product_moment: float
    ringing_pearson: float
    ringing_orientation_degrees: float
    ringing_lag: int
    ringing_derivative_sign_surplus: int
    ringing_rotated_row: int
    ringing_window_start: int
    ringing_window_stop: int
    ringing_reference_sign_changes: int
    ringing_estimate_sign_changes: int
    maximum_excess_backtracking: float
    backtracking_orientation_degrees: float
    backtracking_lag: int
    backtracking_reference: float
    backtracking_estimate: float


def _sign_changes(values: Array) -> int:
    difference = np.diff(np.asarray(values, dtype=np.float64))
    if difference.size == 0:
        return 0
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(values))), 1.0
    )
    sign = np.sign(np.where(np.abs(difference) > tolerance, difference, 0.0))
    sign = sign[sign != 0.0]
    return int(np.count_nonzero(sign[1:] != sign[:-1])) if sign.size else 0


def _rotated_lines(values: Array, angle: float) -> tuple[Array, Array]:
    field = np.asarray(values, dtype=np.float64)
    rotated = rotate(
        field, -float(angle), reshape=True, order=1,
        mode="constant", cval=0.0, prefilter=False,
    )
    mask = rotate(
        np.ones(field.shape, dtype=np.float64), -float(angle),
        reshape=True, order=1, mode="constant", cval=0.0, prefilter=False,
    )
    return rotated, mask >= 1.0 - 128.0 * np.finfo(np.float64).eps


def eikonal_path_oscillation(
    estimate: Array,
    reference: Array,
    *,
    angles: Array | None = None,
    maximum_lag: int = 12,
    cycles: int = 2,
) -> PathOscillation:
    """Return the strongest supported alternating residual correlogram entry.

    A lag ``l`` uses windows of length ``2*cycles*l``.  A window qualifies
    only if its residual derivative contains at least ``2*cycles-1`` sign
    transitions; a single spike therefore cannot qualify as an oscillation.
    """

    trial = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    if trial.shape != truth.shape or trial.ndim != 2:
        raise ValueError("diagnostic requires equal scalar HxW rasters")
    if angles is None:
        angles = np.arange(0.0, 180.0, 5.0, dtype=np.float64)
    maximum_lag = max(int(maximum_lag), 1)
    cycles = max(int(cycles), 2)
    best: dict[str, object] | None = None
    best_ringing: dict[str, object] | None = None
    best_backtracking: dict[str, object] | None = None
    examined = 0
    qualifying = 0

    residual = trial - truth
    for angle in np.asarray(angles, dtype=np.float64):
        turned_residual, valid = _rotated_lines(residual, float(angle))
        turned_truth, _ = _rotated_lines(truth, float(angle))
        turned_trial, _ = _rotated_lines(trial, float(angle))
        for row in range(turned_residual.shape[0]):
            indices = np.flatnonzero(valid[row])
            if indices.size < 4 * cycles:
                continue
            # Convexity of a rotated rectangle makes the fully supported part
            # of each raster row a single interval.
            begin, end = int(indices[0]), int(indices[-1]) + 1
            sequence = turned_residual[row, begin:end]
            truth_line = turned_truth[row, begin:end]
            trial_line = turned_trial[row, begin:end]
            for lag in range(1, min(maximum_lag, sequence.size // (2 * cycles)) + 1):
                width = 2 * cycles * lag
                windows = np.lib.stride_tricks.sliding_window_view(sequence, width)
                truth_windows = np.lib.stride_tricks.sliding_window_view(
                    truth_line, width
                )
                trial_windows = np.lib.stride_tricks.sliding_window_view(
                    trial_line, width
                )
                examined += int(windows.shape[0])
                truth_backtracking = (
                    np.sum(np.abs(np.diff(truth_windows, axis=1)), axis=1)
                    - np.abs(truth_windows[:, -1] - truth_windows[:, 0])
                )
                trial_backtracking = (
                    np.sum(np.abs(np.diff(trial_windows, axis=1)), axis=1)
                    - np.abs(trial_windows[:, -1] - trial_windows[:, 0])
                )
                excess_backtracking = np.maximum(
                    trial_backtracking - truth_backtracking, 0.0
                )
                back_index = int(np.argmax(excess_backtracking))
                back_value = float(excess_backtracking[back_index])
                if back_value > 0.0 and (
                    best_backtracking is None
                    or back_value > best_backtracking["excess"]
                ):
                    best_backtracking = {
                        "excess": back_value,
                        "angle": float(angle) % 180.0,
                        "lag": lag,
                        "reference": float(truth_backtracking[back_index]),
                        "estimate": float(trial_backtracking[back_index]),
                    }
                left = windows[:, :-lag]
                right = windows[:, lag:]
                left_centered = left - np.mean(left, axis=1, keepdims=True)
                right_centered = right - np.mean(right, axis=1, keepdims=True)
                covariance = np.mean(left_centered * right_centered, axis=1)
                candidates = np.flatnonzero(covariance < 0.0)
                if candidates.size == 0:
                    continue
                # Only the strongest supported window for this path and lag
                # can improve the global maximum.  Visit negative moments in
                # descending magnitude and stop at the first with repeated
                # alternation.
                candidates = candidates[np.argsort(covariance[candidates])]
                for start in candidates:
                    product = -float(covariance[start])
                    if (
                        best is not None and best_ringing is not None
                        and product <= best["product"]
                        and product <= best_ringing["product"]
                    ):
                        break
                    window = windows[start]
                    residual_changes = _sign_changes(window)
                    if residual_changes < 2 * cycles - 1:
                        continue
                    qualifying += 1
                    x = left_centered[start]
                    y = right_centered[start]
                    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
                    pearson = float(np.dot(x, y) / denominator) if denominator else 0.0
                    stop = int(start + width)
                    reference_window = truth_line[start:stop]
                    estimate_window = trial_line[start:stop]
                    reference_changes = _sign_changes(reference_window)
                    estimate_changes = _sign_changes(estimate_window)
                    witness = {
                        "product": product,
                        "pearson": pearson,
                        "angle": float(angle) % 180.0,
                        "row": row,
                        "start": begin + int(start),
                        "stop": begin + stop,
                        "lag": lag,
                        "residual_changes": residual_changes,
                        "reference_changes": reference_changes,
                        "estimate_changes": estimate_changes,
                    }
                    if best is None or product > best["product"]:
                        best = witness
                    surplus = max(estimate_changes - reference_changes, 0)
                    if surplus > 0 and (
                        best_ringing is None or product > best_ringing["product"]
                    ):
                        best_ringing = witness
                        break
    if best is None:
        return PathOscillation(
            0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, examined, qualifying,
            0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0,
            0.0, 0.0, 0, 0.0, 0.0,
        )
    return PathOscillation(
        maximum_negative_product_moment=float(best["product"]),
        pearson_at_maximum=float(best["pearson"]),
        orientation_degrees=float(best["angle"]),
        rotated_row=int(best["row"]),
        window_start=int(best["start"]),
        window_stop=int(best["stop"]),
        lag=int(best["lag"]),
        residual_derivative_sign_changes=int(best["residual_changes"]),
        reference_derivative_sign_changes=int(best["reference_changes"]),
        estimate_derivative_sign_changes=int(best["estimate_changes"]),
        derivative_sign_surplus=max(
            int(best["estimate_changes"]) - int(best["reference_changes"]), 0
        ),
        examined_windows=examined,
        qualifying_windows=qualifying,
        maximum_ringing_product_moment=(
            float(best_ringing["product"]) if best_ringing is not None else 0.0
        ),
        ringing_pearson=(
            float(best_ringing["pearson"]) if best_ringing is not None else 0.0
        ),
        ringing_orientation_degrees=(
            float(best_ringing["angle"]) if best_ringing is not None else 0.0
        ),
        ringing_lag=int(best_ringing["lag"]) if best_ringing is not None else 0,
        ringing_derivative_sign_surplus=(
            max(
                int(best_ringing["estimate_changes"])
                - int(best_ringing["reference_changes"]), 0,
            ) if best_ringing is not None else 0
        ),
        ringing_rotated_row=(
            int(best_ringing["row"]) if best_ringing is not None else 0
        ),
        ringing_window_start=(
            int(best_ringing["start"]) if best_ringing is not None else 0
        ),
        ringing_window_stop=(
            int(best_ringing["stop"]) if best_ringing is not None else 0
        ),
        ringing_reference_sign_changes=(
            int(best_ringing["reference_changes"]) if best_ringing is not None else 0
        ),
        ringing_estimate_sign_changes=(
            int(best_ringing["estimate_changes"]) if best_ringing is not None else 0
        ),
        maximum_excess_backtracking=(
            float(best_backtracking["excess"])
            if best_backtracking is not None else 0.0
        ),
        backtracking_orientation_degrees=(
            float(best_backtracking["angle"])
            if best_backtracking is not None else 0.0
        ),
        backtracking_lag=(
            int(best_backtracking["lag"]) if best_backtracking is not None else 0
        ),
        backtracking_reference=(
            float(best_backtracking["reference"])
            if best_backtracking is not None else 0.0
        ),
        backtracking_estimate=(
            float(best_backtracking["estimate"])
            if best_backtracking is not None else 0.0
        ),
    )


def run_audit(side: int = 129, factor: int = 4) -> dict[str, object]:
    angle = 37.5
    radians = math.radians(angle)
    normal = np.array((-math.sin(radians), math.cos(radians)))
    # The analytic audit supplies its exact tangent/normal Eikonal bases plus
    # the two Cartesian controls.  A general orientation census is a separate
    # implementation problem and is not approximated by an arbitrary angle
    # grid here.
    angles = np.unique(np.mod(np.array((
        0.0, 90.0, angle, angle + 90.0, -angle, 90.0 - angle,
    )), 180.0))
    result: dict[str, object] = {}
    for family, width in (("ridge", 1.05), ("edge", 0.8)):
        truth = analytic_feature(side, angle, 0.35, family, width).astype(np.float32)
        coarse_side = (side - 1) // factor + 1
        coarse = conv_basin_average(truth, (coarse_side, coarse_side))
        candidates = {
            "CONV_basin_cycle": conv_resize(coarse, truth.shape),
            "EB_CONV_cycle": eikonal_basin_conv_resize(coarse, truth.shape),
            "Lanczos3_cycle": lanczos3_resize(coarse, truth.shape),
            "exact_eikonal_fiber_cycle": eikonal_fiber_resize(
                coarse, truth.shape, normal
            ),
        }
        result[family] = {
            name: asdict(eikonal_path_oscillation(
                value, truth, angles=angles, maximum_lag=12, cycles=2
            ))
            for name, value in candidates.items()
        }
    return {
        "definition": "maximum supported negative residual autocovariance on straight Eikonal paths",
        "side": side,
        "factor": factor,
        "families": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=129)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "output/support_geometry/eikonal_path_oscillation/audit.json",
    )
    args = parser.parse_args()
    result = run_audit(args.side, args.factor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
