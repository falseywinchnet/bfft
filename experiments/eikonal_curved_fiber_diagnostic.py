"""Straight and radial Eikonal-fiber cure for Cartesian phase residuals."""

from __future__ import annotations

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

from experiments.benchmark_eikonal_frame_bank import (  # noqa: E402
    _cases,
    _evaluate_methods,
    _grid,
)
from experiments.eikonal_current_pushforward_diagnostic import (  # noqa: E402
    _evaluate_profile_many,
    _profiles_along_axis,
    eikonal_fiber_resize,
)


CASES = (
    ("interface_45", (25, 27), (97, 105), "straight", math.pi / 4.0, 0.035),
    ("filament_45", (17, 19), (65, 73), "straight", math.pi / 4.0, 0.08),
    ("circle_r0.62", (17, 19), (65, 73), "radial", 0.0, 0.0),
    ("radial_chirp_p0.00", (17, 19), (65, 73), "radial", 0.0, 0.0),
)


def _radial_fiber_resize(
    values: np.ndarray,
    target_shape: tuple[int, int],
    center: tuple[float, float] = (-0.08, 0.06),
) -> np.ndarray:
    """Positive interpolation on the observed irregular radial actions."""

    source = np.asarray(values, dtype=np.float64)
    source_x, source_y = _grid(source.shape)
    target_x, target_y = _grid(target_shape)
    radius = np.hypot(source_x - center[0], source_y - center[1])
    target_radius = np.hypot(
        target_x - center[0], target_y - center[1]
    )
    # Cartesian samples are nonuniform in radial action.  Rebinning them on
    # the Cartesian step destroys precisely the off-grid phases that make the
    # radial acquisition informative.  Equal actions are averaged; distinct
    # observed actions remain distinct interpolation anchors.
    action = radius.ravel()
    samples = source.ravel()
    order = np.argsort(action, kind="stable")
    action = action[order]
    samples = samples[order]
    unique, inverse = np.unique(action, return_inverse=True)
    mass = np.bincount(inverse).astype(np.float64)
    moment = np.bincount(inverse, weights=samples)
    profile = moment / mass
    result = np.interp(
        target_radius.ravel(), unique, profile,
        left=float(profile[0]), right=float(profile[-1]),
    )
    return result.reshape(target_shape).astype(np.float32)


def _straight_normal(
    angle: float, source_shape: tuple[int, int]
) -> tuple[np.ndarray, float]:
    gradient = np.array((
        math.cos(angle) * 2.0 / (source_shape[1] - 1),
        math.sin(angle) * 2.0 / (source_shape[0] - 1),
    ))
    scale = float(np.linalg.norm(gradient))
    return gradient / scale, scale


def _ordered_path_value(
    coordinate: list[float], value: list[float], query: float
) -> float:
    order = np.argsort(np.asarray(coordinate), kind="stable")
    x = np.asarray(coordinate, dtype=np.float64)[order]
    y = np.asarray(value, dtype=np.float64)[order]
    if x.size == 0:
        raise RuntimeError("characteristic did not cross the source lattice")
    # A grid vertex is reported by one row and one column.  It is one
    # geometric witness, so coincident actions are averaged before transport.
    group = np.zeros(x.size, dtype=np.intp)
    if x.size > 1:
        group[1:] = np.cumsum(np.diff(x) > 128.0 * np.finfo(np.float64).eps)
    mass = np.bincount(group).astype(np.float64)
    action = np.bincount(group, weights=x) / mass
    moment = np.bincount(group, weights=y) / mass
    return float(np.interp(query, action, moment, left=moment[0], right=moment[-1]))


def _straight_two_current_resize(
    values: np.ndarray,
    target_shape: tuple[int, int],
    angle: float,
) -> np.ndarray:
    """Transport along exact straight characteristics at constant companion."""

    source = np.asarray(values, dtype=np.float64)
    height, width = source.shape
    row_profiles = _profiles_along_axis(source, 1)
    column_profiles = _profiles_along_axis(source, 0)
    source_y = np.linspace(-1.0, 1.0, height)
    source_x = np.linspace(-1.0, 1.0, width)
    target_x, target_y = _grid(target_shape)
    normal = np.array((math.cos(angle), math.sin(angle)))
    tangent = np.array((-normal[1], normal[0]))
    phi = normal[0] * target_x + normal[1] * target_y
    psi = tangent[0] * target_x + tangent[1] * target_y
    result = np.empty(target_shape, dtype=np.float64)
    tolerance = 128.0 * np.finfo(np.float64).eps
    for target_index in np.ndindex(target_shape):
        level = float(psi[target_index])
        coordinate: list[float] = []
        value: list[float] = []
        if abs(tangent[0]) > tolerance:
            for row, y_value in enumerate(source_y):
                x_value = (level - tangent[1] * y_value) / tangent[0]
                if -1.0 - tolerance <= x_value <= 1.0 + tolerance:
                    source_position = np.clip(
                        0.5 * (x_value + 1.0) * (width - 1), 0.0, width - 1.0
                    )
                    coordinate.append(float(normal[0] * x_value + normal[1] * y_value))
                    value.append(float(_evaluate_profile_many(
                        row_profiles[row], np.array([source_position])
                    )[0]))
        if abs(tangent[1]) > tolerance:
            for column, x_value in enumerate(source_x):
                y_value = (level - tangent[0] * x_value) / tangent[1]
                if -1.0 - tolerance <= y_value <= 1.0 + tolerance:
                    source_position = np.clip(
                        0.5 * (y_value + 1.0) * (height - 1), 0.0, height - 1.0
                    )
                    coordinate.append(float(normal[0] * x_value + normal[1] * y_value))
                    value.append(float(_evaluate_profile_many(
                        column_profiles[column], np.array([source_position])
                    )[0]))
        result[target_index] = _ordered_path_value(
            coordinate, value, float(phi[target_index])
        )
    return result.astype(np.float32)


def _radial_two_current_resize(
    values: np.ndarray,
    target_shape: tuple[int, int],
    center: tuple[float, float] = (-0.08, 0.06),
) -> np.ndarray:
    """Transport on radial characteristics with polar angle as companion."""

    source = np.asarray(values, dtype=np.float64)
    height, width = source.shape
    row_profiles = _profiles_along_axis(source, 1)
    column_profiles = _profiles_along_axis(source, 0)
    source_y = np.linspace(-1.0, 1.0, height)
    source_x = np.linspace(-1.0, 1.0, width)
    target_x, target_y = _grid(target_shape)
    dx = target_x - center[0]
    dy = target_y - center[1]
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    result = np.empty(target_shape, dtype=np.float64)
    tolerance = 128.0 * np.finfo(np.float64).eps
    for target_index in np.ndindex(target_shape):
        theta = float(angle[target_index])
        cosine, sine = math.cos(theta), math.sin(theta)
        coordinate: list[float] = []
        value: list[float] = []
        if abs(sine) > tolerance:
            for row, y_value in enumerate(source_y):
                action = (y_value - center[1]) / sine
                x_value = center[0] + action * cosine
                if action >= 0.0 and -1.0 - tolerance <= x_value <= 1.0 + tolerance:
                    source_position = np.clip(
                        0.5 * (x_value + 1.0) * (width - 1), 0.0, width - 1.0
                    )
                    coordinate.append(float(action))
                    value.append(float(_evaluate_profile_many(
                        row_profiles[row], np.array([source_position])
                    )[0]))
        if abs(cosine) > tolerance:
            for column, x_value in enumerate(source_x):
                action = (x_value - center[0]) / cosine
                y_value = center[1] + action * sine
                if action >= 0.0 and -1.0 - tolerance <= y_value <= 1.0 + tolerance:
                    source_position = np.clip(
                        0.5 * (y_value + 1.0) * (height - 1), 0.0, height - 1.0
                    )
                    coordinate.append(float(action))
                    value.append(float(_evaluate_profile_many(
                        column_profiles[column], np.array([source_position])
                    )[0]))
        result[target_index] = _ordered_path_value(
            coordinate, value, float(radius[target_index])
        )
    return result.astype(np.float32)


def run(output: Path) -> None:
    functions = {name: function for _, name, function in _cases()}
    rows = []
    for name, coarse_shape, fine_shape, geometry, angle, offset in CASES:
        coarse_x, coarse_y = _grid(coarse_shape)
        fine_x, fine_y = _grid(fine_shape)
        function = functions[name]
        coarse = np.clip(function(coarse_x, coarse_y), 0.0, 1.0).astype(np.float32)
        truth = np.clip(function(fine_x, fine_y), 0.0, 1.0).astype(np.float64)
        results = _evaluate_methods(coarse, fine_shape)
        if geometry == "straight":
            normal, _ = _straight_normal(angle, coarse_shape)
            eikonal = eikonal_fiber_resize(coarse, fine_shape, normal)
            two_current = _straight_two_current_resize(
                coarse, fine_shape, angle
            )
        else:
            eikonal = _radial_fiber_resize(coarse, fine_shape)
            two_current = _radial_two_current_resize(coarse, fine_shape)
        estimates = {
            "owner": np.asarray(results["owner_frames"], dtype=np.float64),
            "direct": np.asarray(results["direct_frames_inverse_2"], dtype=np.float64),
            "Lanczos": np.asarray(results["lanczos3_sinc"], dtype=np.float64),
            "Eikonal fiber": np.asarray(eikonal, dtype=np.float64),
            "Eikonal two-current": np.asarray(two_current, dtype=np.float64),
        }
        rows.append((name, truth, estimates))

    figure, axes = plt.subplots(
        len(rows), 11, figsize=(24.5, 9.5), constrained_layout=True
    )
    headers = (
        "truth", "owner", "direct", "Lanczos", "Eikonal fiber", "Eikonal 2-current",
        "owner residual", "direct residual", "Lanczos residual",
        "fiber residual", "2-current residual",
    )
    for row_index, (name, truth, estimates) in enumerate(rows):
        residuals = {
            label: estimate - truth for label, estimate in estimates.items()
        }
        limit = max(float(np.max(np.abs(value))) for value in residuals.values())
        images = (
            truth,
            estimates["owner"], estimates["direct"], estimates["Lanczos"],
            estimates["Eikonal fiber"], estimates["Eikonal two-current"],
            residuals["owner"], residuals["direct"], residuals["Lanczos"],
            residuals["Eikonal fiber"], residuals["Eikonal two-current"],
        )
        for column, image in enumerate(images):
            axis = axes[row_index, column]
            if column < 6:
                axis.imshow(
                    image, cmap="gray", vmin=0.0, vmax=1.0,
                    interpolation="nearest",
                )
            else:
                axis.imshow(
                    image, cmap="coolwarm", vmin=-limit, vmax=limit,
                    interpolation="nearest",
                )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(headers[column], fontsize=8)
        mse = {
            label: float(np.mean(value * value))
            for label, value in residuals.items()
        }
        axes[row_index, 0].set_ylabel(
            f"{name}\nowner {mse['owner']:.3e}\ndirect {mse['direct']:.3e}\n"
            f"sinc {mse['Lanczos']:.3e}\nfiber {mse['Eikonal fiber']:.3e}\n"
            f"two-current {mse['Eikonal two-current']:.3e}",
            fontsize=7,
        )
    figure.suptitle(
        "Transport on complete signed-Eikonal fibers; residuals share one scale per row",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=210)
    plt.close(figure)


if __name__ == "__main__":
    run(ROOT / "output/support_geometry/eikonal_frame_bank/eikonal_fiber_cure.png")
