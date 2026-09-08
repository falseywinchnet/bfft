"""Diagnose tangential phase modulation in matched CONV raster cycles.

Lanczos is used only as a destination reference.  No optimization or solver
enters the experiment.  Analytic ridges and edges are constant along a known
tangent, so periodic error along that tangent directly measures the observed
phase-coherence defect.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from skimage import data


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from backend import (  # noqa: E402
    _basin_average_axis,
    _resize_axis,
    conv_basin_average,
    conv_resize,
    lanczos3_resize,
)
from gui_common import unit_image  # noqa: E402
from experiments.self_geometric_harmonic_interpolation import (  # noqa: E402
    _cell_directional_chord,
    _evaluate_quintic_variation_lineage_profile,
    _local_fourth_order_2jet,
    _quintic_variation_lineage_profile,
)


Array = np.ndarray
METHOD_ORDER = (
    "CONV basin -> CONV",
    "CONV basin -> phase chord",
    "CONV basin -> support chord",
    "CONV basin -> Lanczos",
    "Lanczos -> CONV",
    "Lanczos -> phase chord",
    "Lanczos -> support chord",
    "Lanczos -> Lanczos",
    "CONV basin HV -> CONV",
    "CONV basin -> CONV VH",
)


def _jet_support_chord(
    row_profiles: list[object],
    column_profiles: list[object],
    shape: tuple[int, int],
    iy: int,
    ix: int,
    u: float,
    v: float,
    tangent: Array,
) -> float:
    """Two-exit characteristic on the exact six-node two-jet support box."""

    left = max(0, ix - 2)
    right = min(shape[1] - 1, ix + 3)
    top = max(0, iy - 2)
    bottom = min(shape[0] - 1, iy + 3)
    x = ix + u
    y = iy + v
    tx, ty = map(float, tangent)
    tolerance = 128.0 * np.finfo(float).eps

    def exit_distance(direction: float) -> float:
        dx, dy = direction * tx, direction * ty
        candidates: list[float] = []
        if dx > tolerance:
            candidates.append((right - x) / dx)
        elif dx < -tolerance:
            candidates.append((left - x) / dx)
        if dy > tolerance:
            candidates.append((bottom - y) / dy)
        elif dy < -tolerance:
            candidates.append((top - y) / dy)
        return min(distance for distance in candidates if distance >= 0.0)

    backward = exit_distance(-1.0)
    forward = exit_distance(1.0)

    def boundary_value(px: float, py: float) -> float:
        if abs(py - top) <= 4.0 * tolerance:
            return float(_evaluate_quintic_variation_lineage_profile(
                row_profiles[top], px
            ))
        if abs(py - bottom) <= 4.0 * tolerance:
            return float(_evaluate_quintic_variation_lineage_profile(
                row_profiles[bottom], px
            ))
        if abs(px - left) <= 4.0 * tolerance:
            return float(_evaluate_quintic_variation_lineage_profile(
                column_profiles[left], py
            ))
        if abs(px - right) <= 4.0 * tolerance:
            return float(_evaluate_quintic_variation_lineage_profile(
                column_profiles[right], py
            ))
        raise RuntimeError("support characteristic did not meet its boundary")

    backward_value = boundary_value(x - backward * tx, y - backward * ty)
    forward_value = boundary_value(x + forward * tx, y + forward * ty)
    return (
        forward * backward_value + backward * forward_value
    ) / (backward + forward)


def _connection_boundary_value(
    row_profiles: list[object],
    column_profiles: list[object],
    bounds: tuple[int, int, int, int],
    point: Array,
) -> float:
    left, right, top, bottom = bounds
    px, py = map(float, point)
    tolerance = 2.0e-10
    if abs(py - top) <= tolerance:
        return float(_evaluate_quintic_variation_lineage_profile(
            row_profiles[top], px
        ))
    if abs(py - bottom) <= tolerance:
        return float(_evaluate_quintic_variation_lineage_profile(
            row_profiles[bottom], px
        ))
    if abs(px - left) <= tolerance:
        return float(_evaluate_quintic_variation_lineage_profile(
            column_profiles[left], py
        ))
    if abs(px - right) <= tolerance:
        return float(_evaluate_quintic_variation_lineage_profile(
            column_profiles[right], py
        ))
    raise RuntimeError("phase connection did not terminate on its support")


def _trace_phase_connection(
    point: Array,
    initial_direction: Array,
    bounds: tuple[int, int, int, int],
    cell_mass: Array,
    cell_tangent: Array,
) -> tuple[Array, float, float]:
    """Traverse a piecewise-constant tangent connection without step size."""

    left, right, top, bottom = bounds
    position = np.asarray(point, dtype=np.float64).copy()
    direction = np.asarray(initial_direction, dtype=np.float64).copy()
    length = 0.0
    admitted_mass = 1.0
    epsilon = 1.0e-10

    def select_cell() -> tuple[int, int, Array, float]:
        x_integer = abs(position[0] - round(position[0])) <= epsilon
        y_integer = abs(position[1] - round(position[1])) <= epsilon
        base_x = int(math.floor(position[0]))
        base_y = int(math.floor(position[1]))
        xs = (base_x - 1, base_x) if x_integer else (base_x,)
        ys = (base_y - 1, base_y) if y_integer else (base_y,)
        candidates: list[tuple[float, int, int, Array, float]] = []
        for cy in ys:
            for cx in xs:
                if not (left <= cx < right and top <= cy < bottom):
                    continue
                local = cell_tangent[cy, cx].copy()
                if float(local @ direction) < 0.0:
                    local = -local
                probe = position + epsilon * local
                if not (
                    cx - 0.25 * epsilon
                    <= probe[0]
                    <= cx + 1.0 + 0.25 * epsilon
                    and cy - 0.25 * epsilon
                    <= probe[1]
                    <= cy + 1.0 + 0.25 * epsilon
                ):
                    continue
                score = float(local @ direction)
                candidates.append((
                    score, -cy, -cx, local, float(cell_mass[cy, cx])
                ))
        if not candidates:
            raise RuntimeError(
                "phase connection has no forward incident cell: "
                f"position={position.tolist()}, direction={direction.tolist()}, "
                f"bounds={bounds}, xs={xs}, ys={ys}"
            )
        _score, negative_y, negative_x, local, mass = max(
            candidates, key=lambda candidate: candidate[:3]
        )
        return -negative_y, -negative_x, local, mass

    for _ in range(32):
        if (
            (position[0] <= left + epsilon and direction[0] < 0.0)
            or (position[0] >= right - epsilon and direction[0] > 0.0)
            or (position[1] <= top + epsilon and direction[1] < 0.0)
            or (position[1] >= bottom - epsilon and direction[1] > 0.0)
        ):
            return position, length, admitted_mass
        iy, ix, local, local_mass = select_cell()
        admitted_mass = min(admitted_mass, local_mass)
        direction = local
        tx, ty = map(float, direction)
        candidates: list[float] = []
        if tx > epsilon:
            candidates.append((ix + 1.0 - position[0]) / tx)
        elif tx < -epsilon:
            candidates.append((ix - position[0]) / tx)
        if ty > epsilon:
            candidates.append((iy + 1.0 - position[1]) / ty)
        elif ty < -epsilon:
            candidates.append((iy - position[1]) / ty)
        positive = [distance for distance in candidates if distance > epsilon]
        if not positive:
            raise RuntimeError("phase connection did not leave its current cell")
        distance = min(positive)
        position += distance * direction
        length += distance
        if (
            position[0] <= left + epsilon
            or position[0] >= right - epsilon
            or position[1] <= top + epsilon
            or position[1] >= bottom - epsilon
        ):
            position[0] = min(max(position[0], left), right)
            position[1] = min(max(position[1], top), bottom)
            return position, length, admitted_mass
    raise RuntimeError("phase connection exceeded compact two-jet support")


def _phase_connection_value(
    row_profiles: list[object],
    column_profiles: list[object],
    source_shape: tuple[int, int],
    cell_mass: Array,
    cell_tangent: Array,
    iy: int,
    ix: int,
    u: float,
    v: float,
) -> tuple[float, float]:
    left = max(0, ix - 2)
    right = min(source_shape[1] - 1, ix + 3)
    top = max(0, iy - 2)
    bottom = min(source_shape[0] - 1, iy + 3)
    bounds = (left, right, top, bottom)
    point = np.array((ix + u, iy + v), dtype=np.float64)
    tangent = cell_tangent[iy, ix]
    backward, backward_length, backward_mass = _trace_phase_connection(
        point, -tangent, bounds, cell_mass, cell_tangent
    )
    forward, forward_length, forward_mass = _trace_phase_connection(
        point, tangent, bounds, cell_mass, cell_tangent
    )
    backward_value = _connection_boundary_value(
        row_profiles, column_profiles, bounds, backward
    )
    forward_value = _connection_boundary_value(
        row_profiles, column_profiles, bounds, forward
    )
    value = (
        forward_length * backward_value
        + backward_length * forward_value
    ) / (backward_length + forward_length)
    return value, min(backward_mass, forward_mass)


def _formal_first_jet(field: Array) -> tuple[Array, Array]:
    """Use the same fourth-interior/second-boundary jet as CONV."""

    value = np.asarray(field, dtype=np.float64)
    _, derivative_y, _ = _local_fourth_order_2jet(value)
    _, derivative_x, _ = _local_fourth_order_2jet(np.moveaxis(value, 1, 0))
    return np.moveaxis(derivative_x, 0, 1), derivative_y


def phase_chord_resize(
    value: Array,
    target: tuple[int, int],
    *,
    support: str = "cell",
) -> Array:
    """Experimental positive tangent transport induced by the CONV two-jet.

    The four corner gradients of each source cell define the positive second
    moment Q=sum(g g^T).  Its normalized eigenvalue gap is the exact mass of
    the single-direction part; the remaining mass retains ordinary CONV.
    The directional value is the positive two-exit measure on the tangent
    chord through the target point.
    """

    source = np.asarray(value, dtype=np.float64)
    if source.ndim != 2 or min(source.shape) < 5:
        raise ValueError("phase-chord diagnostic currently accepts scalar 2-D data")
    baseline = np.asarray(conv_resize(source, target), dtype=np.float64)
    derivative_x, derivative_y = _formal_first_jet(source)
    row_profiles = [
        _quintic_variation_lineage_profile(source[row])
        for row in range(source.shape[0])
    ]
    column_profiles = [
        _quintic_variation_lineage_profile(source[:, column])
        for column in range(source.shape[1])
    ]
    cell_mass = np.zeros((source.shape[0] - 1, source.shape[1] - 1))
    cell_tangent = np.zeros(cell_mass.shape + (2,))
    for iy in range(source.shape[0] - 1):
        for ix in range(source.shape[1] - 1):
            gx = derivative_x[iy : iy + 2, ix : ix + 2].reshape(-1)
            gy = derivative_y[iy : iy + 2, ix : ix + 2].reshape(-1)
            form = np.array((
                (float(gx @ gx), float(gx @ gy)),
                (float(gx @ gy), float(gy @ gy)),
            ))
            eigenvalues, eigenvectors = np.linalg.eigh(form)
            total = float(eigenvalues[0] + eigenvalues[1])
            if total > 0.0:
                cell_mass[iy, ix] = (
                    float(eigenvalues[1] - eigenvalues[0]) / total
                )
                cell_tangent[iy, ix] = eigenvectors[:, 0]
    output = baseline.copy()
    source_y = np.linspace(0.0, source.shape[0] - 1.0, target[0])
    source_x = np.linspace(0.0, source.shape[1] - 1.0, target[1])
    for oy, global_y in enumerate(source_y):
        iy = min(int(math.floor(global_y)), source.shape[0] - 2)
        v = global_y - iy
        for ox, global_x in enumerate(source_x):
            ix = min(int(math.floor(global_x)), source.shape[1] - 2)
            u = global_x - ix
            if (
                (abs(u) < 1.0e-14 or abs(u - 1.0) < 1.0e-14)
                and (abs(v) < 1.0e-14 or abs(v - 1.0) < 1.0e-14)
            ):
                continue
            directional_mass = float(cell_mass[iy, ix])
            if directional_mass <= 0.0:
                continue
            tangent = cell_tangent[iy, ix]
            if support == "cell":
                chord = _cell_directional_chord(
                    row_profiles,
                    column_profiles,
                    iy,
                    ix,
                    u,
                    v,
                    tangent,
                    _evaluate_quintic_variation_lineage_profile,
                )
            elif support == "jet":
                chord = _jet_support_chord(
                    row_profiles,
                    column_profiles,
                    source.shape,
                    iy,
                    ix,
                    u,
                    v,
                    tangent,
                )
            elif support == "connection":
                chord, directional_mass = _phase_connection_value(
                    row_profiles,
                    column_profiles,
                    source.shape,
                    cell_mass,
                    cell_tangent,
                    iy,
                    ix,
                    u,
                    v,
                )
            else:
                raise ValueError("support must be 'cell', 'jet', or 'connection'")
            output[oy, ox] = (
                (1.0 - directional_mass) * baseline[oy, ox]
                + directional_mass * chord
            )
    return output.astype(np.float32)


def _basin_hv(value: Array, target: tuple[int, int]) -> Array:
    along_x = _basin_average_axis(value, target[1], 1)
    return _basin_average_axis(along_x, target[0], 0)


def _conv_vh(value: Array, target: tuple[int, int]) -> Array:
    along_y = _resize_axis(value, target[0], 0, "conv")
    return _resize_axis(along_y, target[1], 1, "conv")


def crossed_cycles(
    value: Array,
    target: tuple[int, int],
    *,
    include_phase_chord: bool = False,
) -> dict[str, Array]:
    """Return crossed analysis/synthesis cycles without choosing among them."""

    source = np.asarray(value, dtype=np.float32)
    conv_coarse = conv_basin_average(source, target)
    lanczos_coarse = lanczos3_resize(source, target)
    conv_hv_coarse = _basin_hv(source, target)
    result = {
        "CONV basin -> CONV": conv_resize(conv_coarse, source.shape[:2]),
        "CONV basin -> Lanczos": lanczos3_resize(
            conv_coarse, source.shape[:2]
        ),
        "Lanczos -> CONV": conv_resize(lanczos_coarse, source.shape[:2]),
        "Lanczos -> Lanczos": lanczos3_resize(
            lanczos_coarse, source.shape[:2]
        ),
        "CONV basin HV -> CONV": conv_resize(
            conv_hv_coarse, source.shape[:2]
        ),
        "CONV basin -> CONV VH": _conv_vh(
            conv_coarse, source.shape[:2]
        ),
    }
    if include_phase_chord:
        result["CONV basin -> phase chord"] = phase_chord_resize(
            conv_coarse, source.shape[:2]
        )
        result["CONV basin -> support chord"] = phase_chord_resize(
            conv_coarse, source.shape[:2], support="jet"
        )
        result["Lanczos -> phase chord"] = phase_chord_resize(
            lanczos_coarse, source.shape[:2]
        )
        result["Lanczos -> support chord"] = phase_chord_resize(
            lanczos_coarse, source.shape[:2], support="jet"
        )
    return result


def analytic_feature(
    side: int,
    angle_degrees: float,
    phase: float,
    family: str,
    width: float,
) -> Array:
    """Sample a ridge or edge whose continuous value is tangent-invariant."""

    y, x = np.indices((side, side), dtype=np.float64)
    center = 0.5 * (side - 1)
    angle = math.radians(angle_degrees)
    normal_x, normal_y = -math.sin(angle), math.cos(angle)
    distance = normal_x * (x - center) + normal_y * (y - center) - phase
    if family == "ridge":
        return 0.12 + 0.76 * np.exp(-0.5 * (distance / width) ** 2)
    if family == "edge":
        return 0.12 + 0.76 * 0.5 * (1.0 + np.tanh(distance / width))
    raise ValueError(f"unknown family {family}")


def _sample_tangent(
    value: Array,
    angle_degrees: float,
    phase: float,
    normal_offset: float,
) -> tuple[Array, Array]:
    side = value.shape[0]
    center = 0.5 * (side - 1)
    angle = math.radians(angle_degrees)
    tangent = np.array((math.cos(angle), math.sin(angle)))
    normal = np.array((-math.sin(angle), math.cos(angle)))
    extent = 0.34 * side
    coordinate = np.linspace(-extent, extent, 2 * math.ceil(extent) + 1)
    points = (
        np.array((center, center))[None]
        + (phase + normal_offset) * normal[None]
        + coordinate[:, None] * tangent[None]
    )
    sampled = map_coordinates(
        np.asarray(value, dtype=np.float64),
        (points[:, 1], points[:, 0]),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return coordinate, sampled


def tangent_ripple(
    estimate: Array,
    truth: Array,
    angle_degrees: float,
    phase: float,
    family: str,
) -> dict[str, float]:
    """Measure periodic residual after removing only a cubic tangent trend."""

    offsets = (
        (-1.5, -0.75, 0.0, 0.75, 1.5)
        if family == "ridge"
        else (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    )
    rms: list[float] = []
    peaks: list[float] = []
    frequencies: list[float] = []
    for normal_offset in offsets:
        coordinate, observed = _sample_tangent(
            estimate, angle_degrees, phase, normal_offset
        )
        _, reference = _sample_tangent(
            truth, angle_degrees, phase, normal_offset
        )
        residual = observed - reference
        polynomial = np.polynomial.Polynomial.fit(
            coordinate, residual, 3
        ).convert()
        detrended = residual - polynomial(coordinate)
        rms.append(float(np.sqrt(np.mean(detrended * detrended))))
        window = np.hanning(detrended.size)
        spectrum = np.abs(np.fft.rfft(detrended * window))
        spectrum *= 2.0 / max(float(np.sum(window)), 1.0)
        spectrum[:4] = 0.0
        frequency = np.fft.rfftfreq(
            detrended.size, d=float(coordinate[1] - coordinate[0])
        )
        index = int(np.argmax(spectrum))
        peaks.append(float(spectrum[index]))
        frequencies.append(float(frequency[index]))
    maximum = int(np.argmax(peaks))
    return {
        "ripple_rms": float(np.mean(rms)),
        "dominant_amplitude": peaks[maximum],
        "dominant_cycles_per_pixel": frequencies[maximum],
    }


def _central_mse(estimate: Array, truth: Array, crop: int = 12) -> float:
    residual = np.asarray(estimate)[crop:-crop, crop:-crop] - np.asarray(truth)[
        crop:-crop, crop:-crop
    ]
    return float(np.mean(residual * residual))


def run_synthetic(side: int, factor: int) -> list[dict[str, object]]:
    target_side = (side - 1) // factor + 1
    target = (target_side, target_side)
    records: list[dict[str, object]] = []
    angles = (15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0)
    phases = tuple(np.linspace(-1.5, 1.5, 7))
    families = (("ridge", 1.15), ("edge", 0.85))
    for family, width in families:
        for angle in angles:
            for phase in phases:
                truth = analytic_feature(side, angle, phase, family, width)
                cycles = crossed_cycles(
                    truth, target, include_phase_chord=True
                )
                for method, estimate in cycles.items():
                    ripple = tangent_ripple(
                        estimate, truth, angle, phase, family
                    )
                    records.append({
                        "family": family,
                        "width": width,
                        "angle_degrees": angle,
                        "phase": phase,
                        "method": method,
                        "mse": _central_mse(estimate, truth),
                        **ripple,
                    })
    return records


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for method in METHOD_ORDER:
        group = [record for record in records if record["method"] == method]
        rms = np.array([float(record["ripple_rms"]) for record in group])
        peak = np.array([
            float(record["dominant_amplitude"]) for record in group
        ])
        mse = np.array([float(record["mse"]) for record in group])
        worst = group[int(np.argmax(rms))]
        summary[method] = {
            "case_count": len(group),
            "geometric_mean_mse": float(np.exp(np.mean(np.log(mse)))),
            "mean_tangent_ripple_rms": float(np.mean(rms)),
            "maximum_tangent_ripple_rms": float(np.max(rms)),
            "mean_dominant_amplitude": float(np.mean(peak)),
            "worst_case": {
                key: worst[key]
                for key in (
                    "family", "angle_degrees", "phase", "ripple_rms",
                    "dominant_amplitude", "dominant_cycles_per_pixel",
                    "mse",
                )
            },
        }
    return summary


def plot_worst_case(
    records: list[dict[str, object]],
    side: int,
    factor: int,
    output: Path,
) -> None:
    canonical = [
        record for record in records
        if record["method"] == "CONV basin -> CONV"
    ]
    worst = max(canonical, key=lambda record: float(record["ripple_rms"]))
    family = str(worst["family"])
    angle = float(worst["angle_degrees"])
    phase = float(worst["phase"])
    width = float(worst["width"])
    truth = analytic_feature(side, angle, phase, family, width)
    target_side = (side - 1) // factor + 1
    cycles = crossed_cycles(
        truth, (target_side, target_side), include_phase_chord=True
    )

    shown = (
        "CONV basin -> CONV",
        "CONV basin -> phase chord",
        "CONV basin -> support chord",
        "CONV basin -> Lanczos",
        "Lanczos -> CONV",
        "Lanczos -> phase chord",
        "Lanczos -> support chord",
        "Lanczos -> Lanczos",
    )
    figure, axes = plt.subplots(2, len(shown) + 1, figsize=(14, 6.2))
    images = ("truth",) + shown
    for column, name in enumerate(images):
        image = truth if name == "truth" else cycles[name]
        axes[0, column].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(name, fontsize=9)
        axes[0, column].set_axis_off()
        coordinate, observed = _sample_tangent(image, angle, phase, 0.0)
        _, reference = _sample_tangent(truth, angle, phase, 0.0)
        axes[1, column].plot(coordinate, observed - reference, linewidth=0.9)
        axes[1, column].axhline(0.0, color="black", linewidth=0.5)
        axes[1, column].set_ylim(-0.35, 0.35)
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_xlabel("tangent coordinate")
        if column == 0:
            axes[1, column].set_ylabel("cycle residual")
    figure.suptitle(
        f"Worst canonical tangent ripple: {family}, {angle:g} deg, "
        f"phase {phase:g}, {side}->{target_side}->{side}"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_natural(output: Path, factor: int) -> dict[str, object]:
    sources = {
        "text": unit_image(data.text()),
        "camera": unit_image(data.camera()),
    }
    shown = (
        "CONV basin -> CONV",
        "CONV basin -> Lanczos",
        "Lanczos -> CONV",
        "Lanczos -> Lanczos",
    )
    metrics: dict[str, object] = {}
    figure, axes = plt.subplots(
        len(sources), len(shown) + 1, figsize=(15, 7.2), squeeze=False
    )
    for row, (source_name, source) in enumerate(sources.items()):
        target = tuple(max(5, round(extent / factor)) for extent in source.shape)
        cycles = crossed_cycles(source, target)
        metrics[source_name] = {
            "source_shape": list(source.shape),
            "target_shape": list(target),
            "mse": {
                method: float(np.mean((cycles[method] - source) ** 2))
                for method in shown
            },
        }
        for column, name in enumerate(("truth",) + shown):
            image = source if name == "truth" else cycles[name]
            axes[row, column].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].set_title(
                source_name if name == "truth" else name, fontsize=9
            )
            axes[row, column].set_axis_off()
    figure.suptitle(f"Natural-image crossed cycles, factor {factor}")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return metrics


def run(output: Path, side: int, factor: int) -> dict[str, object]:
    if (side - 1) % factor:
        raise ValueError("side - 1 must be divisible by factor")
    output.mkdir(parents=True, exist_ok=True)
    records = run_synthetic(side, factor)
    result = {
        "design": {
            "side": side,
            "factor": factor,
            "angles_degrees": [15, 22.5, 30, 37.5, 45, 52.5, 60, 67.5, 75],
            "phases": list(np.linspace(-1.5, 1.5, 7)),
            "families": {"ridge_sigma": 1.15, "edge_tanh_width": 0.85},
            "method_role": "Lanczos is a destination reference only",
            "solver": None,
        },
        "summary": summarize(records),
        "natural": plot_natural(output / "natural_cross_cycles.png", factor),
        "records": records,
    }
    plot_worst_case(records, side, factor, output / "worst_tangent_ripple.png")
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/support_geometry/phase_transport"),
    )
    parser.add_argument("--side", type=int, default=129)
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()
    result = run(args.out, args.side, args.factor)
    print(json.dumps({
        "design": result["design"],
        "summary": result["summary"],
        "natural": result["natural"],
    }, indent=2))


if __name__ == "__main__":
    main()
