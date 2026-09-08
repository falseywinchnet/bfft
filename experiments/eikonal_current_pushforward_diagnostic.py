"""Probe signed-eikonal transport of the admitted CONV current.

This is a geometry experiment, not a candidate production interpolator.  A
straight feature supplies the exact signed eikonal coordinate.  Cartesian
CONV profiles that cross a common eikonal level are pulled back to that level,
averaged, and evaluated on the target lattice.  The construction therefore
tests whether FIR's diagonal coherence can be recovered by spatial phase
transport without signed reconstruction lobes.

Lanczos is retained only as a destination reference.  No optimizer, fitted
coefficient, frequency estimate, or content-selected threshold is used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
for directory in (ROOT, DEMO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend import conv_basin_average, conv_resize, lanczos3_resize  # noqa: E402
from experiments.self_geometric_harmonic_interpolation import (  # noqa: E402
    _evaluate_quintic_variation_lineage_profile,
    _local_fourth_order_2jet,
    _quintic_variation_lineage_profile,
)


Array = np.ndarray


def analytic_feature(
    side: int,
    angle_degrees: float,
    phase: float,
    family: str,
    width: float,
) -> Array:
    y, x = np.indices((side, side), dtype=np.float64)
    center = 0.5 * (side - 1)
    angle = math.radians(angle_degrees)
    normal_x, normal_y = -math.sin(angle), math.cos(angle)
    distance = normal_x * (x - center) + normal_y * (y - center) - phase
    if family == "ridge":
        return 0.12 + 0.76 * np.exp(-0.5 * (distance / width) ** 2)
    if family == "edge":
        return 0.12 + 0.38 * (1.0 + np.tanh(distance / width))
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


def structure_normal(values: Array) -> Array:
    """Return the principal normal from CONV's compact fourth-order 2-jet."""

    field = np.asarray(values, dtype=np.float64)
    if min(field.shape) < 5:
        raise ValueError("the fourth-order structure tensor needs 5x5 samples")
    _, gx, _ = _local_fourth_order_2jet(field.T)
    gx = gx.T
    _, gy, _ = _local_fourth_order_2jet(field)
    # The interior jet is the canonical high-precision differential.  The
    # outer two rings use the separately defined second-order closure and do
    # not contribute to this global straight-owner audit.
    gx = gx[2:-2, 2:-2]
    gy = gy[2:-2, 2:-2]
    tensor = np.array(
        (
            (float(np.sum(gx * gx)), float(np.sum(gx * gy))),
            (float(np.sum(gx * gy)), float(np.sum(gy * gy))),
        ),
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    normal = eigenvectors[:, int(np.argmax(eigenvalues))]
    return normal / max(float(np.linalg.norm(normal)), 1e-30)


def owner_support_normal(values: Array) -> Array:
    """Return the normal induced by an owner's variation-support inertia.

    The compact jet supplies a nonnegative variation measure.  Its spatial
    covariance measures the owner's extent rather than averaging local normal
    estimates.  The dominant covariance eigenvector is the long tangent; its
    orthogonal complement is the signed-Eikonal normal.
    """

    field = np.asarray(values, dtype=np.float64)
    if min(field.shape) < 5:
        raise ValueError("the fourth-order support moment needs 5x5 samples")
    _, gx, _ = _local_fourth_order_2jet(field.T)
    gx = gx.T
    _, gy, _ = _local_fourth_order_2jet(field)
    measure = gx * gx + gy * gy
    total = float(np.sum(measure))
    if total <= np.finfo(np.float64).tiny:
        return np.array((1.0, 0.0), dtype=np.float64)
    y, x = np.indices(field.shape, dtype=np.float64)
    barycenter_x = float(np.sum(measure * x) / total)
    barycenter_y = float(np.sum(measure * y) / total)
    dx = x - barycenter_x
    dy = y - barycenter_y
    covariance = np.array((
        (float(np.sum(measure * dx * dx)), float(np.sum(measure * dx * dy))),
        (float(np.sum(measure * dx * dy)), float(np.sum(measure * dy * dy))),
    )) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    normal = np.array((-tangent[1], tangent[0]))
    return normal / max(float(np.linalg.norm(normal)), 1e-30)


def _profiles_along_axis(values: Array, axis: int) -> list[object]:
    field = np.asarray(values, dtype=np.float64)
    if axis == 1:
        return [
            _quintic_variation_lineage_profile(field[row])
            for row in range(field.shape[0])
        ]
    return [
        _quintic_variation_lineage_profile(field[:, column])
        for column in range(field.shape[1])
    ]


def _evaluate_profile_many(profile: object, positions: Array) -> Array:
    """Vectorized evaluation of one admitted quintic CONV profile."""

    source, increment = profile
    position = np.asarray(positions, dtype=np.float64)
    interval = np.clip(
        np.floor(position).astype(np.int64), 0, source.shape[0] - 2
    )
    u = np.clip(position - interval, 0.0, 1.0)
    control = np.concatenate((
        source[interval, None],
        source[interval, None] + np.cumsum(increment[interval], axis=1),
    ), axis=1)
    for _ in range(5):
        control = (1.0 - u[:, None]) * control[:, :-1] + u[:, None] * control[:, 1:]
    return control[:, 0]


def eikonal_fiber_resize(
    values: Array,
    target_shape: tuple[int, int],
    normal: Array,
) -> Array:
    """Evaluate the mean admitted current on each signed eikonal level.

    For constant metric and a straight transverse seed, the signed Eikonal
    solution is phi(x)=n.x.  Horizontal and vertical source profiles are two
    quadratures of the same level-set fiber.  A row crossing represents
    tangent arclength 1/|n_x| and a column crossing represents arclength
    1/|n_y|.  These geometric weights, followed by mass normalization, are
    symmetric under exchange of the Cartesian axes.  Every target point with
    the same phi receives the same value.
    """

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or min(source.shape) < 5:
        raise ValueError("the diagnostic requires a scalar field at least 5x5")
    out_height, out_width = map(int, target_shape)
    if min(out_height, out_width) < 2:
        raise ValueError("target axes must each contain at least two sites")
    n = np.asarray(normal, dtype=np.float64)
    n /= max(float(np.linalg.norm(n)), 1e-30)
    nx, ny = map(float, n)
    height, width = source.shape
    center_x = 0.5 * (width - 1)
    center_y = 0.5 * (height - 1)

    # Endpoint-aligned target sites expressed in source-lattice coordinates.
    target_y = np.linspace(0.0, height - 1.0, out_height)
    target_x = np.linspace(0.0, width - 1.0, out_width)
    phase = (
        nx * (target_x[None, :] - center_x)
        + ny * (target_y[:, None] - center_y)
    )
    flat_phase = phase.ravel()
    accumulated = np.zeros_like(flat_phase)
    population = np.zeros_like(flat_phase)

    tolerance = 64.0 * np.finfo(np.float64).eps
    if abs(nx) > tolerance:
        profiles = _profiles_along_axis(source, 1)
        transverse = np.arange(height, dtype=np.float64)
        crossing_mass = 1.0 / abs(nx)
        for row in range(height):
            crossings = center_x + (
                flat_phase - ny * (transverse[row] - center_y)
            ) / nx
            admitted = (
                (crossings >= -tolerance)
                & (crossings <= width - 1.0 + tolerance)
            )
            crossings = np.clip(crossings, 0.0, width - 1.0)
            accumulated[admitted] += _evaluate_profile_many(
                profiles[row], crossings[admitted]
            ) * crossing_mass
            population[admitted] += crossing_mass
    if abs(ny) > tolerance:
        profiles = _profiles_along_axis(source, 0)
        transverse = np.arange(width, dtype=np.float64)
        crossing_mass = 1.0 / abs(ny)
        for column in range(width):
            crossings = center_y + (
                flat_phase - nx * (transverse[column] - center_x)
            ) / ny
            admitted = (
                (crossings >= -tolerance)
                & (crossings <= height - 1.0 + tolerance)
            )
            crossings = np.clip(crossings, 0.0, height - 1.0)
            accumulated[admitted] += _evaluate_profile_many(
                profiles[column], crossings[admitted]
            ) * crossing_mass
            population[admitted] += crossing_mass
    if np.any(population == 0.0):
        raise RuntimeError("an Eikonal level did not intersect the source domain")
    return (accumulated / population).reshape(phase.shape).astype(np.float32)


def _range_excess(estimate: Array, source: Array) -> float:
    low, high = float(np.min(source)), float(np.max(source))
    return max(
        0.0,
        low - float(np.min(estimate)),
        float(np.max(estimate)) - high,
    )


def _case(
    side: int,
    factor: int,
    family: str,
    width: float,
    angle: float,
    offset: float,
) -> tuple[list[dict[str, object]], dict[str, Array]]:
    truth = analytic_feature(side, angle, offset, family, width)
    coarse_side = (side - 1) // factor + 1
    coarse = conv_basin_average(truth, (coarse_side, coarse_side))
    radians = math.radians(angle)
    exact_normal = np.array((-math.sin(radians), math.cos(radians)))
    measured_normal = structure_normal(coarse)
    carried_normal = structure_normal(truth)
    owner_normal = owner_support_normal(coarse)
    carried_owner_normal = owner_support_normal(truth)
    agreement = abs(float(measured_normal @ exact_normal))
    angle_error = math.degrees(math.acos(min(max(agreement, -1.0), 1.0)))
    carried_agreement = abs(float(carried_normal @ exact_normal))
    carried_angle_error = math.degrees(math.acos(
        min(max(carried_agreement, -1.0), 1.0)
    ))
    owner_agreement = abs(float(owner_normal @ exact_normal))
    owner_angle_error = math.degrees(math.acos(
        min(max(owner_agreement, -1.0), 1.0)
    ))
    carried_owner_agreement = abs(float(carried_owner_normal @ exact_normal))
    carried_owner_angle_error = math.degrees(math.acos(
        min(max(carried_owner_agreement, -1.0), 1.0)
    ))

    estimates = {
        "CONV Cartesian": conv_resize(coarse, truth.shape),
        "FIR-3 signed": lanczos3_resize(coarse, truth.shape),
        "Eikonal fiber exact": eikonal_fiber_resize(
            coarse, truth.shape, exact_normal
        ),
        "Eikonal fiber measured": eikonal_fiber_resize(
            coarse, truth.shape, measured_normal
        ),
        "Eikonal fiber analysis-carried": eikonal_fiber_resize(
            coarse, truth.shape, carried_normal
        ),
        "Eikonal owner measured": eikonal_fiber_resize(
            coarse, truth.shape, owner_normal
        ),
        "Eikonal owner analysis-carried": eikonal_fiber_resize(
            coarse, truth.shape, carried_owner_normal
        ),
    }
    records = []
    for name, estimate in estimates.items():
        ripple = tangent_ripple(estimate, truth, angle, offset, family)
        records.append({
            "method": name,
            "family": family,
            "angle_degrees": angle,
            "phase": offset,
            "orientation_error_degrees": angle_error,
            "carried_orientation_error_degrees": carried_angle_error,
            "owner_orientation_error_degrees": owner_angle_error,
            "carried_owner_orientation_error_degrees": carried_owner_angle_error,
            "mse": _central_mse(estimate, truth),
            **ripple,
            "range_excess_over_coarse": _range_excess(estimate, coarse),
        })
    return records, {"truth": truth, "coarse": coarse, **estimates}


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    methods = tuple(dict.fromkeys(str(item["method"]) for item in records))
    for method in methods:
        group = [item for item in records if item["method"] == method]
        mse = np.array([float(item["mse"]) for item in group])
        ripple = np.array([float(item["ripple_rms"]) for item in group])
        excess = np.array([
            float(item["range_excess_over_coarse"]) for item in group
        ])
        result[method] = {
            "case_count": len(group),
            "geometric_mean_mse": float(np.exp(np.mean(np.log(mse)))),
            "mean_tangent_ripple_rms": float(np.mean(ripple)),
            "maximum_tangent_ripple_rms": float(np.max(ripple)),
            "maximum_range_excess_over_coarse": float(np.max(excess)),
            "range_escape_cases": int(np.count_nonzero(excess > 1e-8)),
        }
    errors = np.array([
        float(item["orientation_error_degrees"]) for item in records
        if item["method"] == methods[0]
    ])
    carried_errors = np.array([
        float(item["carried_orientation_error_degrees"]) for item in records
        if item["method"] == methods[0]
    ])
    owner_errors = np.array([
        float(item["owner_orientation_error_degrees"]) for item in records
        if item["method"] == methods[0]
    ])
    carried_owner_errors = np.array([
        float(item["carried_owner_orientation_error_degrees"])
        for item in records if item["method"] == methods[0]
    ])
    result["measured_geometry"] = {
        "coarse_mean_orientation_error_degrees": float(np.mean(errors)),
        "coarse_maximum_orientation_error_degrees": float(np.max(errors)),
        "analysis_carried_mean_orientation_error_degrees": float(np.mean(
            carried_errors
        )),
        "analysis_carried_maximum_orientation_error_degrees": float(np.max(
            carried_errors
        )),
        "owner_mean_orientation_error_degrees": float(np.mean(owner_errors)),
        "owner_maximum_orientation_error_degrees": float(np.max(owner_errors)),
        "carried_owner_mean_orientation_error_degrees": float(np.mean(
            carried_owner_errors
        )),
        "carried_owner_maximum_orientation_error_degrees": float(np.max(
            carried_owner_errors
        )),
    }
    return result


def _plot(images: dict[str, Array], output: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = (
        "truth",
        "CONV Cartesian",
        "FIR-3 signed",
        "Eikonal fiber exact",
        "Eikonal fiber measured",
        "Eikonal fiber analysis-carried",
        "Eikonal owner measured",
        "Eikonal owner analysis-carried",
    )
    figure, axes = plt.subplots(1, len(names), figsize=(23, 3.5))
    for axis, name in zip(axes, names):
        axis.imshow(images[name], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(name, fontsize=9)
        axis.set_axis_off()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def run(
    output: Path,
    side: int = 129,
    factor: int = 4,
    *,
    render_plot: bool = True,
) -> dict[str, object]:
    if (side - 1) % factor:
        raise ValueError("side - 1 must be divisible by factor")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    representative: dict[str, Array] | None = None
    for family, width in (("ridge", 1.15), ("edge", 0.85)):
        for angle in (15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0):
            for offset in (-1.0, 0.0, 1.0):
                case_records, images = _case(
                    side, factor, family, width, angle, offset
                )
                records.extend(case_records)
                if family == "ridge" and angle == 37.5 and offset == 0.0:
                    representative = images
    assert representative is not None
    result = {
        "design": {
            "side": side,
            "factor": factor,
            "case_count": len(records) // 7,
            "coordinate": "signed constant-metric Eikonal distance to a transverse line",
            "phase_combination": "level-set arclength mean of admitted Cartesian CONV profiles",
            "matched_geometry": "fourth-order source jet carried through analysis",
            "owner_geometry": "variation-measure barycenter and spatial covariance",
            "frequency_selection": None,
            "optimizer": None,
            "scope": "straight tangent-invariant geometry probe",
        },
        "summary": _summarize(records),
        "records": records,
    }
    if render_plot:
        _plot(
            representative,
            output / "representative_ridge.png",
            f"Signed-eikonal fiber probe, {side}->{(side - 1) // factor + 1}->{side}",
        )
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/support_geometry/eikonal_current_pushforward"),
    )
    parser.add_argument("--side", type=int, default=129)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    result = run(args.out, args.side, args.factor, render_plot=not args.no_plot)
    print(json.dumps({
        "design": result["design"],
        "summary": result["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
