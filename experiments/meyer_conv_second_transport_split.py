"""Second-current Meyer texture transport suggested by the CONV two-jet."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage import data
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conv_distilled_core import (  # noqa: E402
    _variation_jet,
    distilled_conv_resize,
)
from experiments.convstar import _local_two_jet  # noqa: E402
from experiments.meyer_conv_transport_split import ranged, signed  # noqa: E402
from experiments.meyer_eikonal_natural_probe import normalized  # noqa: E402
from experiments.meyer_eikonal_natural_probe import (  # noqa: E402
    correlation,
    gradient_magnitude,
)
from experiments.meyer_eikonal_transport_split import (  # noqa: E402
    metric_meyer,
    metric_rof,
    relative_mse,
    scenes,
)


OUT = ROOT / "experiments" / "out" / "meyer_conv_second_transport_split"
SQRT2 = np.sqrt(2.0)


def _dx(field: np.ndarray) -> np.ndarray:
    return np.roll(field, -1, axis=1) - field


def _dy(field: np.ndarray) -> np.ndarray:
    return np.roll(field, -1, axis=0) - field


def _divx(field: np.ndarray) -> np.ndarray:
    return field - np.roll(field, 1, axis=1)


def _divy(field: np.ndarray) -> np.ndarray:
    return field - np.roll(field, 1, axis=0)


def symmetric_hessian(
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Frobenius-isometric periodic Hessian coordinates."""

    value = np.asarray(field, dtype=np.float64)
    return _dx(_dx(value)), SQRT2 * _dy(_dx(value)), _dy(_dy(value))


def double_divergence(
    qxx: np.ndarray,
    qxy_scaled: np.ndarray,
    qyy: np.ndarray,
) -> np.ndarray:
    """Adjoint of ``symmetric_hessian`` in scaled symmetric coordinates."""

    return (
        _divx(_divx(qxx))
        + SQRT2 * _divx(_divy(qxy_scaled))
        + _divy(_divy(qyy))
    )


def _conv_twojet(field: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return CONV's symmetric two-jet in Frobenius coordinates."""

    source = np.asarray(field, dtype=np.float64)
    height, width = source.shape
    component = source[..., None]
    x_lines = np.moveaxis(component, 1, 0).reshape(width, -1)
    y_lines = component.reshape(height, -1)
    hxx = _local_two_jet(x_lines)[1].reshape(
        width, height).T
    hyy = _local_two_jet(y_lines)[1].reshape(height, width)
    gx, gy = _variation_jet(component)
    dy_gx = _local_two_jet(gx.reshape(height, -1))[0].reshape(height, width)
    dx_gy = _local_two_jet(
        np.moveaxis(gy, 1, 0).reshape(width, -1))[0].reshape(width, height).T
    hxy = SQRT2 * 0.5 * (dy_gx + dx_gy)
    return hxx, hxy, hyy


def _conv_current_alternation(field: np.ndarray) -> np.ndarray:
    """Bounded sign-reversal mass over CONV's declared six-node support."""

    component = np.asarray(field, dtype=np.float64)[..., None]
    gx, gy = _variation_jet(component)
    gx = gx[..., 0]
    gy = gy[..., 0]

    def axis_ratio(current: np.ndarray, axis: int) -> np.ndarray:
        numerator = np.zeros_like(current)
        denominator = np.zeros_like(current)
        # Five consecutive current pairs occupy the six-node jet support.
        for offset in range(-2, 3):
            left = np.roll(current, -offset, axis=axis)
            right = np.roll(current, -(offset + 1), axis=axis)
            numerator += 2.0 * np.sqrt(np.maximum(-left * right, 0.0))
            denominator += np.abs(left) + np.abs(right)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )

    horizontal = axis_ratio(gx, 1)
    vertical = axis_ratio(gy, 0)
    return np.clip(np.maximum(horizontal, vertical), 0.0, 1.0)


def conv_persistent_twojet_state(image: np.ndarray) -> dict[str, np.ndarray | float]:
    """Raise scale-persistent CONV curvature to a volume-neutral tensor state."""

    source = np.asarray(image, dtype=np.float64)
    coarse_shape = tuple(max(5, int(np.ceil(side / 2))) for side in source.shape)
    coarse = distilled_conv_resize(source, coarse_shape)
    transported = distilled_conv_resize(coarse, source.shape)
    fine = _conv_twojet(source)
    coarse_jet = _conv_twojet(transported)
    fine_norm = np.sqrt(sum(component * component for component in fine))
    coarse_norm = np.sqrt(sum(component * component for component in coarse_jet))
    dot = sum(left * right for left, right in zip(fine, coarse_jet))
    numerical = np.finfo(float).eps * max(float(np.ptp(source)) ** 2, 1.0)
    angular = np.clip(np.divide(
        dot,
        fine_norm * coarse_norm + numerical,
        out=np.zeros_like(dot),
        where=(fine_norm > 0.0) & (coarse_norm > 0.0),
    ), 0.0, 1.0)
    amplitude = np.divide(
        2.0 * np.minimum(fine_norm, coarse_norm),
        fine_norm + coarse_norm + numerical,
        out=np.zeros_like(fine_norm),
        where=(fine_norm + coarse_norm) > 0.0,
    )
    persistence = np.clip(angular * amplitude, 0.0, 1.0)
    alternation = _conv_current_alternation(source)
    strength = persistence * (1.0 - alternation)
    # The common two-jet direction is sign-sensitive: opposite curvature is
    # not a persistence certificate and was already removed by ``angular``.
    common = tuple(left + right for left, right in zip(fine, coarse_jet))
    common_norm = np.sqrt(sum(component * component for component in common))
    direction = tuple(np.divide(
        component,
        common_norm + numerical,
        out=np.zeros_like(component),
        where=common_norm > 0.0,
    ) for component in common)
    # Add one unit rank-one certificate to I, then remove its scalar volume.
    # The resulting metric has determinant one and condition 1+s <= 2.
    inverse_parallel = np.power(1.0 + strength, -1.0 / 3.0)
    inverse_orthogonal = np.power(1.0 + strength, 1.0 / 6.0)
    return {
        "direction_xx": direction[0],
        "direction_xy": direction[1],
        "direction_yy": direction[2],
        "inverse_parallel": inverse_parallel,
        "inverse_orthogonal": inverse_orthogonal,
        "strength": strength,
        "persistence": persistence,
        "alternation": alternation,
        "mean_strength": float(np.mean(strength)),
        "mean_persistence": float(np.mean(persistence)),
        "mean_alternation": float(np.mean(alternation)),
        "maximum_strength": float(np.max(strength)),
        "condition_maximum": float(np.max(1.0 + strength)),
    }


def _apply_tensor_inverse_sqrt(
    xx: np.ndarray,
    xy: np.ndarray,
    yy: np.ndarray,
    state: dict[str, np.ndarray | float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if state is None:
        return xx, xy, yy
    nxx = np.asarray(state["direction_xx"])
    nxy = np.asarray(state["direction_xy"])
    nyy = np.asarray(state["direction_yy"])
    parallel = np.asarray(state["inverse_parallel"])
    orthogonal = np.asarray(state["inverse_orthogonal"])
    coordinate = nxx * xx + nxy * xy + nyy * yy
    difference = parallel - orthogonal
    return (
        orthogonal * xx + difference * nxx * coordinate,
        orthogonal * xy + difference * nxy * coordinate,
        orthogonal * yy + difference * nyy * coordinate,
    )


@lru_cache(maxsize=16)
def _conv_jet_matrices(length: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix, float, float]:
    """Return CONV's exact first/second jet matrices and their spectral norms."""

    identity = np.eye(int(length), dtype=np.float64)
    first, second = _local_two_jet(identity)
    first_norm = float(np.linalg.norm(first, 2))
    second_norm = float(np.linalg.norm(second, 2))
    return sparse.csr_matrix(first), sparse.csr_matrix(second), first_norm, second_norm


def conv_symmetric_hessian(field: np.ndarray) -> tuple[np.ndarray, ...]:
    """Apply the exact separable CONV two-jet in Frobenius coordinates."""

    value = np.asarray(field, dtype=np.float64)
    dy, dyy, _ny1, _ny2 = _conv_jet_matrices(value.shape[0])
    dx, dxx, _nx1, _nx2 = _conv_jet_matrices(value.shape[1])
    hxx = np.asarray((dxx @ value.T).T)
    hyy = np.asarray(dyy @ value)
    first_y = np.asarray(dy @ value)
    hxy = SQRT2 * np.asarray((dx @ first_y.T).T)
    return hxx, hxy, hyy


def conv_hessian_adjoint(
    qxx: np.ndarray,
    qxy: np.ndarray,
    qyy: np.ndarray,
) -> np.ndarray:
    """Exact Euclidean adjoint of ``conv_symmetric_hessian``."""

    height, width = qxx.shape
    dy, dyy, _ny1, _ny2 = _conv_jet_matrices(height)
    dx, dxx, _nx1, _nx2 = _conv_jet_matrices(width)
    xx = np.asarray((dxx.T @ qxx.T).T)
    yy = np.asarray(dyy.T @ qyy)
    right_x = np.asarray((dx.T @ qxy.T).T)
    xy = SQRT2 * np.asarray(dy.T @ right_x)
    return xx + xy + yy


def conv_second_order_rof(
    source: np.ndarray,
    fidelity: float,
    state: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    iterations: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], dict[str, float]]:
    """Dual projection for variation measured by CONV's exact two-jet."""

    g = np.asarray(source, dtype=np.float64)
    if state is None:
        qxx = np.zeros_like(g)
        qxy = np.zeros_like(g)
        qyy = np.zeros_like(g)
    else:
        qxx, qxy, qyy = (component.copy() for component in state)
    _dy, _dyy, norm_y1, norm_y2 = _conv_jet_matrices(g.shape[0])
    _dx, _dxx, norm_x1, norm_x2 = _conv_jet_matrices(g.shape[1])
    lipschitz = (
        norm_x2 * norm_x2 + norm_y2 * norm_y2
        + 2.0 * norm_x1 * norm_x1 * norm_y1 * norm_y1)
    step = float(fidelity) / lipschitz
    previous_objective = np.inf
    maximum_increase = 0.0
    for _ in range(int(iterations)):
        u = g - conv_hessian_adjoint(qxx, qxy, qyy) / float(fidelity)
        hxx, hxy, hyy = conv_symmetric_hessian(u)
        candidate_xx = qxx + step * hxx
        candidate_xy = qxy + step * hxy
        candidate_yy = qyy + step * hyy
        norm = np.sqrt(
            candidate_xx * candidate_xx
            + candidate_xy * candidate_xy
            + candidate_yy * candidate_yy)
        scale = 1.0 / np.maximum(norm, 1.0)
        qxx = candidate_xx * scale
        qxy = candidate_xy * scale
        qyy = candidate_yy * scale
        u = g - conv_hessian_adjoint(qxx, qxy, qyy) / float(fidelity)
        objective = 0.5 * float(np.vdot(u, u).real)
        if np.isfinite(previous_objective):
            maximum_increase = max(maximum_increase, objective - previous_objective)
        previous_objective = objective
    return u, (qxx, qxy, qyy), {
        "dual_objective": previous_objective,
        "maximum_objective_increase": maximum_increase,
        "maximum_tensor_flux_norm": float(np.max(np.sqrt(
            qxx * qxx + qxy * qxy + qyy * qyy))),
        "operator_lipschitz_bound": lipschitz,
    }


def second_order_rof(
    source: np.ndarray,
    fidelity: float,
    state: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    iterations: int,
    tensor_geometry: dict[str, np.ndarray | float] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], dict[str, float]]:
    """Project the dual of Hessian-Frobenius variation by a unit tensor ball."""

    g = np.asarray(source, dtype=np.float64)
    if state is None:
        qxx = np.zeros_like(g)
        qxy = np.zeros_like(g)
        qyy = np.zeros_like(g)
    else:
        qxx, qxy, qyy = (component.copy() for component in state)
    # ||H||^2 <= 64.  The tensor inverse-square-root has largest
    # eigenvalue (1+s)^(1/6), whose square is at most 2^(1/3).
    geometry_scale = (
        1.0 if tensor_geometry is None
        else float(np.max(np.asarray(tensor_geometry["inverse_orthogonal"]) ** 2))
    )
    step = float(fidelity) / (64.0 * max(geometry_scale, 1.0))
    previous_objective = np.inf
    maximum_increase = 0.0
    for _ in range(int(iterations)):
        pxx, pxy, pyy = _apply_tensor_inverse_sqrt(
            qxx, qxy, qyy, tensor_geometry)
        u = g - double_divergence(pxx, pxy, pyy) / float(fidelity)
        hxx, hxy, hyy = symmetric_hessian(u)
        hxx, hxy, hyy = _apply_tensor_inverse_sqrt(
            hxx, hxy, hyy, tensor_geometry)
        candidate_xx = qxx + step * hxx
        candidate_xy = qxy + step * hxy
        candidate_yy = qyy + step * hyy
        norm = np.sqrt(
            candidate_xx * candidate_xx
            + candidate_xy * candidate_xy
            + candidate_yy * candidate_yy)
        scale = 1.0 / np.maximum(norm, 1.0)
        qxx = candidate_xx * scale
        qxy = candidate_xy * scale
        qyy = candidate_yy * scale
        pxx, pxy, pyy = _apply_tensor_inverse_sqrt(
            qxx, qxy, qyy, tensor_geometry)
        u = g - double_divergence(pxx, pxy, pyy) / float(fidelity)
        objective = 0.5 * float(np.vdot(u, u).real)
        if np.isfinite(previous_objective):
            maximum_increase = max(maximum_increase, objective - previous_objective)
        previous_objective = objective
    return u, (qxx, qxy, qyy), {
        "dual_objective": previous_objective,
        "maximum_objective_increase": maximum_increase,
        "maximum_tensor_flux_norm": float(np.max(np.sqrt(
            qxx * qxx + qxy * qxy + qyy * qyy))),
        "tensor_geometry_scale": geometry_scale,
    }


def second_current_meyer(
    image: np.ndarray,
    *,
    lam: float = 0.05,
    mu2: float = 40.0,
    outer_iterations: int = 96,
    inner_iterations: int = 8,
    tensor_geometry: dict[str, np.ndarray | float] | None = None,
    conv_operator: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Alternate TV cartoon transport with a bounded second-current texture."""

    source = np.asarray(image, dtype=np.float64)
    identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
    cartoon = np.zeros_like(source)
    texture = np.zeros_like(source)
    cartoon_state = None
    texture_state = None
    maximum_increase = 0.0
    maximum_flux = 0.0
    last_change = np.inf
    for _ in range(int(outer_iterations)):
        next_cartoon, cartoon_state, cartoon_diag = metric_rof(
            source - texture, lam, identity, cartoon_state, inner_iterations)
        remainder = source - next_cartoon
        if conv_operator:
            if tensor_geometry is not None:
                raise ValueError("CONV two-jet operator does not use inferred geometry")
            smooth, texture_state, texture_diag = conv_second_order_rof(
                remainder, 1.0 / float(mu2), texture_state, inner_iterations)
        else:
            smooth, texture_state, texture_diag = second_order_rof(
                remainder, 1.0 / float(mu2), texture_state, inner_iterations,
                tensor_geometry)
        next_texture = remainder - smooth
        last_change = float(
            np.linalg.norm(next_cartoon - cartoon)
            / max(np.linalg.norm(next_cartoon), 1e-30))
        cartoon, texture = next_cartoon, next_texture
        maximum_increase = max(
            maximum_increase,
            cartoon_diag["maximum_objective_increase"],
            texture_diag["maximum_objective_increase"],
        )
        maximum_flux = max(
            maximum_flux,
            cartoon_diag["maximum_whitened_flux_norm"],
            texture_diag["maximum_tensor_flux_norm"],
        )
    cartoon = source - texture
    return cartoon, texture, {
        "mu2": float(mu2),
        "outer_iterations": int(outer_iterations),
        "inner_iterations": int(inner_iterations),
        "last_relative_cartoon_change": last_change,
        "maximum_inner_objective_increase": maximum_increase,
        "maximum_dual_flux_norm": maximum_flux,
        "recomposition_linf": float(np.max(np.abs(source - cartoon - texture))),
        "transport_mean_strength": (
            0.0 if tensor_geometry is None
            else float(tensor_geometry["mean_strength"])),
        "transport_condition_maximum": (
            1.0 if tensor_geometry is None
            else float(tensor_geometry["condition_maximum"])),
        "texture_operator": "CONV two-jet" if conv_operator else "forward Hessian",
    }


def reject_scale_persistent_texture(
    source: np.ndarray,
    texture: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Move the CONV-scale-persistent part of a texture candidate to cartoon."""

    value = np.asarray(source, dtype=np.float64)
    candidate = np.asarray(texture, dtype=np.float64)
    coarse_shape = tuple(max(5, int(np.ceil(side / factor))) for side in value.shape)
    coarse = distilled_conv_resize(candidate, coarse_shape)
    persistent = np.asarray(
        distilled_conv_resize(coarse, value.shape), dtype=np.float64)
    fine_texture = candidate - persistent
    cartoon = value - fine_texture
    return cartoon, fine_texture, {
        "factor": int(factor),
        "persistent_rms": float(np.sqrt(np.mean(persistent * persistent))),
        "retained_texture_rms": float(np.sqrt(np.mean(fine_texture * fine_texture))),
        "recomposition_linf": float(np.max(np.abs(value - cartoon - fine_texture))),
    }


def _render(rows: list[tuple], titles: tuple[str, ...], path: Path) -> None:
    cell, label_width, header = 128, 135, 30
    canvas = Image.new(
        "RGB", (label_width + (len(titles) - 1) * cell,
                header + len(rows) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles[1:]):
        draw.text((label_width + column * cell + 3, 8), title, fill="black")
    for row_index, row in enumerate(rows):
        name, *planes = row
        y0 = header + row_index * cell
        draw.text((4, y0 + 8), name, fill="black")
        texture_indices = set(range(2, len(planes), 2))
        scale = max(
            [float(np.percentile(np.abs(planes[index]), 99.5))
             for index in texture_indices] + [1.0])
        for column, plane in enumerate(planes):
            panel = signed(plane, scale) if column in texture_indices else ranged(plane)
            canvas.paste(
                Image.fromarray(np.uint8(panel * 255)).convert("RGB"),
                (label_width + column * cell, y0),
            )
    canvas.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mu_values = (10.0, 40.0, 160.0)
    records: dict[str, dict[str, dict[str, float]]] = {
        "synthetic": {}, "natural": {}}
    synthetic_rows = []
    for name, (source, truth_u, truth_v) in scenes().items():
        identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
        base_u, base_v, _ = metric_meyer(source, identity)
        variants = [second_current_meyer(source, mu2=mu2) for mu2 in mu_values]
        conv_geometry = conv_persistent_twojet_state(source)
        conv_variant = second_current_meyer(
            source, mu2=40.0, tensor_geometry=conv_geometry)
        conv_operator_variant = second_current_meyer(
            source, mu2=40.0, conv_operator=True)
        conv_clean = reject_scale_persistent_texture(
            source, conv_variant[1], factor=2)
        record = {
            "ordinary_cartoon_relative_mse": relative_mse(base_u, truth_u),
            "ordinary_texture_relative_mse": relative_mse(base_v, truth_v),
        }
        for mu2, (cartoon, texture, diagnostic) in zip(mu_values, variants):
            key = f"second_{int(mu2)}"
            record[f"{key}_cartoon_relative_mse"] = relative_mse(cartoon, truth_u)
            record[f"{key}_texture_relative_mse"] = relative_mse(texture, truth_v)
            record.update({f"{key}_{name}": value
                           for name, value in diagnostic.items()})
        record["conv_second_cartoon_relative_mse"] = relative_mse(
            conv_variant[0], truth_u)
        record["conv_second_texture_relative_mse"] = relative_mse(
            conv_variant[1], truth_v)
        record.update({f"conv_second_{entry}": value
                       for entry, value in conv_variant[2].items()})
        record["conv_operator_cartoon_relative_mse"] = relative_mse(
            conv_operator_variant[0], truth_u)
        record["conv_operator_texture_relative_mse"] = relative_mse(
            conv_operator_variant[1], truth_v)
        record.update({f"conv_operator_{entry}": value
                       for entry, value in conv_operator_variant[2].items()})
        record["conv_clean_cartoon_relative_mse"] = relative_mse(
            conv_clean[0], truth_u)
        record["conv_clean_texture_relative_mse"] = relative_mse(
            conv_clean[1], truth_v)
        record.update({f"conv_clean_{entry}": value
                       for entry, value in conv_clean[2].items()})
        records["synthetic"][name] = record
        synthetic_rows.append((
            name, source, truth_u, truth_v, base_u, base_v,
            variants[0][0], variants[0][1], variants[1][0], variants[1][1],
            variants[2][0], variants[2][1], conv_variant[0], conv_variant[1],
            conv_operator_variant[0], conv_operator_variant[1],
            conv_clean[0], conv_clean[1],
        ))

    natural_sources = {
        "camera": normalized(data.camera()),
        "coins": normalized(data.coins()),
        "page": normalized(data.page()),
        "moon": normalized(data.moon()),
    }
    natural_rows = []
    for name, source in natural_sources.items():
        identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
        base_u, base_v, _ = metric_meyer(source, identity)
        variants = [second_current_meyer(source, mu2=mu2) for mu2 in mu_values]
        conv_geometry = conv_persistent_twojet_state(source)
        conv_variant = second_current_meyer(
            source, mu2=40.0, tensor_geometry=conv_geometry)
        conv_operator_variant = second_current_meyer(
            source, mu2=40.0, conv_operator=True)
        conv_clean = reject_scale_persistent_texture(
            source, conv_variant[1], factor=2)
        record = {"ordinary_texture_rms": float(np.sqrt(np.mean(base_v * base_v)))}
        for mu2, (_cartoon, texture, diagnostic) in zip(mu_values, variants):
            key = f"second_{int(mu2)}"
            record[f"{key}_texture_rms"] = float(np.sqrt(np.mean(texture * texture)))
            record.update({f"{key}_{entry}": value
                           for entry, value in diagnostic.items()})
        record["conv_second_texture_rms"] = float(
            np.sqrt(np.mean(conv_variant[1] * conv_variant[1])))
        record.update({f"conv_second_{entry}": value
                       for entry, value in conv_variant[2].items()})
        record["conv_operator_texture_rms"] = float(
            np.sqrt(np.mean(conv_operator_variant[1] * conv_operator_variant[1])))
        record.update({f"conv_operator_{entry}": value
                       for entry, value in conv_operator_variant[2].items()})
        record["conv_clean_texture_rms"] = float(
            np.sqrt(np.mean(conv_clean[1] * conv_clean[1])))
        record.update({f"conv_clean_{entry}": value
                       for entry, value in conv_clean[2].items()})
        records["natural"][name] = record
        natural_rows.append((
            name, source, base_u, base_v,
            variants[0][0], variants[0][1], variants[1][0], variants[1][1],
            variants[2][0], variants[2][1], conv_variant[0], conv_variant[1],
            conv_operator_variant[0], conv_operator_variant[1],
            conv_clean[0], conv_clean[1],
        ))

    base_cartoon = scenes()["coherent oscillation"][1]
    yy, xx = np.mgrid[:base_cartoon.shape[0], :base_cartoon.shape[1]].astype(float)
    x = (xx + 0.5) / base_cartoon.shape[1]
    y = (yy + 0.5) / base_cartoon.shape[0]
    geometry_sources = {
        "compound interface": base_cartoon,
        "diagonal interface": 45.0 + 135.0 * (y > 0.74 - 0.43 * x),
        "disk": 55.0 + 125.0 * (((x - 0.52) ** 2 + (y - 0.48) ** 2) < 0.24 ** 2),
        "thin bars": 50.0 + 120.0 * (
            ((np.abs(x - 0.30) < 0.035) & (y > 0.18) & (y < 0.82))
            | ((np.abs(y - 0.62 + 0.22 * x) < 0.028) & (x > 0.18) & (x < 0.86))
        ),
    }
    geometry_rows = []
    records["geometry_only"] = {}
    for name, source in geometry_sources.items():
        identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
        base_u, base_v, _ = metric_meyer(source, identity)
        second_u, second_v, second_diag = second_current_meyer(source, mu2=40.0)
        conv_geometry = conv_persistent_twojet_state(source)
        conv_u, conv_v, conv_diag = second_current_meyer(
            source, mu2=40.0, tensor_geometry=conv_geometry)
        conv_operator_u, conv_operator_v, conv_operator_diag = second_current_meyer(
            source, mu2=40.0, conv_operator=True)
        clean_u, clean_v, clean_diag = reject_scale_persistent_texture(
            source, conv_v, factor=2)
        edge = gradient_magnitude(source)
        records["geometry_only"][name] = {
            "ordinary_texture_rms": float(np.sqrt(np.mean(base_v * base_v))),
            "ordinary_texture_edge_correlation": correlation(np.abs(base_v), edge),
            "second_texture_rms": float(np.sqrt(np.mean(second_v * second_v))),
            "second_texture_edge_correlation": correlation(np.abs(second_v), edge),
            "conv_second_texture_rms": float(np.sqrt(np.mean(conv_v * conv_v))),
            "conv_second_texture_edge_correlation": correlation(np.abs(conv_v), edge),
            "conv_operator_texture_rms": float(
                np.sqrt(np.mean(conv_operator_v * conv_operator_v))),
            "conv_operator_texture_edge_correlation": correlation(
                np.abs(conv_operator_v), edge),
            "conv_clean_texture_rms": float(np.sqrt(np.mean(clean_v * clean_v))),
            "conv_clean_texture_edge_correlation": correlation(np.abs(clean_v), edge),
            **{f"second_{entry}": value for entry, value in second_diag.items()},
            **{f"conv_second_{entry}": value for entry, value in conv_diag.items()},
            **{f"conv_operator_{entry}": value
               for entry, value in conv_operator_diag.items()},
            **{f"conv_clean_{entry}": value for entry, value in clean_diag.items()},
        }
        geometry_rows.append((
            name, source, base_u, base_v, second_u, second_v, conv_u, conv_v,
            conv_operator_u, conv_operator_v,
            clean_u, clean_v,
        ))

    (OUT / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    labels = tuple(f"second-{int(mu2)} {part}"
                   for mu2 in mu_values for part in ("U", "V"))
    labels += ("CONV-second U", "CONV-second V")
    labels += ("CONV-operator U", "CONV-operator V")
    labels += ("CONV-clean U", "CONV-clean V")
    _render(
        synthetic_rows,
        ("label", "source", "truth U", "truth V", "ordinary U", "ordinary V")
        + labels,
        OUT / "synthetic.png",
    )
    _render(
        natural_rows,
        ("label", "source", "ordinary U", "ordinary V") + labels,
        OUT / "natural.png",
    )
    _render(
        geometry_rows,
        ("label", "source", "ordinary U", "ordinary V", "second U", "second V",
         "CONV-second U", "CONV-second V", "CONV-operator U", "CONV-operator V",
         "CONV-clean U", "CONV-clean V"),
        OUT / "geometry_only.png",
    )


if __name__ == "__main__":
    main()
