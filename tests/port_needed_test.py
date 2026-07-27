"""Contracts for the native vision port queue."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

from bfft._core import _vision_fast_march_first_label
from bfft.vision import hard_affine_fit_native, hard_basis_refit_native
import port_needed.continuous_eikonal_transport as continuous_transport
from port_needed.anisotropic_edge_cost import build_edge_costs
from port_needed.continuous_eikonal_transport import (
    _simplex_candidate_with_fraction,
    continuous_first_partition,
    inverse_incidence,
    ordered_local_directions,
    prepare_continuous_metric,
)
from port_needed.density_population import (
    curvature_limited_geometry,
    emit_density_population,
)
from port_needed.frozen_meyer_geometry import build_frozen_geometry
from port_needed.hard_region_fit import (
    _eliminate_small_systems,
    _local_affine_basis,
    _reduce,
)
from port_needed.first_arrival_site_force import (
    backtransport_source_force,
    safe_characteristic_site_step,
)
from port_needed.first_arrival_weight_newton import (
    equalize_first_arrival_mass,
)
from port_needed.metric_reduced_stencil import (
    metric_reduced_superbase,
    validate_obtuse_superbase,
)
from port_needed.pipeline import (
    SegmentingConfig,
    build_segmenting_representation,
)
from port_needed.soft_support_diffusion import (
    build_soft_support_conductance,
    diffuse_soft_support,
)
from port_needed.two_label_transport import (
    hard_partition_with_forest,
    local_hard_partition_with_forest,
    walk_two_labels,
)


def _fixture(height=32, width=40):
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack((
        0.2 + 0.7 * (xx >= width // 2),
        0.15 + 0.6 * yy / max(height - 1, 1),
        0.5 + 0.25 * np.sin(0.4 * xx),
    ), axis=2)
    return np.clip(rgb, 0.0, 1.0)


def test_frozen_geometry_uses_one_target_decomposition():
    geometry = build_frozen_geometry(
        _fixture(), tgfd_sweeps=4, flow_sweeps=4, threads=1)
    assert geometry["target_decompositions"] == 1.0
    assert geometry["measure"].dtype == np.float32
    assert np.isclose(np.sum(geometry["measure"]), 1.0, atol=1e-6)


def test_monotone_bucket_matches_heap_distances():
    geometry = build_frozen_geometry(
        _fixture(), tgfd_sweeps=4, flow_sweeps=4, threads=1)
    costs = build_edge_costs(geometry, 1.5)
    centers = np.array(((0.2, 0.2), (0.8, 0.25), (0.5, 0.8)))
    heap = walk_two_labels(centers, costs, queue="heap")
    bucket = walk_two_labels(centers, costs, queue="bucket")
    assert np.allclose(heap[2], bucket[2], atol=1e-9, rtol=0.0)
    assert np.allclose(heap[3], bucket[3], atol=1e-9, rtol=0.0)


def test_local_refresh_preserves_every_parent_pixel():
    geometry = build_frozen_geometry(
        _fixture(), tgfd_sweeps=4, flow_sweeps=4, threads=1)
    costs = build_edge_costs(geometry, 1.5)
    centers = np.array(((0.2, 0.2), (0.8, 0.25), (0.5, 0.8)))
    parent = hard_partition_with_forest(centers, costs)["labels"]
    children = np.vstack((centers, (0.32, 0.28)))
    parent_of_children = np.array((0, 1, 2, 0), dtype=np.int32)
    _, forest = local_hard_partition_with_forest(
        children, parent_of_children, parent, costs)
    assert np.array_equal(parent_of_children[forest["labels"]], parent)


def test_metric_reduction_is_unimodular_and_obtuse():
    angle = np.deg2rad(23.0)
    tangent = np.array((np.cos(angle), np.sin(angle)))
    normal = np.array((-tangent[1], tangent[0]))
    metric = np.outer(tangent, tangent) + 64.0 * np.outer(normal, normal)
    xx = np.full((5, 7), metric[0, 0])
    xy = np.full((5, 7), metric[0, 1])
    yy = np.full((5, 7), metric[1, 1])
    superbase = metric_reduced_superbase(xx, xy, yy)
    check = validate_obtuse_superbase(superbase, xx, xy, yy)
    assert check["maximum_sum_error"] == 0.0
    assert check["maximum_unimodular_error"] == 0.0
    assert check["maximum_pair_inner_product"] <= 1e-10


def test_continuous_front_is_rotation_fair_on_constant_metric():
    size = 49
    angle = np.deg2rad(23.0)
    tangent = np.array((np.cos(angle), np.sin(angle)))
    normal = np.array((-tangent[1], tangent[0]))
    metric = np.outer(tangent, tangent) + 64.0 * np.outer(normal, normal)
    xx = np.full((size, size), metric[0, 0])
    xy = np.full((size, size), metric[0, 1])
    yy = np.full((size, size), metric[1, 1])
    result = continuous_first_partition(
        np.array(((0.5, 0.5),)), xx, xy, yy)
    grid_y, grid_x = np.mgrid[:size, :size]
    displacement = np.stack((
        grid_x - size // 2,
        grid_y - size // 2,
    ), axis=2)
    exact = np.sqrt(np.einsum(
        "...i,ij,...j->...", displacement, metric, displacement))
    radius = np.hypot(displacement[..., 0], displacement[..., 1])
    mask = radius >= 10.0
    relative = np.abs(result["distance"][mask] - exact[mask]) / exact[mask]
    assert np.all(result["labels"] == 0)
    assert float(np.mean(relative)) < 0.03


def test_native_first_arrival_matches_reference_walk_field_by_field():
    assert _vision_fast_march_first_label is not None
    height, width = 29, 37
    yy, xx = np.mgrid[:height, :width]
    angle = 0.7 * np.sin(xx / 8.0) + 0.5 * np.cos(yy / 7.0)
    tangent_x = np.cos(angle)
    tangent_y = np.sin(angle)
    major = 2.0 + 5.0 * (0.5 + 0.5 * np.sin((xx + yy) / 9.0))
    minor = 0.6 + 0.2 * (0.5 + 0.5 * np.cos((xx - yy) / 6.0))
    mxx = major * tangent_x**2 + minor * tangent_y**2
    mxy = (major - minor) * tangent_x * tangent_y
    myy = major * tangent_y**2 + minor * tangent_x**2
    prepared = prepare_continuous_metric(mxx, mxy, myy)
    centers = np.array((
        (0.13, 0.19),
        (0.78, 0.21),
        (0.54, 0.53),
        (0.19, 0.82),
        (0.87, 0.76),
    ))
    native = continuous_transport.continuous_first_partition_prepared(
        centers, prepared)
    native_entry = continuous_transport.fast_march_first_label_native
    try:
        continuous_transport.fast_march_first_label_native = (
            lambda *args, **kwargs: None)
        reference = (
            continuous_transport.continuous_first_partition_prepared(
                centers, prepared))
    finally:
        continuous_transport.fast_march_first_label_native = native_entry
    for key in (
        "labels",
        "parent_first",
        "parent_second",
        "acceptance_order",
    ):
        assert np.array_equal(native[key], reference[key])
    for key in (
        "distance",
        "gradient_x",
        "gradient_y",
        "source_gradient_x",
        "source_gradient_y",
        "parent_fraction",
    ):
        assert np.allclose(native[key], reference[key], atol=3e-14, rtol=0.0)
    assert native["front_pushes"] == reference["front_pushes"]
    assert native["front_maximum_heap"] == reference["front_maximum_heap"]


def test_native_hard_region_fits_match_conditioned_reference():
    height, width = 21, 27
    yy, xx = np.mgrid[:height, :width]
    labels_2d = ((xx // 4) + 7 * (yy // 5)).astype(np.int32)
    target = np.stack((
        0.1 + 0.7 * xx / width,
        0.2 + 0.5 * yy / height,
        0.4 + 0.2 * np.sin((xx + yy) / 4.0),
    ), axis=2)
    native = hard_affine_fit_native(labels_2d, target)
    assert native is not None
    labels, basis, count, radius, centroid, reconstruction = native
    (
        reference_labels,
        reference_basis,
        reference_count,
        reference_radius,
        reference_centroid,
    ) = _local_affine_basis(labels_2d)
    assert np.array_equal(labels, reference_labels)
    assert np.allclose(basis, reference_basis, atol=3e-15, rtol=0.0)
    assert np.array_equal(count, reference_count)
    assert np.allclose(radius, reference_radius, atol=3e-15, rtol=0.0)
    assert np.allclose(centroid, reference_centroid, atol=3e-15, rtol=0.0)

    cells = len(count)
    target_flat = target.reshape(-1, 3)
    ux, uy = basis[:, 1], basis[:, 2]
    rhs0 = np.column_stack([
        _reduce(labels, target_flat[:, channel], cells)
        for channel in range(3)
    ])
    rhsx = np.column_stack([
        _reduce(labels, ux * target_flat[:, channel], cells)
        for channel in range(3)
    ])
    rhsy = np.column_stack([
        _reduce(labels, uy * target_flat[:, channel], cells)
        for channel in range(3)
    ])
    regularization = 1e-5 * count / np.maximum(radius * radius, 1e-30)
    normal_xx = _reduce(labels, ux * ux, cells) + regularization
    normal_xy = _reduce(labels, ux * uy, cells)
    normal_yy = _reduce(labels, uy * uy, cells) + regularization
    determinant = np.maximum(
        normal_xx * normal_yy - normal_xy * normal_xy, 1e-30)
    coefficient = np.empty((cells, 3, 3))
    coefficient[:, 0] = rhs0 / ((1.0 + 1e-7) * count[:, None])
    coefficient[:, 1] = (
        normal_yy[:, None] * rhsx - normal_xy[:, None] * rhsy
    ) / determinant[:, None]
    coefficient[:, 2] = (
        normal_xx[:, None] * rhsy - normal_xy[:, None] * rhsx
    ) / determinant[:, None]
    reference = np.einsum(
        "ni,nic->nc", basis, coefficient[labels], optimize=False)
    assert np.allclose(
        reconstruction.reshape(-1, 3), reference, atol=3e-14, rtol=0.0)

    ridge = np.tanh((xx.ravel() - 0.37 * yy.ravel()) / 3.0)
    design = np.column_stack((basis, ridge))
    refit = hard_basis_refit_native(
        labels, design, target, count, radius)
    assert refit is not None
    normal = np.empty((cells, 4, 4))
    rhs = np.empty((cells, 4, 3))
    for first in range(4):
        for second in range(first, 4):
            value = _reduce(
                labels, design[:, first] * design[:, second], cells)
            normal[:, first, second] = value
            normal[:, second, first] = value
        for channel in range(3):
            rhs[:, first, channel] = _reduce(
                labels,
                design[:, first] * target_flat[:, channel],
                cells,
            )
    normal[:, 0, 0] += 1e-7 * count
    normal[:, 1, 1] += regularization
    normal[:, 2, 2] += regularization
    normal[:, 3, 3] += 2e-5 * count
    coefficient = _eliminate_small_systems(normal, rhs)
    reference_refit = np.einsum(
        "ni,nic->nc", design, coefficient[labels], optimize=False)
    assert np.allclose(refit, reference_refit, atol=3e-13, rtol=0.0)


def test_local_direction_order_matches_angular_reference():
    angle = np.deg2rad(31.0)
    tangent = np.array((np.cos(angle), np.sin(angle)))
    normal = np.array((-tangent[1], tangent[0]))
    metric = np.outer(tangent, tangent) + 37.0 * np.outer(normal, normal)
    xx = np.full((7, 9), metric[0, 0])
    xy = np.full((7, 9), metric[0, 1])
    yy = np.full((7, 9), metric[1, 1])
    superbase = metric_reduced_superbase(xx, xy, yy)
    actual = ordered_local_directions(superbase)
    signed = np.concatenate((superbase, -superbase), axis=2)
    order = np.argsort(
        np.arctan2(signed[..., 1], signed[..., 0]), axis=2)
    reference = np.take_along_axis(
        signed, order[..., None], axis=2)
    assert np.array_equal(actual, reference)


def test_inverse_incidence_is_unique_and_matches_reference_sets():
    angle = np.deg2rad(19.0)
    tangent = np.array((np.cos(angle), np.sin(angle)))
    normal = np.array((-tangent[1], tangent[0]))
    metric = np.outer(tangent, tangent) + 23.0 * np.outer(normal, normal)
    xx = np.full((8, 10), metric[0, 0])
    xy = np.full((8, 10), metric[0, 1])
    yy = np.full((8, 10), metric[1, 1])
    directions = ordered_local_directions(
        metric_reduced_superbase(xx, xy, yy))
    offset, receivers = inverse_incidence(directions)
    height, width = xx.shape
    expected = [set() for _ in range(height * width)]
    cardinal = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for receiver in range(height * width):
        y, x = divmod(receiver, width)
        local = [tuple(value) for value in directions[y, x]]
        for dx, dy in local + list(cardinal):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                expected[ny * width + nx].add(receiver)
    for vertex in range(height * width):
        actual = receivers[offset[vertex]:offset[vertex + 1]]
        assert len(actual) == len(np.unique(actual))
        assert set(actual.tolist()) == expected[vertex]


def test_closed_form_simplex_matches_high_accuracy_search():
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(500):
        matrix = rng.standard_normal((2, 2))
        metric = matrix.T @ matrix + 0.25 * np.eye(2)
        first = rng.integers(-4, 5, size=2).astype(np.float64)
        second = rng.integers(-4, 5, size=2).astype(np.float64)
        if np.linalg.det(np.column_stack((first, second))) == 0.0:
            continue
        delta_value = rng.uniform(-1.0, 1.0)

        def objective(t):
            point = first + t * (second - first)
            return t * delta_value + np.sqrt(point @ metric @ point)

        low, high = 0.0, 1.0
        for _ in range(80):
            left = (2.0 * low + high) / 3.0
            right = (low + 2.0 * high) / 3.0
            if objective(left) <= objective(right):
                high = right
            else:
                low = left
        reference_t = 0.5 * (low + high)
        value, fraction = _simplex_candidate_with_fraction(
            0.0,
            delta_value,
            first[0],
            first[1],
            second[0],
            second[1],
            metric[0, 0],
            metric[0, 1],
            metric[1, 1],
        )
        assert value <= objective(reference_t) + 1e-8
        assert abs(value - objective(fraction)) < 1e-12
        checked += 1
    assert checked >= 400


def test_covector_interface_newton_nearly_equalizes_two_source_mass():
    size = 81
    one = np.ones((size, size))
    zero = np.zeros((size, size))
    from port_needed.continuous_eikonal_transport import (
        prepare_continuous_metric,
    )

    prepared = prepare_continuous_metric(one, zero, one)
    centers = np.array(((0.15, 0.5), (0.65, 0.5)))
    _, _, trace = equalize_first_arrival_mass(
        centers, prepared, one, passes=1)
    assert trace[0]["mass_cv"] > 0.15
    assert trace[-1]["mass_cv"] < 0.02


def test_reverse_characteristic_force_balances_symmetric_domain():
    size = 81
    one = np.ones((size, size))
    zero = np.zeros((size, size))
    result = continuous_first_partition(
        np.array(((0.5, 0.5),)), one, zero, one)
    force = backtransport_source_force(
        np.array(((0.5, 0.5),)),
        result,
        one,
        core_radius_px=2.0,
    )
    assert np.isclose(force["captured_fraction"][0], 1.0)
    assert force["force_per_mass"][0] < 1e-8


def test_density_population_is_locally_emitted_without_a_budget_search():
    geometry = build_frozen_geometry(
        _fixture(), tgfd_sweeps=4, flow_sweeps=4, threads=1)
    centers, report = emit_density_population(
        geometry, safety_cells=4096)
    assert len(centers) == report["realized_cells"]
    assert abs(len(centers) - report["commanded_cells"]) < 0.15 * max(
        report["commanded_cells"], 1.0)
    assert np.all((centers > 0.0) & (centers < 1.0))


def test_curvature_limits_an_otherwise_long_straight_support():
    size = 65
    yy, xx = np.mgrid[:size, :size]
    x = xx - size // 2
    y = yy - size // 2
    theta = np.arctan2(y, x)
    normal_x = np.cos(theta)
    normal_y = np.sin(theta)
    high, low = 1.0, 0.01
    qxx = high * normal_x**2 + low * normal_y**2
    qxy = (high - low) * normal_x * normal_y
    qyy = high * normal_y**2 + low * normal_x**2
    raw = np.sqrt(qxx * qyy - qxy * qxy) / np.pi
    implied = float(np.sum(raw))
    geometry = {
        "precision_xx": qxx,
        "precision_xy": qxy,
        "precision_yy": qyy,
        "measure": raw / implied,
        "implied_cells": implied,
    }
    curved = curvature_limited_geometry(geometry)
    radius = np.hypot(x, y)
    ring = (radius >= 16.0) & (radius <= 24.0)
    assert np.median(curved["curvature_population_factor"][ring]) > 1.5
    assert curved["implied_cells"] > implied
    assert np.isclose(np.sum(curved["measure"]), 1.0, atol=1e-6)


def test_soft_support_is_a_partition_of_unity_and_respects_target_edges():
    height, width = 24, 30
    one = np.ones((height, width))
    zero = np.zeros((height, width))
    geometry = {
        "precision_xx": one,
        "precision_xy": zero,
        "precision_yy": one,
        "metric_trace_p90": 2.0,
        "max_support_px": 4.0,
    }
    target = np.full((height, width, 3), 0.25)
    target[:, width // 2:] = 0.85
    conductance = build_soft_support_conductance(
        geometry, target, colour_percentile=60.0)
    weights = np.zeros((height, width, 2))
    weights[:, :width // 2, 0] = 1.0
    weights[:, width // 2:, 1] = 1.0
    softened = diffuse_soft_support(
        weights, conductance, passes=8, coupling=0.8)
    assert np.allclose(np.sum(softened, axis=2), 1.0, atol=1e-12)
    assert np.min(softened) >= 0.0
    assert np.max(softened) <= 1.0
    # The real colour step blocks more exchange than an equal-colour seam.
    flat_conductance = build_soft_support_conductance(
        geometry, np.full_like(target, 0.25), colour_percentile=60.0)
    flat_softened = diffuse_soft_support(
        weights, flat_conductance, passes=8, coupling=0.8)
    boundary = width // 2
    assert softened[:, boundary - 1, 1].mean() < (
        flat_softened[:, boundary - 1, 1].mean())


def test_accepted_soft_support_preserves_render_record_contract():
    result = build_segmenting_representation(
        _fixture(24, 30),
        SegmentingConfig(
            allocation_max_side=32,
            tgfd_sweeps=4,
            flow_sweeps=4,
            characteristic_passes=0,
            ridge_count=0,
            soft_support_passes=2,
        ),
    )
    assert "rgb" in result["record"]
    assert result["record"]["rgb"].shape == (24, 30, 3)
    assert result["soft_support"] is not None
    assert "rgb" in result["soft_support"]["proposal_record"]


def test_safe_characteristic_step_preserves_germ_and_lowers_action():
    size = 65
    one = np.ones((size, size))
    zero = np.zeros((size, size))
    prepared = prepare_continuous_metric(one, zero, one)
    centers = np.array(((0.27, 0.39),))
    partition = continuous_first_partition(
        centers, one, zero, one)
    moved, refreshed, diagnostic = safe_characteristic_site_step(
        centers, partition, prepared, one)
    assert diagnostic["accepted"]
    assert diagnostic["descent_direction"]
    assert not diagnostic["converged"]
    assert diagnostic["after_action"] < diagnostic["before_action"]
    assert np.all(
        diagnostic["regularized_curvature_determinant"] > 0.0)
    assert np.max(diagnostic["limited_step_px"]) <= np.max(
        diagnostic["trust_radius_px"]) + 1e-12
    assert np.unique(refreshed["labels"]).size == len(centers)
    assert refreshed["front_maximum_heap"] <= size * size
    assert not np.array_equal(moved, centers)


if __name__ == "__main__":
    test_frozen_geometry_uses_one_target_decomposition()
    test_monotone_bucket_matches_heap_distances()
    test_local_refresh_preserves_every_parent_pixel()
    test_metric_reduction_is_unimodular_and_obtuse()
    test_continuous_front_is_rotation_fair_on_constant_metric()
    test_local_direction_order_matches_angular_reference()
    test_inverse_incidence_is_unique_and_matches_reference_sets()
    test_closed_form_simplex_matches_high_accuracy_search()
    test_covector_interface_newton_nearly_equalizes_two_source_mass()
    test_reverse_characteristic_force_balances_symmetric_domain()
    test_density_population_is_locally_emitted_without_a_budget_search()
    test_safe_characteristic_step_preserves_germ_and_lowers_action()
    print("port-needed contracts: ok")
