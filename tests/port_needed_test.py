"""Contracts for the native vision port queue."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

from port_needed.anisotropic_edge_cost import build_edge_costs
from port_needed.continuous_eikonal_transport import (
    continuous_first_partition,
    prepare_continuous_metric,
)
from port_needed.density_population import emit_density_population
from port_needed.frozen_meyer_geometry import build_frozen_geometry
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
    assert diagnostic["after_action"] < diagnostic["before_action"]
    assert np.max(diagnostic["limited_step_px"]) <= np.max(
        diagnostic["trust_radius_px"]) + 1e-12
    assert np.unique(refreshed["labels"]).size == len(centers)
    assert not np.array_equal(moved, centers)


if __name__ == "__main__":
    test_frozen_geometry_uses_one_target_decomposition()
    test_monotone_bucket_matches_heap_distances()
    test_local_refresh_preserves_every_parent_pixel()
    test_metric_reduction_is_unimodular_and_obtuse()
    test_continuous_front_is_rotation_fair_on_constant_metric()
    test_covector_interface_newton_nearly_equalizes_two_source_mass()
    test_reverse_characteristic_force_balances_symmetric_domain()
    test_density_population_is_locally_emitted_without_a_budget_search()
    test_safe_characteristic_step_preserves_germ_and_lowers_action()
    print("port-needed contracts: ok")
