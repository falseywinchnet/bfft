#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_line_heat_bath_kernel import (
    block_heat_bath_matrix,
    block_phase_spectral_statistics,
    constant_rank_dense_lattice_obstruction,
    frame_statistics,
    isotropic_kernel_directions,
    line_heat_bath_matrix,
    line_heat_bath_matrices,
    lifting_flow_certificate,
    observable_relaxation_inflation,
    phase_aware_kernel_directions,
    projection_feature_contraction,
    reversible_gap,
    sweep_observable_contraction,
    walsh_hessian_observable_inflation,
)


def test_line_heat_bath_is_reversible_and_connected() -> None:
    coefficients = np.asarray(
        [(i, j) for i in range(-2, 3) for j in range(-2, 3) if (i + j) % 2 == 0],
        dtype=np.int16,
    )
    mass = np.exp(-np.sum(coefficients.astype(float) ** 2, axis=1))
    directions = np.asarray([(1, 1), (1, -1)], dtype=np.int16)
    transition = line_heat_bath_matrix(coefficients, mass, directions)
    stationary = mass / np.sum(mass)
    flow = stationary[:, None] * transition
    assert np.max(np.abs(np.sum(transition, axis=1) - 1.0)) < 1e-12
    assert np.max(np.abs(flow - flow.T)) < 1e-12
    assert reversible_gap(transition, stationary) > 0.0


def test_isotropic_selection_spans() -> None:
    directions = isotropic_kernel_directions(
        3,
        1,
        np.eye(3, dtype=np.uint8),
        np.eye(3),
        search_cutoff=2,
        count=6,
    )
    assert np.linalg.matrix_rank(directions.astype(float)) == 3


def test_observable_inflation_matches_refresh_chains() -> None:
    stationary = np.asarray([0.2, 0.3, 0.5])
    refresh = np.tile(stationary, (3, 1))
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 2.0]])
    assert abs(
        observable_relaxation_inflation(refresh, stationary, values) - 1.0
    ) < 1e-11
    lazy = 0.5 * np.eye(3) + 0.5 * refresh
    assert abs(
        observable_relaxation_inflation(lazy, stationary, values) - 3.0
    ) < 1e-10
    assert np.isinf(
        observable_relaxation_inflation(np.eye(3), stationary, values)
    )


def test_walsh_hessian_audit_covers_every_character() -> None:
    points = np.asarray(
        [[-1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [1.0, 0.0]]
    )
    stationary = np.full(4, 0.25)
    refresh = np.full((4, 4), 0.25)
    tail_bits = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    report = walsh_hessian_observable_inflation(
        refresh, stationary, points, tail_bits)
    assert len(report["walsh_characters"]) == 4
    assert abs(report["worst_asymptotic_variance_inflation"] - 1.0) < 1e-11


def test_frame_statistics_detect_missing_walsh_phase() -> None:
    directions = np.asarray([[1, 0], [0, 1]], dtype=np.int16)
    incomplete = frame_statistics(
        directions,
        np.eye(2),
        2.0,
        np.asarray([[1, 0], [1, 0]], dtype=np.uint8),
    )
    assert incomplete["minimum_nonzero_walsh_character_edge_fraction"] == 0.0
    complete = frame_statistics(
        directions,
        np.eye(2),
        2.0,
        np.eye(2, dtype=np.uint8),
    )
    assert complete["minimum_nonzero_walsh_character_edge_fraction"] == 0.5


def test_phase_aware_selection_spans_both_geometries() -> None:
    directions = phase_aware_kernel_directions(
        3,
        1,
        np.eye(3, dtype=np.uint8),
        np.eye(3),
        search_cutoff=2,
        count=6,
        s=2.0,
    )
    assert np.linalg.matrix_rank(directions.astype(float)) == 3
    tail = (directions & 1)[:, 1:]
    report = frame_statistics(directions, np.eye(3), 2.0, tail)
    assert report["minimum_nonzero_walsh_character_edge_fraction"] > 0.0


def test_projection_sweep_certificate_on_coordinate_refresh() -> None:
    coefficients = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int16)
    mass = np.ones(4)
    projections = line_heat_bath_matrices(
        coefficients,
        mass,
        np.asarray([[1, 0], [0, 1]], dtype=np.int16),
    )
    values = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    report = sweep_observable_contraction(
        projections, np.full(4, 0.25), values, sweeps=2)
    assert report["sweep_contraction"][0] < 1e-12
    certificate = lifting_flow_certificate(
        projections, np.full(4, 0.25), values)
    assert certificate["decomposition_residual"] < 1e-12
    assert abs(certificate["stable_detail_energy"] - 1.0) < 1e-12
    assert abs(
        certificate["asymptotic_variance_inflation_bound"] - 3.0
    ) < 1e-12


def test_full_rank_block_is_an_independent_refresh() -> None:
    coefficients = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int16)
    mass = np.asarray([1.0, 2.0, 3.0, 4.0])
    transition = block_heat_bath_matrix(
        coefficients,
        mass,
        np.asarray([[1, 0], [0, 1]], dtype=np.int16),
    )
    expected = np.tile(mass / np.sum(mass), (4, 1))
    np.testing.assert_allclose(transition, expected)


def test_twisted_dual_margin_for_coordinate_block() -> None:
    report = block_phase_spectral_statistics(
        np.eye(2, dtype=np.int16),
        np.eye(2),
        np.eye(2, dtype=np.uint8),
        2.0,
        2,
    )
    # The nearest nontrivial half-dual character of Z^2 is distance 1/2.
    assert abs(report["minimum_best_twisted_dual_margin_times_s"] - 1.0) < 1e-12
    assert report["minimum_character_block_coverage"] == 1.0


def test_independent_refresh_kills_centered_features_locally() -> None:
    stationary = np.asarray([0.2, 0.3, 0.5])
    refresh = np.tile(stationary, (3, 1))
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 2.0]])
    assert projection_feature_contraction(refresh, stationary, values) < 1e-12


def test_conway_thompson_constant_rank_obstruction() -> None:
    report = constant_rank_dense_lattice_obstruction(0.23675858)
    assert abs(report["xi_r_squared"] - 3.568746145265497) < 1e-12
    assert abs(
        report["fixed_rank_overlap_exponent_bits"]
        - 0.018589786994486504
    ) < 1e-12
    # The obstruction is exponential but still below the present recovery
    # allowance, so it rejects exponent zero without rejecting every useful
    # constant-rank theorem.
    assert 0.0 < report["fixed_rank_overlap_exponent_bits"] < 0.02648284


if __name__ == "__main__":
    test_line_heat_bath_is_reversible_and_connected()
    test_isotropic_selection_spans()
    test_observable_inflation_matches_refresh_chains()
    test_walsh_hessian_audit_covers_every_character()
    test_frame_statistics_detect_missing_walsh_phase()
    test_phase_aware_selection_spans_both_geometries()
    test_projection_sweep_certificate_on_coordinate_refresh()
    test_full_rank_block_is_an_independent_refresh()
    test_twisted_dual_margin_for_coordinate_block()
    test_independent_refresh_kills_centered_features_locally()
    test_conway_thompson_constant_rank_obstruction()
    print("walsh line heat-bath kernel tests passed")
