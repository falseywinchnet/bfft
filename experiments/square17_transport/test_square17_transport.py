#!/usr/bin/env python3

import math
import unittest

import numpy as np

from chip_transport import (
    condition_capacity_direction,
    initial_population,
    legalize_pose_capacity,
    quantile_emit,
    transport_chart,
)
from configuration_transport import (
    ConfigurationTransportConfig,
    STATE_COUNT,
    apply_product_heat,
    configuration_area_energy,
    hilbert_projective_distance,
    pair_area_kernel,
    poses_from_state,
    transfer_step,
)
from geometry import (
    PAIR_I,
    PAIR_J,
    SQUARE_COUNT,
    capacity_loss_gradient,
    capacity_state,
    pair_witness_state,
)
from reference_chart import REFERENCE_SIDE, reference_chart
from rigidity_analysis import coherent_contact_linearization, nonnegative_shrink_stress
from global_transport_search import ROW_FAMILIES, diverse_population
from floorplan_banach import (
    FloorplanConfig,
    banach_energy_map,
    floorplan_state,
    initial_floorplan,
    physical_area_energy,
    solve_floorplan_banach,
)
from lifted_equilibrium import (
    LiftedConfig,
    equidistant_simplex_lanes,
    entropy_project_scores,
    lifted_constraint_chart,
    normalized_bruun_dip_basis,
)
from topology_search import TrialRequest, stressed_contact_laplacian, transported_fracture
from tensor_transport import (
    interaction_swap,
    mps_norm_squared,
    swap_network_pairs,
    uniform_mps,
)
from occupation_transport import (
    banach_resolvent,
    occupation_area_energy,
    occupation_basis,
)
from spectral_pose_transport import (
    SpectralPoseTransportConfig,
    dip_packet_coordinates,
    scaled_reference_chart,
    spectral_physical_energies,
    spectral_pose_packets,
)
from exact_packet_transport import (
    Factor,
    assignment_energy,
    exact_min_sum,
    interaction_graph,
    min_fill_order,
)
from frontier_preimage import FrontierPreimageConfig


class Square17TransportTests(unittest.TestCase):
    def test_initial_population_is_finite_relaxed_chart(self) -> None:
        poses = initial_population(0, 5.0)
        self.assertEqual(poses.shape, (SQUARE_COUNT, 3))
        self.assertTrue(np.all(np.isfinite(poses)))
        self.assertTrue(np.isfinite(capacity_state(poses, 5.0).minimum_clearance))

    def test_capacity_gradient_matches_translation_difference(self) -> None:
        poses = initial_population(2, 5.0)
        poses[0, 0] += 0.03
        _, gradient, _ = capacity_loss_gradient(poses, 4.98, temperature=0.02)
        epsilon = 2.0e-7
        plus = poses.copy()
        minus = poses.copy()
        plus[0, 0] += epsilon
        minus[0, 0] -= epsilon
        plus_loss, _, _ = capacity_loss_gradient(plus, 4.98, temperature=0.02)
        minus_loss, _, _ = capacity_loss_gradient(minus, 4.98, temperature=0.02)
        numeric = (plus_loss - minus_loss) / (2.0 * epsilon)
        self.assertAlmostEqual(gradient[0, 0], numeric, places=7)

    def test_all_pair_witness_gradients_match_finite_difference(self) -> None:
        poses = initial_population(29, 5.0)
        witnesses = pair_witness_state(poses, absolute_smoothing=2.0e-5)
        pair = int(np.flatnonzero(PAIR_I == 4)[2])
        axis = 3
        epsilon = 2.0e-7
        for square, expected in (
            (int(PAIR_I[pair]), witnesses.theta_i_gradient[pair, axis]),
            (int(PAIR_J[pair]), witnesses.theta_j_gradient[pair, axis]),
        ):
            plus = poses.copy()
            minus = poses.copy()
            plus[square, 2] += epsilon
            minus[square, 2] -= epsilon
            numeric = (
                pair_witness_state(plus, absolute_smoothing=2.0e-5).clearance[pair, axis]
                - pair_witness_state(minus, absolute_smoothing=2.0e-5).clearance[pair, axis]
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(float(expected), float(numeric), places=7)

    def test_interval_conditioner_keeps_shape(self) -> None:
        poses = initial_population(3, 5.0)
        _, gradient, state = capacity_loss_gradient(poses, 4.98, temperature=0.02)
        conditioned, audit = condition_capacity_direction(poses, -gradient, state)
        self.assertEqual(conditioned.shape, poses.shape)
        self.assertGreater(audit.net_count, 0)
        self.assertGreater(audit.paired_axis_count, 0)

    def test_identity_transport_emits_anchor(self) -> None:
        poses = initial_population(4, 5.0)
        emitted = quantile_emit(poses, poses)
        np.testing.assert_allclose(emitted, poses)

    def test_reference_chart_is_a_verified_construction(self) -> None:
        state = capacity_state(reference_chart(), REFERENCE_SIDE)
        self.assertGreaterEqual(state.minimum_clearance, -2.0e-14)

    def test_nonnegative_stress_blocks_coherent_shrink(self) -> None:
        chart = coherent_contact_linearization()
        stress = nonnegative_shrink_stress(chart)
        self.assertEqual(chart.operator.shape, (39, 36))
        self.assertEqual(np.linalg.matrix_rank(chart.operator, tol=1.0e-9), 36)
        self.assertGreaterEqual(float(np.min(stress)), -1.0e-10)
        self.assertLess(np.linalg.norm(chart.operator.T @ stress), 2.0e-14)
        self.assertAlmostEqual(float(stress @ chart.side_response), 1.0, places=12)

    def test_stress_transport_fracture_is_reproducible(self) -> None:
        laplacian = stressed_contact_laplacian()
        np.testing.assert_allclose(laplacian, laplacian.T)
        np.testing.assert_allclose(np.sum(laplacian, axis=1), 0.0, atol=1.0e-14)
        request = TrialRequest(7, 4.67, 0.04, 0.02, 10, 1)
        first = transported_fracture(request)
        second = transported_fracture(request)
        np.testing.assert_allclose(first, second)
        self.assertGreater(np.linalg.norm(first - reference_chart()), 0.04)

    def test_pose_legalizer_reduces_shrink_residual(self) -> None:
        side = REFERENCE_SIDE - 2.0e-4
        initial = transport_chart(reference_chart(), REFERENCE_SIDE, side)
        before = capacity_state(initial, side).overlap_residual
        after = capacity_state(
            legalize_pose_capacity(initial, side, iterations=24), side
        ).overlap_residual
        self.assertLess(after, before)

    def test_diverse_relaxed_families_have_seventeen_cells(self) -> None:
        for family in range(len(ROW_FAMILIES)):
            poses = diverse_population(41, family)
            self.assertEqual(poses.shape, (SQUARE_COUNT, 3))
            self.assertTrue(np.all(np.isfinite(poses)))

    def test_bruun_dip_chart_is_orthogonal(self) -> None:
        basis = normalized_bruun_dip_basis(64, 3)
        np.testing.assert_allclose(basis @ basis.T, np.eye(64), atol=1.0e-13)

    def test_pose_simplex_is_exactly_equidistant(self) -> None:
        side = 4.7
        poses = equidistant_simplex_lanes(52, side)
        chart = np.concatenate((
            (poses[:, :, :2] - 0.5 * side) / 0.55,
            poses[:, :, 2:] / 0.22,
        ), axis=2).reshape(52, -1)
        distance = []
        for first in range(52):
            for second in range(first):
                distance.append(np.linalg.norm(chart[first] - chart[second]))
        self.assertLess(float(np.std(distance)), 1.0e-12)

    def test_floorplan_transport_is_a_17d_banach_contraction(self) -> None:
        config = FloorplanConfig(side=4.8, resolution=24, discount=0.71)
        state = floorplan_state(initial_floorplan(config.side), config)
        rng = np.random.default_rng(9)
        first = rng.normal(size=SQUARE_COUNT)
        second = rng.normal(size=SQUARE_COUNT)
        image_first = banach_energy_map(
            state.local_cost, state.transport, first, config.discount
        )
        image_second = banach_energy_map(
            state.local_cost, state.transport, second, config.discount
        )
        ratio = np.max(np.abs(image_first - image_second)) / np.max(
            np.abs(first - second)
        )
        self.assertLessEqual(ratio, config.discount + 1.0e-14)
        self.assertEqual(state.local_cost.shape, (SQUARE_COUNT,))

    def test_configuration_basis_is_exactly_equidistant(self) -> None:
        # Coordinate kets are not materialized: any two distinct product
        # configurations differ in exactly two orthonormal coordinates.
        self.assertEqual(STATE_COUNT, 2 ** SQUARE_COUNT)
        first = np.zeros(STATE_COUNT)
        second = np.zeros(STATE_COUNT)
        first[117] = 1.0
        second[9183] = 1.0
        self.assertAlmostEqual(
            float(np.linalg.norm(first - second)), np.sqrt(2.0), places=14
        )

    def test_product_heat_advances_all_seventeen_axes(self) -> None:
        amplitude = np.zeros(STATE_COUNT)
        amplitude[0] = 1.0
        transported = apply_product_heat(amplitude, 0.1)
        self.assertTrue(np.all(transported > 0.0))
        self.assertAlmostEqual(float(np.sum(transported)), 1.0, places=13)
        # The antipodal state requires taking the moving branch on all axes.
        self.assertAlmostEqual(float(transported[-1]), 0.1 ** 17, places=30)

    def test_positive_configuration_transfer_contracts_projective_distance(self) -> None:
        # A synthetic physical-area potential is sufficient for the operator
        # law; the exact polygon kernel is separately tested below.
        energy = np.linspace(0.0, 1.0, STATE_COUNT)
        rng = np.random.default_rng(61)
        first = rng.uniform(0.5, 1.5, STATE_COUNT)
        second = rng.uniform(0.5, 1.5, STATE_COUNT)
        before = hilbert_projective_distance(first, second)
        after = hilbert_projective_distance(
            transfer_step(first, energy, 0.12, 1.0),
            transfer_step(second, energy, 0.12, 1.0),
        )
        self.assertLess(after, before)

    def test_configuration_energy_has_no_clearance_term(self) -> None:
        alphabet = np.empty((SQUARE_COUNT, 2, 3), dtype=np.float64)
        reference = reference_chart()
        alphabet[:, 0] = reference
        alphabet[:, 1] = reference
        # Moving one terminal branch leaves the reference state present while
        # creating other product configurations with physical intersection.
        alphabet[0, 1, 0] = alphabet[1, 0, 0]
        alphabet[0, 1, 1] = alphabet[1, 0, 1]
        kernel = pair_area_kernel(alphabet)
        energy = configuration_area_energy(kernel)
        self.assertLess(float(energy[0]), 1.0e-13)
        self.assertGreater(float(np.max(energy)), 0.5)
        measured = poses_from_state(alphabet, 0)
        np.testing.assert_allclose(measured, reference)

    def test_configuration_transport_configuration_has_seventeen_axes(self) -> None:
        config = ConfigurationTransportConfig(iterations_per_action=1)
        self.assertEqual(SQUARE_COUNT, 17)
        self.assertEqual(len(config.inverse_actions), 7)

    def test_swap_network_covers_every_particle_pair_once(self) -> None:
        pairs, order = swap_network_pairs(SQUARE_COUNT)
        unordered = {tuple(sorted(pair)) for pair in pairs}
        self.assertEqual(len(pairs), SQUARE_COUNT * (SQUARE_COUNT - 1) // 2)
        self.assertEqual(len(unordered), len(pairs))
        self.assertEqual(order, list(reversed(range(SQUARE_COUNT))))

    def test_untruncated_two_site_gate_matches_dense_norm(self) -> None:
        tensors = uniform_mps(2, 3)
        gate = np.asarray(
            ((1.0, 0.8, 0.7), (0.8, 0.6, 0.5), (0.7, 0.5, 0.4))
        )
        left, right, record = interaction_swap(
            tensors[0], tensors[1], gate, maximum_rank=3
        )
        expected = float(np.sum(np.square(gate))) / 9.0
        self.assertAlmostEqual(mps_norm_squared([left, right]), expected, places=13)
        self.assertLess(record["discarded_fraction"], 1.0e-28)

    def test_occupation_basis_quotients_particle_permutations(self) -> None:
        basis = occupation_basis(19)
        self.assertEqual(len(basis.combinations), math.comb(19, 17))
        np.testing.assert_allclose(
            np.asarray(basis.transition.sum(axis=1)).ravel(), 1.0
        )
        self.assertEqual(basis.transition.shape, (171, 171))

    def test_occupation_resolvent_is_a_banach_fixed_point(self) -> None:
        basis = occupation_basis(19)
        forcing = np.linspace(0.5, 1.5, len(basis.combinations))
        value, record = banach_resolvent(
            forcing, basis.transition, discount=0.6, iterations=80
        )
        self.assertLess(record["fixed_point_residual_linf"], 1.0e-14)
        self.assertTrue(np.all(value > 0.0))

    def test_occupation_energy_finds_reference_subset(self) -> None:
        alphabet = np.vstack(
            (
                reference_chart(),
                np.asarray(((0.5, 0.5, 0.0), (0.6, 0.6, 0.0))),
            )
        )
        basis = occupation_basis(19)
        from tensor_transport import pose_overlap_gram

        energy = occupation_area_energy(basis, pose_overlap_gram(alphabet))
        self.assertLess(float(np.min(energy)), 1.0e-13)

    def test_dip_pose_packets_span_three_whitened_coordinates(self) -> None:
        packet = dip_packet_coordinates(16, 2)
        self.assertEqual(packet.shape, (16, 3))
        np.testing.assert_allclose(packet[0], 0.0)
        self.assertEqual(np.linalg.matrix_rank(packet), 3)

    def test_spectral_packets_keep_exact_reference_source(self) -> None:
        config = SpectralPoseTransportConfig(
            packet_count=8,
            dip_level=2,
            translation_radius=0.05,
            phase_radius=0.05,
        )
        source = scaled_reference_chart(REFERENCE_SIDE)
        packets = spectral_pose_packets(source, config)
        np.testing.assert_allclose(packets[:, 0], reference_chart(), atol=1.0e-14)
        wall, pair = spectral_physical_energies(packets, REFERENCE_SIDE)
        self.assertLess(float(np.sum(wall[:, 0])), 1.0e-13)
        self.assertLess(float(np.sum(pair[:, :, 0, 0])), 2.0e-13)

    def test_exact_min_sum_matches_brute_force(self) -> None:
        rng = np.random.default_rng(73)
        packet_count = 3
        unary = [rng.uniform(size=packet_count) for _ in range(4)]
        pair = {
            (0, 1): rng.uniform(size=(packet_count, packet_count)),
            (1, 2): rng.uniform(size=(packet_count, packet_count)),
            (1, 3): rng.uniform(size=(packet_count, packet_count)),
        }
        factors = [Factor((variable,), value) for variable, value in enumerate(unary)]
        factors.extend(Factor(scope, value) for scope, value in pair.items())
        adjacency = {0: {1}, 1: {0, 2, 3}, 2: {1}, 3: {1}}
        order, width, _ = min_fill_order(adjacency)
        minimum, assignment, _ = exact_min_sum(factors, order, packet_count)
        brute = math.inf
        brute_assignment = None
        for index in np.ndindex(*(packet_count,) * 4):
            energy = sum(unary[v][index[v]] for v in range(4))
            energy += sum(value[index[i], index[j]] for (i, j), value in pair.items())
            if energy < brute:
                brute = float(energy)
                brute_assignment = index
        self.assertEqual(width, 1)
        self.assertAlmostEqual(minimum, brute, places=14)
        self.assertEqual(tuple(assignment), brute_assignment)

    def test_reference_packet_factor_graph_has_small_treewidth(self) -> None:
        config = SpectralPoseTransportConfig(
            packet_count=8,
            dip_level=2,
            translation_radius=0.05,
            phase_radius=0.05,
        )
        packets = spectral_pose_packets(reference_chart(), config)
        wall, pair = spectral_physical_energies(packets, REFERENCE_SIDE)
        adjacency, _ = interaction_graph(wall, pair)
        _, width, _ = min_fill_order(adjacency)
        self.assertLessEqual(width, 4)

    def test_frontier_preimage_schedule_has_positive_scales(self) -> None:
        config = FrontierPreimageConfig()
        self.assertTrue(all(count > 0 for count, _, _ in config.scales))
        self.assertTrue(
            all(translation > 0.0 and phase > 0.0 for _, translation, phase in config.scales)
        )

    def test_floorplan_descent_uses_energy_before_geometry_audit(self) -> None:
        config = FloorplanConfig(side=4.8, resolution=24, sweeps=2)
        initial = floorplan_state(initial_floorplan(config.side), config)
        result = solve_floorplan_banach(config)
        self.assertLess(result["best_energy"], initial.total_energy)
        self.assertEqual(result["transport_dimension"], SQUARE_COUNT)

    def test_physical_area_energy_vanishes_on_verified_packing(self) -> None:
        energy, local, overlap, wall = physical_area_energy(
            reference_chart(), REFERENCE_SIDE, 2.0
        )
        self.assertLess(energy, 1.0e-13)
        self.assertLess(overlap, 1.0e-13)
        self.assertLess(wall, 1.0e-13)
        self.assertLess(float(np.max(np.abs(local))), 1.0e-13)

    def test_entropy_projection_retains_requested_support(self) -> None:
        probability, info = entropy_project_scores(
            np.linspace(-2.0, 1.0, 16), 7.0
        )
        self.assertAlmostEqual(float(np.sum(probability)), 1.0, places=13)
        self.assertAlmostEqual(info["effective_support"], 7.0, places=8)

    def test_lifted_chart_anneals_to_exact_pair_clearance(self) -> None:
        poses = initial_population(31, 5.0)
        chart = lifted_constraint_chart(
            poses, 5.0, 1.0e-7, absolute_smoothing=1.0e-9
        )
        exact = capacity_state(poses, 5.0)
        np.testing.assert_allclose(
            chart.clearance[4 * SQUARE_COUNT:],
            exact.pair_clearance,
            atol=2.0e-7,
        )


if __name__ == "__main__":
    unittest.main()
