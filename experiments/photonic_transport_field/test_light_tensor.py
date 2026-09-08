from __future__ import annotations

import math
import unittest

import numpy as np

from .light_tensor import (
    LightTensor6,
    LightTransportRegister,
    PolarizationBoundaryResult,
    RegisteredEdge,
    SignedLightLedger,
    adaptive_spectral_filter,
    antimatter_update,
    circular_polarizer,
    diffuse_boundary,
    linear_polarizer,
    mueller_boundary,
    register_from_dense_operator,
    sensor_rgb,
    trapezoidal_integral,
    update_or_recompute,
    volume_transport,
    volume_transport_ledger,
)
from .scenes import facing_patch_grids
from .transport import compile_dense_centroid_field


class SixDimensionalLightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wavelength = np.linspace(380.0, 780.0, 1601)

    def test_orientation_peakedness_chirality_has_a_hard_realizability_constraint(self) -> None:
        with self.assertRaises(ValueError):
            LightTensor6(1.0, 550.0, 40.0**2, 20.0, 0.8, 0.8)

    def test_linear_and_circular_filters_establish_requested_state(self) -> None:
        unpolarized = LightTensor6(2.0, 550.0, 45.0**2)
        angle = math.radians(31.0)
        linear = linear_polarizer(unpolarized, angle)
        self.assertAlmostEqual(linear.power, 1.0, places=14)
        self.assertAlmostEqual(linear.degree_of_polarization, 1.0, places=14)
        self.assertAlmostEqual(linear.linear_orientation_radians, angle, places=14)
        self.assertAlmostEqual(linear.chirality, 0.0, places=14)
        self.assertAlmostEqual(linear.linear_peakedness, 1.0, places=14)

        circular = circular_polarizer(unpolarized, -1)
        self.assertAlmostEqual(circular.power, 1.0, places=14)
        self.assertAlmostEqual(circular.chirality, -1.0, places=14)
        self.assertAlmostEqual(circular.linear_fraction, 0.0, places=14)

    def test_mueller_boundaries_are_extinction_generation_and_transitive(self) -> None:
        incident = LightTensor6(2.0, 550.0, 45.0**2, 73.0, 0.4, 0.2)

        def linear_matrix(angle_degrees: float) -> np.ndarray:
            angle = math.radians(angle_degrees)
            c = math.cos(2.0 * angle)
            s = math.sin(2.0 * angle)
            return 0.5 * np.array(
                [
                    [1.0, c, s, 0.0],
                    [c, c * c, c * s, 0.0],
                    [s, c * s, s * s, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

        first_matrix = linear_matrix(15.0)
        second_matrix = linear_matrix(41.0)
        first = mueller_boundary(incident, first_matrix)
        second = mueller_boundary(first.outgoing, second_matrix)
        composed = mueller_boundary(incident, second_matrix @ first_matrix)

        self.assertIsInstance(first, PolarizationBoundaryResult)
        self.assertIs(first.incident, incident)
        self.assertEqual(
            (incident.orientation_degrees, incident.linear_peakedness, incident.chirality),
            (73.0, 0.4, 0.2),
        )
        self.assertEqual(first.extinguished, incident.scaled(-1.0))
        np.testing.assert_allclose(
            second.outgoing.power * second.outgoing.derived_stokes,
            composed.outgoing.power * composed.outgoing.derived_stokes,
            rtol=1.0e-14,
            atol=1.0e-14,
        )

    def test_unified_extinction_preserves_each_polarization_mode(self) -> None:
        mixed = LightTensor6(
            3.0,
            540.0,
            50.0**2,
            orientation_degrees=27.0,
            linear_peakedness=math.sqrt(0.5),
            chirality=math.sqrt(0.5),
        )
        transported = volume_transport(
            mixed,
            distance=2.0,
            density=0.7,
            extinction=0.4,
            linear_diffusive_extinction=0.9,
            circular_diffusive_extinction=0.2,
        )
        expected_extinction = 0.4 + 0.9 * math.sqrt(0.5) + 0.2 * math.sqrt(0.5)
        self.assertAlmostEqual(
            transported.power,
            3.0 * math.exp(-expected_extinction * 1.4),
            places=14,
        )
        self.assertEqual(transported.orientation_degrees, 27.0)
        self.assertEqual(transported.linear_peakedness, mixed.linear_peakedness)
        self.assertEqual(transported.chirality, mixed.chirality)
        self.assertEqual(transported.degree_of_polarization, mixed.degree_of_polarization)

        linear_mode = LightTensor6(1.0, 540.0, 50.0**2, 27.0, 1.0, 0.0)
        circular_mode = LightTensor6(1.0, 540.0, 50.0**2, 0.0, 0.0, 1.0)
        linear_after = volume_transport(
            linear_mode,
            distance=2.0,
            density=0.7,
            extinction=0.4,
            linear_diffusive_extinction=0.9,
            circular_diffusive_extinction=0.2,
        )
        circular_after = volume_transport(
            circular_mode,
            distance=2.0,
            density=0.7,
            extinction=0.4,
            linear_diffusive_extinction=0.9,
            circular_diffusive_extinction=0.2,
        )
        self.assertGreater(circular_after.power, linear_after.power)
        ensemble = SignedLightLedger.from_packets(
            [LightTensor6(1.0, 540.0, 50.0**2), linear_mode]
        )
        transported_ensemble = volume_transport_ledger(
            ensemble,
            distance=2.0,
            density=0.7,
            extinction=0.4,
            linear_diffusive_extinction=0.9,
            circular_diffusive_extinction=0.2,
        )
        before = ensemble.materialize(self.wavelength).stokes_density
        after = transported_ensemble.materialize(self.wavelength).stokes_density
        before_stokes = trapezoidal_integral(before, self.wavelength, axis=0)
        after_stokes = trapezoidal_integral(after, self.wavelength, axis=0)
        before_degree = float(np.linalg.norm(before_stokes[1:]) / before_stokes[0])
        after_degree = float(np.linalg.norm(after_stokes[1:]) / after_stokes[0])
        self.assertLess(after_degree, before_degree)
        diffuse = diffuse_boundary(transported, 0.6)
        self.assertAlmostEqual(diffuse.power, 0.6 * transported.power, places=14)
        self.assertEqual(diffuse.degree_of_polarization, 0.0)

    def test_signed_absorption_ledger_makes_magenta_without_negative_light(self) -> None:
        broad = LightTensor6(1.0, 550.0, 82.0**2, provenance=1)
        green_correction = LightTensor6(-0.15, 545.0, 18.0**2, provenance=1)
        positive = SignedLightLedger.from_packets([broad]).materialize(self.wavelength)
        reconciled = SignedLightLedger.from_packets(
            [broad, green_correction]
        ).materialize(self.wavelength)
        white_rgb = sensor_rgb(positive)
        magenta_rgb = sensor_rgb(reconciled)
        relative = magenta_rgb / white_rgb
        self.assertLess(relative[1], relative[0])
        self.assertLess(relative[1], relative[2])
        self.assertGreaterEqual(float(np.min(reconciled.stokes_density[:, 0])), 0.0)
        self.assertGreater(reconciled.cancelled_power, 0.14)
        self.assertEqual(reconciled.unmet_negative_power, 0.0)

    def test_adaptive_mixture_resolves_a_non_gaussian_spectral_notch(self) -> None:
        packet = LightTensor6(1.0, 555.0, 72.0**2)

        def notch(wavelength: np.ndarray) -> np.ndarray:
            return 1.0 - 0.92 * np.exp(-0.5 * ((wavelength - 545.0) / 17.0) ** 2)

        compressed = adaptive_spectral_filter(
            packet, notch, tolerance=0.0, max_components=1
        )
        adaptive = adaptive_spectral_filter(
            packet, notch, tolerance=0.08, max_components=8
        )
        oracle = packet.spectral_density(self.wavelength) * notch(self.wavelength)
        one = compressed.ledger.materialize(self.wavelength).stokes_density[:, 0]
        many = adaptive.ledger.materialize(self.wavelength).stokes_density[:, 0]
        denominator = float(trapezoidal_integral(np.abs(oracle), self.wavelength))
        one_error = float(trapezoidal_integral(np.abs(one - oracle), self.wavelength)) / denominator
        many_error = float(trapezoidal_integral(np.abs(many - oracle), self.wavelength)) / denominator
        self.assertGreater(adaptive.component_count, 1)
        self.assertLess(many_error, one_error)
        self.assertLess(many_error, 0.24)
        power_grid = np.linspace(
            max(1.0, packet.centre_nm - 6.0 * math.sqrt(packet.variance_nm2)),
            packet.centre_nm + 6.0 * math.sqrt(packet.variance_nm2),
            10001,
        )
        expected_power = float(
            trapezoidal_integral(packet.spectral_density(power_grid) * notch(power_grid), power_grid)
        )
        self.assertAlmostEqual(adaptive.output_power, expected_power, delta=2.0e-5)


class AntimatterRegistrationTests(unittest.TestCase):
    def test_register_rejects_superunit_outgoing_energy(self) -> None:
        with self.assertRaises(ValueError):
            LightTransportRegister(
                3,
                [
                    RegisteredEdge(0, 1, 0.7, 1),
                    RegisteredEdge(0, 2, 0.4, 2),
                ],
            )

    @staticmethod
    def _emission() -> dict[int, SignedLightLedger]:
        return {
            0: SignedLightLedger.from_packets(
                [LightTensor6(1.0, 530.0, 55.0**2, provenance=1 << 0)]
            ),
            5: SignedLightLedger.from_packets(
                [LightTensor6(0.7, 620.0, 30.0**2, provenance=1 << 5)]
            ),
        }

    @staticmethod
    def _old_register() -> LightTransportRegister:
        return LightTransportRegister(
            64,
            [
                RegisteredEdge(0, 1, 0.42, 10),
                RegisteredEdge(1, 2, 0.51, 11),
                RegisteredEdge(2, 1, 0.18, 12),
                RegisteredEdge(2, 3, 0.63, 13),
                RegisteredEdge(3, 4, 0.27, 14),
                *(RegisteredEdge(node, node + 1, 0.40, 20 + node) for node in range(5, 63)),
            ],
        )

    @staticmethod
    def _new_register() -> LightTransportRegister:
        return LightTransportRegister(
            64,
            [
                RegisteredEdge(0, 1, 0.11, 10),
                RegisteredEdge(1, 2, 0.51, 11),
                RegisteredEdge(2, 1, 0.18, 12),
                RegisteredEdge(2, 3, 0.63, 13),
                RegisteredEdge(3, 4, 0.27, 14),
                RegisteredEdge(1, 4, 0.16, 15),
                *(RegisteredEdge(node, node + 1, 0.40, 20 + node) for node in range(5, 63)),
            ],
        )

    def test_signed_delta_matches_full_recompute_through_feedback(self) -> None:
        old_register = self._old_register()
        new_register = self._new_register()
        emission = self._emission()
        old = old_register.solve(emission, absolute_cutoff=1.0e-14)
        full = new_register.solve(emission, absolute_cutoff=1.0e-14)
        incremental = antimatter_update(
            old_register,
            new_register,
            old.state,
            absolute_cutoff=1.0e-14,
        )
        wavelength = np.linspace(380.0, 780.0, 401)
        for expected, actual in zip(full.state, incremental.state):
            expected_density = expected.materialize(wavelength).stokes_density
            actual_density = actual.materialize(wavelength).stokes_density
            self.assertTrue(np.allclose(actual_density, expected_density, atol=2.0e-13))
        self.assertEqual(incremental.affected_nodes, frozenset({1, 2, 3, 4}))
        self.assertEqual(
            new_register.affected_closure([(0, 1, 10)]),
            frozenset({1, 2, 3, 4}),
        )
        self.assertLess(incremental.edge_applications, full.edge_applications)
        self.assertLess(incremental.correction[1].signed_power, 0.0)
        self.assertEqual(incremental.state[8].provenance, 1 << 5)
        self.assertEqual(incremental.state[4].provenance, 1 << 0)
        planned = update_or_recompute(
            old_register,
            new_register,
            old.state,
            emission,
            absolute_cutoff=1.0e-14,
        )
        self.assertEqual(planned.mode, "antimatter")
        self.assertEqual(planned.edge_applications, incremental.edge_applications)

    def test_six_coordinate_register_dogfoods_existing_geometry(self) -> None:
        patches, blockers = facing_patch_grids(
            side=2,
            separation=4.0,
            extent=1.5,
            occluder_radius=0.32,
        )
        open_field = compile_dense_centroid_field(patches)
        blocked_field = compile_dense_centroid_field(patches, occluders=blockers)
        reflectance = np.full(len(patches), 0.57)
        old_register = register_from_dense_operator(
            open_field.dense_matrix(), reflectance
        )
        new_register = register_from_dense_operator(
            blocked_field.dense_matrix(), reflectance
        )
        packet = LightTensor6(1.0, 575.0, 64.0**2, provenance=1)
        emission = {0: SignedLightLedger.from_packets([packet])}
        old = old_register.solve(emission, absolute_cutoff=1.0e-15)
        full = new_register.solve(emission, absolute_cutoff=1.0e-15)
        incremental = antimatter_update(
            old_register,
            new_register,
            old.state,
            absolute_cutoff=1.0e-15,
        )
        expected = np.linalg.solve(
            np.eye(len(patches))
            - reflectance[:, None] * blocked_field.dense_matrix(),
            np.eye(len(patches))[0],
        )
        full_power = np.array([ledger.signed_power for ledger in full.state])
        incremental_power = np.array(
            [ledger.signed_power for ledger in incremental.state]
        )
        self.assertTrue(np.allclose(full_power, expected, atol=2.0e-14))
        self.assertTrue(np.allclose(incremental_power, full_power, atol=2.0e-14))
        self.assertGreater(len(incremental.affected_nodes), 0)
        planned = update_or_recompute(
            old_register,
            new_register,
            old.state,
            emission,
            absolute_cutoff=1.0e-15,
        )
        self.assertEqual(planned.mode, "full-remarch")
        self.assertEqual(planned.edge_applications, full.edge_applications)


if __name__ == "__main__":
    unittest.main()
