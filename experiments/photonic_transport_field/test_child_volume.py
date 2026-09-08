from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .child_volume import (
    AxisAlignedVolume,
    BoundaryFiber,
    ChildLink,
    ChildEgressBinding,
    ChildIngressBinding,
    ChildPortal,
    ChildSurface,
    ChildTransportBlock,
    ChildVolumeResponseSystem,
    ChildVolumeTopology,
    CoupledVolumeTransportSystem,
    compile_child_volume_topology,
    compile_child_volume_rank_blocks,
    InternalKernel,
    InternalNode,
    VolumeMedium,
    egress,
    ingress,
    internal,
)
from .light_tensor import (
    LightTensor6,
    LightTransportRegister,
    RegisteredEdge,
    SignedLightLedger,
)
from .native_retained import native_available


def _linear_matrix(angle_degrees: float) -> np.ndarray:
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


class ChildVolumeResponseTests(unittest.TestCase):
    @staticmethod
    def _ingress(y: float = 0.5) -> BoundaryFiber:
        return BoundaryFiber(
            np.array((0.0, y, 0.5)),
            np.array((-1.0, 0.0, 0.0)),
            np.array((1.0, 0.0, 0.0)),
            area=0.2,
            solid_angle=0.1,
        )

    @staticmethod
    def _egress(y: float = 0.5) -> BoundaryFiber:
        return BoundaryFiber(
            np.array((1.0, y, 0.5)),
            np.array((1.0, 0.0, 0.0)),
            np.array((1.0, 0.0, 0.0)),
            area=0.2,
            solid_angle=0.1,
        )

    def _slab(self, *, revision: int = 0, exit_gain: float = 0.5) -> ChildVolumeTopology:
        medium = VolumeMedium(
            density=0.4,
            extinction=0.3,
            linear_diffusive_extinction=0.7,
            circular_diffusive_extinction=0.1,
        )
        return ChildVolumeTopology(
            AxisAlignedVolume(np.zeros(3), np.ones(3)),
            (self._ingress(0.35), self._ingress(0.65)),
            (self._egress(),),
            (
                InternalNode(
                    np.array((0.5, 0.5, 0.5)),
                    InternalKernel("diffuse", throughput=0.5),
                ),
            ),
            (
                ChildLink(ingress(0), internal(0), 0.8, 0.5, medium, 10),
                ChildLink(ingress(1), internal(0), 0.6, 0.5, medium, 11),
                ChildLink(internal(0), egress(0), exit_gain, 0.5, medium, 12),
            ),
            revision,
        )

    def test_role_duplicated_boundary_is_directionally_validated(self) -> None:
        wrong = BoundaryFiber(
            np.array((0.0, 0.5, 0.5)),
            np.array((-1.0, 0.0, 0.0)),
            np.array((-1.0, 0.0, 0.0)),
            0.2,
            0.1,
        )
        with self.assertRaises(ValueError):
            ChildVolumeTopology(
                AxisAlignedVolume(np.zeros(3), np.ones(3)),
                (wrong,),
                (self._egress(),),
                (),
                (),
            )
        with self.assertRaises(ValueError):
            ChildLink(egress(0), internal(0), 0.5, 0.1)

    def test_medium_extinction_and_diffuse_generation_are_composed(self) -> None:
        topology = self._slab()
        system = ChildVolumeResponseSystem(topology, absolute_cutoff=1.0e-14)
        incident = LightTensor6(2.0, 530.0, 35.0**2, 27.0, 1.0, 0.0)
        result = system.evaluate(
            {0: SignedLightLedger.from_packets([incident])}
        )
        first_retention = math.exp(-0.4 * 0.5 * (0.3 + 0.7))
        second_retention = math.exp(-0.4 * 0.5 * 0.3)
        expected = 2.0 * 0.8 * first_retention * 0.5 * 0.5 * second_retention
        packet = result.outgoing.ledgers[0].packets[0]
        self.assertAlmostEqual(packet.power, expected, places=14)
        self.assertEqual(
            (packet.orientation_degrees, packet.linear_peakedness, packet.chirality),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(result.outgoing.active_ports, (0,))

    def test_unit_mode_cache_reuses_intensity_changes_and_localizes_ports(self) -> None:
        system = ChildVolumeResponseSystem(self._slab(), absolute_cutoff=1.0e-14)
        mode = LightTensor6(2.0, 530.0, 35.0**2, 27.0, 1.0, 0.0)
        first = system.evaluate({0: SignedLightLedger.from_packets([mode])})
        doubled = system.evaluate(
            {0: SignedLightLedger.from_packets([replace(mode, power=4.0)])}
        )
        correction = system.evaluate(
            {0: SignedLightLedger.from_packets([replace(mode, power=-1.0)])}
        )
        second_port = system.evaluate(
            {
                0: SignedLightLedger.from_packets([mode]),
                1: SignedLightLedger.from_packets([mode]),
            }
        )
        self.assertEqual((first.cache_hits, first.cache_misses), (0, 1))
        self.assertEqual((doubled.cache_hits, doubled.cache_misses), (1, 0))
        self.assertEqual(doubled.kernel_marches, 0)
        self.assertAlmostEqual(
            doubled.outgoing.signed_power,
            2.0 * first.outgoing.signed_power,
            places=14,
        )
        self.assertEqual((correction.cache_hits, correction.cache_misses), (1, 0))
        self.assertEqual(correction.kernel_marches, 0)
        self.assertAlmostEqual(
            correction.outgoing.signed_power,
            -0.5 * first.outgoing.signed_power,
            places=14,
        )
        self.assertEqual((second_port.cache_hits, second_port.cache_misses), (1, 1))
        self.assertEqual(system.cached_mode_response_count, 2)
        self.assertEqual(system.invalidate_ingress([1]), 1)
        self.assertEqual(system.cached_mode_response_count, 1)

    def test_internal_feedback_is_integrated_then_sealed_at_egress(self) -> None:
        topology = ChildVolumeTopology(
            AxisAlignedVolume(np.zeros(3), np.ones(3)),
            (self._ingress(),),
            (self._egress(),),
            (
                InternalNode(
                    np.array((0.5, 0.5, 0.5)),
                    InternalKernel("diffuse", 0.5),
                ),
            ),
            (
                ChildLink(ingress(0), internal(0), 0.5, 0.0, geometry_label=1),
                ChildLink(internal(0), internal(0), 0.2, 0.0, geometry_label=2),
                ChildLink(internal(0), egress(0), 0.4, 0.0, geometry_label=3),
            ),
        )
        system = ChildVolumeResponseSystem(topology, absolute_cutoff=1.0e-14)
        incident = LightTensor6(1.0, 550.0, 30.0**2, 20.0, 1.0, 0.0)
        result = system.evaluate({0: SignedLightLedger.from_packets([incident])})
        expected = 0.5 * 0.5 * 0.4 / (1.0 - 0.2 * 0.5)
        self.assertAlmostEqual(result.outgoing.signed_power, expected, places=12)
        self.assertLess(result.residual_events, 20)
        self.assertEqual(result.outgoing.packet_count, 1)

    def test_mueller_surface_generates_a_transitive_outgoing_mode(self) -> None:
        topology = ChildVolumeTopology(
            AxisAlignedVolume(np.zeros(3), np.ones(3)),
            (self._ingress(),),
            (self._egress(),),
            (
                InternalNode(
                    np.array((0.5, 0.5, 0.5)),
                    InternalKernel("mueller", 1.0, _linear_matrix(31.0)),
                ),
            ),
            (
                ChildLink(ingress(0), internal(0), 1.0, 0.5, geometry_label=1),
                ChildLink(internal(0), egress(0), 1.0, 0.5, geometry_label=2),
            ),
        )
        system = ChildVolumeResponseSystem(topology, absolute_cutoff=1.0e-14)
        incident = LightTensor6(2.0, 550.0, 30.0**2)
        packet = system.evaluate(
            {0: SignedLightLedger.from_packets([incident])}
        ).outgoing.ledgers[0].packets[0]
        self.assertAlmostEqual(packet.power, 1.0, places=14)
        self.assertAlmostEqual(packet.orientation_degrees, 31.0, places=14)
        self.assertAlmostEqual(packet.linear_peakedness, 1.0, places=14)
        self.assertEqual(packet.chirality, 0.0)

    def test_emission_is_cached_and_egress_gathers_into_parent_nodes(self) -> None:
        emitted = SignedLightLedger.from_packets(
            [LightTensor6(0.3, 610.0, 22.0**2, provenance=8)]
        )
        topology = ChildVolumeTopology(
            AxisAlignedVolume(np.zeros(3), np.ones(3)),
            (),
            (self._egress(0.35), self._egress(0.65)),
            (InternalNode(np.array((0.5, 0.5, 0.5)), emission=emitted),),
            (
                ChildLink(internal(0), egress(0), 0.25, 0.0, geometry_label=1),
                ChildLink(internal(0), egress(1), 0.50, 0.0, geometry_label=2),
            ),
        )
        system = ChildVolumeResponseSystem(topology)
        first = system.evaluate({})
        second = system.evaluate({})
        self.assertEqual((first.kernel_marches, second.kernel_marches), (1, 0))
        self.assertAlmostEqual(first.outgoing.signed_power, 0.225, places=14)
        parent = first.outgoing.scatter_to_parent({0: 7, 1: 7})
        self.assertEqual(tuple(parent), (7,))
        self.assertAlmostEqual(parent[7].signed_power, 0.225, places=14)

    def test_topology_replacement_invalidates_only_the_child_response(self) -> None:
        system = ChildVolumeResponseSystem(self._slab(revision=3))
        incident = SignedLightLedger.from_packets(
            [LightTensor6(1.0, 530.0, 35.0**2)]
        )
        old = system.evaluate({0: incident})
        self.assertEqual(system.cached_mode_response_count, 1)
        system.replace_topology(self._slab(revision=4, exit_gain=0.25))
        self.assertEqual(system.cached_mode_response_count, 0)
        new = system.evaluate({0: incident})
        self.assertEqual(new.topology_revision, 4)
        self.assertEqual(new.cache_misses, 1)
        self.assertAlmostEqual(
            new.outgoing.signed_power,
            0.5 * old.outgoing.signed_power,
            places=14,
        )

    def test_repository_geometry_compiler_discovers_private_child_links(self) -> None:
        bounds = AxisAlignedVolume(np.zeros(3), np.ones(3))
        reflected = BoundaryFiber(
            np.array((0.0, 0.5, 0.5)),
            np.array((-1.0, 0.0, 0.0)),
            np.array((-1.0, 0.0, 0.0)),
            area=0.04,
            solid_angle=0.04,
        )
        compiled = compile_child_volume_topology(
            bounds,
            (self._ingress(),),
            (reflected,),
            (
                ChildSurface(
                    np.array((0.8, 0.5, 0.5)),
                    np.array((-1.0, 0.0, 0.0)),
                    area=0.04,
                    kernel=InternalKernel("diffuse", 0.7),
                    object_id=4,
                ),
            ),
            revision=5,
        )
        identities = {
            (link.source.role, link.receiver.role)
            for link in compiled.topology.links
        }
        self.assertIn(("ingress", "internal"), identities)
        self.assertIn(("internal", "egress"), identities)
        self.assertNotIn(("egress", "internal"), identities)
        self.assertNotIn(("internal", "ingress"), identities)
        self.assertGreater(compiled.geometry_diagnostics["exact_visibility_tests"], 0)
        system = ChildVolumeResponseSystem(compiled.topology, absolute_cutoff=1.0e-15)
        result = system.evaluate(
            {0: SignedLightLedger.from_packets([LightTensor6(1.0, 550.0, 30.0**2)])}
        )
        self.assertGreater(result.outgoing.signed_power, 0.0)
        self.assertEqual(result.topology_revision, 5)

    def test_parent_child_feedback_uses_response_only_after_single_emission(self) -> None:
        child_topology = ChildVolumeTopology(
            AxisAlignedVolume(np.zeros(3), np.ones(3)),
            (self._ingress(),),
            (self._egress(),),
            (),
            (ChildLink(ingress(0), egress(0), 0.5, 1.0, geometry_label=1),),
        )
        child = ChildVolumeResponseSystem(child_topology, absolute_cutoff=1.0e-14)
        parent = LightTransportRegister(
            2,
            (RegisteredEdge(1, 0, 0.2, 20),),
        )
        coupled = CoupledVolumeTransportSystem(
            parent,
            (
                ChildPortal(
                    child,
                    (ChildIngressBinding(0, 0, 0.5),),
                    (ChildEgressBinding(0, 1, 0.8),),
                ),
            ),
        )
        source = SignedLightLedger.from_packets(
            [LightTensor6(1.0, 550.0, 30.0**2)]
        )
        result = coupled.solve({0: source}, absolute_cutoff=1.0e-14)
        cycle_gain = 0.5 * 0.5 * 0.8 * 0.2
        expected_zero = 1.0 / (1.0 - cycle_gain)
        expected_one = expected_zero * 0.5 * 0.5 * 0.8
        self.assertAlmostEqual(result.parent_state[0].signed_power, expected_zero, places=12)
        self.assertAlmostEqual(result.parent_state[1].signed_power, expected_one, places=12)
        self.assertGreater(result.child_cache_hits, 0)
        self.assertEqual(result.child_cache_misses, 1)
        self.assertLess(result.residual_events, 30)

    def test_retained_block_matches_its_expanded_pair_operator(self) -> None:
        incoming = (self._ingress(0.35), self._ingress(0.65))
        outgoing = (self._egress(0.35), self._egress(0.65))
        source_factor = np.array((0.8, 0.5))
        receiver_factor = np.array((0.3, 0.4))
        block = ChildTransportBlock(
            (ingress(0), ingress(1)),
            (egress(0), egress(1)),
            source_factor,
            receiver_factor,
            1.0,
            geometry_label=7,
        )
        rank_topology = ChildVolumeTopology(
            AxisAlignedVolume(np.zeros(3), np.ones(3)),
            incoming,
            outgoing,
            (),
            (),
            blocks=(block,),
        )
        expanded_links = tuple(
            ChildLink(
                ingress(source),
                egress(receiver),
                float(source_factor[source] * receiver_factor[receiver]),
                1.0,
                geometry_label=10 * receiver + source,
            )
            for source in range(2)
            for receiver in range(2)
        )
        expanded_topology = ChildVolumeTopology(
            AxisAlignedVolume(np.zeros(3), np.ones(3)),
            incoming,
            outgoing,
            (),
            expanded_links,
        )
        mode = LightTensor6(1.0, 550.0, 30.0**2, 17.0, 0.4, 0.2)
        field = {
            0: SignedLightLedger.from_packets([mode.scaled(2.0)]),
            1: SignedLightLedger.from_packets([mode.scaled(3.0)]),
        }
        retained = ChildVolumeResponseSystem(rank_topology).evaluate(field)
        expanded = ChildVolumeResponseSystem(expanded_topology).evaluate(field)
        for retained_port, expanded_port in zip(
            retained.outgoing.ledgers,
            expanded.outgoing.ledgers,
        ):
            self.assertAlmostEqual(
                retained_port.signed_power,
                expanded_port.signed_power,
                places=14,
            )
        self.assertEqual(retained.evaluation_strategy, "retained")
        self.assertEqual(retained.block_applications, 1)
        self.assertEqual(retained.gathered_source_coefficients, 2)
        self.assertEqual(retained.scattered_receiver_coefficients, 2)
        self.assertEqual(retained.edge_applications, 0)
        self.assertEqual(expanded.edge_applications, 4)

    def test_retained_blocks_share_the_same_outgoing_energy_budget(self) -> None:
        oversized = ChildTransportBlock(
            (ingress(0),),
            (egress(0), egress(1)),
            np.array((1.0,)),
            np.array((0.7, 0.4)),
            1.0,
            geometry_label=9,
        )
        with self.assertRaises(ValueError):
            ChildVolumeTopology(
                AxisAlignedVolume(np.zeros(3), np.ones(3)),
                (self._ingress(),),
                (self._egress(0.35), self._egress(0.65)),
                (),
                (),
                blocks=(oversized,),
            )

    def test_hierarchical_compiler_retains_coherent_boundary_rank(self) -> None:
        bounds = AxisAlignedVolume(
            np.array((0.0, -1.0, -1.0)),
            np.array((4.0, 1.0, 1.0)),
        )
        coordinates = np.linspace(-0.3, 0.3, 4)
        incoming = tuple(
            BoundaryFiber(
                np.array((0.0, y, z)),
                np.array((-1.0, 0.0, 0.0)),
                np.array((1.0, 0.0, 0.0)),
                0.01,
                0.01,
            )
            for y in coordinates
            for z in coordinates
        )
        outgoing = tuple(
            BoundaryFiber(
                np.array((4.0, y, z)),
                np.array((1.0, 0.0, 0.0)),
                np.array((1.0, 0.0, 0.0)),
                0.01,
                0.01,
            )
            for y in coordinates
            for z in coordinates
        )
        compiled = compile_child_volume_rank_blocks(
            bounds,
            incoming,
            outgoing,
            (),
            admissibility=0.35,
        )
        topology = compiled.topology
        self.assertEqual(len(topology.links), 0)
        self.assertGreater(len(topology.blocks), 0)
        self.assertLess(
            topology.retained_stored_coefficients,
            topology.retained_expanded_coefficients,
        )
        field = {
            port: SignedLightLedger.from_packets(
                [LightTensor6(1.0, 550.0, 30.0**2)]
            )
            for port in range(len(incoming))
        }
        system = ChildVolumeResponseSystem(topology)
        result = system.evaluate(field)
        self.assertEqual(result.evaluation_strategy, "retained")
        self.assertLessEqual(result.block_applications, len(topology.blocks))
        self.assertEqual(result.outgoing.active_ports, tuple(range(len(outgoing))))
        self.assertEqual(system.cached_mode_response_count, 0)

    @unittest.skipUnless(native_available(), "native retained-rank library is not built")
    def test_native_rank_blocks_match_python_through_diffuse_feedback(self) -> None:
        incoming = (self._ingress(0.35), self._ingress(0.65))
        outgoing = (self._egress(0.35), self._egress(0.65))
        medium = VolumeMedium(0.35, 0.22, 0.41, 0.09)
        nodes = (
            InternalNode(
                np.array((0.35, 0.5, 0.5)),
                InternalKernel("diffuse", 0.72),
            ),
            InternalNode(
                np.array((0.70, 0.5, 0.5)),
                InternalKernel("identity", 0.81),
            ),
        )
        blocks = (
            ChildTransportBlock(
                (ingress(0), ingress(1)),
                (internal(0),),
                np.array((0.44, 0.31)),
                np.array((0.9,)),
                0.35,
                medium,
                1,
            ),
            ChildTransportBlock(
                (internal(0),),
                (internal(1), egress(0)),
                np.array((0.8,)),
                np.array((0.42, 0.31)),
                0.40,
                medium,
                2,
            ),
            ChildTransportBlock(
                (internal(1),),
                (internal(1), egress(1)),
                np.array((0.7,)),
                np.array((0.16, 0.49)),
                0.25,
                medium,
                3,
            ),
        )
        topology = ChildVolumeTopology(
            AxisAlignedVolume(np.zeros(3), np.ones(3)),
            incoming,
            outgoing,
            nodes,
            (),
            blocks=blocks,
        )
        first = LightTensor6(1.3, 460.0, 19.0**2, 13.0, 0.6, 0.2, 1)
        second = LightTensor6(-0.27, 610.0, 24.0**2, 81.0, 0.1, -0.7, 2)
        field = {
            0: SignedLightLedger.from_packets((first, second)),
            1: SignedLightLedger.from_packets((first.scaled(0.4),)),
        }
        oracle = ChildVolumeResponseSystem(
            topology,
            absolute_cutoff=1.0e-14,
            execution_backend="python",
        ).evaluate(field)
        native = ChildVolumeResponseSystem(
            topology,
            absolute_cutoff=1.0e-14,
            execution_backend="native",
        ).evaluate(field)
        for expected, actual in zip(oracle.outgoing.ledgers, native.outgoing.ledgers):
            self.assertEqual(len(expected.packets), len(actual.packets))
            for expected_packet, actual_packet in zip(expected.packets, actual.packets):
                self.assertEqual(_mode_tuple(expected_packet), _mode_tuple(actual_packet))
                self.assertAlmostEqual(expected_packet.power, actual_packet.power, places=13)
        self.assertNotEqual(native.execution_backend, "python")
        self.assertGreater(native.native_wave_calls, 1)
        self.assertGreater(native.contiguous_source_blocks, 0)
        self.assertEqual(native.block_applications, oracle.block_applications)

    @unittest.skipUnless(native_available(), "native retained-rank library is not built")
    def test_packed_terminal_field_matches_materialized_response(self) -> None:
        incoming = (self._ingress(0.35), self._ingress(0.65))
        outgoing = (self._egress(0.35), self._egress(0.65))
        block = ChildTransportBlock(
            (ingress(0), ingress(1)),
            (egress(0), egress(1)),
            np.array((0.8, 0.5)),
            np.array((0.3, 0.4)),
            0.7,
            VolumeMedium(0.2, 0.3, 0.4, 0.1),
            geometry_label=7,
        )
        topology = ChildVolumeTopology(
            AxisAlignedVolume(np.zeros(3), np.ones(3)),
            incoming,
            outgoing,
            (),
            (),
            blocks=(block,),
        )
        modes = (
            LightTensor6(1.0, 460.0, 18.0**2, 11.0, 0.4, 0.2, 1),
            LightTensor6(1.0, 610.0, 21.0**2, 70.0, 0.1, -0.5, 2),
        )
        powers = np.array(((2.0, 3.0), (-0.4, 0.9)))
        system = ChildVolumeResponseSystem(topology, execution_backend="native")
        packed = system.evaluate_packed(modes, powers)
        ordinary = system.evaluate(
            {
                port: SignedLightLedger.from_packets(
                    mode.scaled(float(powers[index, port]))
                    for index, mode in enumerate(modes)
                )
                for port in range(2)
            }
        )
        materialized = packed.materialize()
        for expected, actual in zip(ordinary.outgoing.ledgers, materialized.ledgers):
            self.assertEqual(len(expected.packets), len(actual.packets))
            for expected_packet, actual_packet in zip(expected.packets, actual.packets):
                self.assertEqual(_mode_tuple(expected_packet), _mode_tuple(actual_packet))
                self.assertAlmostEqual(expected_packet.power, actual_packet.power, places=14)


def _mode_tuple(packet: LightTensor6) -> tuple[float | int, ...]:
    return (
        packet.centre_nm,
        packet.variance_nm2,
        packet.orientation_degrees,
        packet.linear_peakedness,
        packet.chirality,
        packet.provenance,
    )


if __name__ == "__main__":
    unittest.main()
