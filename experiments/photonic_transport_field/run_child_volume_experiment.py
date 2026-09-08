"""Dogfood the cached two-port child-volume response on deterministic scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .child_volume import (
    AxisAlignedVolume,
    BoundaryFiber,
    ChildEgressBinding,
    ChildIngressBinding,
    ChildLink,
    ChildPortal,
    ChildSurface,
    ChildVolumeResponseSystem,
    ChildVolumeTopology,
    CoupledVolumeTransportSystem,
    InternalKernel,
    InternalNode,
    VolumeMedium,
    compile_child_volume_topology,
    compile_child_volume_rank_blocks,
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


def _ingress(y: float = 0.5) -> BoundaryFiber:
    return BoundaryFiber(
        np.array((0.0, y, 0.5)),
        np.array((-1.0, 0.0, 0.0)),
        np.array((1.0, 0.0, 0.0)),
        0.2,
        0.1,
    )


def _egress(y: float = 0.5, *, reflected: bool = False) -> BoundaryFiber:
    if reflected:
        return BoundaryFiber(
            np.array((0.0, y, 0.5)),
            np.array((-1.0, 0.0, 0.0)),
            np.array((-1.0, 0.0, 0.0)),
            0.04,
            0.04,
        )
    return BoundaryFiber(
        np.array((1.0, y, 0.5)),
        np.array((1.0, 0.0, 0.0)),
        np.array((1.0, 0.0, 0.0)),
        0.2,
        0.1,
    )


def _slab(exit_gain: float, revision: int) -> ChildVolumeTopology:
    medium = VolumeMedium(0.4, 0.3, 0.7, 0.1)
    return ChildVolumeTopology(
        AxisAlignedVolume(np.zeros(3), np.ones(3)),
        (_ingress(0.35), _ingress(0.65)),
        (_egress(),),
        (
            InternalNode(
                np.array((0.5, 0.5, 0.5)),
                InternalKernel("diffuse", 0.5),
            ),
        ),
        (
            ChildLink(ingress(0), internal(0), 0.8, 0.5, medium, 10),
            ChildLink(ingress(1), internal(0), 0.6, 0.5, medium, 11),
            ChildLink(internal(0), egress(0), exit_gain, 0.5, medium, 12),
        ),
        revision,
    )


def _rank_boundary(side: int) -> tuple[AxisAlignedVolume, tuple[BoundaryFiber, ...], tuple[BoundaryFiber, ...]]:
    bounds = AxisAlignedVolume(
        np.array((0.0, -1.0, -1.0)),
        np.array((4.0, 1.0, 1.0)),
    )
    coordinates = np.linspace(-0.3, 0.3, side)
    step = 0.6 / max(side - 1, 1)
    area = step * step
    incoming = tuple(
        BoundaryFiber(
            np.array((0.0, y, z)),
            np.array((-1.0, 0.0, 0.0)),
            np.array((1.0, 0.0, 0.0)),
            area,
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
            area,
            0.01,
        )
        for y in coordinates
        for z in coordinates
    )
    return bounds, incoming, outgoing


def run() -> dict[str, object]:
    start = time.perf_counter()
    mode = LightTensor6(2.0, 530.0, 35.0**2, 27.0, 1.0, 0.0, provenance=1)
    child = ChildVolumeResponseSystem(_slab(0.5, 1), absolute_cutoff=1.0e-14)
    first = child.evaluate({0: SignedLightLedger.from_packets([mode])})
    rescaled = child.evaluate(
        {0: SignedLightLedger.from_packets([mode.scaled(2.5)])}
    )
    correction = child.evaluate(
        {0: SignedLightLedger.from_packets([mode.scaled(-0.5)])}
    )
    second_mode = LightTensor6(1.0, 530.0, 35.0**2, 0.0, 0.0, 1.0, provenance=2)
    localized = child.evaluate(
        {
            0: SignedLightLedger.from_packets([mode]),
            1: SignedLightLedger.from_packets([second_mode]),
        }
    )
    removed_port_responses = child.invalidate_ingress([1])
    old_power = first.outgoing.signed_power
    child.replace_topology(_slab(0.25, 2))
    changed = child.evaluate({0: SignedLightLedger.from_packets([mode])})

    direct_topology = ChildVolumeTopology(
        AxisAlignedVolume(np.zeros(3), np.ones(3)),
        (_ingress(),),
        (_egress(),),
        (),
        (ChildLink(ingress(0), egress(0), 0.5, 1.0, geometry_label=1),),
    )
    direct_child = ChildVolumeResponseSystem(direct_topology, absolute_cutoff=1.0e-14)
    parent = LightTransportRegister(2, (RegisteredEdge(1, 0, 0.2, 20),))
    coupled = CoupledVolumeTransportSystem(
        parent,
        (
            ChildPortal(
                direct_child,
                (ChildIngressBinding(0, 0, 0.5),),
                (ChildEgressBinding(0, 1, 0.8),),
            ),
        ),
    )
    parent_source = SignedLightLedger.from_packets(
        [LightTensor6(1.0, 550.0, 30.0**2)]
    )
    coupled_result = coupled.solve({0: parent_source}, absolute_cutoff=1.0e-14)
    cycle_gain = 0.5 * 0.5 * 0.8 * 0.2
    expected_parent_zero = 1.0 / (1.0 - cycle_gain)

    compiled = compile_child_volume_topology(
        AxisAlignedVolume(np.zeros(3), np.ones(3)),
        (_ingress(),),
        (_egress(reflected=True),),
        (
            ChildSurface(
                np.array((0.8, 0.5, 0.5)),
                np.array((-1.0, 0.0, 0.0)),
                0.04,
                InternalKernel("diffuse", 0.7),
                object_id=4,
            ),
        ),
        revision=7,
    )
    compiled_result = ChildVolumeResponseSystem(
        compiled.topology,
        absolute_cutoff=1.0e-15,
    ).evaluate({0: parent_source})

    rank_measurements = []
    for side in (4, 8, 16):
        rank_bounds, rank_incoming, rank_outgoing = _rank_boundary(side)
        compile_start = time.perf_counter()
        rank_compiled = compile_child_volume_rank_blocks(
            rank_bounds,
            rank_incoming,
            rank_outgoing,
            (),
            admissibility=0.35,
        )
        compile_ms = 1000.0 * (time.perf_counter() - compile_start)
        rank_system = ChildVolumeResponseSystem(rank_compiled.topology)
        rank_input = {
            port: parent_source for port in range(len(rank_incoming))
        }
        evaluate_start = time.perf_counter()
        rank_result = rank_system.evaluate(rank_input)
        evaluate_ms = 1000.0 * (time.perf_counter() - evaluate_start)
        stored = rank_compiled.topology.retained_stored_coefficients
        expanded = rank_compiled.topology.retained_expanded_coefficients
        rank_measurements.append(
            {
                "side": side,
                "ingress_fibers": len(rank_incoming),
                "egress_fibers": len(rank_outgoing),
                "retained_blocks": len(rank_compiled.topology.blocks),
                "stored_coefficients": stored,
                "equivalent_pair_coefficients": expanded,
                "coefficient_compression_ratio": expanded / stored,
                "block_applications": rank_result.block_applications,
                "scalar_edge_applications": rank_result.edge_applications,
                "gathered_source_coefficients": rank_result.gathered_source_coefficients,
                "scattered_receiver_coefficients": rank_result.scattered_receiver_coefficients,
                "unit_response_cache_entries": rank_system.cached_mode_response_count,
                "active_output_fibers": len(rank_result.outgoing.active_ports),
                "compile_ms": compile_ms,
                "evaluate_ms": evaluate_ms,
            }
        )

    return {
        "equation": "L_boundary_out=R_V[L_boundary_in]+E_V",
        "boundary_representation": "role_duplicated_directional_ingress_and_egress_fibers",
        "private_geometry_visible_to_parent": False,
        "cached_response": {
            "first_power": old_power,
            "first_cache_misses": first.cache_misses,
            "first_edge_applications": first.edge_applications,
            "rescaled_power_ratio": rescaled.outgoing.signed_power / old_power,
            "rescaled_cache_hits": rescaled.cache_hits,
            "rescaled_kernel_marches": rescaled.kernel_marches,
            "rescaled_edge_applications": rescaled.edge_applications,
            "signed_correction_power_ratio": correction.outgoing.signed_power / old_power,
            "signed_correction_cache_hits": correction.cache_hits,
            "signed_correction_kernel_marches": correction.kernel_marches,
            "localized_cache_hits": localized.cache_hits,
            "localized_cache_misses": localized.cache_misses,
            "removed_port_responses": removed_port_responses,
            "changed_revision": changed.topology_revision,
            "changed_power_ratio": changed.outgoing.signed_power / old_power,
        },
        "parent_child_feedback": {
            "cycle_gain": cycle_gain,
            "parent_node_zero_power": coupled_result.parent_state[0].signed_power,
            "analytic_parent_node_zero_power": expected_parent_zero,
            "absolute_error": abs(
                coupled_result.parent_state[0].signed_power - expected_parent_zero
            ),
            "parent_edge_applications": coupled_result.parent_edge_applications,
            "child_edge_applications": coupled_result.child_edge_applications,
            "child_response_evaluations": coupled_result.child_response_evaluations,
            "child_cache_hits": coupled_result.child_cache_hits,
            "child_cache_misses": coupled_result.child_cache_misses,
            "residual_events": coupled_result.residual_events,
        },
        "repository_geometry_compiler": {
            "private_surface_count": len(compiled.topology.internal_nodes),
            "compiled_role_directed_links": len(compiled.topology.links),
            "geometry_diagnostics": compiled.geometry_diagnostics,
            "outgoing_power": compiled_result.outgoing.signed_power,
            "active_output_fibers": list(compiled_result.outgoing.active_ports),
        },
        "retained_rank_blocks": rank_measurements,
        "elapsed_ms": 1000.0 * (time.perf_counter() - start),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/photonic_child_volume_response.json"),
    )
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
