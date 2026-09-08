"""Dogfood the signed spectral--polarization tensor and light register."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .light_tensor import (
    LightTensor6,
    LightTransportRegister,
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


def _registers() -> tuple[LightTransportRegister, LightTransportRegister]:
    old = LightTransportRegister(
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
    new = LightTransportRegister(
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
    return old, new


def run() -> dict[str, object]:
    start = time.perf_counter()
    wavelength = np.linspace(380.0, 780.0, 1601)
    broad = LightTensor6(1.0, 550.0, 82.0**2, provenance=1 << 0)
    green_absorption = LightTensor6(-0.15, 545.0, 18.0**2, provenance=1 << 0)
    positive = SignedLightLedger.from_packets([broad]).materialize(wavelength)
    signed = SignedLightLedger.from_packets([broad, green_absorption]).materialize(wavelength)
    white_rgb = sensor_rgb(positive)
    magenta_rgb = sensor_rgb(signed)

    def notch(value: np.ndarray) -> np.ndarray:
        return 1.0 - 0.92 * np.exp(-0.5 * ((value - 545.0) / 17.0) ** 2)

    one = adaptive_spectral_filter(broad, notch, tolerance=0.0, max_components=1)
    adaptive = adaptive_spectral_filter(broad, notch, tolerance=0.08, max_components=8)
    oracle = broad.spectral_density(wavelength) * notch(wavelength)
    denominator = float(trapezoidal_integral(np.abs(oracle), wavelength))

    def spectral_error(ledger: SignedLightLedger) -> float:
        represented = ledger.materialize(wavelength).stokes_density[:, 0]
        return float(trapezoidal_integral(np.abs(represented - oracle), wavelength)) / denominator

    incident = LightTensor6(2.0, 550.0, 45.0**2, provenance=1 << 1)
    linear_angle = math.radians(31.0)
    linear_c = math.cos(2.0 * linear_angle)
    linear_s = math.sin(2.0 * linear_angle)
    linear_matrix = 0.5 * np.array(
        [
            [1.0, linear_c, linear_s, 0.0],
            [linear_c, linear_c * linear_c, linear_c * linear_s, 0.0],
            [linear_s, linear_c * linear_s, linear_s * linear_s, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    linear_boundary = mueller_boundary(incident, linear_matrix)
    linear = linear_polarizer(incident, linear_angle)
    circular = circular_polarizer(incident, 1)
    linear_volume = volume_transport(
        linear,
        distance=2.0,
        density=0.7,
        extinction=0.4,
        linear_diffusive_extinction=0.9,
        circular_diffusive_extinction=0.2,
    )
    circular_volume = volume_transport(
        circular,
        distance=2.0,
        density=0.7,
        extinction=0.4,
        linear_diffusive_extinction=0.9,
        circular_diffusive_extinction=0.2,
    )
    diffuse = diffuse_boundary(linear_volume, 0.6)
    unpolarized_background = LightTensor6(
        1.0, 550.0, 45.0**2, provenance=1 << 1
    )
    linear_ensemble = SignedLightLedger.from_packets(
        [unpolarized_background, linear]
    )
    circular_ensemble = SignedLightLedger.from_packets(
        [unpolarized_background, circular]
    )

    def through_volume(ledger: SignedLightLedger) -> SignedLightLedger:
        return volume_transport_ledger(
            ledger,
            distance=2.0,
            density=0.7,
            extinction=0.4,
            linear_diffusive_extinction=0.9,
            circular_diffusive_extinction=0.2,
        )

    def ensemble_degree(ledger: SignedLightLedger) -> float:
        stokes = ledger.materialize(wavelength).stokes_density
        integrated = trapezoidal_integral(stokes, wavelength, axis=0)
        return float(np.linalg.norm(integrated[1:]) / integrated[0])

    linear_ensemble_after = through_volume(linear_ensemble)
    circular_ensemble_after = through_volume(circular_ensemble)

    old_register, new_register = _registers()
    emission = {
        0: SignedLightLedger.from_packets([broad]),
        5: SignedLightLedger.from_packets(
            [LightTensor6(0.7, 620.0, 30.0**2, provenance=1 << 5)]
        ),
    }
    old = old_register.solve(emission, absolute_cutoff=1.0e-14)
    full = new_register.solve(emission, absolute_cutoff=1.0e-14)
    incremental = antimatter_update(
        old_register, new_register, old.state, absolute_cutoff=1.0e-14
    )
    maximum_incremental_error = 0.0
    for expected, actual in zip(full.state, incremental.state):
        expected_stokes = expected.materialize(wavelength).stokes_density
        actual_stokes = actual.materialize(wavelength).stokes_density
        maximum_incremental_error = max(
            maximum_incremental_error,
            float(np.max(np.abs(expected_stokes - actual_stokes))),
        )

    patches, blockers = facing_patch_grids(
        side=2,
        separation=4.0,
        extent=1.5,
        occluder_radius=0.32,
    )
    open_field = compile_dense_centroid_field(patches)
    blocked_field = compile_dense_centroid_field(patches, occluders=blockers)
    reflectance = np.full(len(patches), 0.57)
    open_register = register_from_dense_operator(open_field.dense_matrix(), reflectance)
    blocked_register = register_from_dense_operator(
        blocked_field.dense_matrix(), reflectance
    )
    geometric_emission = {0: SignedLightLedger.from_packets([broad])}
    geometric_old = open_register.solve(geometric_emission, absolute_cutoff=1.0e-15)
    geometric_full = blocked_register.solve(
        geometric_emission, absolute_cutoff=1.0e-15
    )
    geometric_incremental = antimatter_update(
        open_register,
        blocked_register,
        geometric_old.state,
        absolute_cutoff=1.0e-15,
    )
    geometric_planned = update_or_recompute(
        open_register,
        blocked_register,
        geometric_old.state,
        geometric_emission,
        absolute_cutoff=1.0e-15,
    )
    geometric_power_error = max(
        abs(expected.signed_power - actual.signed_power)
        for expected, actual in zip(
            geometric_full.state, geometric_incremental.state
        )
    )

    return {
        "physical_state_coordinates": [
            "signed_power",
            "spectral_centre_nm",
            "spectral_variance_nm2",
            "central_polarization_orientation_degrees",
            "linear_peakedness",
            "signed_chirality",
        ],
        "spectral_signed_ledger": {
            "positive_packet_count": 1,
            "negative_packet_count": 1,
            "cancelled_power": signed.cancelled_power,
            "unmet_negative_power": signed.unmet_negative_power,
            "white_sensor_rgb": white_rgb.tolist(),
            "reconciled_sensor_rgb": magenta_rgb.tolist(),
            "relative_sensor_transmission": (magenta_rgb / white_rgb).tolist(),
        },
        "continuous_spectral_filter": {
            "one_packet_relative_l1_error": spectral_error(one.ledger),
            "adaptive_relative_l1_error": spectral_error(adaptive.ledger),
            "adaptive_component_count": adaptive.component_count,
            "adaptive_internal_closure_error": adaptive.maximum_closure_error,
            "transmitted_power": adaptive.output_power,
        },
        "polarization": {
            "transport_representation": "central_orientation,linear_peakedness,signed_chirality",
            "interaction_ontology": "extinguish_incident_population_then_select_or_generate_outgoing_mode",
            "mueller_evaluation": "transient_boundary_arithmetic_not_a_transport_state",
            "mode_extinction_equation": "w(d)=w(0)*exp(-rho*d*(sigma_t+kappa_L*p+kappa_C*abs(chi)))",
            "polarization_coordinates_change_in_volume": False,
            "linear_boundary_extinguished_power": linear_boundary.extinguished.power,
            "incident_state_after_boundary_evaluation": [
                incident.orientation_degrees,
                incident.linear_peakedness,
                incident.chirality,
            ],
            "linear_polarizer_power": linear.power,
            "linear_orientation_degrees": math.degrees(linear.linear_orientation_radians),
            "linear_mode_power_after_volume": linear_volume.power,
            "linear_mode_state_after_volume": [
                linear_volume.orientation_degrees,
                linear_volume.linear_peakedness,
                linear_volume.chirality,
            ],
            "circular_mode_power_after_volume": circular_volume.power,
            "circular_mode_state_after_volume": [
                circular_volume.orientation_degrees,
                circular_volume.linear_peakedness,
                circular_volume.chirality,
            ],
            "linear_ensemble_degree_before_volume": ensemble_degree(linear_ensemble),
            "linear_ensemble_degree_after_volume": ensemble_degree(linear_ensemble_after),
            "circular_ensemble_degree_before_volume": ensemble_degree(circular_ensemble),
            "circular_ensemble_degree_after_volume": ensemble_degree(circular_ensemble_after),
            "degree_after_diffuse_boundary": diffuse.degree_of_polarization,
        },
        "antimatter_invalidation": {
            "equation": "delta_L=(I-T_new)^-1 (T_new-T_old) L_old",
            "affected_nodes": sorted(incremental.affected_nodes),
            "full_recompute_edge_applications": full.edge_applications,
            "incremental_edge_applications": incremental.edge_applications,
            "incremental_residual_events": incremental.residual_events,
            "maximum_dense_stokes_error": maximum_incremental_error,
            "node_1_signed_correction_power": incremental.correction[1].signed_power,
            "unchanged_branch_provenance": incremental.state[8].provenance,
        },
        "repository_geometry_dogfood": {
            "surface_patch_count": len(patches),
            "open_registered_edges": len(open_register.edges),
            "occluded_registered_edges": len(blocked_register.edges),
            "affected_nodes": sorted(geometric_incremental.affected_nodes),
            "full_recompute_edge_applications": geometric_full.edge_applications,
            "incremental_edge_applications": geometric_incremental.edge_applications,
            "selected_update_mode": geometric_planned.mode,
            "selected_edge_applications": geometric_planned.edge_applications,
            "maximum_signed_power_error": geometric_power_error,
        },
        "elapsed_ms": 1000.0 * (time.perf_counter() - start),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/photonic_light_tensor_foundation.json"),
    )
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
