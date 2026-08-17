#!/usr/bin/env python3
"""Reproducible multiscale preimage of the exact DIP/GDL frontier state.

Every rung constructs a new Bruun/DIP packet chart around the preceding exact
GDL measurement and globally solves that entire chart.  Packet zero is the
incoming state, so physical area cannot increase.  This is a stored
coarse-to-fine transport preimage; it never optimizes SAT clearance and never
selects among decoded Euclidean trial scores.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chip_transport import write_svg
from exact_packet_transport import solve_exact_packet_transport
from spectral_pose_transport import SpectralPoseTransportConfig


@dataclass(frozen=True)
class FrontierPreimageConfig:
    side: float = 4.6755
    packet_count: int = 16
    initial_dip_level: int = 2
    initial_seed: int = 2
    initial_translation_radius: float = 3.0e-5
    initial_phase_radius: float = 1.0e-5
    scales: tuple[tuple[int, float, float], ...] = (
        (20, 1.0e-5, 1.0e-5 / 3.0),
        (20, 3.0e-6, 1.0e-6),
        (160, 1.0e-6, 1.0e-6 / 3.0),
    )
    seed_offset: int = 100


def run_frontier_preimage(config: FrontierPreimageConfig) -> dict:
    initial = SpectralPoseTransportConfig(
        side=config.side,
        packet_count=config.packet_count,
        dip_level=config.initial_dip_level,
        seed=config.initial_seed,
        translation_radius=config.initial_translation_radius,
        phase_radius=config.initial_phase_radius,
    )
    result = solve_exact_packet_transport(initial)
    base = np.asarray(result["poses"], dtype=np.float64)
    trace = [
        {
            "rung": -1,
            "dip_level": config.initial_dip_level,
            "seed": config.initial_seed,
            "translation_radius": config.initial_translation_radius,
            "phase_radius": config.initial_phase_radius,
            "physical_area_energy": result["physical_area_energy"],
            "terminal_sat_audit": result["terminal_sat_audit"],
            "selected_packets": result["selected_packets"],
            "induced_treewidth": result["induced_treewidth"],
        }
    ]
    rung = 0
    for count, translation_radius, phase_radius in config.scales:
        for _ in range(int(count)):
            level = 1 + rung % 4
            seed = config.seed_offset + rung
            packet = SpectralPoseTransportConfig(
                side=config.side,
                packet_count=config.packet_count,
                dip_level=level,
                seed=seed,
                translation_radius=translation_radius,
                phase_radius=phase_radius,
            )
            result = solve_exact_packet_transport(packet, base=base)
            base = np.asarray(result["poses"], dtype=np.float64)
            trace.append(
                {
                    "rung": rung,
                    "dip_level": level,
                    "seed": seed,
                    "translation_radius": translation_radius,
                    "phase_radius": phase_radius,
                    "physical_area_energy": result["physical_area_energy"],
                    "terminal_sat_audit": result["terminal_sat_audit"],
                    "selected_packets": result["selected_packets"],
                    "induced_treewidth": result["induced_treewidth"],
                }
            )
            rung += 1
            if result["terminal_sat_audit"]["overlap_residual"] <= 1.0e-12:
                break
        if result["terminal_sat_audit"]["overlap_residual"] <= 1.0e-12:
            break
    return {
        "method": "exact_bruun_dip_gdl_multiscale_preimage",
        "config": asdict(config),
        "rungs": rung,
        "physical_area_energy": result["physical_area_energy"],
        "terminal_sat_audit": result["terminal_sat_audit"],
        "poses": result["poses"],
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=float, default=4.6755)
    parser.add_argument("--fine-rungs", type=int, default=160)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    config = FrontierPreimageConfig(
        side=args.side,
        scales=(
            (20, 1.0e-5, 1.0e-5 / 3.0),
            (20, 3.0e-6, 1.0e-6),
            (max(int(args.fine_rungs), 0), 1.0e-6, 1.0e-6 / 3.0),
        ),
    )
    result = run_frontier_preimage(config)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.svg is not None:
        write_svg(args.svg, np.asarray(result["poses"]), config.side)


if __name__ == "__main__":
    main()
