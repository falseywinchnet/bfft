#!/usr/bin/env python3
"""Audit one exact-overlap scale-space complex over 128/256/384 V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.audit_resolution_stability import (
    DEFAULT_128,
    DEFAULT_384,
)
from experiments.v3_object_transport.multiscale_proposal_transport import (
    audit_similarity,
    build_multiscale_connection,
    multiscale_point_sources,
    query_multiscale_bloom,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-128", type=Path, default=DEFAULT_128)
    parser.add_argument("--results-256", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--results-384", type=Path, default=DEFAULT_384)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--controls", nargs="+", default=CONTROLS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    roots = {
        128: args.results_128,
        256: args.results_256,
        384: args.results_384,
    }
    output = args.out or (
        args.results_256 / "multiscale_proposal_transport")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "one closed-form heat bloom on the exact-overlap multiplex of "
            "all 128/256/384 proposal graphs; landmarks are query-only"
        ),
        "images": {},
    }
    for name in args.controls:
        labels = {
            side: np.load(root / name / "v3_stages.npz")["compound_labels"]
            for side, root in roots.items()
        }
        proposal = {
            side: np.load(
                root / "proposal_topology_transport" / name / "full.npz"
            )["normalized_connection"].astype(np.float64)
            for side, root in roots.items()
        }
        base = {
            side: np.load(
                root / "participation_algebra" / name
                / "participation_algebra.npz"
            )["complete"].astype(np.float64)
            for side, root in roots.items()
        }
        image_report = {}
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        for arm_name, shuffled in (
            ("full", False), ("shuffled_scale_alignment", True)):
            connection = build_multiscale_connection(
                labels, proposal, shuffled_alignment=shuffled,
                shuffle_key=name)
            names, sources = multiscale_point_sources(
                connection, labels, landmarks[name])
            query = query_multiscale_bloom(connection, sources, base)
            np.savez_compressed(
                image_output / f"{arm_name}.npz",
                names=np.asarray(names, dtype="U64"),
                degree=connection["degree"].astype(np.float32),
                **{
                    key: value.astype(np.float32)
                    for key, value in query.items()
                },
            )
            image_report[arm_name] = {
                "states": int(connection["normalized_connection"].shape[0]),
                "isolated_states": int(np.count_nonzero(
                    connection["degree"] == 0.0)),
                "overlap": connection["overlap"],
                "heat_audit": audit_similarity(
                    names, query["heat_similarity"], landmarks[name]),
                "transported_base_audit": audit_similarity(
                    names, query["transported_base_similarity"],
                    landmarks[name]),
            }
        report["images"][name] = image_report
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
