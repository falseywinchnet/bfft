#!/usr/bin/env python3
"""Gate leader correspondence by V3 boundary-role context before transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.participation_algebra import (
    normalized_linear_kernel,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
)
from experiments.v3_object_transport.run_contour_transport import _audit
from experiments.v3_object_transport.wavelet_split_transport import (
    analytical_connection_split_transport,
    gated_content_connection,
)


def _seed(name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--controls", nargs="+", default=CONTROLS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "wavelet_gated_transport")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "leader content conjoined with independently measured directed-"
            "incidence role by the Schur product, then analytically split "
            "with proposal topology; matched content-alignment shuffle and "
            "evaluation-only landmarks"
        ),
        "images": {},
    }
    for name in args.controls:
        stages = np.load(args.results / name / "v3_stages.npz")
        base = np.load(
            args.results / "participation_algebra" / name
            / "participation_algebra.npz")["complete"].astype(np.float64)
        proposal = np.load(
            args.results / "proposal_topology_transport" / name / "full.npz"
        )["normalized_connection"].astype(np.float64)
        role_embedding = np.load(
            args.results / "connection_bloom" / name / "full.npz"
        )["region_embedding"].astype(np.float64)
        role = normalized_linear_kernel(role_embedding)
        raw = np.load(
            args.results / "wavelet_leader_transport" / name
            / "wavelet_leader_transport.npz")["region_embedding"].astype(
                np.float64)
        law = np.load(
            args.results / "wavelet_leader_scale_law" / name
            / "wavelet_leader_transport.npz")["region_embedding"].astype(
                np.float64)
        embedding = np.column_stack((raw, law))
        connection = gated_content_connection(embedding, role)
        full = analytical_connection_split_transport(
            proposal, connection, base)
        permutation = np.random.default_rng(_seed(name)).permutation(
            len(embedding))
        shuffled_connection = gated_content_connection(
            embedding[permutation], role)
        shuffled = analytical_connection_split_transport(
            proposal, shuffled_connection, base)
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "full.npz",
            role_context_kernel=role.astype(np.float32),
            **{key: value.astype(np.float32) for key, value in full.items()},
        )
        np.savez_compressed(
            image_output / "shuffled_content_alignment.npz",
            role_context_kernel=role.astype(np.float32),
            **{
                key: value.astype(np.float32)
                for key, value in shuffled.items()
            },
        )
        report["images"][name] = {
            "role_context_audit": _audit(role, stages, landmarks[name]),
            "gated_content_audit": _audit(
                connection, stages, landmarks[name]),
            "transported_base_audit": _audit(
                full["transported_base_kernel"], stages, landmarks[name]),
            "shuffled_gated_content_audit": _audit(
                shuffled_connection, stages, landmarks[name]),
            "shuffled_transported_base_audit": _audit(
                shuffled["transported_base_kernel"], stages,
                landmarks[name]),
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
