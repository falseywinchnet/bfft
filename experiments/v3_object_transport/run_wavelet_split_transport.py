#!/usr/bin/env python3
"""Compose leader-content and proposal-topology heat on V3 regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
)
from experiments.v3_object_transport.run_contour_transport import _audit
from experiments.v3_object_transport.wavelet_split_transport import (
    analytical_split_transport,
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
    output = args.out or (args.results / "wavelet_split_transport")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "parameter-free symmetric split heat between explicit proposal "
            "topology and raw-plus-scale-law leader content, with a matched "
            "content-to-region permutation null; landmarks are evaluation-only"
        ),
        "images": {},
    }
    for name in args.controls:
        image_dir = args.results / name
        stages = np.load(image_dir / "v3_stages.npz")
        base = np.load(
            args.results / "participation_algebra" / name
            / "participation_algebra.npz")["complete"].astype(np.float64)
        proposal_data = np.load(
            args.results / "proposal_topology_transport" / name / "full.npz")
        proposal = proposal_data["normalized_connection"].astype(np.float64)
        proposal_baseline = proposal_data[
            "transported_base_kernel"].astype(np.float64)
        raw = np.load(
            args.results / "wavelet_leader_transport" / name
            / "wavelet_leader_transport.npz")["region_embedding"].astype(
                np.float64)
        law = np.load(
            args.results / "wavelet_leader_scale_law" / name
            / "wavelet_leader_transport.npz")["region_embedding"].astype(
                np.float64)
        embedding = np.column_stack((raw, law))
        full = analytical_split_transport(proposal, embedding, base)
        permutation = np.random.default_rng(_seed(name)).permutation(
            len(embedding))
        shuffled = analytical_split_transport(
            proposal, embedding[permutation], base)
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "full.npz",
            **{key: value.astype(np.float32) for key, value in full.items()},
        )
        np.savez_compressed(
            image_output / "shuffled_content_alignment.npz",
            **{
                key: value.astype(np.float32)
                for key, value in shuffled.items()
            },
        )
        report["images"][name] = {
            "proposal_only_audit": _audit(
                proposal_baseline, stages, landmarks[name]),
            "split_heat_audit": _audit(
                full["split_heat_kernel"], stages, landmarks[name]),
            "transported_base_audit": _audit(
                full["transported_base_kernel"], stages, landmarks[name]),
            "shuffled_split_heat_audit": _audit(
                shuffled["split_heat_kernel"], stages, landmarks[name]),
            "shuffled_transported_base_audit": _audit(
                shuffled["transported_base_kernel"], stages,
                landmarks[name]),
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
