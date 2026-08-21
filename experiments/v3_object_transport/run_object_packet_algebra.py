#!/usr/bin/env python3
"""Construct the first complete typed soft-object packet kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.participation_algebra import (
    complete_kernel_algebra,
    normalized_linear_kernel,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
)
from experiments.v3_object_transport.run_contour_transport import _audit
from experiments.v3_object_transport.wavelet_split_transport import (
    normalize_kernel,
)


def _seed(name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def _half_heat(connection: np.ndarray) -> np.ndarray:
    value = 0.5 * (
        np.asarray(connection, dtype=np.float64)
        + np.asarray(connection, dtype=np.float64).T)
    eigenvalue, eigenvector = np.linalg.eigh(value)
    multiplier = np.exp(0.5 * (np.clip(
        eigenvalue, -1.0, 1.0) - 1.0))
    return (eigenvector * multiplier[None, :]) @ eigenvector.T


def _payload(
    half_heat: np.ndarray,
    embedding: np.ndarray,
) -> np.ndarray:
    kernel = normalized_linear_kernel(embedding)
    return normalize_kernel(half_heat @ kernel @ half_heat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--controls", nargs="+", default=CONTROLS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "object_packet_algebra")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "complete positive algebra of structural parts, separately "
            "transported raw-leader and scale-law payloads, and ordered "
            "incidence endpoint role; every region column is a soft packet, "
            "with a matched joint correspondence shuffle"
        ),
        "images": {},
    }
    for name in args.controls:
        stages = np.load(args.results / name / "v3_stages.npz")
        parts = np.load(
            args.results / "participation_algebra" / name
            / "participation_algebra.npz")["complete"].astype(np.float64)
        proposal = np.load(
            args.results / "proposal_topology_transport" / name / "full.npz"
        )["normalized_connection"].astype(np.float64)
        half_heat = _half_heat(proposal)
        raw_embedding = np.load(
            args.results / "wavelet_leader_transport" / name
            / "wavelet_leader_transport.npz")["region_embedding"].astype(
                np.float64)
        law_embedding = np.load(
            args.results / "wavelet_leader_scale_law" / name
            / "wavelet_leader_transport.npz")["region_embedding"].astype(
                np.float64)
        incidence_dir = args.results / "wavelet_incidence_transport" / name
        ordered_role = np.load(
            incidence_dir / "ordered_endpoints.npz"
        )["role_kernel"].astype(np.float64)
        shuffled_ordered_role = np.load(
            incidence_dir / "shuffled_ordered_endpoints.npz"
        )["role_kernel"].astype(np.float64)
        raw_payload = _payload(half_heat, raw_embedding)
        law_payload = _payload(half_heat, law_embedding)
        permutation = np.random.default_rng(_seed(name)).permutation(
            len(raw_embedding))
        shuffled_raw_payload = _payload(
            half_heat, raw_embedding[permutation])
        shuffled_law_payload = _payload(
            half_heat, law_embedding[permutation])
        coordinates = {
            "parts": parts,
            "raw_payload": raw_payload,
            "scale_law_payload": law_payload,
            "ordered_endpoint_role": ordered_role,
        }
        shuffled_coordinates = {
            "parts": parts,
            "raw_payload": shuffled_raw_payload,
            "scale_law_payload": shuffled_law_payload,
            "ordered_endpoint_role": shuffled_ordered_role,
        }
        complete = complete_kernel_algebra(coordinates)["complete"]
        shuffled_complete = complete_kernel_algebra(
            shuffled_coordinates)["complete"]
        without_ordered = complete_kernel_algebra({
            key: value for key, value in coordinates.items()
            if key != "ordered_endpoint_role"
        })["complete"]
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "object_packet_algebra.npz",
            **{
                key: value.astype(np.float32)
                for key, value in coordinates.items()
            },
            complete=complete.astype(np.float32),
            without_ordered=without_ordered.astype(np.float32),
            shuffled_complete=shuffled_complete.astype(np.float32),
        )
        report["images"][name] = {
            "coordinate_audits": {
                key: _audit(value, stages, landmarks[name])
                for key, value in coordinates.items()
            },
            "without_ordered_audit": _audit(
                without_ordered, stages, landmarks[name]),
            "complete_audit": _audit(
                complete, stages, landmarks[name]),
            "shuffled_complete_audit": _audit(
                shuffled_complete, stages, landmarks[name]),
            "packet_self_return": {
                "minimum": float(np.min(np.diag(complete))),
                "median": float(np.median(np.diag(complete))),
                "maximum": float(np.max(np.diag(complete))),
            },
            "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(
                complete))),
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
