#!/usr/bin/env python3
"""Audit scale-causal wavelet-leader content on the V3 control atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.connection_bloom import fit_joint_whitener
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
from experiments.v3_object_transport.wavelet_leader_evidence import (
    region_wavelet_leader_features,
    summarize_wavelet_leaders,
)


def _normalize(kernel: np.ndarray) -> np.ndarray:
    value = 0.5 * (kernel + kernel.T)
    diagonal = np.maximum(np.diag(value), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    return np.divide(
        value, denominator, out=np.zeros_like(value),
        where=denominator > 1e-30)


def _seed(name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--controls", nargs="+", default=CONTROLS)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--representation", choices=("raw_chart", "scale_law"),
        default="scale_law")
    args = parser.parse_args()
    default_name = (
        "wavelet_leader_transport" if args.representation == "raw_chart"
        else "wavelet_leader_scale_law")
    output = args.out or (args.results / default_name)
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    raw = {}
    schema = None
    stages = {}
    control_fields = {}
    for name in args.controls:
        image_dir = args.results / name
        stages[name] = np.load(image_dir / "v3_stages.npz")
        fused = np.load(image_dir / "fused_meyer_evidence.npz")
        control_fields[name] = {
            key: fused[key].astype(np.float64)
            for key in ("target", "cartoon", "texture", "residual")
        }
    analysis_side = max(
        max(field.shape)
        for fields in control_fields.values() for field in fields.values())
    for name in args.controls:
        fields = control_fields[name]
        value, names = region_wavelet_leader_features(
            stages[name]["compound_labels"], fields,
            analysis_side=analysis_side,
            representation=args.representation)
        if schema is None:
            schema = names
        elif names != schema:
            raise RuntimeError("control leader schemas disagree")
        raw[name] = value
    assert schema is not None
    whitener = fit_joint_whitener(raw.values(), schema)
    report = {
        "purpose": (
            "all-scale Haar wavelet-leader content on target plus every fused "
            "Meyer constituent; no Potts labels or iterative inference; "
            "landmarks are evaluation-only"
        ),
        "retained_covariance_modes": int(whitener.basis.shape[1]),
        "analysis_side": int(analysis_side),
        "representation": args.representation,
        "images": {},
    }
    for name in args.controls:
        leader_embedding = whitener.transform(raw[name])
        leader = normalized_linear_kernel(leader_embedding)
        base = np.load(
            args.results / "participation_algebra" / name
            / "participation_algebra.npz")["complete"].astype(np.float64)
        base_leader = complete_kernel_algebra({
            "base": base, "leader": leader})["complete"]
        proposal = np.load(
            args.results / "proposal_topology_transport" / name / "full.npz"
        )["normalized_connection"].astype(np.float64)
        eigenvalue, eigenvector = np.linalg.eigh(0.5 * (proposal + proposal.T))
        multiplier = np.exp(0.5 * (np.clip(
            eigenvalue, -1.0, 1.0) - 1.0))
        half_heat = (eigenvector * multiplier[None, :]) @ eigenvector.T
        transported_leader = _normalize(half_heat @ leader @ half_heat)
        transported_base_leader = _normalize(
            half_heat @ base_leader @ half_heat)

        permutation = np.random.default_rng(_seed(name)).permutation(
            len(leader))
        shuffled_leader = leader[permutation][:, permutation]
        shuffled_base_leader = complete_kernel_algebra({
            "base": base, "leader": shuffled_leader})["complete"]
        shuffled_transport = _normalize(
            half_heat @ shuffled_base_leader @ half_heat)
        kernels = {
            "leader": leader,
            "base_leader_complete": base_leader,
            "proposal_transported_leader": transported_leader,
            "proposal_transported_base_leader": transported_base_leader,
            "shuffled_leader_transport": shuffled_transport,
        }
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "wavelet_leader_transport.npz",
            region_embedding=leader_embedding.astype(np.float32),
            **{key: value.astype(np.float32) for key, value in kernels.items()},
        )
        report["images"][name] = {
            "summary": summarize_wavelet_leaders(raw[name], schema),
            "audits": {
                key: _audit(value, stages[name], landmarks[name])
                for key, value in kernels.items()
            },
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
