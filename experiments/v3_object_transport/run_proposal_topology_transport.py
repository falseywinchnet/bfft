#!/usr/bin/env python3
"""Audit the closed-form bloom of explicit assembly proposal topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.participation_algebra import (
    complete_kernel_algebra,
)
from experiments.v3_object_transport.proposal_topology_transport import (
    analytical_proposal_bloom,
    build_proposal_connection,
    summarize_proposal_connection,
)
from experiments.v3_object_transport.relative_enclosure import (
    build_relative_enclosures,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
    _load_complex,
)
from experiments.v3_object_transport.run_contour_transport import _audit
from experiments.v3_object_transport.support_manifold_transport import (
    build_support_manifold_transport,
)


def _seed(name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--controls", nargs="+", default=CONTROLS)
    args = parser.parse_args()
    output = args.out or (args.results / "proposal_topology_transport")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "closed-form unit heat bloom of every explicit off-diagonal "
            "support--manifold proposal; landmarks are evaluation-only"
        ),
        "images": {},
    }
    for name in args.controls:
        complex_ = _load_complex(
            args.results / name / "compound_region_complex.npz")
        enclosure = build_relative_enclosures(complex_)
        base = np.load(
            args.results / "participation_algebra" / name
            / "participation_algebra.npz"
        )["complete"].astype(np.float64)
        stages = np.load(args.results / name / "v3_stages.npz")

        full = np.load(
            args.results / "support_manifold_transport" / name / "full.npz")
        arms = {
            "full": (
                enclosure["participation"],
                full["support_manifold_weight"].astype(np.float64),
            ),
        }
        permutation = np.random.default_rng(_seed(name)).permutation(
            int(complex_["region_count"]))
        shuffled = dict(enclosure)
        shuffled["participation"] = enclosure["participation"][:, permutation]
        shuffled_transport = build_support_manifold_transport(
            complex_, shuffled, include_centeredness=True, include_scale=True)
        arms["shuffled_manifold_members"] = (
            shuffled["participation"],
            shuffled_transport["support_manifold_weight"],
        )

        image_report = {}
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        for arm_name, (participation, weight) in arms.items():
            connection = build_proposal_connection(participation, weight)
            bloom = analytical_proposal_bloom(
                connection["normalized_connection"], base)
            complete = complete_kernel_algebra({
                "base": base,
                "proposal_heat": bloom["heat_kernel"],
            })["complete"]
            path = image_output / f"{arm_name}.npz"
            np.savez_compressed(
                path,
                normalized_connection=connection[
                    "normalized_connection"].astype(np.float32),
                degree=connection["degree"].astype(np.float32),
                self_measure=connection["self_measure"].astype(np.float32),
                connection_eigenvalue=bloom[
                    "connection_eigenvalue"].astype(np.float32),
                heat_kernel=bloom["heat_kernel"].astype(np.float32),
                transported_base_kernel=bloom[
                    "transported_base_kernel"].astype(np.float32),
                complete_kernel=complete.astype(np.float32),
            )
            image_report[arm_name] = {
                "summary": summarize_proposal_connection(connection),
                "heat_audit": _audit(
                    bloom["heat_kernel"], stages, landmarks[name]),
                "transported_base_audit": _audit(
                    bloom["transported_base_kernel"], stages,
                    landmarks[name]),
                "complete_audit": _audit(
                    complete, stages, landmarks[name]),
            }
        report["images"][name] = image_report
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
