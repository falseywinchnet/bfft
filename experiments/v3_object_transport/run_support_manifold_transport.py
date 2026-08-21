#!/usr/bin/env python3
"""Audit scale-commensurate centered support/manifold assembly."""

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
    summarize_support_manifold_transport,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "support_manifold_transport")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "all covariance-centered, scale-commensurate relations between "
            "bounded manifolds and possible supports; landmarks are evaluation-only"
        ),
        "images": {},
    }
    arms = {
        "full": (True, True),
        "centeredness_only": (True, False),
        "scale_only": (False, True),
    }
    for name in CONTROLS:
        complex_ = _load_complex(
            args.results / name / "compound_region_complex.npz")
        enclosure = build_relative_enclosures(complex_)
        stages = {"compound_labels": complex_["labels"]}
        base = np.load(
            args.results / "participation_algebra" / name
            / "participation_algebra.npz")["complete"].astype(np.float64)
        role = normalized_linear_kernel(np.load(
            args.results / "connection_bloom" / name / "full.npz"
        )["region_embedding"])
        contour = np.load(
            args.results / "contour_transport" / name
            / "contour_transport.npz")["region_kernel"].astype(np.float64)
        enclosure_kernel = enclosure["region_kernel"]
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        image_report = {}
        arm_inputs = {
            arm_name: (enclosure, centeredness, scale)
            for arm_name, (centeredness, scale) in arms.items()
        }
        seed = int.from_bytes(
            hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
        permutation = np.random.default_rng(seed).permutation(
            int(complex_["region_count"]))
        shuffled = dict(enclosure)
        shuffled["participation"] = enclosure["participation"][:, permutation]
        arm_inputs["shuffled_manifold_members"] = (shuffled, True, True)
        for arm_name, (
            arm_enclosure, centeredness, scale,
        ) in arm_inputs.items():
            transport = build_support_manifold_transport(
                complex_, arm_enclosure,
                include_centeredness=centeredness,
                include_scale=scale,
            )
            complete = complete_kernel_algebra({
                "base": base,
                "assembly": transport["region_kernel"],
            })["complete"]
            flat_complete = complete_kernel_algebra({
                "role": role,
                "contour": contour,
                "enclosure": enclosure_kernel,
                "assembly": transport["region_kernel"],
            })["complete"]
            np.savez_compressed(
                image_output / f"{arm_name}.npz",
                **{
                    key: np.asarray(value, dtype=np.float32)
                    for key, value in transport.items()
                },
                complete_kernel=complete.astype(np.float32),
                flat_complete_kernel=flat_complete.astype(np.float32),
            )
            image_report[arm_name] = {
                "summary": summarize_support_manifold_transport(transport),
                "assembly_audit": _audit(
                    transport["region_kernel"], stages, landmarks[name]),
                "complete_audit": _audit(
                    complete, stages, landmarks[name]),
                "flat_complete_audit": _audit(
                    flat_complete, stages, landmarks[name]),
            }
        report["images"][name] = image_report
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
