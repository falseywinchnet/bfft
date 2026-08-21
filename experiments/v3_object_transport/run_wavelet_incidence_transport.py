#!/usr/bin/env python3
"""Lift wavelet-leader content transitions into directed V3 incidence state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.connection_bloom import (
    bloom_region_embedding,
    fit_joint_whitener,
    relation_features,
)
from experiments.v3_object_transport.participation_algebra import (
    complete_kernel_algebra,
    normalized_linear_kernel,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
    _load_bundle,
    _load_complex,
)
from experiments.v3_object_transport.run_contour_transport import _audit


def _seed(name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def _normalize(kernel: np.ndarray) -> np.ndarray:
    value = 0.5 * (kernel + kernel.T)
    diagonal = np.maximum(np.diag(value), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    return np.divide(
        value, denominator, out=np.zeros_like(value),
        where=denominator > 1e-30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--controls", nargs="+", default=CONTROLS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "wavelet_incidence_transport")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]

    controls = {}
    base_relation = {}
    schema = None
    for name in args.controls:
        image_dir = args.results / name
        complex_ = _load_complex(image_dir / "compound_region_complex.npz")
        bundle = _load_bundle(image_dir / "compound_incidence_bundle.npz")
        values, names = relation_features(complex_, bundle)
        if schema is None:
            schema = names
        elif names != schema:
            raise RuntimeError("control incidence schemas disagree")
        base_relation[name] = values
        raw = np.load(
            args.results / "wavelet_leader_transport" / name
            / "wavelet_leader_transport.npz")["region_embedding"].astype(
                np.float64)
        law = np.load(
            args.results / "wavelet_leader_scale_law" / name
            / "wavelet_leader_transport.npz")["region_embedding"].astype(
                np.float64)
        controls[name] = {
            "complex": complex_, "bundle": bundle,
            "stages": np.load(image_dir / "v3_stages.npz"),
            "leader": np.column_stack((raw, law)),
            "contour": np.load(
                args.results / "contour_transport" / name
                / "contour_transport.npz")["region_kernel"].astype(np.float64),
            "enclosure": np.load(
                args.results / "relative_enclosure" / name
                / "relative_enclosure.npz")["region_kernel"].astype(np.float64),
            "proposal": np.load(
                args.results / "proposal_topology_transport" / name
                / "full.npz")["normalized_connection"].astype(np.float64),
        }
        proposal = 0.5 * (
            controls[name]["proposal"] + controls[name]["proposal"].T)
        eigenvalue, eigenvector = np.linalg.eigh(proposal)
        multiplier = np.exp(0.5 * (
            np.clip(eigenvalue, -1.0, 1.0) - 1.0))
        controls[name]["proposal_half_heat"] = (
            eigenvector * multiplier[None, :]) @ eigenvector.T
    assert schema is not None

    arms = {}
    leader_width = next(iter(controls.values()))["leader"].shape[1]
    for arm_name, shuffled, representation in (
        ("transition_only", False, "transition"),
        ("ordered_endpoints", False, "ordered_endpoints"),
        ("shuffled_transition_only", True, "transition"),
        ("shuffled_ordered_endpoints", True, "ordered_endpoints"),
    ):
        raw_relation = {}
        for name in args.controls:
            control = controls[name]
            incidence = control["bundle"]["incidence"]
            region = np.asarray(incidence["region"], dtype=np.int32)
            outside = np.asarray(incidence["outside"], dtype=np.int32)
            leader = control["leader"]
            if shuffled:
                permutation = np.random.default_rng(_seed(name)).permutation(
                    len(leader))
                leader = leader[permutation]
            if representation == "transition":
                evidence = leader[outside] - leader[region]
            else:
                # The ordered endpoint fibre retains absolute material
                # regularity.  Its difference is already in the linear span,
                # so adding it would only introduce a null covariance mode.
                evidence = np.column_stack((
                    leader[region], leader[outside]))
            raw_relation[name] = np.column_stack((
                base_relation[name], evidence))
        if representation == "transition":
            leader_names = tuple(
                f"directed_wavelet_leader_transition_{index}"
                for index in range(leader_width))
        else:
            leader_names = tuple(
                f"directed_wavelet_leader_{endpoint}_{index}"
                for endpoint in ("inside", "outside")
                for index in range(leader_width))
        whitener = fit_joint_whitener(
            raw_relation.values(), schema + leader_names)
        arms[arm_name] = {
            "raw": raw_relation,
            "whitener": whitener,
            "representation": representation,
            "shuffled": shuffled,
        }

    report = {
        "purpose": (
            "raw plus scale-law wavelet-leader evidence inside the directed "
            "V3 incidence fibre; transition-only is compared with the "
            "complete ordered (inside, outside) endpoint state and matched "
            "region-correspondence shuffles; topology heat is analytical and "
            "landmarks are evaluation-only"
        ),
        "arms": {},
    }
    for arm_name, arm in arms.items():
        arm_report = {
            "retained_covariance_modes": int(arm["whitener"].basis.shape[1]),
            "leader_representation": arm["representation"],
            "shuffled_region_correspondence": arm["shuffled"],
            "images": {},
        }
        for name in args.controls:
            control = controls[name]
            whitened = arm["whitener"].transform(arm["raw"][name])
            embedding, topology = bloom_region_embedding(
                control["complex"], control["bundle"], whitened)
            role = normalized_linear_kernel(embedding)
            complete = complete_kernel_algebra({
                "role": role,
                "contour": control["contour"],
                "enclosure": control["enclosure"],
            })["complete"]
            half_heat = control["proposal_half_heat"]
            transported = _normalize(half_heat @ complete @ half_heat)
            image_output = output / name
            image_output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                image_output / f"{arm_name}.npz",
                region_embedding=embedding.astype(np.float32),
                role_kernel=role.astype(np.float32),
                complete_kernel=complete.astype(np.float32),
                proposal_transported_complete_kernel=transported.astype(
                    np.float32),
            )
            arm_report["images"][name] = {
                "topology": topology,
                "role_audit": _audit(
                    role, control["stages"], landmarks[name]),
                "complete_audit": _audit(
                    complete, control["stages"], landmarks[name]),
                "proposal_transported_complete_audit": _audit(
                    transported, control["stages"], landmarks[name]),
            }
        report["arms"][arm_name] = arm_report
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
