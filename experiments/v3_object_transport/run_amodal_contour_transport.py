#!/usr/bin/env python3
"""Audit ternary amodal continuation proposals across the frozen controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.amodal_contour_transport import (
    amodal_pair_residuals,
    build_amodal_transport,
    build_weighted_amodal_transport,
    extract_amodal_ports,
    fit_zero_whitener,
    summarize_amodal_transport,
)
from experiments.v3_object_transport.contour_transport import (
    build_contour_transport,
)
from experiments.v3_object_transport.depth_hodge import (
    build_depth_hodge,
    summarize_depth_hodge,
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


def _stable_seed(name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def _coordinate_kernels(results: Path, name: str) -> dict[str, np.ndarray]:
    embedding = np.load(
        results / "connection_bloom" / name / "full.npz"
    )["region_embedding"]
    return {
        "role": normalized_linear_kernel(embedding),
        "contour": np.load(
            results / "contour_transport" / name
            / "contour_transport.npz")["region_kernel"].astype(np.float64),
        "enclosure": np.load(
            results / "relative_enclosure" / name
            / "relative_enclosure.npz")["region_kernel"].astype(np.float64),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--candidate-mode",
        choices=("same_component", "contour_delaunay"),
        default="same_component",
    )
    args = parser.parse_args()
    default_name = (
        "amodal_contour_transport"
        if args.candidate_mode == "same_component"
        else "amodal_contour_transport_delaunay"
    )
    output = args.out or (args.results / default_name)
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]

    controls = {}
    schema = None
    for name in CONTROLS:
        image_dir = args.results / name
        complex_ = _load_complex(image_dir / "compound_region_complex.npz")
        bundle = _load_bundle(image_dir / "compound_incidence_bundle.npz")
        contour = build_contour_transport(complex_, bundle)
        depth_archive = np.load(
            args.results / "depth_contour_transport" / name
            / "depth_contour_transport.npz")
        depth = {key: depth_archive[key] for key in depth_archive.files}
        ports = extract_amodal_ports(
            complex_, contour, depth,
            focus_arc_margin_first=depth["arc_focus_match_margin_first"],
            focus_arc_reliability=depth["arc_focus_reliability"],
        )
        hodge = build_depth_hodge(ports, int(complex_["region_count"]))
        pair, residual, names = amodal_pair_residuals(
            ports, complex_["labels"],
            candidate_mode=args.candidate_mode,
            port_depth_agreement=hodge["port_agreement"])
        if schema is None:
            schema = names
        elif schema != names:
            raise RuntimeError("amodal residual schemas disagree")
        controls[name] = {
            "complex": complex_, "ports": ports, "pair": pair,
            "residual": residual,
            "hodge": hodge,
        }
    assert schema is not None
    geometry = np.asarray([
        name.startswith("geometry_") for name in schema], dtype=bool)
    arms = {
        "full": {
            name: control["residual"] for name, control in controls.items()
        },
        "geometry_only": {
            name: control["residual"][:, geometry]
            for name, control in controls.items()
        },
        "shuffled_content": {},
    }
    for name, control in controls.items():
        value = control["residual"].copy()
        permutation = np.random.default_rng(
            _stable_seed(name)).permutation(len(value))
        value[:, ~geometry] = value[permutation][:, ~geometry]
        arms["shuffled_content"][name] = value

    report = {
        "purpose": (
            "all smooth ternary T-port continuation proposals conditioned on "
            "an explicit occluder contour; landmarks are evaluation-only"
        ),
        "raw_residual_channels": len(schema),
        "candidate_mode": args.candidate_mode,
        "images": {},
        "arms": {},
    }
    for arm_name, values in arms.items():
        arm_schema = tuple(
            name for name, keep in zip(schema, geometry)
            if keep or arm_name != "geometry_only"
        )
        whitener = fit_zero_whitener(values.values(), arm_schema)
        report["arms"][arm_name] = {
            "channels": len(arm_schema),
            "retained_modes": int(whitener.basis.shape[1]),
        }
        for name, control in controls.items():
            transport = build_amodal_transport(
                control["pair"], values[name], whitener,
                int(control["complex"]["region_count"]),
            )
            coordinates = _coordinate_kernels(args.results, name)
            complete = complete_kernel_algebra({
                **coordinates,
                "amodal": transport["region_kernel"],
            })["complete"]
            stages = {"compound_labels": control["complex"]["labels"]}
            image_output = output / name
            image_output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                image_output / f"{arm_name}.npz",
                compatibility=transport["compatibility"],
                whitened_squared_residual=transport[
                    "whitened_squared_residual"],
                region_kernel=transport["region_kernel"].astype(np.float32),
                complete_kernel=complete.astype(np.float32),
            )
            image_report = report["images"].setdefault(name, {})
            image_report[arm_name] = {
                "summary": summarize_amodal_transport(
                    control["ports"], control["pair"], transport),
                "amodal_audit": _audit(
                    transport["region_kernel"], stages, landmarks[name]),
                "complete_audit": _audit(
                    complete, stages, landmarks[name]),
            }
            image_report["depth_hodge"] = summarize_depth_hodge(
                control["hodge"])

    # These are literal geometric coordinates, not learned compatibility
    # blends.  They remain separate from the covariance-calibrated arms.
    report["arms"]["oriented_crossing"] = {
        "channels": ["orientation_evidence", "cap_crossing_fraction"],
    }
    report["arms"]["hodge_oriented"] = {
        "channels": ["orientation_evidence", "positive_depth_agreement"],
    }
    for name, control in controls.items():
        pair = control["pair"]
        derived_weights = {
            "oriented_crossing": (
                pair["orientation_evidence"]
                * pair["cap_crossing_fraction"]
            ),
            "hodge_oriented": (
                pair["orientation_evidence"]
                * np.sqrt(
                    np.maximum(pair["depth_agreement_first"], 0.0)
                    * np.maximum(pair["depth_agreement_second"], 0.0)
                )
            ),
        }
        coordinates = _coordinate_kernels(args.results, name)
        stages = {"compound_labels": control["complex"]["labels"]}
        for arm_name, weight in derived_weights.items():
            transport = build_weighted_amodal_transport(
                pair, weight, int(control["complex"]["region_count"]))
            complete = complete_kernel_algebra({
                **coordinates, "amodal": transport["region_kernel"],
            })["complete"]
            np.savez_compressed(
                output / name / f"{arm_name}.npz",
                compatibility=weight,
                region_kernel=transport["region_kernel"].astype(np.float32),
                complete_kernel=complete.astype(np.float32),
            )
            report["images"][name][arm_name] = {
                "summary": summarize_amodal_transport(
                    control["ports"], pair, transport),
                "amodal_audit": _audit(
                    transport["region_kernel"], stages, landmarks[name]),
                "complete_audit": _audit(
                    complete, stages, landmarks[name]),
            }
    # Save the raw ternary chart once, independently of each calibrated arm.
    for name, control in controls.items():
        raw = {
            **{f"port_{key}": value for key, value in control["ports"].items()},
            **{f"pair_{key}": value for key, value in control["pair"].items()},
            "raw_residual": control["residual"],
            "raw_residual_names": np.asarray(schema),
            **{
                f"hodge_{key}": value
                for key, value in control["hodge"].items()
                if key != "explained_fraction"
            },
            "hodge_explained_fraction": np.asarray(
                control["hodge"]["explained_fraction"]),
        }
        np.savez_compressed(output / name / "amodal_ports.npz", **raw)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
