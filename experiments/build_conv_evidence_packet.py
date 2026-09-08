"""Assemble the reproducible CONV evidence packet from measured artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "output" / "pdf" / "conv_evidence_v2"
SUPPORT = ROOT / "output" / "support_geometry"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build() -> Path:
    PACKET.mkdir(parents=True, exist_ok=True)
    sources = {
        "evidence_insert": ROOT / "output" / "pdf" / "conv_synthetic_evaluation.tex",
        "synthetic_census": PACKET / "results.json",
        "convergence_figure": PACKET / "convergence.pdf",
        "topology_figure": PACKET / "topology.pdf",
        "technical_synthetics_figure": PACKET / "technical_synthetics.pdf",
        "native_timing": SUPPORT / "conv_evidence_native_timing.json",
        "eikonal_conformance_9x8": (
            SUPPORT / "eikonal_convstar_conformance_9x8.json"
        ),
        "eikonal_conformance_17x4": (
            SUPPORT / "eikonal_convstar_conformance_17x4.json"
        ),
        "eikonal_conformance_33x2": (
            SUPPORT / "eikonal_convstar_conformance_33x2.json"
        ),
        "green_admission_derivation": (
            ROOT / "experiments" / "GREEN_KERNEL_ADMISSION_ANALYSIS.md"
        ),
        "green_admission_reference": (
            SUPPORT / "green_variable_metric_reference.json"
        ),
    }
    copied: dict[str, Path] = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = PACKET / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied[name] = target

    manifest = {
        "packet": "CONV deterministic synthetic evidence, generation 2",
        "method_registry": {
            "CONV": "formal ordered-current operator with all-source Eikonal factor admission",
            "CONV*": "exact FIR CONV factors with the Q1 interpolant of the Eikonal admission's cardinal nodal trace",
            "Lanczos-3": "normalized endpoint-aligned radius-3 Lanczos point interpolation",
            "bilinear": "endpoint-aligned separable linear interpolation",
            "EASU": "FSR 1.0 EASU without RCAS; audit and native timing forms identified in the paper",
            "PCHIP": "separable SciPy PCHIP timing comparator",
            "Green admission": "cardinal weighted-harmonic continuation of the same nodal order coordinate",
        },
        "generators": {
            "synthetic_census": "python -m experiments.conv_synthetic_evidence --out output/pdf/conv_evidence_v2",
            "native_timing": "python -m experiments.benchmark_conv_evidence_native --repeats 21 --out output/support_geometry/conv_evidence_native_timing.json",
            "eikonal_conformance": "python -m experiments.compare_eikonal_convstar --source-side N --scale R --out PATH",
            "green_admission": "python -m experiments.compare_green_variable_metric --source-sides 9,17 --target-side 33 --out output/support_geometry/green_variable_metric_reference.json",
            "paper_composition": "python experiments/compose_conv_paper.py",
        },
        "artifacts": {
            name: {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in copied.items()
        },
    }
    target = PACKET / "evidence_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return target


if __name__ == "__main__":
    print(build())
