#!/usr/bin/env python3
"""Build and save the corrected full-recording Ostensibly trace lattice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.corrected_full_recording import (
    build_corrected_full_recording,
)
from experiments.ostensibly_frontend.full_recording import serializable_intervals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    def report_chunk(record: dict[str, object]) -> None:
        print(json.dumps({"completed_chunk": record}), flush=True)

    result = build_corrected_full_recording(args.wav, progress=report_chunk)
    record = {
        "wav": str(args.wav),
        "sample_rate": result["sample_rate"],
        "diagnostics": result["diagnostics"],
        "chunks": result["chunks"],
        "speech": serializable_intervals(result["speech"]),
        "phones": serializable_intervals(result["phones"]),
    }
    np.savez_compressed(
        args.out / "corrected_full_recording.npz",
        reference=result["reference"],
        n2048_registered_maximum=result["n2048_registered_maximum"],
        registered_maximum=result["registered_maximum"],
        reassigned_support_power=result["reassigned_support_power"],
        baseline_cartoon=result["baseline_cartoon"],
        baseline_texture=result["baseline_texture"],
        trace_field=result["trace_field"],
        activity_score=result["activity_score"],
        active=result["active"],
    )
    (args.out / "corrected_full_recording.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
