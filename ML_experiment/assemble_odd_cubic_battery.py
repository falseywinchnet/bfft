#!/usr/bin/env python3
"""Join the preserved width-38 reference battery to the new odd-cubic runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("ML_experiment/results_continuous_frame_2x"))
    parser.add_argument("--odd", type=Path, default=Path("ML_experiment/results_odd_cubic_battery"))
    parser.add_argument("--out", type=Path, default=Path("ML_experiment/results_odd_cubic_full"))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    reference = json.loads((args.reference / "results.json").read_text())
    odd = json.loads((args.odd / "results.json").read_text())
    for key in ("tasks", "width", "seeds", "steps", "batch", "lr", "eval_every", "grid"):
        if reference["configuration"][key] != odd["configuration"][key]:
            raise ValueError(f"configuration mismatch for {key}")
    variants = list(reference["configuration"]["variants"]) + [odd["configuration"]["variant"]]
    configuration = {**reference["configuration"], "out": str(args.out), "variants": variants,
                     "reference_results": str(args.reference), "odd_results": str(args.odd)}
    (args.out / "results.json").write_text(json.dumps({
        "configuration": configuration, "runs": reference["runs"] + odd["runs"]}, indent=2))
    reference_probes = json.loads((args.reference / "probes.json").read_text())["probes"]
    odd_probes = json.loads((args.odd / "probes.json").read_text())["probes"]
    (args.out / "probes.json").write_text(json.dumps({
        "configuration": configuration, "probes": reference_probes + odd_probes}))
    print(json.dumps({"runs": len(reference["runs"]) + len(odd["runs"]),
                      "probes": len(reference_probes) + len(odd_probes), "variants": variants}))


if __name__ == "__main__": main()

