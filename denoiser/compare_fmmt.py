"""Matched comparison of FMMT support-birth laws on one 2-D observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

try:
    from .fmmt_certified import denoise_fmmt
    from .probes import edge_retention, mse
    from .transport_support import denoise_2d_fmmt
except ImportError:
    from fmmt_certified import denoise_fmmt
    from probes import edge_retention, mse
    from transport_support import denoise_2d_fmmt


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def save_gray(path: Path, value: np.ndarray) -> None:
    Image.fromarray(np.uint8(np.round(np.clip(value, 0.0, 1.0) * 255.0))).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    observed = load_gray(args.input)
    truth = load_gray(args.truth) if args.truth else None
    methods = {
        "plain": lambda: denoise_fmmt(observed, certify_support=False),
        "integrated_checkpoint": lambda: denoise_fmmt(observed),
        "continuous_support": lambda: denoise_2d_fmmt(observed),
    }
    report = {
        "scope": "matched FMMT support-birth comparison; no unrelated operator is scored",
        "input": str(args.input),
        "truth": str(args.truth) if args.truth else None,
        "methods": {},
    }
    for name, run in methods.items():
        start = time.perf_counter()
        output, diagnostics = run()
        seconds = time.perf_counter() - start
        record = {"seconds": seconds, "diagnostics": diagnostics}
        if truth is not None:
            record.update({
                "mse": mse(output, truth),
                "edge_retention": edge_retention(output, truth),
            })
        report["methods"][name] = record
        save_gray(args.out / f"{name}.png", output)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

