"""Command-line entry point for the denoiser evolution experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .fmmt_certified import denoise_fmmt
from .probes import run_probes
from .transport_support import TransportResolution, denoise_2d_fmmt


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(image, 0.0, 1.0) * 255.0))).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    denoise = commands.add_parser("denoise", help="run a 2-D denoising form")
    denoise.add_argument("input", type=Path)
    denoise.add_argument("output", type=Path)
    denoise.add_argument(
        "--method",
        choices=("transport", "integrated", "plain"),
        default="transport",
    )
    denoise.add_argument("--diagnostics", type=Path)
    probe = commands.add_parser("probes", help="run the 1-D/2-D analytic probes")
    probe.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    commands.add_parser("gui", help="open the Dear PyGui laboratory")
    args = parser.parse_args()

    if args.command == "gui":
        from .gui import main as gui_main
        gui_main()
        return
    if args.command == "probes":
        print(json.dumps(run_probes(args.out), indent=2))
        return

    image = _load(args.input)
    if args.method == "transport":
        output, diagnostics = denoise_2d_fmmt(image)
    elif args.method == "integrated":
        output, diagnostics = denoise_fmmt(image)
    else:
        output, diagnostics = denoise_fmmt(image, certify_support=False)
    _save(args.output, output)
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

