"""CLI for the synthetic codec ground-truth benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate_suite
from .generate import generate_suite
from .run_ours import run_ours


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codec-ground-truth", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="create references and TinyPNG upload inputs")
    generate.add_argument("root", type=Path)
    generate.add_argument("--size", type=int, default=384)
    generate.add_argument("--supersample", type=int, default=4)

    evaluate = commands.add_parser("evaluate", help="score candidate directories against ground truth")
    evaluate.add_argument("root", type=Path)
    evaluate.add_argument(
        "--candidate", action="append", default=[], metavar="LABEL=DIR",
        help="repeatable directory containing png__CASE.png and jpeg__CASE.jpg",
    )
    evaluate.add_argument("--include-upload", action="store_true")
    evaluate.add_argument("--include-controls", action="store_true")

    ours = commands.add_parser("run-ours", help="match byte sizes returned by TinyPNG")
    ours.add_argument("root", type=Path)
    ours.add_argument("--codec", choices=("both", "png", "jpeg"), default="both")
    ours.add_argument(
        "--output-dir", type=Path,
        help="write outside a mirrored checkout so later build syncs cannot remove results",
    )
    ours.add_argument(
        "--case", action="append", dest="cases",
        help="run only the named case; repeat for multiple cases",
    )
    return parser


def _candidate_map(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"candidate must be LABEL=DIR: {value}")
        label, path = value.split("=", 1)
        result[label] = Path(path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        result = generate_suite(args.root, size=args.size, supersample=args.supersample)
        print(json.dumps({"root": str(args.root), "cases": len(result["cases"]), "upload_files": 2*len(result["cases"])}, indent=2))
        return 0
    if args.command == "run-ours":
        codecs = ("png", "jpeg") if args.codec == "both" else (args.codec,)
        result = run_ours(
            args.root, codecs=codecs, output_dir=args.output_dir,
            case_names=set(args.cases) if args.cases else None,
            progress=lambda value: print(value, flush=True),
        )
        print(json.dumps({"root": str(args.root), "runs": len(result)}, indent=2))
        return 0
    candidates = _candidate_map(args.candidate)
    if args.include_upload:
        candidates["crude-upload"] = args.root/"upload"
    if args.include_controls:
        candidates.update({
            "control-blur": args.root/"controls"/"blur",
            "control-banded": args.root/"controls"/"banded",
            "control-halo": args.root/"controls"/"halo",
        })
    if not candidates:
        candidates = {
            "tinypng": args.root/"candidates"/"tinypng",
            "ours": args.root/"candidates"/"ours",
        }
    report = evaluate_suite(args.root, candidates)
    print(json.dumps(report["summary"], indent=2))
    return 0
