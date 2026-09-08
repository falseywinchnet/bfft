#!/usr/bin/env python3
"""Install one measured glyph-residual page candidate into a PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pikepdf
from pikepdf import Dictionary, Name, ObjectStreamMode

from pdf_optimizer.mrc import _mrc_pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise SystemExit("source and output must differ")
    output.parent.mkdir(parents=True, exist_ok=True)

    mask_data = (args.study / "candidate-mask.jb2").read_bytes()
    foreground_data = (args.study / "candidate-foreground.jpx").read_bytes()
    globals_path = args.study / "candidate-mask-globals.jb2"
    globals_data = globals_path.read_bytes() if globals_path.exists() else b""
    study_report = json.loads((args.study / "report.json").read_text())

    with pikepdf.open(source) as pdf:
        if not 1 <= args.page <= len(pdf.pages):
            raise SystemExit("page is outside the document")
        pair = _mrc_pair(pdf.pages[args.page - 1])
        if pair is None:
            raise SystemExit("page is not a simple MRC page")
        _background, foreground, mask = pair
        mask.write(mask_data, filter=Name.JBIG2Decode)
        if globals_data:
            globals_stream = pdf.make_stream(globals_data)
            mask.DecodeParms = Dictionary(JBIG2Globals=globals_stream)
        elif "/DecodeParms" in mask:
            del mask["/DecodeParms"]
        for key in ("/Decode", "/Intent"):
            if key in mask:
                del mask[key]

        foreground.write(foreground_data, filter=Name.JPXDecode)
        foreground.ColorSpace = Name.DeviceRGB
        foreground.BitsPerComponent = 8
        foreground.Width = int(study_report["source"]["foreground_size"][0])
        foreground.Height = int(study_report["source"]["foreground_size"][1])
        foreground.Interpolate = True
        for key in ("/DecodeParms", "/Decode", "/Intent"):
            if key in foreground:
                del foreground[key]

        pdf.save(
            output,
            compress_streams=True,
            recompress_flate=False,
            object_stream_mode=ObjectStreamMode.preserve,
            linearize=False,
            preserve_pdfa=True,
            encryption=True if pdf.is_encrypted else None,
        )

    with pikepdf.open(output) as checked:
        problems = checked.check_pdf_syntax()
        if problems:
            raise RuntimeError("PDF syntax check failed: " + "; ".join(problems))
        if len(checked.pages) != study_report["source"]["page_count"]:
            raise RuntimeError("page count changed")

    payload = {
        "source": str(source),
        "output": str(output),
        "page": args.page,
        "source_pdf_bytes": source.stat().st_size,
        "output_pdf_bytes": output.stat().st_size,
        "pdf_delta_bytes": output.stat().st_size - source.stat().st_size,
        "study_pair_delta_bytes": study_report["pair_delta_bytes"],
        "selected_mask": study_report["mask_candidates"]["selected"],
        "selected_foreground": study_report["selected_foreground"],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
