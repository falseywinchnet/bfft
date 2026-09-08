#!/usr/bin/env python3
"""Train JBIG2 GBAT coordinates on row-wise BWT data, then encode original masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from pdf_optimizer.bwt_jbig2 import (
    DEFAULT_GBAT,
    load_pbm_ink,
    rowwise_byte_bwt,
    rank_gbat_coordinates,
    shannon_entropy,
    train_gbat_tables,
    zero_order_huffman_cost,
)


def _encode(encoder: Path, bitmap: Path, coordinates, output: Path) -> None:
    value = ";".join(f"{x},{y}" for x, y in coordinates)
    with output.open("wb") as stream:
        subprocess.run(
            [str(encoder), "-p", "--generic-template", "0", "--generic-at", value, str(bitmap)],
            check=True,
            stdout=stream,
            stderr=subprocess.PIPE,
        )


def _decode_equal(decoder: Path, stream: Path, source: Path, output: Path) -> bool:
    subprocess.run(
        [str(decoder), "-q", "-e", "-t", "pbm", "-o", str(output), str(stream)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with Image.open(source) as left, Image.open(output) as right:
        return left.size == right.size and np.array_equal(np.asarray(left), np.asarray(right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitmap-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--pages", default="10,30,100,180,410,468")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--sample-rows", type=int, default=96)
    parser.add_argument("--terminal-shortlist", type=int, default=24)
    parser.add_argument("--terminal-passes", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pages = [int(value) for value in args.pages.split(",") if value]
    result = {
        "method": "row-wise packed-byte cyclic BWT trains JBIG2 template-0 GBAT; original bitmap is encoded",
        "pages": [],
    }
    started = time.monotonic()
    for page in pages:
        page_started = time.monotonic()
        bitmap = args.bitmap_dir / f"{page:04d}.pbm"
        ink = load_pbm_ink(bitmap)
        packed = np.packbits(ink, axis=1, bitorder="big").tobytes()
        transformed = rowwise_byte_bwt(ink)
        transformed_packed = np.packbits(transformed, axis=1, bitorder="big").tobytes()
        bwt_tables = train_gbat_tables(
            transformed, sample_rows=args.sample_rows, beam_width=args.beam_width
        )
        original_tables = train_gbat_tables(
            ink, sample_rows=args.sample_rows, beam_width=args.beam_width
        )
        bwt_ranked = rank_gbat_coordinates(transformed, sample_rows=args.sample_rows)

        candidates = [("default", DEFAULT_GBAT, None)]
        candidates += [
            (f"bwt_{index}", proposal.coordinates, proposal.conditional_bits_per_pixel)
            for index, proposal in enumerate(bwt_tables)
        ]
        candidates += [
            (f"original_{index}", proposal.coordinates, proposal.conditional_bits_per_pixel)
            for index, proposal in enumerate(original_tables)
        ]
        measurements = []
        with tempfile.TemporaryDirectory(prefix=f"jbig2-bwt-{page:04d}-") as directory:
            work = Path(directory)
            encoded_cache = {}
            def measure(name, coordinates, training_entropy=None):
                key = tuple(tuple(pair) for pair in coordinates)
                encoded = work / f"candidate-{len(encoded_cache):04d}.jb2"
                if key not in encoded_cache:
                    _encode(args.encoder, bitmap, key, encoded)
                    encoded_cache[key] = (encoded.stat().st_size, encoded)
                size, path = encoded_cache[key]
                return {
                    "name": name,
                    "coordinates": key,
                    "training_conditional_bits_per_pixel": training_entropy,
                    "bytes": size,
                    "pixel_exact": None,
                    "_path": path,
                }

            for name, coordinates, training_entropy in candidates:
                measurements.append(measure(name, coordinates, training_entropy))

            # The BWT entropy objective ignores finite-context adaptation cost.
            # Use it as a shortlist, but let actual original-order JBIG2 bytes
            # decide each coordinate-descent mutation.
            shortlist = [coordinate for coordinate, _ in bwt_ranked[:args.terminal_shortlist]]
            current = tuple(DEFAULT_GBAT)
            current_size = measurements[0]["bytes"]
            terminal_trials = 0
            for pass_index in range(args.terminal_passes):
                pass_candidates = []
                for slot in range(4):
                    for coordinate in shortlist:
                        if coordinate in current and coordinate != current[slot]:
                            continue
                        proposal = list(current)
                        proposal[slot] = coordinate
                        measured = measure(
                            f"terminal_pass_{pass_index}_slot_{slot}", tuple(proposal)
                        )
                        terminal_trials += 1
                        pass_candidates.append(measured)
                winner = min(pass_candidates, key=lambda item: (item["bytes"], item["coordinates"]))
                if winner["bytes"] >= current_size:
                    break
                current = winner["coordinates"]
                current_size = winner["bytes"]
            terminal = measure("bwt_shortlist_terminal", current)
            terminal["terminal_trials"] = terminal_trials
            measurements.append(terminal)

            # Decode every reported proposal, and especially the terminal
            # winner; discarded search mutations are size-only measurements.
            for index, measurement in enumerate(measurements):
                decoded = work / f"reported-{index:02d}.pbm"
                measurement["pixel_exact"] = _decode_equal(
                    args.decoder, measurement["_path"], bitmap, decoded
                )
                del measurement["_path"]
        default_bytes = measurements[0]["bytes"]
        best = min(
            (measurement for measurement in measurements if measurement["pixel_exact"]),
            key=lambda measurement: (measurement["bytes"], measurement["name"]),
        )
        page_result = {
            "page": page,
            "width": int(ink.shape[1]),
            "height": int(ink.shape[0]),
            "zero_order": {
                "source_entropy_bits_per_byte": shannon_entropy(packed),
                "bwt_entropy_bits_per_byte": shannon_entropy(transformed_packed),
                "source_huffman_bits": zero_order_huffman_cost(packed),
                "bwt_huffman_bits": zero_order_huffman_cost(transformed_packed),
            },
            "candidates": measurements,
            "best": best,
            "best_delta_from_default": best["bytes"] - default_bytes,
            "seconds": time.monotonic() - page_started,
        }
        result["pages"].append(page_result)
        print(json.dumps({"page": page, "best": best, "delta": page_result["best_delta_from_default"]}), flush=True)

    result["total_seconds"] = time.monotonic() - started
    result["default_total_bytes"] = sum(page["candidates"][0]["bytes"] for page in result["pages"])
    result["best_total_bytes"] = sum(page["best"]["bytes"] for page in result["pages"])
    result["best_total_delta"] = result["best_total_bytes"] - result["default_total_bytes"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
