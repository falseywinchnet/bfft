#!/usr/bin/env python3
"""Compile a label-blind generic pronunciation search catalog."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .generic_pronunciation_catalog import compile_generic_pronunciation_catalog
from .generic_word_template_bank import load_generic_word_template_bank
from .word_lattice import ARPABET_39, parse_cmudict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("generic_word_bank", type=Path)
    parser.add_argument("--lengths", default="3")
    parser.add_argument("--realizations", type=int, default=4)
    parser.add_argument("--time-bins", type=int, default=16)
    parser.add_argument("--row-quantiles", type=int, default=8)
    parser.add_argument(
        "--context-policy",
        choices=("unconditional", "maximum", "mixed"),
        default="unconditional",
    )
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lengths = frozenset(int(value) for value in args.lengths.split(","))
    if not lengths or min(lengths) < 1:
        raise ValueError("catalog lengths must be positive")
    pronunciations = parse_cmudict(
        args.cmudict.read_text().splitlines(),
        frozenset(ARPABET_39),
        max_phones=max(lengths),
    )
    phones = {
        item.phones for item in pronunciations if len(item.phones) in lengths
    }
    bank = load_generic_word_template_bank(args.generic_word_bank)
    started = perf_counter()
    catalog = compile_generic_pronunciation_catalog(
        phones,
        bank,
        realization_count=args.realizations,
        time_bins=args.time_bins,
        row_quantiles=args.row_quantiles,
        context_policy=args.context_policy,
        storage_dtype=args.storage_dtype,
    )
    catalog.save(args.out)
    print(
        {
            "output": str(args.out),
            "pronunciation_count": int(catalog.phone_keys.size),
            "shape": tuple(int(value) for value in catalog.surfaces.shape),
            "seconds": perf_counter() - started,
        }
    )


if __name__ == "__main__":
    main()
