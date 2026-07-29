#!/usr/bin/env python3
"""Decode the four SIG0/SIG1 RAWOUT shadow registers.

The X-A5 firmware writer at 0xc03a43d0 transforms its compact three-word
geometry structure into the four words beginning at 0x302104d4 (SIG0) or the
corresponding SIG1 shadow.  This utility reverses the hardware-facing layout
without assigning names to fields whose semantics are not yet proven.
"""

from __future__ import annotations

import argparse


def integer(text: str) -> int:
    return int(text, 0)


def decode(words: list[int]) -> dict[str, int]:
    r0, r1, r2, r3 = (word & 0xFFFFFFFF for word in words)
    return {
        "pair0_lo14": r0 & 0x3FFF,
        "pair0_hi14": (r0 >> 16) & 0x3FFF,
        "pair1_lo14": r1 & 0x3FFF,
        "pair1_hi14": (r1 >> 16) & 0x3FFF,
        "pair2_lo13": r2 & 0x1FFF,
        "pair2_hi13": (r2 >> 16) & 0x1FFF,
        "enable": r3 & 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode four X-A5 RAWOUT register/shadow words."
    )
    parser.add_argument(
        "words",
        metavar="WORD",
        type=integer,
        nargs=4,
        help="four integer words, with 0x prefixes accepted",
    )
    args = parser.parse_args()

    for name, value in decode(args.words).items():
        print(f"{name:>12} = {value:5d} (0x{value:x})")


if __name__ == "__main__":
    main()
