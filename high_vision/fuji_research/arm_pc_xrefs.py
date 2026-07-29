#!/usr/bin/env python3
"""Find ARM-state ADR-style references to firmware addresses.

The X-A5 image uses many ``add/sub Rd, pc, #imm`` instructions instead of
literal-pool pointers. This small scanner decodes those data-processing
immediates and reports instructions whose computed address is at, or close
to, a requested target.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def ror32(value: int, amount: int) -> int:
    amount &= 31
    if amount == 0:
        return value & 0xFFFFFFFF
    return ((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF


def pc_relative_xrefs(
    image: bytes, base: int, target: int, tolerance: int = 0
) -> list[tuple[int, int, str, int]]:
    hits: list[tuple[int, int, str, int]] = []
    for offset in range(0, len(image) - 3, 4):
        word = int.from_bytes(image[offset : offset + 4], "little")
        # ARM data-processing immediate, Rn == pc, opcode ADD or SUB.
        if ((word >> 25) & 0b111) != 0b001 or ((word >> 16) & 0xF) != 0xF:
            continue
        opcode = (word >> 21) & 0xF
        if opcode not in (0b0010, 0b0100):
            continue
        immediate = ror32(word & 0xFF, 2 * ((word >> 8) & 0xF))
        address = base + offset
        pc = address + 8
        computed = (
            (pc + immediate) & 0xFFFFFFFF
            if opcode == 0b0100
            else (pc - immediate) & 0xFFFFFFFF
        )
        delta = computed - target
        if abs(delta) <= tolerance:
            operation = "add" if opcode == 0b0100 else "sub"
            register = (word >> 12) & 0xF
            hits.append((address, computed, operation, register))
    return hits


def literal_xrefs(
    image: bytes, base: int, target: int, tolerance: int = 0
) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for offset in range(0, len(image) - 3, 4):
        value = int.from_bytes(image[offset : offset + 4], "little")
        if abs(value - target) <= tolerance:
            hits.append((base + offset, value))
    return hits


def parse_int(value: str) -> int:
    return int(value, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("targets", nargs="+", type=parse_int)
    parser.add_argument("--base", type=parse_int, default=0xC0000000)
    parser.add_argument("--tolerance", type=parse_int, default=0)
    args = parser.parse_args()

    image = args.image.read_bytes()
    for target in args.targets:
        hits = pc_relative_xrefs(image, args.base, target, args.tolerance)
        print(f"{target:#010x}:")
        for address, computed, operation, register in hits:
            delta = computed - target
            print(
                f"  {address:#010x}: {operation} r{register}, pc -> "
                f"{computed:#010x} ({delta:+#x})"
            )
        if not hits:
            print("  (no ARM ADR references)")
        for address, value in literal_xrefs(
            image, args.base, target, args.tolerance
        ):
            delta = value - target
            print(f"  {address:#010x}: literal {value:#010x} ({delta:+#x})")


if __name__ == "__main__":
    main()
