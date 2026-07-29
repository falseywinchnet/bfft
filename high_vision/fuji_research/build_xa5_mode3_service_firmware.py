#!/usr/bin/env python3
"""Build and verify the minimal X-A5 mode-3 adjustment-card firmware patch.

This tool never modifies its input. It is deliberately tied to the known
Fujifilm X-A5 2.03 package and refuses any other source hash or layout.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from len_dfi_inspect import decompress_lzss


STOCK_SHA256 = "0d35e4104b98513154c6fe0f827ef771f7531712386a468612ea652f792baeff"
STOCK_SIZE = 22_030_584

OUTER_HEADER_OFFSET = 0x00000000
OUTER_HEADER_SIZE = 0x100
OUTER_PAYLOAD_OFFSET = 0x00000100
OUTER_PAYLOAD_SIZE = 7_721_472
STOCK_OUTER_SUM = 939_407_882
PATCHED_OUTER_SUM = 939_407_868

DFI_OFFSET = 0x100
DFI_HEADER_SIZE = 0x200
PACKED_OFFSET = DFI_OFFSET + DFI_HEADER_SIZE
PACKED_SIZE = 0x57E72E
UNPACKED_SIZE = 0x831000

# This LZSS literal supplies only the immediate byte at executable offset
# 0x1cdc20. Changing 0x11 to 0x03 converts:
#   cmpne r0, #17  ->  cmpne r0, #3
PACKAGE_PATCH_OFFSET = 0x15BEAA
EXECUTABLE_PATCH_OFFSET = 0x1CDC20
STOCK_LITERAL = 0x11
PATCHED_LITERAL = 0x03
STOCK_INSTRUCTION = bytes.fromhex("11005013")
PATCHED_INSTRUCTION = bytes.fromhex("03005013")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def outer_sum(data: bytes) -> int:
    payload = data[
        OUTER_PAYLOAD_OFFSET : OUTER_PAYLOAD_OFFSET + OUTER_PAYLOAD_SIZE
    ]
    return sum(payload) & 0xFFFFFFFF


def declared_outer_sum(data: bytes) -> int:
    header = data[
        OUTER_HEADER_OFFSET : OUTER_HEADER_OFFSET + OUTER_HEADER_SIZE
    ]
    match = re.search(rb"(?:^|\s)SUM=(\d+)(?:\s|\r|\n)", header)
    if match is None:
        raise ValueError("first outer SUM field is missing")
    return int(match.group(1))


def unpack_main(data: bytes) -> bytes:
    packed = data[PACKED_OFFSET : PACKED_OFFSET + PACKED_SIZE]
    if len(packed) != PACKED_SIZE:
        raise ValueError("main DFI packed stream is truncated")
    unpacked = decompress_lzss(packed, UNPACKED_SIZE)
    if len(unpacked) != UNPACKED_SIZE:
        raise ValueError(
            f"main DFI decompressed to {len(unpacked):#x}, "
            f"expected {UNPACKED_SIZE:#x}"
        )
    return unpacked


def verify_stock(stock: bytes) -> bytes:
    if len(stock) != STOCK_SIZE:
        raise ValueError(
            f"stock size is {len(stock)}, expected {STOCK_SIZE}"
        )
    actual_hash = sha256(stock)
    if actual_hash != STOCK_SHA256:
        raise ValueError(
            f"stock SHA-256 is {actual_hash}, expected {STOCK_SHA256}"
        )
    if stock[PACKAGE_PATCH_OFFSET] != STOCK_LITERAL:
        raise ValueError("compressed patch literal is not the expected 0x11")
    if declared_outer_sum(stock) != STOCK_OUTER_SUM:
        raise ValueError("stock outer SUM declaration is unexpected")
    if outer_sum(stock) != STOCK_OUTER_SUM:
        raise ValueError("stock outer payload does not match its SUM")
    executable = unpack_main(stock)
    actual_instruction = executable[
        EXECUTABLE_PATCH_OFFSET : EXECUTABLE_PATCH_OFFSET + 4
    ]
    if actual_instruction != STOCK_INSTRUCTION:
        raise ValueError(
            "stock executable does not contain the expected "
            "cmpne r0, #17 instruction"
        )
    return executable


def build_candidate(stock: bytes) -> bytes:
    candidate = bytearray(stock)
    candidate[PACKAGE_PATCH_OFFSET] = PATCHED_LITERAL

    header = bytes(candidate[:OUTER_HEADER_SIZE])
    old_field = f"SUM={STOCK_OUTER_SUM}".encode()
    new_field = f"SUM={PATCHED_OUTER_SUM}".encode()
    if len(old_field) != len(new_field):
        raise ValueError("replacement SUM changes header width")
    if header.count(old_field) != 1:
        raise ValueError("stock SUM field is not unique in the first header")
    candidate[:OUTER_HEADER_SIZE] = header.replace(
        old_field, new_field, 1
    )
    return bytes(candidate)


def verify_candidate(stock: bytes, stock_executable: bytes, candidate: bytes) -> None:
    if len(candidate) != len(stock):
        raise ValueError("candidate package size changed")
    if declared_outer_sum(candidate) != PATCHED_OUTER_SUM:
        raise ValueError("candidate outer SUM declaration is wrong")
    if outer_sum(candidate) != PATCHED_OUTER_SUM:
        raise ValueError("candidate outer payload does not match its SUM")

    package_differences = [
        index
        for index, (before, after) in enumerate(zip(stock, candidate))
        if before != after
    ]
    expected_header_offsets = {
        index
        for index, (before, after) in enumerate(
            zip(
                f"SUM={STOCK_OUTER_SUM}".encode(),
                f"SUM={PATCHED_OUTER_SUM}".encode(),
            )
        )
        if before != after
    }
    header_start = stock[:OUTER_HEADER_SIZE].index(
        f"SUM={STOCK_OUTER_SUM}".encode()
    )
    expected_package_differences = {
        PACKAGE_PATCH_OFFSET,
        *(header_start + index for index in expected_header_offsets),
    }
    if set(package_differences) != expected_package_differences:
        raise ValueError(
            "candidate has unexpected package differences: "
            f"{[hex(item) for item in package_differences]}"
        )

    candidate_executable = unpack_main(candidate)
    executable_differences = [
        index
        for index, (before, after) in enumerate(
            zip(stock_executable, candidate_executable)
        )
        if before != after
    ]
    if executable_differences != [EXECUTABLE_PATCH_OFFSET]:
        raise ValueError(
            "candidate has unexpected executable differences: "
            f"{[hex(item) for item in executable_differences]}"
        )
    actual_instruction = candidate_executable[
        EXECUTABLE_PATCH_OFFSET : EXECUTABLE_PATCH_OFFSET + 4
    ]
    if actual_instruction != PATCHED_INSTRUCTION:
        raise ValueError(
            "candidate executable does not contain cmpne r0, #3"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock", type=Path, help="stock FWUP0016.DAT")
    parser.add_argument("output", type=Path, help="new candidate path")
    args = parser.parse_args()

    stock_path = args.stock.resolve()
    output_path = args.output.resolve()
    if stock_path == output_path:
        raise ValueError("output must not overwrite the stock firmware")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    stock = stock_path.read_bytes()
    stock_executable = verify_stock(stock)
    candidate = build_candidate(stock)
    verify_candidate(stock, stock_executable, candidate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(candidate)

    # Verify the bytes that actually reached disk, not just the in-memory build.
    written = output_path.read_bytes()
    if written != candidate:
        raise OSError("candidate differs after writing")
    verify_candidate(stock, stock_executable, written)

    print(f"stock_sha256={sha256(stock)}")
    print(f"candidate={output_path}")
    print(f"candidate_sha256={sha256(written)}")
    print(f"package_size={len(written)}")
    print(
        f"package_patch={PACKAGE_PATCH_OFFSET:#x}:"
        f"{STOCK_LITERAL:#04x}->{PATCHED_LITERAL:#04x}"
    )
    print(
        f"executable_patch={EXECUTABLE_PATCH_OFFSET:#x}:"
        "cmpne_r0_17->cmpne_r0_3"
    )
    print(
        f"outer_sum={declared_outer_sum(written)} "
        f"verified={outer_sum(written)}"
    )
    print("executable_difference_count=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
