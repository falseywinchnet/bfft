#!/usr/bin/env python3
"""Read-only inspector for Fujifilm ODM ``LENGTH=`` / ``@DFI`` firmware.

The X-A5 and X-T100 do not use the usual Fujifilm X-Processor update format.
This tool deliberately performs no repacking and no device I/O.  It inventories
the outer records, embedded DFI images, printable strings, and likely ARM code
starts so that the format can be reversed without risking a camera.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


OUTER_HEADER_RE = re.compile(
    rb"LENGTH=(?P<length>\d+)\s+"
    rb"(?P<code>[A-Za-z0-9_-]+)\s+"
    rb"VER=(?P<version>[^\s\x00\r\n]+)"
    rb"(?P<tail>[^\x00\r\n]{0,180})"
)


@dataclass(frozen=True)
class OuterRecord:
    offset: int
    header_end: int
    length: int
    code: str
    version: str
    dvr: str | None
    checksum: int | None
    target_offset: int | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class DfiCandidate:
    offset: int
    version_le: int
    name: str
    data_offset: int
    unpacked_size: int
    load_address: int
    compression: int
    packed_size: int
    first_words_le: tuple[int, ...]
    arm_vector_score: int
    data_entropy: float


def _integer_after(label: bytes, tail: bytes) -> int | None:
    match = re.search(rb"(?:^|\s)" + re.escape(label) + rb"=(\d+)", tail)
    return int(match.group(1)) if match else None


def _text_after(label: bytes, tail: bytes) -> str | None:
    match = re.search(rb"(?:^|\s)" + re.escape(label) + rb"=([^\s]+)", tail)
    return match.group(1).decode("ascii", "replace") if match else None


def find_outer_records(data: bytes) -> list[OuterRecord]:
    records: list[OuterRecord] = []
    for match in OUTER_HEADER_RE.finditer(data):
        offset = match.start()
        # Real headers are padded to a fixed-size record with spaces/NULs.  This
        # rejects diagnostic text containing a syntactically valid LENGTH line.
        padding = data[match.end() : min(match.end() + 64, len(data))]
        if padding and sum(byte in (0, 10, 13, 32) for byte in padding) < (
            len(padding) * 3 // 4
        ):
            continue
        tail = match.group("tail")
        known_tokens = {
            b"DVR",
            b"SUM",
            b"OFFSET",
            b"IPL",
            b"PTBL",
            b"SUB1",
            b"ND1",
        }
        tags: list[str] = []
        for token in tail.split():
            key = token.split(b"=", 1)[0]
            if key in known_tokens and b"=" not in token:
                tags.append(token.decode("ascii", "replace"))
        records.append(
            OuterRecord(
                offset=offset,
                header_end=match.end(),
                length=int(match.group("length")),
                code=match.group("code").decode("ascii", "replace"),
                version=match.group("version").decode("ascii", "replace"),
                dvr=_text_after(b"DVR", tail),
                checksum=_integer_after(b"SUM", tail),
                target_offset=_integer_after(b"OFFSET", tail),
                tags=tuple(tags),
            )
        )
    return records


def _entropy(blob: bytes) -> float:
    if not blob:
        return 0.0
    counts = [0] * 256
    for byte in blob:
        counts[byte] += 1
    total = len(blob)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts
        if count
    )


def _arm_vector_score(blob: bytes) -> int:
    """Score common ARM32 vector-table branch/load instructions."""
    if len(blob) < 32:
        return 0
    words = struct.unpack_from("<8I", blob)
    score = 0
    for word in words:
        opcode = word & 0xFF000000
        if opcode in (0xEA000000, 0xEB000000):
            score += 1
        # LDR pc, [pc, #imm] and close relatives commonly used in vectors.
        if word & 0xFFFFF000 in (0xE59FF000, 0xE51FF000):
            score += 1
    return score


def find_dfi_candidates(data: bytes) -> list[DfiCandidate]:
    results: list[DfiCandidate] = []
    cursor = 0
    while True:
        offset = data.find(b"@DFI", cursor)
        if offset < 0:
            break
        cursor = offset + 4
        if offset + 0x204 > len(data):
            continue
        version = struct.unpack_from("<I", data, offset + 4)[0]
        # Observed LEN images use a 0x200-byte DFI header.  Keep every candidate
        # in the inventory, but score its prospective code start independently.
        data_offset = offset + 0x200
        sample = data[data_offset : data_offset + 0x1000]
        words = struct.unpack_from("<16I", data, offset + 4)
        raw_name = data[offset + 0x10 : offset + 0x20].split(b"\0", 1)[0]
        results.append(
            DfiCandidate(
                offset=offset,
                version_le=version,
                name=raw_name.decode("ascii", "replace"),
                data_offset=data_offset,
                unpacked_size=struct.unpack_from("<I", data, offset + 0x20)[0],
                load_address=struct.unpack_from("<I", data, offset + 0x28)[0],
                compression=struct.unpack_from("<I", data, offset + 0x30)[0],
                packed_size=struct.unpack_from("<I", data, offset + 0x34)[0],
                first_words_le=tuple(words),
                arm_vector_score=_arm_vector_score(sample),
                data_entropy=round(_entropy(sample), 4),
            )
        )
    return results


def decompress_lzss(
    packed: bytes, expected_size: int | None = None
) -> bytes:
    """Decode the 4 KiB-window LZSS stream used by LEN DFI images.

    Flags are consumed least-significant bit first. A set bit is a literal;
    a clear bit introduces a 12-bit ring offset and 4-bit length-minus-three.
    The ring starts at 0xfee and is initialized with spaces, matching the
    classic Okumura LZSS layout used by this firmware family.
    """
    ring = bytearray(b" " * 4096)
    ring_cursor = 0xFEE
    output = bytearray()
    cursor = 0
    while cursor < len(packed):
        flags = packed[cursor]
        cursor += 1
        for bit in range(8):
            if expected_size is not None and len(output) >= expected_size:
                return bytes(output[:expected_size])
            if cursor >= len(packed):
                return bytes(output)
            if flags & (1 << bit):
                value = packed[cursor]
                cursor += 1
                output.append(value)
                ring[ring_cursor] = value
                ring_cursor = (ring_cursor + 1) & 0xFFF
                continue
            if cursor + 1 >= len(packed):
                return bytes(output)
            low = packed[cursor]
            high_length = packed[cursor + 1]
            cursor += 2
            source = low | ((high_length & 0xF0) << 4)
            length = (high_length & 0x0F) + 3
            for index in range(length):
                value = ring[(source + index) & 0xFFF]
                output.append(value)
                ring[ring_cursor] = value
                ring_cursor = (ring_cursor + 1) & 0xFFF
                if expected_size is not None and len(output) >= expected_size:
                    return bytes(output[:expected_size])
    return bytes(output)


def extract_dfi_images(
    data: bytes, candidates: list[DfiCandidate], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, candidate in enumerate(candidates):
        if candidate.version_le != 1:
            continue
        if not (0 < candidate.unpacked_size <= 512 * 1024 * 1024):
            continue
        if not (0 < candidate.packed_size <= len(data) - candidate.data_offset):
            continue
        source = data[
            candidate.data_offset : candidate.data_offset + candidate.packed_size
        ]
        if candidate.compression == 1:
            payload = decompress_lzss(source, candidate.unpacked_size)
        elif candidate.compression == 0:
            payload = source[: candidate.unpacked_size]
        else:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate.name) or "anon"
        path = output_dir / (
            f"{index:02d}_{candidate.offset:08x}_{safe_name}_"
            f"{candidate.load_address:08x}.bin"
        )
        path.write_bytes(payload)
        written.append(path)
    return written


def printable_strings(
    data: bytes, minimum: int = 6
) -> Iterable[tuple[int, str]]:
    pattern = re.compile(rb"[\x20-\x7e]{" + str(minimum).encode() + rb",}")
    for match in pattern.finditer(data):
        yield match.start(), match.group().decode("ascii", "replace")


def interesting_strings(
    data: bytes, minimum: int = 5
) -> list[dict[str, int | str]]:
    terms = re.compile(
        r"(?i)(adjust|autorun|calib|card|debug|factory|jig|script|service|usb"
        r"|\.ash\b|\.bin\b|\.dat\b|\.scp\b|\.scr\b)"
    )
    return [
        {"offset": offset, "text": text}
        for offset, text in printable_strings(data, minimum)
        if terms.search(text)
    ]


def inspect(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "file": str(path.resolve()),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "outer_records": [asdict(item) for item in find_outer_records(data)],
        "dfi_candidates": [asdict(item) for item in find_dfi_candidates(data)],
        "interesting_strings": interesting_strings(data),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("firmware", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of a compact report",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        help="extract structurally valid DFI payloads into this directory",
    )
    args = parser.parse_args()
    firmware_data = args.firmware.read_bytes()
    candidates = find_dfi_candidates(firmware_data)
    report = inspect(args.firmware)
    extracted: list[Path] = []
    if args.extract_dir:
        extracted = extract_dfi_images(
            firmware_data, candidates, args.extract_dir
        )
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"{report['file']}")
    print(f"size={report['size']} sha256={report['sha256']}")
    print("\nOuter LENGTH records:")
    for item in report["outer_records"]:
        print(
            f"  {item['offset']:#010x} len={item['length']:#x} "
            f"{item['code']} {item['version']} dvr={item['dvr']} "
            f"target={item['target_offset']}"
        )
    print("\nDFI candidates:")
    for item in report["dfi_candidates"]:
        print(
            f"  {item['offset']:#010x} version={item['version_le']} "
            f"name={item['name']!r} unpacked={item['unpacked_size']:#x} "
            f"packed={item['packed_size']:#x} compression={item['compression']} "
            f"load={item['load_address']:#010x} "
            f"data={item['data_offset']:#010x} "
            f"arm-score={item['arm_vector_score']} "
            f"entropy={item['data_entropy']}"
        )
    print("\nInteresting printable strings:")
    for item in report["interesting_strings"]:
        print(f"  {item['offset']:#010x} {item['text']}")
    if extracted:
        print("\nExtracted DFI images:")
        for path in extracted:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
