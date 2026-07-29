#!/usr/bin/env python3
"""Build the X-A5 C713A read-only adjustment-backup service card files.

This builder is intentionally incapable of emitting the adjacent adjustment
write event. The only accepted dispatcher pair is opcode 326, parameter 1100.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

MODEL = "C713A"
ACTIVATION_MARKER = f"{MODEL}.ADJ"
SAFE_OPCODE = 326
SAFE_PARAMETER = 1100
WRITE_PARAMETER = 1101


def service_packet(opcode: int = SAFE_OPCODE, parameter: int = SAFE_PARAMETER) -> bytes:
    if (opcode, parameter) != (SAFE_OPCODE, SAFE_PARAMETER):
        raise ValueError(
            f"refusing unsafe dispatcher pair ({opcode}, {parameter}); "
            f"only ({SAFE_OPCODE}, {SAFE_PARAMETER}) is allowed"
        )
    if parameter == WRITE_PARAMETER:
        raise ValueError("refusing WRITE ADJUST DATA event 1101")
    high, low = divmod(opcode, 256)
    if high > 2:
        raise ValueError("opcode cannot be represented by the C..E service packet")
    # ESC, command bank C..E, unused/length word, type 0, opcode low byte,
    # then the 32-bit little-endian dispatcher parameter.
    return bytes((0x1B, ord("C") + high, 0, 0, 0, low)) + parameter.to_bytes(
        4, "little"
    )


def script_dat() -> bytes:
    packet_hex = service_packet().hex().upper()
    return f"#P000-S000\r\n0000:{packet_hex}\r\n".encode("ascii")


def build(destination: Path, identity: str, timestamp: dt.datetime) -> None:
    identity_bytes = identity.encode("ascii")
    if len(identity_bytes) != 12:
        raise ValueError("INPUT identity must be exactly 12 ASCII bytes")
    adjustment_dir = destination / "ADJ"
    model_dir = adjustment_dir / MODEL
    model_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the camera's ordinary writable media layout so a missing DCIM
    # tree does not mask service-card behavior with a capture-path warning.
    (destination / "DCIM" / "100_FUJI").mkdir(parents=True, exist_ok=True)
    # c01cdc04 constructs the packed FAT key C713A + ADJ and scans the
    # active 16-byte directory cache for it before enabling adjustment mode.
    # Root-only placement did not activate on camera. Keep both plausible
    # placements while resolving which directory supplies that cache. The
    # marker's contents are never consulted by the gate.
    (destination / ACTIVATION_MARKER).write_bytes(b"")
    (adjustment_dir / ACTIVATION_MARKER).write_bytes(b"")
    files = {
        "INPUT.DAT": identity_bytes,
        "CARDVER.DAT": MODEL.encode("ascii"),
        "DATE.DAT": timestamp.strftime("%Y/%m/%d %H:%M:%S").encode("ascii"),
        "SCRIPT.DAT": script_dat(),
    }
    for name, content in files.items():
        (model_dir / name).write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--identity",
        default="FF129301X-A5",
        help="12-byte identity recovered from this camera's RAF header",
    )
    parser.add_argument(
        "--timestamp",
        type=dt.datetime.fromisoformat,
        default=dt.datetime.now().replace(microsecond=0),
    )
    args = parser.parse_args()
    build(args.destination, args.identity, args.timestamp)


if __name__ == "__main__":
    main()
