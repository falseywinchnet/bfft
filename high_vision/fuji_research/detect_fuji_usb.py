#!/usr/bin/env python3
"""Read-only detector for Fujifilm USB enumeration modes on macOS."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable


FUJI_VENDOR = 0x04CB
KNOWN_MODES = {
    0xFF80: "conventional Fujifilm service/jig route",
    0x02D5: "ordinary X-A5 PTP identity (not evidence of service mode)",
}


def walk(items: Iterable[object]) -> Iterable[dict[str, object]]:
    for item in items:
        if not isinstance(item, dict):
            continue
        yield item
        children = item.get("_items")
        if isinstance(children, list):
            yield from walk(children)


def parse_hex_field(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    token = value.split(maxsplit=1)[0]
    try:
        return int(token, 16)
    except ValueError:
        return None


def main() -> int:
    result = subprocess.run(
        ["system_profiler", "SPUSBDataType", "-json"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    devices = report.get("SPUSBDataType", [])
    matches = [
        item
        for item in walk(devices)
        if parse_hex_field(item.get("vendor_id")) == FUJI_VENDOR
    ]

    if not matches:
        print("No Fujifilm USB device is currently enumerated.")
        return 1

    for item in matches:
        product = parse_hex_field(item.get("product_id"))
        mode = KNOWN_MODES.get(product, "unclassified Fujifilm USB mode")
        product_text = "unknown" if product is None else f"0x{product:04x}"
        print(f"{item.get('_name', 'Fujifilm device')}: 04cb:{product_text[2:]} — {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
