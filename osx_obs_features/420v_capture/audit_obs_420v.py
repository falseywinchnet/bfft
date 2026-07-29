#!/usr/bin/env python3
"""Read-only audit of Apple 420v selections in an OBS macOS installation."""

from __future__ import annotations

import argparse
import base64
import json
import plistlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


KNOWN_FORMATS = {
    "420v": ("NV12", "video/limited range"),
    "420f": ("NV12", "full range"),
    "yuvs": ("YUY2", "video/limited range"),
    "2vuy": ("UYVY", "video/limited range"),
    "x420": ("P010", "10-bit video/limited range"),
    "xf20": ("P010", "10-bit full range"),
}


@dataclass(frozen=True)
class SelectedFormat:
    collection: str
    source: str
    description: str
    subtype_value: int
    fourcc: str


def fourcc_from_int(value: int) -> str:
    if not 0 <= value <= 0xFFFFFFFF:
        return f"0x{value:x}"
    raw = value.to_bytes(4, "big")
    if all(32 <= byte <= 126 for byte in raw):
        return raw.decode("ascii")
    return f"0x{value:08x}"


def decode_supported_format(encoded: str) -> tuple[str, int, str]:
    description = base64.b64decode(encoded, validate=True).decode("utf-8")
    try:
        subtype_value = int(description.rsplit(maxsplit=1)[-1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"missing numeric subtype in {description!r}") from error
    return description, subtype_value, fourcc_from_int(subtype_value)


def iter_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def selected_formats(scene_file: Path) -> list[SelectedFormat]:
    document = json.loads(scene_file.read_text(encoding="utf-8"))
    found: list[SelectedFormat] = []
    seen: set[tuple[str, str]] = set()
    for item in iter_dicts(document):
        settings = item.get("settings")
        if not isinstance(settings, dict):
            continue
        encoded = settings.get("supported_format")
        if not isinstance(encoded, str) or not encoded:
            continue
        source = str(item.get("name", "<unnamed source>"))
        marker = (source, encoded)
        if marker in seen:
            continue
        seen.add(marker)
        try:
            description, value, fourcc = decode_supported_format(encoded)
        except (ValueError, UnicodeError) as error:
            description = f"<unreadable: {error}>"
            value = -1
            fourcc = "unknown"
        found.append(
            SelectedFormat(scene_file.stem, source, description, value, fourcc)
        )
    return found


def obs_version(app: Path) -> str:
    info = app / "Contents" / "Info.plist"
    try:
        with info.open("rb") as stream:
            plist = plistlib.load(stream)
        return str(
            plist.get("CFBundleShortVersionString")
            or plist.get("CFBundleVersion")
            or "unknown"
        )
    except (OSError, plistlib.InvalidFileException):
        return "not found"


def recent_format_lines(log_dir: Path, limit: int) -> list[tuple[str, str]]:
    logs = sorted(
        log_dir.glob("*.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    pattern = re.compile(r"Using Format\s*:.*(?:420v|420f|yuvs|2vuy)")
    evidence: list[tuple[str, str]] = []
    for log in logs:
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if pattern.search(line):
                evidence.append((log.name, line.strip()))
                if len(evidence) >= limit:
                    return evidence
    return evidence


def describe_format(fourcc: str) -> str:
    if fourcc not in KNOWN_FORMATS:
        return "unrecognized by this audit"
    obs_name, range_name = KNOWN_FORMATS[fourcc]
    return f"OBS {obs_name}; {range_name}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-app", type=Path, default=Path("/Applications/OBS.app"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / "Library" / "Application Support" / "obs-studio",
    )
    parser.add_argument("--log-lines", type=int, default=8)
    args = parser.parse_args(argv)

    plugin = (
        args.obs_app
        / "Contents"
        / "PlugIns"
        / "mac-avcapture.plugin"
        / "Contents"
        / "MacOS"
        / "mac-avcapture"
    )
    print(f"OBS app:       {args.obs_app}")
    print(f"OBS version:   {obs_version(args.obs_app)}")
    print(f"capture plugin:{' present' if plugin.is_file() else ' MISSING'}")

    scenes = args.config / "basic" / "scenes"
    formats: list[SelectedFormat] = []
    for scene in sorted(scenes.glob("*.json")):
        try:
            formats.extend(selected_formats(scene))
        except (OSError, json.JSONDecodeError) as error:
            print(f"warning: could not read {scene}: {error}", file=sys.stderr)

    print("\nSelected custom capture formats:")
    if not formats:
        print("  none found")
    for item in formats:
        print(f"  {item.collection} / {item.source}")
        print(f"    FourCC={item.fourcc} ({item.subtype_value})")
        print(f"    meaning={describe_format(item.fourcc)}")
        print(f"    encoded={item.description}")

    print("\nRecent OBS capture evidence:")
    evidence = recent_format_lines(args.config / "logs", max(args.log_lines, 0))
    if not evidence:
        print("  no matching log lines found")
    for name, line in evidence:
        print(f"  {name}: {line}")

    if any(item.fourcc == "420v" for item in formats):
        print("\nPASS: an OBS scene explicitly selects Apple 420v (OBS NV12).")
        return 0
    print("\nNOTE: no saved scene currently selects 420v; inspect the log evidence above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
