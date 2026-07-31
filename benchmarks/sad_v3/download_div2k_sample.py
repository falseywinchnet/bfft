#!/usr/bin/env python3
"""Range-extract a few images from the official DIV2K validation ZIP."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen
import zipfile

from PIL import Image


ARCHIVE_URL = (
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip")
CONTENT_RANGE = re.compile(r"bytes \d+-\d+/(\d+)")


def _archive_size() -> int:
    request = Request(
        ARCHIVE_URL,
        headers={
            "Range": "bytes=0-0",
            "User-Agent": "bfft-sad-v3-benchmark/1",
        },
    )
    with urlopen(request, timeout=60) as response:
        match = CONTENT_RANGE.fullmatch(
            response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None:
            raise RuntimeError("DIV2K host did not honor a byte-range request")
        return int(match.group(1))


class _RemoteRangeReader(io.RawIOBase):
    """Minimal seekable HTTP range reader sufficient for ``zipfile``."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self.transferred = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid seek mode: {whence}")
        self.position = max(0, position)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        end = min(self.size, self.position + size) - 1
        request = Request(
            self.url,
            headers={
                "Range": f"bytes={self.position}-{end}",
                "User-Agent": "bfft-sad-v3-benchmark/1",
            },
        )
        with urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError(
                    "DIV2K host stopped honoring byte-range requests")
            payload = response.read()
        self.position += len(payload)
        self.transferred += len(payload)
        return payload


def _valid_rgb(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode == "RGB" and max(image.size) > 768
    except (OSError, ValueError):
        return False


def download_samples(output: Path, count: int) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    reader = _RemoteRangeReader(ARCHIVE_URL, _archive_size())
    extracted = []
    with zipfile.ZipFile(reader) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(".png")
        )[:count]
        for name in names:
            destination = output / Path(name).name
            if _valid_rgb(destination):
                print(f"ready      {destination.name}")
                extracted.append(destination)
                continue
            with archive.open(name) as source:
                payload = source.read()
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=output,
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary_path = Path(temporary.name)
            if not _valid_rgb(temporary_path):
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(f"invalid DIV2K member: {name}")
            temporary_path.replace(destination)
            with Image.open(destination) as image:
                print(
                    f"extracted  {destination.name} "
                    f"({image.width}x{image.height}, {len(payload):,} bytes)")
            extracted.append(destination)
    print(f"HTTP range bytes transferred: {reader.transferred:,}")
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "div2k_sample",
    )
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        raise SystemExit("--count must be between 1 and 100")
    paths = download_samples(args.output.resolve(), args.count)
    print(f"DIV2K sample ready: {len(paths)} image(s) in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

