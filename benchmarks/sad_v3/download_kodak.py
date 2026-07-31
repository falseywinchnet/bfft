#!/usr/bin/env python3
"""Download and validate the 24-image Kodak lossless test suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

from PIL import Image


BASE_URL = "https://r0k.us/graphics/kodak/kodak"
IMAGE_NAMES = tuple(f"kodim{index:02d}.png" for index in range(1, 25))


def _valid_kodak_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return (
                image.mode == "RGB"
                and tuple(sorted(image.size)) == (512, 768)
            )
    except (OSError, ValueError):
        return False


def download_kodak(output: Path, *, force: bool = False) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name in IMAGE_NAMES:
        destination = output / name
        if not force and _valid_kodak_image(destination):
            print(f"ready      {name}")
            downloaded.append(destination)
            continue

        request = Request(
            f"{BASE_URL}/{name}",
            headers={"User-Agent": "bfft-sad-v3-benchmark/1"},
        )
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        with tempfile.NamedTemporaryFile(
            prefix=f".{name}.",
            suffix=".tmp",
            dir=output,
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        if not _valid_kodak_image(temporary_path):
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded invalid Kodak image: {name}")
        temporary_path.replace(destination)
        print(f"downloaded {name} ({len(payload):,} bytes)")
        downloaded.append(destination)
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "kodak",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = download_kodak(args.output.resolve(), force=args.force)
    print(f"Kodak suite ready: {len(paths)} images in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

