"""Verified, bounded optimization of PDF-embedded JBIG2 generic regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageChops, ImageOps


_SEGMENT_TYPES = {
    38: "immediate_generic_region",
    48: "page_information",
}


@dataclass(frozen=True)
class Segment:
    number: int
    kind: int
    page: int
    header_start: int
    data_start: int
    data_length: int


@dataclass(frozen=True)
class Jbig2Candidate:
    data: bytes | None
    source_bytes: int
    candidate_bytes: int
    kind: str
    reason: str
    globals_data: bytes | None = None


def parse_embedded_segments(data: bytes) -> list[Segment]:
    """Parse sequential JBIG2 segment headers used by PDF embedded streams."""

    offset = 0
    segments: list[Segment] = []
    while offset < len(data):
        header_start = offset
        if len(data) - offset < 11:
            raise ValueError("truncated JBIG2 segment header")
        number = int.from_bytes(data[offset : offset + 4], "big")
        offset += 4
        flags = data[offset]
        offset += 1
        kind = flags & 0x3F
        long_page_association = bool(flags & 0x40)

        first_referred = data[offset]
        count = first_referred >> 5
        if count == 7:
            if len(data) - offset < 4:
                raise ValueError("truncated JBIG2 referred-segment count")
            count = int.from_bytes(data[offset : offset + 4], "big") & 0x1FFFFFFF
            offset += 4
            offset += (count + 8) // 8
        elif count in {5, 6}:
            raise ValueError("reserved JBIG2 referred-segment count")
        else:
            offset += 1

        reference_width = 1 if number <= 256 else 2 if number <= 65536 else 4
        offset += count * reference_width
        page_width = 4 if long_page_association else 1
        if len(data) - offset < page_width + 4:
            raise ValueError("truncated JBIG2 segment metadata")
        page = int.from_bytes(data[offset : offset + page_width], "big")
        offset += page_width
        data_length = int.from_bytes(data[offset : offset + 4], "big")
        offset += 4
        if data_length == 0xFFFFFFFF:
            raise ValueError("indeterminate JBIG2 segment length is unsupported")
        if len(data) - offset < data_length:
            raise ValueError("truncated JBIG2 segment payload")
        segments.append(
            Segment(
                number=number,
                kind=kind,
                page=page,
                header_start=header_start,
                data_start=offset,
                data_length=data_length,
            )
        )
        offset += data_length
    return segments


def _patch_cropped_stream(data: bytes, *, page_width: int, page_height: int, x: int, y: int) -> bytes:
    segments = parse_embedded_segments(data)
    page_info = [segment for segment in segments if segment.kind == 48]
    regions = [segment for segment in segments if segment.kind == 38]
    if len(segments) != 2 or len(page_info) != 1 or len(regions) != 1:
        kinds = [_SEGMENT_TYPES.get(segment.kind, str(segment.kind)) for segment in segments]
        raise ValueError(f"unexpected generic encoder segments: {kinds}")
    info = page_info[0]
    region = regions[0]
    if info.data_length != 19 or region.data_length < 20:
        raise ValueError("unexpected generic encoder payload size")

    patched = bytearray(data)
    patched[info.data_start : info.data_start + 4] = page_width.to_bytes(4, "big")
    patched[info.data_start + 4 : info.data_start + 8] = page_height.to_bytes(4, "big")
    # A blank default page plus REPLACE-composited ink region permits a crop.
    patched[info.data_start + 16] |= 0x04
    patched[region.data_start + 8 : region.data_start + 12] = x.to_bytes(4, "big")
    patched[region.data_start + 12 : region.data_start + 16] = y.to_bytes(4, "big")
    patched[region.data_start + 16] = 4
    return bytes(patched)


def append_cropped_region(
    page_stream: bytes,
    generic_crop_stream: bytes,
    *,
    x: int,
    y: int,
    composition_operator: int,
) -> bytes:
    """Append a standalone cropped generic region to an existing page stream.

    ``generic_crop_stream`` must be the PDF-mode output of the generic encoder:
    one page-information segment followed by one immediate generic region.  Its
    page-information segment is discarded; the region is renumbered after the
    existing symbol/text page segments and composited as a sparse correction.
    """

    page_segments = parse_embedded_segments(page_stream)
    crop_segments = parse_embedded_segments(generic_crop_stream)
    regions = [segment for segment in crop_segments if segment.kind == 38]
    if len(crop_segments) != 2 or len(regions) != 1:
        raise ValueError("unexpected residual generic stream structure")
    if not page_segments:
        raise ValueError("empty symbol page stream")
    region = regions[0]
    if region.data_length < 20:
        raise ValueError("truncated residual generic region")

    segment_end = region.data_start + region.data_length
    appended = bytearray(generic_crop_stream[region.header_start:segment_end])
    payload = region.data_start - region.header_start
    appended[0:4] = (max(segment.number for segment in page_segments) + 1).to_bytes(4, "big")
    appended[payload + 8 : payload + 12] = x.to_bytes(4, "big")
    appended[payload + 12 : payload + 16] = y.to_bytes(4, "big")
    if composition_operator not in range(5):
        raise ValueError("invalid JBIG2 composition operator")
    appended[payload + 16] = composition_operator
    return page_stream + bytes(appended)


def append_cropped_xor_region(
    page_stream: bytes,
    generic_crop_stream: bytes,
    *,
    x: int,
    y: int,
) -> bytes:
    """Append a standalone cropped generic region using XOR composition."""

    return append_cropped_region(
        page_stream,
        generic_crop_stream,
        x=x,
        y=y,
        composition_operator=2,
    )


def _run(command: list[str], *, stdout: bool = False, timeout: float = 120.0) -> bytes:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.stdout if stdout else b""


def _same_bitmap(left: Path, right: Path) -> bool:
    with Image.open(left) as a, Image.open(right) as b:
        if a.size != b.size:
            return False
        return ImageChops.difference(a.convert("1"), b.convert("1")).getbbox() is None


def optimize_generic_stream(
    raw: bytes,
    *,
    width: int,
    height: int,
    encoder: str | Path,
    decoder: str | Path,
    try_template_1: bool = False,
    try_refined_symbols: bool = False,
    symbol_threshold: float = 0.94,
    try_entropy_regions: bool = False,
    entropy_cell_size: int = 128,
    entropy_frontier: int = 16,
    timeout: float = 120.0,
    temporary_root: str | Path | None = None,
) -> Jbig2Candidate:
    """Return the smallest pixel-identical cropped generic-region candidate."""

    source_size = len(raw)
    try:
        source_segments = parse_embedded_segments(raw)
    except ValueError:
        return Jbig2Candidate(None, source_size, source_size, "source", "unparseable_source")
    if [segment.kind for segment in source_segments] != [48, 38]:
        return Jbig2Candidate(None, source_size, source_size, "source", "not_single_generic_region")

    with tempfile.TemporaryDirectory(prefix="jbig2-page-", dir=temporary_root) as directory:
        work = Path(directory)
        source_stream = work / "source.jb2"
        source_bitmap = work / "source.pbm"
        source_stream.write_bytes(raw)
        try:
            _run(
                [str(decoder), "-q", "-e", "-t", "pbm", "-o", str(source_bitmap), str(source_stream)],
                timeout=timeout,
            )
        except (subprocess.SubprocessError, OSError):
            return Jbig2Candidate(None, source_size, source_size, "source", "decode_failed")

        with Image.open(source_bitmap) as image:
            if image.size != (width, height):
                return Jbig2Candidate(None, source_size, source_size, "source", "dimension_mismatch")
            box = image.getbbox()
            if box is None:
                return Jbig2Candidate(None, source_size, source_size, "source", "empty_mask")
            crop_bitmap = work / "crop.pbm"
            image.crop(box).save(crop_bitmap)

        candidates: list[tuple[int, str, bytes, bytes | None]] = []
        templates = (0, 1) if try_template_1 else (0,)
        for template in templates:
            command = [str(encoder), "-p"]
            if template:
                command += ["--generic-template", str(template)]
            command.append(str(crop_bitmap))
            try:
                encoded = _run(command, stdout=True, timeout=timeout)
                patched = _patch_cropped_stream(
                    encoded,
                    page_width=width,
                    page_height=height,
                    x=box[0],
                    y=box[1],
                )
                candidate_stream = work / f"candidate-{template}.jb2"
                candidate_bitmap = work / f"candidate-{template}.pbm"
                candidate_stream.write_bytes(patched)
                _run(
                    [
                        str(decoder), "-q", "-e", "-t", "pbm", "-o",
                        str(candidate_bitmap), str(candidate_stream),
                    ],
                    timeout=timeout,
                )
            except (ValueError, subprocess.SubprocessError, OSError):
                continue
            if _same_bitmap(source_bitmap, candidate_bitmap):
                candidates.append((len(patched), f"crop_template_{template}", patched, None))

        if try_entropy_regions:
            try:
                import numpy as np
                from .entropy_regions import partition_pixel_boxes, propose_entropy_partitions

                with Image.open(source_bitmap) as source_image:
                    ink = np.logical_not(np.asarray(source_image.convert("1"), dtype=bool))
                partitions = propose_entropy_partitions(
                    ink,
                    cell_size=entropy_cell_size,
                    max_regions=2,
                    first_split_frontier=entropy_frontier,
                )
            except (ImportError, ValueError):
                partitions = ()
            for partition_index, partition in enumerate(partitions[1:], start=1):
                try:
                    stream = None
                    boxes = partition_pixel_boxes(partition, ink)
                    for region_index, (x0, y0, x1, y1) in enumerate(boxes):
                        region_bitmap = work / f"entropy-{partition_index}-{region_index}.pbm"
                        Image.fromarray(
                            np.where(ink[y0:y1, x0:x1], 0, 255).astype(np.uint8), "L"
                        ).convert("1").save(region_bitmap)
                        encoded = _run(
                            [str(encoder), "-p", str(region_bitmap)],
                            stdout=True,
                            timeout=timeout,
                        )
                        if stream is None:
                            stream = _patch_cropped_stream(
                                encoded, page_width=width, page_height=height, x=x0, y=y0
                            )
                        else:
                            stream = append_cropped_region(
                                stream, encoded, x=x0, y=y0, composition_operator=4
                            )
                    if stream is None:
                        continue
                    candidate_stream = work / f"entropy-{partition_index}.jb2"
                    candidate_bitmap = work / f"entropy-{partition_index}.decoded.pbm"
                    candidate_stream.write_bytes(stream)
                    _run(
                        [
                            str(decoder), "-q", "-e", "-t", "pbm", "-o",
                            str(candidate_bitmap), str(candidate_stream),
                        ],
                        timeout=timeout,
                    )
                except (ValueError, subprocess.SubprocessError, OSError):
                    continue
                if _same_bitmap(source_bitmap, candidate_bitmap):
                    candidates.append((len(stream), "entropy_regions_2", stream, None))

        if try_refined_symbols:
            base = work / "refined-symbol"
            try:
                _run(
                    [
                        str(encoder), "-p", "-s", "-r", "-t", str(symbol_threshold),
                        "-b", str(base), str(source_bitmap),
                    ],
                    timeout=timeout,
                )
                globals_path = base.with_suffix(".sym")
                page_path = Path(f"{base}.0000")
                symbol_bitmap = work / "refined-symbol.pbm"
                _run(
                    [
                        str(decoder), "-q", "-e", "-t", "pbm", "-o", str(symbol_bitmap),
                        str(globals_path), str(page_path),
                    ],
                    timeout=timeout,
                )
                globals_data = globals_path.read_bytes()
                page_data = page_path.read_bytes()
            except (subprocess.SubprocessError, OSError):
                pass
            else:
                verified_page_data = page_data
                if not _same_bitmap(source_bitmap, symbol_bitmap):
                    try:
                        with Image.open(source_bitmap) as source_image, Image.open(symbol_bitmap) as symbol_image:
                            difference = ImageChops.difference(
                                source_image.convert("L"), symbol_image.convert("L")
                            )
                            box = difference.getbbox()
                            if box is None:
                                raise ValueError("inconsistent symbol bitmap comparison")
                            guarded = (
                                max(0, box[0] - 4), max(0, box[1] - 4),
                                min(source_image.width, box[2] + 4),
                                min(source_image.height, box[3] + 4),
                            )
                            residual_bitmap = work / "refined-symbol-residual.pbm"
                            ImageOps.invert(difference.crop(guarded)).convert("1").save(residual_bitmap)
                        residual = _run(
                            [str(encoder), "-p", str(residual_bitmap)],
                            stdout=True,
                            timeout=timeout,
                        )
                        verified_page_data = append_cropped_xor_region(
                            page_data, residual, x=guarded[0], y=guarded[1]
                        )
                        hybrid_path = work / "refined-symbol-hybrid.jb2"
                        hybrid_bitmap = work / "refined-symbol-hybrid.pbm"
                        hybrid_path.write_bytes(verified_page_data)
                        _run(
                            [
                                str(decoder), "-q", "-e", "-t", "pbm", "-o", str(hybrid_bitmap),
                                str(globals_path), str(hybrid_path),
                            ],
                            timeout=timeout,
                        )
                    except (ValueError, subprocess.SubprocessError, OSError):
                        verified_page_data = b""
                    else:
                        if not _same_bitmap(source_bitmap, hybrid_bitmap):
                            verified_page_data = b""
                if verified_page_data:
                    candidates.append(
                        (
                            len(globals_data) + len(verified_page_data),
                            "private_refined_symbols",
                            verified_page_data,
                            globals_data,
                        )
                    )

        if not candidates:
            return Jbig2Candidate(None, source_size, source_size, "source", "no_verified_candidate")
        size, kind, data, globals_data = min(candidates, key=lambda item: (item[0], item[1]))
        if size >= source_size:
            return Jbig2Candidate(None, source_size, source_size, "source", "not_smaller")
        return Jbig2Candidate(data, source_size, size, kind, "replaced", globals_data)
