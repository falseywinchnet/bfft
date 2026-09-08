"""Lossless PDF analysis and representation optimization.

The safe optimizer never decodes and re-encodes JPEG, JPEG 2000, JBIG2, or
CCITT image streams.  It searches several syntax-equivalent representations,
then retains the smallest measured file, including the original as a candidate.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable
import warnings
import zlib

import pikepdf
from pikepdf import Array, Dictionary, Name, ObjectStreamMode, Stream
from pypdf import PdfReader
import zopfli.zlib


class SignedPdfError(RuntimeError):
    """Raised when a rewrite would invalidate a PDF signature."""


@dataclass(frozen=True)
class OptimizationConfig:
    """Bounds and search controls for a lossless optimization run."""

    use_zopfli: bool = True
    zopfli_iterations: int = 8
    workers: int = max(1, min(4, os.cpu_count() or 1))
    max_encoded_stream_bytes: int = 32 * 1024 * 1024
    max_inflated_stream_bytes: int = 128 * 1024 * 1024
    text_sample_pages: int = 24
    force_signed: bool = False
    jbig2_encoder: str | Path | None = None
    jbig2_decoder: str | Path | None = None
    jbig2_timeout_seconds: float = 120.0
    jbig2_try_refined_symbols: bool = False
    jbig2_symbol_threshold: float = 0.94
    jbig2_try_entropy_regions: bool = False
    jbig2_entropy_cell_size: int = 128
    jbig2_entropy_frontier: int = 16

    def __post_init__(self) -> None:
        if self.zopfli_iterations < 1:
            raise ValueError("zopfli_iterations must be positive")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.max_encoded_stream_bytes < 1 or self.max_inflated_stream_bytes < 1:
            raise ValueError("stream byte ceilings must be positive")
        if (self.jbig2_encoder is None) != (self.jbig2_decoder is None):
            raise ValueError("JBIG2 optimization requires both encoder and decoder")
        if self.jbig2_timeout_seconds <= 0:
            raise ValueError("jbig2_timeout_seconds must be positive")
        if not 0.4 <= self.jbig2_symbol_threshold <= 0.97:
            raise ValueError("jbig2_symbol_threshold must be between 0.4 and 0.97")
        if self.jbig2_entropy_cell_size < 32 or self.jbig2_entropy_frontier < 1:
            raise ValueError("invalid JBIG2 entropy-region search bounds")

    @property
    def use_jbig2(self) -> bool:
        return self.jbig2_encoder is not None and self.jbig2_decoder is not None


@dataclass(frozen=True)
class OptimizationResult:
    source: Path
    output: Path
    source_bytes: int
    output_bytes: int
    selected_candidate: str
    candidate_bytes: dict[str, int]
    flate_streams_replaced: int
    flate_stream_bytes_before: int
    flate_stream_bytes_after: int
    jbig2_streams_replaced: int
    jbig2_stream_bytes_before: int
    jbig2_stream_bytes_after: int
    skipped_streams: dict[str, int]
    analysis: dict[str, Any]

    @property
    def saved_bytes(self) -> int:
        return self.source_bytes - self.output_bytes

    @property
    def saved_fraction(self) -> float:
        return self.saved_bytes / self.source_bytes if self.source_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = str(self.source)
        payload["output"] = str(self.output)
        payload["saved_bytes"] = self.saved_bytes
        payload["saved_fraction"] = self.saved_fraction
        return payload


def _filter_label(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, Array):
        return " ".join(str(item) for item in value)
    return str(value)


def _stream_category(stream: Stream) -> str:
    subtype = str(stream.get("/Subtype", ""))
    stream_type = str(stream.get("/Type", ""))
    if subtype == "/Image":
        return "image"
    if stream_type == "/ObjStm":
        return "object_stream"
    if stream_type == "/XRef":
        return "xref_stream"
    if stream_type == "/Metadata":
        return "metadata"
    if subtype in {"/Type1C", "/CIDFontType0C", "/OpenType"}:
        return "font_program"
    if any(key in stream for key in ("/Length1", "/Length2", "/Length3")):
        return "font_program"
    if subtype == "/Form":
        return "form"
    return "content_or_auxiliary"


def _walk_resources(resources: Any, fonts: dict[tuple[int, int], Any], seen: set[tuple[int, int]]) -> None:
    if not resources:
        return
    for font in (resources.get("/Font", {}) or {}).values():
        if font.objgen != (0, 0):
            fonts[font.objgen] = font
    for xobject in (resources.get("/XObject", {}) or {}).values():
        if xobject.objgen in seen:
            continue
        seen.add(xobject.objgen)
        if str(xobject.get("/Subtype", "")) == "/Form":
            _walk_resources(xobject.get("/Resources"), fonts, seen)


def _font_inventory(pdf: pikepdf.Pdf) -> dict[str, Any]:
    fonts: dict[tuple[int, int], Any] = {}
    seen: set[tuple[int, int]] = set()
    for page in pdf.pages:
        _walk_resources(page.get("/Resources"), fonts, seen)

    programs: dict[tuple[int, int], tuple[str, int]] = {}
    families: Counter[str] = Counter()
    to_unicode: dict[tuple[int, int], int] = {}
    for font in fonts.values():
        base = str(font.get("/BaseFont", ""))
        descriptor = font.get("/FontDescriptor")
        descendants = font.get("/DescendantFonts") or []
        if not descriptor and descendants:
            descendant = descendants[0]
            base = str(descendant.get("/BaseFont", base))
            descriptor = descendant.get("/FontDescriptor")
        program = None
        if descriptor:
            for key in ("/FontFile", "/FontFile2", "/FontFile3"):
                if descriptor.get(key):
                    program = descriptor[key]
                    break
        if program is not None and program.objgen not in programs:
            size = len(program.read_raw_bytes())
            programs[program.objgen] = (base, size)
            family = re.sub(r"^/[A-Z]{6}\+", "/", base)
            families[family] += 1
        cmap = font.get("/ToUnicode")
        if cmap is not None:
            to_unicode[cmap.objgen] = len(cmap.read_raw_bytes())
    fragmented = {family: count for family, count in families.items() if count > 1}
    return {
        "font_resource_count": len(fonts),
        "embedded_font_program_count": len(programs),
        "embedded_font_program_bytes": sum(size for _, size in programs.values()),
        "to_unicode_stream_count": len(to_unicode),
        "to_unicode_stream_bytes": sum(to_unicode.values()),
        "fragmented_subset_families": fragmented,
    }


def _has_signature(pdf: pikepdf.Pdf) -> bool:
    if pdf.Root.get("/Perms"):
        return True
    form = pdf.Root.get("/AcroForm")
    pending = list((form.get("/Fields") or [])) if form else []
    seen: set[tuple[int, int]] = set()
    while pending:
        field = pending.pop()
        if field.objgen in seen:
            continue
        seen.add(field.objgen)
        value = field.get("/V")
        value_is_signature = isinstance(value, Dictionary) and str(value.get("/Type", "")) == "/Sig"
        if str(field.get("/FT", "")) == "/Sig" or value_is_signature:
            return True
        pending.extend(field.get("/Kids") or [])
    return False


def _sample_text(path: Path, password: str, maximum_pages: int) -> dict[str, int]:
    try:
        reader = PdfReader(str(path), password=password or None, strict=False)
    except Exception:
        return {"sampled_pages": 0, "sampled_text_characters": 0}
    sample_count = min(maximum_pages, len(reader.pages))
    characters = 0
    for page in reader.pages[:sample_count]:
        try:
            characters += len(page.extract_text() or "")
        except Exception:
            continue
    return {"sampled_pages": sample_count, "sampled_text_characters": characters}


def analyze_pdf(
    path: str | Path,
    *,
    password: str = "",
    text_sample_pages: int = 24,
) -> dict[str, Any]:
    """Inventory the stored representation without rasterizing the document."""

    source = Path(path)
    filters: Counter[str] = Counter()
    filter_bytes: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    category_bytes: Counter[str] = Counter()
    image_filters: Counter[str] = Counter()
    image_bytes = 0
    image_pixels = 0
    jbig2_without_globals = 0
    dct_images = 0
    flate_images = 0

    with pikepdf.open(source, password=password) as pdf:
        signed = _has_signature(pdf)
        stream_count = 0
        for obj in pdf.objects:
            if not isinstance(obj, Stream):
                continue
            stream_count += 1
            raw_size = len(obj.read_raw_bytes())
            filter_name = _filter_label(obj.get("/Filter"))
            category = _stream_category(obj)
            filters[filter_name] += 1
            filter_bytes[filter_name] += raw_size
            categories[category] += 1
            category_bytes[category] += raw_size
            if category != "image":
                continue
            image_filters[filter_name] += 1
            image_bytes += raw_size
            image_pixels += int(obj.get("/Width", 0)) * int(obj.get("/Height", 0))
            dct_images += filter_name == "/DCTDecode"
            flate_images += filter_name == "/FlateDecode"
            if filter_name == "/JBIG2Decode":
                decode_parms = obj.get("/DecodeParms")
                globals_obj = decode_parms.get("/JBIG2Globals") if isinstance(decode_parms, Dictionary) else None
                jbig2_without_globals += globals_obj is None

        font_inventory = _font_inventory(pdf)
        page_count = len(pdf.pages)
        payload: dict[str, Any] = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "pdf_version": str(pdf.pdf_version),
            "pages": page_count,
            "objects": len(pdf.objects),
            "streams": stream_count,
            "encrypted": bool(pdf.is_encrypted),
            "linearized": bool(pdf.is_linearized),
            "signed": signed,
            "filters": dict(sorted(filters.items())),
            "filter_bytes": dict(sorted(filter_bytes.items())),
            "stream_categories": dict(sorted(categories.items())),
            "stream_category_bytes": dict(sorted(category_bytes.items())),
            "image_count": sum(image_filters.values()),
            "image_stream_bytes": image_bytes,
            "image_sample_pixels": image_pixels,
            "image_filters": dict(sorted(image_filters.items())),
            "jbig2_images_without_global_dictionary": jbig2_without_globals,
            **font_inventory,
        }

    payload.update(_sample_text(source, password, text_sample_pages))
    pages = max(1, payload["pages"])
    image_dominated = (
        payload["image_count"] >= pages
        and payload["image_sample_pixels"] / pages >= 500_000
        and payload["embedded_font_program_count"] <= 3
    )
    native_text = payload["image_count"] == 0 and payload["font_resource_count"] > 0
    payload["representation_class"] = (
        "scanned_mixed_raster" if image_dominated else "native_text_vector" if native_text else "hybrid"
    )
    payload["method_applicability"] = {
        "manual_jpeg_transport_images": dct_images,
        "svg_flat_raster_candidates": flate_images,
        "jbig2_shared_symbol_dictionary_candidates": jbig2_without_globals,
        "font_subset_repacking_is_material": payload["embedded_font_program_bytes"] >= source.stat().st_size * 0.2,
        "safe_structural_rewrite_candidate": True,
    }
    return payload


def _inflate_bounded(raw: bytes, ceiling: int) -> bytes:
    decoder = zlib.decompressobj()
    output = bytearray()
    offset = 0
    chunk_size = 1 << 20
    while offset < len(raw):
        chunk = raw[offset : offset + chunk_size]
        offset += len(chunk)
        while chunk:
            remaining = ceiling - len(output)
            if remaining < 0:
                raise OverflowError("inflated stream exceeds configured ceiling")
            piece = decoder.decompress(chunk, remaining + 1)
            output.extend(piece)
            if len(output) > ceiling:
                raise OverflowError("inflated stream exceeds configured ceiling")
            chunk = decoder.unconsumed_tail
    remaining = ceiling - len(output)
    output.extend(decoder.flush(remaining + 1))
    if len(output) > ceiling:
        raise OverflowError("inflated stream exceeds configured ceiling")
    if not decoder.eof:
        raise zlib.error("incomplete Flate stream")
    return bytes(output)


def _zopfli_candidate(raw: bytes, iterations: int, ceiling: int) -> tuple[bytes | None, str]:
    try:
        decoded = _inflate_bounded(raw, ceiling)
    except OverflowError:
        return None, "inflated_ceiling"
    except zlib.error:
        return None, "invalid_flate"
    candidate = zopfli.zlib.compress(decoded, numiterations=iterations)
    if len(candidate) >= len(raw):
        return None, "not_smaller"
    if _inflate_bounded(candidate, ceiling) != decoded:
        return None, "verification_failed"
    return candidate, "replaced"


def _eligible_flate_streams(pdf: pikepdf.Pdf, config: OptimizationConfig) -> Iterable[Stream]:
    for obj in pdf.objects:
        if not isinstance(obj, Stream):
            continue
        stream_filter = obj.get("/Filter")
        pure_flate = str(stream_filter) == "/FlateDecode" or (
            isinstance(stream_filter, Array)
            and len(stream_filter) == 1
            and str(stream_filter[0]) == "/FlateDecode"
        )
        if not pure_flate:
            continue
        if str(obj.get("/Type", "")) in {"/ObjStm", "/XRef"}:
            continue
        if len(obj.read_raw_bytes()) > config.max_encoded_stream_bytes:
            continue
        yield obj


def _recompress_flate_streams(pdf: pikepdf.Pdf, config: OptimizationConfig) -> dict[str, Any]:
    streams = list(_eligible_flate_streams(pdf, config))
    skipped: Counter[str] = Counter()
    before = 0
    after = 0
    replaced = 0

    def apply(stream: Stream, raw: bytes, result: tuple[bytes | None, str]) -> None:
        nonlocal before, after, replaced
        candidate, reason = result
        before += len(raw)
        if candidate is None:
            after += len(raw)
            skipped[reason] += 1
            return
        decode_parms = stream.get("/DecodeParms")
        original_filter = stream.get("/Filter")
        stream.write(candidate, filter=original_filter, decode_parms=decode_parms)
        if stream.read_raw_bytes() != candidate:
            raise RuntimeError("pikepdf did not retain the proposed Flate representation")
        after += len(candidate)
        replaced += 1

    if config.workers == 1 or len(streams) < 2:
        for stream in streams:
            raw = stream.read_raw_bytes()
            apply(stream, raw, _zopfli_candidate(raw, config.zopfli_iterations, config.max_inflated_stream_bytes))
    else:
        with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="pdf-zopfli") as pool:
            pending: dict[Future[tuple[bytes | None, str]], tuple[Stream, bytes]] = {}
            iterator = iter(streams)

            def submit_one() -> bool:
                try:
                    stream = next(iterator)
                except StopIteration:
                    return False
                raw = stream.read_raw_bytes()
                future = pool.submit(
                    _zopfli_candidate,
                    raw,
                    config.zopfli_iterations,
                    config.max_inflated_stream_bytes,
                )
                pending[future] = (stream, raw)
                return True

            for _ in range(min(len(streams), config.workers * 2)):
                submit_one()
            while pending:
                completed = next(as_completed(tuple(pending)))
                stream, raw = pending.pop(completed)
                apply(stream, raw, completed.result())
                submit_one()

    return {
        "replaced": replaced,
        "before": before,
        "after": after,
        "skipped": dict(skipped),
    }


def _eligible_jbig2_streams(pdf: pikepdf.Pdf) -> Iterable[Stream]:
    for obj in pdf.objects:
        if not isinstance(obj, Stream):
            continue
        if str(obj.get("/Subtype", "")) != "/Image" or str(obj.get("/Filter", "")) != "/JBIG2Decode":
            continue
        decode_parms = obj.get("/DecodeParms")
        if isinstance(decode_parms, Dictionary) and decode_parms.get("/JBIG2Globals") is not None:
            continue
        if int(obj.get("/Width", 0)) <= 0 or int(obj.get("/Height", 0)) <= 0:
            continue
        yield obj


def _recompress_jbig2_streams(
    pdf: pikepdf.Pdf,
    config: OptimizationConfig,
    *,
    temporary_root: Path,
) -> dict[str, Any]:
    from .jbig2 import Jbig2Candidate, optimize_generic_stream

    streams = list(_eligible_jbig2_streams(pdf))
    skipped: Counter[str] = Counter()
    before = 0
    after = 0
    replaced = 0

    def propose(stream: Stream, raw: bytes) -> Jbig2Candidate:
        return optimize_generic_stream(
            raw,
            width=int(stream.Width),
            height=int(stream.Height),
            encoder=str(config.jbig2_encoder),
            decoder=str(config.jbig2_decoder),
            try_template_1=False,
            try_refined_symbols=config.jbig2_try_refined_symbols,
            symbol_threshold=config.jbig2_symbol_threshold,
            try_entropy_regions=config.jbig2_try_entropy_regions,
            entropy_cell_size=config.jbig2_entropy_cell_size,
            entropy_frontier=config.jbig2_entropy_frontier,
            timeout=config.jbig2_timeout_seconds,
            temporary_root=temporary_root,
        )

    def apply(stream: Stream, raw: bytes, candidate: Jbig2Candidate) -> None:
        nonlocal before, after, replaced
        before += len(raw)
        if candidate.data is None:
            after += len(raw)
            skipped[candidate.reason] += 1
            return
        original_filter = stream.get("/Filter")
        decode_parms = stream.get("/DecodeParms")
        globals_stream = None
        if candidate.globals_data is not None:
            globals_stream = pdf.make_stream(candidate.globals_data)
            decode_parms = Dictionary(decode_parms) if isinstance(decode_parms, Dictionary) else Dictionary()
            decode_parms[Name.JBIG2Globals] = globals_stream
        stream.write(candidate.data, filter=original_filter, decode_parms=decode_parms)
        if stream.read_raw_bytes() != candidate.data:
            raise RuntimeError("pikepdf did not retain the proposed JBIG2 representation")
        if globals_stream is not None and globals_stream.read_raw_bytes() != candidate.globals_data:
            raise RuntimeError("pikepdf did not retain the proposed JBIG2 globals")
        after += candidate.candidate_bytes
        replaced += 1

    if config.workers == 1 or len(streams) < 2:
        for stream in streams:
            raw = stream.read_raw_bytes()
            apply(stream, raw, propose(stream, raw))
    else:
        with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="pdf-jbig2") as pool:
            pending: dict[Future[Jbig2Candidate], tuple[Stream, bytes]] = {}
            iterator = iter(streams)

            def submit_one() -> bool:
                try:
                    stream = next(iterator)
                except StopIteration:
                    return False
                raw = stream.read_raw_bytes()
                pending[pool.submit(propose, stream, raw)] = (stream, raw)
                return True

            for _ in range(min(len(streams), config.workers * 2)):
                submit_one()
            while pending:
                completed = next(as_completed(tuple(pending)))
                stream, raw = pending.pop(completed)
                apply(stream, raw, completed.result())
                submit_one()

    return {
        "replaced": replaced,
        "before": before,
        "after": after,
        "skipped": dict(skipped),
    }


def _save_pdf(pdf: pikepdf.Pdf, path: Path, mode: ObjectStreamMode) -> None:
    pdf.remove_unreferenced_resources()
    encryption: bool | None = True if pdf.is_encrypted else None
    pdf.save(
        path,
        compress_streams=True,
        recompress_flate=False,
        object_stream_mode=mode,
        linearize=False,
        preserve_pdfa=True,
        encryption=encryption,
    )


def _check_output(path: Path, expected_pages: int, password: str) -> None:
    with pikepdf.open(path, password=password) as pdf:
        if len(pdf.pages) != expected_pages:
            raise RuntimeError("page count changed during optimization")
        for page in pdf.pages:
            _ = page.obj
        with warnings.catch_warnings():
            # libqpdf still validates syntax and all generalized streams when
            # optional JPX/JBIG2 decoder executables are absent.
            warnings.filterwarnings(
                "ignore",
                message=r"pikepdf is missing some specialized decoders.*",
                category=UserWarning,
            )
            problems = pdf.check_pdf_syntax()
        if problems:
            raise RuntimeError("PDF syntax check failed: " + "; ".join(problems))


def optimize_pdf(
    source: str | Path,
    output: str | Path,
    *,
    config: OptimizationConfig | None = None,
    password: str = "",
) -> OptimizationResult:
    """Search lossless PDF representations and write only the byte winner."""

    config = config or OptimizationConfig()
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError("source and output must be different paths")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis = analyze_pdf(source_path, password=password, text_sample_pages=config.text_sample_pages)
    if analysis["signed"] and not config.force_signed:
        raise SignedPdfError("refusing to rewrite a signed PDF; use force_signed only after accepting signature invalidation")

    candidate_bytes: dict[str, int] = {"original": source_path.stat().st_size}
    zopfli_stats = {"replaced": 0, "before": 0, "after": 0, "skipped": {}}
    jbig2_stats = {"replaced": 0, "before": 0, "after": 0, "skipped": {}}
    with tempfile.TemporaryDirectory(prefix=".pdf-repr-", dir=output_path.parent) as temporary:
        temporary_path = Path(temporary)
        candidates: dict[str, Path] = {"original": source_path}

        structural = temporary_path / "structural.pdf"
        with pikepdf.open(source_path, password=password) as pdf:
            _save_pdf(pdf, structural, ObjectStreamMode.generate)
        candidates["structural_packed"] = structural
        candidate_bytes["structural_packed"] = structural.stat().st_size

        if config.use_zopfli or config.use_jbig2:
            with pikepdf.open(source_path, password=password) as pdf:
                if config.use_zopfli:
                    zopfli_stats = _recompress_flate_streams(pdf, config)
                if config.use_jbig2:
                    jbig2_stats = _recompress_jbig2_streams(
                        pdf,
                        config,
                        temporary_root=temporary_path,
                    )
                label = "zopfli_jbig2" if config.use_zopfli and config.use_jbig2 else "zopfli" if config.use_zopfli else "jbig2"
                preserved = temporary_path / f"{label}-preserved.pdf"
                packed = temporary_path / f"{label}-packed.pdf"
                _save_pdf(pdf, preserved, ObjectStreamMode.preserve)
                _save_pdf(pdf, packed, ObjectStreamMode.generate)
            candidates[f"{label}_preserved"] = preserved
            candidates[f"{label}_packed"] = packed
            candidate_bytes[f"{label}_preserved"] = preserved.stat().st_size
            candidate_bytes[f"{label}_packed"] = packed.stat().st_size

        selected = min(candidate_bytes, key=lambda name: (candidate_bytes[name], name != "original"))
        shutil.copyfile(candidates[selected], output_path)

    _check_output(output_path, analysis["pages"], password)
    return OptimizationResult(
        source=source_path,
        output=output_path,
        source_bytes=source_path.stat().st_size,
        output_bytes=output_path.stat().st_size,
        selected_candidate=selected,
        candidate_bytes=candidate_bytes,
        flate_streams_replaced=int(zopfli_stats["replaced"]),
        flate_stream_bytes_before=int(zopfli_stats["before"]),
        flate_stream_bytes_after=int(zopfli_stats["after"]),
        jbig2_streams_replaced=int(jbig2_stats["replaced"]),
        jbig2_stream_bytes_before=int(jbig2_stats["before"]),
        jbig2_stream_bytes_after=int(jbig2_stats["after"]),
        skipped_streams={
            **{f"flate:{key}": value for key, value in dict(zopfli_stats["skipped"]).items()},
            **{f"jbig2:{key}": value for key, value in dict(jbig2_stats["skipped"]).items()},
        },
        analysis=analysis,
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
