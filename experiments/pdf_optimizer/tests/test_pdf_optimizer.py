from __future__ import annotations

from pathlib import Path
import zlib

import pikepdf
from pikepdf import Dictionary, Name, ObjectStreamMode
import pytest

from pdf_optimizer.core import (
    OptimizationConfig,
    SignedPdfError,
    _inflate_bounded,
    analyze_pdf,
    optimize_pdf,
)
from pdf_optimizer.jbig2 import (
    _patch_cropped_stream,
    append_cropped_xor_region,
    parse_embedded_segments,
)
from pdf_optimizer.bwt_jbig2 import cyclic_bwt, rowwise_byte_bwt, zero_order_huffman_cost
from pdf_optimizer.mrc import MrcConfig, _background_repair, _field_grid
from pdf_optimizer.entropy_regions import propose_entropy_partitions
from pdf_optimizer.glyph_residual import (
    GlyphResidualConfig,
    canonicalize_mask,
    denoise_foreground_texture_family,
    foreground_owned_composite,
    fourier_circle_descriptor,
    signed_distance,
)


def _make_pdf(path: Path, *, signed: bool = False) -> bytes:
    decoded = (b"q 0.1 0.2 0.3 rg 10 10 200 200 re f Q\n" * 1500)
    with pikepdf.Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(300, 300))
        stream = pdf.make_stream(zlib.compress(decoded, 1))
        stream.Filter = Name.FlateDecode
        page.Contents = stream
        if signed:
            pdf.Root.Perms = Dictionary()
            pdf.Root.Perms.DocMDP = pdf.make_indirect(Dictionary(Type=Name.Sig))
        pdf.save(path, compress_streams=False, object_stream_mode=ObjectStreamMode.disable)
    return decoded


def test_analyze_and_optimize_losslessly(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    decoded = _make_pdf(source)
    analysis = analyze_pdf(source)
    assert analysis["pages"] == 1
    assert analysis["filters"]["/FlateDecode"] >= 1

    result = optimize_pdf(
        source,
        output,
        config=OptimizationConfig(workers=2, zopfli_iterations=3, text_sample_pages=1),
    )
    assert result.output_bytes <= result.source_bytes
    assert result.flate_streams_replaced >= 1
    with pikepdf.open(output) as pdf:
        assert pdf.pages[0].Contents.read_bytes() == decoded


def test_signed_pdf_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "signed.pdf"
    _make_pdf(source, signed=True)
    with pytest.raises(SignedPdfError):
        optimize_pdf(source, tmp_path / "out.pdf", config=OptimizationConfig(use_zopfli=False))


def test_inflate_ceiling_is_enforced() -> None:
    raw = zlib.compress(b"a" * 10_000)
    with pytest.raises(OverflowError):
        _inflate_bounded(raw, 100)


def test_single_element_flate_filter_array_is_supported(tmp_path: Path) -> None:
    source = tmp_path / "array-filter.pdf"
    output = tmp_path / "array-filter-out.pdf"
    decoded = _make_pdf(source)
    with pikepdf.open(source, allow_overwriting_input=True) as pdf:
        pdf.pages[0].Contents.Filter = pikepdf.Array([Name.FlateDecode])
        pdf.save(source)
    result = optimize_pdf(
        source,
        output,
        config=OptimizationConfig(workers=1, zopfli_iterations=3, text_sample_pages=0),
    )
    assert result.output_bytes <= result.source_bytes
    with pikepdf.open(output) as pdf:
        assert pdf.pages[0].Contents.read_bytes() == decoded


def _jbig2_segment(number: int, kind: int, payload: bytes) -> bytes:
    return (
        number.to_bytes(4, "big")
        + bytes([kind, 0, 1])
        + len(payload).to_bytes(4, "big")
        + payload
    )


def test_jbig2_generic_crop_patch_is_structural() -> None:
    page_info = bytearray(19)
    page_info[:4] = (20).to_bytes(4, "big")
    page_info[4:8] = (30).to_bytes(4, "big")
    generic = bytearray(20)
    generic[:4] = (10).to_bytes(4, "big")
    generic[4:8] = (12).to_bytes(4, "big")
    embedded = _jbig2_segment(0, 48, page_info) + _jbig2_segment(1, 38, generic)

    patched = _patch_cropped_stream(embedded, page_width=100, page_height=200, x=7, y=9)
    segments = parse_embedded_segments(patched)
    assert [segment.kind for segment in segments] == [48, 38]
    info, region = segments
    assert int.from_bytes(patched[info.data_start : info.data_start + 4], "big") == 100
    assert int.from_bytes(patched[info.data_start + 4 : info.data_start + 8], "big") == 200
    assert patched[info.data_start + 16] & 0x04
    assert int.from_bytes(patched[region.data_start + 8 : region.data_start + 12], "big") == 7
    assert int.from_bytes(patched[region.data_start + 12 : region.data_start + 16], "big") == 9
    assert patched[region.data_start + 16] == 4


def test_jbig2_xor_residual_discards_page_info_and_renumbers() -> None:
    page_info = bytearray(19)
    text_region = bytearray(25)
    page = _jbig2_segment(4, 48, page_info) + _jbig2_segment(5, 6, text_region)
    crop_info = bytearray(19)
    crop_region = bytearray(24)
    crop_region[:4] = (7).to_bytes(4, "big")
    crop_region[4:8] = (9).to_bytes(4, "big")
    crop = _jbig2_segment(0, 48, crop_info) + _jbig2_segment(1, 38, crop_region)

    hybrid = append_cropped_xor_region(page, crop, x=123, y=456)
    segments = parse_embedded_segments(hybrid)
    assert [segment.kind for segment in segments] == [48, 6, 38]
    residual = segments[-1]
    assert residual.number == 6
    assert int.from_bytes(hybrid[residual.data_start + 8 : residual.data_start + 12], "big") == 123
    assert int.from_bytes(hybrid[residual.data_start + 12 : residual.data_start + 16], "big") == 456
    assert hybrid[residual.data_start + 16] == 2


def test_jbig2_tools_must_be_paired() -> None:
    with pytest.raises(ValueError):
        OptimizationConfig(jbig2_encoder="jbig2")
    with pytest.raises(ValueError):
        OptimizationConfig(jbig2_symbol_threshold=0.99)
    with pytest.raises(ValueError):
        OptimizationConfig(jbig2_entropy_cell_size=16)


def test_bwt_preserves_histogram_and_huffman_cost() -> None:
    source = b"banana_bandana\x00\xff\x00"
    transformed = cyclic_bwt(source)
    assert sorted(transformed) == sorted(source)
    assert zero_order_huffman_cost(transformed) == zero_order_huffman_cost(source)


def test_rowwise_byte_bwt_preserves_shape_and_histogram() -> None:
    import numpy as np

    bits = np.zeros((3, 24), dtype=np.uint8)
    bits[0, 2:9] = 1
    bits[1, 12:21] = 1
    transformed = rowwise_byte_bwt(bits)
    assert transformed.shape == bits.shape
    assert np.array_equal(
        np.sort(np.packbits(bits, axis=1), axis=1),
        np.sort(np.packbits(transformed, axis=1), axis=1),
    )


def test_entropy_partition_proposes_distribution_change() -> None:
    import numpy as np

    bits = np.zeros((256, 256), dtype=np.uint8)
    bits[:, :128:8] = 1
    bits[::2, 128:] = 1
    partitions = propose_entropy_partitions(bits, cell_size=64, max_regions=3)
    assert partitions
    assert len(partitions[-1].regions) >= 2


def test_mrc_field_grid_tracks_smooth_visible_ink() -> None:
    import numpy as np

    height, width = 90, 60
    y, x = np.mgrid[:height, :width]
    image = np.stack(
        [40 + x // 6, 50 + y // 9, 60 + (x + y) // 15],
        axis=2,
    ).astype(np.uint8)
    mask = ((x % 9) < 4) & ((y % 11) < 6)
    grid, rmse = _field_grid(image, mask, 16)
    assert grid.shape == (24, 16, 3)
    assert rmse < 2.0


def test_mrc_background_repair_removes_mask_correlated_ghosts() -> None:
    import numpy as np
    from PIL import Image

    height, width = 180, 120
    background = np.full((height, width, 3), (205, 188, 174), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[40:140:12, 20:100] = 255
    background[mask > 0] -= 18
    repaired, contaminated, ghost_rmse, outside_psnr = _background_repair(
        Image.fromarray(background, "RGB"),
        Image.fromarray(mask, "L"),
        scale=3,
    )
    assert repaired.shape == (60, 40, 3)
    assert contaminated.any()
    assert ghost_rmse > 2.0
    assert outside_psnr > 40.0


def test_mrc_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        MrcConfig(jbig2_decoder="jbig2dec", foreground_rmse_limit=0)


def test_signed_distance_is_positive_inside() -> None:
    import numpy as np

    mask = np.zeros((11, 13), dtype=bool)
    mask[3:8, 4:9] = True
    distance = signed_distance(mask)
    assert distance[5, 6] > 0
    assert distance[0, 0] < 0


def test_fourier_circle_descriptor_has_rotation_stable_radial_energy() -> None:
    import numpy as np

    mask = np.zeros((19, 19), dtype=bool)
    mask[4:15, 7:11] = True
    mask[11:15, 5:14] = True
    radial, _ = fourier_circle_descriptor(mask, size=24, bins=8)
    rotated, _ = fourier_circle_descriptor(np.rot90(mask), size=24, bins=8)
    np.testing.assert_allclose(radial, rotated, atol=0.08)


def test_canonical_glyph_envelope_is_a_source_superset() -> None:
    import numpy as np

    mask = np.zeros((60, 100), dtype=bool)
    for index, x0 in enumerate((5, 27, 49, 71)):
        mask[12:27, x0 + 3 : x0 + 7] = True
        mask[23:27, x0 : x0 + 12] = True
        mask[12:16, x0 + 1 : x0 + 9] = True
        mask[16 + index % 2, x0 + 7] = True

    def identity_split(field, _passes):
        return field, np.zeros_like(field)

    result = canonicalize_mask(
        mask,
        GlyphResidualConfig(
            minimum_area=5,
            maximum_area=500,
            minimum_repetitions=3,
            descriptor_distance=1.0,
            spatial_distance=1.0,
            envelope_dilation=1,
        ),
        meyer_splitter=identity_split,
    )
    assert result.clustered_components == 4
    assert result.added_pixels > 0
    assert not np.any(mask & ~result.canonical_mask)


def test_foreground_ownership_transfer_is_exact_before_coding() -> None:
    import numpy as np

    background = np.full((24, 30, 3), 220, dtype=np.uint8)
    foreground = np.full((24, 30, 3), 45, dtype=np.uint8)
    source = np.zeros((24, 30), dtype=bool)
    source[7:18, 10:14] = True
    canonical = source.copy()
    canonical[5:20, 8:16] = True
    composite, candidate, enlarged = foreground_owned_composite(
        background, foreground, source, canonical
    )
    rebuilt = np.where(canonical[..., None], candidate, enlarged)
    assert np.array_equal(composite, rebuilt)
    assert np.all(candidate[canonical & ~source] == 220)


def test_foreground_texture_family_reuses_one_split_and_keeps_gain_one_exact() -> None:
    import numpy as np

    background = np.full((32, 32, 3), 210, dtype=np.uint8)
    candidate = background.copy()
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 10:22] = True
    candidate[mask] = 55
    calls = []

    def split(field, passes):
        calls.append((field.shape, passes))
        cartoon = np.full_like(field, float(np.mean(field)))
        return cartoon, field - cartoon

    family = denoise_foreground_texture_family(
        candidate,
        background,
        mask,
        texture_gains=(1.0, 0.5, 0.0),
        meyer_splitter=split,
    )
    assert len(calls) == 1
    assert np.array_equal(family[1.0], candidate)
    assert not np.array_equal(family[0.0][mask], candidate[mask])
