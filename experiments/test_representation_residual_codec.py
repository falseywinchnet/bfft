"""Exactness tests for the representation-plus-correction codec experiment."""

from __future__ import annotations

import numpy as np

from experiments.representation_residual_codec import (
    apply_modular_difference,
    benchmark_pair,
    cell_residual_decode,
    cell_residual_encode,
    compact_geometry_decode,
    compact_geometry_encode,
    dct_decode,
    dct_encode,
    finite_predictor_decode,
    finite_predictor_encode,
    modular_difference,
    png_predictive_decode,
    png_predictive_encode,
    signed_finite_decode,
    signed_finite_encode,
    signed_png_decode,
    signed_png_encode,
    standard_png_decode,
    standard_png_encode,
)


def fixture(height: int = 37, width: int = 43) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.stack(
        (
            (7 * x + 3 * y) & 255,
            (x * y + 19) & 255,
            ((x > width // 2) * 173 + 5 * y) & 255,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_modular_correction_is_exact() -> None:
    source = fixture()
    base = np.roll(source, (3, -5), axis=(0, 1))
    correction = modular_difference(source, base)
    np.testing.assert_array_equal(
        apply_modular_difference(base, correction), source
    )


def test_png_predictive_roundtrip() -> None:
    source = fixture()
    np.testing.assert_array_equal(
        png_predictive_decode(png_predictive_encode(source)), source
    )


def test_standard_png_roundtrip() -> None:
    source = fixture()
    np.testing.assert_array_equal(
        standard_png_decode(standard_png_encode(source)), source
    )


def test_finite_predictor_roundtrip() -> None:
    source = fixture()
    np.testing.assert_array_equal(
        finite_predictor_decode(
            finite_predictor_encode(source, tile_size=11)
        ),
        source,
    )


def test_signed_correction_packets_are_exact() -> None:
    source = fixture()
    base = np.roll(source, (2, 7), axis=(0, 1))
    for encode, decode in (
        (signed_png_encode, signed_png_decode),
        (signed_finite_encode, signed_finite_decode),
    ):
        packet = encode(source, base)
        np.testing.assert_array_equal(decode(base, packet), source)


def test_cell_ordered_residual_is_exact() -> None:
    source = fixture()
    base = np.roll(source, (2, 7), axis=(0, 1))
    y, x = np.mgrid[:source.shape[0], :source.shape[1]]
    labels = ((x // 9) + 5 * (y // 8)).astype(np.int32)
    packet = cell_residual_encode(source, base, labels)
    np.testing.assert_array_equal(
        cell_residual_decode(base, labels, packet), source
    )


def test_compact_geometry_packet_and_correction_are_exact() -> None:
    source = fixture()
    y, x = np.mgrid[:source.shape[0], :source.shape[1]]
    labels = ((x // 9) + 5 * (y // 8)).astype(np.int32)
    cells = int(labels.max()) + 1
    centers = np.zeros((cells, 2), dtype=np.float64)
    for cell in range(cells):
        selected = labels == cell
        centers[cell, 0] = np.mean(x[selected]) / (source.shape[1] - 1)
        centers[cell, 1] = np.mean(y[selected]) / (source.shape[0] - 1)
    for selection in ("area", "mass", "farthest"):
        geometry = compact_geometry_encode(
            source,
            labels,
            centers,
            site_count=min(12, cells),
            selection=selection,
        )
        base, compact_labels = compact_geometry_decode(geometry)
        correction = cell_residual_encode(source, base, compact_labels)
        np.testing.assert_array_equal(
            cell_residual_decode(base, compact_labels, correction), source
        )


def test_dct_packet_has_expected_shape_and_range() -> None:
    source = fixture()
    decoded = dct_decode(dct_encode(source, quality=70))
    assert decoded.shape == source.shape
    assert decoded.dtype == np.uint8


def test_every_benchmark_total_is_exact() -> None:
    source = fixture()
    support = np.clip(
        np.rint(source.astype(np.float64) * 0.85 + 16.0), 0, 255
    ).astype(np.uint8)
    rows, break_even = benchmark_pair(source, support, qualities=(55,))
    assert rows
    assert all(row["exact"] for row in rows)
    assert "geometry_budget_vs_source_png" in break_even
