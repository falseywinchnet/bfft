from __future__ import annotations

import unittest

import numpy as np

from pdf_optimizer.glyph_residual import (
    GlyphResidualConfig,
    canonicalize_mask,
    canonicalize_masks_global_core,
    canonicalize_masks_global,
    decode_core_atlas,
    decode_core_corrections,
    decode_refined_core_atlas,
    decode_fixed_refinement_atlas,
    decode_split_refinement_atlas,
    denoise_foreground_texture_family,
    foreground_owned_composite,
    fourier_circle_descriptor,
    signed_distance,
    serialize_core_atlas,
    serialize_core_corrections,
    serialize_fixed_refinement_atlas,
    serialize_split_refinement_atlas,
    subset_core_atlas,
    transcode_fixed_refinement_to_split,
)


class GlyphResidualTests(unittest.TestCase):
    def test_signed_distance_is_positive_inside(self):
        mask = np.zeros((11, 13), dtype=bool)
        mask[3:8, 4:9] = True
        distance = signed_distance(mask)
        self.assertGreater(distance[5, 6], 0)
        self.assertLess(distance[0, 0], 0)

    def test_fourier_circle_radial_energy_is_rotation_stable(self):
        mask = np.zeros((19, 19), dtype=bool)
        mask[4:15, 7:11] = True
        mask[11:15, 5:14] = True
        radial, _ = fourier_circle_descriptor(mask, size=24, bins=8)
        rotated, _ = fourier_circle_descriptor(np.rot90(mask), size=24, bins=8)
        np.testing.assert_allclose(radial, rotated, atol=0.08)

    def test_canonical_envelope_is_source_superset(self):
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
        self.assertEqual(result.clustered_components, 4)
        self.assertGreater(result.added_pixels, 0)
        self.assertFalse(np.any(mask & ~result.canonical_mask))

    def test_foreground_ownership_transfer_is_exact(self):
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
        self.assertTrue(np.array_equal(composite, rebuilt))
        self.assertTrue(np.all(candidate[canonical & ~source] == 220))

    def test_global_envelopes_are_source_supersets(self):
        pages = []
        for offset in (0, 1):
            mask = np.zeros((48, 72), dtype=bool)
            for x0 in (6, 30, 54):
                mask[10:25, x0 : x0 + 4] = True
                mask[21:25, x0 - 2 : x0 + 8] = True
                mask[12 + offset, x0 + 5] = True
            pages.append(mask)

        def identity_split(field, _passes):
            return field, np.zeros_like(field)

        result = canonicalize_masks_global(
            pages,
            GlyphResidualConfig(
                minimum_area=5,
                maximum_area=500,
                minimum_repetitions=4,
                descriptor_distance=1.0,
                spatial_distance=1.0,
            ),
            meyer_splitter=identity_split,
        )
        self.assertGreaterEqual(result.clustered_components, 6)
        for source, canonical in zip(result.source_masks, result.canonical_masks):
            self.assertFalse(np.any(source & ~canonical))

    def test_global_core_atlas_and_residual_are_exact(self):
        pages = []
        for offset in (0, 1):
            mask = np.zeros((48, 72), dtype=bool)
            for index, x0 in enumerate((6, 30, 54)):
                mask[10:25, x0 : x0 + 4] = True
                mask[21:25, x0 - 2 : x0 + 8] = True
                mask[12 + (offset + index) % 2, x0 + 5] = True
            pages.append(mask)

        def identity_split(field, _passes):
            return field, np.zeros_like(field)

        result = canonicalize_masks_global_core(
            pages,
            GlyphResidualConfig(
                minimum_area=5,
                maximum_area=500,
                minimum_repetitions=4,
                descriptor_distance=1.0,
                spatial_distance=1.0,
            ),
            meyer_splitter=identity_split,
        )
        self.assertGreaterEqual(len(result.glyphs), 1)
        self.assertGreaterEqual(len(result.occurrences), 6)
        for source, core, residual in zip(
            result.source_masks, result.core_masks, result.residual_masks
        ):
            self.assertFalse(np.any(core & ~source))
            self.assertTrue(np.array_equal(source, core | residual))

        encoded = serialize_core_atlas(result)
        decoded = decode_core_atlas(encoded)
        for expected, actual in zip(result.core_masks, decoded):
            self.assertTrue(np.array_equal(expected, actual))
        correction_data = serialize_core_corrections(result)
        correction_masks = decode_core_corrections(
            correction_data, (mask.shape for mask in result.source_masks)
        )
        for source, core, correction, remainder in zip(
            result.source_masks,
            decoded,
            correction_masks,
            result.remainder_masks,
        ):
            self.assertTrue(np.array_equal(source, core | correction | remainder))

        median = canonicalize_masks_global_core(
            pages,
            GlyphResidualConfig(
                minimum_area=5,
                maximum_area=500,
                minimum_repetitions=4,
                descriptor_distance=1.0,
                spatial_distance=1.0,
            ),
            meyer_splitter=identity_split,
            core_mode="median",
        )
        median_atlas = serialize_core_atlas(median)
        median_refined = decode_refined_core_atlas(
            median_atlas, serialize_core_corrections(median)
        )
        for source, refined, remainder in zip(
            median.source_masks,
            median_refined,
            median.remainder_masks,
        ):
            self.assertTrue(np.array_equal(source, refined | remainder))
        fixed_refined = decode_fixed_refinement_atlas(
            serialize_fixed_refinement_atlas(median)
        )
        for source, refined, remainder in zip(
            median.source_masks, fixed_refined, median.remainder_masks
        ):
            self.assertTrue(np.array_equal(source, refined | remainder))
        split_refined = decode_split_refinement_atlas(
            serialize_split_refinement_atlas(median)
        )
        for source, refined, remainder in zip(
            median.source_masks, split_refined, median.remainder_masks
        ):
            self.assertTrue(np.array_equal(source, refined | remainder))
        transcoded_refined = decode_split_refinement_atlas(
            transcode_fixed_refinement_to_split(
                serialize_fixed_refinement_atlas(median)
            )
        )
        for expected, actual in zip(split_refined, transcoded_refined):
            self.assertTrue(np.array_equal(expected, actual))
        subset = subset_core_atlas(median, (0,))
        subset_refined = decode_refined_core_atlas(
            serialize_core_atlas(subset), serialize_core_corrections(subset)
        )
        for source, refined, remainder in zip(
            subset.source_masks, subset_refined, subset.remainder_masks
        ):
            self.assertTrue(np.array_equal(source, refined | remainder))

    def test_texture_family_reuses_one_split(self):
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
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.array_equal(family[1.0], candidate))
        self.assertFalse(np.array_equal(family[0.0][mask], candidate[mask]))


if __name__ == "__main__":
    unittest.main()
