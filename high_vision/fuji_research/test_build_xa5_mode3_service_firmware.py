#!/usr/bin/env python3

import unittest

import build_xa5_mode3_service_firmware as builder


class Mode3ServiceFirmwareTests(unittest.TestCase):
    def test_constants_encode_expected_arm_instructions(self) -> None:
        self.assertEqual(
            int.from_bytes(builder.STOCK_INSTRUCTION, "little"),
            0x13500011,
        )
        self.assertEqual(
            int.from_bytes(builder.PATCHED_INSTRUCTION, "little"),
            0x13500003,
        )

    def test_outer_sum_delta_matches_literal_delta(self) -> None:
        self.assertEqual(
            builder.PATCHED_OUTER_SUM - builder.STOCK_OUTER_SUM,
            builder.PATCHED_LITERAL - builder.STOCK_LITERAL,
        )

    def test_sum_replacement_has_fixed_width(self) -> None:
        self.assertEqual(
            len(f"SUM={builder.STOCK_OUTER_SUM}"),
            len(f"SUM={builder.PATCHED_OUTER_SUM}"),
        )


if __name__ == "__main__":
    unittest.main()
