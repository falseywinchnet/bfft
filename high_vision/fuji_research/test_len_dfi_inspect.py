import struct
import unittest

from len_dfi_inspect import (
    decompress_lzss,
    find_dfi_candidates,
    find_outer_records,
)


class LenDfiInspectTests(unittest.TestCase):
    def test_finds_outer_record_and_rejects_embedded_length_string(self):
        data = (
            b"LENGTH=32 C713A VER=C713A-vA4 DVR=2.03 SUM=12 IPL\r\n"
            + b" " * 80
            + b"xLENGTH=999 BAD VER=nope\nnot-a-padded-header"
        )
        records = find_outer_records(data)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].length, 32)
        self.assertEqual(records[0].dvr, "2.03")
        self.assertEqual(records[0].checksum, 12)
        self.assertEqual(records[0].tags, ("IPL",))

    def test_scores_arm_vector_at_dfi_plus_0x200(self):
        header = b"@DFI" + struct.pack("<I", 1) + b"\0" * (0x200 - 8)
        vectors = struct.pack("<8I", *([0xE59FF018] * 8))
        candidates = find_dfi_candidates(header + vectors + b"\0" * 64)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].version_le, 1)
        self.assertEqual(candidates[0].data_offset, 0x200)
        self.assertEqual(candidates[0].arm_vector_score, 8)

    def test_decompresses_literals_and_ring_copy(self):
        # ff: 8 literals; ef: 4 literals, an 8-byte copy from 0xfee, 3 literals.
        packed = (
            b"\xffABCDEFGH"
            b"\xefIJKL"
            b"\xee\xf5"
            b"MNO"
        )
        self.assertEqual(
            decompress_lzss(packed),
            b"ABCDEFGHIJKLABCDEFGHMNO",
        )


if __name__ == "__main__":
    unittest.main()
