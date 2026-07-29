import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from build_readonly_dump_card import build, service_packet


class ReadOnlyDumpCardTests(unittest.TestCase):
    def test_packet_encodes_only_dump_pair(self):
        self.assertEqual(
            service_packet(),
            bytes.fromhex("1B44000000464C040000"),
        )
        with self.assertRaises(ValueError):
            service_packet(326, 1101)
        with self.assertRaises(ValueError):
            service_packet(325, 1100)

    def test_builds_personalized_card_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build(root, "FF129301X-A5", datetime(2026, 7, 28, 6, 0, 0))
            directory = root / "ADJ" / "C713A"
            self.assertTrue((root / "DCIM" / "100_FUJI").is_dir())
            self.assertEqual((root / "C713A.ADJ").read_bytes(), b"")
            self.assertEqual((root / "ADJ" / "C713A.ADJ").read_bytes(), b"")
            self.assertEqual((directory / "INPUT.DAT").read_bytes(), b"FF129301X-A5")
            self.assertEqual((directory / "CARDVER.DAT").read_bytes(), b"C713A")
            self.assertEqual(
                (directory / "DATE.DAT").read_bytes(), b"2026/07/28 06:00:00"
            )
            self.assertEqual(
                (directory / "SCRIPT.DAT").read_bytes(),
                b"#P000-S000\r\n0000:1B44000000464C040000\r\n",
            )


if __name__ == "__main__":
    unittest.main()
