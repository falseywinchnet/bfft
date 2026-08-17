from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.manual_jpeg_optimizer.jpegli_bridge import (
    ownership_dead_zone_field,
    write_jldz,
)


class JpegliBridgeTest(unittest.TestCase):
    def test_field_is_nonnegative_and_preserves_dc(self) -> None:
        yy, xx = np.mgrid[:16, :16]
        rgb = np.stack((xx * 8, yy * 8, (xx + yy) * 4), axis=-1).astype(np.float64)
        labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
        field = ownership_dead_zone_field(rgb, labels, quality=80, strength=0.3)
        self.assertEqual(field.shape, (3, 2, 2, 64))
        self.assertTrue(np.all(field >= 0))
        self.assertTrue(np.all(field[..., 0] == 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.jldz"
            write_jldz(path, field)
            self.assertEqual(path.read_bytes()[:4], b"JLDZ")
            self.assertEqual(path.stat().st_size, 16 + field.nbytes)


if __name__ == "__main__":
    unittest.main()
