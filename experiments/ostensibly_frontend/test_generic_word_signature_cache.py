from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from .generic_word_signature_cache import GenericWordSignatureCache


class GenericWordSignatureCacheTests(unittest.TestCase):
    def test_quantized_round_trip_and_configuration_gate(self) -> None:
        cache = GenericWordSignatureCache("abc", 2, 8, 4)
        surface = np.linspace(0.0, 1.0, 32).reshape(8, 4)
        mass = np.full(8, 0.125)
        stored = cache.put(
            ("A", "B"), ((surface, mass), (surface + 0.1, mass))
        )
        self.assertEqual(stored[0][0].dtype, np.float16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            cache.save(path)
            loaded = GenericWordSignatureCache.load(path)
        loaded.require_configuration("abc", 2, 8, 4, "float16")
        self.assertTrue(
            np.array_equal(loaded.get(("A", "B"))[1][0], stored[1][0])
        )
        with self.assertRaises(ValueError):
            loaded.require_configuration("other", 2, 8, 4, "float16")
        with self.assertRaises(ValueError):
            loaded.require_configuration(
                "abc", 2, 8, 4, "float16", "contextual"
            )


if __name__ == "__main__":
    unittest.main()
