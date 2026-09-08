import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.ostensibly_frontend.generic_pronunciation_catalog import (
    GenericPronunciationCatalog,
)


class GenericPronunciationCatalogTests(unittest.TestCase):
    def test_rank_is_label_blind_and_round_trips(self) -> None:
        surfaces = np.zeros((2, 2, 8, 4), dtype=np.float16)
        masses = np.zeros((2, 2, 8), dtype=np.float16)
        surfaces[1] = 1.0
        catalog = GenericPronunciationCatalog(
            phone_keys=np.asarray(("OW K EY", "S T AH")),
            lengths=np.asarray((3, 3), dtype=np.int16),
            surfaces=surfaces,
            masses=masses,
            context_policy="unconditional",
        )
        query = (np.full((8, 4), 0.9), np.zeros(8))
        self.assertEqual(catalog.rank(query, (3,), count=1)[0][0], ("S", "T", "AH"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.npz"
            catalog.save(path)
            loaded = GenericPronunciationCatalog.load(path)
            self.assertEqual(loaded.rank(query, (3,), count=1), catalog.rank(query, (3,), count=1))

    def test_rank_filters_lengths(self) -> None:
        catalog = GenericPronunciationCatalog(
            phone_keys=np.asarray(("OW K EY", "OW")),
            lengths=np.asarray((3, 1), dtype=np.int16),
            surfaces=np.zeros((2, 1, 8, 4), dtype=np.float32),
            masses=np.zeros((2, 1, 8), dtype=np.float32),
            context_policy="unconditional",
        )
        query = (np.zeros((8, 4)), np.zeros(8))
        self.assertEqual(catalog.rank(query, (1,), count=4)[0][0], ("OW",))


if __name__ == "__main__":
    unittest.main()
