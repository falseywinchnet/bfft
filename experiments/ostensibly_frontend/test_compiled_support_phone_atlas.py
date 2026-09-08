from __future__ import annotations

import unittest

import numpy as np

from .compiled_support_phone_atlas import compile_support_phone_atlas


class CompiledSupportPhoneAtlasTests(unittest.TestCase):
    def test_ranks_matching_local_support_first(self) -> None:
        t = np.linspace(0.1, 0.9, 200)
        diagonal = np.column_stack((t, t, np.ones_like(t)))
        opposite = np.column_stack((t, 1.0 - t, np.ones_like(t)))
        atlas = compile_support_phone_atlas(
            (("A", "a", diagonal), ("B", "b", opposite)),
            bins=32,
            minimum_cell_count=1,
        )
        self.assertEqual(atlas.rank(diagonal)[0]["phone"], "A")


if __name__ == "__main__":
    unittest.main()
