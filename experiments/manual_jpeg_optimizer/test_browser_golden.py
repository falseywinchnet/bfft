from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.manual_jpeg_optimizer.browser_golden import DEFAULT_FIXTURE, build_fixture, verify_fixture


class BrowserStructuralGoldenTests(unittest.TestCase):
    def test_frozen_fixture_matches_current_authoritative_core(self):
        self.assertTrue(DEFAULT_FIXTURE.is_file())
        verify_fixture(DEFAULT_FIXTURE)

    def test_fixture_declares_the_remaining_native_boundary(self):
        fixture = build_fixture()
        self.assertIn("connected_ownership_regions", fixture["scope"])
        self.assertIn("regional_chroma_covariance_projection", fixture["scope"])
        self.assertIn("jpegli_dead_zone_and_trellis", fixture["excluded_scope"])


if __name__ == "__main__":
    unittest.main()
