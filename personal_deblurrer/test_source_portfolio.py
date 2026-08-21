from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from .source_portfolio import research_source_portfolio
from .workbench import V3_SKIMAGE_PORTFOLIO


class ResearchSourcePortfolioTests(unittest.TestCase):
    def test_materialized_v3_sources_are_complete_and_method_agnostic(self) -> None:
        root = (
            Path(__file__).resolve().parent / "source_assets" / "v3_skimage")
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["method_inheritance"], "none")
        self.assertEqual(manifest["count"], len(V3_SKIMAGE_PORTFOLIO))
        self.assertEqual(
            [record["name"] for record in manifest["records"]],
            list(V3_SKIMAGE_PORTFOLIO),
        )
        for record in manifest["records"]:
            path = root / record["file"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["sha256"],
            )
        portfolio = research_source_portfolio(32)
        self.assertEqual(len(portfolio), 6 + len(V3_SKIMAGE_PORTFOLIO))
        self.assertEqual(
            sum(name.startswith("v3_skimage/") for name in portfolio),
            len(V3_SKIMAGE_PORTFOLIO),
        )
        for image in portfolio.values():
            self.assertEqual(image.shape[:2], (32, 32))


if __name__ == "__main__":
    unittest.main()
