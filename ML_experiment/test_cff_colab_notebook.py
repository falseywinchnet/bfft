"""Regression and reduced-runtime checks for the standalone CFF notebook."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from ML_experiment.cff import ContinuousFrameFlow as StandaloneCFF
from ML_experiment.continuous_frame_flow import ContinuousFrameFlow as ReferenceCFF


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "CFF_8_turn_spiral_colab.ipynb"


class CFFColabTests(unittest.TestCase):
    def test_extracted_cff_matches_reference(self):
        torch.manual_seed(123)
        standalone = StandaloneCFF(2, 2, width=38)
        torch.manual_seed(123)
        reference = ReferenceCFF(2, 2, width=38)
        sample = torch.randn(9, 2)

        self.assertEqual(
            sum(parameter.numel() for parameter in standalone.parameters()),
            sum(parameter.numel() for parameter in reference.parameters()),
        )
        torch.testing.assert_close(standalone(sample), reference(sample))

    def test_every_code_cell_compiles(self):
        notebook = json.loads(NOTEBOOK.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{index}", "exec")

    def test_reduced_training_workflow_executes(self):
        notebook = json.loads(NOTEBOOK.read_text())
        namespace = {"__name__": "cff_colab_smoke_test"}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # The Mini's lean Torch runtime intentionally has no matplotlib;
            # plotting cells are covered by the compilation test above.
            source = source.replace("import matplotlib.pyplot as plt\n", "")
            if "plt." in source:
                continue
            source = source.replace("STEPS = 2_000", "STEPS = 2")
            source = source.replace("EVAL_EVERY = 50", "EVAL_EVERY = 1")
            source = source.replace("GRID = 181", "GRID = 31")
            exec(compile(source, "cff_colab_smoke", "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
