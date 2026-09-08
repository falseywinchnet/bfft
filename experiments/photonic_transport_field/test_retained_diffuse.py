from __future__ import annotations

import unittest

import numpy as np

from .native_retained import native_available
from .retained_diffuse import NativeDiffuseTransportPlan
from .scenes import facing_patch_grids
from .transport import compile_hierarchical_field, solve_transport


@unittest.skipUnless(native_available(), "native retained-rank library is not built")
class NativeDiffuseTransportTests(unittest.TestCase):
    def test_native_march_consumes_geometric_blocks_without_expansion(self) -> None:
        patches, _ = facing_patch_grids(side=8, separation=8.0, extent=2.0)
        field = compile_hierarchical_field(patches, admissibility=0.20)
        self.assertTrue(any(
            block.source.size > 1 and block.receiver.size > 1
            for block in field.blocks
        ))
        emission = np.stack([patch.emission for patch in patches])
        albedo = np.stack([patch.albedo for patch in patches])
        reference = solve_transport(
            field,
            emission,
            albedo,
            relative_tolerance=1.0e-13,
            absolute_tolerance=1.0e-15,
        )
        with NativeDiffuseTransportPlan(field) as plan:
            self.assertEqual(
                plan.native_plan.stored_coefficient_count,
                field.stored_coefficients,
            )
            actual = plan.solve(
                emission,
                albedo,
                relative_tolerance=1.0e-13,
                absolute_tolerance=1.0e-15,
            )
        self.assertTrue(np.allclose(
            actual.outgoing, reference.outgoing, rtol=2.0e-13, atol=2.0e-15
        ))
        self.assertTrue(np.allclose(
            actual.incident, reference.incident, rtol=2.0e-13, atol=2.0e-15
        ))
        self.assertLess(
            float(np.max(np.abs(actual.conservation_error))), 1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
