from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from backend import (
    backend_description, conv_basin_average, conv_basin_average_lines_f64,
    conv_passband_basin_lines_f64, conv_passband_compensate_lines_f64,
    conv_resize, conv_transport_resize,
    conv_evaluate_lines, conv_evaluate_profile, conv_evaluate_profile_2d,
    easu_resize, lanczos3_resize,
    linear_resize, polyphase_fir_resize,
)
from conservative import (
    admit_2d, admit_2d_reference, matched_cycle, proposal_2d,
    restrict_2x2, zero_detail_synthesis, zero_detail_synthesis_reference,
)


def sign_changes(value: np.ndarray, tolerance: float = 3.0e-6) -> int:
    sign = np.sign(np.where(np.abs(value)>tolerance,value,0.0))
    sign = sign[sign != 0]
    return int(np.sum(sign[1:] != sign[:-1])) if sign.size else 0


class BackendTest(unittest.TestCase):
    def test_native_backend_is_available(self) -> None:
        self.assertIn("native C ABI 1", backend_description())

    def test_conv_reproduces_affine_data_at_arbitrary_coordinates(self) -> None:
        source = np.linspace(-0.4,1.7,37,dtype=np.float32)
        for target in (5,19,73,148):
            expected = np.linspace(-0.4,1.7,target,dtype=np.float32)
            np.testing.assert_allclose(conv_resize(source,target),expected,atol=8e-7,rtol=0)

    def test_nested_conv_anchors_are_exact(self) -> None:
        rng=np.random.default_rng(9)
        source=rng.normal(size=65).astype(np.float32)
        for factor in (2,4,8):
            result=conv_resize(source,factor*(source.size-1)+1)
            np.testing.assert_array_equal(result[::factor],source)

    def test_native_arbitrary_site_evaluators_match_regular_synthesis(self) -> None:
        rng = np.random.default_rng(84)
        source = rng.random((17, 7), dtype=np.float32)
        positions = np.linspace(0.0, 16.0, 65, dtype=np.float32)
        arbitrary = conv_evaluate_profile(source, positions)
        regular = np.stack([
            conv_resize(source[:, lane], 65) for lane in range(source.shape[1])
        ], axis=1)
        np.testing.assert_allclose(arbitrary, regular, atol=1.2e-7, rtol=0)

        sites = np.array((0.0, 1.25, 4.5, 7.75, 10.0, 13.5, 16.0), np.float32)
        one_per_line = conv_evaluate_lines(source, sites)
        reference = np.array([
            conv_evaluate_profile(source[:, lane], sites[lane:lane + 1])[0]
            for lane in range(source.shape[1])
        ])
        np.testing.assert_array_equal(one_per_line, reference)

    def test_native_two_dimensional_profile_is_affine_exact_and_axis_equivariant(self) -> None:
        y,x=np.indices((9,11),dtype=np.float32)
        source=np.stack((.2+.07*x-.03*y,-.4+.02*x+.11*y,1.3-.05*x+.04*y),axis=-1)
        qx=np.array((0.0,.4,2.75,5.125,8.8,10.0),dtype=np.float32)
        qy=np.array((0.0,1.7,3.25,6.5,7.1,8.0),dtype=np.float32)
        result=conv_evaluate_profile_2d(source,qx,qy)
        expected=np.stack((.2+.07*qx-.03*qy,-.4+.02*qx+.11*qy,1.3-.05*qx+.04*qy),axis=-1)
        np.testing.assert_allclose(result,expected,atol=1.5e-6,rtol=0)

        rng=np.random.default_rng(95)
        field=rng.normal(size=(9,11,3)).astype(np.float32)
        forward=conv_evaluate_profile_2d(field,qx,qy)
        exchanged=conv_evaluate_profile_2d(np.swapaxes(field,0,1),qy,qx)
        np.testing.assert_allclose(forward,exchanged,atol=1.5e-6,rtol=0)

    def test_conv_does_not_add_ordered_sign_transitions(self) -> None:
        rng=np.random.default_rng(4)
        for _ in range(50):
            source=rng.normal(size=41).astype(np.float32)
            result=conv_resize(source,321)
            self.assertLessEqual(sign_changes(np.diff(result)),sign_changes(np.diff(source)))

    def test_all_backends_accept_arbitrary_image_shapes(self) -> None:
        rng=np.random.default_rng(1)
        image=rng.random((17,23,3),dtype=np.float32)
        for operation in (
            conv_resize, polyphase_fir_resize, lanczos3_resize,
            linear_resize, easu_resize,
        ):
            result=operation(image,(29,37))
            self.assertEqual(result.shape,(29,37,3))
            self.assertTrue(np.all(np.isfinite(result)))

    def test_constant_preservation(self) -> None:
        image=np.full((19,21,3),0.375,dtype=np.float32)
        for operation in (conv_resize,polyphase_fir_resize,lanczos3_resize,linear_resize,easu_resize):
            np.testing.assert_allclose(operation(image,(11,39)),0.375,atol=2e-6,rtol=0)

    def test_basin_analysis_integrates_affine_profiles_exactly(self) -> None:
        source = np.linspace(0.0, 1.0, 33, dtype=np.float32)
        result = conv_basin_average(source, 9)
        step = 32.0 / 8.0
        expected = np.arange(9, dtype=np.float64) * step / 32.0
        expected[0] = (step / 4.0) / 32.0
        expected[-1] = (32.0 - step / 4.0) / 32.0
        np.testing.assert_allclose(result, expected, atol=2.0e-7, rtol=0)

    def test_float64_basin_analysis_retains_binary64_affine_accuracy(self) -> None:
        source = np.linspace(-0.125, 1.375, 33, dtype=np.float64)[:, None]
        result = conv_basin_average_lines_f64(source, 9)[:, 0]
        step = 32.0 / 8.0
        expected = -0.125 + np.arange(9, dtype=np.float64) * step * 1.5 / 32.0
        expected[0] = -0.125 + (step / 4.0) * 1.5 / 32.0
        expected[-1] = -0.125 + (32.0 - step / 4.0) * 1.5 / 32.0
        np.testing.assert_allclose(result, expected, atol=2.0e-14, rtol=0)

    def test_passband_basin_zero_strength_is_canonical_bit_for_bit(self) -> None:
        rng = np.random.default_rng(20260901)
        source = rng.normal(size=(97, 5))
        canonical = conv_basin_average_lines_f64(source, 23)
        recovered = conv_passband_basin_lines_f64(
            source, 23, compensation=0.0
        )
        np.testing.assert_array_equal(recovered, canonical)

    def test_passband_target_operator_matches_integrated_variant(self) -> None:
        rng = np.random.default_rng(20260902)
        source = rng.normal(size=(193, 11))
        canonical = conv_basin_average_lines_f64(source, 47)
        from_existing = conv_passband_compensate_lines_f64(
            canonical, compensation=0.05
        )
        integrated = conv_passband_basin_lines_f64(
            source, 47, compensation=0.05
        )
        np.testing.assert_array_equal(from_existing, integrated)

    def test_passband_basin_preserves_affine_data_and_line_moments(self) -> None:
        source = np.stack([
            np.linspace(-0.125, 1.375, 97, dtype=np.float64),
            np.linspace(2.0, -3.0, 97, dtype=np.float64),
        ], axis=1)
        canonical = conv_basin_average_lines_f64(source, 23)
        recovered = conv_passband_basin_lines_f64(
            source, 23, compensation=0.25
        )
        np.testing.assert_allclose(recovered, canonical, atol=3.0e-14, rtol=0)

        rng = np.random.default_rng(17)
        source = rng.normal(size=(97, 7))
        canonical = conv_basin_average_lines_f64(source, 23)
        recovered = conv_passband_basin_lines_f64(
            source, 23, compensation=0.25
        )
        coordinate = np.arange(23, dtype=np.float64)[:, None]
        np.testing.assert_allclose(
            recovered.sum(axis=0), canonical.sum(axis=0), atol=2.0e-14, rtol=0
        )
        np.testing.assert_allclose(
            (coordinate * recovered).sum(axis=0),
            (coordinate * canonical).sum(axis=0),
            atol=3.0e-13,
            rtol=0,
        )

    def test_passband_basin_recovers_target_lattice_gradient_energy(self) -> None:
        x = np.linspace(-1.0, 1.0, 513, dtype=np.float64)
        source = (
            0.35 * x
            + 0.25 * np.tanh(22.0 * (x + 0.31))
            - 0.20 * np.tanh(31.0 * (x - 0.18))
            + 0.025 * np.sin(39.0 * np.pi * x)
        )[:, None]
        canonical = conv_basin_average_lines_f64(source, 81)[:, 0]
        recovered = conv_passband_basin_lines_f64(
            source, 81, compensation=0.20
        )[:, 0]
        canonical_energy = float(np.square(np.diff(canonical)).sum())
        recovered_energy = float(np.square(np.diff(recovered)).sum())
        self.assertGreater(recovered_energy, canonical_energy)
        self.assertLess(float(np.max(np.abs(recovered - canonical))), 0.02)

    def test_transport_resize_uses_basin_analysis_only_on_reduction(self) -> None:
        rng = np.random.default_rng(29)
        source = rng.random((41, 53, 3), dtype=np.float32)
        reduced = conv_transport_resize(source, (17, 19))
        np.testing.assert_array_equal(
            reduced, conv_basin_average(source, (17, 19))
        )
        enlarged = conv_transport_resize(reduced, source.shape[:2])
        np.testing.assert_array_equal(
            enlarged, conv_resize(reduced, source.shape[:2])
        )

    def test_native_moment_admission_matches_the_reference_exactly(self) -> None:
        rng = np.random.default_rng(20260827)
        coarse = rng.normal(size=(17, 19, 3)).astype(np.float32)
        proposal = proposal_2d(coarse)
        native = admit_2d(coarse, proposal)
        reference = admit_2d_reference(coarse, proposal)
        for name in ("horizontal", "vertical", "mixed"):
            np.testing.assert_array_equal(
                getattr(native, name), getattr(reference, name)
            )

    def test_conservative_support_preserves_every_parent_mean(self) -> None:
        rng = np.random.default_rng(8)
        coarse = rng.random((17, 21, 3), dtype=np.float32)
        fine = zero_detail_synthesis(coarse)
        np.testing.assert_allclose(
            restrict_2x2(fine), coarse, atol=3.0e-7, rtol=0
        )

    def test_fused_four_child_atlas_matches_decomposed_native_oracle(self) -> None:
        rng = np.random.default_rng(20260901)
        for shape in ((5, 7), (17, 19), (17, 19, 3)):
            coarse = rng.normal(size=shape).astype(np.float32)
            fused = zero_detail_synthesis(coarse)
            oracle = zero_detail_synthesis_reference(coarse)
            np.testing.assert_array_equal(fused, oracle)

    def test_fused_four_child_atlas_preserves_affine_planes(self) -> None:
        y, x = np.mgrid[:13, :17].astype(np.float32)
        coarse = np.stack(
            (0.25 + 0.03*x - 0.02*y, -0.4 + 0.01*x + 0.04*y),
            axis=-1,
        )
        fused = zero_detail_synthesis(coarse)
        np.testing.assert_array_equal(
            fused, zero_detail_synthesis_reference(coarse)
        )
        np.testing.assert_allclose(
            restrict_2x2(fused), coarse, atol=1.2e-7, rtol=0
        )

    def test_fused_four_child_atlas_has_no_factorwise_sign_surplus(self) -> None:
        rng = np.random.default_rng(20260902)
        for coarse in (
            rng.normal(size=(19, 23, 3)).astype(np.float32),
            np.broadcast_to(
                np.cos(np.linspace(0.0, 3.0*np.pi, 23, dtype=np.float32)),
                (19, 23),
            ).copy(),
        ):
            fine = zero_detail_synthesis(coarse)
            scalar = coarse.ndim == 2
            components = 1 if scalar else coarse.shape[-1]
            for component in range(components):
                source = coarse if scalar else coarse[..., component]
                result = fine if scalar else fine[..., component]
                horizontal = np.diff(result, axis=1)
                for row in range(source.shape[0]):
                    limit = sign_changes(np.diff(source[row]))
                    self.assertLessEqual(
                        sign_changes(horizontal[2*row]), limit
                    )
                    self.assertLessEqual(
                        sign_changes(horizontal[2*row+1]), limit
                    )
                vertical = np.diff(result, axis=0)
                for column in range(source.shape[1]):
                    limit = sign_changes(np.diff(source[:, column]))
                    self.assertLessEqual(
                        sign_changes(vertical[:, 2*column]), limit
                    )
                    self.assertLessEqual(
                        sign_changes(vertical[:, 2*column+1]), limit
                    )

    def test_fused_atlas_retains_constant_transverse_lines_across_simd_tail(self) -> None:
        line = np.cos(np.linspace(0.0, 3.15*np.pi, 23, dtype=np.float32))
        coarse = np.broadcast_to(line, (17, 23)).copy()
        fine = zero_detail_synthesis(zero_detail_synthesis(coarse))
        np.testing.assert_array_equal(
            fine, np.broadcast_to(fine[:1], fine.shape)
        )

    def test_conservative_matched_cycle_uses_exact_dyadic_shapes(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.random((80, 96, 3), dtype=np.float32)
        coarse, returned = matched_cycle(source, (20, 24))
        self.assertEqual(coarse.shape, (20, 24, 3))
        self.assertEqual(returned.shape, source.shape)
        self.assertTrue(np.all(np.isfinite(returned)))

    def test_python_sources_do_not_import_the_parent_repository(self) -> None:
        root=Path(__file__).resolve().parent
        forbidden={"experiments","denoiser","bfft","src"}
        for path in root.glob("*.py"):
            # Launchers deliberately connect the copy-isolated numerical core
            # to repository-level comparison methods.  The reusable modules
            # themselves remain independent.
            if path.name.startswith("run_"):
                continue
            tree=ast.parse(path.read_text(),filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    names={alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node,ast.ImportFrom) and node.module:
                    names={node.module.split(".")[0]}
                else:
                    continue
                self.assertTrue(names.isdisjoint(forbidden),f"{path.name} imports {names & forbidden}")


if __name__ == "__main__":
    unittest.main()
