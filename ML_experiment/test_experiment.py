from __future__ import annotations

import unittest
import sys
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ML_experiment.models import parameter_count
from ML_experiment.optimizers import MuonWithAuxAdamW, zeropower_newton_schulz5
from ML_experiment.continuous_frame_flow import ContinuousFrameFlow
from ML_experiment.run_benchmark import chart_loss, secant_loss
from ML_experiment.tasks import TASK_BUILDERS
from ML_experiment.variants import JET_VARIANTS, RADIAL_ENERGY_VARIANTS, RADIAL_FRAME_VARIANTS, RADIAL_INTEGRAL_VARIANTS, RADIAL_LAB_VARIANTS, RADIAL_PARALLEL_VARIANTS, RADIAL_SHELL_VARIANTS, TRANSPORT_VARIANTS, VARIANTS, make_variant


class SupersetTests(unittest.TestCase):
    def test_catalog_is_the_union_superset(self):
        required = {"spiral", "checkerboard", "two_moons", "pinwheel", "xor_quads", "sinusoid_bounds",
                    "radial_stripes", "swiss_cheese", "lorenz_lobes", "periodic_wells", "ripple", "ring_sdf",
                    "complex_spiral_3d", "periodic_nd", "hyperchecker", "multiscale_1d", "chirp_1d",
                    "poly_drifted_chirp_1d",
                    "localized_steps_1d", "fourier_mix_1d", "nd_spiral_low_rank", "nd_spiral_high_rank",
                    "hypercube_checker"}
        self.assertEqual(required, set(TASK_BUILDERS))

    def test_exact_parameter_budget(self):
        for dimensions in ((1, 1), (2, 2), (16, 2)):
            counts = {parameter_count(make_variant(name, *dimensions, 16)) for name in VARIANTS}
            self.assertEqual(len(counts), 1)
            jet_counts = {parameter_count(make_variant(name, *dimensions, 16)) for name in JET_VARIANTS}
            self.assertEqual(len(jet_counts), 1)

    def test_context_variants_change_function_without_parameters(self):
        x = torch.randn(11, 3); torch.manual_seed(7); base = make_variant("self_context", 3, 2, 16)
        for name in ("self_context_hard", "self_context_iterated", "self_context_uncertainty"):
            torch.manual_seed(7); other = make_variant(name, 3, 2, 16)
            self.assertEqual(parameter_count(base), parameter_count(other)); self.assertFalse(torch.allclose(base(x), other(x)))

    def test_relational_objectives_backpropagate(self):
        for name, objective in (("self_context_secant", secant_loss), ("self_context_chart", chart_loss)):
            model = make_variant(name, 2, 2, 16); x = torch.randn(32, 2); y = torch.randint(2, (32,)); g = torch.Generator().manual_seed(3)
            loss = objective(model, x, y, "classification", g, x.std(0, keepdim=True) * .055) if name.endswith("chart") else objective(model, x, y, "classification", g)
            loss.backward(); self.assertTrue(torch.isfinite(loss)); self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_curvature_is_state_without_a_parameter_or_loss_change(self):
        x = torch.randn(13, 3)
        torch.manual_seed(17); baseline = make_variant("self_context", 3, 2, 16)
        for name in JET_VARIANTS[2:]:
            torch.manual_seed(17); model = make_variant(name, 3, 2, 16)
            output = model(x); output.square().mean().backward()
            self.assertEqual(parameter_count(baseline), parameter_count(model))
            self.assertFalse(torch.allclose(baseline(x), output))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_transport_variants_are_parameter_matched_and_finite(self):
        x = torch.randn(13, 3)
        torch.manual_seed(23); baseline = make_variant("self_context", 3, 2, 16)
        for name in TRANSPORT_VARIANTS:
            torch.manual_seed(23); model = make_variant(name, 3, 2, 16)
            output = model(x); output.square().mean().backward()
            self.assertEqual(parameter_count(baseline), parameter_count(model))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_detached_curvature_preserves_forward_state_but_changes_backward_route(self):
        x = torch.randn(17, 3); target = torch.randn(17, 2)
        torch.manual_seed(29); full = make_variant("self_context_jet_curvature_context", 3, 2, 16)
        torch.manual_seed(29); detached = make_variant("self_context_jet_curvature_detached", 3, 2, 16)
        full_output, detached_output = full(x), detached(x)
        self.assertTrue(torch.allclose(full_output, detached_output, atol=1e-6, rtol=1e-5))
        (full_output - target).square().mean().backward()
        (detached_output - target).square().mean().backward()
        full_gradient = full.up.metric.weight.grad
        detached_gradient = detached.up.metric.weight.grad
        self.assertFalse(torch.allclose(full_gradient, detached_gradient))

    def test_bounded_curvature_reports_subunit_authority(self):
        model = make_variant("self_context_jet_curvature_bounded", 3, 2, 16)
        model.set_diagnostics_enabled(True)
        _ = model(torch.randn(19, 3))
        for layer in model.diagnostics().values():
            authority = layer["curvature_authority"]
            self.assertTrue(((authority >= 0) & (authority <= 1)).all())

    def test_radial_lab_variants_separate_selection_from_values_without_parameters(self):
        x = torch.randn(17, 3)
        counts = set()
        outputs = []
        for name in RADIAL_LAB_VARIANTS:
            torch.manual_seed(31); model = make_variant(name, 3, 2, 16)
            outputs.append(model(x)); counts.add(parameter_count(model))
        self.assertEqual(len(counts), 1)
        self.assertFalse(torch.allclose(outputs[0], outputs[1]))
        self.assertFalse(torch.allclose(outputs[2], outputs[3]))

    def test_eikonal_shell_consensus_stays_on_the_simplex(self):
        x = torch.randn(23, 3)
        baseline = make_variant("self_context", 3, 2, 16)
        for name in RADIAL_SHELL_VARIANTS:
            model = make_variant(name, 3, 2, 16)
            output = model(x)
            self.assertEqual(parameter_count(model), parameter_count(baseline))
            self.assertTrue(torch.isfinite(output).all())
            for weight in model.allocation_weights():
                self.assertTrue(torch.allclose(weight.sum(1), torch.ones(len(x)), atol=1e-5))

    def test_parallel_curvature_is_parameter_matched_and_finite(self):
        x = torch.randn(23, 3)
        baseline = make_variant("self_context", 3, 2, 16)
        for name in RADIAL_PARALLEL_VARIANTS:
            model = make_variant(name, 3, 2, 16)
            output = model(x)
            self.assertEqual(parameter_count(model), parameter_count(baseline))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_orthogonal_eikonal_frame_is_parameter_matched_and_finite(self):
        x = torch.randn(23, 3)
        baseline = make_variant("self_context", 3, 2, 16)
        for name in RADIAL_FRAME_VARIANTS:
            model = make_variant(name, 3, 2, 16)
            output = model(x)
            self.assertEqual(parameter_count(model), parameter_count(baseline))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_frozen_transport_shell_and_single_scale_variants_are_finite(self):
        names = (
            "self_context_stiefel_flow_curvature_frozen",
            "self_context_stiefel_flow_curvature_up",
            "self_context_stiefel_flow_curvature_down",
            "self_context_stiefel_flow_curvature_frozen_up",
            "self_context_stiefel_flow_curvature_frozen_down",
            "self_context_stiefel_flow_curvature_hutch2",
            "self_context_stiefel_flow_curvature_frozen_hutch2",
        )
        baseline = make_variant("self_context_stiefel_flow_curvature", 3, 2, 16)
        for name in names:
            model = make_variant(name, 3, 2, 16)
            output = model(torch.randn(23, 3))
            self.assertEqual(parameter_count(model), parameter_count(baseline))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_muon_orthogonalization_and_auxiliary_updates_are_finite(self):
        gradient = torch.randn(12, 7)
        update = zeropower_newton_schulz5(gradient)
        self.assertEqual(update.shape, gradient.shape)
        self.assertTrue(torch.isfinite(update).all())
        model = make_variant("self_context_stiefel_flow_curvature_frozen", 3, 2, 16)
        before = {name: parameter.detach().clone()
                  for name, parameter in model.named_parameters()}
        optimizer = MuonWithAuxAdamW(model, lr=3e-3)
        loss = model(torch.randn(31, 3)).square().mean()
        loss.backward(); optimizer.step()
        self.assertTrue(all(torch.isfinite(parameter).all()
                            for parameter in model.parameters()))
        self.assertTrue(any(not torch.equal(before[name], parameter)
                            for name, parameter in model.named_parameters()
                            if parameter.ndim == 2))
        self.assertTrue(any(not torch.equal(before[name], parameter)
                            for name, parameter in model.named_parameters()
                            if parameter.ndim == 1))

    def test_standalone_frame_flow_matches_experiment_implementation(self):
        torch.manual_seed(47)
        experiment = make_variant("self_context_stiefel_flow_curvature", 3, 2, 16)
        standalone = ContinuousFrameFlow(3, 2, width=16)
        source = experiment.state_dict()
        translated = {}
        for name in standalone.state_dict():
            source_name = name.replace("frame_atlas", "primitive")
            translated[name] = source[source_name]
        standalone.load_state_dict(translated)
        x = torch.randn(29, 3)
        self.assertTrue(torch.allclose(experiment(x), standalone(x), atol=1e-6, rtol=1e-5))

    def test_eikonal_energy_values_are_parameter_matched_and_finite(self):
        x = torch.randn(23, 3)
        baseline = make_variant("self_context", 3, 2, 16)
        for name in RADIAL_ENERGY_VARIANTS:
            model = make_variant(name, 3, 2, 16)
            output = model(x)
            self.assertEqual(parameter_count(model), parameter_count(baseline))
            self.assertTrue(torch.isfinite(output).all())

    def test_eikonal_integral_is_parameter_matched_and_finite(self):
        x = torch.randn(23, 3)
        baseline = make_variant("self_context", 3, 2, 16)
        for name in RADIAL_INTEGRAL_VARIANTS:
            model = make_variant(name, 3, 2, 16)
            output = model(x)
            self.assertEqual(parameter_count(model), parameter_count(baseline))
            self.assertTrue(torch.isfinite(output).all())

    def test_nd_tasks_have_inner_and_outer_support(self):
        for name in ("nd_spiral_low_rank", "nd_spiral_high_rank", "hypercube_checker"):
            task = TASK_BUILDERS[name](0); self.assertEqual(task.input_dim, 16); self.assertEqual(len(task.tail_x), 10)
            self.assertEqual(task.output_dim, 2); self.assertGreater(len(task.x_test), 1000)

    def test_stored_probe_atlas_is_the_full_product(self):
        stored = json.loads((Path(__file__).parent / "results_confirm/probes.json").read_text())["probes"]
        expected_variants = {"ordinary_mlp", "self_context", "self_context_hard", "self_context_chart"}
        stored_tasks = set(TASK_BUILDERS) - {"poly_drifted_chirp_1d"}
        self.assertEqual(len(stored), len(stored_tasks) * len(expected_variants))
        self.assertEqual(
            {(task, variant) for task in stored_tasks for variant in expected_variants},
            {(row["task"], row["variant"]) for row in stored},
        )


if __name__ == "__main__": unittest.main()
