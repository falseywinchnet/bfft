from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from ML_experiment.sparse_sine_gradient_zonotope import (
    _flatten_gradients,
    _named_parameters,
    commit_adamw,
    gradient_covariance_frame,
    preview_adamw,
    smooth_witness_objective,
    train_gradient_zonotope,
)
from ML_experiment.sparse_sine_study import AcquisitionData, SupportWhitened, make_model
from ML_experiment.sparse_sine_witness_descent import (
    InterleavedWitnessAtlas,
    train_witness_descent,
)
from ML_experiment.tasks import sparse_sine_1d
from ML_experiment.run_gradient_zonotope_battery import ProjectiveWitnessAtlas
from ML_experiment.tasks import TASK_BUILDERS


class GradientZonotopeTests(unittest.TestCase):
    def test_covariance_frame_separates_common_and_anisotropic_parts(self):
        common = torch.tensor([2.0, -1.0, 0.5, 3.0])
        a = torch.tensor([1.0, 0.0, -1.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0, -1.0])
        gradients = torch.stack((common + a, common - a, common + b, common - b))
        mean, directions, retained = gradient_covariance_frame(gradients, rank=2)
        self.assertTrue(torch.allclose(mean, common))
        self.assertEqual(directions.shape, (2, 4))
        self.assertAlmostEqual(float(retained), 1.0, places=5)
        self.assertLess(float(directions[0] @ directions[1]).__abs__(), 1e-6)

    def test_adamw_preview_matches_committed_update(self):
        torch.manual_seed(12)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(8, 2)).square().mean().backward()
            optimizer.step()
        flat = torch.cat([torch.randn_like(p).flatten() for p in model.parameters()])
        preview = preview_adamw(model, optimizer, flat)
        commit_adamw(model, optimizer, flat)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.allclose(parameter, preview[name], atol=2e-7, rtol=2e-6))

    def test_full_covariance_frame_reconstructs_candidate_scores(self):
        torch.manual_seed(61001)
        data = AcquisitionData(sparse_sine_1d(1))
        model = SupportWhitened(make_model("self_context", 8), data)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        atlas = InterleavedWitnessAtlas(data, cells=16, folds=4)
        named = _named_parameters(model)
        gradients = []
        for sample in range(3):
            generator = torch.Generator().manual_seed(71000 + 1000003 + 37 * sample)
            phase = sample * 0.6180339887498949 % 1.0
            index = atlas.stratified_train(256, 0, generator, phase=phase)
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(model(data.x[index]), data.y[index]).backward()
            gradients.append(_flatten_gradients(named))
        gradients = torch.stack(gradients)
        common, directions, _ = gradient_covariance_frame(gradients, 2)
        denominator = directions.square().sum(-1).clamp_min(1e-12)
        coordinates = torch.einsum(
            "bp,rp->br", gradients - common, directions
        ) / denominator
        reconstructed = common + torch.einsum("br,rp->bp", coordinates, directions)
        self.assertTrue(torch.allclose(reconstructed, gradients, atol=2e-6, rtol=2e-5))
        for gradient in reconstructed:
            parameters = preview_adamw(model, optimizer, gradient)
            score, _, _ = smooth_witness_objective(
                model, parameters, atlas, 0, worst_weight=0.10
            )
            clone = SupportWhitened(make_model("self_context", 8), data)
            clone.load_state_dict(model.state_dict())
            clone_optimizer = torch.optim.AdamW(
                clone.parameters(), lr=3e-3, weight_decay=1e-4
            )
            commit_adamw(clone, clone_optimizer, gradient)
            expected = atlas.score(clone, 0, worst_weight=0.10)["score"]
            self.assertAlmostEqual(float(score), expected, places=5)

    def test_full_rank_ray_identifies_the_discrete_best_endpoint(self):
        task = sparse_sine_1d(1)
        discrete_result = train_witness_descent(
            task, model_name="self_context", width=8, seed=1,
            macro_steps=1, branches=3, micro_steps=1, batch=64, lr=3e-3,
            cells=16, folds=4, worst_weight=0.10, evaluate_every=1,
            witness_period=50,
        )
        discrete, discrete_history = discrete_result[0], discrete_result[1]
        continuous_result = train_gradient_zonotope(
            task, model_name="self_context", width=8, seed=1, steps=1,
            gradient_samples=3, covariance_rank=2, radius=4.0,
            selector="ray", quadratic_probe=0.75, selection_temperature=0.002,
            selection_decay=0.90,
            outer_steps=2,
            outer_lr=0.3, outer_optimizer_name="adam",
            coefficient_penalty=0.0, batch=64, lr=3e-3, cells=16,
            folds=4, witness_period=50, worst_weight=0.10,
            evaluate_every=1,
        )
        continuous = continuous_result["model"]
        self.assertEqual(
            discrete_history[0]["winner"],
            int(continuous_result["history"][0]["selected_ray"]),
        )
        self.assertLessEqual(
            continuous_result["history"][0]["selected_score"],
            discrete_history[0]["candidate_score_min"] + 1e-5,
        )

    def test_projective_atlas_balances_class_folds(self):
        task = TASK_BUILDERS["spiral"](0)
        atlas = ProjectiveWitnessAtlas(task, folds=4)
        seen = torch.zeros(len(task.x_train), dtype=torch.long)
        counts_by_fold = []
        for fold in range(4):
            witness = atlas.witness(fold)
            seen[witness] += 1
            counts_by_fold.append(
                torch.bincount(task.y_train[witness], minlength=2)
            )
            draw = atlas.sample(128, fold, torch.Generator().manual_seed(9 + fold))
            self.assertFalse(torch.isin(draw, witness).any())
        self.assertTrue(torch.equal(seen, torch.ones_like(seen)))
        counts = torch.stack(counts_by_fold)
        self.assertTrue(torch.all(counts.max(0).values - counts.min(0).values <= 1))


if __name__ == "__main__":
    unittest.main()
