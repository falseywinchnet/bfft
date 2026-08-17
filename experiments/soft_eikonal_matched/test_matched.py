from __future__ import annotations

import inspect, unittest
import torch

import models
from metrics import jacobian_variability
from tasks import TASK_BUILDERS


class MatchedTests(unittest.TestCase):
    def test_parameter_counts_match_exactly(self):
        for input_dim, output_dim in ((2,2),(2,1),(1,3),(10,2)):
            for width in (16,36):
                linear,soft=models.make_matched_pair(input_dim,output_dim,width)
                self.assertEqual(models.parameter_count(linear),models.parameter_count(soft))
                self.assertGreater(linear.extra.numel(),0)

    def test_ordinary_mlp_budget_matches_exactly(self):
        for input_dim,output_dim in ((2,2),(2,1),(1,3),(10,2)):
            for width in (16,36):
                mlp,soft=models.make_mlp_pair(input_dim,output_dim,width)
                self.assertEqual(models.parameter_count(mlp),models.parameter_count(soft))
                self.assertGreaterEqual(mlp.expansion,2*width)
                self.assertLess(mlp.extra.numel(),2*width+1)
                self.assertIsInstance(mlp.activation,models.LELU)

    def test_ordinary_mlp_is_genuinely_nonlinear(self):
        mlp,_=models.make_mlp_pair(2,2,16);x=torch.randn(32,2)
        variability,rank=jacobian_variability(mlp,x)
        self.assertGreater(variability,1e-4);self.assertGreater(rank,0)

    def test_affine_control_is_exactly_affine(self):
        linear,_=models.make_matched_pair(5,3,16); x=torch.randn(11,5); y=torch.randn(11,5)
        origin=linear(torch.zeros_like(x)); residual=linear(x+y)-origin-(linear(x)-origin)-(linear(y)-origin)
        self.assertLess(float(residual.abs().max()),2e-5)
        weight,bias=linear.collapsed(); self.assertTrue(torch.allclose(linear(x),x@weight.T+bias,atol=2e-5))
        variability,rank=jacobian_variability(linear,x); self.assertLess(variability,1e-5);self.assertEqual(rank,0)

    def test_soft_modes_and_jacobian_are_nontrivial(self):
        _,soft=models.make_matched_pair(2,2,16);x=torch.randn(32,2)
        outputs=[]
        for mode in ("matched","mismatched","uniform","base_only"):
            soft.set_diagnostic_mode(mode);outputs.append(soft(x))
        self.assertEqual([tuple(value.shape) for value in outputs],[(32,2)]*4)
        soft.set_diagnostic_mode("matched"); variability,_=jacobian_variability(soft,x)
        self.assertGreater(variability,1e-4)

    def test_no_forbidden_activation(self):
        source=inspect.getsource(models);self.assertNotIn("nn."+"GELU",source);self.assertNotIn("F."+"gelu",source)
        x=torch.linspace(-3,3,17);expected=x*torch.sigmoid((torch.pi/torch.sqrt(torch.tensor(3.0)))*x)
        self.assertTrue(torch.allclose(models.LELU()(x),expected))

    def test_problem_catalog(self):
        for name in ("periodic_wells","two_moons","pinwheel","xor_quads","sinusoid_bounds","radial_stripes",
                     "swiss_cheese","lorenz_lobes","complex_spiral_3d","periodic_nd","hyperchecker"):
            task=TASK_BUILDERS[name](0);self.assertGreater(len(task.x_train),500);self.assertEqual(task.output_dim,task.y_train.shape[1] if task.kind=="regression" else int(task.y_train.max())+1)

    def test_one_dimensional_function_suite(self):
        for name in ("multiscale_1d","chirp_1d","localized_steps_1d","fourier_mix_1d"):
            task=TASK_BUILDERS[name](0);self.assertEqual(task.input_dim,1);self.assertEqual(task.output_dim,1)
            self.assertEqual(len(task.tail_x),10);self.assertTrue(torch.isfinite(task.y_train).all())
            truth=task.truth(task.x_test).reshape_as(task.y_test)
            recovered=task.y_test*task.target_std+task.target_mean
            self.assertTrue(torch.allclose(truth,recovered,atol=1e-5,rtol=1e-5),name)


if __name__=="__main__":unittest.main()
