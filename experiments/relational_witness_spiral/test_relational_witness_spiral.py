import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("experiment", HERE / "run_experiment.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class ExperimentTests(unittest.TestCase):
    def test_spiral_is_antipodal_and_bounded(self):
        x, y, u = M.spiral_points(17, .2, .4, 3)
        np.testing.assert_allclose(x[:17], -x[17:])
        np.testing.assert_array_equal(y, [0]*17 + [1]*17)
        self.assertTrue(np.all((u >= .2) & (u <= .4)))

    def test_all_models_backpropagate(self):
        x, y, _ = M.spiral_points(8, .1, .3, 9)
        for kind in ["reference_linear", "witness_static", "witness_marginal", "witness_relational"]:
            model = M.Model(kind, 8, 0)
            logits = model.forward(x)
            loss, grad = M.cross_entropy(logits, y)
            model.backward(grad)
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(all(np.isfinite(p.grad).all() for p in model.parameters()))

    def test_relational_gradient(self):
        rng = np.random.default_rng(4)
        layer = M.WitnessLinear(4, 6, rng, "relational", response=5, seed=2)
        x = rng.normal(size=(3, 4)); upstream = rng.normal(size=(3, 6))
        layer.forward(x); analytic = layer.backward(upstream)
        eps = 1e-6
        for i in range(x.size):
            xp=x.copy(); xm=x.copy(); xp.flat[i]+=eps; xm.flat[i]-=eps
            fp=(layer.forward(xp)*upstream).sum(); fm=(layer.forward(xm)*upstream).sum()
            self.assertAlmostEqual(analytic.flat[i], (fp-fm)/(2*eps), places=5)


if __name__ == "__main__": unittest.main()
