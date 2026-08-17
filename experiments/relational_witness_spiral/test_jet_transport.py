import unittest

import torch

from run_jet_transport import JetLinear, JetNetwork, jet_relation_loss, observed_neighbor_triples
from run_hypersphere_atlas import spiral_points


class JetTransportTests(unittest.TestCase):
    def test_unbounded_jet_path_and_live_gradient(self):
        torch.manual_seed(4)
        layer = JetLinear(3, 4)
        x = torch.randn(8, 3)
        base = layer.base(x)
        difference_small = (layer(x) - base).norm()
        difference_large = (layer(10 * x) - layer.base(10 * x)).norm()
        self.assertGreater(float(difference_large), 10 * float(difference_small))
        difference_large.backward()
        self.assertGreater(float(layer.coefficients[-1].weight.grad.norm()), 0)

    def test_relations_use_observed_points_and_are_finite(self):
        x, _ = spiral_points(30, .02, .5, 3)
        tr = torch.arange(len(x))
        jj, kk = observed_neighbor_triples(x, tr)
        self.assertTrue(torch.all(jj < len(x)) and torch.all(kk < len(x)))
        model = JetNetwork(width=8)
        _, diagnostics = model(torch.cat([x, x[jj], x[kk]]), True)
        batch = len(x)
        loss = torch.zeros(())
        for d in diagnostics:
            di = {n: v[:batch] for n, v in d.items()}
            dj = {n: v[batch:2*batch] for n, v in d.items()}
            dk = {n: v[2*batch:] for n, v in d.items()}
            loss = loss + jet_relation_loss(di, dj, dk, True)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.up.operator_connection[-1].weight.grad.norm()), 0)

    def test_integrated_model_uses_connections_as_carrier(self):
        torch.manual_seed(7)
        model = JetNetwork(width=8, integrated=True)
        x = torch.randn(12, 2)
        loss = model(x).square().mean()
        loss.backward()
        self.assertGreater(float(model.up.input_connection[-1].weight.grad.norm()), 0)
        self.assertGreater(float(model.up.operator_connection[-1].weight.grad.norm()), 0)
        self.assertIsNone(model.up.representation[-1].weight.grad)
        self.assertIsNone(model.up.coefficients[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
