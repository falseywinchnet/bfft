import unittest

import torch

from run_jet_transport import JetNetwork
from run_probabilistic_jet import JetPosterior, posterior_weights, split_relation_anchors


class ProbabilisticJetTests(unittest.TestCase):
    def test_evidence_split_is_disjoint_and_complete(self):
        fit, evidence = split_relation_anchors(101, 8, .2)
        self.assertEqual(len(set(fit.tolist()) & set(evidence.tolist())), 0)
        self.assertEqual(set(torch.cat([fit, evidence]).tolist()), set(range(101)))

    def test_posterior_prefers_lower_predictive_loss(self):
        weights = posterior_weights([.08, .01, .04], .02)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(int(weights.argmax()), 1)
        self.assertGreater(float(weights[1]), float(weights[2]))

    def test_mixture_is_weighted_logit_average(self):
        torch.manual_seed(3)
        hypotheses = [JetNetwork(8, integrated=True), JetNetwork(8, integrated=True)]
        weights = torch.tensor([.25, .75])
        mixture = JetPosterior(hypotheses, weights)
        x = torch.randn(7, 2)
        expected = .25*hypotheses[0](x) + .75*hypotheses[1](x)
        self.assertTrue(torch.allclose(mixture(x), expected))


if __name__ == "__main__":
    unittest.main()
