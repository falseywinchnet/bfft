from __future__ import annotations

import unittest

from ML_experiment.muon_primacy import make_operator_problem
from ML_experiment.run_optimizer_operator_battery import (
    OPERATOR_NATIVE_LRS,
    OPTIMIZERS,
    STANDING_LRS,
    _float_problem,
    run,
)


class OptimizerOperatorBatteryTests(unittest.TestCase):
    def test_every_declared_optimizer_takes_a_finite_step(self):
        problem = _float_problem(make_operator_problem(8, 1e2, seed=17))
        for name in OPTIMIZERS:
            result = run(problem, [problem.covariance], name, STANDING_LRS[name])
            self.assertEqual(result["status"], "complete", name)
            self.assertEqual(len(result["history"]), 1, name)

    def test_operator_native_rates_cover_exactly_the_battery_arms(self):
        self.assertEqual(set(OPERATOR_NATIVE_LRS), set(OPTIMIZERS))
        self.assertEqual(set(STANDING_LRS), set(OPTIMIZERS))


if __name__ == "__main__":
    unittest.main()
