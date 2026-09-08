import unittest
import numpy as np
from walk import ordered_control,compile_walk,orient
from gauge import Potentials,best_gauge

class Tests(unittest.TestCase):
    def test_cycle_obstruction(self):
        p=Potentials(3)
        self.assertTrue(p.admit(0,1,2))
        self.assertTrue(p.admit(1,2,3))
        self.assertTrue(p.admit(0,2,5))
        self.assertFalse(p.admit(0,2,4))
    def test_fixed_endpoint_and_actual_oblique_frames(self):
        for n in (4,6,8,16):
            r=ordered_control(n);route=orient(r['permutation'],[(g.i,g.j) for g in reversed(r['gates'])])
            gates=compile_walk(r,route);original,changed=best_gauge(gates,n,8)
            self.assertIsNotNone(changed)
            self.assertLess(changed['max_matrix_change'],1e-10)
            self.assertGreater(changed['nonzero_log_gauges'],0)
            self.assertLess(changed['counts']['estimated_real_multiplies'],original['estimated_real_multiplies'])

if __name__=='__main__':unittest.main()
