import unittest
import numpy as np
from .increment_transport import IncrementChart

class IncrementTests(unittest.TestCase):
    def test_unit_drift_is_retained(self):
        r=np.array([2.,-3.,.5]);states=[j*r for j in range(9)]
        chart=IncrementChart.fit(states,contractive=True)
        np.testing.assert_allclose(chart.candidate(100),108*r,rtol=1e-12)
        candidate,m,calls,_=chart.propose(lambda x:x+r)
        self.assertEqual(m,64);self.assertEqual(calls,3)
        np.testing.assert_allclose(candidate,72*r,rtol=1e-12)

    def test_coupled_rotation_learned_without_tangent(self):
        angle=.3;T=.98*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        states=[np.array([1.,2.])]
        for _ in range(8):states.append(T@states[-1])
        chart=IncrementChart.fit(states,contractive=True)
        np.testing.assert_allclose(chart.candidate(64),np.linalg.matrix_power(T,64)@states[-1],atol=2e-13)
        self.assertLess(chart.fit_defect,1e-13)

    def test_unseen_event_rejected_by_live_probe(self):
        states=[np.array([float(j)]) for j in range(9)]
        chart=IncrementChart.fit(states)
        def step(x):return x+1 if x[0]<15 else x-1
        _,m,calls,_=chart.propose(step)
        self.assertEqual(m,0);self.assertGreater(calls,0)

    def test_contractivity_and_constant_states(self):
        states=[np.array([1.1**j]) for j in range(9)]
        chart=IncrementChart.fit(states,contractive=True)
        self.assertLessEqual(np.linalg.norm(chart.T,2),1.)
        self.assertGreater(chart.clipping,0)
        chart=IncrementChart.fit([np.ones(3)]*9)
        np.testing.assert_array_equal(chart.candidate(64),np.ones(3))

if __name__=='__main__':unittest.main()
