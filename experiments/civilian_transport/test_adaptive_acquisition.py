import unittest
import numpy as np
from .adaptive_acquisition import BoundarySampler,BoundaryFixture


class AdaptiveAcquisitionTests(unittest.TestCase):
    def test_semiperiodic_spacing(self):
        sampler=BoundarySampler(seed=10)
        t=0.
        for _ in range(50):
            nxt=sampler.next_time(t)
            self.assertTrue(.4-1e-12 <= nxt-t <= .6+1e-12)
            t=nxt

    def test_trigger_ramp_is_continuous(self):
        s=BoundarySampler(high_rate=32.)
        self.assertEqual(s.rate(5.),2.)
        s.observe_side(6.,0.)
        self.assertEqual(s.rate(6.),2.)
        self.assertLess(s.rate(6.+1e-9)-2.,1e-6)
        self.assertGreater(s.rate(7.),s.rate(6.))
        self.assertLess(s.rate(7.),32.)
        s.observe_side(7.,-1.)
        self.assertEqual(s.trigger_time,6.)

    def test_ramped_phase_spacing(self):
        s=BoundarySampler(high_rate=32.,seed=12)
        s.observe_side(6.,.1)
        t=6.
        for _ in range(100):
            nxt=s.next_time(t)
            phase=s.integrated_rate(t,nxt)
            self.assertTrue(.8-1e-10<=phase<=1.2+1e-10)
            self.assertGreater(nxt,t)
            t=nxt

    def test_untriggered_policies_share_history(self):
        a=BoundarySampler(high_rate=4.,seed=9)
        b=BoundarySampler(high_rate=32.,seed=9)
        ta=tb=0.
        for _ in range(15):
            ta=a.next_time(ta);tb=b.next_time(tb)
            self.assertEqual(ta,tb)

    def test_closed_loop_shares_history_until_estimated_crossing(self):
        from .acquisition_study import simulate
        fixture=BoundaryFixture('helix',110001)
        a=simulate(fixture,.35,'adaptive_4',110001,cells=32,keep_trace=True)
        b=simulate(fixture,.35,'adaptive_32',110001,cells=32,keep_trace=True)
        self.assertEqual(a['trigger_time'],b['trigger_time'])
        for result in (a,b):
            self.assertEqual(result['updates'],result['samples']-1)
            self.assertGreater(result['update_ms'],0.)
            events=result['trace']['events']
            self.assertTrue(all(y['time']>x['time'] for x,y in zip(events,events[1:])))
        ea=[e for e in a['trace']['events'] if e['time']<=a['trigger_time']]
        eb=[e for e in b['trace']['events'] if e['time']<=b['trigger_time']]
        self.assertEqual(len(ea),len(eb))
        np.testing.assert_array_equal([e['time'] for e in ea],[e['time'] for e in eb])
        np.testing.assert_array_equal([e['observation'] for e in ea],[e['observation'] for e in eb])
        self.assertGreater(b['samples'],a['samples'])

    def test_boundary_truth_is_policy_independent(self):
        f=BoundaryFixture('helix',110001)
        self.assertLess(abs(float(f.position(f.crossing_time)[0])),1e-12)
        self.assertTrue(5.8<f.crossing_time<6.2)
        a=f.position([0.,6.,12.])
        b=f.position(np.linspace(0,12,121))[[0,60,120]]
        np.testing.assert_array_equal(a,b)


if __name__=='__main__':
    unittest.main()
