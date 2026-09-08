import unittest
import numpy as np
from .model import ReducedMeyerMap,grad,div,project_disk,project_disk_derivative
from .intermediate_state import trace,longitudinal,sub,vnorm

class IntermediateTests(unittest.TestCase):
    def test_within_pass_and_remainder_identities(self):
        f=np.random.default_rng(32).uniform(0,255,(12,16))
        for lam,mu in [(.02,20),(.05,40),(.1,80)]:
            m=ReducedMeyerMap(f,lam,mu);z=m.initial()
            for k in range(12):
                r=trace(m,z)
                for key in ['u_balance_identity_rms','w_balance_identity_rms','full_remainder_identity_rms']:
                    self.assertLess(r[key],2e-11)
                for name in ['u_branch','w_branch']:
                    for key in ['shrink_identity_error','update_identity_error','prox_complementarity_error','incoming_capacity_excess','nonintegrability_identity_error']:
                        self.assertLess(r[name][key],2e-11)
                p=r['phases']['after_projection']
                self.assertLessEqual(p['cartoon_complementarity'],r['cartoon_complementarity_bound']+1e-10)
                self.assertLessEqual(p['texture_complementarity'],r['texture_complementarity_bound']+1e-10)
                z=m.step(z)

    def test_transverse_field_can_change_projected_divergence(self):
        rng=np.random.default_rng(33);m=ReducedMeyerMap(np.zeros((12,16)),.05,40)
        h=(rng.normal(size=m.shape),rng.normal(size=m.shape))
        h=sub(h,longitudinal(m,h));self.assertLess(vnorm(grad(div(*h))),1e-12)
        t=(20*rng.normal(size=m.shape),20*rng.normal(size=m.shape))
        dh=project_disk_derivative(*t,*h,m.ru)
        self.assertGreater(float(np.linalg.norm(div(*dh))),.01)
        eps=1e-6
        pp=project_disk(t[0]+eps*h[0],t[1]+eps*h[1],m.ru)
        pm=project_disk(t[0]-eps*h[0],t[1]-eps*h[1],m.ru)
        fd=div(*sub(pp,pm))/(2*eps)
        np.testing.assert_allclose(fd,div(*dh),atol=3e-9,rtol=1e-6)

    def test_chart_can_close_or_have_a_left_null_obstruction(self):
        from .intermediate_chart import probe
        x=np.arange(64)[None,:]
        for name,f in [('ramp',255*x/64),('edge',50+150.*(x>32))]:
            m=ReducedMeyerMap(f,.05,40);z=m.initial()
            for _ in range(63):z=m.step(z)
            r=probe(m,z)
            self.assertEqual(r['changed_clip_branches'],0)
            self.assertLess(r['left_null_obstruction_check_rms'],1e-10)
            if name=='ramp':
                self.assertLess(r['linear_equation_residual_rms'],1e-10)
                self.assertLess(r['candidate_fixed_point_residual_rms'],1e-10)
                self.assertLess(r['candidate_gap']['gap_per_pixel'],1e-10)
            else:
                self.assertGreater(r['linear_equation_residual_rms'],.01)
                self.assertGreater(r['candidate_fixed_point_residual_rms'],.01)
                self.assertEqual(r['first_actual_branch_event_within_2048_steps'],150)

if __name__=='__main__':unittest.main()
