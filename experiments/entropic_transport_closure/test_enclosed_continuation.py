import unittest
import numpy as np
from .context_descent import evaluate, objective_difference
from .enclosed_continuation import (merge_zonotopes,reduce_generators,merged_cloud,
    prepare,gradient_support,increment_bound,solve,audit)

class EnclosureTests(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(415)

    def problem(self,n=11):
        L=self.rng.normal(size=(9,n))*2
        a=self.rng.random(9)+.2; a/=a.sum()
        b=self.rng.random(n)+.2; b/=b.sum()
        return L,a,b

    def test_merge_and_reduction_preserve_support(self):
        for r in [2,4]:
            c1,c2=self.rng.normal(size=(2,r)); G1=self.rng.normal(size=(r,7)); G2=self.rng.normal(size=(r,5))
            c,G=merge_zonotopes(c1,G1,c2,G2,2*r)
            self.assertLessEqual(G.shape[1],2*r)
            for d in self.rng.normal(size=(100,r)):
                target=max(c1@d+np.abs(G1.T@d).sum(),c2@d+np.abs(G2.T@d).sum())
                self.assertGreaterEqual(c@d+np.abs(G.T@d).sum()+1e-12,target)
            points=self.rng.normal(size=(17,r)); c,G=merged_cloud(points,2*r)
            for d in self.rng.normal(size=(100,r)):
                self.assertGreaterEqual(c@d+np.abs(G.T@d).sum()+1e-12,max(points@d))

    def test_chart_hessian_and_gradient_enclosure(self):
        L,a,b=self.problem(); z=self.rng.normal(size=len(b)); st=evaluate(L,a,b,z)
        for kind in ['exact','box','merged']:
            candidates=self.rng.normal(size=(4,len(b)))
            m=prepare(st,a,b,4,kind,candidates=candidates); U=m['U']; V=m['V']
            np.testing.assert_allclose(U.T@U,np.eye(4),atol=1e-12)
            np.testing.assert_allclose(U.T@np.sqrt(b),0,atol=1e-12)
            P=st['P']
            H=np.diag(st['c'])-P.T@(a[:,None]*P)
            np.testing.assert_allclose(m['A'],V.T@H@V,atol=2e-12)
            for R in [.01,.1,.5,1.5]:
                s=self.rng.normal(size=4); s*=R/np.ptp(V@s)
                at=evaluate(L,a,b,z+U@s)
                for _ in range(20):
                    d=self.rng.normal(size=4)
                    exact=float((at['c']-b)@(V@d)); upper,_=gradient_support(m,s,d)
                    self.assertLessEqual(exact,upper+2e-12)
                    d*=.2/np.ptp(V@d)
                    difference=objective_difference(at,V@d,a,b)
                    upper,*_=increment_bound(m,s,d)
                    self.assertLessEqual(difference,upper+2e-13)

    def test_complete_solver_and_independent_internal_audit(self):
        L,a,b=self.problem()
        for kind in ['exact','box','merged']:
            out=solve(L,a,b,enclosure=kind,warmup=0)
            self.assertTrue(out['converged'],(kind,out['residual']))
            self.assertTrue(out['blocks'])
            checks=audit(L,a,b,out)
            self.assertLessEqual(checks['max_increment_bound_violation'],1e-13)
            self.assertLessEqual(checks['max_directional_support_violation'],1e-13)
            self.assertEqual(checks['positive_changes_above_1e_minus24'],0)

if __name__=='__main__': unittest.main()
