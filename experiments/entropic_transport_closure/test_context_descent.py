import unittest
import numpy as np
from .context_descent import evaluate, objective_difference, inverse_action, solve, ordinary_blas, evaluate_factored, conditional_action

class ContextTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(73)
        self.L=rng.normal(size=(9,11))*3
        self.a=rng.random(9)+.1; self.a/=self.a.sum()
        self.b=rng.random(11)+.1; self.b/=self.b.sum()
        self.rng=rng

    def test_gradient_and_retilting(self):
        z=self.rng.normal(size=11); d=self.rng.normal(size=11)
        st=evaluate(self.L,self.a,self.b,z); root=np.sqrt(self.b)
        e=1e-5
        fd=(evaluate(self.L,self.a,self.b,z+e*d)['value']-evaluate(self.L,self.a,self.b,z-e*d)['value'])/(2*e)
        self.assertAlmostEqual(fd,float(st['g']@d),places=8)
        h=.2*d; new=evaluate(self.L,self.a,self.b,z+root*h)
        tilt=st['P']*np.exp(h)[None,:]; tilt/=tilt.sum(1)[:,None]
        np.testing.assert_allclose(tilt,new['P'],atol=1e-14)
        self.assertAlmostEqual(objective_difference(st,h,self.a,self.b),new['value']-st['value'],places=13)
        self.assertAlmostEqual(objective_difference(st,h+19,self.a,self.b),objective_difference(st,h,self.a,self.b),places=13)

    def test_lifted_rule_and_descent(self):
        st=evaluate(self.L,self.a,self.b,self.rng.normal(size=11)); P=st['P']; c=st['c']
        H=np.diag(c)-P.T@(self.a[:,None]*P)
        root=np.sqrt(self.b); d=-st['g']/root
        Pd=P*(d[None,:]-(P@d)[:,None])
        np.testing.assert_allclose(self.a@Pd,H@d,atol=1e-14)
        self.assertLess(float((c-self.b)@d),0)
        np.testing.assert_allclose(H@np.ones(11),0,atol=1e-14)

    def test_inverse_memory_is_positive_and_secant(self):
        A=self.rng.normal(size=(11,11)); A=A.T@A+np.eye(11)
        pairs=[]
        for _ in range(5):
            s=self.rng.normal(size=11); pairs.append((s,A@s))
        B=np.column_stack([inverse_action(e,pairs) for e in np.eye(11)])
        np.testing.assert_allclose(B,B.T,atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(B).min(),0)
        np.testing.assert_allclose(B@pairs[-1][1],pairs[-1][0],atol=1e-12)

    def test_factored_conditional_matches_explicit(self):
        z=self.rng.normal(size=11)
        f=evaluate_factored(np.exp(self.L),self.L,self.a,self.b,z)
        e=evaluate(self.L,self.a,self.b,z)
        np.testing.assert_allclose(f["g"],e["g"],atol=1e-14)
        h=self.rng.normal(size=11)
        np.testing.assert_allclose(conditional_action(f,h),e["P"]@h,atol=1e-14)
        for scale in [.01,2]:
            self.assertAlmostEqual(objective_difference(f,scale*h,self.a,self.b),objective_difference(e,scale*h,self.a,self.b),places=13)

    def test_blas_matches_log_domain(self):
        out=ordinary_blas(self.L,self.a,self.b)
        self.assertTrue(out["converged"])
        control=solve(self.L,self.a,self.b,"ordinary")
        np.testing.assert_allclose(np.array(out["y"])-np.mean(out["y"]), np.array(control["y"])-np.mean(control["y"]),atol=1e-7)

    def test_complete_descent_and_gauge(self):
        for method in ['ordinary','gradient','scalar','lbfgs','adaptive','frozen']:
            out=solve(self.L,self.a,self.b,method,max_steps=4000)
            self.assertTrue(out['converged'],(method,out['residual'],out['failure']))
            self.assertTrue(all(x['objective_difference']<0 for x in out['history']))
            self.assertLessEqual(max(x['memory'] for x in out['history']),8)
        a=solve(self.L,self.a,self.b,y0=np.ones(11)*100)
        b=solve(self.L,self.a,self.b,y0=np.zeros(11))
        np.testing.assert_allclose(np.array(a['y'])-np.mean(a['y']),np.array(b['y'])-np.mean(b['y']),atol=1e-7)

if __name__=='__main__': unittest.main()
