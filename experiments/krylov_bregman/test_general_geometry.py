import unittest
import numpy as np
from .general_geometry import (RationalMirror, mirror_inverse, mirror_gradient,
    mirror_hessian, observable, fit_recurrence, predict, compatibility_defect, discover_mobius, mobius_transport)

KINDS = ('euclidean','quartic','entropy','hypentropy')

class GeneralGeometryTests(unittest.TestCase):
    def test_mirror_inverses(self):
        s = np.linspace(.01,3,51)
        for k in KINDS:
            np.testing.assert_allclose(mirror_gradient(mirror_inverse(s,k),k),s,atol=1e-13)

    def test_actual_mirror_update_and_convex_hessian(self):
        for k in KINDS:
            m = RationalMirror(k,n=6)
            x = m.initial
            np.testing.assert_allclose(m.encode(m.step(x)),m.dual_step(m.encode(x)),atol=1e-13)
            eps = 1e-5
            H = np.column_stack([(m.gradient(x+eps*v)-m.gradient(x-eps*v))/(2*eps)
                                 for v in np.eye(6)])
            self.assertLess(np.linalg.norm(H-H.T),1e-8)
            self.assertGreater(np.linalg.eigvalsh(H).min(),0)

    def test_reciprocal_identity_across_geometries(self):
        for k in KINDS:
            m=RationalMirror(k)
            s=m.initial_s
            for _ in range(32):
                actual=m.encode(m.step(m.decode(s)))
                np.testing.assert_allclose(1/actual,1/s/m.a+m.b/m.a,rtol=2e-12)
                s=actual

    def test_discovery_without_supplied_rates(self):
        m=RationalMirror()
        s=m.initial_s.copy(); data=[]
        for _ in range(9):
            data.append(observable(s,'reciprocal')); s=m.dual_step(s)
        Q,T,report=fit_recurrence(data)
        self.assertEqual(report['rank'],4)
        self.assertLess(report['training_defect'],1e-12)
        predicted=predict(Q,T,data[0],64)
        s=m.initial_s.copy()
        for _ in range(64): s=m.dual_step(s)
        np.testing.assert_allclose(predicted,observable(s,'reciprocal'),rtol=2e-8)

    def test_chart_synthesis_from_rational_relation(self):
        m=RationalMirror(); s=m.initial_s.copy(); states=[]
        for _ in range(9):
            states.append(s.copy()); s=m.dual_step(s)
        roots,rates,info=discover_mobius(states)
        for horizon in (16,64,128):
            predicted=mobius_transport(states[0],roots,rates,horizon)
            s=states[0].copy()
            for _ in range(horizon): s=m.dual_step(s)
            np.testing.assert_allclose(predicted,s,rtol=1e-8,atol=1e-11)
        self.assertLess(info['maximum_relation_defect'],1e-14)

    def test_unidentified_chart_rejected(self):
        with self.assertRaises(ValueError): discover_mobius(np.ones((9,4)))

    def test_geometry_compatibility_is_necessary(self):
        # A stable linear map can be gradient-compatible in one metric only.
        J=np.diag([.5,.8]); H=np.array([[2.,.6],[.6,1.]])
        self.assertEqual(compatibility_defect(np.eye(2),J),0)
        self.assertGreater(compatibility_defect(H,J),.1)
        # The corresponding conjugated map restores compatibility.
        C=np.array([[1.,.3],[0.,2.]])
        J2=np.linalg.inv(C)@J@C
        self.assertLess(compatibility_defect(C.T@C,J2),1e-14)

    def test_nonlinear_step_satisfies_compatibility(self):
        for k in KINDS:
            m=RationalMirror(k,n=6); x=m.initial; eps=1e-5
            J=np.column_stack([(m.step(x+eps*v)-m.step(x-eps*v))/(2*eps)
                               for v in np.eye(6)])
            self.assertLess(compatibility_defect(m.hessian(m.step(x)),J),1e-8)

    def test_directional_defect_identity(self):
        # Exact vector accumulation preserves cancellation lost by norm sums.
        T=np.array([[.9,.2],[0.,.8]])
        defects=[np.array([1.,0.]),np.array([-.9,0.])]
        e=np.zeros(2)
        for d in defects: e=T@e+d
        np.testing.assert_allclose(e,np.zeros(2),atol=1e-15)
        self.assertGreater(sum(np.linalg.norm(d) for d in defects),1)

    def test_euclidean_quartic_degree_growth(self):
        # Exact integer polynomial composition F(x)=x-x^3; no finite linear
        # coordinate-observable closure: distinct leading degrees 3**j.
        from numpy.polynomial import polynomial as p
        q=np.array([0,1],dtype=object)
        for j in range(5):
            self.assertEqual(len(q)-1,3**j)
            cube=p.polymul(p.polymul(q,q),q)
            q=p.polysub(q,cube)

if __name__=='__main__': unittest.main()
