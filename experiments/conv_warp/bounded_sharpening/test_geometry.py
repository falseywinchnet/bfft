"""Exact-rational admission checks and independent derivative identities."""
from fractions import Fraction as F
from math import comb
import random
import unittest
import numpy as np
from .study import derivative_rows, half_controls, refined_derivative_rows


def dot(a,b):
    return sum((x*y for x,y in zip(a,b)),F(0))


def finite_admission(P,directions,A,b):
    margin=[dot(row,P)-rhs for row,rhs in zip(A,b)]
    assert min(margin)>=0
    effects=[[dot(row,d) for d in directions] for row in A]
    harm=[sum(max(-v,F(0)) for v in row) for row in effects]
    caps=[min([F(1)]+[s/H for s,H,row in zip(margin,harm,effects) if row[g]<0]) for g in range(len(directions))]
    Q=[p+sum(c*d[j] for c,d in zip(caps,directions)) for j,p in enumerate(P)]
    remaining=[p+sum(d[j] for d in directions)-q for j,(p,q) in enumerate(zip(P,Q))]
    ratios=[(dot(row,Q)-rhs)/-dot(row,remaining) for row,rhs in zip(A,b) if dot(row,remaining)<0]
    tau=min([F(1)]+ratios)
    return [q+tau*r for q,r in zip(Q,remaining)],tau,Q,remaining


class GeometryTests(unittest.TestCase):
    def test_exact_overlapping_moment_preserving_bundles(self):
        rng=random.Random(923)
        for _ in range(120):
            n=7;P=[F(rng.randrange(1,10),10) for _ in range(n)]
            A=[[F(int(i==j)*sign) for j in range(n)] for sign in (1,-1) for i in range(n)]
            b=[F(0)]*n+[F(-1)]*n
            # Non-axis constraints: each contains the given feasible base.
            for _ in range(9):
                row=[F(rng.randrange(-3,4)) for _ in range(n)]
                A.append(row);b.append(dot(row,P)-F(rng.randrange(6),10))
            ds=[]
            for _ in range(5):
                d=[F(rng.randrange(-9,10),5) for _ in range(n-1)]
                ds.append(d+[-sum(d)])
            Q,tau,Q0,r=finite_admission(P,ds,A,b)
            self.assertEqual(sum(Q),sum(P))
            self.assertTrue(all(dot(row,Q)>=rhs for row,rhs in zip(A,b)))
            if tau<1:
                # Exact contact and failure of every larger step on this ray.
                self.assertTrue(any(dot(row,r)<0 and dot(row,Q)==rhs for row,rhs in zip(A,b)))
                beyond=[q+F(1,1000)*v for q,v in zip(Q,r)]
                self.assertTrue(any(dot(row,beyond)<rhs for row,rhs in zip(A,b)))

    def test_zero_margin_harm_and_help(self):
        Q,_,_,_=finite_admission([F(0)],[[F(-1)],[F(2)]],[[F(1)]],[F(0)])
        self.assertGreaterEqual(Q[0],0)
        Q,_,_,_=finite_admission([F(0)],[[F(-1)]],[[F(1)]],[F(0)])
        self.assertEqual(Q,[F(0)])

    def test_requested_ray_strength_saturates(self):
        outputs=[]
        for strength in (F(1,4),F(1),F(4),F(16)):
            Q,_,_,_=finite_admission([F(1,2)],[[strength]],[[F(1)],[F(-1)]],[F(0),F(-1)])
            outputs.append(Q[0])
        self.assertEqual(outputs,[F(3,4),F(1),F(1),F(1)])

    def test_mixed_derivative_coefficients(self):
        # x^3*y^2 represented independently in degree-(5,5) Bernstein form.
        P=np.array([[comb(i,3)/comb(5,3)*comb(j,2)/comb(5,2) for i in range(6)] for j in range(6)])
        D=derivative_rows(P,((3,2),))@P.ravel()
        np.testing.assert_allclose(D,12,atol=2e-13)
        for a,b in ((1,0),(0,1),(2,1),(4,0),(0,3),(5,5)):
            value=derivative_rows(P,((a,b),))@P.ravel()
            factor=(6 if a==3 else (3 if a==1 else 6 if a==2 else 1))*(2 if b in (1,2) else 1)
            expected=np.array([0 if a>3 or b>2 else factor*comb(i,3-a)/comb(5-a,3-a)*comb(j,2-b)/comb(5-b,2-b) for j in range(6-b) for i in range(6-a)])
            np.testing.assert_allclose(value,expected,atol=2e-11)

    def test_quintic_current_and_slope_bound(self):
        controls=np.array([0.,0.,0.,1.,1.,1.])
        t=np.linspace(0,1,1001)
        potential=sum(controls[i]*comb(5,i)*t**i*(1-t)**(5-i) for i in range(6))
        np.testing.assert_allclose(potential,10*t**3-15*t**4+6*t**5,atol=3e-15)
        slope=5*sum(np.diff(controls)[i]*comb(4,i)*t**i*(1-t)**(4-i) for i in range(5))
        self.assertAlmostEqual(slope.max(),15/8)
        self.assertTrue(np.all(slope>=0))

    def test_shared_edge_bundle_has_zero_cell_moments(self):
        # Two incident cells, sharing one complete edge.
        P=np.zeros((6,11));edge=np.array([1.,-2.,3.,4.]);P[1:5,5]=edge
        P[1:5,1:5]=-edge.sum()/16;P[1:5,6:10]=-edge.sum()/16
        self.assertEqual(P[:,:6].sum(),0)
        self.assertEqual(P[:,5:].sum(),0)
        self.assertTrue(np.all(P[::5,::5]==0))

    def test_half_interval_derivative_certificate_is_sharp_for_step(self):
        # Original quartic derivative controls bound the slope by 5.
        # Exact half-interval controls certify the true maximum 15/8.
        controls=np.array([0.,0.,5.,0.,0.])
        refined=half_controls(4)@controls
        self.assertEqual(refined.max(),15/8)
        self.assertEqual(refined.min(),0)
        P=np.tile([0.,0.,0.,1.,1.,1.],(6,1))
        self.assertEqual(np.max(refined_derivative_rows(P)@P.ravel()),15/8)

    def test_subdivision_preserves_polynomial(self):
        rng=np.random.default_rng(31)
        for degree in range(1,6):
            controls=rng.normal(size=degree+1);refined=(half_controls(degree)@controls).reshape(2,degree+1)
            t=np.linspace(0,1,33)
            for half in (0,1):
                s=(t+half)/2
                original=sum(controls[i]*comb(degree,i)*s**i*(1-s)**(degree-i) for i in range(degree+1))
                value=sum(refined[half,i]*comb(degree,i)*t**i*(1-t)**(degree-i) for i in range(degree+1))
                np.testing.assert_allclose(value,original,atol=2e-15)


if __name__=='__main__':unittest.main()
