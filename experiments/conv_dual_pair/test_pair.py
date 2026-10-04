import unittest
import numpy as np
import sympy as sp

from experiments.conv_dual_pair.pair import (
    BASIN, LINEAR_DUAL, MAXFLAT, BAND_OPTIMAL, proposal, moment,
    restrict, grow, analyze, synthesize, grow2d, restrict2d,
)


def variations(values):
    signs = np.sign(np.diff(values))
    signs = signs[np.abs(np.diff(values)) > 2e-12]
    return np.count_nonzero(signs[1:] != signs[:-1])


class PairTests(unittest.TestCase):
    def test_exact_basin_from_quintic(self):
        y = {i: sp.Symbol(f"y{i}") for i in range(-3, 4)}
        def controls(i):
            m = lambda j: (y[j-2]-8*y[j-1]+8*y[j+1]-y[j+2])/12
            q = lambda j: (-y[j+2]+16*y[j+1]-30*y[j]+16*y[j-1]-y[j-2])/12
            return [y[i], y[i]+m(i)/5, y[i]+2*m(i)/5+q(i)/20,
                    y[i+1]-2*m(i+1)/5+q(i+1)/20, y[i+1]-m(i+1)/5, y[i+1]]
        value = sp.expand((sum(controls(-1))+sum(controls(0)))/12)
        expected = [sp.Rational(n,2880) for n in (11,-82,709,1604,709,-82,11)]
        self.assertEqual([value.coeff(y[k]) for k in range(-3,4)], expected)

    def test_exact_linear_dual_and_seven_tap_obstruction(self):
        h = {k:sp.Rational(n,2880) for k,n in zip(range(-3,4),(11,-82,709,1604,709,-82,11))}
        g = {k:sp.Rational(n,1103232) for k,n in zip(range(-4,5),(-2981,-22222,-173472,573838,1456138,573838,-173472,-22222,-2981))}
        for shift in range(-4,5):
            self.assertEqual(sum(h[k]*g.get(k-2*shift,0) for k in h), int(shift==0))
        for phase in (0,1):
            self.assertEqual(sum(v for k,v in g.items() if k%2==phase),1)
        unknowns = sp.symbols('g0:4')
        short = {k:unknowns[abs(k)] for k in range(-3,4)}
        equations = [sum(h[k]*short.get(k-2*j,0) for k in h)-int(j==0) for j in range(-3,4)]
        equations += [sum(v for k,v in short.items() if k%2==p)-1 for p in (0,1)]
        self.assertEqual(sp.linsolve(equations,unknowns),sp.EmptySet)

    def test_band_optimum_exact_stationarity(self):
        t = sp.Symbol('t', real=True)
        a = sp.Matrix([5*t+sp.Rational(11,64),-4*t-sp.Rational(3,128),t])
        b = sp.Matrix([4/sp.pi-1,8/(3*sp.pi)-1,52/(15*sp.pi)-1])
        objective = (a-b).dot(a-b)
        optimum = (4096-945*sp.pi)/(13440*sp.pi)
        self.assertEqual(sp.simplify(sp.diff(objective,t).subs(t,optimum)),0)
        self.assertEqual(sp.diff(objective,t,2),84)
        np.testing.assert_allclose(np.array(a.subs(t,optimum),dtype=float).ravel(),BAND_OPTIMAL,atol=1e-16)

    def test_polynomial_cell_averages_through_degree_six(self):
        # Interior and one-sided closure checked independently of admission.
        centers = np.arange(13,dtype=float)-6
        for power in range(7):
            parent = ((centers+.5)**(power+1)-(centers-.5)**(power+1))/(power+1)
            exact = ((centers+.5)**(power+1)-2*centers**(power+1)+(centers-.5)**(power+1))/(power+1)
            np.testing.assert_allclose(proposal(parent),exact,rtol=3e-12,atol=2e-10)

    def test_cycles_and_variation(self):
        rng = np.random.default_rng(72019)
        fields = [rng.normal(size=31) for _ in range(40)]
        fields += [np.ones(31), np.arange(31.), np.r_[np.zeros(15),np.ones(16)],
                   np.eye(1,31,15).ravel(),np.arange(31.)%2,
                   np.cumsum(rng.lognormal(size=31))]
        for mode in ('maxflat','band'):
            for z in fields:
                fine = grow(z,mode)
                np.testing.assert_allclose(restrict(fine),z,rtol=3e-15,atol=3e-14)
                self.assertLessEqual(variations(fine),variations(z))
                mu=moment(z,mode)
                self.assertLessEqual(np.max(np.abs(mu)),np.max(np.abs(np.diff(z)))+1e-12)
                x=rng.normal(size=(62,3))
                np.testing.assert_allclose(synthesize(analyze(x,mode)),x,rtol=3e-14,atol=3e-14)

    def test_two_dimensional_coarse_cycle(self):
        rng=np.random.default_rng(99)
        for mode in ('maxflat','band'):
            z=rng.normal(size=(11,13,4))
            np.testing.assert_allclose(restrict2d(grow2d(z,mode)),z,atol=2e-15)

    def test_linear_dual_rings_on_step(self):
        # An exact FIR reciprocal alone does not enforce CONV admission.
        z=np.r_[np.zeros(25),np.ones(25)]
        fine=np.zeros(2*len(z));fine[::2]=z
        fine=np.convolve(fine,LINEAR_DUAL,'same')
        window=fine[36:64]
        self.assertLess(window.min(),-.1)
        self.assertGreater(window.max(),1.1)


if __name__=='__main__':
    unittest.main()
