from fractions import Fraction as F
import unittest
from experiments.conv_exact_fusion.polynomial_proof import (
    poly, add, sub, mul, scale, total, evaluate, tails, raw_bank,
    compile_line, certified, scalar_profile, first_cell, restrict, nonnegative,
)


class PolynomialFusionProofTests(unittest.TestCase):
    def test_mass_identity_and_four_dimensional_bubble(self):
        t = tails()
        # Four independent coordinates of the exact mass-zero correction.
        d = [F(3,7),F(-2,9),F(5,11),F(7,13)]
        d.append(-sum(d))
        e = total(scale(t[k],d[k]) for k in range(5))
        reduced = total(scale(sub(t[k],t[4]),d[k]) for k in range(4))
        self.assertEqual(e,reduced)
        self.assertEqual(e[0],0)
        self.assertEqual(evaluate(e,F(1)),0)
        # Divide e by u(1-u), exactly, without sampled fitting.
        q = [e[1]]
        for j in range(2,5): q.append(e[j]+q[-1])
        self.assertEqual(mul(poly([0,1,-1]),poly(q)),e)

    def test_transverse_analysis_commutes_with_synthesis(self):
        rows = [[F((i*i+3*j*i+7*j*j)%19) for i in range(7)] for j in range(7)]
        p = first_cell(rows,3)
        bank,delta = raw_bank(p)
        for u in [F(0),F(1,7),F(1,2),F(6,7),F(1)]:
            direct,dd = raw_bank([poly([evaluate(q,u)]) for q in p])
            self.assertEqual([[evaluate(q,u) for q in c] for c in bank],
                             [[q[0] for q in c] for c in direct])
            self.assertEqual([evaluate(q,u) for q in delta],[q[0] for q in dd])

    def test_certified_nonlinear_face_is_exact(self):
        # Nontrivial admission correction; same input line up to positive
        # polynomial amplitude and spatially constant polynomial offset.
        signal = [0,0,1,4,3,2,2]
        amp = poly([1,F(1,3),F(1,5)])
        offset = poly([3,-1,0,F(1,11)])
        y = [add(scale(amp,x),offset) for x in signal]
        c,checks = compile_line(y)
        self.assertTrue(certified(checks))
        a,_ = raw_bank(y)
        self.assertNotEqual(c,a)
        for u in [F(k,17) for k in range(18)]:
            direct = scalar_profile([evaluate(p,u) for p in y])
            self.assertEqual([[evaluate(p,u) for p in cell] for cell in c],
                             [[p[0] for p in cell] for cell in direct])

    def test_certificate_does_not_accept_midpoint_only(self):
        y = [poly([0]),poly([1]),poly([2]),poly([3]),poly([4]),poly([5,-9])]
        _,checks = compile_line(y)
        self.assertFalse(certified(checks))
        self.assertFalse(nonnegative(poly([F(-1,10),1]),strict=True))

    def test_projection_and_synthesis_fuse_to_one_stencil(self):
        for signal in ([0,0,1,4,3,2,2], [3,-2,5,-1,4,0,6], list(range(7))):
            a,d = raw_bank([poly([x]) for x in signal])
            c = scalar_profile(signal)
            for i in range(6):
                active = [k for k in range(5) if c[i][k][0] != 0]
                for u in (F(1,7),F(1,2),F(6,7)):
                    t = [evaluate(p,u) for p in tails()]
                    original = signal[i]+sum(t[k]*c[i][k][0] for k in range(5))
                    if not active:
                        self.assertEqual(original,signal[i])
                        continue
                    mean = sum(t[k] for k in active)/len(active)
                    fused = signal[i]+mean*d[i][0]+sum((t[k]-mean)*a[i][k][0] for k in active)
                    self.assertEqual(original,fused)
                    reference = active[0]
                    reduced = signal[i]+mean*d[i][0]+sum(
                        (t[k]-mean)*(a[i][k][0]-a[i][reference][0])
                        for k in active if k != reference)
                    self.assertEqual(original,reduced)

    def test_subinterval_substitution(self):
        p = poly([2,3,-7,5,1,8])
        q = restrict(p,F(2,7),F(5,7))
        for u in [F(k,9) for k in range(10)]:
            self.assertEqual(evaluate(q,u),evaluate(p,F(2,7)+F(3,7)*u))

    def test_actual_two_orders_share_all_source_grid_lines(self):
        rows = [[F((i*i+3*j*i+7*j*j)%19) for i in range(7)] for j in range(7)]
        cols = list(map(list,zip(*rows)))
        # Complete nonlinear two-factor evaluations, rational and nonseparable.
        def order(data, i,j,u,v):
            line = [evaluate(p,u) for p in first_cell(data,i)]
            c = scalar_profile(line)[j]
            return line[j]+sum(evaluate(t,v)*p[0] for t,p in zip(tails(),c))
        interior_defects = []
        for u,v in [(F(0),F(2,7)),(F(1),F(2,7)),(F(3,7),F(0)),
                    (F(3,7),F(1)),(F(2,7),F(3,7))]:
            a = order(rows,0,0,u,v)
            b = order(cols,0,0,v,u)
            if u in (0,1) or v in (0,1): self.assertEqual(a,b)
            else: interior_defects.append(a-b)
        self.assertTrue(any(interior_defects))

if __name__ == '__main__': unittest.main()
