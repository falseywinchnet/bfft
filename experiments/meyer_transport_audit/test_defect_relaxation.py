import unittest
from unittest.mock import patch
import numpy as np
from .model import ReducedMeyerMap, State, grad, div, project_disk
from .defect_relaxation import DefectRelaxation


class DefectRelaxationTests(unittest.TestCase):
    def test_ordinary_matches_original_and_fft_budget(self):
        m = ReducedMeyerMap(np.random.default_rng(2).normal(100,20,(12,16)), .05,40)
        z = m.initial()
        op = DefectRelaxation(m,'ordinary')
        for _ in range(16):
            expected, actual = m.step(z), op.step(z)
            np.testing.assert_allclose(m.pack(actual),m.pack(expected),rtol=0,atol=1e-12)
            z = expected
        fft, ifft = np.fft.fft2, np.fft.ifft2
        for name in ['ordinary','fused','refresh','aligned_refresh','remembered_refresh']:
            with patch('numpy.fft.fft2', wraps=fft) as forward, patch('numpy.fft.ifft2', wraps=ifft) as inverse:
                DefectRelaxation(m,name).step(z)
                self.assertEqual(forward.call_count,2)
                self.assertEqual(inverse.call_count,2)

    def test_nonconstant_source_exact_fixed_point(self):
        f = 100+2*np.sin(2*np.pi*np.arange(32)[None,:]/32)
        m = ReducedMeyerMap(f,.05,40)
        zero = np.zeros_like(f)
        u = np.full_like(f,100)
        tw = -np.cumsum((m.cw/m.etaw)*(f-u),axis=1)
        z = State(u,zero.copy(),zero.copy(),zero.copy(),tw,zero.copy())
        self.assertLess(np.max(np.abs(tw)),m.rw)
        for name in ['ordinary','reverse','alternating','fused','refresh','fused_refresh',
                     'transport','reobserve','aligned_refresh','remembered_refresh']:
            op = DefectRelaxation(m,name)
            current = z
            for _ in range(8):
                current = op.step(current)
                np.testing.assert_allclose(m.pack(current),m.pack(z),rtol=0,atol=1e-11)

    def test_memory_feasibility_and_fused_equations(self):
        m = ReducedMeyerMap(np.random.default_rng(4).normal(100,50,(12,16)),.05,40)
        for name in ['ordinary','fused','refresh','aligned_refresh','remembered_refresh']:
            op = DefectRelaxation(m,name)
            z = m.initial()
            for _ in range(20):
                old = z
                z = op.step(z)
                for x,tx,ty,radius in [(z.u,z.tux,z.tuy,m.ru),(z.w,z.twx,z.twy,m.rw)]:
                    gx,gy = grad(x)
                    self.assertLessEqual(float(np.max(np.hypot(tx-gx,ty-gy))),radius+1e-12)
                if name == 'fused':
                    pu = project_disk(old.tux,old.tuy,m.ru)
                    pw = project_disk(old.twx,old.twy,m.rw)
                    hu = m.cu*z.u-m.etau*div(*grad(z.u))
                    hw = m.cw*z.w-m.etaw*div(*grad(z.w))
                    ru = m.cu*(old.u+z.w)-m.etau*div(old.tux-2*pu[0],old.tuy-2*pu[1])
                    rw = m.cw*(m.image-z.u)-m.etaw*div(old.twx-2*pw[0],old.twy-2*pw[1])
                    np.testing.assert_allclose(hu,ru,atol=1e-12,rtol=0)
                    np.testing.assert_allclose(hw,rw,atol=1e-12,rtol=0)

    def test_retained_memory_and_local_projection_work(self):
        m = ReducedMeyerMap(np.random.default_rng(9).normal(100,30,(12,16)),.05,40)
        op = DefectRelaxation(m,'aligned_refresh')
        z = m.initial()
        for _ in range(40):
            z = op.step(z)
            for x,tx,ty,b in [(z.u,z.tux,z.tuy,op.memory[0]),(z.w,z.twx,z.twy,op.memory[1])]:
                gx,gy = grad(x)
                np.testing.assert_allclose(tx-gx,b[0],rtol=0,atol=1e-12)
                np.testing.assert_allclose(ty-gy,b[1],rtol=0,atol=1e-12)
        rng = np.random.default_rng(17)
        px,py = project_disk(*rng.normal(size=(2,40,40)),2.)
        gx,gy = rng.normal(size=(2,40,40))
        qx,qy = project_disk(px+gx,py+gy,2.)
        ex,ey = qx-px,qy-py
        self.assertGreaterEqual(float(np.min(gx*ex+gy*ey-ex*ex-ey*ey)),-1e-12)


if __name__ == '__main__':
    unittest.main()
