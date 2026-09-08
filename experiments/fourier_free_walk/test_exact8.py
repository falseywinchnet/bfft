import unittest
import numpy as np
import sympy as s
import exact8

class Tests(unittest.TestCase):
    def test_actual_program_exactly(self):
        previous=exact8.HALF_ROOT2
        try:
            exact8.HALF_ROOT2=s.sqrt(2)/2
            for j in range(8):
                x=[s.Integer(k==j) for k in range(8)]
                result=exact8.transform_inplace(x)
                target=[s.S.One]
                for k in range(1,4):target.extend([s.cos(2*s.pi*k*j/8),-s.sin(2*s.pi*k*j/8)])
                target.append(s.Integer(-1)**j)
                self.assertTrue(all(s.simplify(a-b)==0 for a,b in zip(result,target)))
        finally:exact8.HALF_ROOT2=previous
    def test_random_real_input(self):
        rng=np.random.default_rng(860)
        for _ in range(100):
            x=rng.normal(size=8);r=np.fft.rfft(x)
            expected=[r[0].real]
            for z in r[1:-1]:expected.extend([z.real,z.imag])
            expected.append(r[-1].real)
            np.testing.assert_allclose(exact8.transform_inplace(x.copy()),expected,atol=3e-14)

if __name__=='__main__':unittest.main()
