import unittest
import sympy as s
from projection_rank import matrices,modular_witness,probe

class Tests(unittest.TestCase):
    def test_field_reconstructs_real_fourier_endpoint(self):
        for n in (4,8):
            parts=matrices(n);c=s.cos(2*s.pi/n)
            m=s.zeros(n)
            for power,a in enumerate(parts):m+=c**power*a
            rows=[[1]*n]
            for k in range(1,n//2):rows.extend([[s.cos(2*s.pi*k*j/n) for j in range(n)],[-s.sin(2*s.pi*k*j/n) for j in range(n)]])
            rows.append([(-1)**j for j in range(n)])
            self.assertTrue(all(s.simplify(x)==0 for x in m-s.Matrix(rows)))
    def test_nonzero_minor_is_exact(self):
        r=probe(8,2);self.assertEqual(r['rank_lower_bound'],2)
        m=s.zeros(8)
        for w,a in zip(r['weights'],matrices(8)[1:]):m+=w*a
        witness=r['nonzero_minor_witness'];minor=m.extract(witness['minor_rows'],witness['minor_columns'])
        self.assertNotEqual(minor.det()%witness['prime'],0)
    def test_modular_rank_is_only_a_lower_bound(self):
        r=modular_witness(s.diag(1000003,1))
        self.assertEqual(r['rank'],1)

if __name__=='__main__':unittest.main()
