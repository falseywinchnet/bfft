import unittest
import numpy as np
from packet import Packet, direct_matrices


class PacketTests(unittest.TestCase):
    def test_explicit_matrix_oracle(self):
        for r in range(1,5):
            for slope in (1,3,-1):
                p=Packet(r,slope)
                self.assertEqual(len(set(p.pack.tolist())),p.n)
                Q=p.matrix()
                F,G=direct_matrices(p.m)
                col,phase=p.monomial()
                P=np.zeros_like(Q)
                P[np.arange(p.n),col]=phase
                np.testing.assert_allclose(Q@Q.conj().T,np.eye(p.n),atol=3e-14)
                np.testing.assert_allclose(Q@G@Q.conj().T,P,atol=3e-14)
                C=Q@F@Q.conj().T@P.conj().T
                for i in range(p.n):
                    x=np.eye(1,p.n,i).ravel()
                    np.testing.assert_allclose(p.carry_blocks(x),C[:,i],atol=3e-14)

    def test_size_growth_and_patterns(self):
        rng=np.random.default_rng(7043)
        for r in range(1,9):
            p=Packet(r)
            for x in (np.ones(p.n),(-1.)**np.arange(p.n),
                      np.exp(2j*np.pi*7*np.arange(p.n)/p.n),
                      rng.normal(size=p.n)+1j*rng.normal(size=p.n)):
                np.testing.assert_allclose(p.inverse(p.forward(x)),x,atol=2e-13)
                np.testing.assert_allclose(p.carry_blocks(x),p.carry(x),atol=2e-13)
                actual=p.transform_blocks(x)
                expected=np.fft.fft(x,norm="ortho")
                np.testing.assert_allclose(actual,expected,atol=2e-11)
                self.assertLess(abs(np.linalg.norm(actual)-np.linalg.norm(x)),2e-11)
                np.testing.assert_allclose(p.transform_cancelled(x)/p.m,expected,atol=2e-11)

    def test_carry_invariant_labels(self):
        for r in range(1,7):
            p=Packet(r)
            for pp, beta in ((0,0),(p.a-1,p.a-1)):
                z=np.zeros((p.b,p.a,p.b,p.a),complex)
                z[0,pp,0,beta]=1
                out=p.carry_blocks(z.ravel()).reshape(z.shape)
                allowed=np.zeros(z.shape,bool)
                allowed[:,pp,:,beta]=True
                self.assertEqual(np.count_nonzero(np.abs(out[~allowed])>1e-14),0)


if __name__ == "__main__":
    unittest.main()
