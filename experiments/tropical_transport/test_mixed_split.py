"""General-marginal split of mixed coefficients at a Sinkhorn fixed point:
2<v_k,B(v_i,v_j)>_c = (s_k - s_i s_j)^2 T_c + 2 s_k s_i s_j (T_c - T_a),
T_c=E_c[v_k v_i v_j], T_a=E_a[u_k u_i u_j], u=Pv/s (left singular modes)."""
import unittest
import numpy as np
from .probe import Pair, problem
from .normal_form import conditionals, bilinear
from .mixed import fixed


class Split(unittest.TestCase):
    def test_split(self):
        worst = 0; asym = []
        for seed, eps in ((0, 0.05), (1, 0.02), (2, 0.1)):
            C, p, q = problem(24, seed); S = Pair(C, p, q, eps)
            ys = fixed(S, 24); P, Q = conditionals(S, ys); a = np.exp(S.lp); c = a@P
            J = Q@P; w = np.sqrt(c); M = (w[:, None]*J)/w[None, :]
            lam, U = np.linalg.eigh(.5*(M+M.T)); o = np.argsort(-lam)[1:7]
            s = np.sqrt(np.clip(lam[o], 0, None)); V = U[:, o]/w[:, None]; Ul = (P@V)/s
            # left modes a-orthonormal
            G = Ul.T@(a[:, None]*Ul); self.assertLess(np.abs(G-np.eye(6)).max(), 1e-8)
            for i in range(6):
                for j in range(6):
                    Bij = bilinear(P, Q, V[:, i], V[:, j])
                    for k in range(6):
                        lhs = 2*V[:, k]@(c*Bij)
                        Tc = c@(V[:, k]*V[:, i]*V[:, j]); Ta = a@(Ul[:, k]*Ul[:, i]*Ul[:, j])
                        rhs = (s[k]-s[i]*s[j])**2*Tc+2*s[k]*s[i]*s[j]*(Tc-Ta)
                        worst = max(worst, abs(lhs-rhs)/max(1e-12, abs(lhs)+abs(rhs)))
                        asym.append(abs(Tc-Ta)/max(abs(Tc), abs(Ta), 1e-12))
        print('max rel error', worst, 'median |Tc-Ta|/max', np.median(asym))
        self.assertLess(worst, 1e-6)


if __name__ == '__main__':
    unittest.main()
