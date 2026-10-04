"""Exact checks from the authorized Codex/Claude dialogue, 2026-09-30.

Scope: symmetric doubly stochastic positive kernels, uniform marginals,
quotient by the constant gauge, at the fixed point y=0. Write
    g(y) = -log(K exp(y)), F = g o g, J = K^2.
For probability-orthonormal K eigenvectors v_i with eigenvalues s_i,
the quadratic coefficient B = D^2 F/2 obeys
    <v_k,B(v_i,v_j)> = (s_k-s_i*s_j)^2 <v_k,v_i*v_j>/2.
Proof: expand B(u,w)=[K^2(uw)-2K((Ku)(Kw))+(K^2u)(K^2w)]/2
and use self-adjointness. Products inside vectors are componentwise.

For phi(F)=J phi and phi(y)=y+h(y,y)+O(||y||^3),
    h_kij=(s_k-s_i*s_j)/(2*(s_k+s_i*s_j))*<v_k,v_i*v_j>.
This continuous choice remains valid at the positive-branch resonance.
If s_i,s_j,s_k>0, the scalar prefactor has magnitude at most 1/2.
This is coefficientwise, NOT a dimension-independent tensor norm bound.
Zero modes require separate treatment. Entrywise positivity does not imply
positive spectrum: the negative-branch example below has a genuine
nonzero resonant quadratic term and cannot be C^2-linearized locally.

The resonant quadratic normal form has a finite monomial lift; this is
classical normal-form/Carleman algebra, not a novelty or performance claim.
The conjectural research target is an acquired, parity-aware finite lift
with useful full-trajectory remainder and complete-cost guarantees.
Claude independently verified the counterexample and generalized the split.
For the whitened coupling singular modes P v_i = s_i u_i, Q u_i=s_i v_i,
let T_b=E_b[v_k v_i v_j] and T_a=E_a[u_k u_i u_j]. Direct adjointness gives
    2 B_kij=(s_k-s_i*s_j)^2 T_b+2*s_k*s_i*s_j*(T_b-T_a).
The first term cancels at positive-branch resonance; the left/right triple
moment mismatch need not. This identity was independently checked
algebraically here; Claude reports numerical verification in its own tests.
Its Dirichlet bound |<w,B(u,u)>_c| <= ||w||inf <u,(I-J)u>_c is stronger
than the first self-mode estimate used to initiate this dialogue.

No assertion of full Sinkhorn finite closure follows from these checks.

Run this small standard-library exact oracle with python3 -m
experiments.entropic_transport_closure.resonance_dialogue.
"""
from fractions import Fraction as Q


def matvec(A, v):
    return [sum(a*x for a, x in zip(row, v)) for row in A]


def dot(u, v):
    return sum(a*b for a, b in zip(u, v))


def quadratic(K, u, w):
    pu, pw = matvec(K, u), matvec(K, w)
    ju, jw = matvec(K, pu), matvec(K, pw)
    first = matvec(K, matvec(K, [a*b for a, b in zip(u, w)]))
    second = matvec(K, [a*b for a, b in zip(pu, pw)])
    return [(a-2*b+c*d)/2 for a,b,c,d in zip(first,second,ju,jw)]


def kernel(s):
    # eigenvectors u=(1,-1,0), w=(1,1,-2); eigenvalues .3,s.
    u, w = [1,-1,0], [1,1,-2]
    return [[Q(1,3)+Q(3,20)*u[i]*u[j]+s*w[i]*w[j]/6
             for j in range(3)] for i in range(3)]


def main():
    u, w = [Q(1),Q(-1),Q(0)], [Q(1),Q(1),Q(-2)]
    lam = Q(9,100)
    for s in [Q(-9,100),Q(9,100)]:
        K=kernel(s)
        assert all(x>0 for row in K for x in row)
        assert all(sum(row)==1 for row in K)
        assert matvec(K,u)==[Q(3,10)*x for x in u]
        assert matvec(K,w)==[s*x for x in w]
        coefficient=dot(w,quadratic(K,u,u))/dot(w,w)
        expected=(s-Q(9,100))**2/Q(6)
        assert coefficient==expected
        assert s*s==lam*lam
        print('half-step eigenvalue:',s,'resonant w coefficient:',coefficient)
    beta=Q(27,5000)
    assert dot(w,quadratic(kernel(Q(-9,100)),u,u))/dot(w,w)==beta
    mu=lam*lam
    x0,z0=Q(2,7),Q(-3,11)
    for horizon in [1,2,7,32]:
        x,z=x0,z0
        for _ in range(horizon):
            x,z=lam*x,mu*z+beta*x*x
        expected=mu**horizon*z0+horizon*mu**(horizon-1)*beta*x0*x0
        assert z==expected
        assert x==lam**horizon*x0
    print('Exact resonance, positive-branch cancellation, and lifted powers: PASS')


if __name__=='__main__':
    main()
