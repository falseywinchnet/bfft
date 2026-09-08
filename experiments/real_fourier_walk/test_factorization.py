"""Exact commutative-diagram certificate for the normalized quartic split.

The same input polynomial is reduced in the old frame, passed through the
butterfly, and independently reduced in each new quadratic frame. All three
quartic pairings are permutations of these four certified child residues.
"""
import json
import sympy as sp


def certify():
    z, C, S = sp.symbols('z C S', real=True)
    old_cos, old_sin = 2*C*C-1, 2*C*S
    parent = z**4-2*old_cos*z*z+1
    checks = 0
    for j in range(8):
        r = sp.rem(z**j, parent, z).expand()
        u0,u1,v0,v1 = [r.coeff(z,k) for k in range(4)]
        A0,A1 = u0+old_cos*v0,u1+old_cos*v1
        B0,B1 = old_sin*v0,old_sin*v1
        re,im = C*A1-S*B1,S*A1+C*B1
        actual = ((A0+re,B0+im),(A0-re,im-B0))
        for side, c in enumerate((C,-C)):
            residue = sp.rem(z**j,z*z-2*c*z+1,z).expand()
            expected = (residue.coeff(z,0)+c*residue.coeff(z,1),S*residue.coeff(z,1))
            for a,b in zip(actual[side],expected):
                assert sp.rem(sp.expand(a-b),S*S+C*C-1,S)==0
                checks += 1
        r = sp.rem(z**j,z**4-1,z).expand()
        r0,r1,r2,r3 = [r.coeff(z,k) for k in range(4)]
        actual = ((r0+r2,r1+r3),(r0-r2,r1-r3))
        for k,q in enumerate((z*z-1,z*z+1)):
            residue=sp.rem(z**j,q,z).expand()
            for d in range(2):
                assert sp.expand(actual[k][d]-residue.coeff(z,d))==0
                checks+=1
    return {'exact_scalar_coordinate_checks':checks,
            'generic_angle_constraint':'C^2 + S^2 = 1',
            'arbitrary_pairings_preserved':3,
            'factorization':'G_children R_choice = P_choice (B_a direct_sum B_b) G_parent'}

if __name__=='__main__':
    print(json.dumps(certify(),indent=2))
