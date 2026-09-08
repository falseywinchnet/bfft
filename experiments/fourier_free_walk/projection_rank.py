"""Exact lower-bound witnesses independent of a radix tree or intermediate basis.

For a Q-linear functional ell on the constant field with ell(1)=0, apply ell
entrywise to all current wire coefficient rows. Rational arithmetic cannot
increase their row rank; one nonrational scalar multiplication can add at most
one row direction. Thus rank(ell(F)) bounds such multiplications from below.
This is a linear constant-coefficient circuit bound, not a latency bound.
"""
import argparse,json,pathlib,random
import sympy as s
import numpy as np


def matrices(n):
    if n<4 or n&(n-1):raise ValueError('this field certificate uses power-of-two N>=4')
    t=s.Symbol('c');degree=n//4;mod=s.Poly(s.chebyshevt(degree,t),t)
    # 2*T_degree(y/2) is monic Eisenstein at 2 for degree=2^k>1.
    # This certifies that the quotient basis really is the Fourier number field.
    if degree>1:
        eisenstein=s.Poly(2*s.chebyshevt(degree,t/2),t)
        coefficients=eisenstein.all_coeffs()
        assert coefficients[0]==1 and all(c%2==0 for c in coefficients[1:])
        assert coefficients[-1]%4==2
    cosines=[s.rem(s.Poly(s.chebyshevt(k,t),t),mod) for k in range(n)]
    def cos(k):return cosines[k%n]
    values=[[s.Poly(1,t)]*n]
    for k in range(1,n//2):
        values.extend([[cos(k*j) for j in range(n)],[-cos(n//4-k*j) for j in range(n)]])
    values.append([s.Poly((-1)**j,t) for j in range(n)])
    return [s.Matrix(n,n,lambda i,j:values[i][j].nth(r)) for r in range(degree)]


def modular_witness(m,prime=1000003):
    """An exact nonzero minor modulo a prime certifies a rational rank bound."""
    a=np.array([[int(s.numer(x))*pow(int(s.denom(x)),-1,prime)%prime for x in row]
                for row in m.tolist()],dtype=np.int64)
    row_ids=list(range(a.shape[0]));pivot_rows=[];pivot_cols=[];rank=0
    for col in range(a.shape[1]):
        choices=np.flatnonzero(a[rank:,col])
        if not len(choices):continue
        pivot=rank+int(choices[0]);a[[rank,pivot]]=a[[pivot,rank]]
        row_ids[rank],row_ids[pivot]=row_ids[pivot],row_ids[rank]
        pivot_rows.append(row_ids[rank]);pivot_cols.append(col)
        a[rank]=a[rank]*pow(int(a[rank,col]),-1,prime)%prime
        if rank+1<len(a):a[rank+1:]=(a[rank+1:]-a[rank+1:,col:col+1]*a[rank])%prime
        rank+=1
        if rank==len(a):break
    return {'rank':rank,'prime':prime,'minor_rows':pivot_rows,'minor_columns':pivot_cols}


def probe(n,trials=8):
    a=matrices(n);rng=random.Random(890+n);best_rank=0;best_weights=[];checks=[];witness=None
    if len(a)==1:return {'N':n,'field_degree':1,'rank_lower_bound':0,'weights':[],'trials':[]}
    for trial in range(trials):
        weights=[1 if trial==0 else rng.randint(-5,5) for _ in a[1:]]
        m=s.zeros(n)
        for w,b in zip(weights,a[1:]):m+=w*b
        current=modular_witness(m);rank=current['rank'];checks.append(rank)
        if rank>best_rank:best_rank=rank;best_weights=weights;witness=current
    return {'N':n,'field_degree':len(a),'rank_lower_bound':best_rank,
            'weights':best_weights,'trials':checks,'nonzero_minor_witness':witness,
            'scope':'nonrational scalar multiplications in any exact linear constant-coefficient circuit; rational arithmetic and copies are free in this bound'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--sizes',default='4,8,16,32,64');args=p.parse_args()
    rows=[]
    for n in map(int,args.sizes.split(',')):
        row=probe(n);rows.append(row);print(json.dumps(row),flush=True)
        pathlib.Path(args.out).write_text(json.dumps(rows,indent=2)+'\n')
