import argparse
import json
import random
import math
import sympy as sp
import numpy as np
from newton import Walk,swap,T,walk_rfft
from quotient_fft import transform


def exact_cells():
    a0,a1,b0,b1,p0,p1,q0,q1=sp.symbols("a0 a1 b0 b1 p0 p1 q0 q1",real=True)
    a,b=swap((a0,a1),(b0,b1),(p0,p1),(q0,q1))
    difference=sp.expand(a0+a1*T+(T*T+p1*T+p0)*(b0+b1*T)
                         -(a[0]+a[1]*T+(T*T+q1*T+q0)*(b[0]+b[1]*T)))
    assert difference==0
    aa,bb=swap(a,b,(q0,q1),(p0,p1))
    assert all(sp.expand(x-y)==0 for x,y in zip((*aa,*bb),(a0,a1,b0,b1)))
    rng=random.Random(7043)
    w=Walk.samples([sp.Integer(rng.randrange(-4,5)) for _ in range(16)])
    original=w.polynomial_oracle()
    for _ in range(80):
        if rng.randrange(2):w.exchange(rng.randrange(7))
        else:w.replace(rng.randrange(8),(sp.Rational(rng.randrange(-2,3),2),sp.Rational(rng.randrange(-3,4),3)))
        assert w.polynomial_oracle()==original
    # Actual real-to-DFT endpoint using only local exchanges/replacements.
    n=8
    factors=[(-1,0)]+[(1,-2*sp.cos(2*sp.pi*k/n)) for k in range(1,n//2)]
    checks=0;steps=0
    for j in range(n):
        w=Walk.samples([sp.Integer(i==j) for i in range(n)])
        for i,f in enumerate(factors):w.replace(i,f)
        for factor in factors:
            u,v=w.expose(w.factors.index(factor))
            q=T*T+factor[1]*T+factor[0]
            expected=sp.rem(T**j,q,T)
            assert sp.simplify(u-expected.coeff(T,0))==0
            assert sp.simplify(v-expected.coeff(T,1))==0
            checks+=1
        steps=w.steps
    # Coxeter/braid path relation: two different local three-swap routes.
    vals=[(1,2),(3,4),(5,6)];polys=[(0,0),(1,sp.Rational(1,3)),(1,-1)]
    l=Walk(vals,polys);r=Walk(vals,polys)
    for i in (0,1,0):l.exchange(i)
    for i in (1,0,1):r.exchange(i)
    assert l.factors==r.factors and l.values==r.values
    # Same four-stream law at larger scales: replace t by t^h, allowing each
    # coefficient to be a polynomial of degree<h. This is a whole-packet edge.
    for h in (1,2,4,8):
        values=[sum(sp.Integer(rng.randrange(-4,5))*T**j for j in range(h)) for _ in range(4)]
        aa,bb=swap(values[:2],values[2:],(1,sp.Rational(2,3)),(1,-sp.Rational(2,3)))
        old=values[0]+T**h*values[1]+(T**(2*h)+sp.Rational(2,3)*T**h+1)*(values[2]+T**h*values[3])
        new=aa[0]+T**h*aa[1]+(T**(2*h)-sp.Rational(2,3)*T**h+1)*(bb[0]+T**h*bb[1])
        assert sp.expand(old-new)==0
    return {"symbolic_general_cell_identity":True,"symbolic_inverse_identity":True,
            "random_turns_and_replacements":80,"real_state_dimension":16,
            "braid_relation_exact":True,"N8_fourier_residue_checks":checks,
            "N8_walk_exchanges":steps,"whole_packet_exchange_h":[1,2,4,8]}


def walk_sweep():
    rng=np.random.default_rng(9206);results=[]
    for n in (8,16,32,64,128):
        x=rng.normal(size=n);z=np.fft.rfft(x)
        reference=np.empty(n);reference[0]=z[0].real;reference[-1]=z[-1].real
        reference[1:-1:2]=z[1:-1].real;reference[2:-1:2]=-z[1:-1].imag
        for order in ("natural","bitreverse"):
            actual,counts=walk_rfft(x,order)
            relative=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
            results.append({"N":n,"order":order,"relative_l2_error":relative,**counts})
    return results


def sweep():
    rng=np.random.default_rng(7043)
    rows=[]
    for exponent in range(2,15):
        n=2**exponent
        samples=[rng.normal(size=n),np.ones(n),np.eye(1,n,n//3).ravel()]
        for normalized in (False,True):
            maxerr=0.;maxrel=0.
            for x in samples:
                y,counts=transform(x,normalized)
                reference=np.fft.rfft(x)
                packed=np.empty(n);packed[0]=reference[0].real;packed[-1]=reference[-1].real
                packed[1:-1:2]=reference[1:-1].real;packed[2:-1:2]=-reference[1:-1].imag
                maxerr=max(maxerr,float(np.max(np.abs(y-packed))))
                maxrel=max(maxrel,float(np.linalg.norm(y-packed)/np.linalg.norm(packed)))
                for order in ("breadth","turn"):
                    alternate,_=transform(x,normalized,order)
                    assert np.array_equal(y,alternate)
                if normalized:np.testing.assert_allclose(y,packed,atol=2e-10,rtol=2e-11)
            rows.append({"N":n,"normalized":normalized,"max_abs_error":maxerr,
                         "max_rel_l2_error":maxrel,**counts})
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);args=parser.parse_args()
    exact=exact_cells();print(json.dumps(exact),flush=True)
    rows=sweep()
    with open(args.out,"w") as f:json.dump({"exact":exact,"fft_sweep":rows,"walk_sweep":walk_sweep()},f,indent=2);f.write("\n")
    print("Real FFT sweep and all schedule-equivalence checks passed",flush=True)
