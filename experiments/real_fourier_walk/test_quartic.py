import argparse,json,itertools
import numpy as np
import sympy as sp
from quartic_fft import transform,remainder


def exact_pairings():
    # Parent (y²-1)(y²+1); halve the variable y=z². All three pairings
    # of its four real child quadratics must yield a valid local split.
    t=sp.Symbol('t');s=sp.sqrt(2)
    factors=[t*t-1,t*t+1,t*t-s*t+1,t*t+s*t+1]
    pairings=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
    C,D=sp.symbols('C D',real=True)
    generic=[t*t-2*C*t+1,t*t+2*C*t+1,t*t-2*D*t+1,t*t+2*D*t+1]
    checks=0
    for family in (factors,generic):
        for pairs in pairings:
            moduli=[sp.expand(family[a]*family[b]) for a,b in pairs]
            if family==factors:assert sp.expand(moduli[0]*moduli[1])==t**8-1
            else:
                parent=(t**4+(2-4*C*C)*t*t+1)*(t**4+(2-4*D*D)*t*t+1)
                assert sp.expand(moduli[0]*moduli[1]-parent)==0
            for j in range(8):
                values=np.zeros((8,1),dtype=object);values[j,0]=sp.S.One
                for q in moduli:
                    coefficients=np.array([sp.expand(q).coeff(t,k) for k in range(5)],dtype=object)
                    actual=remainder(values,coefficients)[:,0]
                    expected=sp.rem(t**j,q,t)
                    assert all(sp.expand(actual[k]-expected.coeff(t,k))==0 for k in range(4))
                    checks+=1
    return {"pairings":3,"exact_basis_remainder_checks":checks,"generic_child_angles_symbolic":True}


def all_small_plans():
    count=0;maximum=0.
    for plan in itertools.product(range(3),repeat=3):
        for j in range(16):
            x=np.eye(1,16,j).ravel();reference=np.fft.rfft(x)
            y,_=transform(x,lambda node,factors:plan[node])
            target=np.empty(16);target[0]=reference[0].real;target[-1]=reference[-1].real
            target[1:-1:2]=reference[1:-1].real;target[2:-1:2]=-reference[1:-1].imag
            error=float(np.max(np.abs(y-target)))
            assert error<1e-12,(plan,j,error)
            maximum=max(maximum,error);count+=1
    return {"N":16,"plans":27,"basis_plan_checks":count,"maximum_absolute_error":maximum}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    rng=np.random.default_rng(9506);rows=[]
    for exponent in range(3,15):
        n=2**exponent;x=rng.normal(size=n);z=np.fft.rfft(x)
        target=np.empty(n);target[0]=z[0].real;target[-1]=z[-1].real
        target[1:-1:2]=z[1:-1].real;target[2:-1:2]=-z[1:-1].imag
        for policy in ('sibling','random','separated'):
            actual,counts=transform(x,policy)
            relative=float(np.linalg.norm(actual-target)/np.linalg.norm(target))
            absolute=float(np.max(np.abs(actual-target)))
            if policy=='separated':assert relative<1e-10,(n,relative)
            assert counts['peak_frontier_coordinates']==n
            rows.append({'N':n,'policy':policy,'relative_l2_error':relative,'max_abs_error':absolute,**counts})
    result={'exact':exact_pairings(),'all_small_plans':all_small_plans(),'sweep':rows}
    with open(args.out,'w') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(rows[-3:]),flush=True)
