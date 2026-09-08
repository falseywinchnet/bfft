"""Exact basis certificates for changing locations, collisions, FFT/ODFT."""
import json
import argparse
import sympy as sp
from atlas import T
from moving import MovingAtlas


def certificate():
    a=MovingAtlas(8)
    fft=a.fourier_chart()
    odd=a.fourier_chart(True)
    # Continuous family t²+lambda: its two roots collide inside one packet
    # at lambda=0. The other packet stays coprime across [-1/4,1/4].
    collision_charts=[a.chart([T*T+v,(T-2)**6]) for v in
                      (-sp.Rational(1,4),0,sp.Rational(1,4))]
    # Move the k=2 conjugate pair to cos(theta)=1/3, still on the unit circle.
    single_move=a.chart([*fft[:2],T*T-sp.Rational(2,3)*T+1,*fft[3:]])
    path=[fft,single_move,*collision_charts,odd,fft]
    checks=0
    for j in range(a.n):
        x=[int(i==j) for i in range(a.n)]
        initial=a.input(x)
        state=initial
        for target in path:
            state=a.transport(state,target)
            assert state[1]==a.oracle(x,target)
            checks+=1
        assert a.transport(state,initial[0])==initial
        direct_odd=a.transport(initial,odd)
        via_fft=a.transport(a.transport(initial,fft),odd)
        assert via_fft==direct_odd
        for odd_flag in (False,True):
            endpoint=a.transport(initial,a.fourier_chart(odd_flag))
            actual=a.readout(endpoint,odd_flag)
            expected=[]
            if not odd_flag:expected.append(1)
            for k in (range(1,8,2) if odd_flag else range(2,8,2)):
                expected.extend([sp.chebyshevt(k*j,a.c),sp.chebyshevt(abs(4-k*j),a.c)])
            if not odd_flag:expected.append((-1)**j)
            assert all(a.domain.from_sympy(u-v)==a.domain.zero for u,v in zip(actual,expected))
    # All old packets contribute to the changed quadratic, despite other
    # target packets being preserved exactly. Establish by source-coordinate
    # basis perturbations, not by looking at nonzero plan coefficients alone.
    dependencies=[]
    for i,p in enumerate(fft):
        nonzero=False
        for degree in range(p.degree()):
            residues=[a.poly(0) for _ in fft];residues[i]=a.poly(T**degree)
            nonzero|=not a.transport((fft,tuple(residues)),single_move)[1][2].is_zero
        if nonzero:dependencies.append(i)
    assert dependencies==list(range(len(fft)))
    try:a.chart([T,T,(T-2)**6])
    except ValueError:rejected=True
    else:raise AssertionError("Shared root across distinct packets was accepted")
    return {"N":8,"exact_basis_transition_checks":checks,"real_state_dimension":8,
            "FFT_and_ODFT_readout_exact":True,"route_independent":True,
            "loop_exact":True,"internal_collision_traversed":True,
            "cross_packet_collision_rejected":rejected,
            "changed_quadratic_source_packets":dependencies,
            "path":[[str(p.as_expr()) for p in chart] for chart in path]}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);args=p.parse_args()
    result=certificate()
    with open(args.out,"w") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="path"}),flush=True)
