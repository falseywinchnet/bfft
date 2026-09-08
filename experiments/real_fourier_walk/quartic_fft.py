"""Bounded quartic packets with selectable cross-parent root regrouping.

Every live packet is a product of two real quadratic factors in t^h.
Splitting halves h and permits any pairing of four child quadratics.
Thus packet order stays four and data dimension stays N at every frontier.
"""
import math
import numpy as np


def quadratic(theta):
    return np.array([-1.,0.,1.]) if theta is None else np.array([1.,-2*math.cos(theta),1.])


def children(theta):
    return (None,math.pi/2) if theta is None else (theta/2,math.pi-theta/2)


def remainder(rows,modulus):
    r=rows.copy()
    for k in range(len(r)-1,len(modulus)-2,-1):
        for j in range(len(modulus)-1):
            r[k-len(modulus)+1+j]-=modulus[j]*r[k]
    return r[:len(modulus)-1]


def transform(x,policy="separated",seed=7043):
    x=np.asarray(x,float);n=len(x)
    if n<4 or n&(n-1):raise ValueError("N must be a power of two >=4")
    rng=np.random.default_rng(seed)
    tasks=[(x.copy(),(None,math.pi/2))]
    out=np.empty(n);splits=0;crossings=0;min_gap=2.;frontier=n;peak=n
    while tasks:
        v,thetas=tasks.pop();h=len(v)//4;frontier-=len(v)
        if h==1:
            for theta in thetas:
                u,w=remainder(v.reshape(4,1),quadratic(theta))[:,0]
                if theta is None:out[0]=u+w;out[-1]=u-w
                else:
                    k=round(theta*n/(2*math.pi))
                    out[2*k-1]=u+math.cos(theta)*w;out[2*k]=math.sin(theta)*w
            continue
        factors=children(thetas[0])+children(thetas[1])
        pairings=(((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))
        if callable(policy):choice=int(policy(splits,factors));assert choice in (0,1,2)
        elif policy=="sibling":choice=0
        elif policy=="random":choice=int(rng.integers(3))
        elif policy=="separated":
            if None in factors:
                # Keep the real endpoints paired with the quarter-turn factor.
                choice=next(i for i,pairs in enumerate(pairings) if any(
                    factors[a] is None and factors[b]==math.pi/2 or
                    factors[b] is None and factors[a]==math.pi/2 for a,b in pairs))
            else:
                score=lambda pairs:min(abs(math.cos(factors[a])-math.cos(factors[b])) for a,b in pairs)
                choice=max(range(3),key=lambda i:score(pairings[i]))
        else:raise ValueError(policy)
        splits+=1;crossings+=choice!=0
        rows=v.reshape(8,h//2)
        for a,b in pairings[choice]:
            target=(factors[a],factors[b])
            if None not in target:min_gap=min(min_gap,abs(math.cos(target[0])-math.cos(target[1])))
            modulus=np.convolve(quadratic(target[0]),quadratic(target[1]))
            child=remainder(rows,modulus).ravel()
            tasks.append((child,target));frontier+=len(child)
        peak=max(peak,frontier)
    return out,{"splits":splits,"cross_parent_regroupings":crossings,
                "minimum_cosine_gap":min_gap,"peak_frontier_coordinates":peak,
                "maximum_packet_polynomial_degree":4}
