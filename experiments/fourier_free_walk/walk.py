"""Offline real-basis walk synthesis and in-place N-real-coordinate execution.

No Bruun, radix, root-packet, FFT, or prescribed pair-matching rules are used.
Search is finite; the admissible elementary GL(2,R) state model is broader.
"""
from dataclasses import dataclass
import itertools
import math
import numpy as np

TOL=2e-11


def fourier(n):
    """Orthonormal real DFT; physical rows DC, Re1, Im1, ..., Nyquist."""
    if n<4 or n%2:raise ValueError('even N >=4 required')
    t=np.arange(n,dtype=float)
    rows=[np.ones(n)/math.sqrt(n)]
    for k in range(1,n//2):
        rows.extend([np.cos(2*np.pi*k*t/n)*math.sqrt(2/n),
                     -np.sin(2*np.pi*k*t/n)*math.sqrt(2/n)])
    rows.append((-1.)**t/math.sqrt(n))
    return np.array(rows)


def inverse2(g):
    a,b,c,d=np.asarray(g).ravel();det=a*d-b*c
    if abs(det)<1e-14:raise ValueError('singular cell')
    return np.array([[d,-b],[-c,a]])/det


@dataclass
class Gate:
    i:int
    j:int
    g:np.ndarray


def annihilators(a,b,oblique):
    """All distinct zero-making directions from the current pair of rows.

    These depend on the current full basis, not a preselected Fourier angle.
    Orthogonal completion gives O(2) cells. Oblique completion independently
    chooses both directions, from a bounded shortlist to control branching.
    """
    directions=[np.array([1.,0.]),np.array([0.,1.])]
    seen={(1.,0.),(0.,1.)}
    for x,y in zip(a,b):
        h=math.hypot(x,y)
        if h<TOL:continue
        v=np.array([y,-x])/h
        if v[np.argmax(np.abs(v))]<0:v=-v
        key=tuple(np.round(v,10))
        if key not in seen:directions.append(v);seen.add(key)
    for v in directions:
        if min(abs(v))<TOL:continue
        yield np.array([v,[-v[1],v[0]]])
    if oblique:
        # Count and retain this explicit truncation in the experiment contract.
        rank=sorted(directions,key=lambda v:(np.count_nonzero(abs(v[0]*a+v[1]*b)>TOL),
                                             np.sum(abs(v[0]*a+v[1]*b))))[:6]
        for u,v in itertools.combinations(rank,2):
            g=np.array([u,v]);det=np.linalg.det(g)
            if abs(det)>.15 and abs(np.dot(u,v))>1e-8:yield g


def synthesize(n,seed=0,oblique=False,explore=0.):
    """Sparse-support guided walk backwards from the actual ordered endpoint.

    Positive explore samples near-best moves, including fill-in when offered.
    Target diagonalization is not assumed; terminal permutation is returned
    explicitly and must be routed or rejected by the separate orientation walk.
    """
    rng=np.random.default_rng(seed);m=fourier(n);gates=[];trace=[];seen=set();peak_condition=1.
    attempted=0
    for step in range(4*n*n):
        counts=np.count_nonzero(abs(m)>TOL,axis=1)
        if np.all(counts==1) and len(set(np.argmax(abs(m),axis=1)))==n:break
        key=np.round(m,9).tobytes()
        if key in seen:break
        seen.add(key)
        candidates=[]
        for i in range(n):
            for j in range(i+1,n):
                a,b=m[i],m[j]
                if not np.any((abs(a)>TOL)&(abs(b)>TOL)):continue
                old_count=counts[i]+counts[j]
                old_l1=np.sum(abs(a))+np.sum(abs(b))
                for g in annihilators(a,b,oblique):
                    y=g@np.array([a,b]);attempted+=1
                    new_count=np.count_nonzero(abs(y)>TOL)
                    # Support first, then L1. A small determinant penalty stops
                    # nearly dependent oblique rows looking free in this screen.
                    gain=old_count-new_count
                    l1gain=old_l1-np.sum(abs(y))
                    penalty=-math.log(abs(np.linalg.det(g)))
                    score=gain+.15*l1gain-.2*penalty
                    if score>1e-9:candidates.append((score,i,j,g,y,gain))
        if not candidates:break
        candidates.sort(key=lambda x:x[0],reverse=True)
        best=candidates[0][0]
        short=[c for c in candidates[:32] if c[0]>=best-explore]
        # Even explore=0 visits tied choices, rather than fixing a radix rule.
        pick=short[int(rng.integers(len(short)))]
        score,i,j,g,y,gain=pick
        m[[i,j]]=y
        # Numeric zero threshold is only a search decision. Do not round the
        # executed matrices or erase any residual from validation.
        gates.append(Gate(i,j,g))
        condition=float(np.linalg.cond(m));peak_condition=max(peak_condition,condition)
        trace.append({'pair':[i,j],'support':int(np.count_nonzero(abs(m)>TOL)),
                      'support_gain':int(gain),'condition':condition,
                      'oblique':bool(abs(np.dot(g[0],g[1]))>1e-8)})
        if condition>1e10:break
    permutation=tuple(int(x) for x in np.argmax(abs(m),axis=1))
    residual=m.copy();residual[np.arange(n),permutation]=0
    complete=len(set(permutation))==n and np.max(abs(residual))<TOL
    return {'n':n,'gates':gates,'terminal':m,'permutation':permutation,
            'complete':bool(complete),'trace':trace,'attempted_cells':attempted,
            'peak_condition':peak_condition}


def orient(p,edges,cap=100000):
    """All retained cell orientations, with no terminal placement operation.

    Mapping p[logical wire] = current physical address. At an existing gate,
    retain or exchange its two outputs. Both choices use the same two stores.
    Exact enumeration while frontier <= cap; beyond that this is a beam.
    """
    n=len(p);identity=tuple(range(n));states={tuple(p):0};peak=1;pruned=0
    if tuple(p)==identity:
        return {'success':True,'bits':0,'peak_states':1,'pruned_states':0,
                'exhaustive':True,'identity_witness':True}
    # Lower-bound pruning: a label can only travel through future edge contacts.
    future=[None]*(len(edges)+1);reach=[1<<i for i in range(n)]
    future[-1]=tuple(reach)
    for t in range(len(edges)-1,-1,-1):
        i,j=edges[t];reach=reach.copy();reach[i]=reach[j]=reach[i]|reach[j]
        future[t]=tuple(reach)
    for step,(i,j) in enumerate(edges):
        next_states={}
        for state,path in states.items():
            swapped=list(state);swapped[i],swapped[j]=swapped[j],swapped[i]
            for choice,new in ((0,state),(1,tuple(swapped))):
                # To end with identity, token currently at wire k must be able
                # to reach logical wire new[k] in the remaining gate network.
                if any(not(future[step+1][k]>>new[k]&1) for k in range(n)):continue
                next_states.setdefault(new,path|(choice<<step))
        peak=max(peak,len(next_states))
        if len(next_states)>cap:
            ranked=sorted(next_states,key=lambda s:(sum(k!=x for k,x in enumerate(s)),
                                                   sum(abs(k-x) for k,x in enumerate(s))))
            pruned+=len(next_states)-cap
            next_states={s:next_states[s] for s in ranked[:cap]}
        states=next_states
        if not states:break
    result=states.get(identity)
    return {'success':result is not None,'bits':result,'peak_states':peak,
            'pruned_states':pruned,'exhaustive':pruned==0}


def compile_walk(result,routing):
    if not result['complete'] or not routing['success']:raise ValueError('walk not admitted')
    n=result['n'];mapping=list(result['permutation']);physical=[]
    # M_terminal x = D P x. The logical values begin at their source physical
    # slots. Scaling is fused below into first real cells, not a gather pass.
    initial=np.empty(n)
    for i,k in enumerate(mapping):initial[k]=result['terminal'][i,k]
    for step,gate in enumerate(reversed(result['gates'])):
        i,j=gate.i,gate.j;u,v=mapping[i],mapping[j];g=inverse2(gate.g)
        if routing['bits']>>step&1:
            g=g[::-1].copy();mapping[i],mapping[j]=v,u
        physical.append(Gate(u,v,g))
    assert mapping==list(range(n))
    first={};last={}
    for k,gate in enumerate(physical):
        for row,wire in enumerate((gate.i,gate.j)):
            first.setdefault(wire,(k,row));last[wire]=(k,row)
    scale=np.full(n,math.sqrt(n/2));scale[0]=scale[-1]=math.sqrt(n)
    for wire in range(n):
        if wire not in first:raise ValueError('untouched wire requires separate scale')
        k,col=first[wire];physical[k].g[:,col]*=initial[wire]
        k,row=last[wire];physical[k].g[row,:]*=scale[wire]
    return physical


def execute(x,gates):
    values=np.array(x,dtype=float,copy=True)
    for gate in gates:
        a,b=values[gate.i],values[gate.j];g=gate.g
        values[gate.i]=g[0,0]*a+g[0,1]*b
        values[gate.j]=g[1,0]*a+g[1,1]*b
    return values


def circuit_matrix(n,gates):
    m=np.eye(n)
    for gate in gates:m[[gate.i,gate.j]]=gate.g@m[[gate.i,gate.j]]
    return m


def verify(result,routing):
    gates=compile_walk(result,routing);n=result['n']
    scales=np.full(n,math.sqrt(n/2));scales[0]=scales[-1]=math.sqrt(n)
    target=scales[:,None]*fourier(n);actual=circuit_matrix(n,gates)
    error=float(np.max(abs(actual-target)))
    if error>1e-9:raise AssertionError(('circuit reconstruction',n,error))
    return gates,error


def serialize_gates(gates):
    return [{'i':int(g.i),'j':int(g.j),'matrix':g.g.tolist()} for g in gates]


def ordered_control(n):
    """Unrestricted plane-rotation elimination to an actual diagonal.

    Quadratic-size constructive control, not advertised as a discovered FFT.
    It also supplies complete words for topology-changing turnover walks.
    """
    m=fourier(n);gates=[]
    for col in range(n-1):
        for row in range(n-1,col,-1):
            a,b=m[col,col],m[row,col]
            if abs(b)<TOL:continue
            h=math.hypot(a,b);g=np.array([[a/h,b/h],[-b/h,a/h]])
            m[[col,row]]=g@m[[col,row]];gates.append(Gate(col,row,g))
    return {'n':n,'gates':gates,'terminal':m,'permutation':tuple(range(n)),
            'complete':True,'trace':[],'attempted_cells':len(gates),'peak_condition':1.}


def ordered_sparse(n,seed=0):
    """Sparse walk plus explicitly PAID quarter-turn placement cells.

    This is a starting control for algebraic absorption, not a free permutation.
    All of these cells enter the rewrite search and the execution ledger.
    """
    result=synthesize(n,seed,False)
    if not result['complete']:return ordered_control(n)
    m=result['terminal'].copy();gates=list(result['gates'])
    permutation=list(result['permutation'])
    for i in range(n):
        if permutation[i]==i:continue
        j=permutation.index(i);g=np.array([[0.,1.],[-1.,0.]])
        m[[i,j]]=g@m[[i,j]];permutation[i],permutation[j]=permutation[j],permutation[i]
        gates.append(Gate(i,j,g))
    result.update(gates=gates,terminal=m,permutation=tuple(permutation))
    return result
