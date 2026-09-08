"""Exact-form SO(3) chart changes inside an ordered real Fourier walk.

Two or three overlapping plane cells can be expressed in any of six Euler
charts on the same three physical wires. This changes angles AND pair topology.
The search may take a two-to-three step before finding later cancellations.
"""
import math
import itertools
import numpy as np
from walk import Gate,ordered_control,ordered_sparse,orient,verify


def rotation(c,s):return np.array([[c,s],[-s,c]])


def embedded(gate,wires):
    r=np.eye(len(wires));ids=[wires.index(gate.i),wires.index(gate.j)]
    r[np.ix_(ids,ids)]=gate.g
    return r


def product(gates,wires):
    m=np.eye(len(wires))
    for gate in gates:m=embedded(gate,wires)@m
    return m


def euler(word,wires):
    """Return B(alpha), A(beta), B(gamma), in execution order.

    A acts on wires 0,1; B on 1,2. For M=B(gamma) A(beta) B(alpha),
    M00=cos(beta), (M01,M02)=sin(beta)*(cos(alpha),sin(alpha)),
    (-M10,M20)=sin(beta)*(cos(gamma),sin(gamma)).
    """
    m=product(word,wires);s=math.hypot(m[0,1],m[0,2]);c=m[0,0]
    h=math.hypot(c,s);c/=h;s/=h
    if s<1e-12:
        ca,sa=1.,0.;cg,sg=m[2,2],m[1,2]
        h=math.hypot(cg,sg);cg/=h;sg/=h
    else:
        ca,sa=m[0,1],m[0,2];h=math.hypot(ca,sa);ca/=h;sa/=h
        cg,sg=-m[1,0],m[2,0];h=math.hypot(cg,sg);cg/=h;sg/=h
    a,b,d=wires
    result=[Gate(b,d,rotation(ca,sa)),Gate(a,b,rotation(c,s)),Gate(b,d,rotation(cg,sg))]
    if np.max(abs(product(result,wires)-m))>2e-10:raise AssertionError('Euler chart mismatch')
    return result


def canon(gate):
    if gate.i<gate.j:return Gate(gate.i,gate.j,gate.g.copy())
    return Gate(gate.j,gate.i,gate.g[::-1,::-1].copy())


def simplify(gates):
    result=[canon(g) for g in gates if np.max(abs(g.g-np.eye(2)))>2e-12]
    changed=True
    while changed:
        changed=False
        for i,g in enumerate(result):
            for j in range(i+1,len(result)):
                h=result[j]
                if (h.i,h.j)==(g.i,g.j):
                    merged=Gate(g.i,g.j,h.g@g.g)
                    result[i]=merged;result.pop(j)
                    if np.max(abs(merged.g-np.eye(2)))<2e-12:result.pop(i)
                    changed=True;break
                if {h.i,h.j}&{g.i,g.j}:break
            if changed:break
    return result


def counts(gates):
    mul=add=swaps=0
    for gate in gates:
        a=gate.g;nonzero=abs(a)>2e-10
        add+=sum(max(0,int(k)-1) for k in np.sum(nonzero,axis=1))
        values=abs(a[nonzero]);m=int(np.count_nonzero(abs(values-1)>2e-10))
        if len(values)==4 and np.max(values)-np.min(values)<2e-10 and m==4:m=2
        mul+=m
        swaps+=bool(not nonzero[0,0] and not nonzero[1,1])
    return {'cells':len(gates),'estimated_real_multiplies':mul,'estimated_real_additions':add,
            'pair_reads':len(gates),'pair_writes':len(gates),'pure_exchange_cells':swaps}


def objective(gates):
    c=counts(gates)
    return c['cells']+.12*c['estimated_real_multiplies']+.04*c['estimated_real_additions']


def search(n,seed=0,steps=5000,start_kind="sparse",general_frames=False):
    rng=np.random.default_rng(seed);control=ordered_sparse(n,seed) if start_kind=="sparse" else ordered_control(n)
    initial_counts=counts(control['gates'])
    current=simplify(control['gates']);best=current;cost=objective(current);best_cost=cost
    accepted=turnovers=births=commutes=general_moves=0;history=[]
    for step in range(steps):
        if len(current)<2:break
        proposal=list(current)
        start=int(rng.integers(len(current)-1))
        width=3 if start+2<len(current) and rng.random()<.65 else 2
        selected=current[start:start+width]
        wires=sorted(set(w for g in selected for w in (g.i,g.j)))
        move=None
        if len(wires)==3:
            wires=list(rng.permutation(wires))
            orthogonal=all(np.max(abs(g.g@g.g.T-np.eye(2)))<1e-12 and np.linalg.det(g.g)>0 for g in selected)
            if general_frames and (not orthogonal or rng.random()<.4):
                from general_turnover import general
                replacement=general(selected,wires)
                if replacement is None:continue
                general_move=True
            else:
                replacement=euler(selected,wires);general_move=False
            proposal[start:start+width]=replacement
            move='birth' if width==2 else 'turnover'
        elif not ({selected[0].i,selected[0].j}&{selected[1].i,selected[1].j}):
            proposal[start],proposal[start+1]=proposal[start+1],proposal[start];move='commute'
        if move is None:continue
        proposal=simplify(proposal);new_cost=objective(proposal)
        temperature=.10+1.4*(1-(step%1000)/1000)
        if new_cost<=cost or rng.random()<math.exp(min(0,(cost-new_cost)/temperature)):
            current=proposal;cost=new_cost;accepted+=1
            turnovers+=move=='turnover';births+=move=='birth';commutes+=move=='commute'
            general_moves+=move!='commute' and general_move
        if new_cost<best_cost-1e-10:
            best=proposal;best_cost=new_cost
            history.append({'step':step,**counts(best)})
        if (step+1)%1000==0:current=list(best);cost=best_cost
    control['gates']=best
    route=orient(control['permutation'],[(g.i,g.j) for g in reversed(best)])
    physical,error=verify(control,route)
    return {'n':n,'reverse_gates':best,'physical_gates':physical,'route':route,
            'initial':initial_counts,'final':counts(physical),
            'reverse_final':counts(best),'max_matrix_error':error,'history':history,
            'accepted_moves':accepted,'turnovers':turnovers,'births':births,'commutes':commutes,
            'general_frame_moves':general_moves}
