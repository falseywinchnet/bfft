"""Joint topology + nonorthogonal wire-frame search on an ordered real DFT.

Every accepted state is an executable whole transform. Topology changes are
followed by endpoint-fixed scale-gauge admission; no final placement is added.
"""
import math
import numpy as np
from walk import Gate,inverse2,ordered_sparse,orient,verify,circuit_matrix
from turnover import simplify,euler,counts,objective
from general_turnover import general
from gauge import gauge,best_gauge


def search(n,seed=0,steps=3000,excursions=False):
    rng=np.random.default_rng(seed);control=ordered_sparse(n,seed)
    route=orient(control['permutation'],[(g.i,g.j) for g in reversed(control['gates'])])
    original,_=verify(control,route);target=circuit_matrix(n,original)
    initial_counts=counts(original);_,initial=best_gauge(original,n,16)
    current=initial['gates'];best=current;cost=objective(current);best_cost=cost
    accepted=oblique=uphill=0;history=[];rejected_numeric=0;excursion_moves=0;excursion_examples=[]
    for step in range(steps):
        if len(current)<2:break
        start=int(rng.integers(len(current)-1));width=3 if start+2<len(current) and rng.random()<.6 else 2
        selected=current[start:start+width];wires=sorted(set(w for g in selected for w in (g.i,g.j)))
        proposal=list(current);kind=None
        excursion=None
        if excursions and rng.random()<.15:
            # Add an arbitrary continuous mixer and its inverse, then rechart
            # across the old cell so the identity excursion is not adjacent.
            # The third wire may be anywhere in the N-real-coordinate state.
            old=current[start];third=int(rng.choice([i for i in range(n) if i not in (old.i,old.j)]))
            common=old.i if rng.random()<.5 else old.j
            parameter=float(rng.uniform(-2.5,2.5))
            if rng.random()<.5:
                h=np.array([[math.cos(parameter),math.sin(parameter)],[-math.sin(parameter),math.cos(parameter)]])
                family='rotation'
            else:h=np.array([[1.,parameter],[0.,1.]]);family='shear'
            bridge=Gate(common,third,h)
            triple=list(rng.permutation([old.i,old.j,third]))
            replacement=general([old,bridge],triple)
            if replacement is None:continue
            proposal[start:start+1]=replacement+[Gate(common,third,inverse2(h))]
            kind='excursion';excursion={'step':step,'pair':[int(common),int(third)],
                                      'parameter':parameter,'family':family}
        elif len(wires)==3:
            wires=list(rng.permutation(wires));replacement=general(selected,wires)
            if replacement is None:continue
            proposal[start:start+width]=replacement;kind='general'
        elif not ({selected[0].i,selected[0].j}&{selected[1].i,selected[1].j}):
            proposal[start],proposal[start+1]=proposal[start+1],proposal[start];kind='commute'
        else:continue
        proposal=simplify(proposal)
        try:
            frame=gauge(proposal,n,int(rng.integers(100000)))
        except AssertionError:
            rejected_numeric+=1;continue
        if frame is None:rejected_numeric+=1;continue
        proposal=frame['gates'];new_cost=objective(proposal)
        error=float(np.max(abs(circuit_matrix(n,proposal)-target)))
        if error>1e-9:rejected_numeric+=1;continue
        temperature=.05+.9*(1-(step%500)/500)
        if new_cost<=cost or rng.random()<math.exp(min(0,(cost-new_cost)/temperature)):
            uphill+=new_cost>cost+1e-10;accepted+=1;oblique+=kind=='general'
            current=proposal;cost=new_cost
            if kind=='excursion':
                excursion_moves+=1
                if len(excursion_examples)<8:excursion_examples.append(dict(excursion,cells=len(current)))
        if new_cost<best_cost-1e-10:
            best=proposal;best_cost=new_cost;history.append({'step':step,**counts(best)})
        if (step+1)%500==0:current=list(best);cost=best_cost
    return {'n':n,'initial':initial_counts,'gauged_start':counts(initial['gates']),
            'final':counts(best),'physical_gates':best,'history':history,
            'max_matrix_error':float(np.max(abs(circuit_matrix(n,best)-target))),
            'accepted_moves':accepted,'general_frame_moves':oblique,'uphill_moves':uphill,
            'numeric_rejections':rejected_numeric,'continuous_excursions_accepted':excursion_moves,
            'excursion_examples':excursion_examples}
