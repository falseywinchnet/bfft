"""Exercise independent positive scale gauges at every intermediate wire value.

Endpoint gauges are fixed to one. Edge equations are admitted by a weighted
union-find with cycle checks; this is a finite geometry-only construction.
An inconsistent cycle is retained as non-unit arithmetic, never erased.
"""
import math
import numpy as np
from walk import Gate,circuit_matrix
from turnover import counts


class Potentials:
    def __init__(self,n):
        self.parent=list(range(n));self.weight=[0.]*n;self.rank=[0]*n
    def find(self,x):
        if self.parent[x]!=x:
            p=self.parent[x];r,w=self.find(p);self.weight[x]+=w;self.parent[x]=r
        return self.parent[x],self.weight[x]
    def admit(self,a,b,d):
        """lambda[a]-lambda[b]=d; false means a conflicting existing cycle."""
        ra,wa=self.find(a);rb,wb=self.find(b)
        if ra==rb:return abs(wa-wb-d)<1e-9
        offset=d-wa+wb
        if self.rank[ra]<self.rank[rb]:self.parent[ra]=rb;self.weight[ra]=offset
        else:
            self.parent[rb]=ra;self.weight[rb]=-offset
            if self.rank[ra]==self.rank[rb]:self.rank[ra]+=1
        return True


def frame_graph(n,gates):
    current=list(range(n));edges=[];cells=[];vertices=n
    for gate in gates:
        ins=[current[gate.i],current[gate.j]];outs=[vertices,vertices+1];vertices+=2
        for row in range(2):
            for col in range(2):
                a=gate.g[row,col]
                if abs(a)>1e-12:edges.append((outs[row],ins[col],-math.log(abs(a))))
        cells.append((ins,outs));current[gate.i],current[gate.j]=outs
    return vertices,list(range(n))+current,edges,cells


def gauge(gates,n,seed=0):
    vertices,boundary,edges,cells=frame_graph(n,gates);dsu=Potentials(vertices+1);anchor=vertices
    for v in boundary:assert dsu.admit(v,anchor,0.)
    order=list(range(len(edges)));rng=np.random.default_rng(seed)
    if seed==0:order.sort(key=lambda k:abs(edges[k][2]))
    else:rng.shuffle(order)
    accepted=0
    for k in order:accepted+=dsu.admit(*edges[k])
    # The anchored component's root is not necessarily anchor itself.
    root,aw=dsu.find(anchor);lambdas=[]
    for v in range(vertices):
        r,w=dsu.find(v);lambdas.append(w-aw if r==root else w)
    peak=max(abs(x) for x in lambdas)
    if peak>math.log(1e8):return None
    result=[]
    for gate,(ins,outs) in zip(gates,cells):
        a=gate.g.copy()
        for row in range(2):
            for col in range(2):a[row,col]*=math.exp(lambdas[outs[row]]-lambdas[ins[col]])
        result.append(Gate(gate.i,gate.j,a))
    error=float(np.max(abs(circuit_matrix(n,result)-circuit_matrix(n,gates))))
    if error>2e-9:raise AssertionError(('gauge changed endpoint',error))
    return {'gates':result,'counts':counts(result),'admitted_unit_equations':accepted,
            'conflicting_cycle_edges':len(edges)-accepted,'log_scale_max':peak,
            'scale_dynamic_range':math.exp(max(lambdas)-min(lambdas)),
            'max_matrix_change':error,'free_intermediate_gauges':vertices-len(set(boundary)),
            'nonzero_log_gauges':sum(abs(x)>1e-10 for x in lambdas),
            'lambdas':lambdas,'seed':seed}


def best_gauge(gates,n,seeds=32):
    original=counts(gates);best=None
    for seed in range(seeds):
        candidate=gauge(gates,n,seed)
        if candidate is None:continue
        key=(candidate['counts']['estimated_real_multiplies'],candidate['log_scale_max'])
        if best is None or key<(best['counts']['estimated_real_multiplies'],best['log_scale_max']):best=candidate
    return original,best
