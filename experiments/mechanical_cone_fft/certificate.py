#!/usr/bin/env python3
"""NumPy-only certificates for the mechanical cone-FFT forms.

Checks the cone quotient algebra, an exact normalized N=8 Bruun factorization,
the backward mass gauge, the forward kinematic gauge, reciprocal virtual work,
the convex tap inventory, and a dense reciprocal spring existence result.
"""
from __future__ import annotations
import json, math
from typing import Iterable, Sequence
import numpy as np

TOL=1e-12
SQRT2=math.sqrt(2.0)
SILVER=SQRT2-1.0

def real_fourier_matrix(n:int)->np.ndarray:
    t=np.arange(n); rows=[np.ones(n)/np.sqrt(n)]
    for k in range(1,n//2):
        phase=2*np.pi*k*t/n
        rows += [np.sqrt(2/n)*np.cos(phase), -np.sqrt(2/n)*np.sin(phase)]
    rows.append(((-1.0)**t)/np.sqrt(n))
    return np.vstack(rows)

def cone_lift(m:np.ndarray)->np.ndarray:
    p=np.maximum(m,0.0); n=np.maximum(-m,0.0)
    return np.block([[p,n],[n,p]])

def projection(n:int)->np.ndarray: return np.hstack([np.eye(n),-np.eye(n)])
def positive_injection(n:int)->np.ndarray: return np.vstack([np.eye(n),np.zeros((n,n))])

def block_diag(*blocks:np.ndarray)->np.ndarray:
    out=np.zeros((sum(x.shape[0] for x in blocks),sum(x.shape[1] for x in blocks)))
    r=c=0
    for b in blocks:
        h,w=b.shape; out[r:r+h,c:c+w]=b; r+=h; c+=w
    return out

def bruun_cell(theta:float)->np.ndarray:
    c,s=math.cos(theta),math.sin(theta)
    return np.array([[1,0,c,-s],[0,1,s,c],[1,0,-c,s],[0,-1,s,c]],float)/SQRT2

def half_split(n:int)->np.ndarray:
    h=n//2; out=np.zeros((n,n))
    for j in range(h):
        out[j,j]=out[j,j+h]=out[h+j,j]=1/SQRT2; out[h+j,j+h]=-1/SQRT2
    return out

def bruun8():
    target=real_fourier_matrix(8); s0=half_split(8)
    odd=np.diag([1.,-1.,1.,-1.])@bruun_cell(np.pi/4)@np.eye(4)[[0,2,1,3],:]
    e0=half_split(4)
    e1=np.array([[1/SQRT2,1/SQRT2,0,0],[0,0,1,0],[0,0,0,-1],[1/SQRT2,-1/SQRT2,0,0]])
    s1=block_diag(e0,odd); s2=block_diag(e1,np.eye(4))
    perm=np.eye(8)[[0,4,5,1,2,6,7,3],:]
    return [s0,s1,s2],perm,target

def product(stages:Iterable[np.ndarray])->np.ndarray:
    stages=list(stages); out=np.eye(stages[0].shape[1])
    for stage in stages: out=stage@out
    return out

def mass_gauge(stages:Sequence[np.ndarray]):
    g=[np.empty(0)]*(len(stages)+1); g[-1]=np.ones(stages[-1].shape[0])
    for s in range(len(stages)-1,-1,-1): g[s]=np.abs(stages[s]).T@g[s+1]
    a=[g[s+1][:,None]*m/g[s][None,:] for s,m in enumerate(stages)]
    return g,a

def kinematic_gauge(stages:Sequence[np.ndarray]):
    d=[np.ones(2*stages[0].shape[1])]; b=[]
    for m in stages:
        c=cone_lift(m); nxt=1/(c@(1/d[-1])); physical=nxt[:,None]*c*(1/d[-1])[None,:]
        d.append(nxt); b.append(physical)
    return d,b

def choose_three(w:np.ndarray):
    best=None
    for pair in ((0,1),(0,2),(1,2)):
        third=({0,1,2}-set(pair)).pop(); i,j=pair; ps=w[i]+w[j]
        bp=float(w[j]/ps); bf=float(w[third]/w.sum()); score=min(bp,1-bp,bf,1-bf)
        cand=(score,bp,bf)
        if best is None or cand[0]>best[0]: best=cand
    return best[1],best[2]

def taps(w:np.ndarray):
    n=len(w)
    if n==1:return []
    if n==2:return [float(w[1]/w.sum())]
    if n==3:return list(choose_three(w))
    if n==4:
        best=None
        for p1,p2 in (((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))):
            s1=float(w[list(p1)].sum()); s2=float(w[list(p2)].sum())
            vals=[w[p1[1]]/s1,w[p2[1]]/s2,s2/(s1+s2)]
            score=min(*(vals),*(1-x for x in vals)); cand=(score,vals)
            if best is None or cand[0]>best[0]: best=cand
        return [float(x) for x in best[1]]
    raise NotImplementedError

def sinkhorn_abs(m:np.ndarray):
    a=np.abs(m); r=np.ones(a.shape[0]); c=np.ones(a.shape[1])
    for _ in range(10000):
        r/=r*(a@c); c/=c*(a.T@r); b=r[:,None]*a*c[None,:]
        if max(np.max(abs(b.sum(0)-1)),np.max(abs(b.sum(1)-1)))<1e-14:return r,c
    raise RuntimeError('Sinkhorn balancing failed')

def check(name:str,value:float,limit:float):
    if not np.isfinite(value) or value>limit: raise AssertionError(f'{name}: {value} > {limit}')

def main():
    rng=np.random.default_rng(20260812); out={}
    A=rng.normal(size=(3,4)); B=rng.normal(size=(4,2))
    d1=np.linalg.norm(projection(3)@cone_lift(A)-A@projection(4))
    d2=np.linalg.norm(projection(3)@cone_lift(A)@cone_lift(B)-A@B@projection(2))
    check('intertwining',d1,1e-13); check('projected composition',d2,1e-13)
    a=np.array([[1.,1.]]); b=np.array([[1.],[-1.]])
    gap=np.linalg.norm(cone_lift(a)@cone_lift(b)-cone_lift(a@b))
    if gap<=1: raise AssertionError('canonical lift counterexample vanished')
    out.update(cone_intertwining_defect=float(d1),cone_composed_projection_defect=float(d2),canonical_lift_multiplicativity_gap=float(gap))

    stages,perm,target=bruun8(); ferr=np.linalg.norm(perm@product(stages)-target); check('F8',ferr,2e-14)
    out['n8_factorization_error']=float(ferr)

    g,sg=mass_gauge(stages); cg=[cone_lift(x) for x in sg]
    cdef=max(float(np.max(abs(x.sum(0)-1))) for x in cg); check('mass columns',cdef,2e-14)
    boundary=projection(8)@product(cg)@positive_injection(8)@np.diag(g[0])
    merr=np.linalg.norm(perm@boundary-target); check('mass recovery',merr,2e-14)
    out.update(mass_gauge_column_defect=cdef,mass_gauge_boundary_error=float(merr),mass_gauge_input_min=float(g[0].min()),mass_gauge_input_max=float(g[0].max()))

    d,kg=kinematic_gauge(stages); rdef=max(float(np.max(abs(x.sum(1)-1))) for x in kg); check('kinematic rows',rdef,2e-14)
    boundary=np.diag(1/d[-1])@product(kg); signed=projection(8)@boundary@positive_injection(8)
    kerr=np.linalg.norm(perm@signed-target); check('kinematic recovery',kerr,2e-14)
    out.update(kinematic_gauge_row_defect=rdef,kinematic_gauge_boundary_error=float(kerr))

    vw=[]
    for stage in cg:
        f=rng.normal(size=stage.shape[1]); dqout=rng.normal(size=stage.shape[0]); dqin=stage.T@dqout
        vw.append(abs(f@dqin-(stage@f)@dqout))
    v=max(vw); check('virtual work',v,2e-13); out['reciprocal_virtual_work_defect']=float(v)

    all_taps=[]; passes=0; per=[]
    for stage in kg:
        before=len(all_taps)
        for row in stage:
            support=np.flatnonzero(row>TOL)
            if len(support)==1: passes+=1
            all_taps.extend(taps(row[support]))
        per.append(len(all_taps)-before)
    mid=sum(abs(x-.5)<1e-12 for x in all_taps); silver=sum(abs(x-SILVER)<1e-12 for x in all_taps)
    if (len(all_taps),mid,silver)!=(44,36,8): raise AssertionError('tap inventory changed')
    out.update(convex_binary_bars=len(all_taps),convex_bars_per_stage=per,convex_midpoint_taps=mid,convex_silver_taps=silver,convex_pass_through_rows=passes,minimum_tap_arm_fraction=float(min(min(x,1-x) for x in all_taps)))

    r,c=sinkhorn_abs(target); bs=r[:,None]*target*c[None,:]; bc=cone_lift(bs)
    check('dense rows',float(np.max(abs(bc.sum(1)-1))),2e-14); check('dense cols',float(np.max(abs(bc.sum(0)-1))),2e-14)
    K=np.block([[np.diag(bc.sum(0)),-bc.T],[-bc,np.diag(bc.sum(1))]])
    eig=np.linalg.eigvalsh(K)
    if eig[0]<-2e-13: raise AssertionError('spring Laplacian is indefinite')
    derr=np.linalg.norm(np.diag(1/r)@bs@np.diag(1/c)-target); check('dense reconstruction',derr,2e-14)
    out.update(dense_spring_edges=int(np.count_nonzero(bc>TOL)),dense_spring_min_eigenvalue=float(eig[0]),dense_spring_first_nonrigid_eigenvalue=float(eig[1]),dense_spring_reconstruction_error=float(derr))
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
