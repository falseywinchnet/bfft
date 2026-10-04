"""Joint image meshes over a shared 2-D scene chart, solved in two stages.

This is a layered 2-D registration representation, not recovered metric depth.
"""
import heapq
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from skimage.transform import SimilarityTransform


def lattice(shape):
    h,w=shape;nx=max(5,int(np.ceil(w/32))+1);ny=max(5,int(np.ceil(h/32))+1)
    yy,xx=np.meshgrid(np.linspace(0,h-1,ny),np.linspace(0,w-1,nx),indexing='ij')
    return np.stack([xx,yy],-1),[ny,nx]


def weights(points,shape,meshshape):
    h,w=shape;ny,nx=meshshape;q=np.asarray(points)/[w-1,h-1]*[nx-1,ny-1]
    q=np.clip(q,[0,0],[nx-1-1e-8,ny-1-1e-8]);ix=np.floor(q).astype(int);t=q-ix
    ids=ix[:,1]*nx+ix[:,0];ids=ids[:,None]+[0,1,nx,nx+1]
    tx,ty=t[:,0],t[:,1]
    coeff=np.where((tx+ty<=1)[:,None],np.c_[1-tx-ty,tx,ty,tx*0],np.c_[tx*0,1-ty,1-tx,tx+ty-1])
    return ids,coeff


def evaluate(mesh,points,shape):
    ids,c=weights(points,shape,mesh.shape[:2]);return (mesh.reshape(-1,2)[ids]*c[...,None]).sum(1)


def initialize(shapes,edges,roots):
    """Minimum-cost paths initialize gauges; all graph edges enter refinement."""
    maps={i:np.eye(3) for i in roots};cost={i:0. for i in roots};queue=[(0.,i) for i in roots]
    adj={i:[] for i in range(len(shapes))}
    for e in edges:
        a,b=e['a'],e['b'];model=SimilarityTransform()
        if not model.estimate(np.array(e['a_points']),np.array(e['b_points'])):continue
        m=model.params
        if not .15<np.linalg.det(m[:2,:2])<7:continue
        c=1+8/max(e['inliers'],1)+e['error']/4
        adj[a].append((b,m,c));adj[b].append((a,np.linalg.inv(m),c))
    while queue:
        d,i=heapq.heappop(queue)
        if d!=cost[i]:continue
        for j,m,c in adj[i]:
            if d+c<cost.get(j,float('inf')):
                cost[j]=d+c;maps[j]=maps[i]@np.linalg.inv(m);heapq.heappush(queue,(d+c,j))
    meshes={}
    for i,m in maps.items():
        p,_=lattice(shapes[i]);meshes[i]=(np.c_[p.reshape(-1,2),np.ones(p.shape[0]*p.shape[1])]@m.T)[:,:2].reshape(p.shape)
    return meshes,cost


def minimum_area_ratio(base,candidate):
    def areas(p):
        u=p[:-1,1:]-p[:-1,:-1];v=p[1:,:-1]-p[:-1,:-1]
        u2=p[1:,:-1]-p[1:,1:];v2=p[:-1,1:]-p[1:,1:]
        return np.r_[(u[...,0]*v[...,1]-u[...,1]*v[...,0]).ravel(),(u2[...,0]*v2[...,1]-u2[...,1]*v2[...,0]).ravel()]
    return float(np.min(areas(candidate)/areas(base)))


def solve(shapes,edges,base,fixed,iterations=6,period=None):
    ids=sorted(base);offset={};total=0
    for i in ids:offset[i]=total;total+=base[i].shape[0]*base[i].shape[1]
    original=np.concatenate([base[i].reshape(-1,2) for i in ids]);current=original.copy();history=[]
    edges=[e for e in edges if e['a'] in base and e['b'] in base]
    rows=[];cols=[];vals=[];targets=[];tie_ranges=[]
    def add(ii,cc,target):
        row=len(targets);rows.extend([row]*len(ii));cols.extend(ii);vals.extend(cc);targets.append(target)
    for e in edges:
        a,b=e['a'],e['b'];pa=np.array(e['a_points']);pb=np.array(e['b_points'])
        ia,ca=weights(pa,shapes[a],base[a].shape[:2]);ib,cb=weights(pb,shapes[b],base[b].shape[:2]);start=len(targets)
        wt=min(1.,np.sqrt(25/len(pa)))
        for k in range(len(pa)):
            target=np.zeros(2)
            if period is not None:
                difference=ca[k]@base[a].reshape(-1,2)[ia[k]]-cb[k]@base[b].reshape(-1,2)[ib[k]]
                target=np.rint(difference[0]/period[0])*period
            add(np.r_[ia[k]+offset[a],ib[k]+offset[b]],np.r_[ca[k],-cb[k]]*wt,target*wt)
        tie_ranges.append((start,len(targets),wt,e))
    ties=len(targets)
    # Preserve local affine shape softly; allow supported non-affine parallax.
    for i in ids:
        ny,nx=base[i].shape[:2];off=offset[i]
        for y in range(ny):
            for x in range(nx):
                v=off+y*nx+x
                if i in fixed:add([v],[100.],original[v]*100)
                else:
                    add([v],[.025],original[v]*.025)
                    for dx,dy in ((1,0),(0,1)):
                        if (0<x<nx-1 and dx) or (0<y<ny-1 and dy):
                            ns=[v-dy*nx-dx,v,v+dy*nx+dx];cc=np.array([1,-2,1])*.7
                            add(ns,cc,cc@original[ns])
                    if x+1<nx:
                        add([v,v+1],[.06,-.06],(original[v]-original[v+1])*.06)
                    if y+1<ny:
                        add([v,v+nx],[.06,-.06],(original[v]-original[v+nx])*.06)
    A=sparse.coo_matrix((vals,(rows,cols)),shape=(len(targets),total)).tocsr();target=np.array(targets);rw=np.ones(len(targets))
    for iteration in range(iterations):
        if ties:
            err=np.linalg.norm(A[:ties]@current-target[:ties],axis=1)
            rw[:ties]=1/np.sqrt(1+(err/5)**2)
        Aw=A.multiply(rw[:,None]).tocsr();rhs=target*rw[:,None]
        solved=[lsqr(Aw,rhs[:,k],atol=1e-6,btol=1e-6,iter_lim=1500,x0=current[:,k]) for k in range(2)]
        proposal=np.stack([r[0] for r in solved],-1)
        for i in ids:
            start=offset[i];stop=start+base[i].shape[0]*base[i].shape[1]
            if i in fixed:proposal[start:stop]=original[start:stop]
            else:
                safe,_=admissible_step(base[i],current[start:stop].reshape(base[i].shape),proposal[start:stop].reshape(base[i].shape))
                proposal[start:stop]=safe.reshape(-1,2)
        current=proposal
        history.append(dict(iteration=iteration,median_tie_residual=float(np.median(np.linalg.norm(A[:ties]@current-target[:ties],axis=1))),linear_iterations=[r[2] for r in solved],stop_codes=[r[1] for r in solved]))
    candidate={i:current[offset[i]:offset[i]+base[i].shape[0]*base[i].shape[1]].reshape(base[i].shape) for i in ids}
    # Fixed charts are exact; there is no tolerance-driven movement of stage 1.
    for i in fixed:candidate[i]=base[i].copy()
    fractions={}
    for i in ids:
        fraction=1.
        while minimum_area_ratio(base[i],base[i]+fraction*(candidate[i]-base[i]))<.1 and fraction>1/128:fraction*=.5
        if minimum_area_ratio(base[i],base[i]+fraction*(candidate[i]-base[i]))<.1:fraction=0.
        candidate[i]=base[i]+fraction*(candidate[i]-base[i]);fractions[i]=fraction
    residuals=[]
    for _,_,_,e in tie_ranges:
        a,b=e['a'],e['b'];delta=evaluate(candidate[a],e['a_points'],shapes[a])-evaluate(candidate[b],e['b_points'],shapes[b])
        if period is not None:delta-=np.rint(delta[:,0]/period[0])[:,None]*period
        err=np.linalg.norm(delta,axis=1)
        residuals.append(dict(a=a,b=b,median=float(np.median(err)),p90=float(np.quantile(err,.9)),count=len(err)))
    return candidate,dict(history=history,fold_prevention_fractions=fractions,residuals=residuals,
        minimum_area_ratio=min(minimum_area_ratio(base[i],candidate[i]) for i in ids),fixed=sorted(fixed))


def panorama_initialization(shapes,edges,anchor):
    """Robust global translation gauge before local distortion estimation.

Normalize vertical angular sampling provisionally. This initializes meshes;
neither equal scale nor translation-only behavior constrains the final fit.
"""
    ids=sorted({e[k] for e in edges for k in ('a','b')});index={i:k for k,i in enumerate(ids)};height=shapes[anchor][0]
    rows=[];targets=[];confidence=[]
    for e in edges:
        a,b=e['a'],e['b'];delta=np.array(e['a_points'])*height/shapes[a][0]-np.array(e['b_points'])*height/shapes[b][0]
        median=np.median(delta,axis=0);scatter=np.median(np.linalg.norm(delta-median,axis=1))
        row=np.zeros(len(ids));row[index[b]]=1;row[index[a]]=-1;rows.append(row);targets.append(median)
        confidence.append(np.sqrt(e['inliers'])/(1+scatter/8))
    A=np.array(rows);b=np.array(targets);confidence=np.array(confidence);w=confidence.copy();positions=np.zeros((len(ids),2))
    anchorrow=np.zeros((1,len(ids)));anchorrow[0,index[anchor]]=100
    for _ in range(12):
        positions=np.linalg.lstsq(np.r_[A*w[:,None],anchorrow],np.r_[b*w[:,None],[[0.,0.]]],rcond=None)[0]
        err=np.linalg.norm(A@positions-b,axis=1);w=confidence/np.sqrt(1+(err/12)**2)
    return {i:lattice(shapes[i])[0]*height/shapes[i][0]+positions[index[i]] for i in ids}


def admissible_step(reference,current,proposal):
    """Damp only vertices incident to invalid triangles; preserve safe updates."""
    def area(p):
        a=p[:-1,:-1];b=p[:-1,1:];c=p[1:,:-1];d=p[1:,1:]
        cross=lambda u,v:u[...,0]*v[...,1]-u[...,1]*v[...,0]
        return np.stack([cross(b-a,c-a),cross(c-d,b-d)],-1)
    denominator=area(reference);fraction=np.ones(current.shape[:2]);delta=proposal-current
    for iteration in range(28):
        candidate=current+fraction[...,None]*delta
        bad=np.any(area(candidate)/denominator<.1,axis=-1)
        if not bad.any():return candidate,fraction
        incident=np.zeros_like(fraction,dtype=bool)
        incident[:-1,:-1]|=bad;incident[:-1,1:]|=bad;incident[1:,:-1]|=bad;incident[1:,1:]|=bad
        fraction[incident]*=.5
        if iteration>20:fraction[incident]=0
    return current.copy(),np.zeros_like(fraction)


def periodic_panorama_initialization(shapes,edges,anchor):
    """Estimate a repeated chart period from redundant translation closures."""
    initial=panorama_initialization(shapes,edges,anchor);ids=sorted(initial);ix={i:k for k,i in enumerate(ids)};height=shapes[anchor][0]
    A=[];b=[];confidence=[]
    for e in edges:
        a,c=e['a'],e['b'];d=np.array(e['a_points'])*height/shapes[a][0]-np.array(e['b_points'])*height/shapes[c][0]
        v=np.median(d,0);scatter=np.median(np.linalg.norm(d-v,axis=1));row=np.zeros(len(ids));row[ix[c]]=1;row[ix[a]]=-1;A.append(row);b.append(v);confidence.append(np.sqrt(e['inliers'])/(1+scatter/8))
    A=np.array(A);b=np.array(b);confidence=np.array(confidence);positions=np.array([initial[i][0,0] for i in ids]);closure=A@positions-b
    wide=np.abs(closure[:,0]);candidates=wide[wide>shapes[anchor][1]*.7]
    lookup={(e['a'],e['b']):b[k] for k,e in enumerate(edges)}
    triangle_periods=[]
    for a in ids:
        for c in ids:
            if c<=a or (a,c) not in lookup:continue
            for d in ids:
                if d<=c or (c,d) not in lookup or (a,d) not in lookup:continue
                loop=lookup[a,c]+lookup[c,d]-lookup[a,d]
                if abs(loop[0])>.7*shapes[anchor][1] and abs(loop[1])<.2*abs(loop[0]):triangle_periods.append(abs(loop[0]))
    if len(triangle_periods)>=3:candidates=np.array(triangle_periods)
    if len(candidates)<3:return initial,None,dict(accepted=False,reason='no_repeated_closure')
    period=np.array([np.median(candidates),0.]);first_period=period.copy();history=[]
    for iteration in range(16):
        winding=np.rint((A@positions-b)[:,0]/period[0]);matrix=np.c_[A,-winding]
        residual=A@positions-b-winding[:,None]*period;w=confidence/np.sqrt(1+(np.linalg.norm(residual,axis=1)/16)**2)
        anchorrow=np.zeros((1,len(ids)+1));anchorrow[0,ix[anchor]]=100
        prior=np.zeros((1,len(ids)+1));prior[0,-1]=.05
        solution=np.linalg.lstsq(np.r_[matrix*w[:,None],anchorrow,prior],np.r_[b*w[:,None],[[0.,0.]],first_period[None,:]*.05],rcond=None)[0]
        positions=solution[:-1];period=solution[-1]
        history.append(dict(period=period.tolist(),median_wrapped_residual=float(np.median(np.linalg.norm(matrix@solution-b,axis=1)))))
    if not .7*shapes[anchor][1]<period[0]<3*shapes[anchor][1] or abs(period[1])>.2*period[0]:return initial,None,dict(accepted=False,reason='inadmissible_period',history=history)
    base={i:lattice(shapes[i])[0]*height/shapes[i][0]+positions[ix[i]] for i in ids}
    return base,period,dict(accepted=True,period=period.tolist(),history=history,closure_candidates=candidates.tolist(),basis='periodic horizontal scene chart inferred from redundant correspondence closures')


def unwrap_points(points,period):
    p=np.asarray(points).copy()
    if period is None:return p
    angle=p[:,0]*2*np.pi/period[0];center=np.arctan2(np.sin(angle).mean(),np.cos(angle).mean())*period[0]/(2*np.pi)
    p-=np.rint((p[:,0]-center)/period[0])[:,None]*period
    return p
