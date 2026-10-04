"""Sequential graph-cut seams; preserve source detail instead of ghost averaging."""
import numpy as np
from scipy import ndimage,sparse
from scipy.sparse.csgraph import maximum_flow,breadth_first_order


def cut(a,b,wa,wb):
    h,w=wa.shape;n=h*w;ids=np.arange(n).reshape(h,w)
    va=wa>0;vb=wb>0;union=va|vb
    error=np.mean(np.abs(a-b),axis=-1)+.008
    row=[];col=[];data=[]
    def add(u,v,c):row.extend(np.ravel(u));col.extend(np.ravel(v));data.extend(np.ravel(c))
    for u,v,e,valid in [(ids[:,:-1],ids[:,1:],(error[:,:-1]+error[:,1:])*.5,union[:,:-1]&union[:,1:]),(ids[:-1],ids[1:],(error[:-1]+error[1:])*.5,union[:-1]&union[1:])]:
        capacity=np.where(valid,np.maximum(1,np.rint(e*10000)),0).astype(np.int64)
        add(u,v,capacity);add(v,u,capacity)
    cost_a=np.rint((1-wa)*10).astype(np.int64);cost_b=np.rint((1-wb)*10).astype(np.int64)
    cost_a[~va&vb]=10000000;cost_b[va&~vb]=10000000
    cost_a[~union]=0;cost_b[~union]=10000000
    add(np.full(n,n),ids.ravel(),cost_b.ravel());add(ids.ravel(),np.full(n,n+1),cost_a.ravel())
    capacity=sparse.coo_matrix((data,(row,col)),shape=(n+2,n+2),dtype=np.int64).tocsr()
    flow=maximum_flow(capacity,n,n+1)
    residual=capacity-flow.flow;residual.data=(residual.data>0).astype(np.int64);residual.eliminate_zeros()
    reachable=breadth_first_order(residual,n,directed=True,return_predecessors=False)
    choose_b=np.ones(n+2,bool);choose_b[reachable]=False
    return choose_b[:n].reshape(h,w)&vb


def seam_mosaic(stack,weights,step=4):
    reduced=stack[:,::step,::step];rw=weights[:,::step,::step]
    composite=reduced[0].copy();w=rw[0].copy();labels=np.zeros(w.shape,np.int16)
    for i in range(1,len(stack)):
        choose=cut(composite,reduced[i],w,rw[i]);labels[choose]=i;composite[choose]=reduced[i][choose];w[choose]=rw[i][choose]
    full=np.repeat(np.repeat(labels,step,axis=0),step,axis=1)[:stack.shape[1],:stack.shape[2]]
    valid=weights>0;selected=np.take_along_axis(valid,full[None,...],axis=0)[0]
    fallback=weights.argmax(0);full=np.where(selected,full,fallback)
    # One output-pixel feather only; no broad averaging across depth conflicts.
    blend=np.stack([ndimage.gaussian_filter((full==i).astype(np.float32),.65)*valid[i] for i in range(len(stack))])
    blend/=np.maximum(blend.sum(0),1e-20)
    return np.sum(stack*blend[...,None],axis=0),full
