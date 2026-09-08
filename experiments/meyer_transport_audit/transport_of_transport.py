"""Carry complete transport directions and their adjoint global responses.

The reduced nonlinear map exactly equals the orthogonal projection of the
full map on the retained affine subspace. Every disk is re-evaluated at every
modeled state. Only restriction to that subspace is approximate. No factors.
"""
from pathlib import Path
import sys,json,argparse,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import (
    ReducedMeyerMap,State,grad,div,project_disk,project_disk_derivative,solve_screened,rms)
from experiments.meyer_transport_audit.certificate import certificate
from experiments.benchmark_meyer_flow_jump import benchmark_scene


def pullback(m,q):
    """<q,T(z)> = <linear,z> + constant + <weights,Pi(t)>.

    Two screened adjoint solves per direction. Both solves are self-adjoint
    with the same periodic operator as the authoritative map.
    """
    rw=solve_screened(q.w-div(q.twx,q.twy),m.cw,m.etaw,m.symbol)
    ru=solve_screened(q.u-div(q.tux,q.tuy)-m.cw*rw,m.cu,m.etau,m.symbol)
    gu=grad(ru);gw=grad(rw)
    linear=State(m.cu*ru,m.cu*ru,m.etau*gu[0],m.etau*gu[1],m.etaw*gw[0],m.etaw*gw[1])
    weights=np.stack([q.tux-2*m.etau*gu[0],q.tuy-2*m.etau*gu[1],
                      q.twx-2*m.etaw*gw[0],q.twy-2*m.etaw*gw[1]])
    constant=m.cw*float(np.sum(rw*m.image))
    return m.pack(linear),weights,constant


class CarriedTransport:
    def __init__(self,m,z,depth):
        self.m=m;self.z=z;self.zvec=m.pack(z)
        current=z;increments=[]
        for _ in range(depth):
            nxt=m.step(current);increments.append(m.pack(m.subtract(nxt,current)));current=nxt
        self.training_end=current;self.training_steps=depth
        matrix=np.stack(increments,axis=1)
        U,s,_=np.linalg.svd(matrix,full_matrices=False)
        # Numerical rank only; not a content-dependent step factor.
        keep=s>max(float(s[0])*1e-12,1e-25)
        self.Q=U[:,keep];self.rank=int(np.sum(keep));self.singular_values=s
        self.ytrain=self.Q.T@(m.pack(current)-self.zvec)
        L=[];V=[];constants=[]
        for j in range(self.rank):
            l,v,c=pullback(m,m.unpack(self.Q[:,j]));L.append(l);V.append(v);constants.append(c)
        self.L=np.asarray(L).reshape(self.rank,6*m.count)
        self.V=np.asarray(V).reshape(self.rank,4*m.count)
        self.linear=self.L@self.Q
        self.t0=np.stack([z.tux,z.tuy,z.twx,z.twy])
        self.Qt=self.Q.reshape(6,m.count,self.rank)[2:].reshape(4*m.count,self.rank)
        self.p0=self.projections(self.t0).ravel()
        # This offset follows from the exact pullback, with no extra map call.
        self.offset=self.L@self.zvec+np.asarray(constants)+self.V@self.p0-self.Q.T@self.zvec

    def projections(self,t):
        pu=project_disk(t[0],t[1],self.m.ru)
        pw=project_disk(t[2],t[3],self.m.rw)
        return np.stack([*pu,*pw])

    def field(self,y):
        return self.t0+(self.Qt@y).reshape(4,*self.m.shape)

    def reduced_map(self,y):
        return self.offset+self.linear@y+self.V@(self.projections(self.field(y)).ravel()-self.p0)

    def jacobian(self,y):
        t=self.field(y);extra=np.empty((4*self.m.count,self.rank))
        for j in range(self.rank):
            h=self.Qt[:,j].reshape(4,*self.m.shape)
            pu=project_disk_derivative(t[0],t[1],h[0],h[1],self.m.ru)
            pw=project_disk_derivative(t[2],t[3],h[2],h[3],self.m.rw)
            extra[:,j]=np.stack([*pu,*pw]).ravel()
        return self.linear+self.V@extra

    def reconstruct(self,y):return self.m.unpack(self.zvec+self.Q@y)

    def rollout(self,horizon,nonlinear=True):
        y=self.ytrain.copy()
        if not nonlinear:
            base=y.copy();fbase=self.reduced_map(base);J=self.jacobian(base)
        for _ in range(max(0,horizon-self.training_steps)):
            y=self.reduced_map(y) if nonlinear else fbase+J@(y-base)
        return self.reconstruct(y)


class FullMemoryTransport:
    """Retain the entire Bregman memory; jointly reduce only the primal pair.

    The primal update is its exact orthogonal projection. The full pointwise
    memory update is unprojected, and t=Dx+b with feasible b holds every step.
    """
    def __init__(self,m,z,depth):
        self.m=m;self.z=z;self.training_steps=depth
        self.x0=np.concatenate([z.u.ravel(),z.w.ravel()])
        current=z;increments=[]
        for _ in range(depth):
            nxt=m.step(current)
            increments.append(np.concatenate([(nxt.u-current.u).ravel(),(nxt.w-current.w).ravel()]))
            current=nxt
        self.training_end=current
        U,s,_=np.linalg.svd(np.stack(increments,axis=1),full_matrices=False)
        keep=s>max(float(s[0])*1e-12,1e-25);self.Q=U[:,keep];self.rank=int(np.sum(keep))
        self.ytrain=self.Q.T@(np.concatenate([current.u.ravel(),current.w.ravel()])-self.x0)
        def gradients(u,w):return np.stack([*grad(u),*grad(w)])
        self.t0=gradients(z.u,z.w)
        self.btrain=np.stack([current.tux,current.tuy,current.twx,current.twy])-gradients(current.u,current.w)
        Qgrad=[];L=[];V=[];constants=[]
        zero=np.zeros(m.shape)
        for j in range(self.rank):
            qu,qw=self.Q[:,j].reshape(2,*m.shape)
            Qgrad.append(gradients(qu,qw).ravel())
            l,v,c=pullback(m,State(qu,qw,zero,zero,zero,zero))
            L.append(l);V.append(v.ravel());constants.append(c)
        self.Qgrad=np.asarray(Qgrad).reshape(self.rank,4*m.count).T
        L=np.asarray(L).reshape(self.rank,6*m.count)
        self.Lb=L[:,2*m.count:];self.V=np.asarray(V).reshape(self.rank,4*m.count)
        self.linear=L[:,:2*m.count]@self.Q+self.Lb@self.Qgrad
        self.offset=L[:,:2*m.count]@self.x0+self.Lb@self.t0.ravel()+np.asarray(constants)-self.Q.T@self.x0

    def advance(self,y,b):
        t=self.t0+(self.Qgrad@y).reshape(4,*self.m.shape)+b
        p=np.stack([*project_disk(t[0],t[1],self.m.ru),*project_disk(t[2],t[3],self.m.rw)])
        yn=self.offset+self.linear@y+self.Lb@b.ravel()+self.V@p.ravel()
        return yn,p

    def reconstruct(self,y,b):
        u,w=(self.x0+self.Q@y).reshape(2,*self.m.shape)
        tu=np.stack(grad(u))+b[:2];tw=np.stack(grad(w))+b[2:]
        return State(u,w,*tu,*tw)

    def rollout(self,horizon,nonlinear=True):
        y=self.ytrain.copy();b=self.btrain.copy()
        for _ in range(max(0,horizon-self.training_steps)):y,b=self.advance(y,b)
        return self.reconstruct(y,b)


def compare(m,z,depth,horizon,repeats=3):
    times={'ordinary':[],'frozen':[],'carried':[],'full_memory':[]};ends={};models=[]
    for repeat in range(repeats):
        start=time.perf_counter();ordinary=z
        for _ in range(horizon+1):ordinary=m.step(ordinary)
        times['ordinary'].append(time.perf_counter()-start);ends['ordinary']=ordinary
        start=time.perf_counter();model=CarriedTransport(m,z,depth);setup=time.perf_counter()-start
        models.append(model)
        for name,nonlinear in [('frozen',False),('carried',True)]:
            start=time.perf_counter();candidate=model.rollout(horizon,nonlinear)
            candidate=m.step(candidate) # restores t=Dx+b, |b|<=a after the projected evolution
            times[name].append(setup+time.perf_counter()-start);ends[name]=candidate
        start=time.perf_counter();full=FullMemoryTransport(m,z,depth)
        candidate=m.step(full.rollout(horizon))
        times['full_memory'].append(time.perf_counter()-start);ends['full_memory']=candidate
    ref=ends['ordinary'];vref=m.image-ref.u-ref.w;rows={}
    for name,end in ends.items():
        rows[name]={'ms':1000*float(np.median(times[name])),**certificate(m,end),
                    'full_state_rms_to_ordinary':rms(m.pack(m.subtract(end,ref))),
                    'texture_rms_to_ordinary':rms((m.image-end.u-end.w)-vref)}
    return {'depth':depth,'rank':models[-1].rank,'horizon':horizon,
            'setup_map_equivalents':depth+models[-1].rank,
            'method_map_equivalents_including_refresh':depth+models[-1].rank+1,
            'ordinary_map_calls':horizon+1,'methods':rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,default=64)
    ap.add_argument('--out',required=True);ap.add_argument('--repeats',type=int,default=3)
    a=ap.parse_args();n=a.size;y,x=np.mgrid[:n,:n];rng=np.random.default_rng(983)
    scenes={'ramp':255*x/n,'edge':50.+150.*(x>n//2),
            'carrier':100+2*np.sin(2*np.pi*(x+2*y)/n),
            'crossing':benchmark_scene(n),'noise':rng.uniform(0,255,(n,n))}
    result={'size':n,'note':'Identical NumPy backend. All training, SVD, adjoint setup, inner projections, reconstruction and one refresh are timed. Diagnostics excluded. No native speed claim.', 'scenes':{}}
    for name,f in scenes.items():
        m=ReducedMeyerMap(f,.05,40);z=m.initial();rows=[]
        for k in range(1,33):
            if k>1:z=m.step(z)
            if k in [4,32]:
                for depth in [2,4,8]:
                    for horizon in [16,32]:
                        r={'start_pass':k,**compare(m,z,depth,horizon,a.repeats)};rows.append(r)
                        print(name,k,depth,horizon,{key:round(v['gap_per_pixel'],5) for key,v in r['methods'].items()},flush=True)
        result['scenes'][name]=rows
        Path(a.out).write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
