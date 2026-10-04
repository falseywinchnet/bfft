"""Mutation: reduced flow with exact primitive nonlinearity and retained memory.

For Meyer F(z+d)=F(z)+Jd+L q(d). Reuse the fixed spatial response L through
its adjoint on the acquired basis; reevaluate the true disk remainder every
reduced step. This is exact Galerkin evaluation, not exact full-state closure.
"""
from dataclasses import dataclass
import numpy as np
from .core import arnoldi
from experiments.meyer_transport_audit.model import grad,div,project_disk,project_disk_derivative


def screened(model,x,weight):
    return np.fft.ifft2(np.fft.fft2(x,axes=(-2,-1))/(1+weight*model.full.symbol),axes=(-2,-1)).real


def response(model,q):
    """Whitened projection remainder -> full whitened quotient response."""
    qux,quy,qwx,qwy=np.asarray(q).reshape(4,*model.shape)
    du=2*np.sqrt(model.a)*screened(model,div(qux,quy),model.a)
    dw=-screened(model,du,model.b)+2*np.sqrt(model.b)*screened(model,div(qwx,qwy),model.b)
    ux,uy=grad(du);wx,wy=grad(dw)
    return np.stack([du+dw,np.sqrt(model.a)*ux+qux,np.sqrt(model.a)*uy+quy,
                     np.sqrt(model.b)*wx+qwx,np.sqrt(model.b)*wy+qwy]).ravel()


def response_adjoint_basis(model,Q):
    """Batched L^T Q; two screened transforms per basis column.

    The batch implementation uses last-axis differences explicitly: the
    ordinary scalar grad/div helpers use fixed image axes.
    """
    k=Q.shape[1]
    if k==0:return np.empty((4*model.n,0))
    v=Q.T.reshape(k,5,*model.shape)
    def ds(x,y):return -(x-np.roll(x,1,axis=-1)+y-np.roll(y,1,axis=-2))
    def dg(x):return np.roll(x,-1,axis=-1)-x,np.roll(x,-1,axis=-2)-x
    A=v[:,0]+np.sqrt(model.a)*ds(v[:,1],v[:,2])
    B=v[:,0]+np.sqrt(model.b)*ds(v[:,3],v[:,4])
    sb=screened(model,B,model.b)
    sa=screened(model,A-sb,model.a)
    ax,ay=dg(sa);bx,by=dg(sb)
    return np.stack([v[:,1]-2*np.sqrt(model.a)*ax,v[:,2]-2*np.sqrt(model.a)*ay,
        v[:,3]-2*np.sqrt(model.b)*bx,v[:,4]-2*np.sqrt(model.b)*by],axis=1).reshape(k,4*model.n).T.copy()


@dataclass
class FactorChart:
    model:object
    anchor:np.ndarray
    basis:object
    disks:np.ndarray
    projected_disks:np.ndarray
    disk_basis:np.ndarray
    derivative_basis:np.ndarray
    reduced_response:np.ndarray
    adjoint_actions:int

    @classmethod
    def acquire(cls,model,z,depth=4):
        lin=model.linearize(z);b=arnoldi(lin.action,lin.next_state-z,depth)
        return cls.from_basis(model,z,b)

    @classmethod
    def from_basis(cls,model,z,b,response_basis=None):
        k=b.actions;fields=z.reshape(5,*model.shape)/model.weights[:,None,None]
        Q=b.Q.reshape(5,model.n,k)/model.weights[:,None,None]
        base=[];projected=[];D=[]
        for index,radius in [(1,model.full.ru),(3,model.full.rw)]:
            tx,ty=fields[index:index+2]
            px,py=project_disk(tx,ty,radius)
            base.extend([tx.ravel(),ty.ravel()]);projected.extend([px.ravel(),py.ravel()])
            dx=[];dy=[]
            for j in range(k):
                a,c=project_disk_derivative(tx,ty,Q[index,:,j].reshape(model.shape),Q[index+1,:,j].reshape(model.shape),radius)
                dx.append(a.ravel());dy.append(c.ravel())
            D.extend([np.column_stack(dx) if k else np.empty((model.n,0)),
                      np.column_stack(dy) if k else np.empty((model.n,0))])
        return cls(model,z.copy(),b,np.stack(base),np.stack(projected),Q[1:].reshape(4*model.n,k),
            np.stack(D).reshape(4*model.n,k),
            response_adjoint_basis(model,b.Q) if response_basis is None else response_basis,k)

    def remainder(self,c):
        m=self.model
        t=self.disks+(self.disk_basis@c).reshape(4,m.n)
        pu=project_disk(t[0],t[1],m.full.ru);pw=project_disk(t[2],t[3],m.full.rw)
        q=np.stack([*pu,*pw])-self.projected_disks-(self.derivative_basis@c).reshape(4,m.n)
        q*=m.weights[1:,None]
        return q.ravel()

    def coordinates(self,horizon):
        source=np.zeros(self.basis.actions)
        if len(source):source[0]=self.basis.beta
        c=np.zeros_like(source)
        for _ in range(horizon):
            c=source+self.basis.H@c+self.reduced_response.T@self.remainder(c)
        return c

    def candidate(self,horizon):
        return self.anchor+self.basis.Q@self.coordinates(horizon)


def enrich_from_defect(chart,probe_horizon=8,additional=2):
    """Acquire a coupled escape direction and its tangent response.

    The probe is an actual map evaluation, not a replayed future trajectory.
    Existing adjoint responses are retained. At most `additional` new tangent
    and adjoint actions are needed; one live map probe is additionally charged.
    """
    from .core import Arnoldi,identity
    m=chart.model;b=chart.basis;c=chart.coordinates(probe_horizon)
    point=chart.anchor+b.Q@c
    actual=m.step(point)-chart.anchor
    seed=actual-b.Q@(b.Q.T@actual)
    qs=[b.Q[:,j].copy() for j in range(b.actions)]
    aq=[(b.Q@b.H+b.remainder)[:,j].copy() for j in range(b.actions)]
    anchor=m.representative(chart.anchor)
    for _ in range(additional):
        v=seed.copy()
        for _ in range(2):
            for q in qs:v-=(q@v)*q
        norm=np.linalg.norm(v)
        if norm<1e-12*max(np.linalg.norm(actual),1e-30):break
        v/=norm;qs.append(v)
        av=m.encode(m.full.tangent(anchor,m.representative(v)))
        aq.append(av);seed=av
    Q=np.column_stack(qs);AQ=np.column_stack(aq);H=Q.T@AQ
    bn=Arnoldi(Q,H,AQ-Q@H,b.beta,len(qs),identity)
    R=np.column_stack([chart.reduced_response,response_adjoint_basis(m,Q[:,b.actions:])]) if len(qs)>b.actions else chart.reduced_response
    return FactorChart.from_basis(m,chart.anchor,bn,R)
