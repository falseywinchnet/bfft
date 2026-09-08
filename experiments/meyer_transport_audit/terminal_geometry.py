"""Diagnose terminal optimality, with no extrapolation or step factors.

The direct candidate imposes texture complementarity and residual balance
using the current feasible q. It is a falsifiable terminal reconstruction,
not a new convergent solver. Zero-gradient flux is retained, not normalized.
"""
from pathlib import Path
import sys,json,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap,grad,div,project_disk,laplacian_symbol
from experiments.meyer_transport_audit.certificate import tv

def fields(m,z):
    px,py=project_disk(z.tux,z.tuy,m.ru)
    gx,gy=project_disk(z.twx,z.twy,m.rw)
    return m.etau*px,m.etau*py,m.etaw/m.cw*gx,m.etaw/m.cw*gy

def measure(f,u,px,py,gx,gy,lam,mu):
    q=-div(px,py);v=-div(gx,gy)
    a=tv(u)-float(np.sum(u*q))
    b=mu*tv(q)-float(np.sum(v*q))
    c=.5*lam*float(np.sum((f-u-v-q/lam)**2))
    return {'cartoon_complementarity':a/f.size,
            'texture_complementarity':b/f.size,
            'balance':c/f.size,'gap_per_pixel':(a+b+c)/f.size,
            'q_rms_error_bound':float(np.sqrt(max(0,2*lam*(a+b+c)/f.size)))}

def direct(m,z):
    px,py,gx,gy=fields(m,z);q=-div(px,py);dx,dy=grad(q)
    mag=np.hypot(dx,dy);nonzero=mag>0
    gx=np.divide(m.mu*dx,mag,out=gx.copy(),where=nonzero)
    gy=np.divide(m.mu*dy,mag,out=gy.copy(),where=nonzero)
    u=m.image+div(gx,gy)-q/m.lam
    return measure(m.image,u,px,py,gx,gy,m.lam,m.mu)

def constant_bottom(f,lam,mu):
    """Sufficient capacity certificate for the absolute zero-energy bottom.

    The Poisson flux minimizes L2 norm, not maximum pointwise norm.
    Failure of its capacity test is inconclusive, not proof of infeasibility.
    Project before measuring so the reported upper bound is always feasible.
    """
    f=np.asarray(f,float);u=np.full_like(f,float(np.mean(f)))
    symbol=laplacian_symbol(f.shape);rhs=np.fft.fft2(f-u)
    spectrum=np.divide(rhs,symbol,out=np.zeros_like(rhs),where=symbol>0)
    phi=np.fft.ifft2(spectrum).real;gx,gy=grad(phi)
    peak=float(np.max(np.hypot(gx,gy)))
    feasible=peak<=mu
    gx,gy=project_disk(gx,gy,mu)
    stats=measure(f,u,np.zeros_like(f),np.zeros_like(f),gx,gy,lam,mu)
    return {'poisson_flux_peak':peak,'capacity':mu,
            'poisson_flux_fits':feasible,**stats}

def run(n):
    y,x=np.mgrid[:n,:n];rng=np.random.default_rng(983)
    scenes={'ramp':255*x/n,'edge':50.+150*(x>n//2),
            'low_carrier':100+2*np.sin(2*np.pi*(x+2*y)/n),
            'noise':rng.uniform(0,255,(n,n))}
    out={}
    for lam,mu in [(.02,20),(.05,40),(.1,80)]:
        for name,f in scenes.items():
            m=ReducedMeyerMap(f,lam,mu);z=m.initial();rows=[];qs=[]
            for k in range(1,2049):
                if k>1:z=m.step(z)
                if k in [16,32,64,128,512,2048]:
                    px,py,gx,gy=fields(m,z);q=-div(px,py)
                    rows.append({'passes':k,'ordinary':measure(f,z.u,px,py,gx,gy,lam,mu),
                                 'direct_terminal_candidate':direct(m,z)})
                    qs.append(q)
            for row,q in zip(rows,qs):
                row['q_rms_to_2048']=float(np.sqrt(np.mean((q-qs[-1])**2)))
            out[f'{name}_lambda{lam}_mu{mu}']={'trajectory':rows,
                'constant_bottom':constant_bottom(f,lam,mu)}
            print(name,lam,rows[2],flush=True)
    return {'size':n,'note':'2048 is a finite checkpoint, not an exact optimum.', 'scenes':out}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,default=32)
    ap.add_argument('--out',required=True);a=ap.parse_args()
    Path(a.out).write_text(json.dumps(run(a.size),indent=2))
