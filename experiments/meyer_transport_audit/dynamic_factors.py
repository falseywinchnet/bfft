"""First dynamic-factor study: exact certificate statistics on the current ray.

The quadratic suggests a factor; it is NOT a majorizer. Actual acceptance
checks the exact recovered feasible gap against the ordinary step from the
same state. This is a one-step certificate, not a convergence theorem.
All candidate/statistic costs are charged. No additional map/FFT is used.
"""
from pathlib import Path
import sys, time, json, argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.meyer_transport_audit.model import ReducedMeyerMap, State, grad, div, project_disk
from experiments.meyer_transport_audit.certificate import certificate, tv
from experiments.benchmark_meyer_flow_jump import benchmark_scene


def disk_direction(tx,ty,hx,hy,radius):
    """Exact right directional derivative, including radial boundary motion."""
    mag=np.hypot(tx,ty)
    nx=tx/np.maximum(mag,1e-300); ny=ty/np.maximum(mag,1e-300)
    radial=nx*hx+ny*hy
    exterior=mag>radius
    boundary=(mag==radius)&(radial>0)
    mask=exterior|boundary
    factor=radius/np.maximum(mag,1e-300)
    return (np.where(mask,factor*(hx-nx*radial),hx),
            np.where(mask,factor*(hy-ny*radial),hy))


def tv_direction(x,h):
    gx,gy=grad(x);hx,hy=grad(h);mag=np.hypot(gx,gy)
    # Norm has a one-sided derivative |Dh| at a zero spatial gradient.
    return float(np.sum(np.where(mag>0,
        (gx*hx+gy*hy)/np.maximum(mag,1e-300),np.hypot(hx,hy))))


def gap_and_direction(m,z,h):
    """Exact gap and its right derivative along z+s*h; only local stencils."""
    px,py=project_disk(z.tux,z.tuy,m.ru)
    gx,gy=project_disk(z.twx,z.twy,m.rw)
    pdx,pdy=disk_direction(z.tux,z.tuy,h.tux,h.tuy,m.ru)
    gdx,gdy=disk_direction(z.twx,z.twy,h.twx,h.twy,m.rw)
    q=-m.etau*div(px,py);dq=-m.etau*div(pdx,pdy)
    v=-(m.etaw/m.cw)*div(gx,gy);dv=-(m.etaw/m.cw)*div(gdx,gdy)
    e=m.image-z.u-v
    gap=tv(z.u)+.5*m.lam*np.sum(e*e)-np.sum(m.image*q)+.5/m.lam*np.sum(q*q)+m.mu*tv(q)
    slope=(tv_direction(z.u,h.u)+m.lam*np.sum(e*(-h.u-dv))
           -np.sum(m.image*dq)+np.sum(q*dq)/m.lam+m.mu*tv_direction(q,dq))
    return float(gap),float(slope)


def first_branch_event(z,h,m):
    """First positive ray distance to a disk boundary; exact quadratic roots.

    Starts at the ordinary endpoint. This certifies only boundary location,
    not constancy of the exterior derivative and not descent.
    """
    best=np.inf
    for tx,ty,hx,hy,radius in [(z.tux,z.tuy,h.tux,h.tuy,m.ru),
                              (z.twx,z.twy,h.twx,h.twy,m.rw)]:
        aa=hx*hx+hy*hy;bb=2*(tx*hx+ty*hy);cc=tx*tx+ty*ty-radius*radius
        disc=bb*bb-4*aa*cc
        valid=(aa>0)&(disc>=0)
        root=np.sqrt(np.maximum(disc,0))
        # Stable quadratic formula, including tangent and zero roots.
        qq=-.5*(bb+np.copysign(root,bb))
        for roots in [np.divide(qq,aa,out=np.full_like(aa,np.inf),where=valid),
                      np.divide(cc,qq,out=np.full_like(aa,np.inf),where=valid&(qq!=0))]:
            roots=np.where(valid&(roots>1e-14),roots,np.inf)
            best=min(best,float(np.min(roots)))
    return best


def choose_factor(m,z,next_z,max_trials=2,event_cap=False):
    h=m.subtract(next_z,z)
    base_gap,slope=gap_and_direction(m,next_z,h)
    current_gap=certificate(m,z)['gap']
    # Match value at alpha=0, value at alpha=1, and right slope at alpha=1.
    curvature=current_gap-base_gap+slope
    proposal=1.
    if slope<0:
        advance=-slope/(2*curvature) if curvature>0 else 1.
        proposal=1.+min(1.,max(0.,advance))
    event=first_branch_event(next_z,h,m) if event_cap else np.inf
    if event_cap:proposal=min(proposal,1.+event)
    alpha=proposal;trials=0;selected=next_z;selected_gap=base_gap
    roundoff=64*np.finfo(float).eps*max(abs(base_gap),abs(current_gap),1.)
    for _ in range(max_trials):
        if alpha<=1.+1e-12:alpha=1.;break
        candidate=m.add_scaled(z,h,alpha)
        trial_gap=certificate(m,candidate)['gap'];trials+=1
        if np.isfinite(trial_gap) and trial_gap<=base_gap+roundoff:
            selected=candidate;selected_gap=trial_gap;break
        alpha=1.+.5*(alpha-1.)
    else:alpha=1.
    if selected is next_z:alpha=1.
    return selected,{'alpha':alpha,'proposal':proposal,'base_gap':base_gap,
        'selected_gap':selected_gap,'slope':slope,'curvature_model':curvature,
        'trials':trials,'certificate_evaluations':2+trials,
        'event_distance':event if np.isfinite(event) else None,
        'roundoff_allowance':roundoff}


def run_scene(f,lam,mu,steps):
    m=ReducedMeyerMap(f,lam,mu);report={}
    for name in ['ordinary','fixed175','dynamic_one','dynamic_two','boundary_cap']:
        z=m.initial();elapsed=0.;records=[];checkpoints=[]
        for k in range(2,steps+1):
            start=time.perf_counter()
            next_z=m.step(z)
            if k<=4 or name=='ordinary':z=next_z
            elif name=='fixed175':z=m.add_scaled(z,m.subtract(next_z,z),1.75)
            else:
                z,d=choose_factor(m,z,next_z,1 if name=='dynamic_one' else 2,name=='boundary_cap')
                records.append(d)
            elapsed+=time.perf_counter()-start
            if k in [16,32,64,128]:
                checkpoints.append({'passes':k,'ms':elapsed*1000,**certificate(m,z)})
        report[name]={'checkpoints':checkpoints,'decisions':records}
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,default=64)
    ap.add_argument('--steps',type=int,default=128);ap.add_argument('--out',required=True);a=ap.parse_args()
    n=a.size;y,x=np.mgrid[:n,:n];rng=np.random.default_rng(983)
    scenes={'ramp':np.asarray(255*x/n,float),'edge':50.+150.*(x>n//2),
            'low_carrier':100+2*np.sin(2*np.pi*(x+2*y)/n),
            'crossing':benchmark_scene(n),'noise':rng.uniform(0,255,(n,n))}
    result={'size':n,'steps':a.steps,'backend':'Identical NumPy FFT map; all decision costs included. No native speed claim.',
        'contract':'Selected feasible gap <= ordinary endpoint gap from same input state, up to recorded rounding allowance.',
        'scenes':{}}
    for lam,mu in [(.02,20),(.05,40),(.1,80)]:
        for name,f in scenes.items():
            key=f'{name}_lambda{lam}_mu{mu}'
            result['scenes'][key]=run_scene(f,lam,mu,a.steps)
            Path(a.out).write_text(json.dumps(result,indent=2))
            print(key,{m:round(r['checkpoints'][-1]['gap_per_pixel'],6) for m,r in result['scenes'][key].items()},flush=True)

if __name__=='__main__':main()
