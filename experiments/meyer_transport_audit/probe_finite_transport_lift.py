"""Measure geometry-change forcing and order dependence in the exact lift."""
from pathlib import Path
import sys,json,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap,rms
from experiments.meyer_transport_audit.finite_transport_lift import secant_matrix,increment_transfer,lifted_step
from experiments.benchmark_meyer_flow_jump import benchmark_scene

def matrices(m,z,h):
    return (secant_matrix(z.tux,z.tuy,h.tux,h.tuy,m.ru),
            secant_matrix(z.twx,z.twy,h.twx,h.twy,m.rw))

def run(n):
    y,x=np.mgrid[:n,:n];rng=np.random.default_rng(983)
    scenes={'ramp':255*x/n,'edge':50.+150*(x>n//2),
            'carrier':100+2*np.sin(2*np.pi*(x+2*y)/n),
            'crossing':benchmark_scene(n),'noise':rng.uniform(0,255,(n,n))}
    out={}
    for name,f in scenes.items():
        m=ReducedMeyerMap(f,.05,40);z=m.initial();h=m.subtract(m.step(z),z);ordinary=z;rows=[];maxerr=0.
        for k in range(1,130):
            maxerr=max(maxerr,rms(m.pack(m.subtract(z,ordinary))))
            if k in [4,16,64,128]:
                zn,hn=lifted_step(m,z,h);_,hnn=lifted_step(m,zn,hn)
                B0=matrices(m,z,h);B1=matrices(m,zn,hn)
                a=m.subtract(hn,h);an=m.subtract(hnn,hn)
                forcing=m.subtract(increment_transfer(m,h,*B1),increment_transfer(m,h,*B0))
                swapped=increment_transfer(m,increment_transfer(m,h,*B1),*B0)
                tangent=m.tangent(z,h)
                rows.append({'pass':k,
                    'increment_rms':rms(m.pack(h)),
                    'next_increment_change_rms':rms(m.pack(an)),
                    'geometry_change_forcing_rms':rms(m.pack(forcing)),
                    'geometry_forcing_over_next_increment_change':rms(m.pack(forcing))/max(rms(m.pack(an)),1e-30),
                    'order_commutator_over_second_increment':rms(m.pack(m.subtract(swapped,hnn)))/max(rms(m.pack(hnn)),1e-30),
                    'frozen_tangent_next_increment_relative_error':rms(m.pack(m.subtract(tangent,hn)))/max(rms(m.pack(hn)),1e-30)})
            if k<129:z,h=lifted_step(m,z,h);ordinary=m.step(ordinary)
        out[name]={'max_128_step_state_rms_discrepancy':maxerr,'checkpoints':rows}
        print(name,maxerr,rows[0],flush=True)
    return {'size':n,'lambda':.05,'mu':40,'note':'Exact lifted ordinary trajectories. Operator-order and geometry-forcing diagnostics use additional transforms; no speed claim.', 'scenes':out}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=64);p.add_argument('--out',required=True);a=p.parse_args()
    Path(a.out).write_text(json.dumps(run(a.size),indent=2))
