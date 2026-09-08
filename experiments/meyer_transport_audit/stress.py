"""Parameter and source stress screen; no tuning per source."""
from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.meyer_transport_audit.native_audit import Native,library
from experiments.meyer_transport_audit.model import ReducedMeyerMap
from experiments.meyer_transport_audit.certificate import certificate
from experiments.benchmark_meyer_flow_jump import benchmark_scene

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--lib',default='/tmp/meyer_audit.dylib');a=ap.parse_args()
    n=128;y,x=np.mgrid[:n,:n];rng=np.random.default_rng(983)
    scenes={'constant':np.full((n,n),100.),'ramp':255*x/n+np.zeros_like(y),
            'edge':50.+150.*(x>n//2),'checker':127.+100.*(-1.)**(x+y),
            'white_noise':rng.uniform(0,255,(n,n)),
            'low_carrier':100+2*np.sin(2*np.pi*(x+2*y)/n),
            'crossing':benchmark_scene(n),'noisy_crossing':benchmark_scene(n)+rng.normal(0,15,(n,n))}
    lib=library(a.lib);report={}
    for lam,mu in [(.02,20),(.05,40),(.1,80)]:
        for name,f in scenes.items():
            f=np.ascontiguousarray(f,dtype=float);p=Native(lib,n,1,lam,mu);m=ReducedMeyerMap(f,lam,mu);rows={}
            for alpha in [1.,1.5,1.75,1.9]:
                v=p.split(f,0 if alpha==1 else 2,128,alpha)
                z=m.unpack(p.state().ravel());rows[str(alpha)]={**certificate(m,z),'residual_rms':m.residual_norm(z),'finite':bool(np.isfinite(v).all())}
            report[f'{name}_lambda{lam}_mu{mu}']=rows;p.close()
        Path(a.out).write_text(json.dumps(report,indent=2));print(lam,mu,'done',flush=True)

if __name__=='__main__':main()
