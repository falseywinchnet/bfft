"""Small explicit Jacobians to diagnose geometry, not runtime acquisition."""
import argparse,json
import numpy as np
from .sinkhorn_transport import Sinkhorn
from .sinkhorn_study import problem
from .curved_meyer import MeyerQuotient
from .curved_study import scenes

def run():
    rows=[]
    s=Sinkhorn(*problem(64,0,'image'),.01)
    image=next(im for name,im in scenes(8) if name=='camera')
    for name,m in [('sinkhorn',s),('meyer',MeyerQuotient(image))]:
        z=m.initial.copy();last=0
        for prefix in (0,4,32):
            for _ in range(prefix-last):z=m.step(z)
            last=prefix;lin=m.linearize(z);eye=np.eye(len(z))
            J=np.column_stack([lin.action(e) for e in eye])
            eig=np.linalg.eigvals(J)
            rows.append(dict(model=name,prefix=prefix,dimension=len(z),
                max_imaginary=float(np.max(np.abs(eig.imag))),
                complex_eigenvalues=int(np.sum(np.abs(eig.imag)>1e-8)),
                minimum_real=float(np.min(eig.real)),maximum_modulus=float(np.max(np.abs(eig))),
                nonnormality=float(np.linalg.norm(J.T@J-J@J.T)/np.linalg.norm(J)**2)))
    return {'note':'small explicit diagnostic Jacobians, not charged to any solver or used in its decisions','rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    with open(a.out,'w') as f:json.dump(run(),f,indent=2)
