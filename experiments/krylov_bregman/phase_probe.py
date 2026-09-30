"""Separate finite-basis error from change in nonlinear geometry."""
import argparse,json
import numpy as np
from .curved_meyer import MeyerQuotient
from .curved_study import scenes
from .core import arnoldi

def run():
 rows=[]
 for name,image in scenes(32):
  m=MeyerQuotient(image);z=m.initial.copy();last=0
  for prefix in (4,32,128):
   for _ in range(prefix-last):z=m.step(z)
   last=prefix;lin=m.linearize(z);r=lin.next_state-z
   bases={k:arnoldi(lin.action,r,k) for k in (4,8,12)}
   actual=z.copy();affine=np.zeros_like(z)
   for j in range(1,65):
    actual=m.step(actual);affine=r+lin.action(affine)
    if j in (16,64):
     for k,b in bases.items():
      pred=b.displacement(j)
      rows.append(dict(scene=name,prefix=prefix,horizon=j,depth=k,
       displacement=float(np.linalg.norm(actual-z)),
       compression_error=float(np.linalg.norm(pred-affine)),
       geometry_error=float(np.linalg.norm(z+affine-actual)),
       total_error=float(np.linalg.norm(z+pred-actual))))
 return {'rows':rows,'note':'full tangent and actual replay are diagnostic only'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
 with open(a.out,'w') as f:json.dump(run(),f,indent=2)
