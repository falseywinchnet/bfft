"""Activate downstream admission with fixed two-carrier signals and test reduction."""
import json,argparse
from pathlib import Path
import numpy as np
from experiments.conv_downstream.core import Downstream,NAMES
from experiments.conv_downstream.run_study import carrier
from experiments.conv_admission_band.run_study import quality

def run(out):
 ops={k:Downstream(k) for k in NAMES};rows=[]
 y,x=np.mgrid[:33,:33];yf,xf=np.mgrid[:257,:257]/8;crop=np.s_[48:-48,48:-48]
 for angle in (0,30,45,60,90):
  ct,st=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
  for freq in (.5,.85,1.2):
   def signal(x,y):return .5+.28*np.cos(np.pi*freq*(ct*x+st*y)+.17)+.12*np.cos(np.pi*.7*(-st*x+ct*y)+1.03)
   src=signal(x,y);truth=signal(xf,yf)
   for name,op in ops.items():
    z=op.synthesize(src,(257,257));rows.append(dict(test='mixture',angle=angle,frequency=freq,method=name,**quality(z[crop],truth[crop])))
  for freq in (.3,.5,.7,.85,.95,1.05,1.2,1.5):
   phase=np.pi*freq*(ct*xf+st*yf)+.17;truth=.5+.4*np.cos(phase)
   for name,op in ops.items():
    z=op.synthesize(op.reduce(truth,(33,33)),(257,257))
    rows.append(dict(test='roundtrip_wave',angle=angle,frequency=freq,method=name,**quality(z[crop],truth[crop]),**carrier(z[crop],phase[crop])))
 Path(out).write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
