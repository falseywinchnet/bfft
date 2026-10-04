"""Matched-resolution point and area assets for the comparison viewer."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from .paper_images import *
from .paper_images import _scene,_global_lattice_from_atlas

OUT=Path('output/support_geometry/conv_bounded_sharpening/paper-images')

def run():
 rows=[]
 for kind in CASES:
  source=_scene(kind,17,**CASES[kind])
  control,_=finite_joint_control_nets(source);P=_global_lattice_from_atlas(control[...,None])[...,0]
  A,b=constraints(source);D=refined_derivative_rows(P);base_bound=np.max(abs(derivative_rows(P)@P.ravel()));nets={'joint-base':P}
  for name,cap,strength in [('cap1',1,1),('cap2',2,1),('cap4',4,1),('cap2-strong',2,4)]:
   AA=sparse.vstack([A,D,-D],format='csr');bb=np.r_[b,np.full(2*D.shape[0],-cap*base_bound)]
   nets[name]=admit(P,concentration(P,source,strength),AA,bb)[0]
  for side in (65,513):
   tp=_scene(kind,side,**CASES[kind]);ta=true_area(kind,side,8 if side==65 else 4)
   values={'truth':{'point':tp,'area':ta}}
   for name,Q in nets.items():values[name]={'point':point(Q,side),'area':area(Q,side)}
   W=lanczos_weights(np.linspace(0,16,side));AW=area_weights(side,lanczos_weights,8)
   values['lanczos3']={'point':W@source@W.T,'area':AW@source@AW.T}
   values['paper-convstar']={'point':np.asarray(distilled_conv_synthesis(source,(side,side)))}
   for name,views in values.items():
    for view,value in views.items():
     reference=tp if view=='point' else ta;error=value-reference
     save_image(OUT/f'{kind}-{name}-{view}-{side}.png',value)
     e=np.clip(error/.10,-1,1);rgb=np.ones(e.shape+(3,))*.95
     rgb[:,:,0]-=.75*np.maximum(-e,0);rgb[:,:,2]-=.75*np.maximum(e,0);rgb[:,:,1]-=.75*abs(e)
     Image.fromarray(np.uint8(rgb*255)).save(OUT/f'{kind}-{name}-{view}-error-{side}.png')
     rows.append({'field':kind,'method':name,'view':view,'side':side,'mse':float(np.mean(error**2)),
      'maxError':float(np.max(abs(error))),'sourceRangeExcursion':float(max(0,value.max()-source.max(),source.min()-value.min()))})
  print(kind,flush=True)
 (OUT/'resolution-metrics.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':run()
