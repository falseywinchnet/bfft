"""Same-resolution carrier audit: proposal, admission, area, and display."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .paper_images import (CASES,_scene,point,area,true_area,flat_controls,
 area_weights,basis,save_image,distilled_conv_synthesis)
from .study import finite_joint_control_nets,_global_lattice_from_atlas
from experiments.conv_warp.joint_reference import canonical_sampled_control_net,_shared_support_bounds

OUT=Path('output/support_geometry/conv_bounded_sharpening/paper-images')

def run():
 source=_scene('carrier',17,**CASES['carrier']);raw=_global_lattice_from_atlas(canonical_sampled_control_net(source[...,None]))[...,0];raw[::5,::5]=source
 lo,hi=_shared_support_bounds(source[...,None]);clipped=np.clip(raw,lo[...,0],hi[...,0])
 control,diag=finite_joint_control_nets(source);base=_global_lattice_from_atlas(control[...,None])[...,0]
 rows=[];fig,axes=plt.subplots(2,4,figsize=(12,6.6),layout='constrained')
 for side in (65,129,513):
  t=_scene('carrier',side,**CASES['carrier']);a=true_area('carrier',side,4)
  # Order 8 tensor quadrature on polynomial coefficients, independently of
  # the order 4 area() execution; every basin split at source knots.
  W=area_weights(side,basis,8);checked=W@flat_controls(base)@W.T
  bp=point(base,side);ba=area(base,side)
  row={'side':side,'pointMse':float(np.mean((bp-t)**2)),'areaMse':float(np.mean((ba-a)**2)),
   'baseAreaMinusPointRms':float(np.sqrt(np.mean((ba-bp)**2))),
   'truthAreaMinusPointRms':float(np.sqrt(np.mean((a-t)**2))),
   'areaVersusPointErrorChangeRms':float(np.sqrt(np.mean(((ba-a)-(bp-t))**2))),
   'quadrature4vs8Max':float(np.max(abs(checked-ba))),
   'stages':{name:{'pointMse':float(np.mean((point(P,side)-t)**2)),'areaMse':float(np.mean((area(P,side)-a)**2))} for name,P in [('raw',raw),('clipped',clipped),('admitted',base)]}}
  rows.append(row)
  for name,img in [('truth',t),('truth-area',a),('joint-base',bp),('joint-base-area',ba)]:save_image(OUT/f'carrier-audit-{name}-{side}.png',img)
  if side==65:
   for col,(name,img) in enumerate([('Analytic point',t),('Analytic area',a),('Baseline point',bp),('Baseline area',ba)]):
    axes[0,col].imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[0,col].set_title(name)
    error=img-(a if col%2 else t)
    axes[1,col].imshow(error,cmap='RdBu_r',vmin=-.015,vmax=.015,interpolation='nearest')
    axes[1,col].set_title('Error · fixed ±0.015')
    for ax in axes[:,col]:ax.set_xticks([]);ax.set_yticks([])
 fig.suptitle('Carrier: identical 65 × 65 grids and identical pixel display',fontsize=15)
 fig.savefig(OUT/'carrier-integral-audit.png',dpi=160);plt.close(fig)
 (OUT/'carrier-integral-audit.json').write_text(json.dumps({'admission':diag,'rows':rows},indent=2)+'\n')
 print(json.dumps(rows,indent=2))

if __name__=='__main__':run()
