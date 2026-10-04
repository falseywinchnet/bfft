"""Controlled shade-only fitting audit; never modifies the room correction."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage
from .run import HERE,save
from .floor_affine import fit_floor,affine_points
from .structure_alignment import polygon_mask
from .shade_diagnostic import POLYGON,CROP
from .average_alignment import average
from .core import srgb_to_lab

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('/tmp/shade_model_audit'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 z=np.load(HERE/'out/shade-diagnostic/observations.npz');s=z['current'];w=z['weights'];reference=2;roi=polygon_mask(w.shape[1:],POLYGON);yy,xx=np.indices(roi.shape);points=np.stack((xx,yy),-1);records=[]
 for i in [0,1,3]:
  try:
   p,c,r=fit_floor(s[reference],s[i],w[reference],w[i],POLYGON,structural=True)
   q=affine_points(points,p,c);warped=np.stack([ndimage.map_coordinates(s[i,...,k],[q[...,1],q[...,0]],order=1,mode='constant') for k in range(3)],-1)
   validity=ndimage.map_coordinates(w[i],[q[...,1],q[...,0]],order=1,mode='constant');common=np.minimum(w[i],validity);valid=roi&(common>.2)&(w[reference]>.2)
   errors=[]
   for label,moving in [('before',s[i]),('affine',warped)]:
    pair=average(np.stack([s[reference],moving]),np.stack([w[reference],common]));save(pair[CROP[1]:CROP[3],CROP[0]:CROP[2]],args.out/f'pair-{i}-{label}.png')
    a=srgb_to_lab(s[reference])[...,0];b=srgb_to_lab(moving)[...,0];a-=ndimage.gaussian_filter(a,3);b-=ndimage.gaussian_filter(b,3);errors.append(float(np.mean(np.abs(a[valid]-b[valid]))))
   records.append(dict(view=i,fit=r,parameters=p.tolist(),common_pixels=int(valid.sum()),contrast_before=errors[0],contrast_after=errors[1],at_parameter_bound=bool(np.any(np.abs(p[:2])>27.9) or np.any(np.abs(p[2:])>29.9))))
   print(json.dumps(dict(view=i,selection_before=r['before_selection'],selection_after=r['selected']['selection_error'],report_before=r['before_test'],report_after=r['after_test'],contrast=errors,bound=records[-1]['at_parameter_bound'])),flush=True)
  except ValueError as e:records.append(dict(view=i,error=str(e)))
 report=dict(reference=reference,polygon=POLYGON,views=records,seconds=time.perf_counter()-start,sampler='linear diagnostic only',limitations='Manually selected audit ROI. Affine fits use current source observations and three spatial block splits; reporting blocks are from the same photographs. Pair images demonstrate fit behavior, not a replacement fused view. No result is applied to the room.')
 (args.out/'receipt.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
