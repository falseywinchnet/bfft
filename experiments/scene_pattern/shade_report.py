"""Summarize controlled shade ablations; fitted alternatives are not promoted."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw
from .run import HERE,save
from .shade_diagnostic import POLYGON,CROP
from .structure_alignment import polygon_mask,REGIONS
from .floor_affine import affine_points
from .registration_audit import contrast_error,field_evidence

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'out/v2/room/alignment/shade');args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
 data=HERE/'out/shade-diagnostic';fits=HERE/'out/shade-model-audit';z=np.load(data/'observations.npz');s=z['current'];w=z['weights'];original=z['before'];ow=z['before_weights'];reference=2
 report=json.loads((fits/'receipt.json').read_text());roi=polygon_mask(w.shape[1:],POLYGON);shaft=polygon_mask(w.shape[1:],REGIONS[0]['polygon']);yy,xx=np.indices(roi.shape);points=np.stack((xx,yy),-1)
 old=np.load(HERE/'out/v2/room/alignment/floor/fields.npz')['fields'];new=np.load(HERE/'out/v2/room/alignment/structures/fields.npz')['fields'];ablation=[]
 for i in [0,1,3]:
  support=shaft&(ow[reference]>.2)&(ow[i]>.2) if i in [1,3] else np.zeros(roi.shape,bool)
  inspected=roi&(w[reference]>.2)&(w[i]>.2);valid=inspected&(ow[reference]>.2)&(ow[i]>.2)
  ablation.append(dict(view=i,evidence=field_evidence(new[i]-old[i],support,inspected),before_shaft_correction=contrast_error(original[reference],original[i],valid),after_shaft_correction=contrast_error(s[reference],s[i],valid)))
 for v in report['views']:
  if 'error' in v:continue
  i=v['view'];q=affine_points(points,np.asarray(v['parameters']),np.asarray(v['fit']['center']))
  warped=np.stack([ndimage.map_coordinates(s[i,...,k],[q[...,1],q[...,0]],order=1,mode='constant') for k in range(3)],-1)
  vw=ndimage.map_coordinates(w[i],[q[...,1],q[...,0]],order=1,mode='constant');valid=shaft&(w[reference]>.2)&(w[i]>.2)&(vw>.2)
  v['shaft_crosscheck']=dict(pixels=int(valid.sum()),before=contrast_error(s[reference],s[i],valid),after_unrestricted_shade_affine=contrast_error(s[reference],warped,valid))
  for label,moving in [('before',s[i]),('affine',warped)]:
   save((s[reference]+moving)[132:297,180:274]/2,out/f'shaft-{i}-{label}.png')
 report['shaft_ablation']=ablation;report['systemic_findings']=['Fitted-domain coverage is distinct from smoothness and positive Jacobian.','Other observed regions require validation before accepting a local model extension.','A fit rejected in one region does not determine visibility or fit validity in another region.']
 (out/'receipt.json').write_text(json.dumps(report,indent=2))
 for path in fits.glob('*.png'):shutil.copy2(path,out/path.name)
 for path in data.glob('*.png'):shutil.copy2(path,out/path.name)
 grid=Image.new('RGB',(1040,440),'#0b1015');draw=ImageDraw.Draw(grid)
 for j,n in enumerate(['before','current']):
  for i in range(4):
   a=Image.open(data/f'{n}-{i}.png').resize((256,181));grid.paste(a,(i*260,j*220+25));draw.text((i*260+3,j*220+3),f'{n}: photo {i+1}',fill='white')
 grid.save(out/'source-comparison.jpg');print(json.dumps(dict(ablation=ablation,crosschecks=[dict(view=v['view'],shaft=v.get('shaft_crosscheck'),error=v.get('error')) for v in report['views']]),indent=2))
if __name__=='__main__':main()
