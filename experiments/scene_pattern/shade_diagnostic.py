"""Inspect shade evidence and separate shaft extrapolation from residual alignment."""
import json,argparse
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw
from .run import HERE,save
from .average_alignment import atlas_sources,average
from .structure_alignment import polygon_mask
from .floor_affine import fit_floor,affine_points

CROP=(65,45,270,190)
POLYGON=[(104,80),(199,52),(239,79),(248,146),(212,179),(155,177),(105,143)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('/tmp/shade_diagnostic'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 base=json.loads((HERE/'out/v2/room/receipt.json').read_text());stages={}
 for label,path in [('before','floor'),('current','structures')]:
  fields=np.load(HERE/f'out/v2/room/alignment/{path}/fields.npz')['fields']
  s,w=atlas_sources(HERE/'data/desktop-scene',base,700,method='linear',fields=fields);stages[label]=(s,w)
  save(average(s,w)[CROP[1]:CROP[3],CROP[0]:CROP[2]],args.out/f'{label}.png')
  for i in range(4):save(s[i,CROP[1]:CROP[3],CROP[0]:CROP[2]],args.out/f'{label}-{i}.png')
 np.savez_compressed(args.out/'observations.npz',before=stages['before'][0],before_weights=stages['before'][1],current=stages['current'][0],weights=stages['current'][1])
 report=[];roi=polygon_mask(w.shape[1:],POLYGON);reference=2
 for label,(s,w) in stages.items():
  for i in [0,1,3]:
   support=roi&(w[reference]>.2)&(w[i]>.2)
   f=np.load(HERE/f"out/v2/room/alignment/{'floor' if label=='before' else 'structures'}/fields.npz")['fields'][i]
   report.append(dict(stage=label,view=i,pixels=int(support.sum()),median_applied_shift=float(np.median(np.linalg.norm(f[support],axis=-1))),max_applied_shift=float(np.max(np.linalg.norm(f[support],axis=-1)))))
 (args.out/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
