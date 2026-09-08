"""Report retained planar virtual-view field results from saved frames."""
from pathlib import Path
import json,statistics
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'reflected_gather_m4';rows=[]
for name in ['mirror-relay','aperture-canyon','standard','occlusion-garden']:
 d=json.loads((p/f'reflected_gather_{name}.json').read_text());frames={n:[r for r in d['frames'] if r['mode']==n] for n in [0,1]}
 a=np.asarray(Image.open(p/f'reflected_gather_{name}.json.0.ppm')).astype(float);b=np.asarray(Image.open(p/f'reflected_gather_{name}.json.1.ppm')).astype(float)
 delta=abs(a-b).max(axis=2)
 rows.append({'scene':name,'origins':d['origins'],'cold_ms':{n:frames[n][0]['ms'] for n in [0,1]},
  'warm_ms':{n:statistics.mean(r['ms'] for r in frames[n][1:]) for n in [0,1]},'allocation_ms':d['allocation_ms'],
  'gathers':frames[1][0]['reflected_gathers'],'source_quadrature_saved_per_warm_frame':frames[0][0]['source_quadrature']-frames[1][0]['source_quadrature'],
  'new_field_construction_quadrature':frames[1][0]['reflected_field_quadrature'],'field_samples':frames[1][0]['reflected_samples_total'],
  'max_byte_difference':int(delta.max()),'pixels_changed':int((delta>0).sum()),'pixels_over_1':int((delta>1).sum())})
(p/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for ax,(mode,title) in zip(axes,[(0,'One per band — existing camera field'),(1,'One per band — reflected fields retained')]):
 ax.imshow(Image.open(p/f'reflected_gather_mirror-relay.json.{mode}.ppm'));ax.set_title(title);ax.axis('off')
fig.savefig(p/'comparison.png',dpi=160);plt.close(fig)
print(json.dumps(rows,indent=2))
