"""Summarize saved direct-band benchmarks and render a comparison figure."""
from pathlib import Path
import json,statistics
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'direct_spectral_m4'
summary={'scenes':[],'aperture_vs_256':[],'ray_vs_1024':[]}
for name in ['aperture','standard','mirror-relay','occlusion-garden']:
 d=json.loads((p/f'direct_spectral_{name}.json').read_text());rows=[]
 for n in [0,4,8,16,32,64]:
  frames=[r for r in d['frames'] if r['samples_per_band']==n]
  rows.append({'samples_per_band':n,'mean_ms':statistics.mean(r['ms'] for r in frames),'runs_ms':[r['ms'] for r in frames],
               'classification_calls':frames[0]['classification_calls'],'emitter_quadrature':frames[0]['quadrature']})
 summary['scenes'].append({'scene':d['scene'],'modes':rows})
ref=np.array(Image.open(p/'direct_spectral_aperture256.json.256.ppm')).astype(float)
images={n:np.array(Image.open(p/f'direct_spectral_aperture.json.{n}.ppm')) for n in [0,4,8,16,32,64]}
for n,b in images.items():
 delta=abs(b-ref);summary['aperture_vs_256'].append({'samples_per_band':n,'mean_byte_error':float(delta.mean()),
  'rms_byte_error':float(np.sqrt((delta*delta).mean())),'max_byte_error':int(delta.max()),'bytes_over_1':int((delta>1).sum()),
  'pixels_over_1':int((delta.max(axis=2)>1).sum())})
rays=json.loads((p/'direct_spectral_rays.json').read_text())['rays']
assert len(rays)==56
for ray in rays:
 for s in ray['samples']:
  assert not s['n'] or s['classes']==0
  assert np.isfinite(s['value']).all()
for n in [0,4,8,16,32,64,128,256,512]:
 delta=np.array([np.array(next(s['value'] for s in ray['samples'] if s['n']==n))-np.array(ray['samples'][-1]['value']) for ray in rays])
 summary['ray_vs_1024'].append({'samples_per_band':n,'linear_rmse':float(np.sqrt((delta*delta).mean())),'linear_max_error':float(abs(delta).max())})
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained',facecolor='#f5f5f5')
for axis,(title,b) in zip(ax[0],[('Current classification + transport',images[0]),('Direct: 8 samples / band',images[8]),('Direct: 256 samples / band',ref.astype(np.uint8))]):
 axis.imshow(b);axis.set_title(title);axis.axis('off')
crop=np.s_[315:380,250:340]
for axis,(title,b) in zip(ax[1],[('Same crop — current',images[0]),('Same crop — direct 8',images[8]),('Same crop — direct 256',ref.astype(np.uint8))]):
 axis.imshow(b[crop],interpolation='nearest');axis.set_title(title);axis.axis('off')
fig.suptitle('Unconditional spectral transport · aperture-canyon · 800 × 600\nAll three bands integrated; spatial boundary coverage retained',fontsize=16)
fig.savefig(p/'comparison.png',dpi=160);plt.close(fig)
print('Saved summary and comparison; 56 ray probes finite, every direct mode has zero spectral classifications.')
