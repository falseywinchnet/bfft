"""Summarize the unconditional single-wavelength-per-band comparison."""
from pathlib import Path
import json,statistics
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'direct_spectral_m4'
summary=[]
for name,old in [('aperture-canyon','aperture'),('standard','standard'),('mirror-relay','mirror-relay'),('occlusion-garden','occlusion-garden')]:
 d=json.loads((p/f'one_per_band_{name}.json').read_text());assert len(d['frames'])==6
 timings={n:[r['ms'] for r in d['frames'] if r['samples_per_band']==n] for n in [0,1]}
 rows={n:next(r for r in d['frames'] if r['samples_per_band']==n) for n in [0,1]}
 a=np.asarray(Image.open(p/f'one_per_band_{name}.json.1.ppm')).astype(float)
 metrics={}
 for target,b in [('current',np.asarray(Image.open(p/f'one_per_band_{name}.json.0.ppm'))),('direct8',np.asarray(Image.open(p/f'direct_spectral_{old}.json.8.ppm')))]:
  delta=abs(a-b);maximum=delta.max(axis=2)
  metrics[target]={'rms_byte_error':float(np.sqrt((delta*delta).mean())),'mean_byte_error':float(delta.mean()),'max_byte_error':int(delta.max()),'pixels_changed':int((maximum>0).sum()),'pixels_over_1':int((maximum>1).sum()),'pixels_over_8':int((maximum>8).sum())}
 summary.append({'scene':name,'median_ms':{n:statistics.median(t) for n,t in timings.items()},'runs_ms':timings,
  'speedup':statistics.median(timings[0])/statistics.median(timings[1]),'classification_calls':{n:rows[n]['classification_calls'] for n in [0,1]},
  'emitter_quadrature':{n:rows[n]['quadrature'] for n in [0,1]},'image_differences':metrics})
 for r in d['frames']:
  assert r['samples_per_band'] or r['changed_bytes']==0
(p/'one_per_band_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(18,4.8),layout='constrained')
for ax,(title,file) in zip(axes,[('Current renderer','one_per_band_aperture-canyon.json.0.ppm'),('One unconditional sample per band','one_per_band_aperture-canyon.json.1.ppm'),('Eight unconditional samples per band','direct_spectral_aperture.json.8.ppm')]):
 ax.imshow(Image.open(p/file));ax.set_title(title,fontsize=13);ax.axis('off')
fig.savefig(p/'one_per_band_comparison.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
for ax,row in zip(axes.flat,summary):
 ax.imshow(Image.open(p/f"one_per_band_{row['scene']}.json.1.ppm"));ax.axis('off');ax.set_title(f"{row['scene']} · {row['median_ms'][1]:.0f} ms · {row['speedup']:.2f}×")
fig.suptitle('One unconditional sample per band · 800 × 600 · M4 Mini',fontsize=16)
fig.savefig(p/'one_per_band_scenes.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))
