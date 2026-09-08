"""Render saved measurements without rerunning the native experiment."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/support_geometry/conv_downstream'
r=json.loads((OUT/'conv_downstream4/results.json').read_text())
one=json.loads((OUT/'conv_downstream1.json').read_text())
q=json.loads((OUT/'conv_downstream_mixtures.json').read_text())
names=list(r['timing'][0]['methods']); labels=['V1','Ledger diagonal','Projection diagonal','Both diagonal','Ledger full energy','Projection value','Equal blend','Amplitude blend','Exact basin integral','Hat basin']
base={x['image']:x for x in r['images'] if x['method']=='v1'}
gains={n:[100*(1-x['roundtrip']['mse']/base[x['image']]['roundtrip']['mse']) for x in r['images'] if x['method']==n] for n in names}
timing=lambda data:{n:data['timing'][0]['methods'][n]['median']/data['timing'][0]['methods']['v1']['median'] for n in names}
t4=timing(r);t1=timing(one)
fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained');fig.suptitle('Fixed downstream CONV formulas — source stencil and raw currents unchanged',fontsize=16)
y=np.arange(len(names));a=axs[0,0]
a.barh(y-.17,[t4[n] for n in names],.32,label='4 workers',color='#227c9d');a.barh(y+.17,[t1[n] for n in names],.32,label='1 worker',color='#80c4b5')
a.set_yticks(y,labels);a.invert_yaxis();a.set_xlim(.9,1.25);a.axvline(1,color='black',lw=.7);a.set_xlabel('Complete 512→64→512 time / V1 (31 repeats)');a.legend(loc='lower right',fontsize=8);a.grid(axis='x',alpha=.2)
a=axs[0,1];means=[np.mean(gains[n]) for n in names]
a.barh(y,means,color=['#227c9d' if v>=0 else '#c56b46' for v in means]);a.set_yticks(y,labels);a.invert_yaxis();a.set_xscale('symlog',linthresh=.015);a.set_xlim(-15,.08);a.axvline(0,color='black',lw=.7);a.set_xlabel('Mean per-image MSE improvement, % (symlog scale)')
for i,v in enumerate(means):a.text(-2 if v < -1 else v,i,f' {v:+.4f}%',va='center',fontsize=8,color='white' if v < -1 else 'black',ha='left' if v>=0 or v < -1 else 'right')
a.grid(axis='x',alpha=.2)
a=axs[1,0]
for name,color in [('v1','#227c9d'),('basin_hat','#c56b46')]:
 for angle,style in [(0,'-'),(45,'--')]:
  z=[x for x in q if x['test']=='roundtrip_wave' and x['method']==name and x['angle']==angle]
  a.plot([x['frequency'] for x in z],[x['gain'] for x in z],style,color=color,marker='.',label=f'{name}, {angle}°')
a.set_xlabel('Frequency / coarse-grid axial Nyquist');a.set_ylabel('Recovered fundamental gain');a.set_title('Hat basin attenuation: 257→33→257');a.legend(fontsize=8);a.grid(alpha=.2)
a=axs[1,1]
for kind,color,offset in [('step','#80c4b5',-.17),('strip','#227c9d',.17)]:
 values=[max(x['excursion'] for x in r['edges'] if x['method']==n and x['kind']==kind) for n in names]
 a.barh(y+offset,values,.32,label=kind,color=color)
a.set_yticks(y,labels);a.invert_yaxis();a.set_xlabel('Worst excursion outside [0, 1] across 15 placements');a.set_title('Synthesis: step and thin strip');a.legend(fontsize=8);a.grid(axis='x',alpha=.2)
fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
lines=['| Fixed formula | Full time, 4 workers | Full time, 1 worker | Mean MSE improvement | Image range |','|---|---:|---:|---:|---:|']
for n,l in zip(names,labels):
 g=gains[n];lines.append(f'| {l} | {t4[n]:.3f}× | {t1[n]:.3f}× | {np.mean(g):+.4f}% | {min(g):+.4f} to {max(g):+.4f}% |')
(OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n\nPositive MSE improvement is better. Arithmetic mean of six relative image improvements; this is not a confidence interval. Times are complete native/Python pipeline medians.\n')
print('\n'.join(lines))
