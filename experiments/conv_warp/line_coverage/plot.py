from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('output/support_geometry/conv_line_coverage')
r=json.loads((root/'results.json').read_text());c=next(c for c in r['cases'] if c['width']==1 and c['phase']==.585)
p=json.loads((root/'profile.json').read_text())
x=np.array([v['x'] for v in c['profile']]);current=np.array([v['deficit'] for v in c['profile']]);L=p['profile']['L'];R=p['profile']['R'];e=p['profile']['edgeWidth']
def S(t):
 t=np.clip(t,0,1);return t**3*(10+t*(-15+6*t))
y=S((x-L)/e+.5)-S((x-R)/e+.5)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained',gridspec_kw={'width_ratios':[1.3,1]})
a=axes[0];a.axvspan(L-32,R-32,color='#d5dfe3',alpha=.65,label='Original line extent')
a.plot(x-32,current,color='#a65032',lw=2.4,label='Current CONV field')
a.plot(x-32,y,color='#176a7a',lw=2.4,label='Coverage-preserving C² quintic')
a.scatter([0,1],[.415,.585],marker='s',s=50,color='#263b43',zorder=5,label='Source pixel averages')
a.set(xlim=(-1.5,2.7),ylim=(-.03,1.08),xlabel='Source pixel coordinate',ylabel='Line contrast (0 = background, 1 = full line)',title='Same two source pixel averages')
a.legend(loc='upper left',fontsize=8.5,frameon=False);a.grid(alpha=.14)
a=axes[1];selected=[next(v for v in p['views'] if v['delta']==d and v['phase']==0) for d in [1,.5,2]]
labels=['Same size','2× larger','2× smaller'];positions=np.arange(3)
for offset,key,color,label in [(-.25,'geometric','#71838d','Exact line coverage'),(0,'current','#a65032','Current CONV area'),(.25,'coverageProfile','#176a7a','Coverage-preserving quintic')]:
 vals=[v[key] for v in selected];bars=a.bar(positions+offset,vals,width=.24,color=color,label=label)
 for b,v in zip(bars,vals):a.text(b.get_x()+b.get_width()/2,v+.018,f'{v:.3f}',ha='center',fontsize=8)
a.set(xticks=positions,xticklabels=labels,ylim=(0,1.22),ylabel='Peak output-pixel contrast',title='After output pixel-area integration');a.legend(loc='upper right',fontsize=8,frameon=False);a.grid(axis='y',alpha=.14);a.set_axisbelow(True)
fig.suptitle('Fine-line attenuation: source coverage versus nodal reconstruction',fontsize=14)
fig.savefig(root/'line-coverage.png',dpi=170)
fig.savefig(root/'line-coverage.pdf')
