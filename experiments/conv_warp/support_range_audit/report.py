import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('output/support_geometry/conv_support_range_audit')

def run():
 d=json.loads((OUT/'results.json').read_text());low=json.loads((OUT/'low-contrast/results.json').read_text())
 summaries=[]
 for family,n,rows in [('smooth',17,d['rows']),('smooth',33,d['rows']),('sharp',17,d['rows']),('sharp',33,d['rows']),('low-contrast',0,low['rows'])]:
  rows=[r for r in rows if family=='low-contrast' or r['family']==family and r['sourceSide']==n];base={r['case']:r for r in rows if r['method']=='nominal'}
  for m in d['methods']:
   rs=[r for r in rows if r['method']==m];ratio=np.array([r['mse']/base[r['case']]['mse'] for r in rs]);summaries.append({'family':family,'side':n,'method':m,'cases':len(rs),'geometricMseRatio':float(np.exp(np.log(ratio).mean())),'improved':int(np.sum(ratio<1-1e-6)),'worse':int(np.sum(ratio>1+1e-6)),'maxPhysicalExcursion':max(r['physicalExcursion'] for r in rs)})
 (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
 imgs=np.load(OUT/'images.npz');lim=np.load(OUT/'low-contrast/images.npz')
 cases=sorted({k.split('/')[0] for k in imgs.files});carrier=next(k for k in cases if k.startswith('carrier'));curve=next(k for k in cases if k.startswith('curved'));bar=next(k for k in cases if k.startswith('thin'))
 small=next(k.split('/')[0] for k in lim.files if k.startswith('thin') and '0.39999' in k)
 chosen=[(imgs,carrier,'Carrier',(0,1)),(imgs,curve,'Curved edge',(0,1)),(imgs,bar,'Binary thin bar',(0,1)),(lim,small,'Low-contrast bar',(0.2,.8))]
 methods=['truth','raw','nominal','current-only','physical-box','refined-support'];labels=['Analytic truth','Raw atlas proposal','Nominal joint reference','Current constraints only','Physical [0,1] box','Refined sample range']
 fig,axes=plt.subplots(4,6,figsize=(15,10.7),layout='constrained')
 for y,(image,case,title,bounds) in enumerate(chosen):
  for x,m in enumerate(methods):
   ax=axes[y,x];ax.imshow(image[case+'/'+m],cmap='gray',vmin=bounds[0],vmax=bounds[1],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   if y==0:ax.set_title(labels[x],fontsize=9)
   if x==0:ax.set_ylabel(title+'\n'+str(bounds),fontsize=10)
 fig.suptitle('Support-range ablation · no experimental sharpening',fontsize=16);fig.savefig(OUT/'comparison.png',dpi=150);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
 html='''<!doctype html><meta charset="utf-8"><title>CONV support-range audit</title><style>body{max-width:1300px;margin:30px auto;padding:0 22px;font:16px/1.6 system-ui;background:#171b20;color:#e4e9ee}img{width:100%}a{color:#8bdade}table{border-collapse:collapse}td,th{padding:8px 18px;border-bottom:1px solid #555;text-align:left}</style><h1>Support-range clipping audit</h1><p>92 declared cases, six construction stages or variants. The smooth cases use the paper's analytic families at source sizes 17 and 33. Sharp cases include binary and low-contrast steps and bars. No experimental concentration is applied. These comparisons execute the Float64 joint reference.</p><p>The original CONV paper permits an existing extremum to move between samples. The perspective warp adds a stricter sample-range clipping step. The local kernel with that step is byte-identical to the deployed WASM checked in this audit.</p><img src="comparison.png" alt="Same fields under six range policies"><p>Each row has a common display scale, printed at left. Numerical measurements use unclipped floating-point arrays. The last row uses [0.2,0.8] to show low-contrast halos.</p><h2>Smooth-field MSE relative to the nominal rule</h2><table><tr><th>Variant</th><th>17 × 17</th><th>33 × 33</th></tr>'''
 for m in ['raw','current-only','physical-box','refined-support']:
  v=[next(r['geometricMseRatio'] for r in summaries if r['family']=='smooth' and r['side']==n and r['method']==m) for n in (17,33)]
  html+=f'<tr><td>{m}</td><td>{v[0]:.3f}×</td><td>{v[1]:.3f}×</td></tr>'
 html+='''</table><p>Smaller is better. Raw proposals omit current admission and are diagnostic. Removing sample-range clipping while keeping current constraints improves aggregate smooth fidelity, but sharp-case excursions reach 0.259 outside [0,1]. A [0,1] box suppresses those binary excursions but leaves low-contrast halos up to 25.9% of edge contrast. Refined support certificates preserve the selected bound but this finite admission rule worsens aggregate fidelity.</p><p>Conclusion: strict sample-range clipping is an additional range policy, not a general consequence of CONV's variation theorem. Removing it alone is not a validated replacement. The next construction must preserve permitted extrema while retaining the applicable current topology and physical bounds.</p><p><a href="results.json">Main results</a> · <a href="low-contrast/results.json">Low-contrast results</a> · <a href="summary.json">Summary</a> · <a href="comparison.pdf">Vector figure</a></p>'''
 (OUT/'index.html').write_text(html)
 dest=Path('output/support_geometry/conv_bounded_sharpening/paper-images/range-audit');shutil.copytree(OUT,dest,dirs_exist_ok=True)

if __name__=='__main__':run()
