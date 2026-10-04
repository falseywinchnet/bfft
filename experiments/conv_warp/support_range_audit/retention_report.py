"""Report the isolated representation experiment without changing the demo."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('output/support_geometry/conv_blend_retention')


def run():
    packet=json.loads((ROOT/'results.json').read_text());images=np.load(ROOT/'images.npz');summary=[]
    for n in (17,33):
        rows=[r for r in packet['rows'] if r['sourceSide']==n]
        ratios=[r['continuous']['retainedMse']/r['continuous']['quinticMse'] for r in rows]
        area=[r['retainedAreaTruth']['mse']/r['quinticAreaTruth']['mse'] for r in rows]
        summary.append(dict(sourceSide=n,cases=len(rows),continuousMseChangePercent=100*(math.exp(statistics.mean(map(math.log,ratios)))-1),
            areaMseChangePercent=100*(math.exp(statistics.mean(map(math.log,area)))-1),
            bestContinuousPercent=100*(min(ratios)-1),worstContinuousPercent=100*(max(ratios)-1),
            bestAreaPercent=100*(min(area)-1),worstAreaPercent=100*(max(area)-1),
            maxSampledCorrection8BitLevels=255*max(r['continuous']['maxSampledCorrection'] for r in rows),
            maxCorrectionRms8BitLevels=255*max(math.sqrt(r['continuous']['correctionMse']) for r in rows),
            continuousRatioRefinement=max(abs(r['continuous']['retainedMse']/r['continuous']['quinticMse']-r['continuousRefinement16']['retainedMse']/r['continuousRefinement16']['quinticMse']) for r in rows)))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(4,3,figsize=(10,12),layout='constrained')
    for row,kind in enumerate(('edge','carrier','curved','crossing')):
        for col,m in enumerate(('quintic','retained','correction')):
            ax=axes[row,col]
            if m=='correction':im=ax.imshow(255*images[kind+'/'+m],cmap='RdBu_r',vmin=-.25,vmax=.25,interpolation='nearest')
            else:ax.imshow(images[kind+'/'+m],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(('Quintic blend refit','Exact degree-six blend','Retained minus quintic')[col])
            if col==0:ax.set_ylabel(kind.capitalize())
    fig.colorbar(im,ax=axes[:,2],shrink=.65,label='Value change in 8-bit levels (one level = 1/255)')
    fig.suptitle('Isolated blend retention · 17 × 17 sources · 513 × 513 point views\nCommon [0,1] image range; difference shown on a separate magnified scale',fontsize=13)
    fig.savefig(ROOT/'comparison.png',dpi=140);fig.savefig(ROOT/'comparison.pdf');plt.close(fig)
    table=''.join(f"<tr><td>{r['sourceSide']} × {r['sourceSide']}</td><td>{r['continuousMseChangePercent']:+.6f}%</td><td>{r['areaMseChangePercent']:+.6f}%</td><td>{r['maxSampledCorrection8BitLevels']:.4f}</td></tr>" for r in summary)
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>CONV blend retention</title><style>body{font:16px system-ui;line-height:1.6;max-width:1060px;margin:35px auto;padding:0 20px;background:#161a20;color:#e1e6ee}h1{font-size:28px}td,th{text-align:left;padding:12px;border-bottom:1px solid #48515d}table{width:100%;border-collapse:collapse}img{width:100%;height:auto;background:white}a{color:#9cdae8}</style>
<h1>Retaining the degree-six blend</h1><p>The same two factor representations are either blended and refitted into a quintic, or blended exactly into degree six. Both omit support clipping and joint-current admission. No sharpening or new local source analysis is introduced.</p><p>The 56 cases are the paper’s full declared smooth family at source sides 17 and 33. Continuous-field MSE is integrated against the analytic field with 32-point Gauss quadrature per source-cell axis, checked at order 16. Pixel-area MSE uses 65 × 65 outputs with matching analytic area references. Negative changes mean improvement.</p>
<table><tr><th>Source</th><th>Continuous MSE change</th><th>Area MSE change</th><th>Largest sampled change, 8-bit levels</th></tr>TABLE</table>
<p>Aggregate changes are geometric means of paired error ratios. The isolated RGBA point kernel took approximately 35% longer on the M4 Mini. This is a CPU kernel comparison, not a browser FPS or full-warp timing.</p><p>The first two image columns share the [0,1] grayscale range. The third shows signed differences on its own magnified scale. It does not depict full-strength visible artifacts.</p><img src="comparison.png" alt="Four paper fields, quintic and retained blends, and magnified signed changes"><p><a href="results.json">All cases and source hashes</a> · <a href="summary.json">Summary</a> · <a href="cost.json">Kernel timings</a> · <a href="comparison.pdf">PDF plate</a></p></html>'''.replace('TABLE',table)
    (ROOT/'index.html').write_text(html)
    served=Path('output/support_geometry/conv_bounded_sharpening/paper-images/retention');served.mkdir(exist_ok=True)
    for name in ('index.html','comparison.png','comparison.pdf','results.json','summary.json','cost.json'):
        shutil.copyfile(ROOT/name,served/name)


if __name__=='__main__':run()
