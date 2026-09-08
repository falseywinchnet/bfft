"""Plot one improvement and one regression from the saved validation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root/'meyer_defect_validation128_cached.json').read_text())
fig, axes = plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
methods = [('ordinary','Ordinary','#333333'),('refresh','Fresh memory','#0072B2'),
           ('aligned_refresh','Defect-gated memory','#D55E00')]
for ax,key,title in zip(axes,['crossing_lambda0.05_mu40','ramp_lambda0.02_mu20'],
                       ['Crossing: improved relaxation','Ramp: a regression']):
    case = data['cases'][key]
    end = max(next((r['seconds'] for r in case[m] if r['gap']<1e-4),case[m][-1]['seconds']) for m,_,_ in methods)
    for method,label,color in methods:
        rows = case[method]
        ax.semilogy([r['seconds'] for r in rows],[max(r['gap'],1e-15) for r in rows],label=label,color=color,lw=1.7)
    ax.axhline(1e-3,color='#777777',lw=.8,ls=':')
    ax.set(xlim=(0,end*1.05),ylim=(1e-5,max(case[m][0]['gap'] for m,_,_ in methods)*1.2),
           xlabel='Accumulated update time (seconds)',ylabel='Feasible primal–dual gap / sample',title=title)
    ax.grid(axis='y',alpha=.18)
    ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False,fontsize=9)
fig.savefig(root/'defect_relaxation_comparison.png',dpi=170)
fig.savefig(root/'defect_relaxation_comparison.svg')
