"""Show certificate-stage sensitivity beside observed transport reversal."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
tr=json.loads((root/'meyer_ramp_diagnosis.json').read_text())['cases']['lambda0.02_mu20']['trajectories']
ring=json.loads((root/'meyer_ramp_ringing.json').read_text())['cases']['lambda0.02_mu20']
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for method,label,color,phase,style in [
    ('ordinary','Ordinary, best available','#333333','best','-'),
    ('refresh','Refresh, next projection','#0072B2','projected','--'),
    ('refresh','Refresh, best available','#0072B2','best','-'),
    ('aligned_refresh','Defect gate, best available','#D55E00','best','-')]:
    rows=tr[method]
    values=[min(r['projected']['primal'],r['incoming']['primal'])-max(r['projected']['dual'],r['incoming']['dual'])
            if phase=='best' else r[phase]['gap'] for r in rows]
    axes[0].semilogy([r['pass'] for r in rows],values,label=label,color=color,ls=style,lw=1.6)
axes[0].set(xlim=(32,256),ylim=(1e-6,.1),xlabel='Pass',ylabel='Feasible primal–dual gap / sample',title='Part of the apparent regression is the witness')
axes[0].axhline(.001,color='#888888',lw=.7,ls=':')
axes[0].legend(frameon=False,fontsize=8)
for method,label,color in [('ordinary','Ordinary','#333333'),('refresh','Refresh both memories','#0072B2'),('u_refresh','Refresh cartoon memory only','#D55E00')]:
    rows=ring[method]['checkpoints']
    axes[1].plot([r['pass'] for r in rows],[r['successive_increment_cosine'] for r in rows],label=label,color=color,marker='o',ms=4,lw=1.5)
axes[1].set(xlabel='Pass',ylabel='Cosine between consecutive full-state updates',ylim=(-1.08,1.08),title='The remaining reversal is real')
axes[1].axhline(0,color='#888888',lw=.7)
axes[1].legend(frameon=True,framealpha=1,edgecolor='none',fontsize=8,loc='center right')
for ax in axes:
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'ramp_refresh_diagnosis.png',dpi=170)
fig.savefig(root/'ramp_refresh_diagnosis.svg')
