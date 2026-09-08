"""Render saved intermediate-state measurements; no solver rerun."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'meyer_intermediate64.json').read_text())['scenes']
chart=json.loads((ROOT/'meyer_intermediate_chart64.json').read_text())['scenes']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
rows=data['crossing_lambda0.05_mu40'];x=[r['pass'] for r in rows]
ax=axs[0,0]
for key,label,color in [('memory_change_rms','Retained-field change','#007c91'),
                         ('gradient_change_rms','Primal-gradient change','#d26928')]:
    ax.loglog(x,[r['w_branch'][key] for r in rows],'-o',label=label,color=color,ms=4)
ax.set(title='A. Texture branch: memory keeps changing',xlabel='Ordinary pass',ylabel='RMS vector change')
ax.legend(frameon=False);ax.grid(alpha=.15)
ax=axs[0,1]
for b,label,color in [('u_branch','Cartoon branch','#007c91'),('w_branch','Texture branch','#d26928')]:
    ax.semilogx(x,[100*r[b]['retained_transverse_energy_fraction'] for r in rows],'-o',label=label,color=color,ms=4)
ax.set(title='B. Retained information absent from divergence',xlabel='Ordinary pass',ylabel='Divergence-free share of squared norm (%)')
ax.legend(frameon=False);ax.grid(alpha=.15)
ax=axs[1,0]
r=rows[6];branches=['u_branch','w_branch']
shares=[100*r[b]['remainder_energy_crossings'] for b in branches]
ax.bar(['Cartoon','Texture'],shares,color='#007c91',label='Sites that cross a projection boundary')
ax.bar(['Cartoon','Texture'],[100-a for a in shares],bottom=shares,color='#c5dfe4',label='Sites that remain exterior')
for i,b in enumerate(branches):
    ax.text(i,50,f"{100*r[b]['crossing_fraction']:.3f}% of sites\n{shares[i]:.1f}% of error",ha='center',va='center',color='white',fontsize=11)
ax.set(title='C. Nonlinear error is concentrated here (pass 64)',ylabel='Projection-remainder energy (%)',ylim=(0,105))
ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,-.30),fontsize=9)
ax=axs[1,1]
chosen=[chart['ramp'][1],chart['ramp'][2],chart['edge'][2],chart['carrier'][0]]
xx=np.arange(4)
ax.bar(xx-.17,[r['incoming_gap']['gap_per_pixel'] for r in chosen],.34,label='Incoming state',color='#d26928')
ax.bar(xx+.17,[max(r['candidate_gap']['gap_per_pixel'],1e-14) for r in chosen],.34,label='Dense local-pattern solve',color='#007c91')
ax.set_yscale('log');ax.set_xticks(xx,['Ramp 16\nleaves pattern','Ramp 64\nvalid closure','Edge 64\ninconsistent','Carrier 4\nvalid closure'])
ax.set(title='D. Does the current pattern contain the answer?',ylabel='Feasible gap per pixel',ylim=(1e-14,1e3))
ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15)
fig.suptitle('Inside the Meyer transport state',fontsize=17,fontweight='bold')
fig.supxlabel('A–C: 64 × 64 crossing. D: 1 × 64 signals. λ = 0.05, μ = 40. Dense solves are diagnostic; no timing claim.',fontsize=9)
fig.savefig(ROOT/'intermediate_summary.png',dpi=180)
