"""Render existing directional-limit measurements without rerunning probes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    folder=Path(__file__).parent/'results/theory'
    rows=json.loads((folder/'directional_limit.json').read_text())['cases']
    labels=[f"s{r['seed']} / {r['eps']:g}" for r in rows]; x=np.arange(len(rows))
    selected=[next(o for o in r['observability'] if o['reference_rule_rank']==1 and o['horizon']==r['true_relaxation_horizon']) for r in rows]
    broad=[100*r['best_scalar_rms_loss'] for r in selected]
    actual=[100*r['empirical_excitation']['best_scalar_rms_loss'] for r in selected]
    affine=[100*r['affine_minimax_exact_eigenvalues']['error_over_last_observed_change'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    axes[0].bar(x-.18,broad,.36,color='#a45335',label='Isotropic perturbations')
    axes[0].bar(x+.18,actual,.36,color='#166678',label='Observed excitation')
    axes[0].axhline(1,color='#333333',ls='--',label='1% RMS target')
    axes[0].set(title='Best possible one-coordinate linear rule prediction',ylabel='Minimum relative RMS loss (%)')
    axes[0].legend(fontsize=9)
    axes[1].bar(x,affine,color='#166678')
    axes[1].axhline(10,color='#a45335',ls='--',label='10% of last eigenvalue change')
    for xx,v in zip(x,affine): axes[1].text(xx,v+2,f'{v:.1f}%',ha='center',fontsize=9)
    axes[1].set(title='Best affine rule fit to all three exact anchors',ylabel='Unavoidable max error / last change (%)',ylim=(0,150))
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=35,ha='right'); ax.set_xlabel('Original seed / regularization'); ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Six original failures · oracle information supplied · stated representation limits',fontsize=13)
    fig.savefig(folder/'directional_limit.png',dpi=180);plt.close(fig)


if __name__=='__main__': main()
