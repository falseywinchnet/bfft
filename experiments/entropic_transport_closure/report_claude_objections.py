"""Plot retained objection tests without rerunning any numerical experiment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    folder=Path(__file__).parent/'results'/'theory'
    cases=json.loads((folder/'claude_objections.json').read_text())['cases']
    duplicates=json.loads((folder/'refinement_objection.json').read_text())['cases']
    gaussian=json.loads((folder/'gaussian_closure.json').read_text())['cases']
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for seed,color in [(0,'#166678'),(1,'#a45335')]:
        rows=sorted([c for c in cases if not c['density'] and c['seed']==seed],key=lambda c:c['eps'],reverse=True)
        x=[c['eps'] for c in rows]
        counts=[c['anchors'][0]['captures']['1']['rank99'] for c in rows]
        axs[0,0].semilogx(x,[v if v is not None else 33 for v in counts],'o-',color=color,label=f'Seed {seed}, early')
        for xx,v in zip(x,counts):
            if v is None: axs[0,0].annotate('>32',(xx,33),xytext=(8,0),textcoords='offset points')
        obs=[[r for r in c['observability'] if r['output']=='rule' and r['reference_rank']==4 and r['horizon']==c['horizon']][0]['rank99'] for c in rows]
        axs[0,1].semilogx(x,obs,'o-',color=color,label=f'Seed {seed}')
    axs[0,0].axhline(1,ls='--',color='#777777',label='Both seeds, late')
    axs[0,0].set(title='Current modes for 99% reference-direction capture',xlabel='Regularization epsilon',ylabel='Required leading dimension')
    axs[0,1].axhline(4,ls='--',color='#777777',label='State dimension = 4')
    axs[0,1].set(title='Rule observability exceeds state dimension',xlabel='Regularization epsilon',ylabel='Rank retaining 99% of rule Gramian trace')
    n=[r['states'] for r in duplicates]
    axs[1,0].loglog(n,[10**r['log10_uniform_bound'] for r in duplicates],'o-',color='#a45335',label='Uniform theorem bound')
    axs[1,0].loglog(n,[r['second_jet_error_64'] for r in duplicates],'o-',color='#166678',label='Actual error over 64 steps')
    axs[1,0].set(title='Duplicating atoms leaves physical dynamics unchanged',xlabel='Represented states',ylabel='Oscillation error / bound')
    g=next(r for r in gaussian if r['case']=='isotropic' and r['t']==.001)
    h=[r['horizon'] for r in g['horizons']]
    axs[1,1].loglog(h,[r['ordinary_relative_error_to_fixed'] for r in g['horizons']],'o-',color='#a45335',label='Ordinary iterate vs fixed point')
    axs[1,1].loglog(h,[r['jump_relative_error'] for r in g['horizons']],'o-',color='#166678',label='Powered vs ordinary iterate')
    axs[1,1].set(title='Exact isotropic Gaussian: no preferred direction',xlabel='Horizon, fixed t = .001',ylabel='Relative precision error')
    for ax in axs.ravel(): ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Acquisition objections and continuous Gaussian controls',fontsize=15)
    fig.savefig(folder/'claude_objections.png',dpi=160); plt.close(fig)


if __name__=='__main__': main()
