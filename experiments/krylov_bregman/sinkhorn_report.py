"""Summarize retained seven-repeat timings and draw the transfer figure."""
import json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'experiments/krylov_bregman/results'
OUT=ROOT/'output/support_geometry/krylov_bregman/sinkhorn'

def main():
    rows=json.loads((RESULTS/'sinkhorn_full.json').read_text())['runs']
    keys=sorted(set((r['n'],r['seed'],r['kind'],r['epsilon']) for r in rows))
    cases=[];methods=['ordinary','fixed','discovered','anderson']
    for key in keys:
        subset=[r for r in rows if (r['n'],r['seed'],r['kind'],r['epsilon'])==key]
        times={m:statistics.median(r['seconds'] for r in subset if r['method']==m) for m in methods}
        cases.append(dict(n=key[0],seed=key[1],kind=key[2],epsilon=key[3],median_seconds=times,
            speedups={m:times['ordinary']/times[m] for m in methods},
            work={m:next(r['work'] for r in subset if r['method']==m) for m in methods},
            statuses={m:sorted(set(r['status'] for r in subset if r['method']==m)) for m in methods}))
    summary={'cases':cases,'total_runs':len(rows),'successful_runs':sum(r['status']=='target' for r in rows),
        'max_column_l1':max(r['column_l1'] for r in rows),'max_row_l1':max(r['row_l1'] for r in rows),
        'max_plan_l1_to_reference':max(r['plan_l1_to_reference'] for r in rows),
        'max_objective_difference_to_reference':max(abs(r['objective']-r['reference_objective']) for r in rows),
        'speedups':{m:dict(minimum=min(c['speedups'][m] for c in cases),
              median=statistics.median(c['speedups'][m] for c in cases),
              maximum=max(c['speedups'][m] for c in cases),
              faster_cases=sum(c['speedups'][m]>1 for c in cases)) for m in methods[1:]},
        'fixed_faster_than_anderson_cases':sum(c['median_seconds']['fixed']<c['median_seconds']['anderson'] for c in cases)}
    (RESULTS/'sinkhorn_summary.json').write_text(json.dumps(summary,indent=2))
    OUT.mkdir(parents=True,exist_ok=True)
    colors={'fixed':'#1565c0','discovered':'#b85c00','anderson':'#267342'}
    names={'fixed':'Polynomial transport (fixed 4 / 16)','discovered':'Polynomial transport (discovery)','anderson':'Anderson (depth 4)'}
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.8),sharey=True)
    for ax,n in zip(axs,[128,512]):
        cc=[c for c in cases if c['n']==n]
        for i,m in enumerate(methods[1:]):
            ax.plot(np.arange(len(cc))+(i-1)*.12,[c['speedups'][m] for c in cc],
                    'o',color=colors[m],label=names[m],markersize=6)
        ax.axhline(1,color='#666666',lw=1,ls='--');ax.set_yscale('log');ax.set_ylim(.55,12)
        ax.set_xticks(range(len(cc)),[f"{'Img' if c['kind']=='image' else 'Mix'}\n{c['epsilon']:g}\ns{c['seed']}" for c in cc],fontsize=8)
        ax.set_title(f'{n} source and {n} target points')
        ax.grid(axis='y',alpha=.15);ax.set_xlabel('Distribution / regularization / seed')
    axs[0].set_ylabel('Speedup over ordinary Sinkhorn (higher is faster)')
    axs[0].set_yticks([.5,1,2,4,8],['0.5x','1x','2x','4x','8x'])
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.94),ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.075,right=.99,bottom=.20,top=.80,wspace=.05)
    fig.suptitle('Sinkhorn transfer: seven-repeat median times at identical accuracy',y=.99,fontsize=14)
    fig.savefig(OUT/'speedups.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/'speedups.svg',bbox_inches='tight');plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
