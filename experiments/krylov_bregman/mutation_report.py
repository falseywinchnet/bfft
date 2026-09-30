"""Retained mutation evidence: model fidelity versus complete execution cost."""
import json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'experiments/krylov_bregman/results'
OUT=ROOT/'output/support_geometry/krylov_bregman/mutation'

def main():
    factor=json.loads((R/'krylov_factor_full.json').read_text())
    increment=json.loads((R/'krylov_increment_full.json').read_text())
    summary={}
    for name,data,fields in [('factor',factor,['scene']),('increment',increment,['kind','scene'])]:
        cases=[]
        for key in sorted(set(tuple(r[f] for f in fields) for r in data['runs'])):
            rows=[r for r in data['runs'] if tuple(r[f] for f in fields)==key]
            methods=sorted(set(r['method'] for r in rows))
            times={m:statistics.median(r['seconds'] for r in rows if r['method']==m) for m in methods}
            cases.append(dict(zip(fields,key),median_seconds=times,
                speedups={m:times['ordinary']/t for m,t in times.items()},
                accepted={m:next((r.get('accepted') for r in rows if r['method']==m),None) for m in methods}))
        summary[name]=cases
    anchors=factor['anchors']
    summary['anchor_improvement']={'count':len(anchors),
        'factor_over_linear':sum(a['factor_error']<a['linear_error'] for a in anchors),
        'enriched_over_linear':sum(a['enriched_error']<a['linear_error'] for a in anchors),
        'enriched_over_factor':sum(a['enriched_error']<a['factor_error'] for a in anchors)}
    summary['timed_runs']=len(factor['runs'])+len(increment['runs'])
    summary['successful_runs']=sum(r['status']=='target' for r in factor['runs']+increment['runs'])
    (R/'mutation_summary.json').write_text(json.dumps(summary,indent=2))
    OUT.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    valid=[a for a in anchors if a['displacement']>1e-10 and a['factor_error']>1e-12 and a['enriched_error']>1e-12]
    x=np.array([a['factor_error']/a['displacement'] for a in valid]);y=np.array([a['enriched_error']/a['displacement'] for a in valid])
    axs[0].loglog(x,y,'o',color='#176e99');lo=min(x.min(),y.min())*.7;hi=max(x.max(),y.max())*1.3
    axs[0].plot([lo,hi],[lo,hi],'--',color='#888888');axs[0].set_xlim(lo,hi);axs[0].set_ylim(lo,hi)
    axs[0].set_xlabel('Nonlinear reduced model: relative trajectory error')
    axs[0].set_ylabel('After defect enrichment: relative trajectory error')
    axs[0].set_title('Enrichment improves prediction\nBelow the diagonal is better')
    cases=summary['factor'];xx=np.arange(len(cases));methods=['linear16','factor16','factor64','enriched64']
    for j,(m,col) in enumerate(zip(methods,['#718096','#d09531','#bd6446','#176e99'])):
        axs[1].bar(xx+(j-1.5)*.19,[1/c['speedups'][m] for c in cases],width=.18,color=col,label=m)
    axs[1].axhline(1,color='#555555',ls='--');axs[1].set_xticks(xx,[c['scene'].replace('camera_permuted','Permuted').replace('straight_carrier','Carrier').title() for c in cases],rotation=20,ha='right')
    axs[1].set_ylabel('Solve time / ordinary Meyer (lower is better)');axs[1].set_title('None of these mutations amortizes yet\nThree-repeat median complete solve times')
    axs[1].legend(frameon=False,fontsize=8,ncol=2,loc='upper left')
    for ax in axs:ax.grid(axis='y',alpha=.15)
    fig.subplots_adjust(left=.075,right=.985,bottom=.2,top=.84,wspace=.34)
    fig.suptitle('Meyer mutation: improved local fidelity does not yet produce faster solves',y=.98)
    fig.savefig(OUT/'fidelity_and_cost.png',dpi=170,bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
