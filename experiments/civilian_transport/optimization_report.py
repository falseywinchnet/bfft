"""Render the retained exact-elimination measurements; no experiment rerun."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

def main():
    out=Path(__file__).parent/'optimization_results'
    data=json.loads((out/'results.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.6),gridspec_kw={'width_ratios':[1,1.2]})
    colors=['#a65043','#bd9255','#3b8783','#244b85']
    names=['Dense Zak','Dense real','Four-state NumPy','Four-state native']
    for backend,label,color in zip(('dense_zak','dense_current','markov_python','markov_native'),names,colors):
        rows=[r for r in data['micro'] if r['backend']==backend]
        axes[0].plot([r['cells'] for r in rows],[r['median_us'][0] for r in rows],marker='o',label=label,color=color,lw=2)
    axes[0].set(xscale='log',yscale='log',xlabel='Represented current cells',ylabel='Update time per observation (µs)',title='Same posterior, much less update work')
    axes[0].xaxis.set_minor_locator(NullLocator());axes[0].set_xticks([32,128,512],['32','128','512']);axes[0].grid(axis='y',alpha=.2);axes[0].legend(frameon=False,fontsize=9)
    policies=['constant_2','adaptive_32','constant_32'];labels=['Constant 2 Hz','Adaptive 2→32 Hz','Constant 32 Hz']
    for j,backend in enumerate(('dense','native')):
        bottom=np.zeros(3);x=np.arange(3)+(-.19 if j==0 else .19)
        for key,label,color in [('initialization_ms','Initialize','#d6dce3'),('update_ms','Assimilate','#244b85'),('readout_ms','Read out','#74a9b8'),('scheduler_ms','Schedule','#d39243')]:
            vals=[np.mean([r[key] for r in data['runs'] if r['backend']==backend and r['policy']==p and r['sigma']==.35]) for p in policies]
            axes[1].bar(x,vals,bottom=bottom,width=.34,color=color,label=label if j==0 else None)
            bottom+=vals
        for xx,v in zip(x,bottom):axes[1].text(xx,v+3,f'{v:.1f}',ha='center',fontsize=9)
    axes[1].set(xticks=range(3),xticklabels=labels,ylabel='Total online CPU time over 12 s (ms)',title='Complete acquisition loop, unchanged decisions',ylim=(0,245))
    axes[1].legend(frameon=False,ncol=2,loc='upper left',fontsize=9);axes[1].grid(axis='y',alpha=.15)
    axes[1].text(.5,-.19,'Each pair: original (left), optimized (right)',transform=axes[1].transAxes,ha='center',fontsize=9)
    fig.suptitle('Exact elimination of the relational current history',fontsize=16,x=.06,ha='left',y=1.01)
    fig.text(.06,-.03,'M4 Mini • float64 • 5 retained length branches • matched 480-case replay • sensor acquisition cost excluded',fontsize=9,color='#535b63')
    fig.tight_layout();fig.savefig(out/'exact_optimization.png',dpi=180,bbox_inches='tight');fig.savefig(out/'exact_optimization.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
