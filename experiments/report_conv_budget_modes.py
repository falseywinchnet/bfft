"""Save matched visual data on compute host; render saved data locally."""
import argparse
from pathlib import Path
import numpy as np


def prepare(out):
    from experiments.probe_conv_budget_modes import cases
    from experiments.conv_budget_modes import resize
    from experiments.convstar import convstar_resize
    arrays={}
    for name,source,truth in list(cases(17))[:4]:
        key=name.replace(' ','_');arrays[key+'_source']=source;arrays[key+'_truth']=truth
        arrays[key+'_conv']=convstar_resize(source,3);arrays[key+'_budget']=resize(source,3)
    np.savez_compressed(out,**arrays)


def plot(path,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=np.load(path);names=('diagonal_sigmoid','curved_sigmoid','diagonal_step','disk')
    fig,axes=plt.subplots(2,4,figsize=(14,6.3),constrained_layout=True)
    for col,name in enumerate(names):
        truth=data[name+'_truth'];conv=data[name+'_conv'];budget=data[name+'_budget']
        q=np.linspace(0,1,truth.shape[0]);j=len(q)//2
        ax=axes[0,col]
        ax.plot(q,truth[j],color='#172431',lw=2.2,label='Truth')
        ax.plot(q,conv[j],color='#c46534',lw=1.5,label='CONV')
        ax.plot(q,budget[j],color='#167f83',lw=1.6,ls='--',label='Budgeted joint mode')
        source=data[name+'_source'];ax.scatter(np.linspace(0,1,17),source[8],s=12,c='#172431',zorder=4,label='Source nodes')
        ax.set_title(name.replace('_',' ').capitalize());ax.set_xlim(.12,.88);ax.set_ylim(min(-.03,float(conv[j].min())-.03),max(1.03,float(conv[j].max())+.03))
        ax.set_xlabel('x at y = 0.5');ax.grid(alpha=.14)
        ax=axes[1,col]
        ax.plot(q,conv[j]-truth[j],color='#c46534',lw=1.5,label='CONV error')
        ax.plot(q,budget[j]-truth[j],color='#167f83',lw=1.5,label='Joint-mode error')
        ax.axhline(0,color='#85919b',lw=.7);ax.set_xlim(.12,.88);ax.grid(alpha=.14);ax.set_xlabel('x at y = 0.5')
    axes[0,0].set_ylabel('Value');axes[1,0].set_ylabel('Signed value error')
    axes[0,0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Shared phase, slope and curvature at CONV-scale cost\n17 × 17 source → 49 × 49 output; central horizontal cuts',fontsize=14)
    fig.savefig(out,dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare');p.add_argument('--plot');p.add_argument('--out')
    a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    else:plot(a.plot,a.out)
