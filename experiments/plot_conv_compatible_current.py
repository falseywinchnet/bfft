"""Render saved M4 evidence without rerunning the numerical experiments."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder):
    folder=Path(folder)
    data=json.loads((folder/'conv_compatible_discovery_v3.json').read_text())
    x=np.array(data['x'])
    selections=[('sine f=0.4,p=0.5',(8,16),'High-frequency sine: 0.4 cycles/sample'),
                ('sigmoid w=0.7,p=0.5',(10,15),'Steep sigmoid: width 0.7'),
                ('step p=0.5',(10,15),'Step: phase 0.5')]
    methods=[('CONV','Current CONV','#1672b8'),
             ('compatible envelope','Compatible current + excursion bound','#d96c13'),
             ('Lanczos-3','Lanczos-3','#32834b')]
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for row,(key,limits,title) in enumerate(selections):
        p=data['plots'][key];truth=np.array(p['truth'])
        for method,label,color in methods:
            z=np.array(p['estimates'][method])
            axes[row,0].plot(x,z,color=color,label=label,lw=1.7)
            axes[row,1].plot(x,z-truth,color=color,lw=1.5)
        axes[row,0].plot(x,truth,'k--',lw=1,label='Analytic truth')
        axes[row,0].scatter(np.arange(data['n']),p['source'],s=13,color='k',zorder=5)
        for col in range(2):
            axes[row,col].set_xlim(*limits);axes[row,col].grid(alpha=.2)
            axes[row,col].set_title(title+('' if col==0 else ' — error'))
            axes[row,col].set_xlabel('Source coordinate')
        mask=(x>=limits[0])&(x<=limits[1])
        error=np.concatenate([(np.array(p['estimates'][m])-truth)[mask] for m,_,_ in methods])
        span=max(np.ptp(error),1e-6)
        axes[row,1].set_ylim(error.min()-.08*span,error.max()+.08*span)
    axes[0,0].legend(loc='upper left',fontsize=8)
    fig.tight_layout();fig.savefig(folder/'compatible_profiles.png',dpi=160)
    fig,ax=plt.subplots(figsize=(10,4.5))
    validation=json.loads((folder/'conv_compatible_validation_v3.json').read_text())
    families=['sigmoid','sine','chirp','mixed','step','box']
    indices=np.arange(len(families))
    for j,(m,label,color) in enumerate(methods):
        ratio=[validation['summary'][f][m]['mean_nmse']/validation['summary'][f]['CONV']['mean_nmse'] for f in families]
        ax.bar(indices+(j-1)*.24,ratio,width=.24,color=color,label=label)
    ax.set_xticks(indices, families);ax.set_ylabel('Mean value NMSE / current CONV')
    ax.set_title('Expanded validation: 68 cases, 33 source nodes, 32× refinement')
    ax.axhline(1,color='black',lw=.7);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(folder/'validation_ratios.png',dpi=160)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder')
    plot(p.parse_args().folder)
