"""Render saved measurements only; no recomputation of benchmark claims."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'meyer_fused_validate256.json').read_text())
names={'barbara_512':'Barbara (resized to 256²)','camera_256':'Cameraman 256²',
       'synthetic_256':'Analytic source 256²','crossing':'Multiscale crossing 256²'}
fig,axes=plt.subplots(2,4,figsize=(15,7.4),layout='constrained')
for col,(name,s) in enumerate(d['scenes'].items()):
    r=s['methods']
    for row,(metric,title) in enumerate([('error4096','Relative texture error to pass 4096'),('gap_per_pixel','Certified objective gap / pixel')]):
        ax=axes[row,col]
        for prefix,keys,color,label in [
            ('ordinary',['ordinary32','ordinary40','ordinary48','ordinary64'],'#727780','Ordinary coupled transport'),
            ('relax',['relax1.75_24','relax1.75_32','relax1.75_48','relax1.75_64'],'#17805b','Current-state step ×1.75')]:
            ax.plot([r[k]['median_ms'] for k in keys],[r[k][metric] for k in keys],'-o',color=color,label=label,ms=4)
            for k in keys:
                labelnum=k.split('_')[-1] if prefix=='relax' else k[8:]
                ax.annotate(labelnum,(r[k]['median_ms'],r[k][metric]),xytext=(3,5),textcoords='offset points',fontsize=7,color=color)
        for k,marker,label in [('fast','v','Published fast'),('quality','D','Published quality')]:
            ax.scatter(r[k]['median_ms'],r[k][metric],marker=marker,color='#315cc2',s=36,label=label,zorder=5)
        ax.set_xlabel('Median elapsed time (ms)');ax.grid(alpha=.2)
        if row==0:ax.set_title(names[name],fontsize=11)
        if col==0:ax.set_ylabel(title)
        if row==1:ax.set_yscale('log')
axes[0,0].legend(fontsize=7,loc='upper right')
fig.suptitle('Recompute the coupled transport, then advance it farther\nNative M4 CPU · 1 thread · 15 shuffled measurements · other CPU jobs active',fontsize=15)
fig.savefig(HERE/'frontier.png',dpi=160);plt.close(fig)

data=np.load(HERE/'meyer_fused_validate256.npz')
fig,axes=plt.subplots(4,4,figsize=(11,10),layout='constrained')
for row,name in enumerate(d['scenes']):
    f=data[name+'_source'];q=data[name+'_quality'];r=data[name+'_relax1.75_32'];target=data[name+'_reference4096']
    axes[row,0].imshow(f-q,cmap='gray',vmin=0,vmax=255)
    axes[row,1].imshow(f-r,cmap='gray',vmin=0,vmax=255)
    limit=max(np.quantile(abs(q-target),.995),np.quantile(abs(r-target),.995),1.)
    axes[row,2].imshow(q-target,cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[row,3].imshow(r-target,cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[row,0].set_ylabel(names[name],fontsize=9)
    for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    if row==0:
        for ax,title in zip(axes[row],['Quality cartoon','Step ×1.75 / 32 cartoon','Quality minus pass 4096','Step ×1.75 / 32 minus pass 4096']):ax.set_title(title,fontsize=9)
fig.suptitle('Same source, parameters, and transport state\nError colors share a scale within each row; pass 4096 is a trajectory reference, not ground truth',fontsize=12)
fig.savefig(HERE/'comparison.png',dpi=160);plt.close(fig)
