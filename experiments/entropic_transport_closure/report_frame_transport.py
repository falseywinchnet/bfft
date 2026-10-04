import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    base=Path(__file__).parent/'results/theory'
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
    for ax,name,title in zip(axes,['frame_direction_fan','frame_direction_fan_matched'],['Fixed RMS amplitude','Matched log-range = 6']):
        rows=json.loads((base/(name+'.json')).read_text())
        for g,color,label in [(0.,'#2b6cb0','Binary fields'),(.5,'#d97706','50% Gaussian variance'),(1.,'#b83280','Gaussian fields')]:
            ds=[1,2,4,8,16]
            means=[np.mean([r['products']/256 for r in rows if r['directions']==d and r['gaussian_fraction']==g]) for d in ds]
            ax.plot(ds,means,'o-',color=color,label=label)
        ax.set_xscale('log',base=2);ax.set_xticks(ds,ds);ax.set_ylim(0,1.04)
        ax.set_xlabel('Prescribed varying modes');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].set_ylabel('Fraction of kernel products still required')
    axes[1].legend(loc='lower right',fontsize=9)
    fig.suptitle('Primitive response reuse loses coverage as the request family grows',fontsize=13)
    fig.text(.5,.01,'Controlled positive-query probes, not Sinkhorn trajectories. 128 coordinates, 256 requests, rank budget 32; means of two seeds.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(base/'frame_direction_fan.png',dpi=160)
if __name__=='__main__':main()
