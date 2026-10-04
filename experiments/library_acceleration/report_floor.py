"""Render the measured floor/gap distinction; no extrapolated speed claims."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).parent/'results'
    d=json.loads((root/'chambolle_camera_floor_v2.json').read_text())
    rows=d['trace'];x=np.array([r['iteration'] for r in rows])
    scale=d['initial_objective']-d['floor_lower']
    gap=np.array([r['gap'] for r in rows])/scale
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    fig.suptitle('Chambolle on a noisy 512 × 512 camera image',fontsize=15,fontweight='bold')
    ax=axes[0];ax.loglog(x,gap,color='#2368a2',lw=2)
    ax.axvline(d['default_iterations'],color='#bd5a21',ls='--',label='Default stops: 7 iterations')
    ax.axhline(1e-5,color='#567a42',ls=':',label='Tight gap target: 5,331 iterations')
    ax.scatter([5331],[gap[5330]],color='#567a42',zorder=3)
    ax.set(xlabel='Iterations',ylabel='Primal–dual gap / available objective decrease',title='The certified tail remains long')
    ax.legend(fontsize=8,loc='lower left');ax.grid(alpha=.2,which='both')
    rr=d['image_distance_trace'];xx=[r['iteration'] for r in rr];yy=[r['relative_distance'] for r in rr]
    ax=axes[1];ax.loglog(xx,yy,color='#6d5293',lw=2)
    ax.axhline(.2,color='#888',ls=':',lw=1);ax.axvline(7,color='#bd5a21',ls='--')
    for count,label in [(7,'80% closer: 7'),(19,'90% closer: 19'),(47,'95% closer: 47')]:
        ax.scatter([count],[yy[count-1]],color='#6d5293',zorder=3)
        ax.annotate(label,(count,yy[count-1]),xytext=(6,9),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Iterations',ylabel='Distance to reference / initial distance',title='Reconstruction accuracy is a different measure')
    ax.grid(alpha=.2,which='both')
    fig.savefig(root/'chambolle_floor.png',dpi=170)
    plt.close(fig)

if __name__=='__main__':main()
