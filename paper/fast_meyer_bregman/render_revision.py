"""September paper figures and clean table, rendered only from saved records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'})

def figures():
    a=np.load(ROOT/'experiments/out/meyer_hard_jump_defect_proof/proof_arrays.npz')
    names=['pure_edge','pure_carrier','symmetric_support']
    fig,axes=plt.subplots(3,5,figsize=(10,6.15),layout='constrained')
    titles=['Source','Carrier term','Structural halo','Capacity correction','Actual texture error']
    for row,name in enumerate(names):
        err=a[name+'_hard_texture']-a[name+'_fused64_texture']
        arrays=[a[name+'_source'],-a[name+'_retained_texture'],a[name+'_structural_halo'],a[name+'_capacity_correction'],err]
        vmax=max(float(np.max(abs(v))) for v in arrays[1:]) or 1.
        for col,v in enumerate(arrays):
            ax=axes[row,col]
            ax.imshow(v,cmap='gray' if col==0 else 'RdBu_r',vmin=0 if col==0 else -vmax,vmax=255 if col==0 else vmax)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=10)
        axes[row,0].set_ylabel(name.replace('_',' '),fontsize=10)
    fig.savefig(HERE/'results/revision_defect.png',dpi=200);plt.close(fig)
    a=np.load(HERE/'results/flow_natural_m4/arrays.npz')
    fig,axes=plt.subplots(3,4,figsize=(9,6.7),layout='constrained')
    titles=['Fused 64 cartoon','Finite-flow quality cartoon','Scalar shortcut texture','Finite-flow quality texture']
    for row,(name,label) in enumerate([('barbara_512','Barbara 512²'),('camera_256','Cameraman 256²'),('synthetic_256','Analytic 256²')]):
        keys=['fused_64_cartoon','flow_quality_cartoon','hard_jump_texture','flow_quality_texture']
        vmax=max(float(np.quantile(abs(a[name+'_'+k]),.995)) for k in keys[2:])
        for col,k in enumerate(keys):
            axes[row,col].imshow(a[name+'_'+k],cmap='gray' if col<2 else 'RdBu_r',vmin=0 if col<2 else -vmax,vmax=255 if col<2 else vmax)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(titles[col].replace(' cartoon','\ncartoon').replace(' texture','\ntexture'),fontsize=9)
        axes[row,0].set_ylabel(label,fontsize=10)
    fig.savefig(HERE/'results/revision_natural.png',dpi=200);plt.close(fig)

def table():
    r=json.loads((ROOT/'experiments/meyer_transport_audit/meyer_clean256.json').read_text())
    names=['Barbara','Cameraman','Analytic','Crossing']
    keys=['ordinary32','ordinary40','ordinary48','ordinary64','fast','quality','relax1.75_32']
    labels=['Ordinary 32','Ordinary 40','Ordinary 48','Ordinary 64','Finite flow fast','Finite flow quality','Current state 1.75 / 32']
    s=r'\begin{table*}[t]\centering\scriptsize'+'\n'+r'\caption{Clean M4 rerun: eight threads, $256^2$ sources, 15 shuffled timed repetitions after two warmup rounds. Each source reports median ms, $E_{64}$, and recovered objective gap per pixel $G=(P-L)/(HW)$. Barbara is resized to $256^2$.}\label{tab:clean}'+'\n'+r'\begin{tabular}{lrrrrrrrrrrrr}\toprule'+'\n'
    s+='&'+'&'.join(r'\multicolumn{3}{c}{'+n+'}' for n in names)+r'\\'+'\n'
    s+='Method &'+'&'.join(['ms & $E_{64}$ & $G$']*4)+r'\\\midrule'+'\n'
    for k,l in zip(keys,labels):
        vals=[]
        for sc in r['scenes'].values():
            v=sc['methods'][k];vals.extend([f"{v['median_ms']:.2f}",f"{v['error64']:.4f}",f"{v['gap_per_pixel']:.4f}"])
        s+=l+'&'+'&'.join(vals)+r'\\'+'\n'
    s+=r'\bottomrule\end{tabular}\end{table*}'+'\n'
    (HERE/'results/september_clean_table.tex').write_text(s)

if __name__=='__main__':figures();table()
