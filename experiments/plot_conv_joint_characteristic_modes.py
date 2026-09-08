"""Plots and complete accepted/rejected summaries from saved mode probes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('output/support_geometry/conv_joint_characteristic_modes')
    matched=json.loads((root/'matched.json').read_text())
    validation=json.loads((root/'validation.json').read_text())
    summary={}
    for name,data in [('matched',matched),('validation',validation)]:
        accepted=[]
        for r in data['rows']:
            if r.get('method')!='characteristic':
                continue
            a={'case':r['case'],'chart':r['chart'],'cardinality_error':r['cardinality_error'],
               'excursion':r['excursion']}
            for method in ('CONV','Lanczos-3'):
                b=next(z for z in data['rows'] if z['case']==r['case'] and z.get('method')==method)
                for metric in ('mse','full_mse','sampled_gradient_mse'):
                    a[method+' / '+metric]=r[metric]/b[metric]
            accepted.append(a)
        summary[name]={'accepted':accepted,'rejected':[r for r in data['rows'] if not r['accepted']]}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    palette={'CONV':'#b45309','Lanczos-3':'#7c3aed','characteristic':'#007d82'}
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    q=np.linspace(0,16,49)
    for col,name in enumerate(('diagonal sigmoid','curved sigmoid','diagonal step')):
        data=matched['images'][name];truth=np.array(data['truth']);row=24
        axes[0,col].plot(q,truth[row],color='black',lw=2,label='Analytic truth (sampled)')
        for method in ('CONV','Lanczos-3','characteristic'):
            z=np.array(data[method])
            axes[0,col].plot(q,z[row],color=palette[method],lw=1.5,label=method)
            axes[1,col].plot(q,z[row]-truth[row],color=palette[method],lw=1.5)
        axes[0,col].scatter(q[::3],np.array(data['source'])[8],c='black',s=12,zorder=5)
        axes[0,col].set_title(name);axes[0,col].set_ylabel('Value')
        axes[1,col].axhline(0,color='black',lw=.6);axes[1,col].set_ylabel('Reconstruction − truth')
        axes[1,col].set_xlabel('Source coordinate x; y = 8')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Direct C2 characteristic modes · 4 of 8 matched fields admitted; others rejected',fontsize=14)
    fig.savefig(root/'profiles.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,key,title in zip(axes,('matched','validation'),('Matched: 4 admitted / 8 fields','Separate check: 12 admitted / 16 fields')):
        rows=summary[key]['accepted'];pos=np.arange(len(rows))
        for j,method in enumerate(('CONV','Lanczos-3')):
            ax.barh(pos+.34*(j-.5),[r[method+' / sampled_gradient_mse'] for r in rows],height=.34,
                    color=palette[method],label='vs '+method)
        ax.axvline(1,color='black',ls='--',lw=.7);ax.set_xscale('log');ax.set_xlim(1e-6,2)
        ax.set_yticks(pos,[r['case'].replace('radial ','circle ').replace('0.5416666666666666, 0.4583333333333333','6.5/12, 5.5/12').replace('0.7071067811865476','sqrt(1/2)') for r in rows],fontsize=8)
        ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Sampled-gradient MSE ratio — lower is better')
    axes[0].legend(fontsize=8)
    fig.suptitle('Accepted modes only; every rejected case is listed in the findings',fontsize=13)
    fig.savefig(root/'slope_errors.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
