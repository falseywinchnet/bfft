"""Render saved measurements; no fitting or numerical admission runs here."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('output/support_geometry/conv_joint_potential')
    old=json.loads(Path('output/support_geometry/conv_compatible_current/conv_compatible_2d.json').read_text())
    new=json.loads((root/'final/natural.json').read_text())
    validation=json.loads((root/'final/validation.json').read_text())
    palette={'CONV':'#b45309','Lanczos-3':'#7c3aed','joint potential':'#007d82'}
    summary=[]
    for r in new['rows']:
        baseline={q['method']:q for q in old['rows'] if q['case']==r['case']}
        summary.append({'case':r['case'],'converged':r['converged'],
                        'interior_ratio_conv':r['mse']/baseline['CONV']['mse'],
                        'full_ratio_conv':r['full_mse']/baseline['CONV']['full_mse'],
                        'interior_ratio_lanczos':r['mse']/baseline['Lanczos-3']['mse'],
                        'full_ratio_lanczos':r['full_mse']/baseline['Lanczos-3']['full_mse'],
                        'excursion':r['excursion'],'maximum_violation':r['maximum_violation']})
    aggregate=[]
    for family in ('wave','sigmoid','chirp'):
        for method in ('CONV','Lanczos-3'):
            a=[r for r in validation['rows'] if r['family']==family and r['method']==method]
            b=[r for r in validation['rows'] if r['family']==family and r['method']=='joint natural']
            row={'family':family,'baseline':method,'cases':len(a),
                 'wins':sum(r['mse']<s['mse'] for r,s in zip(b,a)),
                 'full_wins':sum(r['full_mse']<s['full_mse'] for r,s in zip(b,a)),
                 'all_converged':all(r['converged'] for r in b)}
            for key in ('mse','full_mse','discrete_gradient_mse'):
                row[key+'_ratio']=sum(r[key] for r in b)/sum(r[key] for r in a)
            aggregate.append(row)
    (root/'summary.json').write_text(json.dumps({'matched':summary,'validation':aggregate},indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    names=[r['case'] for r in summary];pos=np.arange(len(names))
    for ax,key,title in zip(axes,('interior_ratio_conv','full_ratio_conv'),('Interior value error','Whole-image value error')):
        ax.barh(pos,[r[key] for r in summary],color=palette['joint potential'])
        ax.axvline(1,color='#b45309',linestyle='--',label='CONV baseline')
        ax.set_yticks(pos,names);ax.invert_yaxis();ax.set_xlabel('MSE / CONV MSE — lower is better');ax.set_title(title)
        ax.set_xlim(0,1.05)
    fig.suptitle('One shared potential: tighter reconstruction with a joint range bound',fontsize=15)
    fig.savefig(root/'matched_errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for col,name in enumerate(('diagonal wave','diagonal sigmoid','chirped field')):
        data=old['images'][name];truth=np.array(data['truth']);joint=np.array(new['images'][name]['image'])
        # Same central scanline and same evaluation sites for every method.
        xs=np.linspace(0,16,49);row=24
        axes[0,col].plot(xs,truth[row],color='black',lw=2,label='Analytic truth')
        for method,z in (('CONV',np.array(data['CONV'])),('Lanczos-3',np.array(data['Lanczos-3'])),('joint potential',joint)):
            axes[0,col].plot(xs,z[row],color=palette[method],lw=1.3,label=method)
            axes[1,col].plot(xs,z[row]-truth[row],color=palette[method],lw=1.3)
        axes[0,col].scatter(xs[::3],np.array(data['source'])[8],s=12,c='black',zorder=5)
        axes[0,col].set_title(name);axes[0,col].set_ylabel('Value')
        axes[1,col].axhline(0,color='black',lw=.6);axes[1,col].set_ylabel('Reconstruction − truth')
        axes[1,col].set_xlabel('Source coordinate x; y = 8')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Shared C1 tensor-quintic potential · natural boundary proposal · depth-one certificates',fontsize=13)
    fig.savefig(root/'joint_profiles.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key,title in zip(axes,('mse_ratio','full_mse_ratio','discrete_gradient_mse_ratio'),('Interior values','Whole-image values','Interior sampled gradients')):
        for j,method in enumerate(('CONV','Lanczos-3')):
            rows=[r for r in aggregate if r['baseline']==method]
            ax.bar(np.arange(3)+.35*(j-.5),[r[key] for r in rows],width=.35,label='vs '+method,color=palette[method])
        ax.axhline(1,color='black',ls='--',lw=.8);ax.set_xticks(range(3),['6 waves','3 sigmoids','3 chirps'])
        ax.set_title(title);ax.set_ylabel('Sum of joint errors / sum of baseline errors')
    axes[0].legend(fontsize=8)
    fig.suptitle('Separate 13 × 13 source battery; settings fixed before its results',fontsize=14)
    fig.savefig(root/'validation_errors.png',dpi=180);plt.close(fig)
    print(json.dumps({'matched':summary,'validation':aggregate},indent=2))


if __name__=='__main__':
    main()
