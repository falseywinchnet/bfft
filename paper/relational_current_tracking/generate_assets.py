"""Generate manuscript tables/figures from the accompanying retained evidence."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence';T=ROOT/'tables';F=ROOT/'figures'
LABELS={'relational':'Relational current','severed':'Severed current','single_length':'Single length, 1.5 s','cv':'CV Kalman','ca':'CA Kalman','imm':'IMM','robust_imm':'Robust IMM','turn_ukf':'3-D turn UKF'}
METHODS=list(LABELS)
COND=[(.35,65),(1.4,65),(.35,9)]
FAMILIES=['line','helix','figure8','stop_go','switching']
POLICIES=['constant_2','adaptive_4','adaptive_8','adaptive_16','adaptive_32','constant_32']
PNAMES=['Constant 2','2 to 4','2 to 8','2 to 16','2 to 32','Constant 32']
def load(n):return json.loads((E/(n+'.json')).read_text())
def texrows(name,rows):
    (T/(name+'.tex')).write_text('% Generated from retained evidence; do not edit numerals manually.\n'+'\n'.join(' & '.join(r)+r' \\' for r in rows)+'\n')

def main():
    for n,v in load('manifest').items():
        assert hashlib.sha256((E/n).read_bytes()).hexdigest()==v['sha256'],n
    raw=load('relational_raw')['runs'];summ=load('relational_summary')
    stats={}
    for c in COND:
        for method in METHODS:
            r=[x for x in raw if (x['sigma'],x['count'])==c and x['method']==method]
            assert len(r)==30
            stats[c,method]=dict(rmse=np.sqrt(np.mean([x['sq'] for x in r])),radius=np.mean([x['radius'] for x in r]),covered=sum(x['covered'] for x in r))
    texrows('forecast_accuracy',[[LABELS[m]]+[f"{stats[c,m]['rmse']:.3f}" for c in COND] for m in METHODS])
    texrows('forecast_coverage',[[LABELS[m]]+[f"{stats[c,m]['covered']}/30" for c in COND] for m in METHODS])
    # Use archived paired intervals where already published; derive remaining
    # comparator intervals by the same paired, within-family resampling rule.
    pairs={(x['sigma'],x['count'],x['other']):x for x in summ['paired_ratios']}
    rng=np.random.default_rng(880042)
    for c in COND:
        a=sorted([r for r in raw if (r['sigma'],r['count'])==c and r['method']=='relational'],key=lambda r:(r['family'],r['seed']))
        av=np.array([r['sq'] for r in a])
        indices=np.concatenate([rng.choice([i for i,r in enumerate(a) if r['family']==f],size=(4000,6)) for f in FAMILIES],axis=1)
        for m in ('cv','ca','imm','robust_imm','turn_ukf'):
            if (*c,m) in pairs:continue
            b=sorted([r for r in raw if (r['sigma'],r['count'])==c and r['method']==m],key=lambda r:(r['family'],r['seed']))
            assert [(r['family'],r['seed']) for r in a]==[(r['family'],r['seed']) for r in b]
            bv=np.array([r['sq'] for r in b]);boot=np.sqrt(av[indices].mean(axis=1)/bv[indices].mean(axis=1))
            pairs[c+(m,)]=dict(ratio=float(np.sqrt(av.mean()/bv.mean())),ci=np.quantile(boot,[.025,.975]).tolist())
    texrows('paired_intervals',[[f'C{i+1}',LABELS[m],f"{pairs[c+(m,)]['ratio']:.3f}",f"[{pairs[c+(m,)]['ci'][0]:.3f}, {pairs[c+(m,)]['ci'][1]:.3f}]"] for i,c in enumerate(COND) for m in ('cv','ca','imm','robust_imm','turn_ukf')])
    growth=[]
    rng_growth=np.random.default_rng(880043)
    for m in ('relational','cv','ca','imm','robust_imm','turn_ukf'):
        base=sorted([r for r in raw if (r['sigma'],r['count'])==COND[0] and r['method']==m],key=lambda r:(r['family'],r['seed']))
        indices=np.concatenate([rng_growth.choice([i for i,r in enumerate(base) if r['family']==f],size=(4000,6)) for f in FAMILIES],axis=1)
        av=np.array([r['sq'] for r in base]);row=[]
        for c in COND[1:]:
            changed=sorted([r for r in raw if (r['sigma'],r['count'])==c and r['method']==m],key=lambda r:(r['family'],r['seed']))
            assert [(r['family'],r['seed']) for r in base]==[(r['family'],r['seed']) for r in changed]
            bv=np.array([r['sq'] for r in changed]);boot=np.sqrt(bv[indices].mean(axis=1)/av[indices].mean(axis=1))
            ratio=float(np.sqrt(bv.mean()/av.mean()));ci=np.quantile(boot,[.025,.975]).tolist()
            growth.append(dict(method=m,sigma=c[0],count=c[1],ratio=ratio,ci=ci))
            row.append(f"{ratio:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    texrows('degradation',[[LABELS[m]]+[f"{g['ratio']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]" for g in growth if g['method']==m] for m in ('relational','cv','ca','imm','robust_imm','turn_ukf')])
    (E/'derived_degradation.json').write_text(json.dumps(growth,indent=2)+'\n')
    acq=load('acquisition_summary')['summaries'];opt=load('optimization_raw');cv=load('cv_cost_controls')
    texrows('acquisition',[[f'{s:g}',name,f"{r['samples']:.1f}",f"{r['post_track_rmse']:.3f}",f"{r['post_forecast_rmse']:.3f}"] for s in (.35,1.4) for p,name in zip(POLICIES,PNAMES) for r in acq if r['sigma']==s and r['policy']==p])
    texrows('settling',[[f'{s:g}',name,f"{r['first_second_track_rmse']:.3f}",f"{r['later_track_rmse']:.3f}"] for s in (.35,1.4) for p,name in zip(('constant_2','adaptive_32','constant_32'),('Constant 2','2 to 32','Constant 32')) for r in acq if r['sigma']==s and r['policy']==p])
    backends=['dense_zak','dense_current','markov_python','markov_native'];bnames=['Dense Zak','Dense real current','Four-state NumPy','Four-state compiled']
    texrows('micro',[[label]+[f'{v:.2f}' for v in r['median_us']] for b,label in zip(backends,bnames) for r in opt['micro'] if r['backend']==b and r['cells']==128])
    texrows('micro_variability',[[label,f"{r['median_us'][0]:.2f}",f"[{r['p10_us'][0]:.2f}, {r['p90_us'][0]:.2f}]"] for b,label in zip(backends,bnames) for r in opt['micro'] if r['backend']==b and r['cells']==128])
    cost={}
    for p in POLICIES:
        for b in ('dense','native'):
            rr=[r for r in opt['runs'] if r['sigma']==.35 and r['policy']==p and r['backend']==b]
            assert len(rr)==40
            cost[p,b]={k:float(np.mean([r[k] for r in rr])) for k in ('initialization_ms','update_ms','readout_ms','scheduler_ms','total_online_ms')}
    texrows('online',[[name,f"{cost[p,'dense']['total_online_ms']:.2f}",f"{cost[p,'native']['total_online_ms']:.2f}",f"{cost[p,'dense']['total_online_ms']/cost[p,'native']['total_online_ms']:.2f}"] for p,name in zip(('constant_2','adaptive_32','constant_32'),('Constant 2','2 to 32','Constant 32'))])
    hist=load('geometric_confirmation')['overall']
    texrows('historical',[[LABELS.get(m,'Geometric transport')]+[f'{hist[m][k]:.3f}' for k in ('track_rmse','forecast2_rmse','energy2','boundary_crossing_brier')] for m in ('cv','ca','imm','robust_imm','turn_ukf','geometric')])
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.75),sharey=True)
    for j,(ax,c,title) in enumerate(zip(axes,COND,['C1: 65 samples, $\\sigma=0.35$','C2: 65 samples, $\\sigma=1.4$','C3: 9 samples, $\\sigma=0.35$'])):
        for i,m in enumerate(('cv','ca','imm','robust_imm','turn_ukf')):
            r=pairs[c+(m,)];lo,hi=r['ci'];z=r['ratio']
            ax.errorbar(z,4-i,xerr=[[z-lo],[hi-z]],fmt='o',ms=3,capsize=2,color='#244b85')
        ax.axvline(1.,ls='--',lw=.8,color='#9b4c43');ax.set_title(title,fontsize=8.6);ax.set_xlim(.3,1.2);ax.set_xticks([.4,.7,1.]);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Endpoint RMSE ratio')
        ax.set_yticks(range(5),['Turn UKF','Robust IMM','IMM','CA','CV'])
    fig.tight_layout(w_pad=.9);fig.savefig(F/'paired_forecasts.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7))
    for s,color,marker in ((.35,'#244b85','o'),(1.4,'#a65043','s')):
        rr=[next(r for r in acq if r['sigma']==s and r['policy']==p) for p in POLICIES]
        for ax,key in zip(axes,('post_track_rmse','post_forecast_rmse')):
            ax.plot([r['samples'] for r in rr[:5]],[r[key] for r in rr[:5]],marker=marker,ms=3,color=color,label=f'$\\sigma={s:g}$ m')
            ax.scatter(rr[-1]['samples'],rr[-1][key],marker=marker,facecolors='none',edgecolors=color,s=28)
            ax.set_xscale('log');ax.xaxis.set_minor_locator(NullLocator());ax.set_xticks([25,50,100,200,400],['25','50','100','200','400']);ax.grid(axis='y',alpha=.15);ax.set_xlabel('Mean acquisitions over 12 s')
    axes[0].set_ylabel('Post-crossing position RMSE (m)');axes[1].set_ylabel('Post-crossing 2 s forecast RMSE (m)');axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(F/'acquisition_accuracy.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),gridspec_kw={'width_ratios':[1,1.1]})
    colors=['#a65043','#bd9255','#3b8783','#244b85']
    for b,label,color in zip(backends,bnames,colors):
        rr=[r for r in opt['micro'] if r['backend']==b]
        axes[0].plot([r['cells'] for r in rr],[r['median_us'][0] for r in rr],marker='o',ms=3,label=label,color=color,lw=1.5)
    axes[0].set(xscale='log',yscale='log',xlabel='Represented current cells',ylabel='Update time ($\\mu$s)')
    axes[0].xaxis.set_minor_locator(NullLocator());axes[0].set_xticks([32,128,512],['32','128','512']);axes[0].grid(axis='y',alpha=.15);axes[0].legend(frameon=False,fontsize=6.8)
    policies=['constant_2','adaptive_32','constant_32']
    for j,b in enumerate(('dense','native')):
        bottom=np.zeros(3);xx=np.arange(3)+(-.19 if j==0 else .19)
        for k,label,color in [('initialization_ms','Initialize','#d6dce3'),('update_ms','Assimilate','#244b85'),('readout_ms','Read out','#74a9b8'),('scheduler_ms','Schedule','#d39243')]:
            vals=np.array([cost[p,b][k] for p in policies]);axes[1].bar(xx,vals,bottom=bottom,width=.34,color=color,label=label if j==0 else None);bottom+=vals
        for x,v in zip(xx,bottom):axes[1].text(x,v+3,f'{v:.1f}',ha='center',fontsize=7)
    axes[1].set(xticks=range(3),xticklabels=['2 Hz','2 to 32 Hz','32 Hz'],ylabel='Total online time (ms)',ylim=(0,260));axes[1].legend(frameon=False,ncol=2,fontsize=6.8,loc='upper left');axes[1].grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(F/'exact_cost.pdf',bbox_inches='tight');plt.close(fig)
    (E/'derived_paired_intervals.json').write_text(json.dumps([dict(v, sigma=k[0],count=k[1],other=k[2]) for k,v in pairs.items()],indent=2)+'\n')
    print('Verified source hashes; generated 10 tables and 3 vector figures.')
if __name__=='__main__':main()
