"""Render paired acquisition/noise results without smoothing or curve fitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from .study import METHODS, save
from .data import FAMILIES

LABELS = dict(cv='CV Kalman', ca='CA Kalman', imm='IMM', robust_imm='Robust IMM',
              turn_ukf='Turn UKF', geometric='Geometric transport')
COLORS = dict(zip(METHODS, ['#8a939e','#bb8540','#9063b8','#007d78','#3b74bd','#dc482f']))


def run():
    root = Path(__file__).parent/'degradation_results'
    data = json.loads((root/'results.json').read_text())
    rows, certs = data['runs'], data['certificates']
    seeds = data['metadata']['seeds_per_family']
    rng = np.random.default_rng(20260907)
    bootstrap = np.concatenate([i*seeds+rng.integers(0,seeds,(4000,seeds))
                                for i in range(len(FAMILIES))],axis=1)
    summary = []
    for biased in (False, True):
        for sigma,count in [(s,65) for s in data['metadata']['sigmas']]+[(.35,n) for n in data['metadata']['counts'][1:]]:
            selected = sorted([r for r in rows if (r['biased'],r['sigma'],r['count'])==(biased,sigma,count)],
                              key=lambda r:(FAMILIES.index(r['family']),r['seed']))
            geo = np.array([r['forecast_sq'] for r in selected if r['method']=='geometric'])
            for method in METHODS:
                batch = [r for r in selected if r['method']==method]
                loss = np.array([r['forecast_sq'] for r in batch])
                boot_rmse = np.sqrt(loss[bootstrap].mean(axis=1))
                ratio_boot = np.sqrt(geo[bootstrap].mean(axis=1)/loss[bootstrap].mean(axis=1))
                summary.append(dict(biased=biased,sigma=sigma,count=count,method=method,
                    rmse=float(np.sqrt(loss.mean())), rmse_ci=np.quantile(boot_rmse,[.025,.975]).tolist(),
                    geo_ratio=float(np.sqrt(geo.mean()/loss.mean())),
                    geo_ratio_ci=np.quantile(ratio_boot,[.025,.975]).tolist(),
                    tracking_rmse=float(np.sqrt(np.mean([r['track_sq'] for r in batch]))),
                    gaussian_coverage=float(np.mean([r['gaussian_95_covered'] for r in batch])),
                    gaussian_radius=float(np.mean([r['gaussian_equiv_radius'] for r in batch])),
                    certified_radius=float(np.mean([r['certified_radius'] for r in batch])),
                    certified_coverage=float(np.mean([r['certified_covered'] for r in batch])),
                    milliseconds=float(np.mean([r['seconds'] for r in batch])*1000)))
    changes=[]
    for biased in (False,True):
        for method in METHODS:
            for label,condition in [('noise', (2.8,65)),('thinning',(.35,5))]:
                def losses(sigma,count):
                    batch=sorted([r for r in rows if (r['biased'],r['method'],r['sigma'],r['count'])
                                  ==(biased,method,sigma,count)],
                                 key=lambda r:(FAMILIES.index(r['family']),r['seed']))
                    return np.array([r['forecast_sq'] for r in batch])
                baseline=losses(.35,65); stressed=losses(*condition)
                ratios=np.sqrt(stressed[bootstrap].mean(axis=1)/baseline[bootstrap].mean(axis=1))
                changes.append(dict(biased=biased,method=method,change=label,
                    percent=float(100*(np.sqrt(stressed.mean()/baseline.mean())-1)),
                    percent_ci=(100*(np.quantile(ratios,[.025,.975])-1)).tolist()))
    save(root/'summary.json', {'conditions':summary,'degradation':changes,
        'bootstrap':'4000 paired resamples within each family; pointwise 95% intervals'})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white'})

    def panel_data(biased,col,method):
        ss = [s for s in summary if s['biased']==biased and s['method']==method
              and (s['count']==65 if col==0 else s['sigma']==.35)]
        ss.sort(key=lambda s:s['sigma'] if col==0 else -s['count'])
        return np.array([s['sigma'] if col==0 else s['count'] for s in ss]),ss

    def axes_style(ax,biased,col):
        ax.grid(alpha=.2)
        ax.set_xscale('log')
        ticks=data['metadata']['sigmas'] if col==0 else data['metadata']['counts']
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        if col==1: ax.invert_xaxis()
        ax.set_xlabel('Position noise σ per coordinate (m) → more noise' if col==0
                      else 'Observations in the same 6 s → fewer observations')
        ax.set_title(('Independent noise' if not biased else 'Independent noise + shared 0.75 m bias')+
                     (' · 65 observations' if col==0 else ' · σ = 0.35 m'), fontsize=11)

    for relative in (False,True):
        fig,axes=plt.subplots(2,2,figsize=(13,8.7),sharey=True,layout='constrained')
        for rr,biased in enumerate((False,True)):
            for col in range(2):
                ax=axes[rr,col]
                for method in METHODS:
                    if relative and method=='geometric': continue
                    x,ss=panel_data(biased,col,method)
                    key='geo_ratio' if relative else 'rmse'
                    y=np.array([s[key] for s in ss]); ci=np.array([s[key+'_ci'] for s in ss])
                    ax.plot(x,y,'o-',color=COLORS[method],lw=2.5 if method=='geometric' else 1.7,
                            markersize=4,label=LABELS[method])
                    ax.fill_between(x,ci[:,0],ci[:,1],color=COLORS[method],alpha=.07)
                if relative:
                    ax.axhline(1,color='#222222',ls='--',lw=1)
                    ax.set_ylabel('Geometric RMSE / comparator RMSE\n< 1: geometric better; > 1: worse')
                else:
                    ax.set_ylim(bottom=0)
                    ax.set_ylabel('2 s forecast position RMSE (m) · lower is better')
                axes_style(ax,biased,col)
        title='Relative forecast quality as information decreases' if relative else 'Forecast error as information decreases'
        fig.suptitle(title+'\n30 paired trajectories · frozen motion parameters · shaded pointwise 95% bootstrap intervals',fontsize=14)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
        name='relative_quality' if relative else 'forecast_error'
        fig.savefig(root/(name+'.png'),dpi=180)
        fig.savefig(root/(name+'.pdf'))
        plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,8.7),layout='constrained')
    for rr,biased in enumerate((False,True)):
        for col in range(2):
            ax=axes[rr,col]
            for method in METHODS:
                x,ss=panel_data(biased,col,method)
                ax.plot(x,[100*s['gaussian_coverage'] for s in ss],'o-',color=COLORS[method],label=LABELS[method])
            x,ss=panel_data(biased,col,'geometric')
            coverage=[]
            for s in ss:
                cc=[c for c in certs if (c['biased'],c['sigma'],c['count'])==(biased,s['sigma'],s['count'])]
                coverage.append(100*np.mean([c['intersection_covered'] for c in cc]))
            ax.plot(x,coverage,'s--',color='black',label='Analytic pair intersection')
            ax.axhline(95,color='black',ls=':',lw=1)
            ax.set_ylim(0,104)
            ax.set_ylabel('Observed endpoint coverage (%)')
            axes_style(ax,biased,col)
    fig.suptitle('Coverage of nominal 95% regions\nMoment ellipsoids are empirical; pair certificates have conditional ≥95% coverage',fontsize=14)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(root/'coverage.png',dpi=180); fig.savefig(root/'coverage.pdf'); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True,layout='constrained')
    for col in range(2):
        ax=axes[col]
        for method in METHODS:
            x,ss=panel_data(False,col,method)
            ax.plot(x,[s['gaussian_radius'] for s in ss],'o-',color=COLORS[method],label=LABELS[method]+' ellipsoid')
        x,ss=panel_data(False,col,'geometric')
        radii=[]
        for s in ss:
            cc=[c for c in certs if (c['biased'],c['sigma'],c['count'])==(False,s['sigma'],s['count'])]
            radii.append(np.mean([c['radius'] for c in cc]))
        ax.plot(x,radii,'s--',color='black',label='Smallest certified pair ball')
        ax.plot(x,[s['certified_radius'] for s in ss],'d:',color=COLORS['geometric'],label='Certificate around geometric mean')
        ax.set_ylim(bottom=0)
        ax.set_ylabel('Region radius (m)')
        axes_style(ax,False,col)
    fig.suptitle('Coverage has a size cost\nIndependent noise · ellipsoids use equal-volume sphere radii · coverage differs',fontsize=14)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(root/'region_size.png',dpi=180); fig.savefig(root/'region_size.pdf'); plt.close(fig)

    text=['# Matched noise and observation-density study','',
          'The geometric predictor loses ground to IMM as observations become sparse. With independent noise σ=0.35 m, reducing observations from 65 to 5 increases its two-second RMSE from 3.373 to 4.201 m (+24.5%, paired 95% interval +11.1% to +36.5%). Robust IMM rises from 3.111 to 3.295 m. At five observations geometric RMSE is 27.5% higher than robust IMM (paired interval 5.2% to 50.8%).', '',
          'Increasing noise from σ=0.35 to 2.80 m at 65 observations increases geometric RMSE by 53.1% (paired interval 41.3% to 65.7%). Relative performance is not uniformly decreasing: robust IMM degrades faster, while ordinary IMM remains more accurate at the highest noise level. The geometric method does not dominate these comparators.', '',
          'All 540 analytic pair intersections contained the true endpoint. This is accompanied by large regions: the smallest certified pair ball has radius 27.11 m at the ordinary dense condition, rising to 71.44 m at σ=2.80 m. At fixed σ=0.35, thinning barely changes this simple certificate (27.11 to 27.13 m), because the sparse schedule still contains an almost optimal observation pair. The exact intersection volume is not measured. The geometric moment ellipsoid covers only 22 of 30 endpoints (73.3%) in the ordinary dense condition; its smaller size cannot be interpreted as a valid 95% guarantee.', '',
          '30 fresh trajectories (six seeds in each of five families), two sensor regimes, nine unique conditions, six methods: 3,240 filter runs. All predict exactly two seconds after a fixed six-second history. The dense acquisition has 65 positions; nested thinning retains 33, 17, 9 or 5, including both endpoints. Independent noise per coordinate is 0.10, 0.35, 0.70, 1.40 or 2.80 m. Noise and acquisition sweeps are separate, not a full factorial grid. The second regime adds one shared bias of norm 0.75 m. All methods receive the same correct nominal sigma. Motion settings were frozen from the earlier development study. The geometric filter uses 6,144 particles; every method uses 1,024 forecast paths.','',
          'Rows use position RMSE across 30 endpoints. Ratios and pointwise 95% intervals use 4,000 paired bootstrap samples, stratified by family. They are not simultaneous confidence bands. Thirty cases provide coarse coverage estimates; analytic coverage follows from assumptions, not observed success.','',
          '| Bias | σ (m) | Observations | Method | Forecast RMSE (m) | Geometric / method [95% CI] | Moment-region coverage | Certified radius about method (m) |',
          '|---|---:|---:|---|---:|---|---:|---:|']
    for s in summary:
        lo,hi=s['geo_ratio_ci']
        text.append(f"| {s['biased']} | {s['sigma']:.2f} | {s['count']} | {LABELS[s['method']]} | {s['rmse']:.3f} | {s['geo_ratio']:.3f} [{lo:.3f}, {hi:.3f}] | {100*s['gaussian_coverage']:.1f}% | {s['certified_radius']:.2f} |")
    text+=['','The certificates are a wrapper available to every method. They do not change the geometric point prediction. All-pair intersection coverage and the smallest pair ball are reported separately from Gaussian moment ellipsoids. The exact intersection volume is not computed. A smallest pair ball is an outer approximation, not a minimum enclosing ball of the intersection.', '',
           'See [BOUND_EXAMINATION.md](BOUND_EXAMINATION.md) for proofs and scope, and `degradation_results/results.json` for raw paired results. Plots connect measured values only; no monotonic fit or smoothing is imposed.']
    (Path(__file__).parent/'DEGRADATION_FINDINGS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__': run()
