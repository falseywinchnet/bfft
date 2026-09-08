"""Render retained first relational-current experiment; no re-fitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import FAMILIES
from .study import save

ROOT = Path(__file__).parent
OUT = ROOT/'relational_results'
LABELS = {'relational':'Relational current', 'severed':'Past–future severed',
          'single_length':'Single length 1.5 s', 'cv':'CV Kalman', 'ca':'CA Kalman',
          'imm':'IMM', 'robust_imm':'Robust IMM', 'turn_ukf':'Turn UKF'}
CONDITIONS = [(.35, 65), (1.4, 65), (.35, 9)]
NAMES = ['65 observations · σ = 0.35 m', '65 observations · σ = 1.4 m', '9 observations · σ = 0.35 m']


def main():
    data = json.loads((OUT/'results.json').read_text())
    rows = data['runs']
    summaries = []
    methods = list(LABELS)
    for condition in CONDITIONS:
        for method in methods:
            a = [r for r in rows if (r['sigma'],r['count'])==condition and r['method']==method]
            summaries.append(dict(sigma=condition[0], count=condition[1], method=method,
                                  rmse=float(np.sqrt(np.mean([r['sq'] for r in a]))),
                                  path_rmse=float(np.sqrt(np.mean([r['path_sq'] for r in a]))),
                                  coverage=float(np.mean([r['covered'] for r in a])),
                                  radius=float(np.mean([r['radius'] for r in a])),
                                  median_ms=float(1000*np.median([r['seconds'] for r in a]))))
    comparisons = []
    rng = np.random.default_rng(42)
    for condition in CONDITIONS:
        a = sorted([r for r in rows if (r['sigma'],r['count'])==condition and r['method']=='relational'],key=lambda r:(r['family'],r['seed']))
        for other in ('robust_imm', 'imm', 'severed', 'single_length'):
            b = sorted([r for r in rows if (r['sigma'],r['count'])==condition and r['method']==other],key=lambda r:(r['family'],r['seed']))
            assert [(r['family'],r['seed']) for r in a]==[(r['family'],r['seed']) for r in b]
            av,bv=np.array([r['sq'] for r in a]),np.array([r['sq'] for r in b])
            indices=np.concatenate([rng.choice([i for i,r in enumerate(a) if r['family']==f],size=(4000,6)) for f in FAMILIES],axis=1)
            boot=np.sqrt(av[indices].mean(axis=1)/bv[indices].mean(axis=1))
            comparisons.append(dict(sigma=condition[0],count=condition[1],other=other,
                                    ratio=float(np.sqrt(av.mean()/bv.mean())),ci=np.quantile(boot,[.025,.975])))
    save(OUT/'summary.json',dict(summaries=summaries,paired_ratios=comparisons))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,5.4),sharex=True)
    colors=['#1269a7','#b99a9a','#87b9d7','#777777','#9b7bb7','#519168','#e1892c','#ae6179']
    x=np.arange(3)
    for j,m in enumerate(methods):
        s=[next(s for s in summaries if (s['sigma'],s['count'])==c and s['method']==m) for c in CONDITIONS]
        for ax,key in zip(axes,['rmse','coverage','radius']):
            ax.plot(x,[r[key]*(100 if key=='coverage' else 1) for r in s],marker='o',label=LABELS[m],color=colors[j],lw=2.5 if m=='relational' else 1.2)
    for ax,title,ylabel in zip(axes,['2 s endpoint error','Moment-ellipsoid coverage','Mean region size'],['3-D RMSE (m)','Observed coverage (%)','Equal-volume sphere radius (m)']):
        ax.set(title=title,ylabel=ylabel,xticks=x,xticklabels=['Dense\nσ=.35','Dense\nσ=1.4','Sparse\nσ=.35'])
        ax.grid(alpha=.18)
    axes[0].set_ylim(bottom=0); axes[1].set_ylim(0,105); axes[2].set_ylim(bottom=0)
    axes[1].axhline(95,color='black',ls=':',lw=1)
    fig.suptitle('Relational-current Kalman: first frozen experiment · 30 paired paths per condition',fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=[0,.15,1,.94])
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    traces=json.loads((OUT/'traces.json').read_text())
    fig,axes=plt.subplots(3,5,figsize=(17,9),sharex=True)
    for col,f in enumerate(FAMILIES):
        a=next(r for r in traces if r['family']==f and r['method']=='relational')
        b=next(r for r in traces if r['family']==f and r['method']=='robust_imm')
        t=np.array(a['times']); truth=np.array(a['truth']); obs=np.array(a['observations'])
        q=np.array(a['forecast_times']); mean=np.array(a['mean']); cov=np.array(a['covariance'])
        for d in range(3):
            ax=axes[d,col]
            ax.plot(t,truth[:,d],color='#30343b',lw=1.7,label='Truth')
            ax.scatter(t[:65],obs[:,d],s=5,color='#92969e',alpha=.7,label='Observed')
            ax.plot(q,mean[:,d],color='#1269a7',lw=2,label='Relational current')
            sd=np.sqrt(cov[:,d,d]);ax.fill_between(q,mean[:,d]-1.96*sd,mean[:,d]+1.96*sd,color='#1269a7',alpha=.15,label='±1.96 marginal SD')
            ax.plot(q,np.array(b['mean'])[:,d],color='#e1892c',lw=1.6,label='Robust IMM')
            ax.axvline(6,color='#808080',ls=':',lw=1);ax.grid(alpha=.15)
            if d==0: ax.set_title(f.replace('_',' ').title())
            if col==0: ax.set_ylabel(['x (m)','y (m)','z (m)'][d])
            if d==2: ax.set_xlabel('Physical time (s)')
    fig.suptitle('First retained seed in every family · forecasts use only observations at t ≤ 6 s',fontsize=14)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False)
    fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(OUT/'trajectories.png',dpi=160);fig.savefig(OUT/'trajectories.pdf');plt.close(fig)
    lines=['# First relational-current Kalman experiment','',
           'This implements the explicit Gaussian structural special case of KALMAN_FOUNDATION.md. It does not implement or establish a general nonlinear relational BTB denoiser. No parameter was changed after inspecting this evaluation.','',
           '## Construction','',
           'The state contains an anchor and 64 orthonormal piecewise-constant current coefficients on [0,8] s. Position observations integrate these basis functions exactly. The current covariance is a shared constant component of variance 4 plus a Matérn-3/2 component of variance 4 (m/s)^2; coefficient covariance scales with cell width. Five fixed correlation lengths (0.3, 0.7, 1.5, 3, 6 s) receive a uniform prior. Their exact history marginal likelihoods determine posterior weights, and total covariance includes between-component uncertainty. This is a declared motion prior, not a law supplied by the motivating papers.','',
           'The first measurement conditions a flat absolute-position prior once. Subsequent measurements are independent Gaussian errors of known sigma. Zak analysis/synthesis retains full covariance and the real-signal conjugacy constraints. Sequential Kalman conditioning retains the complete current covariance as observations arrive and evaluates the Gaussian fixed point; it is tested against joint conditioning. A separate actual relaxed proximal iteration tests the BTB-compatible Gaussian map; it is not substituted into the reported forecast before convergence.','',
           '## Evaluation contract','',
           'Five existing motion families, six fresh seeds per family (80000+1000×family index+repeat), three paired conditions: 65 observations with sigma 0.35 m; 65 with sigma 1.4 m; 9 with sigma 0.35 m. History is 0–6 s; forecast is 6–8 s. Truth values after 6 s never reach an estimator. All 720 method/case runs are retained. Baselines use their previously frozen configurations, changing only supplied sensor sigma, with 2048 forecast draws. This is an exploratory synthetic screen; no real sensor or correlated-noise claim is made.','',
           '## Endpoint results','',
           '| Method | Dense σ=.35 RMSE | Dense σ=1.4 RMSE | Sparse σ=.35 RMSE |','|---|---:|---:|---:|']
    for m in methods:
        values=[next(s['rmse'] for s in summaries if s['method']==m and (s['sigma'],s['count'])==c) for c in CONDITIONS]
        lines.append('| '+LABELS[m]+' | '+' | '.join(f'{v:.3f} m' for v in values)+' |')
    lines += ['', 'RMSE is sqrt(mean squared 3-D endpoint distance), with equal case weights.','',
              '## Paired uncertainty','',
              'Ratios below are relational RMSE / comparator RMSE. Pointwise 95% percentile intervals use 4000 paired bootstrap resamples, six seeds resampled within each of the five fixed families. They do not establish generalization over arbitrary motion families or simultaneous significance.','',
              '| Condition | Comparator | Ratio | Interval |','|---|---|---:|---|']
    for r in comparisons:
        lines.append(f"| σ={r['sigma']}, n={r['count']} | {LABELS[r['other']]} | {r['ratio']:.3f} | [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] |")
    lines += ['', '## Interpretation and unresolved work','',
              '- The declared current model is competitive in this first screen. Read the paired intervals before describing the small differences versus robust IMM as a win.',
              '- Severing current covariance across 6 s worsens point prediction and greatly enlarges forecast regions. Its future current is independent and zero mean, so this ablation intentionally removes the continuation mechanism. It establishes the importance of this prior connection, not its uniqueness or optimality.',
              '- The length mixture improves over fixed 1.5 s at low noise but does not win every condition. Hyperparameter uncertainty is retained; there is no post-hoc choice of the best length using future truth.',
              '- All five 64→128 cell audits change endpoint means by less than 0.022 m. This is a limited resolution check, not a convergence theorem or uniform truncation guarantee.',
              '- Zak/current and four-step/single-step assimilation agree to less than 3e-11 in the retained audits. They are equivalent computations of the same posterior, not independent accuracy gains.',
              '- The unpreconditioned relaxed proximal iteration is slow: after 2000 steps its relative whitened-state mean error is 55–83%, despite normalized residuals around 0.00016–0.00111. Its worst contraction factor is 0.9999795. Small residuals do not certify a recovered state. Exact Gaussian conditioning supplies the reported predictions; this iterative path has no demonstrated practical advantage.',
              '- Moment ellipsoids use the chi-square-3 95% threshold. For mixtures and these out-of-model motion fixtures they are diagnostics, not exact 95% probability sets. The relational model covers 30/30, 28/30, 30/30 endpoints; its mean equal-volume radii are 5.77, 8.42, 6.61 m. Small-sample coverage does not prove a guarantee.',
              '- Recorded timing is the actual complete query workload: exact analytical moments for the current model versus baseline Monte Carlo forecasts. It is not a matched-kernel latency or complexity claim. Dense covariance and growing current dimension remain scaling costs.',
              '- A justified nonlinear relational operator, robust correlated sensor model, and calibration on external motion remain unresolved. This experiment tests an explicit integrated Gaussian-process prior with uncertain correlation length; it does not claim the full original mathematical objective has been solved.','',
              '## Reproduction','', '```sh','sh experiments/civilian_transport/run_relational_m4.sh',
              '.venv-jpeg/bin/python -m experiments.civilian_transport.relational_report','```','',
              'The runner uses m4build/m4host and immediately copies results home. The retained JSON includes source hashes. Eleven tests cover streaming/batch equivalence, integration, Zak identity, likelihood accounting, spatial equivariance, covariance/information order, future nullspace, mixture variance, proximal convergence on a small case, and prior-predictive calibration.','',
              '![Comparison](relational_results/comparison.png)','', '![Trajectories](relational_results/trajectories.png)']
    (ROOT/'RELATIONAL_FINDINGS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
