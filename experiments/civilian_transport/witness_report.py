"""Render retained witness-geometry exploration, without selecting a new model."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .study import save

ROOT=Path(__file__).parent
OUT=ROOT/'witness_results'
KINDS=['ordinary','sensor_spike','maneuver','sensor_drift']
KIND_LABELS=['Ordinary motion','Training sensor spike','New maneuver','Coherent sensor drift*']
METHODS=['prefix','raw_train','witness_ray','witness_span','reused_witness','all_history']
LABELS={'prefix':'Prefix only','raw_train':'Training update','witness_ray':'Witnessed single direction',
        'witness_span':'Witnessed correction space','reused_witness':'Reused training as witness',
        'all_history':'Original Kalman, all history'}
COLORS={'raw_train':'#87939e','witness_ray':'#1476b8','witness_span':'#189c89','all_history':'#db812e',
        'prefix':'#b998ad','reused_witness':'#b75a5a'}


def main():
    data=json.loads((OUT/'results.json').read_text())
    rows,certs=data['runs'],data['certificates']
    summaries=[]
    for sigma in (.35,1.4):
        for kind in KINDS:
            for method in METHODS:
                a=[r for r in rows if (r['sigma'],r['kind'],r['method'])==(sigma,kind,method)]
                summaries.append(dict(sigma=sigma,kind=kind,method=method,
                    witness_rmse=float(np.sqrt(np.mean([r['witness_sq'] for r in a]))),
                    forecast_rmse=float(np.sqrt(np.mean([r['forecast_sq'] for r in a])))))
    certificate_summary=[]
    for sigma in (.35,1.4):
        for kind in KINDS:
            for method in ('witness_ray','witness_span','reused_witness'):
                a=[r for r in certs if (r['sigma'],r['kind'],r['method'])==(sigma,kind,method)]
                certificate_summary.append(dict(sigma=sigma,kind=kind,method=method,count=len(a),
                    accepted=sum(r['certified'] for r in a),bound_failed=sum(r['lower_bound_failed'] for r in a),
                    local_harmed=sum(r['witness_gain']<0 for r in a),future_harmed=sum(r['future_gain']<0 for r in a),
                    median_amplification=float(np.median([r.get('endpoint_amplification',0) for r in a]))))
    paired=[]
    rng=np.random.default_rng(813)
    # Pointwise paired intervals over seeds within the five fixed families.
    for sigma in (.35,1.4):
        for kind in KINDS:
            for method in ('witness_ray','witness_span'):
                aa=sorted([r for r in rows if (r['sigma'],r['kind'],r['method'])==(sigma,kind,method)],key=lambda r:(r['family'],r['seed']))
                bb=sorted([r for r in rows if (r['sigma'],r['kind'],r['method'])==(sigma,kind,'all_history')],key=lambda r:(r['family'],r['seed']))
                av,bv=np.array([r['forecast_sq'] for r in aa]),np.array([r['forecast_sq'] for r in bb])
                families=sorted(set(r['family'] for r in aa))
                indices=np.concatenate([rng.choice([i for i,r in enumerate(aa) if r['family']==f],size=(4000,12)) for f in families],axis=1)
                ratios=np.sqrt(av[indices].mean(axis=1)/bv[indices].mean(axis=1))
                paired.append(dict(sigma=sigma,kind=kind,method=method,ratio=float(np.sqrt(av.mean()/bv.mean())),
                                   ci=np.quantile(ratios,[.025,.975])))
    save(OUT/'summary.json',dict(summaries=summaries,certificate_summary=certificate_summary,paired_forecast_ratios=paired))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    shown=['raw_train','witness_ray','witness_span','all_history']
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    for row,sigma in enumerate((.35,1.4)):
        for col,key in enumerate(('witness_rmse','forecast_rmse')):
            ax=axes[row,col]
            for j,m in enumerate(shown):
                values=[next(r[key] for r in summaries if (r['sigma'],r['kind'],r['method'])==(sigma,k,m)) for k in KINDS]
                ax.bar(np.arange(4)+(j-1.5)*.19,values,width=.18,color=COLORS[m],label=LABELS[m])
            ax.set_title(('Clean error at witness times' if col==0 else '2 s future endpoint error')+f' · σ = {sigma} m')
            ax.set_ylabel('3-D RMSE (m)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            ax.set_xticks(range(4),['Ordinary','Sensor spike','Maneuver','Sensor drift*'])
    axes[0,0].set_ylim(0,2);axes[1,0].set_ylim(0,2)
    axes[0,1].set_ylim(0,11);axes[1,1].set_ylim(0,11)
    fig.suptitle('Local evidence does not certify continuation · 60 paired paths per group',fontsize=15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False)
    fig.text(.5,.012,'* Coherent drift violates the independent zero-mean witness-noise contract.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.12,1,.95]);fig.savefig(OUT/'local_vs_future.png',dpi=180);fig.savefig(OUT/'local_vs_future.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    null=data['null_probe']
    axes[0].bar([0,1,2,3],100*np.array([null['honest_ray_false_acceptance'],null['honest_span_false_acceptance'],null['reused_ray_false_acceptance'],null['reused_span_false_acceptance']]),color=['#1476b8','#189c89','#b75a5a','#b75a5a'])
    axes[0].axhline(5,color='black',ls=':',lw=1)
    axes[0].set(xticks=range(4),xticklabels=['Honest\nray','Honest\nspace','Reused\nray','Reused\nspace'],ylabel='False support (%)',ylim=(0,105),title='Pure-noise control · 4,000 trials')
    for method,label,color in [('witness_ray','Single direction','#1476b8'),('witness_span','Correction space','#189c89')]:
        a=[r for r in certs if r['method']==method and r['witness_contract']]
        axes[1].scatter([r['lower_gain']/10 for r in a],[r['actual_gain']/10 for r in a],s=12,alpha=.45,color=color,label=label)
    limit=max(axes[1].get_xlim()[1],axes[1].get_ylim()[1]);axes[1].plot([0,limit],[0,limit],color='black',ls=':',lw=1)
    axes[1].set(xlabel='Claimed lower gain (m² per witness)',ylabel='Actual clean gain (m² per witness)',title='Independent witness contract holds')
    axes[1].legend(frameon=False,fontsize=9)
    for method,label,color in [('witness_ray','Single direction','#1476b8'),('witness_span','Correction space','#189c89')]:
        a=[r for r in certs if r['method']==method and r['kind']=='sensor_drift']
        axes[2].scatter([r['lower_gain']/10 for r in a],[r['actual_gain']/10 for r in a],s=16,alpha=.6,color=color)
    limit=max(axes[2].get_xlim()[1],axes[2].get_ylim()[1]);axes[2].plot([0,limit],[0,limit],color='black',ls=':',lw=1)
    axes[2].set(xlabel='Claimed lower gain (m² per witness)',ylabel='Actual clean gain (m² per witness)',title='Coherent drift breaks that contract')
    for ax in axes:ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(OUT/'evidence_controls.png',dpi=180);fig.savefig(OUT/'evidence_controls.pdf');plt.close(fig)

    traces=json.loads((OUT/'traces.json').read_text())
    maneuver=next(r for r in traces if r['family']=='line' and r['kind']=='maneuver')
    drift=next(r for r in traces if r['family']=='line' and r['kind']=='sensor_drift')
    t=np.array(maneuver['times']);ta=np.array(maneuver['truth']);tb=np.array(drift['truth'])
    direction=ta[-1]-tb[-1];gap=float(np.linalg.norm(direction));direction/=gap
    fig,ax=plt.subplots(figsize=(11,5.3))
    ax.plot(t,ta @ direction,color='#303b4a',lw=2.4,label='Clean truth: physical maneuver')
    ax.plot(t,tb @ direction,color='#885b9e',lw=2.4,label='Clean truth: sensor drift')
    obs=np.array(maneuver['observations'])
    ax.scatter(t[:61],obs @ direction,s=13,color='#8d949e',label='Identical observed history')
    for m in ('all_history','witness_span'):
        ax.plot(t[60:],np.array(maneuver['methods'][m])[60:] @ direction,lw=2,color=COLORS[m],label=LABELS[m]+' (same in both worlds)')
    ax.axvline(6,color='#8d949e',ls=':',lw=1)
    ax.plot([8,8],[tb[-1] @ direction,ta[-1] @ direction],color='#303b4a',ls=':',lw=1.5)
    ax.annotate(f'{gap:.2f} m truth separation',xy=(8,.5*(ta[-1]+tb[-1]) @ direction),xytext=(-8,0),textcoords='offset points',ha='right',va='center',fontsize=10)
    ax.set(xlim=(3.5,8.15),xlabel='Physical time (s)',ylabel='Position along injected change (m)',title='Same observations cannot identify whether a coherent change is motion or sensor drift')
    ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=9,frameon=False)
    fig.tight_layout();fig.savefig(OUT/'identical_observations.png',dpi=180);fig.savefig(OUT/'identical_observations.pdf');plt.close(fig)

    a=[r for r in certs if r['method']=='witness_span' and r['witness_contract']]
    ray=[r for r in certs if r['method']=='witness_ray' and r['witness_contract']]
    amp=np.array([r['endpoint_amplification'] for r in a])
    event_fail=sum(not r['confidence_event'] for r in a)
    lines=['# Witnessed correction: findings','',
           'The experiment borrows the local-restoration observation from BTB; it implements no BTB iteration. It constructs a finite confidence geometry from independent history witnesses and tests its limits. The main result is a separation: local clean-error improvement can be supported without assuming the current prior is correct, but extending that correction through the same current relationships does not establish improved prediction.','',
           '## Construction and proof','',
           'See [WITNESS_GEOMETRY.md](WITNESS_GEOMETRY.md) for the derivation. Directions are the complete current-mean revisions induced by ten training observations. The witness values never enter the direction construction. For a rank-r witness space, the projected clean residual lies inside a chi-square confidence ball. The maximum guaranteed-gain correction is the projected observed residual shortened by that ball’s radius. This is an algebraic calculation, not repeated denoising. A separate single-direction procedure only shortens the combined training correction.','',
           'The guarantee is conditional on independent, zero-mean Gaussian witness errors of known sigma. It concerns summed clean squared position error at the ten witness times. It is uniform over all corrections in the training-defined space, so selecting the correction using the witness confidence ball is permitted. It makes no claim about the unobserved future or the physical cause of a residual.','',
           '## Protocol','',
           'Five existing motion families; 12 fresh seeds each; sigma 0.35 and 1.4 m; ordinary motion, one 3 m training-only sensor spike, a new maneuver, and coherent sensor drift. This gives 480 cases, 2880 retained method/case records, and 1440 certificate records. Prefix observations are 0–4 s; training observations are 4.2,4.4,...,6 s; witnesses are 4.1,4.3,...,5.9 s. All forecasts are made at 6 s for 8 s. Future observation values are replaced with NaN before proposal construction. No threshold or prior was tuned on these results.','',
           'Original Kalman uses all 61 history observations; training update uses the 41-point prefix plus ten training points; witnessed methods additionally use the ten witnesses in the support calculation. The original Kalman witness error is evaluated against clean truth at positions it has observed noisily, so it is a denoising comparison, not an independent validation score for that baseline.','',
           '## Pure-noise control','',
           f"With 4000 pure-noise trials, false support is {100*data['null_probe']['honest_ray_false_acceptance']:.2f}% for the honest ray and {100*data['null_probe']['honest_span_false_acceptance']:.2f}% for the honest correction space, consistent with their separate 5% levels. Reusing the noise that created the direction as its own witness produces 100% false support for both. This is a concrete self-confirmation failure; it is not a theorem about every denoiser or EM system.",'',
           '## Local error and forecast error','',
           'Each entry is clean witness RMSE / two-second endpoint RMSE, both in metres. Each condition has 60 paired paths.','',
           '| Noise | Scenario | Training update | Witnessed ray | Witnessed space | Original Kalman, all history |',
           '|---|---|---:|---:|---:|---:|']
    for sigma in (.35,1.4):
        for kind,label in zip(KINDS,KIND_LABELS):
            values=[]
            for m in shown:
                r=next(r for r in summaries if (r['sigma'],r['kind'],r['method'])==(sigma,kind,m))
                values.append(f"{r['witness_rmse']:.3f} / {r['forecast_rmse']:.3f}")
            lines.append(f'| {sigma} | {label} | '+' | '.join(values)+' |')
    lines += ['',
           'For ordinary motion at sigma 0.35, the prefix-only witness/forecast RMSE is 1.939 / 7.113 m. The witnessed space improves local error to 0.513 m but worsens the future RMSE to 7.832 m. Original Kalman reaches 0.188 / 3.413 m. A valid local improvement certificate therefore does not establish a competitive denoiser or an improved forecast.',
           '','## Bounds and transport conditioning','',
           f"Among the 360 cases satisfying the witness contract, the ray lower bound fails in {sum(r['lower_bound_failed'] for r in ray)}/360 records; the space lower bound fails in {sum(r['lower_bound_failed'] for r in a)}/360. The underlying space confidence event fails in {event_fail}/360. The latter event guarantees all directional lower bounds; a failure of that sufficient event need not violate the particular selected bound. These are measured counts, not replacements for the mathematical assumptions.",'',
           f"Across these cases, future amplification ||E(8)||/sigma has median {np.median(amp):.2f}, 90th percentile {np.quantile(amp,.9):.2f}, and maximum {np.max(amp):.2f}. It measures endpoint change per unit stacked witness change within the proposed correction space. Weakly witnessed combinations can therefore generate large endpoint changes. This operator norm is a diagnostic, not a direct error bound on an arbitrary future path.",'',
           'The one-direction correction avoids reweighting individually weak directions and forecasts much better than the full-space correction. It still does not consistently improve over the original all-history Kalman model. Its lack of a future guarantee remains unchanged.',
           '','## Identical observations, incompatible truth','',
           f"The maneuver and drift cases have exactly identical histories and exactly identical corrected outputs, verified for all 120 matched pairs. Their clean endpoints differ by {gap:.6f} m. Any common point prediction must be at least {gap/2:.6f} m from one of those two truths. Any set containing both endpoints must have diameter at least {gap:.6f} m. These conclusions need no estimator-specific argument.",'',
           'The drift breaks the witness-noise assumption. At sigma 0.35 the ray and space lower bounds fail in 48/60 and 46/60 drift cases respectively. Independent random noise on top of that drift does not make the drift independent zero-mean sensor error. The experiment cannot identify its cause from those observations alone.',
           '','## Consequence for the construction','',
           'The proof extends to any calibrated closed convex projected-noise region K: the maximum guaranteed-gain correction is v minus its projection onto K. An allowed coherent-nuisance subspace makes K a cylinder; its nuisance component cannot support a guaranteed correction. These are mathematical extensions, not additional fitted benchmark results. See Sections 3a and 6a of WITNESS_GEOMETRY.md.','',
           'A current prior can propose a correction, and untouched observations can establish a local benefit without treating that prior as truth. That is a useful interface. It does not identify an unrestricted corruption process, certify a physical continuation law, or justify transporting every locally supported change into the future. A further construction must specify which relationships support continuation, retain future-invisible directions, and account for coherent sensor alternatives. No new denoising loop is called for by this result.',
           '','## Reproduction and artifacts','',
           '```sh','sh experiments/civilian_transport/run_witness_m4.sh',
           '.venv-jpeg/bin/python -m experiments.civilian_transport.witness_report','```','',
           'Nine focused tests cover the uniform confidence bound, basis and spatial equivariance, zero/rank-deficient fields, future nullspace, the ray bound, exact null rate, absence of witness/future leakage, and the indistinguishable observation pair. Existing relational Kalman tests run alongside them. Source hashes, full results, certificate records, traces, and paired bootstrap summaries are retained in `witness_results/`. The confidence theorem is pointwise per procedure/record; it is not simultaneous across the entire experiment.','',
           '![Local versus future](witness_results/local_vs_future.png)','',
           '![Evidence controls](witness_results/evidence_controls.png)','',
           '![Identical observations](witness_results/identical_observations.png)']
    (ROOT/'WITNESS_FINDINGS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
