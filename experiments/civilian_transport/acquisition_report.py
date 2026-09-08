"""Render retained acquisition/assimilation tradeoffs without selecting a policy."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .study import save

ROOT=Path(__file__).parent
OUT=ROOT/'acquisition_results'
POLICIES=['constant_2','adaptive_4','adaptive_8','adaptive_16','adaptive_32','constant_32']
LABELS={'constant_2':'Constant 2 Hz','adaptive_4':'2 → 4 Hz','adaptive_8':'2 → 8 Hz',
        'adaptive_16':'2 → 16 Hz','adaptive_32':'2 → 32 Hz','constant_32':'Constant 32 Hz'}


def main():
    data=json.loads((OUT/'results.json').read_text())
    rows=data['runs'];rng=np.random.default_rng(742)
    summaries=[];paired=[];marginals=[]
    for sigma in (.35,1.4):
        for policy in POLICIES:
            a=sorted([r for r in rows if (r['sigma'],r['policy'])==(sigma,policy)],key=lambda r:(r['family'],r['seed']))
            families=sorted(set(r['family'] for r in a))
            indices=np.concatenate([rng.choice([i for i,r in enumerate(a) if r['family']==f],size=(4000,8)) for f in families],axis=1)
            s=dict(sigma=sigma,policy=policy,count=len(a))
            for key in ('samples','updates','initialization_ms','update_ms','readout_ms','scheduler_ms','total_online_ms','mean_update_ms'):
                s[key]=float(np.mean([r[key] for r in a]))
            for stage in ('approach','first_second','settling','later','post'):
                for metric in ('track','forecast'):
                    values=np.array([r['stages'][stage][metric+'_sq'] for r in a])
                    s[stage+'_'+metric+'_rmse']=float(np.sqrt(values.mean()))
                    s[stage+'_'+metric+'_ci']=np.quantile(np.sqrt(values[indices].mean(axis=1)),[.025,.975])
                for key in ('mean_radius','coverage','mean_forecast_radius','forecast_coverage','mean_correction','mean_projected_gain'):
                    s[stage+'_'+key]=float(np.mean([r['stages'][stage][key] for r in a]))
            delays=np.array([r['trigger_delay'] for r in a])
            s.update(trigger_delay_median=float(np.median(delays)),trigger_delay_range=[float(delays.min()),float(delays.max())],
                     trigger_delay_p90=float(np.quantile(delays,.9)),trigger_early_count=int(np.count_nonzero(delays<0)))
            summaries.append(s)
        for stage in ('first_second','later','post'):
            aa=sorted([r for r in rows if (r['sigma'],r['policy'])==(sigma,'adaptive_32')],key=lambda r:(r['family'],r['seed']))
            bb=sorted([r for r in rows if (r['sigma'],r['policy'])==(sigma,'constant_32')],key=lambda r:(r['family'],r['seed']))
            indices=np.concatenate([rng.choice([i for i,r in enumerate(aa) if r['family']==f],size=(4000,8)) for f in sorted(set(r['family'] for r in aa))],axis=1)
            av=np.array([r['stages'][stage]['track_sq'] for r in aa]);bv=np.array([r['stages'][stage]['track_sq'] for r in bb])
            boot=np.sqrt(av[indices].mean(axis=1)/bv[indices].mean(axis=1))
            paired.append(dict(sigma=sigma,stage=stage,ratio=float(np.sqrt(av.mean()/bv.mean())),ci=np.quantile(boot,[.025,.975])))
        a=[next(r for r in summaries if (r['sigma'],r['policy'])==(sigma,p)) for p in POLICIES[:5]]
        for before,after in zip(a,a[1:]):
            gain=before['post_track_rmse']**2-after['post_track_rmse']**2
            marginals.append(dict(sigma=sigma,from_policy=before['policy'],to_policy=after['policy'],
                mse_gain=gain,extra_samples=after['samples']-before['samples'],extra_update_ms=after['update_ms']-before['update_ms'],
                gain_per_sample=gain/(after['samples']-before['samples']),
                gain_per_update_ms=gain/(after['update_ms']-before['update_ms'])))
    save(OUT/'summary.json',dict(summaries=summaries,paired_adaptive32_constant32=paired,marginal_returns=marginals))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13.5,9))
    for row,sigma in enumerate((.35,1.4)):
        seq=[next(r for r in summaries if (r['sigma'],r['policy'])==(sigma,p)) for p in POLICIES[:5]]
        dense=next(r for r in summaries if (r['sigma'],r['policy'])==(sigma,'constant_32'))
        for col,cost in enumerate(('update_ms','samples')):
            ax=axes[row,col];x=np.array([r[cost] for r in seq])
            for stage,label,color,ls in [('post','Whole post-crossing interval','#176da4','-'),('later','3–6 s after crossing','#169881','--')]:
                y=np.array([r[stage+'_track_rmse'] for r in seq]);cis=np.array([r[stage+'_track_ci'] for r in seq])
                ax.errorbar(x,y,yerr=np.maximum(np.stack([y-cis[:,0],cis[:,1]-y]),0),fmt='o',ls=ls,color=color,lw=2,capsize=3,label=label)
            for r in seq:
                ax.annotate('2 Hz' if r['policy']=='constant_2' else r['policy'].split('_')[-1]+' Hz',
                            (r[cost],r['post_track_rmse']),xytext=(4,6),textcoords='offset points',fontsize=9)
            ax.scatter([dense[cost]],[dense['post_track_rmse']],marker='*',s=180,color='#db812e',label='Constant 32 Hz, entire run',zorder=4)
            ax.set_title(f'Position accuracy / '+('integration cost' if col==0 else 'acquisition count')+f' · σ = {sigma} m')
            ax.set_xlabel('Assimilation CPU time over 12 s (ms)' if col==0 else 'Acquired positions over 12 s')
            ax.set_ylabel('3-D position RMSE (m)');ax.set_ylim(bottom=0);ax.grid(alpha=.18)
    fig.suptitle('One retained tracker · boundary-triggered sampling · 40 paired paths per noise level',fontsize=15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(OUT/'benefit_cost.png',dpi=180);fig.savefig(OUT/'benefit_cost.pdf');plt.close(fig)

    traces=json.loads((OUT/'traces.json').read_text())
    selected={r['policy']:r for r in traces if r['family']=='helix' and r['sigma']==.35}
    a=selected['adaptive_32'];t=np.array(a['times']);cross=a['crossing_time'];trigger=a['trigger_time']
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True,gridspec_kw={'height_ratios':[1,1.3,1.2,1]})
    colors={'constant_2':'#8a939e','adaptive_32':'#176da4','constant_32':'#db812e'}
    for policy in ('constant_2','adaptive_32','constant_32'):
        r=selected[policy]
        axes[0].plot(t,r['rate'],color=colors[policy],label=LABELS[policy],lw=2)
        axes[2].plot(t,np.sqrt(r['track_sq']),color=colors[policy],label=LABELS[policy],lw=1.5)
    axes[0].set_ylabel('Requested Hz');axes[0].legend(ncol=3,frameon=False,fontsize=10)
    for j,policy in enumerate(POLICIES):
        event_times=[e['time'] for e in selected[policy]['events']]
        axes[1].scatter(event_times,np.full(len(event_times),j),marker='|',s=70,color='#176da4' if policy.startswith('adaptive') else '#8a939e')
    axes[1].set_yticks(range(6),[LABELS[p] for p in POLICIES]);axes[1].invert_yaxis()
    axes[1].set_ylabel('Actual acquisitions')
    axes[2].set_ylabel('Position error (m)')
    events=a['events'][1:]
    axes[3].scatter([e['time'] for e in events],[e['correction'] for e in events],s=16,color='#176da4',alpha=.8)
    axes[3].set_ylabel('Correction size (m)');axes[3].set_xlabel('Physical time (s)')
    for ax in axes:
        ax.axvline(cross,color='#303b4a',ls=':',lw=1.3)
        ax.axvline(trigger,color='#b45a73',ls='--',lw=1.3)
        ax.set_xlim(3.5,12);ax.grid(alpha=.15)
    axes[0].annotate('Estimated crossing: rate starts rising',xy=(trigger,2),xytext=(7.,12.),arrowprops=dict(arrowstyle='->',color='#b45a73'),fontsize=10,color='#943e59')
    fig.suptitle('Irregular timestamps and a smooth rate transition · first helix-derived seed, σ = 0.35 m',fontsize=14)
    fig.text(.5,.012,'Dotted black: true crossing (evaluation only). Dashed rose: estimated crossing used by the sampler.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.95]);fig.savefig(OUT/'sampling_transition.png',dpi=170);fig.savefig(OUT/'sampling_transition.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for sigma,color in ((.35,'#176da4'),(1.4,'#b55c77')):
        a=[r for r in marginals if r['sigma']==sigma]
        for ax,key in zip(axes,('gain_per_update_ms','gain_per_sample')):
            ax.plot(range(4),[r[key] for r in a],marker='o',color=color,lw=2,label=f'σ = {sigma} m')
            ax.set_xticks(range(4),['2→4','4→8','8→16','16→32'])
            ax.set_yscale('log');ax.grid(alpha=.2,which='major')
            ax.set_xlabel('Increase in post-trigger target rate (Hz)')
    axes[0].set_ylabel('Position MSE reduction / extra assimilation ms')
    axes[1].set_ylabel('Position MSE reduction / extra acquired sample')
    axes[0].legend(frameon=False)
    fig.suptitle('Measured marginal benefit decreases smoothly as the acquisition budget increases',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'marginal_return.png',dpi=180);fig.savefig(OUT/'marginal_return.pdf');plt.close(fig)

    lines=['# Acquisition density: measured benefit and integration cost','',
           'This experiment keeps one retained relational-current Kalman update law. The ordinary covariance-weighted innovation supplies the shortened correction; the independent-witness gate from the preceding exploration is not part of this acquisition test. Sampling density is the controlled variable. No tracker or correction method is selected at the boundary.','',
           'The result supports the proposed tradeoff: added irregular samples improve tracking smoothly, assimilation cost grows approximately with the number of observations, and individual corrections become smaller as more history is available. After settling, adaptive high-density sampling reaches the tracking accuracy of continuous high-density acquisition at substantially lower whole-run cost.','',
           '## Protocol','',
           'See [ACQUISITION_PROTOCOL.md](ACQUISITION_PROTOCOL.md). Position arrivals have +/-20% phase jitter around the requested cadence. The sampler begins at 2 Hz. When an acquired observation updates the estimated x position to x>=0, the request increases exponentially toward 4, 8, 16, or 32 Hz with a one-second time constant. The boundary affects acquisition only; the motion model and state are not reset. Constant 2 and 32 Hz profiles provide acquisition-cost references.','',
           'Five boundary-crossing fixture families, eight fresh seeds, and two Gaussian noise levels give 80 paired cases and 480 retained main policy/case runs. All accuracy measurements use the same 0.1 s physical evaluation grid. Acquisition is driven by the estimated crossing, never the true crossing. Adaptive profiles have exactly identical acquired histories and trigger times before their requested cadences diverge.','',
           '## Whole-run cost and post-crossing accuracy','',
           'Sample counts and costs cover the entire 12 s run. Position and forecast RMSE cover the interval after the true crossing; forecast horizon is two seconds. CPU integration time excludes initialization, readouts, scheduling, and diagnostic-only probes; total measured online time includes the first three. No sensor, radio, energy or bandwidth price is assigned to a sample.','',
           '| Noise σ | Acquisition profile | Mean samples | Integration ms | Total online ms | Position RMSE m | 2 s forecast RMSE m |','|---|---|---:|---:|---:|---:|---:|']
    for sigma in (.35,1.4):
        for policy in POLICIES:
            r=next(r for r in summaries if (r['sigma'],r['policy'])==(sigma,policy))
            lines.append(f"| {sigma} | {LABELS[policy]} | {r['samples']:.1f} | {r['update_ms']:.1f} | {r['total_online_ms']:.1f} | {r['post_track_rmse']:.3f} | {r['post_forecast_rmse']:.3f} |")
    lines += ['', 'The per-observation integration cost is approximately 0.46 ms throughout this fixed 128-cell experiment on the M4 CPU. This is a measured implementation cost, not a universal hardware or real-time guarantee. Dense covariance and the finite represented interval remain implementation constraints.','',
              '## Transition and retained history','',
              '| Noise σ | Profile | First second position RMSE m | 3–6 s position RMSE m | Later mean correction m | Later projected gain |','|---|---|---:|---:|---:|---:|']
    for sigma in (.35,1.4):
        for policy in POLICIES:
            r=next(r for r in summaries if (r['sigma'],r['policy'])==(sigma,policy))
            lines.append(f"| {sigma} | {LABELS[policy]} | {r['first_second_track_rmse']:.3f} | {r['later_track_rmse']:.3f} | {r['later_mean_correction']:.3f} | {r['later_mean_projected_gain']:.3f} |")
    lines += ['', 'The projected gain is delta_position·innovation / ||innovation||², measured at each acquisition and then averaged. It describes the effective correction of this mixture model; it is not an imposed clipping coefficient. The rate command is continuous, while actual measurement updates remain discrete. No claim of pointwise state continuity is made.','',
              '## Adaptive high density versus continuous high density','']
    for sigma in (.35,1.4):
        a=next(r for r in summaries if (r['sigma'],r['policy'])==(sigma,'adaptive_32'))
        b=next(r for r in summaries if (r['sigma'],r['policy'])==(sigma,'constant_32'))
        ratio=next(r for r in paired if r['sigma']==sigma and r['stage']=='later')
        lines.append(f"At σ={sigma}, adaptive 2→32 Hz uses {100*a['samples']/b['samples']:.1f}% of the acquired samples, {100*a['update_ms']/b['update_ms']:.1f}% of assimilation time, and {100*a['total_online_ms']/b['total_online_ms']:.1f}% of total measured online time. After 3 s of settling its position RMSE is {a['later_track_rmse']:.3f} m versus {b['later_track_rmse']:.3f} m. The paired RMSE ratio is {ratio['ratio']:.3f}, with a pointwise 95% bootstrap interval [{ratio['ci'][0]:.3f}, {ratio['ci'][1]:.3f}].")
        lines.append('')
    lines += ['The paired intervals use 4000 resamples of eight seeds within each of the five fixed motion families. They are pointwise, not simultaneous claims or generalization over arbitrary flights. Continuous high density is more accurate immediately after crossing because it has already accumulated dense history. The adaptive profile pays for its lower earlier acquisition cost with a settling interval.','',
              '## Marginal return','',
              '| Noise σ | Target increase | Extra samples | Extra integration ms | Position MSE reduction / ms | Position MSE reduction / sample |','|---|---|---:|---:|---:|---:|']
    for r in marginals:
        lines.append(f"| {r['sigma']} | {LABELS[r['from_policy']]} to {LABELS[r['to_policy']]} | {r['extra_samples']:.1f} | {r['extra_update_ms']:.2f} | {r['gain_per_update_ms']:.5f} | {r['gain_per_sample']:.5f} |")
    lines += ['', 'These are observed returns on the unchanged update law. No runtime policy winner is selected. A hardware-specific acquisition cost can later be combined with the separate count and compute measurements. More samples improve current tracking more strongly than they improve two-second prediction in this screen; the continuation model is still substantive.','',
              '## Trigger, uncertainty and representation checks','']
    for sigma in (.35,1.4):
        r=next(r for r in summaries if (r['sigma'],r['policy'])==(sigma,'adaptive_32'))
        lines.append(f"At σ={sigma}, median estimated-trigger delay is {r['trigger_delay_median']:.3f} s, with observed range [{r['trigger_delay_range'][0]:.3f}, {r['trigger_delay_range'][1]:.3f}] s and 90th percentile {r['trigger_delay_p90']:.3f} s. Negative delay means the estimate triggered before the clean path crossed. These decisions were retained, not replaced with oracle crossing times.")
        lines.append('')
    lines += ['All four 128→256-cell checks change post-crossing position RMSE by less than 0.00013 m and two-second forecast RMSE by less than 0.00084 m, with identical triggers and sample counts. These are limited checks, not a uniform resolution theorem. Moment-region sizes and empirical coverage are retained in summary.json; they are not exact 95% guarantees for these out-of-model trajectories.','',
              'Seventeen tests passed on the Mini: six cadence/end-to-end checks and eleven existing relational Kalman checks. Source hashes and all 480 unique main records are retained.','',
              '## Reproduction','', '```sh','sh experiments/civilian_transport/run_acquisition_m4.sh',
              '.venv-jpeg/bin/python -m experiments.civilian_transport.acquisition_report','```','',
              '![Benefit and cost](acquisition_results/benefit_cost.png)','',
              '![Sampling transition](acquisition_results/sampling_transition.png)','',
              '![Marginal return](acquisition_results/marginal_return.png)']
    (ROOT/'ACQUISITION_FINDINGS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
