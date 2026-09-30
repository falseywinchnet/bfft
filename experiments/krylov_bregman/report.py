"""Render retained records and generate the manuscript's numerical tables."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['ordinary','flow_euclidean','flow_mirror','stationary_krylov','anderson']
LABELS={'ordinary':'Ordinary','flow_euclidean':'Finite flow (Euclidean)',
        'flow_mirror':'Finite flow (mirror)','stationary_krylov':'Stationary Krylov',
        'anderson':'Anderson'}
COLORS=['#334155','#2563eb','#7c3aed','#d97706','#059669']
NAMES={'quadratic_mirror':'Quadratic mirror','entropy_dual_affine':'Entropy, affine dual',
       'entropy_simplex':'Entropy simplex','admm_lasso':'ADMM Lasso',
       'entropy_primal_ablation':'Entropy, primal','entropy_boundary_drift':'Boundary drift'}


def get(data,family,method,seed=None):
    return [r for r in data['runs'] if r['family']==family and r['method']==method
            and (seed is None or r['seed']==seed)]


def time_cell(data,family,method):
    rows=get(data,family,method)
    success=[r for r in rows if r['status']=='target']
    if len(success)!=len(rows):return f'-- ({len(success)}/{len(rows)})'
    return f'{1000*statistics.median(r["seconds_median"] for r in rows):.2f}'


def speed(data,family,method):
    rows=get(data,family,method)
    base={r['seed']:r for r in get(data,family,'ordinary')}
    if any(r['status']!='target' or base[r['seed']]['status']!='target' for r in rows):return None
    return [base[r['seed']]['seconds_median']/r['seconds_median'] for r in rows]


def speed_cell(data,family,method):
    values=speed(data,family,method)
    if values is None:
        rows=get(data,family,method)
        return f'-- ({sum(r["status"]=="target" for r in rows)}/{len(rows)})'
    return f'{statistics.median(values):.2f}'


def render(root):
    root=Path(root)
    records=root/'experiments/krylov_bregman/results'
    paper=root/'paper/krylov_bregman'
    figures=paper/'figures';figures.mkdir(parents=True,exist_ok=True)
    small=json.loads((records/'suite_v2/primary_n96.json').read_text())
    large=json.loads((records/'suite_v2/primary_n512.json').read_text())
    shadows={rho:json.loads((records/f'shadow_v2/rho_{rho:g}.json').read_text())
             for rho in [.001,.01,.1,1.,10.]}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#334155',
                         'text.color':'#172033','axes.titleweight':'semibold',
                         'pdf.fonttype':42,'savefig.facecolor':'white'})
    cases=[('quadratic_mirror',0),('entropy_simplex',0),('entropy_simplex',2)]
    fig,axes=plt.subplots(2,3,figsize=(10.6,6.0))
    for col,(family,seed) in enumerate(cases):
        for method,color in zip(METHODS,COLORS):
            row=get(large,family,method,seed)[0]
            trace=[x for x in row['trace'] if x['work']>0]
            y=[max(x['gap_ratio'],1e-12) for x in trace]
            for i,key in enumerate(['work','seconds']):
                x=[t[key]*(1000 if key=='seconds' else 1) for t in trace]
                axes[i,col].loglog(x,y,label=LABELS[method],color=color,lw=1.5,
                                  ls='--' if method=='flow_mirror' else '-')
                axes[i,col].axhline(1e-6,color='#94a3b8',ls=':',lw=.9)
                axes[i,col].grid(True,which='major',alpha=.17)
                axes[i,col].set_ylim(1e-8,1e3)
        axes[0,col].set_title(f'{NAMES[family]} · seed {seed}',fontsize=10)
        axes[0,col].set_xlabel('Map calls + tangent actions')
        axes[1,col].set_xlabel('Elapsed milliseconds')
    axes[0,0].set_ylabel('Objective-gap ratio')
    axes[1,0].set_ylabel('Objective-gap ratio')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.tight_layout(rect=(0,.10,1,1))
    fig.savefig(figures/'pilot_curves.pdf',bbox_inches='tight')
    fig.savefig(figures/'pilot_curves.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(10.6,3.1))
    finite_cases=[(small,'quadratic_mirror','Quadratic mirror'),
                  (small,'entropy_simplex','Entropy simplex'),
                  (shadows[1.],'admm_shadow',r'ADMM shadow, $\rho=1$')]
    for ax,(data,family,title) in zip(axes,finite_cases):
        rows=[r for r in data['finite_screens'][family]
              if r['prefix']==4 and r['depth']==4 and r['metric']=='euclidean']
        rows=sorted(rows,key=lambda x:x['horizon'])
        for key,label,color,style in [
            ('relative_total_error','Total','#334155','-'),
            ('relative_frozen_error','Frozen-geometry error','#d97706','--'),
            ('relative_krylov_error','Krylov compression error','#2563eb',':')]:
            ax.loglog([r['horizon'] for r in rows],[max(r[key],1e-16) for r in rows],
                      marker='o',ms=4,label=label,color=color,ls=style)
        ax.set_title(title,fontsize=10)
        ax.set_xlabel('Predicted horizon')
        ax.set_xticks([4,16,64],labels=['4','16','64'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True,which='major',alpha=.2)
        ax.set_ylim(1e-16,10)
    axes[0].set_ylabel('Relative finite-trajectory error')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04))
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(figures/'error_sources.pdf',bbox_inches='tight')
    fig.savefig(figures/'error_sources.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

    tex=[r'\subsection{Measured outcomes}',
         r'All retained comparisons in this section use protocol v2 on the M4 Mini CPU, '
         r'with one BLAS thread. Each primary family has three seeds and five timing '
         r'repeats per seed; the penalty and shadow screens use three timing repeats. '
         r'The primary budgets are 16,384 work units at $n=96$ and 32,768 at $n=512$. '
         r'The penalty screen uses $n=256$ and 16,384 work units. Here $n$ is the primal '
         r'dimension: the full ADMM state has size $2n$ and simplex dual states have '
         r'size $n-1$.',
         r'\begin{table}[htbp]\centering\small',
         r'\begin{tabular}{lr rrrrr}\toprule',
         r'Geometry & $n$ & Ordinary & Finite E & Finite M & Stationary & Anderson\\\midrule']
    for data,n in [(small,96),(large,512)]:
        for family in data['configuration']['families']:
            tex.append(NAMES[family]+f' & {n} & '+' & '.join(time_cell(data,family,m) for m in METHODS)+r'\\')
        if n==96:tex.append(r'\midrule')
    tex += [r'\bottomrule\end{tabular}',
            r'\caption{Milliseconds to the $10^{-6}$ objective-gap target. Each cell is '
            r'the median across three seeds of each seed\textquotesingle s median runtime. '
            r'E and M denote Euclidean and mirror metrics. A cell with any failure gives '
            r'the success count instead of a success-only timing. These are small synthetic '
            r'experiments, not confidence intervals or a universal performance claim.}',
            r'\label{tab:times}\end{table}']
    eq=statistics.median(speed(large,'quadratic_mirror','flow_euclidean'))
    es=statistics.median(speed(large,'entropy_simplex','flow_euclidean'))
    mq=statistics.median(speed(large,'quadratic_mirror','flow_mirror'))
    ms=statistics.median(speed(large,'entropy_simplex','flow_mirror'))
    tex += [f'At $n=512$, the finite Euclidean-metric schedule gives median elapsed-time '
            f'speedups of {eq:.2f}$\\times$ on quadratic mirrors and {es:.2f}$\\times$ on '
            f'the coupled entropy simplex, relative to ordinary iteration. The corresponding '
            f'mirror-metric speedups are {mq:.2f}$\\times$ and {ms:.2f}$\\times$. '
            r'These gains establish a cross-problem mechanism in this pilot. They do not '
            r'establish an advantage over the strongest comparator: Anderson is substantially '
            r'faster on these primary smooth examples, and the stationary Krylov correction '
            r'is faster when it succeeds. The latter misses the target on entropy seed two '
            r'at both dimensions.',
            r'The finite schedules reduce map-equivalent work by about $2.8\times$ on '
            r'these smooth problems. At $n=96$, overhead eliminates the elapsed-time advantage '
            r'on quadratic mirrors and the cheap separable entropy control. The added mirror '
            r'metric does not improve the work-to-target counts in this primary screen; '
            r'its coordinate covariance and symmetry are structural properties, not a '
            r'demonstrated additional speedup here. With the well-performing baseline ADMM '
            r'penalty $\rho=0.1$, the ordinary method reaches the target too quickly to '
            r'amortize the fixed Krylov schedule.',
            r'\begin{figure}[htbp]\centering',
            r'\includegraphics[width=\textwidth]{figures/pilot_curves.pdf}',
            r'\caption{Representative primary traces at $n=512$. The top row counts maps '
            r'and tangent actions; the bottom row uses elapsed time from the first timed '
            r'repeat. Euclidean and mirror finite-flow work curves nearly coincide. '
            r'Entropy seed two exposes the stationary comparator\textquotesingle s failure. '
            r'Tables use all seeds and repeated timing medians.}',
            r'\label{fig:curves}\end{figure}',
            r'\subsection{Penalty sensitivity and the shadow-coordinate result}',
            r'The penalty screen keeps the acceleration schedule fixed. For $\rho=1$, '
            r'both finite and stationary Krylov schedules fail on all three seeds while '
            r'ordinary ADMM and Anderson reach the target. At $\rho=10$, finite flow '
            r'succeeds on only one seed. The exact shadow reparameterization preserves '
            r'ordinary ADMM numerically and reduces state dimension, but does not resolve '
            r'these failures. Thus an off-graph extrapolated memory state is not a sufficient '
            r'explanation: the frozen geometric prediction itself remains inadequate.',
            r'\begin{table}[htbp]\centering\small',
            r'\begin{tabular}{r rrrrr}\toprule',
            r'$\rho$ & Finite, full & Finite, shadow & Stationary, full & Stationary, shadow & AA, shadow\\\midrule']
    for rho,data in shadows.items():
        tex.append(f'{rho:g} & '+' & '.join([
            speed_cell(data,'admm_lasso','flow_euclidean'),
            speed_cell(data,'admm_shadow','flow_euclidean'),
            speed_cell(data,'admm_lasso','stationary_krylov'),
            speed_cell(data,'admm_shadow','stationary_krylov'),
            speed_cell(data,'admm_shadow','anderson')])+r'\\')
    tex += [r'\bottomrule\end{tabular}',
            r'\caption{Median elapsed-time speedup relative to ordinary iteration in '
            r'the same representation, at $n=256$. Values above one are faster. '
            r'Any unsuccessful seed replaces the ratio by its success count. The full '
            r'and shadow representations have the same ordinary trajectory and target.}',
            r'\label{tab:penalty}\end{table}',
            r'\begin{figure}[htbp]\centering',
            r'\includegraphics[width=\textwidth]{figures/error_sources.pdf}',
            r'\caption{Independent one-block diagnostics at ordinary prefix four, '
            r'Krylov depth four, seed zero. Errors are normalized by the actual finite '
            r'displacement; curves at $10^{-16}$ represent the plotting floor. '
            r'The affine quadratic primarily exposes compression error. Nonlinear '
            r'and active-mask geometry can dominate in the other examples. All reference '
            r'map calls and full tangent recurrences used for this figure are separately '
            r'charged diagnostics, excluded from accelerated solve timings.}',
            r'\label{fig:defects}\end{figure}',
            r'The boundary-drift control has an exact rank-one finite flow and a '
            r'singular stationary projected system. Its horizon-sixteen Python '
            r'implementation still does not beat the extremely cheap ordinary update '
            r'in elapsed time. This makes the distinction between an exact transport '
            r'representation and a practical acceleration particularly clear.']
    (paper/'results.tex').write_text('\n\n'.join(tex)+'\n')
    summary={'quadratic_n512_euclidean_speedups':speed(large,'quadratic_mirror','flow_euclidean'),
             'simplex_n512_euclidean_speedups':speed(large,'entropy_simplex','flow_euclidean'),
             'quadratic_n512_mirror_speedups':speed(large,'quadratic_mirror','flow_mirror'),
             'simplex_n512_mirror_speedups':speed(large,'entropy_simplex','flow_mirror')}
    (records/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    render(p.parse_args().root)
