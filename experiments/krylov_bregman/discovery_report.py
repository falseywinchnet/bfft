"""Render retained discovery/certification evidence, without rerunning solvers."""
from pathlib import Path
import json
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RECORDS=ROOT/'experiments/krylov_bregman/results'
PAPER=ROOT/'paper/krylov_bregman'
METHODS=['ordinary','fixed','certified','anderson']
LABELS=['Ordinary','Fixed depth 4 / horizon 16','Discovered, piece-certified','Anderson depth 4']
COLORS=['#334155','#7c3aed','#007f86','#c66a14']


def summaries(data,kind,method):
    return [st.median(r['seconds'] for r in data['runs'] if r['kind']==kind and r['method']==method and r['seed']==s) for s in range(data['configuration']['seeds'])]


def main():
    small=json.loads((RECORDS/'certified_huber_v2.json').read_text())
    large=json.loads((RECORDS/'certified_huber_v2_n256.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(9.4,6.2))
    for col,kind in enumerate(['positive','signed']):
        for method,label,color in zip(METHODS,LABELS,COLORS):
            row=next(r for r in large['runs'] if r['kind']==kind and r['seed']==0 and r['repeat']==0 and r['method']==method)
            for i,key in enumerate(['work','seconds']):
                axes[i,col].loglog([r[key] for r in row['trace']],[r['gap_ratio'] for r in row['trace']],label=label,color=color,lw=1.5)
                axes[i,col].axhline(1e-3,color='#aaa',ls=':',lw=.8)
                axes[i,col].grid(alpha=.2);axes[i,col].spines[['top','right']].set_visible(False)
        axes[0,col].set_title(kind.capitalize()+' sensing, n=256',fontsize=11)
        axes[0,col].set_xlabel('Map calls + tangent actions')
        axes[1,col].set_xlabel('Elapsed seconds')
    axes[0,0].set_ylabel('Objective-gap ratio');axes[1,0].set_ylabel('Objective-gap ratio')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='lower center',frameon=False)
    fig.tight_layout(rect=(0,.09,1,1))
    fig.savefig(PAPER/'figures/discovery_costs.pdf',bbox_inches='tight')
    fig.savefig(PAPER/'figures/discovery_costs.png',dpi=170,bbox_inches='tight');plt.close(fig)
    tex=[r'\subsection{Camera-derived inverse problems and complete cost}',
         r'The Huber study uses a camera image resized to $8\times8$ or $16\times16$, '
         r'with $n=64$ or $256$ unknowns and $4n$ synthetic measurements. '
         r'The sensing matrix has independent Gaussian entries scaled by $1/\sqrt{4n}$; '
         r'the positive variant takes their absolute values. The mirror diagonal spans '
         r'$[1,16]$. Observations are noiseless, $b=Ax_*$, so the minimum objective is '
         r'zero. These are controlled camera-derived inverse problems, not a benchmark '
         r'of naturally corrupted measurements. Both sensing variants are retained.',
         r'The first positive-sensing blocks at $n=64$ discover horizons $392$, $366$, '
         r'and $386$ on the three seeds. Each uses one map and one tangent action. '
         r'Independent replay disagrees by at most $8.6\times10^{-12}$ in absolute '
         r'whitened dual norm. At $n=256$, the corresponding discovered horizons are '
         r'$838$, $831$, and $821$, with maximum absolute replay error '
         r'$3.7\times10^{-11}$. These are exact-arithmetic drift skips verified '
         r'numerically; the long horizon is determined by the first feature event.',
         r'For the complete solve, the certified method tries depths $2,4,8$, caps '
         r'horizons at $4096$, and requires the bound in \eqref{eq:piecebound} to be '
         r'at most $0.02\norm{Qc_m}$. It takes sixteen ordinary steps after an '
         r'unprofitable acquisition. The target is an objective-gap ratio of $10^{-3}$, '
         r'checked every sixteen work units. Three seeds and three shuffled timing '
         r'repeats per seed are used on the M4 Mini with one BLAS thread. '
         r'Timings include discovery, boundary checks, unsuccessful attempts, '
         r'and objective checks, but exclude matrix setup. All methods reach the '
         r'target on all seeds in this study.',
         r'\begin{table}[htbp]\centering\small',
         r'\begin{tabular}{lr rrrr}\toprule',
         r'Sensing & $n$ & Ordinary & Fixed & Discovered & Anderson\\\midrule']
    output={}
    for data,n in [(small,64),(large,256)]:
        for kind in ['positive','signed']:
            med={m:st.median(summaries(data,kind,m)) for m in METHODS}
            base=summaries(data,kind,'ordinary');cert=summaries(data,kind,'certified');fixed=summaries(data,kind,'fixed')
            output[f'{kind}_{n}']={'milliseconds':{m:1000*v for m,v in med.items()},
               'paired_certified_speedups_vs_ordinary':[b/c for b,c in zip(base,cert)],
               'paired_certified_speedups_vs_fixed':[b/c for b,c in zip(fixed,cert)]}
            tex.append(kind.capitalize()+f' & {n} & '+' & '.join(f'{1000*med[m]:.2f}' for m in METHODS)+r'\\')
    speed=st.median(output['positive_256']['paired_certified_speedups_vs_ordinary'])
    fixedspeed=st.median(output['positive_256']['paired_certified_speedups_vs_fixed'])
    tex += [r'\bottomrule\end{tabular}',
            r'\caption{Complete solve milliseconds, median of the per-seed timing '
            r'medians. Fixed means depth four/horizon sixteen with one settling '
            r'step. Discovered means the affine-piece certificate.}',r'\label{tab:discovery}\end{table}',
            f'At $n=256$ with positive sensing, the discovered method is {speed:.2f}$\\times$ '
            f'faster than ordinary iteration and {fixedspeed:.2f}$\\times$ faster than '
            r'the fixed finite schedule, using paired per-seed median time ratios. '
            r'At $n=64$ its acquisition overhead prevents an overall speedup. On signed '
            r'sensing, frequent piece events make this certificate costly and the '
            r'discovered method is slower at both sizes. Anderson wins the complete '
            r'solves here after the initial constant-drift phase ends. The proved '
            r'local drift obstruction therefore does not imply an end-to-end ranking.',
            r'The full solve also exposes a stability distinction hidden by final '
            r'runtime alone. On positive-sensing seeds zero and one at $n=256$, '
            r'Anderson reaches recorded objective gaps $25.26$ and $7.35$ times '
            r'the initial gap before recovering. The certified transport has no '
            r'comparable excursion at the recorded checkpoints. Checks occur every '
            r'sixteen work units, so this is not a bound on all intermediate gaps '
            r'or a claim of monotonicity for approximate certified proposals.',
            r'Measurements from the initial scalar implementation of the reduced-path checks are retained '
            r'as protocol v1. Protocol v2 uses block doubling for $c_j$ and batched '
            r'feature tests without changing mathematical admission conditions. '
            r'Both raw records remain available; v2 is used in Table~\ref{tab:discovery}.',
            r'\begin{figure}[htbp]\centering',
            r'\includegraphics[width=\textwidth]{figures/discovery_costs.pdf}',
            r'\caption{Huber mirror descent at $n=256$, seed zero, first timed repeat. '
            r'The same camera source and target tolerance are used for positive '
            r'and signed sensing. A discovered valid polynomial can save work and '
            r'time, while frequent boundary changes can eliminate that benefit. '
            r'The timing table retains all seeds and repeats.}',r'\end{figure}',
            r'This extension establishes a discoverable, checkable finite transport '
            r'construction for a nontrivial convex mirror class. The central unresolved '
            r'problem is to discover comparably economical validity descriptions for '
            r'curved proximal geometry and changing observable spaces, including the '
            r'original coupled Meyer iteration.']
    (PAPER/'discovery_results.tex').write_text('\n\n'.join(tex)+'\n')
    (RECORDS/'discovery_summary.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
