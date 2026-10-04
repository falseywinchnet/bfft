"""Retained curved-Meyer certificate coverage and honest cost report."""
from pathlib import Path
import json,statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RECORDS=ROOT/'experiments/krylov_bregman/results';PAPER=ROOT/'paper/krylov_bregman'
NAMES={'camera':'Camera','camera_permuted':'Permuted camera','barbara':'Barbara','straight_carrier':'Straight carrier','crossing':'Crossing'}


def main():
    coverage=json.loads((RECORDS/'curved_meyer_v1.json').read_text())
    local=json.loads((RECORDS/'curved_discovery_v1.json').read_text())
    solves=json.loads((RECORDS/'curved_solve_v1.json').read_text())
    accepted=[r for r in local['rows'] if r['accepted']]
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.1))
    for name,color in zip(NAMES,['#2563eb','#64748b','#a855f7','#059669','#d97706']):
        rows=[r for r in coverage['rows'] if r['scene']==name and r['depth']==4]
        axes[0].loglog([max(r['actual_error'],1e-12) for r in rows],[max(r['bound'],1e-12) for r in rows],'.',color=color,label=NAMES[name],ms=5)
    axes[0].loglog([1e-10,1e5],[1e-10,1e5],ls=':',color='#999',lw=1)
    axes[0].set_xlabel('Actual quotient error');axes[0].set_ylabel('Proved upper bound')
    axes[0].set_title('Curved path certificate, depth 4',fontsize=10)
    axes[0].legend(fontsize=7,frameon=False);axes[0].grid(alpha=.15)
    labels=[NAMES[r['scene']]+' / '+str(r['prefix']) for r in accepted]
    speeds=[r['seconds_ordinary_horizon_settle']/r['seconds_discovery_settle'] for r in accepted]
    axes[1].barh(labels,speeds,color=['#008b8b' if v>1 else '#94a3b8' for v in speeds])
    axes[1].invert_yaxis();axes[1].axvline(1,ls=':',color='#444');axes[1].set_xlabel('Elapsed-time speedup over replay')
    axes[1].set_title('Accepted blocks, including discovery',fontsize=10)
    axes[1].tick_params(axis='y',labelsize=8)
    for i,v in enumerate(speeds):axes[1].text(v+.04,i,f'{v:.2f}',va='center',fontsize=8)
    axes[1].set_xlim(0,4.1)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(PAPER/'figures/curved_validity.pdf',bbox_inches='tight')
    fig.savefig(PAPER/'figures/curved_validity.png',dpi=170,bbox_inches='tight');plt.close(fig)
    tex=[r'\subsection{Coverage, local amortization, and complete solves}',
         r'The retained coverage battery uses camera, Barbara, a histogram-preserving '
         r'camera permutation, a straight carrier, and crossing carriers with an edge, '
         r'all at $64\times64$. Ordinary anchors are taken after $4,32,128,512$ '
         r'passes, with depths $2,4,8$ and horizons $8,16,32,64$. All 240 measured '
         r'quotient errors satisfy \eqref{eq:curvedpath} within numerical tolerance. '
         r'The texture errors after settling are independently checked as well. '
         r'The certificate is an analytic bound; these finite tests verify its '
         r'implementation rather than establish the theorem.',
         r'For example, at camera prefix $512$, depth two and horizon sixteen, '
         r'the bound is $0.0954$ times predicted displacement and the actual error '
         r'is $0.019$ times actual displacement. At camera prefix four the same '
         r'horizon has a much larger curvature budget. Increasing depth can '
         r'reduce compression without removing that changing geometry.',
         r'The live discovery policy acquires depths $2,4,8$ incrementally and '
         r'checks the reduced path in chunks of eight up to horizon $64$. It '
         r'stops a depth after the first chunk whose accumulated bound exceeds '
         r'$0.1$ times predicted displacement, then selects a checked horizon '
         r'whose nominal operator savings exceed $1.5\times$ acquisition and '
         r'settling work. This early stopping rule need not find the longest '
         r'admissible horizon. It does not consult a future trajectory or gap oracle.',
         r'The policy accepts seven of the twenty anchors. With five timing '
         r'repeats, the late camera block discovers depth two/horizon sixteen '
         r'and is $1.39\times$ faster than ordinary replay. The straight carrier '
         r'at prefixes $128$ and $512$ discovers depth two/horizon sixty-four '
         r'and gives $3.47\times$ and $3.48\times$ speedups. Three of the seven '
         r'accepted blocks are slower despite passing the nominal work screen. '
         r'Timings include chart acquisition, radial checks, reconstruction, '
         r'and one settling pass for both methods.',
         r'\begin{figure}[htbp]\centering',
         r'\includegraphics[width=\textwidth]{figures/curved_validity.pdf}',
         r'\caption{Left: independently replayed errors and the curved certificate '
         r'for depth four, across all anchors and horizons. The dotted diagonal '
         r'is equality; plotting floors are $10^{-12}$. Right: all seven accepted '
         r'live-discovery blocks, labeled by source and ordinary anchor prefix. '
         r'Thirteen rejected anchors remain in the raw records.}',r'\end{figure}',
         r'For complete solves, all methods use the same original recurrence '
         r'and the same recovered primal--dual gap. The target is $10^{-4}$ '
         r'times the initial gap, checked every thirty-two work units. The '
         r'fixed methods use depth four, horizon sixteen, and one settling '
         r'pass; they differ only in six-field Euclidean versus whitened '
         r'quotient coordinates. The discovered method takes thirty-two '
         r'ordinary steps after an unprofitable acquisition. Its fallback '
         r'also charges an additional pass to recover the full primal output. '
         r'The work ceiling is 4096; all methods reach the target on every source.',
         r'\begin{table}[htbp]\centering\small',
         r'\begin{tabular}{l rrrr}\toprule',
         r'Source & Ordinary & Fixed, six fields & Fixed, quotient & Discovered\\\midrule']
    methods=['ordinary','full_fixed','quotient_fixed','curved_discovered'];summary={}
    for name,label in NAMES.items():
        times={m:st.median(r['seconds'] for r in solves['runs'] if r['scene']==name and r['method']==m) for m in methods}
        summary[name]=times
        tex.append(label+' & '+' & '.join(f'{1000*times[m]:.1f}' for m in methods)+r'\\')
    tex += [r'\bottomrule\end{tabular}',
            r'\caption{Complete Python solve milliseconds on the M4 Mini, one '
            r'BLAS thread, median of three shuffled timed runs after warm-up. '
            r'Setup is excluded; gap checks, discovery, rejected candidates, '
            r'and all update work are included. These are not native-kernel timings.}',r'\label{tab:curvedsolve}\end{table}',
            r'The conservative discovered policy loses to ordinary iteration '
            r'on all five complete solves, despite profitable certified blocks '
            r'in some later regimes. This is a retained negative result. A '
            r'valid curved description is now available for the original map; '
            r'the current acquisition policy and sufficient path-error bound '
            r'are not yet an economical general accelerator. As in the earlier '
            r'Meyer probe, successful acceleration need not closely reproduce '
            r'the entire ordinary trajectory. The strict certificate is a '
            r'controlled research instrument, not a replacement imposed on '
            r'the established native operator.',
            r'The continuation suite passes 41 tests, including the exact '
            r'future-driving quotient, the stronger averaged-map inequality, '
            r'curvature formulas across boundary events, sharp attainment of '
            r'the Fourier response gain, and independently replayed path bounds.']
    (PAPER/'curved_results.tex').write_text('\n\n'.join(tex)+'\n')
    (RECORDS/'curved_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
