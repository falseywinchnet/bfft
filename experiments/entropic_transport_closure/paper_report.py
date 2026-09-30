"""Regenerate manuscript table and figures from the retained receipts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .report_sinkhorn import summarize, METHODS, LABELS, COLORS


def main():
    repo = Path(__file__).resolve().parents[2]
    data = repo/'experiments/entropic_transport_closure/results'
    paper = repo/'paper/krylov_bregman'
    figures = paper/'figures'
    summary, paired = summarize(data)
    assert summary['all_general_converged'] and summary['all_blocks_converged']
    assert summary['models']['failed_bounds'] == 0
    rows = [r for r in summary['table'] if r['size'] == 2048 and r['case'] != 'positive_dense']
    order = ['line_eps_0.02', 'line_eps_0.003', 'plane_eps_0.03', 'plane_eps_0.006', 'plane_eps_0.003']
    rows.sort(key=lambda r: order.index(r['case']))
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
    x = np.arange(len(rows))
    for j, method in enumerate(METHODS):
        label = {'linear':'Frozen, k=4', 'quadratic':'Changing, k=4',
                 'quadratic2':'Changing, k=2'}[method]
        ax.bar(x+(j-1)*.24, [r[method]['speedup'] for r in rows], .22,
               label=label, color=COLORS[method])
    ax.axhline(1, color='.3', linestyle='--', linewidth=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r['case'].replace('_eps_', '\nε=') for r in rows])
    ax.set_ylabel('Ordinary time / method time')
    ax.set_ylim(0, 1.75)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01),
              ncol=3, fontsize=7, frameon=False)
    for suffix in ['pdf', 'png']:
        fig.savefig(figures/f'entropic_cost.{suffix}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), constrained_layout=True)
    ax = axes[0]
    for metric, label, color in [('state_error','State','#426baf'), ('rule_error','Derivative','#287d72')]:
        ax.scatter([max(p['linear'][metric],1e-16) for p in paired],
                   [max(p['quadratic'][metric],1e-16) for p in paired],
                   s=9, alpha=.7, color=color, label=label)
    ax.plot([1e-13,1e5],[1e-13,1e5],'--',color='.5',linewidth=.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Frozen-rule error'); ax.set_ylabel('Changing-rule error')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('(a) Matched model diagnostics',fontsize=8)
    ax = axes[1]
    for coupling, color in [(.03,'#426baf'),(.001,'#287d72')]:
        b = [r for r in summary['blocks'] if r['coupling'] == coupling]
        ax.plot([r['size'] for r in b], [r['speedup'] for r in b], 'o-',
                color=color, markersize=3, label=f'Coupling {coupling}')
    ax.set_xscale('log',base=2)
    ax.set_xlabel('Original kernel dimension')
    ax.set_ylabel('Reduced ordinary time / power time')
    ax.set_title('(b) Exact proportional-block family',fontsize=8)
    ax.legend(fontsize=7,frameon=False)
    for suffix in ['pdf', 'png']:
        fig.savefig(figures/f'entropic_closure.{suffix}',dpi=180)
    plt.close(fig)

    lines = [r'\begin{table}[!hbp]\centering\small',
        r'\begin{tabular}{lrrrr}\toprule',
        r'Problem & Ordinary & Frozen 4 & Changing 4 & Changing 2\\\midrule']
    for row in rows:
        kind, eps = row['case'].split('_eps_')
        vals = [row['ordinary_ms']]+[row[m]['milliseconds'] for m in METHODS]
        lines.append(f'{kind.capitalize()}, $\\varepsilon={eps}$ & '+' & '.join(f'{v:.2f}' for v in vals)+r'\\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Complete solve times in milliseconds at $n=2048$, reported as',
        r'the median of per-seed five-repeat medians. The matched four-direction',
        r'comparison isolates the added changing-rule model; the two-direction',
        r'version also changes representation capacity.}\label{tab:entropic-cost}',
        r'\end{table}']
    (paper/'entropic_table.tex').write_text('\n'.join(lines)+'\n')
    (data/'paper_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Generated manuscript table and two figures from retained receipts.')


if __name__ == '__main__':
    main()
