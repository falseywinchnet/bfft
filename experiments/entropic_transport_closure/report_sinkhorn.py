"""Render retained measurements locally; does not rerun the solvers."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np


METHODS = ['linear', 'quadratic', 'quadratic2']
LABELS = {'linear': 'Frozen rule, 4 directions', 'quadratic': 'Changing rule, 4 directions',
          'quadratic2': 'Changing rule, 2 directions'}
COLORS = {'linear': '#426baf', 'quadratic': '#b77635', 'quadratic2': '#287d72'}


def summarize(root):
    general, blocks = [], []
    for path in sorted(root.glob('entropic_sinkhorn_final*.json')):
        data = json.loads(path.read_text())
        general += data['general']
        blocks += data['blocks']
    if not general:
        raise ValueError('No retained final measurements found.')
    grouped = {}
    for row in general:
        key = (row['size'], row['case'], row['seed'], row['method'])
        grouped.setdefault(key, []).append(row)
    medians = {key: statistics.median(r['seconds'] for r in rows)
               for key, rows in grouped.items()}
    table = []
    for size, case in sorted(set((r['size'], r['case']) for r in general)):
        seeds = sorted(set(r['seed'] for r in general if r['size'] == size and r['case'] == case))
        out = dict(size=size, case=case,
                   ordinary_ms=1000*statistics.median(medians[size, case, s, 'ordinary'] for s in seeds))
        for method in METHODS:
            ratios = [medians[size, case, s, 'ordinary']/medians[size, case, s, method] for s in seeds]
            selected = [r for s in seeds for r in grouped[size, case, s, method]]
            out[method] = dict(speedup=statistics.median(ratios),
                speedup_seed_min=min(ratios), speedup_seed_max=max(ratios),
                milliseconds=1000*statistics.median(medians[size, case, s, method] for s in seeds),
                evaluations=statistics.median(r['evaluations'] for r in selected),
                acquisitions=statistics.median(r['acquisitions'] for r in selected),
                accepted=statistics.median(r['accepted'] for r in selected),
                kernel_vector_products=statistics.median(r['kernel_vector_products'] for r in selected))
        table.append(out)
    block_table = []
    for size, coupling in sorted(set((r['size'], r['coupling']) for r in blocks)):
        times = {method: statistics.median(r['seconds'] for r in blocks
                    if r['size'] == size and r['coupling'] == coupling and r['method'] == method)
                 for method in ['ordinary', 'power']}
        block_table.append(dict(size=size, coupling=coupling, speedup=times['ordinary']/times['power'],
                                ordinary_ms=1000*times['ordinary'], power_ms=1000*times['power']))
    probes = json.loads((root/'models.json').read_text())['records']
    pairs = {}
    for row in probes:
        key = tuple(row[k] for k in ['case', 'seed', 'anchor', 'horizon'])
        pairs.setdefault(key, {})[row['method']] = row
    paired = [p for p in pairs.values()
              if len(p) == 2 and all(r['finite_rollout'] for r in p.values())]
    model_summary = dict(paired=len(paired), total_pairs=len(pairs),
        state_improvements=sum(p['quadratic']['state_error'] < p['linear']['state_error'] for p in paired),
        rule_improvements=sum(p['quadratic']['rule_error'] < p['linear']['rule_error'] for p in paired),
        failed_bounds=sum(not r.get('bound_holds', True) for r in probes),
        nonfinite_or_incomplete=sum(not r['finite_rollout'] for r in probes))
    summary = dict(general_runs=len(general), all_general_converged=all(r['converged'] for r in general),
                   worst_general_residual=max(r['residual'] for r in general),
                   block_runs=len(blocks), all_blocks_converged=all(r['converged'] for r in blocks),
                   table=table, blocks=block_table, models=model_summary)
    return summary, paired


def render(root, summary, paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    max_size = max(r['size'] for r in summary['table'])
    rows = [r for r in summary['table'] if r['size'] == max_size and r['case'] != 'positive_dense']
    x = np.arange(len(rows))
    ax = axes[0, 0]
    for j, method in enumerate(METHODS):
        vals = [r[method]['speedup'] for r in rows]
        ax.bar(x+(j-1)*.24, vals, width=.22, label=LABELS[method], color=COLORS[method])
    ax.axhline(1, color='#444444', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r['case'].replace('_eps_', '\nε=') for r in rows])
    ax.set_ylabel('Ordinary time / method time; higher is faster')
    ax.set_title(f'General kernels: {max_size} × {max_size}, actual marginal tolerance 10⁻⁹')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    sizes = sorted(set(r['size'] for r in summary['table']))
    hard = ['line_eps_0.003', 'plane_eps_0.006', 'plane_eps_0.003']
    for method in METHODS:
        vals = [np.exp(np.mean([np.log(r[method]['speedup']) for r in summary['table']
                    if r['size'] == size and r['case'] in hard])) for size in sizes]
        ax.plot(sizes, vals, 'o-', label=LABELS[method], color=COLORS[method])
    ax.axhline(1, color='#444444', linestyle='--', linewidth=1)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes); ax.set_xticklabels(sizes)
    ax.set_xlabel('Points per marginal')
    ax.set_ylabel('Geometric mean speedup on three narrow-kernel cases')
    ax.set_title('Larger kernels can amortize acquisition and rollout')
    ax = axes[1, 0]
    for metric, label, color in [('state_error', 'State prediction', '#426baf'),
                                  ('rule_error', 'Derivative prediction', '#287d72')]:
        xx = [max(p['linear'][metric], 1e-16) for p in paired]
        yy = [max(p['quadratic'][metric], 1e-16) for p in paired]
        ax.scatter(xx, yy, s=18, alpha=.7, label=label, color=color)
    limits = [1e-12, 1e4]
    ax.plot(limits, limits, '--', color='#777777', linewidth=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Frozen-rule error')
    ax.set_ylabel('Changing-rule error; below diagonal is better')
    ax.set_title('Matched four-direction model probes, outside timed runs')
    ax.legend(fontsize=8)
    ax.text(.98, .04, f"{summary['models']['paired']} completed pairs; "
            f"{summary['models']['nonfinite_or_incomplete']} rollouts did not complete.\n"
            'Unrestricted diagnostics, not accepted solver steps.',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    ax = axes[1, 1]
    for coupling, color in [(.03, '#426baf'), (.001, '#287d72')]:
        rows = [r for r in summary['blocks'] if r['coupling'] == coupling]
        ax.plot([r['size'] for r in rows], [r['speedup'] for r in rows], 'o-',
                label=f'Block coupling {coupling}', color=color)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Original kernel dimension')
    ax.set_ylabel('Reduced ordinary time / exact projective-power time')
    ax.set_title('Exact block family: both methods use the same reduction')
    ax.legend(fontsize=8)
    fig.suptitle('Applying evolving transport closure to Sinkhorn\n'
                 'M4 CPU · five shuffled repetitions per seed · setup and checks included', fontsize=14)
    fig.savefig(root/'application.png', dpi=160)
    fig.savefig(root/'application.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary, paired = summarize(args.root)
    (args.root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    render(args.root, summary, paired)
    print(json.dumps({k:v for k,v in summary.items() if k != 'table'}, indent=2))
    for row in summary['table']:
        print(row['size'], row['case'], round(row['ordinary_ms'], 3),
              ' '.join(f'{m}={row[m]["speedup"]:.3f}x' for m in METHODS))
