"""Read-only statistical analysis and scientific plots of saved measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .study import METHODS, save
from .data import FAMILIES
from .geometry import rotate_and_integrate


LABELS = {'cv': 'CV Kalman', 'ca': 'CA Kalman', 'imm': 'IMM',
          'robust_imm': 'Robust IMM', 'turn_ukf': '3-D turn UKF', 'geometric': 'Geometric transport'}
COLORS = ['#9aa5ab', '#bc9476', '#687ca5', '#8d71ac', '#db8748', '#087f70']


def paired(rows):
    groups = {}
    for row in rows:
        key = (row['family'], row['corrupt'], row['seed'])
        groups.setdefault(key, {})[row['method']] = row
    keys = sorted(groups)
    strata = [[j for j, key in enumerate(keys) if key[:2] == (family, corrupt)]
              for family in FAMILIES for corrupt in (False, True)]
    rng = np.random.default_rng(4209)
    indices = np.concatenate([rng.choice(group, (4000, len(group)), replace=True) for group in strata], axis=1)
    result = {}
    for method in METHODS[:-1]:
        own = np.array([groups[k]['geometric']['forecast2_sq'] for k in keys])
        other = np.array([groups[k][method]['forecast2_sq'] for k in keys])
        delta = 100*(1-np.sqrt(own[indices].mean(axis=1)/other[indices].mean(axis=1)))
        result[method] = {'forecast_rmse_reduction_percent': float(100*(1-np.sqrt(own.mean()/other.mean()))),
                          'stratified_bootstrap_95': np.quantile(delta, [.025, .975]).tolist(),
                          'paired_case_wins': int(np.sum(own < other)), 'cases': len(keys)}
    return result


def report(directory):
    directory = Path(directory)
    payload = json.loads((directory/'results.json').read_text())
    comparisons = paired(payload['runs'])
    save(directory/'paired_comparisons.json', comparisons)
    summary = payload['overall']
    geo_rows = [r for r in payload['runs'] if r['method'] == 'geometric']
    ablations = {}
    for variant in ('uncoupled', 'straight'):
        ablations[variant] = {
            'forecast2_rmse': float(np.sqrt(np.mean([r['ablations'][variant]['forecast2_sq'] for r in geo_rows]))),
            'energy2': float(np.mean([r['ablations'][variant]['energy2'] for r in geo_rows])),
            'boundary_crossing_brier': float(np.mean([r['ablations'][variant]['boundary_crossing_brier'] for r in geo_rows]))}
    save(directory/'ablations.json', ablations)
    lines = ['# Position-only civilian tracking: measured findings', '',
             f'The fixed test has {len(geo_rows)} independent synthetic trajectories: '
             f'{payload["metadata"]["seeds_per_family_noise"]} seeds for each of five motion families '
             'under clean and corrupted observations. Configuration selection used ten separate development cases.', '',
             'The approximately two-second forecast uses only past positions and requested future timestamps. '
             'Numbers below are vector position RMSE in meters; lower scores are better. '
             'No claims of a universal SOTA result or real-airspace validation follow from this battery.', '',
             '| Method | Tracking RMSE | ~1 s forecast | ~2 s forecast | Energy score (~2 s) | Boundary crossing Brier |',
             '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        row = summary[m]
        lines.append(f'| {LABELS[m]} | {row["track_rmse"]:.3f} | {row["forecast1_rmse"]:.3f} | {row["forecast2_rmse"]:.3f} | {row["energy2"]:.3f} | {row["boundary_crossing_brier"]:.4f} |')
    lines += ['', '## Paired forecast comparison', '',
              'Positive percentage means lower RMSE for geometric transport. '
              'Intervals resample whole trajectories within each family/noise stratum (4,000 paired bootstrap draws). '
              'They describe this synthetic case distribution, conditional on the frozen development choice.', '',
              '| Comparator | Geometric RMSE reduction | 95% interval | Cases won |',
              '|---|---:|---:|---:|']
    for m, c in comparisons.items():
        lo, hi = c['stratified_bootstrap_95']
        lines.append(f'| {LABELS[m]} | {c["forecast_rmse_reduction_percent"]:.1f}% | [{lo:.1f}%, {hi:.1f}%] | {c["paired_case_wins"]}/{c["cases"]} |')
    lines += ['', '## Posterior-coupling and straightening ablations', '',
              'These start from the same geometric filtered posterior; only future propagation is changed.', '',
              '| Forecast law | ~2 s RMSE | Energy score | Crossing Brier |', '|---|---:|---:|---:|']
    for label, row in [('Intact joint posterior', summary['geometric'])]+list(ablations.items()):
        lines.append(f'| {label} | {row["forecast2_rmse"]:.3f} | {row["energy2"]:.3f} | {row["boundary_crossing_brier"]:.4f} |')
    lines += ['', '## Calibration, components, and cost', '',
              'Containment is empirical coverage of a nominal 95% moment-Gaussian ellipsoid, '
              'not an exact mixture credible set. Bias/noise errors test attribution separately from tracking. '
              'Update times include current-state diagnostics; forecast times include 256 sampled paths '
              'over twenty irregular intervals and exclude the ablation calls. Timings are single-run Python '
              'measurements on the M4, not production-kernel or isolated-host speed guarantees.', '',
              '| Method | Tracking coverage | Forecast coverage | Bias RMSE | Noise RMSE | Update ms | Forecast ms |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        r = summary[m]
        lines.append(f'| {LABELS[m]} | {100*r["track_95"]:.1f}% | {100*r["forecast2_95"]:.1f}% | {r["bias_rmse"]:.3f} | {r["noise_rmse"]:.3f} | {r["update_ms"]:.2f} | {r["forecast_ms"]:.2f} |')
    lines += ['', '## Breakdown by motion family and noise', '',
              '| Scenario | CV | CA | IMM | Robust IMM | Turn UKF | Geometric |', '|---|---:|---:|---:|---:|---:|---:|']
    for key, values in payload['by_case'].items():
        lines.append('| '+key+' | '+' | '.join(f'{values[m]["forecast2_rmse"]:.3f}' for m in METHODS)+' |')
    lines += ['', '## Scope of the establishment', '',
              'Eleven structural tests pass: exact finite-map composition, rotation equivariance, speed '
              'preservation, the zero-turn limit, the mean-generator boundary counterexample, positive '
              'process covariance, independent Gaussian conditioning, posterior signal/noise mean accounting, '
              'missing-observation handling, forecast isolation from filtering/RNG, and all-method smoke checks.', '',
              'The inferred object is a distribution over a restricted continuous family of geometric '
              'continuation laws. The resistance functional itself is prescribed up to development-selected '
              'diffusivities. This is not unrestricted discovery of the geometry from one trajectory. '
              'Possible-history reduction is Monte Carlo resampling, so exact posterior support is not guaranteed. '
              'Independent future waypoint commands remain unavailable to all methods.', '',
              'The baseline implementations and their limits are specified in THEORY.md. '
              'The test compares standard practical filter families plus a continuous-turn UKF; '
              'it does not reproduce every contemporary arXiv method or claim a universal leaderboard.', '']
    filename = 'FINDINGS.md' if directory.name == 'results' else 'CONFIRMATION.md'
    (directory.parent/filename).write_text('\n'.join(lines))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    fig.suptitle(f'Civilian 3-D tracking from noisy positions only · {len(geo_rows)} held-out tracks', fontsize=18)
    for ax, key, title, ylabel in zip(axes.flat,
        ['track_rmse', 'forecast2_rmse', 'energy2', 'boundary_crossing_brier'],
        ['Current position', 'Forecast without future measurements', 'Predictive distribution', 'Civilian boundary crossing'],
        ['Tracking RMSE (m)', 'Approx. 2 s forecast RMSE (m)', 'Energy score (lower is better)', 'Brier score (lower is better)']):
        values = [summary[m][key] for m in METHODS]
        bars = ax.bar(np.arange(6), values, color=COLORS)
        ax.set_xticks(np.arange(6), ['CV', 'CA', 'IMM', 'Robust\nIMM', 'Turn\nUKF', 'Geometric'])
        ax.set(title=title, ylabel=ylabel)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.set_ylim(0, max(values)*1.19)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(directory/'comparison.png', dpi=145)
    plt.close(fig)

    # Mathematical counterexample, independent of benchmark success or failure.
    t = np.linspace(0, 2, 160)
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    for sign, color in [(-1, '#537ba5'), (1, '#087f70')]:
        points = np.array([rotate_and_integrate([1, 0, 0], [0, 0, sign], h)[1] for h in t])
        ax.plot(points[:, 0], points[:, 1], color=color, lw=2.5, label='Supported '+('left' if sign > 0 else 'right')+' turn')
    ax.plot(t, np.zeros_like(t), '--', color='#d87745', lw=2, label='Propagate average generator')
    ax.axvline(1.5, color='#754c5d', label='Boundary: x = 1.5')
    ax.plot(np.sin(t), np.zeros_like(t), ':', color='#263746', lw=3, label='Mean of actual transported paths')
    ax.set(title='Averaging the transport can invent a boundary crossing', xlabel='x', ylabel='y', aspect='equal')
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, .5))
    ax.grid(alpha=.15)
    fig.savefig(directory/'support_counterexample.png', dpi=150)
    plt.close(fig)

    traces = json.loads((directory/'traces.json').read_text())
    for family in traces:
        data = traces[family]
        truth = np.array(data['truth'])
        origin = 60
        visible = [truth[:origin+21]]
        for method in ['robust_imm', 'turn_ukf', 'geometric']:
            visible.append(np.array(data[method][origin]['forecast']['paths'])[:24].reshape(-1, 3))
        visible = np.concatenate(visible)
        lower, upper = visible.min(axis=0), visible.max(axis=0)
        center = (lower+upper)/2
        radius = max((upper-lower).max()/2, .1)*1.05
        fig = plt.figure(figsize=(16, 5), layout='constrained')
        for index, method in enumerate(['robust_imm', 'turn_ukf', 'geometric']):
            ax = fig.add_subplot(1, 3, index+1, projection='3d')
            # Fixed before inspecting success; identical axes across comparators.
            record = data[method][origin]['forecast']
            paths = np.array(record['paths'])
            ax.plot(*truth[:origin+1].T, color='#a4adb4', lw=1, label='Past truth')
            ax.plot(*truth[origin:origin+21].T, color='#172f3b', lw=3, label='Future truth (scoring only)')
            for path in paths[:24]:
                ax.plot(*path.T, color=COLORS[METHODS.index(method)], lw=.7, alpha=.3)
            ax.plot(*np.array(record['mean']).T, color=COLORS[METHODS.index(method)], lw=2, label='Predictive mean')
            ax.set(title=LABELS[method], xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
            ax.set_xlim(center[0]-radius, center[0]+radius)
            ax.set_ylim(center[1]-radius, center[1]+radius)
            ax.set_zlim(center[2]-radius, center[2]+radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=25, azim=-65)
            ax.legend(fontsize=8)
        fig.suptitle(f'{family}: retained future paths from the same noisy observation history', fontsize=16)
        fig.savefig(directory/(family+'_forecasts.png'), dpi=135)
        plt.close(fig)
    print('\n'.join(lines[:22]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', default=str(Path(__file__).parent/'results'))
    report(parser.parse_args().directory)
