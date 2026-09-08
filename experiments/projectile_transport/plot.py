"""Render saved measurements only; performs no estimator fitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .study import METHODS, SCENARIOS


COLORS = ['#87959b', '#dc7849', '#9274b2', '#4b9eb0', '#087f65']
LABELS = ['Ballistic KF', 'Fixed joint EKF', 'Motion mixture', 'Noise mixture', 'Joint transport']


def render(directory):
    directory = Path(directory)
    results = json.loads((directory/'results.json').read_text())
    traces = json.loads((directory/'trace.json').read_text())
    truth = traces['truth']
    times = np.array([row['t'] for row in truth])
    xy = np.array([row['state'][:2] for row in truth])
    obs = np.array([row['observation'] if row['observation'] is not None else [np.nan, np.nan] for row in truth])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    fig.suptitle('Projectile tracking: transporting motion and noise distributions', fontsize=19)
    ax = axes[0, 0]
    ax.plot(*xy.T, color='#172f3b', lw=2, label='Truth')
    ax.scatter(*obs.T, color='#a4abb0', s=12, alpha=.55, label='Observed position')
    for name, color, label in zip(METHODS, COLORS, LABELS):
        if name not in ('ballistic_kf', 'fixed_joint_ekf', 'joint_transport'):
            continue
        means = np.array([row['mean'] for row in traces[name]])
        ax.plot(*means[:, :2].T, color=color, lw=1.5, label=label)
    ax.set(title='Combined disturbance example: estimated path', xlabel='Horizontal position (m)', ylabel='Height (m)')
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for name, color, label in zip(METHODS, COLORS, LABELS):
        means = np.array([row['mean'] for row in traces[name]])
        ax.plot(times, np.linalg.norm(means[:, :2]-xy, axis=1), color=color, label=label)
    ax.axvspan(1.1, 2.1, color='#dfb863', alpha=.13, label='High sensor noise')
    ax.set(title='Tracking error through noise changes and gaps', xlabel='Time (s)', ylabel='Position error (m)')
    ax.legend(fontsize=8, ncol=2)
    ax = axes[1, 0]
    ax.plot(times, [row['sigma']**2 for row in truth], color='#172f3b', label='True white-noise variance')
    for name, color, label in zip(METHODS, COLORS, LABELS):
        if name in ('fixed_joint_ekf', 'noise_only', 'joint_transport'):
            ax.plot(times, [row['expected_sigma2'] for row in traces[name]], color=color, label=label)
    ax.set(title='Inferred sensor-noise distribution', xlabel='Time (s)', ylabel='Posterior mean variance (m²)')
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    x = np.arange(len(SCENARIOS))
    for index, (name, color, label) in enumerate(zip(METHODS, COLORS, LABELS)):
        values = [results['summary'][s][name]['forecast_1.0s_rmse'] for s in SCENARIOS]
        ax.bar(x+(index-2)*.15, values, .145, color=color, label=label)
    ax.set_xticks(x, ['Ballistic', 'Drag', 'Force\nchange', 'Sensor\nchange', 'Combined', 'Off-catalog\ndrag'])
    ax.set(title=f'One-second forecasts without future observations ({results["metadata"]["seeds"]} seeds)', ylabel='Position RMSE (m)')
    ax.legend(fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.grid(alpha=.14)
        ax.set_axisbelow(True)
    fig.savefig(directory/'overview.png', dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    fig.suptitle('Component attribution: truth, posterior mean and marginal ±1.96σ', fontsize=17)
    joint = traces['joint_transport']
    means = np.array([row['components_mean'] for row in joint])
    covs = np.array([row['components_cov'] for row in joint])
    for column, (key, label, unit) in enumerate([('drag', 'Drag acceleration', 'm/s²'), ('force', 'Residual acceleration', 'm/s²'), ('bias', 'Camera bias', 'm')]):
        actual = np.array([row[key] for row in truth])
        for axis in range(2):
            ax = axes[axis, column]
            i = 2*column+axis
            sd = np.sqrt(np.maximum(covs[:, i, i], 0))
            ax.fill_between(times, means[:, i]-1.96*sd, means[:, i]+1.96*sd, color='#087f65', alpha=.16)
            ax.plot(times, actual[:, axis], color='#172f3b', label='Truth')
            ax.plot(times, means[:, i], color='#087f65', label='Joint transport')
            ax.set(title=label+' ('+['x', 'y'][axis]+')', xlabel='Time (s)', ylabel=unit)
            ax.grid(alpha=.14)
            ax.legend(fontsize=9)
    fig.savefig(directory/'components.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', default=str(Path(__file__).parent/'results'))
    render(parser.parse_args().directory)
