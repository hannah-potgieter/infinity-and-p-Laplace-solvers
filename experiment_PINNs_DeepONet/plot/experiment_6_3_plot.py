"""
Plot MSE vs p from saved iterative p-Laplacian training histories (Section 6.3).

Reads from:  outputs/expr_6_3_iter/{stage}_{domain}/npy/training_history.npz
             outputs/expr_6_3_iter/{stage}_{domain}/train.log
Saves to:    outputs/images/

Usage:
    python plot/experiment_6_3_plot.py
    python plot/experiment_6_3_plot.py --domain square
    python plot/experiment_6_3_plot.py --domain disc
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'cm'

STAGES = ['p2to20', 'p25to50', 'p60to100', 'p150to500', 'p600to1000']


def _parse_inf_lap_from_logs(base_dir, domain):
    """Extract per-p Inf-Lap raw PDE residual from training log files."""
    p_vals, inf_lap_vals = [], []
    pattern = re.compile(
        r'Stage \d+ \(p=(\d+)\) PDE diagnostics:.*?Inf-Lap raw:\s+([\d.eE+\-]+)',
        re.DOTALL,
    )
    for stage in STAGES:
        path = os.path.join(base_dir, f'{stage}_{domain}', 'train.log')
        if not os.path.exists(path):
            print(f"  [SKIP] Missing log {path}")
            continue
        with open(path) as f:
            text = f.read()
        for m in pattern.finditer(text):
            p_vals.append(int(m.group(1)))
            inf_lap_vals.append(float(m.group(2)))
    order = np.argsort(p_vals)
    return np.array(p_vals)[order], np.array(inf_lap_vals)[order]


def _extract_final_per_p(base_dir, domain):
    """Load history npz files; return last-epoch total_loss and test_loss per p."""
    p_values, total_loss, test_loss = [], [], []

    for stage in STAGES:
        fname = os.path.join(base_dir, f'{stage}_{domain}', 'npy', 'training_history.npz')
        if not os.path.exists(fname):
            print(f"  [SKIP] Missing {fname}")
            continue
        data = np.load(fname, allow_pickle=True)
        p_arr = data['p']
        for pv in np.unique(p_arr):
            idx = np.where(p_arr == pv)[0][-1]
            p_values.append(int(pv))
            total_loss.append(data['total_loss'][idx])
            test_loss.append(data['test_loss'][idx])

    order = np.argsort(p_values)
    return (np.array(p_values)[order],
            np.array(total_loss)[order],
            np.array(test_loss)[order])


def plot_mse_vs_p(base_dir, out_dir, domain):
    """Generate the Loss/MSE vs p figure."""
    p_npz, mse_total, mse_inf = _extract_final_per_p(base_dir, domain)
    p_log, mse_delta_inf = _parse_inf_lap_from_logs(base_dir, domain)

    if len(p_npz) == 0:
        print(f"  No data found for domain '{domain}'.")
        return

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.semilogy(p_npz, mse_total, 'o-', color='blue', markersize=4,
                label=r'$\mathrm{MSE(total)}$')
    ax.semilogy(p_npz, mse_inf, 's-', color='orange', markersize=4,
                label=r'$\mathrm{MSE}_\infty$')
    if len(p_log) > 0:
        ax.semilogy(p_log, mse_delta_inf, '^-', color='green', markersize=4,
                    label=r'$\mathrm{MSE}_{\Delta_\infty}$')

    ax.set_xlabel(r'$p$', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.set_title(r'MSE vs $p$', fontsize=20)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, 'p_lap_pinns_test_loss3.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot MSE vs p for Section 6.3 iterative training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--domain', type=str, default='square',
                        choices=['square', 'disc'],
                        help='Domain name')
    parser.add_argument('--base-dir', type=str, default='outputs/expr_6_3_iter',
                        help='Base directory for Section 6.3 iterative outputs')
    parser.add_argument('--out-dir', type=str, default='outputs/images',
                        help='Directory to save the figure')
    args = parser.parse_args()

    print(f"Plotting MSE vs p for domain '{args.domain}':")
    plot_mse_vs_p(args.base_dir, args.out_dir, args.domain)
