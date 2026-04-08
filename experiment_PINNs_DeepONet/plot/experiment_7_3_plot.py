"""
Plot MSE vs p for Experiment 7.3 (DeepONet distance-to-origin & distance-to-boundary).

For 7.3.1 (origin): 3 lines — FEM MSE, DeepONet MSE (with exact inf), DeepONet MSE (no exact inf)
For 7.3.2 (boundary): 2 lines — FEM MSE, DeepONet MSE (with exact inf)

Reads from:  outputs/expr_7_3_1/{domain}/{run_tag}/npy/mse.npz
             outputs/expr_7_3_2/{domain}/{run_tag}/npy/mse.npz
Saves to:    outputs/images/

Usage:
    python plot/experiment_7_3_plot.py --type origin --domain disc
    python plot/experiment_7_3_plot.py --type origin --domain all
    python plot/experiment_7_3_plot.py --type boundary --domain ellipse1
    python plot/experiment_7_3_plot.py --type boundary --domain all
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'cm'

ORIGIN_DOMAINS = ['square', 'disc']
BOUNDARY_DOMAINS = ['disc', 'ellipse1', 'ellipse2', 'ellipse3',
                    'sphere', 'cylinder', 'torus']

OUTPUT_NAMES = {
    ('origin', 'disc'):        'p_lap_deeponet_test1.png',
    ('origin', 'square'):      'p_lap_deeponet_test2.png',
    ('boundary', 'disc'):      'p_lap_deeponet_2ddisc.png',
    ('boundary', 'ellipse1'):  'p_lap_deeponet_2dellipse1.png',
    ('boundary', 'ellipse2'):  'p_lap_deeponet_2dellipse2.png',
    ('boundary', 'ellipse3'):  'p_lap_deeponet_2dellipse3.png',
    ('boundary', 'sphere'):    'p_lap_deeponet_3dsphere.png',
    ('boundary', 'cylinder'):  'p_lap_deeponet_3dcylinder.png',
    ('boundary', 'torus'):     'p_lap_deeponet_3dtorus.png',
}


def _plot_deeponet_line(ax, don_p, don_mse, fem_p, marker, color, label):
    """Plot DeepONet line; markers only at p values present in fem_p plus p=500."""
    ax.semilogy(don_p, don_mse, '-', color=color, linewidth=1.2)

    marker_ps = set(fem_p.tolist())
    marker_ps.add(500.0)
    mask = np.isin(don_p, list(marker_ps))
    ax.semilogy(don_p[mask], don_mse[mask], marker, color=color,
                markersize=4, linestyle='None', label=label)


def plot_origin(domain, base_dir, out_dir):
    """Experiment 7.3.1: 3 lines (FEM, DeepONet w/ inf, DeepONet w/o inf)."""
    pinns_path = os.path.join(base_dir, domain, 'pinns_inf', 'npy', 'mse.npz')
    no_inf_path = os.path.join(base_dir, domain, 'no_inf', 'npy', 'mse.npz')

    if not os.path.exists(pinns_path):
        print(f"  [SKIP] Missing {pinns_path}")
        return
    if not os.path.exists(no_inf_path):
        print(f"  [SKIP] Missing {no_inf_path}")
        return

    pinns = np.load(pinns_path)
    no_inf = np.load(no_inf_path)
    fem_p = pinns['fem_p']

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.semilogy(fem_p, pinns['fem_mse'], 'o-', color='blue',
                markersize=3, label=r'$\mathrm{MSE}_{\infty,p}^{\mathrm{FEM}}$')
    _plot_deeponet_line(ax, pinns['deeponet_p'], pinns['deeponet_mse'],
                        fem_p, 's', 'green',
                        r'$\mathrm{MSE}_{\infty,p}$')
    _plot_deeponet_line(ax, no_inf['deeponet_p'], no_inf['deeponet_mse'],
                        fem_p, '^', 'orange',
                        r'$\mathrm{MSE}_{\infty,p}$ (no exact solution)')

    ax.set_xlabel(r'$p$', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.set_title(r'MSE vs $p$', fontsize=20)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = OUTPUT_NAMES.get(('origin', domain), f'mse_vs_p_origin_{domain}.png')
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_boundary(domain, base_dir, out_dir):
    """Experiment 7.3.2: 2 lines (FEM, DeepONet w/ inf)."""
    pinns_path = os.path.join(base_dir, domain, 'pinns_inf', 'npy', 'mse.npz')

    if not os.path.exists(pinns_path):
        print(f"  [SKIP] Missing {pinns_path}")
        return

    pinns = np.load(pinns_path)
    fem_p = pinns['fem_p']

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.semilogy(fem_p, pinns['fem_mse'], 'o-', color='blue',
                markersize=3, label=r'$\mathrm{MSE}_{\infty,p}^{\mathrm{FEM}}$')
    _plot_deeponet_line(ax, pinns['deeponet_p'], pinns['deeponet_mse'],
                        fem_p, 's', 'green',
                        r'$\mathrm{MSE}_{\infty,p}$')

    ax.set_xlabel(r'$p$', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.set_title(r'MSE vs $p$', fontsize=20)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = OUTPUT_NAMES.get(('boundary', domain), f'mse_vs_p_boundary_{domain}.png')
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot MSE vs p for Experiment 7.3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type', type=str, required=True,
                        choices=['origin', 'boundary'],
                        help='Experiment type: origin (7.3.1) or boundary (7.3.2)')
    parser.add_argument('--domain', type=str, default='all',
                        help='Domain name or "all"')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory (default: outputs/expr_7_3_1 for origin, outputs/expr_7_3_2 for boundary)')
    parser.add_argument('--out-dir', type=str, default='outputs/images',
                        help='Directory to save figures')
    args = parser.parse_args()

    if args.type == 'origin':
        base_dir = args.base_dir or 'outputs/expr_7_3_1'
        domains = ORIGIN_DOMAINS if args.domain == 'all' else [args.domain]
        for d in domains:
            print(f"Plotting 7.3.1 origin — {d}:")
            plot_origin(d, base_dir, args.out_dir)
    else:
        base_dir = args.base_dir or 'outputs/expr_7_3_2'
        domains = BOUNDARY_DOMAINS if args.domain == 'all' else [args.domain]
        for d in domains:
            print(f"Plotting 7.3.2 boundary — {d}:")
            plot_boundary(d, base_dir, args.out_dir)
