"""
Plot results for Section 6.4.3 (All Ellipses DeepONet).

1. MSE vs θ      — line plot with markers at training θ values.
2. MSE vs (a, b) — heatmap with colorbar.
3. MSE vs p      — FEM vs DeepONet comparison.

Reads from:  outputs/expr_6_4_3/total{N}/{run_tag}/npy/mse.npz
Saves to:    outputs/images/

Usage:
    python plot/experiment_6_4_3_plot.py --total-div 10
    python plot/experiment_6_4_3_plot.py --total-div 8
    python plot/experiment_6_4_3_plot.py --total-div all
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

rcParams['mathtext.fontset'] = 'cm'

ALL_TOTAL_DIVS = [10, 8, 6, 4, 2]


def plot_mse_vs_theta(theta, mse_theta, total_div, out_path, p_label='500'):
    """MSE vs θ (averaged over a, b), markers at training θ values."""
    train_thetas = np.array(
        [k * np.pi / 2 / total_div for k in range(total_div + 1)])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(theta, mse_theta, '-', color='blue', linewidth=1.5)

    marker_idx = [int(np.argmin(np.abs(theta - t))) for t in train_thetas]
    ax.plot(theta[marker_idx], mse_theta[marker_idx],
            'o', color='blue', markersize=5, label=fr'Training $\theta$ ({total_div+1} pts)')

    ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax.set_xticklabels([r'$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.set_ylim(0, 0.0040)
    ax.set_xlabel(r'$\theta$', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.set_title(fr'$\overline{{\mathrm{{MSE}}}}(\theta)$ ($p={p_label}$)', fontsize=20)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_mse_vs_ab(a_real, b_real, mse_ab, out_path, p_label='500'):
    """Heatmap of MSE(a, b) averaged over θ."""
    fig, ax = plt.subplots(figsize=(6, 5))

    from scipy.ndimage import zoom
    n_interp = 200
    factor = n_interp / mse_ab.shape[0]
    mse_smooth = zoom(mse_ab.T, factor, order=3)
    a_fine = np.linspace(a_real[0], a_real[-1], mse_smooth.shape[1])
    b_fine = np.linspace(b_real[0], b_real[-1], mse_smooth.shape[0])
    A, B = np.meshgrid(a_fine, b_fine)
    c = ax.contourf(A, B, mse_smooth, levels=60, cmap='viridis')

    ax.set_xticks([0.9, 0.95, 1.0, 1.05, 1.1])
    ax.set_yticks([0.2, 0.225, 0.25, 0.275, 0.3])
    ax.set_xlabel(r'$a$', fontsize=18)
    ax.set_ylabel(r'$b$', fontsize=18)
    ax.set_title(fr'$\overline{{\mathrm{{MSE}}}}(a,b)$ ($p={p_label}$)', fontsize=20)
    ax.tick_params(labelsize=14)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(c, cax=cax, format='%.1e')
    cb.ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_mse_vs_p(deeponet_p, deeponet_mse, fem_p, fem_mse, total_div, out_path):
    """MSE vs p: FEM and DeepONet comparison."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.semilogy(fem_p, fem_mse, 'o-', color='blue', markersize=3,
                label=r'$\mathrm{MSE}_{\infty,p}^{\mathrm{FEM}}$')
    ax.semilogy(deeponet_p, deeponet_mse, '-', color='green', linewidth=1.2)
    marker_ps = set(fem_p.tolist())
    marker_ps.add(500.0)
    mask = np.isin(deeponet_p, list(marker_ps))
    ax.semilogy(deeponet_p[mask], deeponet_mse[mask], 's', color='green',
                markersize=4, linestyle='None',
                label=r'$\mathrm{MSE}_{\infty,p}$')

    ax.set_xlabel(r'$p$', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.set_title(r'MSE vs $p$', fontsize=20)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_total_div(total_div, base_dir, run_tag, out_dir):
    """Generate figures for one total_div.

    Supports two data layouts:
      - New:    theta_mse(θ) + ab_mse(a,b) saved directly at p=500
      - Legacy: mse_4d(p, a, b, θ) → slice by p, then marginalise
    """
    npz_path = os.path.join(base_dir, f'total{total_div}', run_tag, 'npy', 'mse.npz')
    if not os.path.exists(npz_path):
        print(f"  [SKIP] Missing {npz_path}")
        return

    data = np.load(npz_path)
    os.makedirs(out_dir, exist_ok=True)

    td = total_div
    suffix = '' if td == 10 else f'_int{td}'

    print(f"Plotting Section 6.4.3 total_div={td}:")

    if 'theta_mse' in data and 'ab_mse' in data:
        theta = data['theta']
        theta_mse = data['theta_mse']
        a_real = data['a_real']
        b_real = data['b_real']
        ab_mse = data['ab_mse']

        if not np.all(np.isnan(theta_mse)):
            plot_mse_vs_theta(
                theta, theta_mse, td,
                os.path.join(out_dir, f'p_lap_deeponet_2dellipseall_theta{suffix}.png'))
        if td == 10 and not np.all(np.isnan(ab_mse)):
            plot_mse_vs_ab(
                a_real, b_real, ab_mse,
                os.path.join(out_dir, f'p_lap_deeponet_2dellipseall_ab.png'))

    elif 'mse_4d' in data and not np.all(np.isnan(data['mse_4d'])):
        mse_4d = data['mse_4d']
        sweep_p = data['sweep_p']
        a_real = data['a_real']
        b_real = data['b_real']
        theta = data['theta']

        pi = int(np.argmin(np.abs(sweep_p - 500)))
        p_tag = int(sweep_p[pi])
        mse_slice = mse_4d[pi]

        mse_theta = np.nanmean(mse_slice, axis=(0, 1))
        plot_mse_vs_theta(
            theta, mse_theta, td,
            os.path.join(out_dir, f'p_lap_deeponet_2dellipseall_theta{suffix}.png'),
            p_label=str(p_tag))

        mse_ab = np.nanmean(mse_slice, axis=2)
        plot_mse_vs_ab(
            a_real, b_real, mse_ab,
            os.path.join(out_dir, f'p_lap_deeponet_2dellipseall_ab{suffix}.png'),
            p_label=str(p_tag))

    elif 'theta_mse' in data:
        plot_mse_vs_theta(
            data['theta'], data['theta_mse'], td,
            os.path.join(out_dir, f'p_lap_deeponet_2dellipseall_theta{suffix}.png'))
    else:
        print("  [SKIP] No plottable θ data found")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Section 6.4.3 all ellipses (MSE vs θ, MSE vs (a,b))',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--total-div', type=str, default='all',
                        help='total_div value or "all"')
    parser.add_argument('--run-tag', type=str, default='exact_inf',
                        help='Run tag subdirectory')
    parser.add_argument('--base-dir', type=str, default='outputs/expr_6_4_3',
                        help='Base directory for Section 6.4.3 outputs')
    parser.add_argument('--out-dir', type=str, default='outputs/images',
                        help='Directory to save figures')
    args = parser.parse_args()

    if args.total_div == 'all':
        divs = ALL_TOTAL_DIVS
    else:
        divs = [int(args.total_div)]

    for td in divs:
        plot_total_div(td, args.base_dir, args.run_tag, args.out_dir)
