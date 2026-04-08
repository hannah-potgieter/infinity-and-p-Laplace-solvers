"""
Plot figures from saved .npz data for Experiment 7.1 (Infinity Laplacian).

Reads from:  outputs/expr_7_1/{example}/{run_tag}/npy/
Saves to:    outputs/images/

Usage:
    python plot/experiment_7_1_plot.py --example arctan
    python plot/experiment_7_1_plot.py --example all
    python plot/experiment_7_1_plot.py --example absolute --run-tag normal
    python plot/experiment_7_1_plot.py --example all --base-dir outputs/expr_7_1
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

rcParams['mathtext.fontset'] = 'cm'


def plot_training_history(history_path, out_path):
    """Figure 1: Loss vs Epoch."""
    data = np.load(history_path)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.semilogy(data['bc_loss'], label='Train BC', color='blue', alpha=0.8)
    ax.semilogy(data['pde_loss'], label='Train PINNs', color='orange', alpha=0.8)
    ax.semilogy(data['test_loss'], label='Test MSE', color='green', linewidth=1.5)

    val_loss = data.get('val_loss', None)
    if val_loss is not None and not np.all(np.isnan(val_loss)):
        ax.semilogy(val_loss, label='Val', color='deeppink', linewidth=1.5)
        best_epoch = int(np.nanargmin(val_loss))
        ax.axvline(x=best_epoch, color='deeppink', linestyle='--', alpha=0.5,
                   label=f'Best (ep {best_epoch + 1})')

    if 'interface_loss' in data:
        intf = data['interface_loss']
        if np.any(intf > 0):
            ax.semilogy(intf, label='Interface', color='cyan', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Loss vs Epoch', fontsize=16)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs='auto', numticks=12))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_solution(ax, x, y, z, title, cmap='rainbow', vmin=None, vmax=None):
    """Helper: contourf for grid and non-grid data. Non-grid data is interpolated."""
    from scipy.interpolate import griddata

    n_pts = len(x)
    side = int(np.sqrt(n_pts))

    is_disc = False
    if side * side == n_pts:
        X = x.reshape(side, side)
        Y = y.reshape(side, side)
        Z = z.reshape(side, side)
    else:
        n_grid = 200
        r = max(abs(x).max(), abs(y).max(), 1.0) + 0.04
        xi = np.linspace(-r, r, n_grid)
        yi = np.linspace(-r, r, n_grid)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x, y), z, (X, Y), method='cubic')
        nan_mask = np.isnan(Z)
        if nan_mask.any():
            Z_nearest = griddata((x, y), z, (X, Y), method='nearest')
            Z[nan_mask] = Z_nearest[nan_mask]
        mask = X**2 + Y**2 > 1.0
        Z[mask] = np.nan
        is_disc = True

    levels = np.linspace(vmin if vmin is not None else np.nanmin(Z),
                         vmax if vmax is not None else np.nanmax(Z), 21)
    c = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)

    if is_disc:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r'$\mathbf{x}$', fontsize=14)
    ax.set_ylabel(r'$\mathbf{y}$', fontsize=14)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    return c


def _get_common_ranges(exact_path, pred_path):
    """Compute shared axis and colorbar ranges for exact and prediction plots."""
    exact_data = np.load(exact_path)
    pred_data = np.load(pred_path)
    x_all = np.concatenate([exact_data['x'], pred_data['x']])
    y_all = np.concatenate([exact_data['y'], pred_data['y']])
    u_pred = pred_data['u_pred_best'] if 'u_pred_best' in pred_data else pred_data['u_pred']
    u_all = np.concatenate([exact_data['u_exact'], u_pred])
    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()
    for snap in [-2, -1, 0, 1, 2]:
        if abs(xmin - snap) < 0.05:
            xmin = float(snap)
        if abs(xmax - snap) < 0.05:
            xmax = float(snap)
        if abs(ymin - snap) < 0.05:
            ymin = float(snap)
        if abs(ymax - snap) < 0.05:
            ymax = float(snap)
    return (xmin, xmax, ymin, ymax, u_all.min(), u_all.max())


def _sync_ticks(ax, xlim, ylim):
    """Force x and y axes to use the same tick values when ranges match."""
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    xr = ax.get_xlim()
    yr = ax.get_ylim()
    if abs(xr[0] + xr[1]) < 1e-6 and abs(yr[0] + yr[1]) < 1e-6:
        ticks = np.arange(np.ceil(xr[0] * 2) / 2, xr[1] + 0.25, 0.5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)


def plot_exact_solution(exact_path, out_path, xlim=None, ylim=None, vlim=None):
    """Figure 3: Exact u(x,y)."""
    data = np.load(exact_path)
    x, y, u = data['x'], data['y'], data['u_exact']

    vmin, vmax = (vlim if vlim else (None, None))
    fig, ax = plt.subplots(figsize=(6, 5))
    c = _plot_solution(ax, x, y, u, r'Exact $u(x,y)$', vmin=vmin, vmax=vmax)
    _sync_ticks(ax, xlim, ylim)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(c, cax=cax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_prediction(pred_path, exact_path, out_path, xlim=None, ylim=None, vlim=None):
    """Figure 2: Predicted u(x,y) from best checkpoint, with same colorbar range as exact."""
    pred_data = np.load(pred_path)
    exact_data = np.load(exact_path)

    x = pred_data['x']
    y = pred_data['y']
    u_pred = pred_data['u_pred_best'] if 'u_pred_best' in pred_data else pred_data['u_pred']
    u_exact = exact_data['u_exact']

    if vlim:
        vmin, vmax = vlim
    else:
        vmin = min(u_exact.min(), u_pred.min())
        vmax = max(u_exact.max(), u_pred.max())

    fig, ax = plt.subplots(figsize=(6, 5))
    c = _plot_solution(ax, x, y, u_pred, r'Predicted $u(x,y)$', vmin=vmin, vmax=vmax)
    _sync_ticks(ax, xlim, ylim)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(c, cax=cax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


OUTPUT_NAMES = {
    'arctan': 'inf_lap_arctan',
    'aronsson_square': 'inf_lap_aronsson',
    'aronsson_disc': 'inf_lap_aronsson_circle',
    'absolute': 'inf_lap_abs',
    'absolute_dd': 'inf_lap_abs_dd',
}


def plot_example(example_name, base_dir, run_tag, out_dir):
    """Generate all 3 figures for one example."""
    run_dir = os.path.join(base_dir, example_name, run_tag)
    npy_dir = os.path.join(run_dir, 'npy')

    history_path = os.path.join(npy_dir, 'training_history.npz')
    exact_path = os.path.join(npy_dir, 'exact_solution.npz')
    pred_path = os.path.join(npy_dir, 'predictions.npz')

    for p in [history_path, exact_path, pred_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] {example_name}/{run_tag}: missing {p}")
            return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Plotting {example_name}/{run_tag}:")

    xmin, xmax, ymin, ymax, vmin, vmax = _get_common_ranges(exact_path, pred_path)
    xlim = (xmin, xmax)
    ylim = (ymin, ymax)
    vlim = (vmin, vmax)

    tag = OUTPUT_NAMES.get(example_name, example_name)

    plot_training_history(
        history_path,
        os.path.join(out_dir, f'loss_{tag}.png'))

    plot_prediction(
        pred_path, exact_path,
        os.path.join(out_dir, f'appr_{tag}.png'),
        xlim=xlim, ylim=ylim, vlim=vlim)

    plot_exact_solution(
        exact_path,
        os.path.join(out_dir, f'exact_{tag}.png'),
        xlim=xlim, ylim=ylim, vlim=vlim)


ALL_EXAMPLES = ['arctan', 'absolute', 'absolute_dd', 'aronsson_square', 'aronsson_disc']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot figures from saved .npz data for Experiment 7.1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--example', type=str, default='all',
                        choices=ALL_EXAMPLES + ['all'],
                        help='Which example to plot')
    parser.add_argument('--run-tag', type=str, default='normal',
                        help='Run tag subdirectory')
    parser.add_argument('--base-dir', type=str, default='outputs/expr_7_1',
                        help='Base directory for experiment 7.1 outputs')
    parser.add_argument('--out-dir', type=str, default='outputs/images',
                        help='Directory to save figures')
    args = parser.parse_args()

    examples = ALL_EXAMPLES if args.example == 'all' else [args.example]
    for ex in examples:
        plot_example(ex, args.base_dir, args.run_tag, args.out_dir)
