"""
Shared plotting utilities for PINN experiments.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (side-effect import)
import torch

from .utils import get_device


# ============================================================================
# PDE loss distribution (diagnostic per-epoch plot)
# ============================================================================

def plot_pde_loss_distribution(
    model,
    x_interior: torch.Tensor,
    epoch: int,
    example_name: str,
    output_dir: str = '.',
    outlier_color: str = 'red',
    device: Optional[torch.device] = None,
) -> str:
    """
    Save a 3-panel diagnostic plot of per-point PDE loss.

    Panels: outlier spatial map, log-loss heatmap, loss histogram.

    The model must expose ``get_pde_loss_with_outlier_info(x)`` and
    ``outlier_percentile``.

    Returns:
        Filename of the saved image.
    """
    if device is None:
        device = get_device()

    was_training = model.training
    model.eval()

    n_points = min(10000, len(x_interior))
    if len(x_interior) > n_points:
        indices = torch.randperm(len(x_interior))[:n_points]
        x_sample = x_interior[indices].to(device)
    else:
        x_sample = x_interior.to(device)

    x_coords, y_coords, per_point_loss, is_outlier, threshold = \
        model.get_pde_loss_with_outlier_info(x_sample)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: outlier locations
    ax = axes[0]
    ax.scatter(x_coords[~is_outlier], y_coords[~is_outlier],
               c='blue', s=1, alpha=0.5, label='Normal')
    n_outliers = is_outlier.sum()
    ax.scatter(x_coords[is_outlier], y_coords[is_outlier],
               c=outlier_color, s=10, alpha=0.8, label=f'Outlier ({n_outliers})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    op = getattr(model, 'outlier_percentile', 0)
    ax.set_title(f'Epoch {epoch}: Outlier Locations\n(Top {op}% = {n_outliers} points)')
    ax.legend()
    ax.set_aspect('equal')

    # Panel 2: log-loss heatmap
    ax = axes[1]
    log_loss = np.log10(per_point_loss + 1e-10)
    sc = ax.scatter(x_coords, y_coords, c=log_loss, s=2, cmap='hot', alpha=0.7)
    plt.colorbar(sc, ax=ax, label='log10(loss)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Epoch {epoch}: PDE Loss Magnitude')
    ax.set_aspect('equal')

    # Panel 3: histogram
    ax = axes[2]
    log_all = np.log10(per_point_loss + 1e-10)
    ax.hist(log_all[~is_outlier], bins=50, alpha=0.7, color='blue', label='Normal')
    ax.hist(log_all[is_outlier], bins=50, alpha=0.7, color=outlier_color, label='Outlier')
    ax.axvline(np.log10(threshold + 1e-10), color='black', linestyle='--',
               label=f'Threshold ({threshold:.2e})')
    ax.set_xlabel('log10(PDE loss)')
    ax.set_ylabel('Count')
    ax.set_title(f'Epoch {epoch}: Loss Distribution')
    ax.legend()

    plt.suptitle(f'{example_name} - PDE Loss Analysis (Epoch {epoch})', fontsize=14)
    plt.tight_layout()

    import os
    filename = f'pde_loss_epoch_{epoch:04d}_{example_name}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved PDE loss plot: {filepath}")

    if was_training:
        model.train()

    return filename


# ============================================================================
# Results figure (loss curves + solution comparison)
# ============================================================================

def plot_results(
    model,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    history: dict,
    title: str,
    n_grid: int = 100,
    output_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """
    6-panel summary figure: loss curves, exact/predicted fields, error, 3D views.

    Automatically adds insets for ReLoBRaLo weights, LR schedule, or
    p-schedule if those history keys have varying values.

    Returns:
        The matplotlib Figure object.
    """
    if device is None:
        device = get_device()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # -- Panel (0,0): training curves ----------------------------------------
    ax = axes[0, 0]
    ax.semilogy(history['bc_loss'], label='Train BC', color='violet', alpha=0.7)
    ax.semilogy(history['pde_loss'], label='Train PDE', color='orange', alpha=0.7)

    if 'interface_loss' in history and any(v > 0 for v in history.get('interface_loss', [])):
        ax.semilogy(history['interface_loss'], label='Interface', color='cyan', alpha=0.7)

    if 'val_loss' in history and history['val_loss'] and history['val_loss'][0] is not None:
        ax.semilogy(history['val_loss'], label='Val Loss', color='blue', linewidth=2)
        best_epoch = int(np.argmin(history['val_loss']))
        ax.axvline(x=best_epoch, color='blue', linestyle='--', alpha=0.5,
                   label=f'Best (ep {best_epoch + 1})')

    ax.semilogy(history['test_loss'], label='Test (exact)', color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend(fontsize=8)
    ax.grid(True)

    # p-schedule twin axis (experiment 6_2 iterative training)
    if 'p' in history and len(set(history.get('p', []))) > 1:
        ax_p = ax.twinx()
        ax_p.plot(history['p'], color='red', linestyle='--', alpha=0.5, label='p value')
        ax_p.set_ylabel('p', color='red')
        ax_p.tick_params(axis='y', labelcolor='red')

    # Weight inset (ReLoBRaLo)
    has_weight_inset = False
    if 'weight_bc' in history and len(set(history.get('weight_bc', []))) > 1:
        has_weight_inset = True
        ax_w = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        ax_w.plot(history['weight_bc'], label='w_BC', color='violet', alpha=0.7)
        ax_w.plot(history['weight_pde'], label='w_PDE', color='orange', alpha=0.7)
        ax_w.set_xlabel('Epoch', fontsize=8)
        ax_w.set_ylabel('Weight', fontsize=8)
        ax_w.legend(fontsize=7)
        ax_w.tick_params(labelsize=7)
        ax_w.set_title('Weights', fontsize=8)

    # LR inset
    if 'lr' in history and len(history['lr']) > 0 and len(set(history['lr'])) > 1:
        pos = [0.55, 0.1, 0.4, 0.35] if has_weight_inset else [0.55, 0.55, 0.4, 0.4]
        ax_lr = ax.inset_axes(pos)
        ax_lr.semilogy(history['lr'], label='LR', color='red', alpha=0.7)
        ax_lr.set_xlabel('Epoch', fontsize=8)
        ax_lr.set_ylabel('LR', fontsize=8)
        ax_lr.tick_params(labelsize=7)
        ax_lr.set_title('LR Schedule', fontsize=8)
        ax_lr.grid(True, alpha=0.3)

    # -- get predictions -----------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test.to(device)).cpu()

    x_np = x_test.numpy()
    y_exact_np = y_test.numpy().flatten()
    y_pred_np = y_pred.numpy().flatten()

    # -- panels (0,1)-(1,2): solution comparison -----------------------------
    n_pts = len(x_np)
    try:
        side = int(np.sqrt(n_pts))
        if side * side != n_pts:
            raise ValueError("Non-square grid")

        X = x_np[:, 0].reshape(side, side)
        Y = x_np[:, 1].reshape(side, side)
        Z_exact = y_exact_np.reshape(side, side)
        Z_pred = y_pred_np.reshape(side, side)

        ax = axes[0, 1]
        c = ax.contourf(X, Y, Z_exact, levels=20, cmap='rainbow')
        plt.colorbar(c, ax=ax)
        ax.set_title('Exact Solution')
        ax.set_xlabel('x'); ax.set_ylabel('y')

        ax = axes[0, 2]
        c = ax.contourf(X, Y, Z_pred, levels=20, cmap='rainbow')
        plt.colorbar(c, ax=ax)
        ax.set_title('PINN Prediction')
        ax.set_xlabel('x'); ax.set_ylabel('y')

        ax = axes[1, 0]
        c = ax.contourf(X, Y, np.abs(Z_exact - Z_pred), levels=20, cmap='hot')
        plt.colorbar(c, ax=ax)
        ax.set_title('Absolute Error')
        ax.set_xlabel('x'); ax.set_ylabel('y')

        ax = fig.add_subplot(2, 3, 5, projection='3d')
        ax.plot_surface(X, Y, Z_exact, cmap='rainbow', alpha=0.8)
        ax.set_title('Exact (3D)')
        ax.set_xlabel('x'); ax.set_ylabel('y')

        ax = fig.add_subplot(2, 3, 6, projection='3d')
        ax.plot_surface(X, Y, Z_pred, cmap='rainbow', alpha=0.8)
        ax.set_title('Predicted (3D)')
        ax.set_xlabel('x'); ax.set_ylabel('y')

    except Exception:
        ax = axes[0, 1]
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], c=y_exact_np, cmap='rainbow', s=1)
        plt.colorbar(sc, ax=ax)
        ax.set_title('Exact Solution')
        ax.set_aspect('equal')

        ax = axes[0, 2]
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], c=y_pred_np, cmap='rainbow', s=1)
        plt.colorbar(sc, ax=ax)
        ax.set_title('PINN Prediction')
        ax.set_aspect('equal')

        ax = axes[1, 0]
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], c=np.abs(y_exact_np - y_pred_np),
                        cmap='hot', s=1)
        plt.colorbar(sc, ax=ax)
        ax.set_title('Absolute Error')
        ax.set_aspect('equal')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    return fig
