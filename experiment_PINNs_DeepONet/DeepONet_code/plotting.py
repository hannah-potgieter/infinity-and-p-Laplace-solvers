"""
Plotting utilities for DeepONet experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(loss_history, title='', output_path=None):
    """
    Plot all loss curves stored in *loss_history* on a log-scale.

    Args:
        loss_history: dict mapping label -> list of values (one per epoch).
        title: plot title.
        output_path: if given, save figure instead of showing.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, vals in loss_history.items():
        ax.semilogy(range(len(vals)), vals, label=key)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(title or 'Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_mse_vs_p(p_values, mse_values, title='', output_path=None):
    """
    Semi-log plot of MSE versus the p-Laplacian parameter.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(p_values, mse_values, 'b-o', markersize=3)
    ax.set_xlabel('p')
    ax.set_ylabel('MSE (vs exact solution at p=∞)')
    ax.set_title(title or 'DeepONet MSE vs p')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_prediction_2d(
    pred_grid,
    exact_grid,
    X_mesh,
    Y_mesh,
    title='',
    output_path=None,
    mask=None,
):
    """
    Side-by-side comparison: prediction | exact | absolute error.

    All grid arrays should have shape ``(Nx, Ny)``.
    *mask* is an optional boolean array; ``False`` entries are set to NaN.
    """
    if mask is not None:
        mask_np = mask if isinstance(mask, np.ndarray) else mask.numpy()
        pred_grid = np.where(mask_np, pred_grid, np.nan)
        exact_grid = np.where(mask_np, exact_grid, np.nan)

    X_np = X_mesh if isinstance(X_mesh, np.ndarray) else X_mesh.numpy()
    Y_np = Y_mesh if isinstance(Y_mesh, np.ndarray) else Y_mesh.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].contourf(X_np, Y_np, pred_grid, levels=20, cmap='rainbow')
    axes[0].set_title('Prediction')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X_np, Y_np, exact_grid, levels=20, cmap='rainbow')
    axes[1].set_title('Exact (p=∞)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1])

    error_grid = np.abs(pred_grid - exact_grid)
    im2 = axes[2].contourf(X_np, Y_np, error_grid, levels=20, cmap='hot')
    axes[2].set_title('|Error|')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2])

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()
