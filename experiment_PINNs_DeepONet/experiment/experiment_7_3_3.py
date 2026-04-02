#!/usr/bin/env python3
"""
Experiment 6.3.3 — Distance to Boundary, All Ellipses (DeepONet)

Learns the operator over a *family* of 2-D ellipses parameterised by
rotation angle θ ∈ [0, π/2] and shape parameters (a, b).

Trunk input:   (x, y)                    — 2-D spatial coordinates
Branch input:  (p/500, θ/(π/2), a, b)    — 4 PDE / geometry parameters

Usage:
    python experiment_6_3_3.py --epochs 20 --total-div 10
    python experiment_6_3_3.py --epochs 20 --total-div 2   # smaller dataset
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from DeepONet_code import (
    DeepONet, DeepONetDataset, load_mat_files_per_p,
    train_deeponet, evaluate_over_p_range,
    set_seed, get_device, create_output_dirs, setup_logging,
    determine_run_tag,
    plot_training_history, plot_mse_vs_p,
    dist_to_ellipse_boundary,
)

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

torch.set_default_dtype(torch.float32)


# =====================================================================
# Exact solution for a rotated ellipse
# =====================================================================

def exact_solution_ellipse(x, y, a_param, b_param, theta=0.0):
    """
    Distance to boundary of a rotated ellipse.

    Parameters (a_param, b_param) ∈ [0, 1] are mapped to semi-axes::

        a = a_param * 0.2 + 0.9   →  a ∈ [0.9, 1.1]
        b = b_param * 0.1 + 0.2   →  b ∈ [0.2, 0.3]

    The ellipse is  x²/a² + y²/b² = 1  *after* rotating by θ.
    """
    a = a_param * 0.2 + 0.9
    b = b_param * 0.1 + 0.2
    xr = np.cos(theta) * x - np.sin(theta) * y
    yr = np.sin(theta) * x + np.cos(theta) * y
    return dist_to_ellipse_boundary(xr, yr, a, b)


# =====================================================================
# Training data generation
# =====================================================================

def generate_training_data(
    mat_path, total_div=10, include_exact_inf=True, p_normalize=500.0,
):
    """
    Build training data by combining .mat FEM solutions with exact p = ∞
    solutions at various (θ, a, b) settings.

    Returns ``(X_train, Y_train)`` NumPy arrays.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required to load .mat files")
    if not os.path.isdir(mat_path):
        raise FileNotFoundError(f"Data directory not found: {mat_path}")

    mat_files = sorted(glob.glob(os.path.join(mat_path, '*.mat')))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {mat_path}")

    # find highest-p reference file
    best_p, ref_file = 0, None
    for mf in mat_files:
        m = re.search(r'p\s*(\d+)', os.path.basename(mf))
        if m and int(m.group(1)) > best_p:
            best_p, ref_file = int(m.group(1)), mf

    print(f"  Reference file: {ref_file}  (p={best_p})")
    mat = scipy.io.loadmat(ref_file)
    pts = mat['pts']
    store_y = mat['solution'][0].reshape(-1, 1)

    base_a, base_b = 0.5, 0.5
    x_t, y_t = torch.tensor(pts[:, 0]), torch.tensor(pts[:, 1])
    y_exact = exact_solution_ellipse(x_t, y_t, base_a, base_b).numpy().reshape(-1, 1)

    valid = np.abs(y_exact[:, 0] - store_y[:, 0]) < 0.01
    pts_norm = pts / 2.0 + 0.5

    X_list, Y_list = [], []

    if include_exact_inf:
        # base ellipse, theta=0
        n = pts.shape[0]
        X_base = np.column_stack([
            pts_norm,
            np.full(n, 500 / p_normalize),
            np.zeros(n),
            np.full(n, base_a),
            np.full(n, base_b),
        ])
        X_base = X_base[valid][::10]
        Y_base = y_exact[valid][::10]
        X_list.append(X_base)
        Y_list.append(Y_base)
        print(f"  Base ellipse (θ=0): {X_base.shape[0]} pts")

        # rotations of base ellipse
        for ti in range(total_div):
            theta = (ti + 1) * np.pi / 2 / total_div
            xo = (pts_norm[:, 0] - 0.5) * 2
            yo = (pts_norm[:, 1] - 0.5) * 2
            xr = np.cos(theta) * xo - np.sin(theta) * yo
            yr = np.sin(theta) * xo + np.cos(theta) * yo
            X_rot = np.column_stack([
                xr / 2 + 0.5, yr / 2 + 0.5,
                np.full(n, 500 / p_normalize),
                np.full(n, theta / (np.pi / 2)),
                np.full(n, base_a),
                np.full(n, base_b),
            ])
            X_list.append(X_rot[valid][::10])
            Y_list.append(Y_base.copy())
        print(f"  + {total_div} rotations of base ellipse")

    # vary (a_param, b_param) × rotations
    pts_f = pts[valid][::10]
    pts_nf = pts_norm[valid][::10]
    n_f = pts_f.shape[0]

    for ai in range(total_div):
        a_p = (0.5 - (ai + 1) / total_div * 2 * 0.5
               if ai < total_div // 2
               else 0.5 + (ai - total_div // 2 + 1) / total_div * 2 * 0.5)
        for bi in range(total_div):
            b_p = (0.5 - (bi + 1) / total_div * 2 * 0.5
                   if bi < total_div // 2
                   else 0.5 + (bi - total_div // 2 + 1) / total_div * 2 * 0.5)

            if not include_exact_inf:
                continue

            y_ab = exact_solution_ellipse(
                torch.tensor(pts_f[:, 0]),
                torch.tensor(pts_f[:, 1]),
                a_p, b_p, theta=0.0,
            ).numpy().reshape(-1, 1)

            # theta = 0
            X_ab = np.column_stack([
                pts_nf,
                np.full(n_f, 500 / p_normalize),
                np.zeros(n_f),
                np.full(n_f, a_p),
                np.full(n_f, b_p),
            ])
            X_list.append(X_ab)
            Y_list.append(y_ab)

            # rotations
            for ti in range(total_div):
                theta = (ti + 1) * np.pi / 2 / total_div
                xo = (pts_nf[:, 0] - 0.5) * 2
                yo = (pts_nf[:, 1] - 0.5) * 2
                xr = np.cos(theta) * xo - np.sin(theta) * yo
                yr = np.sin(theta) * xo + np.cos(theta) * yo
                X_rot = np.column_stack([
                    xr / 2 + 0.5, yr / 2 + 0.5,
                    np.full(n_f, 500 / p_normalize),
                    np.full(n_f, theta / (np.pi / 2)),
                    np.full(n_f, a_p),
                    np.full(n_f, b_p),
                ])
                X_list.append(X_rot)
                Y_list.append(y_ab.copy())

        if (ai + 1) % max(total_div // 5, 1) == 0:
            print(f"  a-param {ai + 1}/{total_div}")

    X_train = np.vstack(X_list).astype(np.float32)
    Y_train = np.vstack(Y_list).astype(np.float32)
    print(f"  Total training samples: {X_train.shape[0]}")
    return X_train, Y_train


# =====================================================================
# FEM vs exact MSE per p (base ellipse a=0.5, b=0.5, theta=0)
# =====================================================================

def _find_ellipse1_dir(data_dir):
    """Return the ``Ellipse 1`` subdirectory under *data_dir*."""
    return os.path.join(data_dir, 'Ellipse 1')


def _compute_fem_mse_from(mat_path):
    """Compute MSE(FEM, u_∞) per p for the base ellipse.

    Returns ``(fem_p_list, fem_mse_list, per_p)`` — lists are empty and
    ``per_p`` is ``[]`` when no ``.mat`` files are found.
    """
    per_p = load_mat_files_per_p(mat_path)
    if not per_p:
        return [], [], []

    base_a, base_b = 0.5, 0.5
    p_list, mse_list = [], []
    for entry in per_p:
        pts, sol_fem = entry['pts'], entry['sol']
        sol_exact = exact_solution_ellipse(
            torch.tensor(pts[:, 0], dtype=torch.float32),
            torch.tensor(pts[:, 1], dtype=torch.float32),
            base_a, base_b, theta=0.0,
        ).numpy().reshape(-1, 1)
        mse = float(np.mean((sol_fem - sol_exact) ** 2))
        p_list.append(entry['p'])
        mse_list.append(mse)
        print(f"  Loaded p={entry['p']}: {pts.shape[0]} points, mse: {mse:.6e}")
    return p_list, mse_list, per_p


# =====================================================================
# Experiment runner
# =====================================================================

def run_experiment(args):
    tag = f'total{args.total_div}'
    run_tag = determine_run_tag(args)
    base = os.path.join(args.output_dir, 'expr_7_3_3', tag, run_tag)
    setup_logging(base)
    output_dirs = create_output_dirs(base)

    print(f"\n{'=' * 60}")
    print(f"All Ellipses  —  total_div={args.total_div}")
    print(f"{'=' * 60}")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ---- FEM MSE (before training, so it prints during loading) --------
    mat_path = _find_ellipse1_dir(args.data_dir)
    fem_p_list, fem_mse_list, per_p = _compute_fem_mse_from(mat_path)

    # ---- data ----------------------------------------------------------
    X_train, Y_train = generate_training_data(
        mat_path, total_div=args.total_div,
        include_exact_inf=args.add_exact_inf,
    )
    print(f"Training data: X={X_train.shape}  Y={Y_train.shape}")

    loader = DataLoader(
        DeepONetDataset(X_train, Y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    # ---- test (sub-sample from training) -------------------------------
    n_test = min(5000, len(X_train) // 10)
    idx = np.random.choice(len(X_train), n_test, replace=False)
    test_x = torch.from_numpy(X_train[idx]).float()
    test_y = torch.from_numpy(Y_train[idx]).float()
    test_data = [(test_x, test_y, 'holdout')]

    # ---- model ---------------------------------------------------------
    trunk = [int(x) for x in args.trunk_layers.split(',')]
    branch = [int(x) for x in args.branch_layers.split(',')]
    model = DeepONet(trunk_layers=trunk, branch_layers=branch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: trunk={trunk}  branch={branch}  params={n_params:,}")

    # ---- train ---------------------------------------------------------
    t_train_start = time.time()
    history = train_deeponet(
        model, loader, args.epochs,
        lr=args.lr, device=device,
        test_data=test_data, show_every=1,
        checkpoint_dir=output_dirs['checkpoints'],
        checkpoint_name='model',
    )
    t_train = time.time() - t_train_start
    print(f"Total training time: {t_train:.2f}s")

    # ---- inference time at p=500 on test set ----------------------------
    test_x_500 = test_x.clone()
    test_x_500[:, 2] = 500.0 / 500.0
    model.eval()
    x_test_dev = test_x_500.to(device)
    with torch.no_grad():
        for _ in range(10):
            model(x_test_dev)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            model(x_test_dev[:1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_pt = (time.time() - t0) / 100
        t0 = time.time()
        for _ in range(100):
            model(x_test_dev)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_full = (time.time() - t0) / 100
    print(f"Inference (per point): {t_pt:.4e}s | (full domain): {t_full:.4e}s")

    # ---- evaluate over p (column 2) ------------------------------------
    ev = evaluate_over_p_range(
        model, test_x, test_y, device, p_col_idx=2,
    )
    for i, (p, mse) in enumerate(zip(ev['p'], ev['mse'])):
        if i % 20 == 0:
            print(f"  p={p:3d}:  MSE={mse:.6e}")

    # ---- prediction vs FEM training data per p (base ellipse) ----------
    pred_train_p, pred_train_mse = [], []
    if per_p:
        base_a, base_b = 0.5, 0.5
        model.eval()
        with torch.no_grad():
            for entry in per_p:
                p, pts, sol = entry['p'], entry['pts'], entry['sol']
                pts_norm = pts / 2.0 + 0.5
                inp_cols = np.column_stack([
                    pts_norm,
                    np.full(pts.shape[0], p / 500.0),
                    np.zeros(pts.shape[0]),
                    np.full(pts.shape[0], base_a),
                    np.full(pts.shape[0], base_b),
                ]).astype(np.float32)
                inp = torch.from_numpy(inp_cols).to(device)
                pred = model(inp).cpu().numpy()
                mse = float(np.mean((pred - sol) ** 2))
                pred_train_p.append(p)
                pred_train_mse.append(mse)

    # ---- 4D sweep: MSE(p, a, b, θ) ----------------------------------------
    sweep_p_values = np.array([200, 300, 400, 450, 495, 500])
    n_ab = 21
    n_theta_steps = 101
    a_param_grid = np.linspace(0, 1, n_ab)
    b_param_grid = np.linspace(0, 1, n_ab)
    a_real_grid = a_param_grid * 0.2 + 0.9
    b_real_grid = b_param_grid * 0.1 + 0.2
    theta_grid = np.linspace(0, np.pi / 2, n_theta_steps)

    base_a, base_b = 0.5, 0.5
    base_mask = (
        (np.abs(X_train[:, 2] - 1.0) < 1e-6) &
        (np.abs(X_train[:, 3]) < 1e-6) &
        (np.abs(X_train[:, 4] - base_a) < 1e-6) &
        (np.abs(X_train[:, 5] - base_b) < 1e-6)
    )
    x_base = X_train[base_mask]
    n_p, n_a, n_b, n_t = len(sweep_p_values), n_ab, n_ab, n_theta_steps
    mse_4d = np.full((n_p, n_a, n_b, n_t), np.nan)

    print(f"4D sweep: {n_p}×{n_a}×{n_b}×{n_t} = {n_p*n_a*n_b*n_t:,} evaluations, "
          f"{x_base.shape[0]} base pts")

    if x_base.shape[0] > 0:
        model.eval()
        with torch.no_grad():
            for pi, p_val in enumerate(sweep_p_values):
                for ai, ap in enumerate(a_param_grid):
                    for bi, bp in enumerate(b_param_grid):
                        for ti, theta_val in enumerate(theta_grid):
                            inp = x_base.copy()
                            xo = (inp[:, 0] - 0.5) * 2.0
                            yo = (inp[:, 1] - 0.5) * 2.0
                            inp[:, 0] = (np.cos(theta_val) * xo - np.sin(theta_val) * yo) / 2.0 + 0.5
                            inp[:, 1] = (np.sin(theta_val) * xo + np.cos(theta_val) * yo) / 2.0 + 0.5
                            inp[:, 2] = p_val / 500.0
                            inp[:, 3] = theta_val / (np.pi / 2)
                            inp[:, 4] = ap
                            inp[:, 5] = bp

                            y_exact = exact_solution_ellipse(
                                torch.tensor(((inp[:, 0] - 0.5) * 2).astype(np.float32)),
                                torch.tensor(((inp[:, 1] - 0.5) * 2).astype(np.float32)),
                                ap, bp, theta=theta_val,
                            ).numpy().reshape(-1, 1)

                            pred = model(torch.from_numpy(inp).float().to(device))
                            mse_4d[pi, ai, bi, ti] = float(torch.mean(
                                (pred - torch.from_numpy(y_exact).float().to(device)) ** 2
                            ).item())
                    if (ai + 1) % 7 == 0:
                        print(f"  p={p_val}: a {ai+1}/{n_a}")
                print(f"  p={p_val} done")
    else:
        print("  [WARN] No base points for 4D sweep (need exact_inf)")

    # ---- save all MSE curves to one .npz -------------------------------
    save_dict = dict(
        deeponet_p=np.array(ev['p']),
        deeponet_mse=np.array(ev['mse']),
        fem_p=np.array(fem_p_list),
        fem_mse=np.array(fem_mse_list),
        pred_train_p=np.array(pred_train_p),
        pred_train_mse=np.array(pred_train_mse),
        sweep_p=sweep_p_values,
        a_real=a_real_grid,
        b_real=b_real_grid,
        theta=theta_grid,
        mse_4d=mse_4d,
    )
    npz_path = os.path.join(output_dirs['npy'], 'mse.npz')
    np.savez(npz_path, **save_dict)
    print(f"Saved MSE curves → {npz_path}")

    # ---- plots ---------------------------------------------------------
    plot_training_history(
        history, f'6.3.3 All Ellipses ({tag})',
        os.path.join(output_dirs['plots'], 'training.png'),
    )
    plot_mse_vs_p(
        ev['p'], ev['mse'], f'6.3.3 All Ellipses ({tag})',
        os.path.join(output_dirs['plots'], 'mse_vs_p.png'),
    )

    return model, history, ev


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Dist-to-Boundary All-Ellipse DeepONet',
    )
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--trunk-layers', default='2,512,512,512,128')
    ap.add_argument('--branch-layers', default='4,128,128,128,128')
    ap.add_argument('--total-div', type=int, default=10)
    ap.add_argument('--data-dir', default='./experiment_6_3_3')
    ap.add_argument('--output-dir', default='outputs')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--no-exact-inf', dest='add_exact_inf', action='store_false')
    ap.add_argument('--run-tag', default='auto',
                    help='Tag for MSE output files (auto-detected from flags '
                         'when "auto": no_inf / exact_inf)')
    args = ap.parse_args()

    run_experiment(args)
    print("\nExperiment completed!")


if __name__ == '__main__':
    main()
