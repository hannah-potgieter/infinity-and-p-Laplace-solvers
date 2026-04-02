#!/usr/bin/env python3
"""
Experiment 6.3.1 — Distance to Origin (DeepONet)

BVP:
    Δ_p u_p = −1  in Ω,   u_p = 0 at origin,   ∂u_p/∂n = 0 elsewhere.

Exact limiting solution:  u_∞(x, y) = √(x² + y²)

Domains:  disc (x² + y² ≤ 1)  |  square ([−1,1]²)

Usage:
    python experiment_6_3_1.py --domain disc   --epochs 20
    python experiment_6_3_1.py --domain square --epochs 20
    python experiment_6_3_1.py --domain all    --epochs 20
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from DeepONet_code import (
    DeepONet, DeepONetDataset, load_mat_files_per_p,
    infer_pinns_on_grid,
    train_deeponet, evaluate_over_p_range,
    set_seed, get_device, create_output_dirs, setup_logging,
    determine_run_tag, save_mse_npz,
    plot_training_history, plot_mse_vs_p, plot_prediction_2d,
)

torch.set_default_dtype(torch.float32)


# =====================================================================
# Exact solution
# =====================================================================

def exact_solution(x, y):
    """u_∞ = √(x² + y²)"""
    return torch.sqrt(x ** 2 + y ** 2)


# =====================================================================
# Data helpers
# =====================================================================

def _get_pinns_prediction(args, last_pts, domain, npy_dir=None):
    """
    Obtain PINNs prediction at p=∞ on the FEM grid.

    Priority:
        1. ``--pinns-checkpoint`` → run inference (cached to ``.npy``)
        2. ``--pinns-file``       → load pre-saved prediction
        3. fall back to exact closed-form solution
    """
    if args.pinns_checkpoint:
        cache_dir = npy_dir or 'outputs/npy'
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f'pinns_inf_origin_{domain}.npy')
        return infer_pinns_on_grid(
            args.pinns_checkpoint, last_pts, cache_path=cache,
        )

    if args.pinns_file and os.path.exists(args.pinns_file):
        print(f"  Loading PINNs prediction from {args.pinns_file}")
        arr = np.load(args.pinns_file) if args.pinns_file.endswith('.npy') \
            else np.load(args.pinns_file)
        return arr.flatten().reshape(-1, 1).astype(np.float32)

    Y_inf = exact_solution(
        torch.tensor(last_pts[:, 0]),
        torch.tensor(last_pts[:, 1]),
    ).numpy().reshape(-1, 1)
    print(f"  Using exact solution for p=∞")
    return Y_inf


def generate_test_grid(domain, n_points=101):
    """Return (test_x, test_y) on a uniform grid. test_x has 3 cols (x, y, 0)."""
    lin = torch.linspace(-1, 1, n_points)
    X, Y = torch.meshgrid(lin, lin, indexing='ij')

    if domain == 'disc':
        mask = X ** 2 + Y ** 2 <= 1
        Xf, Yf = X[mask], Y[mask]
    else:
        Xf, Yf = X.flatten(), Y.flatten()

    sol = exact_solution(Xf, Yf).unsqueeze(1)
    coords = torch.stack([Xf, Yf], dim=1) / 2.0 + 0.5
    test_x = torch.cat([coords, torch.zeros(coords.shape[0], 1)], dim=1)
    return test_x, sol


# =====================================================================
# Single-domain runner
# =====================================================================

def run_experiment(domain, args):
    run_tag = determine_run_tag(args)
    base = os.path.join(args.output_dir, 'expr_7_3_1', domain, run_tag)
    setup_logging(base)
    output_dirs = create_output_dirs(base)

    print(f"\n{'=' * 60}")
    print(f"Distance to Origin  —  {domain.capitalize()}")
    print(f"{'=' * 60}")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ---- data (single pass: load, compute FEM MSE, build train) --------
    mat_path = os.path.join(args.data_dir, domain.capitalize())
    per_p = load_mat_files_per_p(mat_path)
    if not per_p:
        raise FileNotFoundError(
            f"No .mat files in {mat_path}/."
        )

    X_list, Y_list = [], []
    fem_p_list, fem_mse_list = [], []

    for entry in per_p:
        p, pts, sol = entry['p'], entry['pts'], entry['sol']

        sol_exact = exact_solution(
            torch.tensor(pts[:, 0], dtype=torch.float32),
            torch.tensor(pts[:, 1], dtype=torch.float32),
        ).numpy().reshape(-1, 1)
        mse = float(np.mean((sol - sol_exact) ** 2))
        fem_p_list.append(p)
        fem_mse_list.append(mse)
        print(f"  Loaded p={p}: {pts.shape[0]} points, mse: {mse:.6e}")

        pts_norm = pts / 2.0 + 0.5
        p_col = np.full((pts.shape[0], 1), p / 500.0, dtype=np.float32)
        X_list.append(np.hstack([pts_norm, p_col]).astype(np.float32))
        Y_list.append(sol.astype(np.float32))

    last_pts = per_p[-1]['pts']
    X_train = np.vstack(X_list)
    Y_train = np.vstack(Y_list)

    if args.add_exact_inf and last_pts is not None:
        pts_norm = last_pts / 2.0 + 0.5
        p_col = np.ones((pts_norm.shape[0], 1), dtype=np.float32)
        X_inf = np.hstack([pts_norm, p_col]).astype(np.float32)
        Y_inf = _get_pinns_prediction(args, last_pts, domain, npy_dir=output_dirs['npy'])

        sol_exact_inf = exact_solution(
            torch.tensor(last_pts[:, 0], dtype=torch.float32),
            torch.tensor(last_pts[:, 1], dtype=torch.float32),
        ).numpy().reshape(-1, 1)
        mse_inf = float(np.mean((Y_inf - sol_exact_inf) ** 2))
        print(f"  Added p=∞ point: {X_inf.shape[0]} pts, mse: {mse_inf:.6e}")

        X_train = np.vstack([X_train, X_inf])
        Y_train = np.vstack([Y_train, Y_inf])

    print(f"Training data: X={X_train.shape}  Y={Y_train.shape}")

    loader = DataLoader(
        DeepONetDataset(X_train, Y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    # ---- test data -----------------------------------------------------
    test_x, test_y = generate_test_grid(domain)
    test_data = []
    for pv, lbl in [(150, 'p150'), (200, 'p200'), (250, 'p250'), (500, 'p500')]:
        tx = test_x.clone()
        tx[:, 2] = pv / 500.0
        test_data.append((tx, test_y, lbl))

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

    # ---- inference time at p=500 on full domain -------------------------
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

    # ---- evaluate over p -----------------------------------------------
    ev = evaluate_over_p_range(model, test_x, test_y, device, p_col_idx=2)
    for i, (p, mse) in enumerate(zip(ev['p'], ev['mse'])):
        if i % 20 == 0:
            print(f"  p={p:3d}:  MSE={mse:.6e}")

    # ---- prediction vs FEM training data per p -------------------------
    pred_train_p, pred_train_mse = [], []
    model.eval()
    with torch.no_grad():
        for entry in per_p:
            p, pts, sol = entry['p'], entry['pts'], entry['sol']
            pts_norm = pts / 2.0 + 0.5
            p_col = np.full((pts.shape[0], 1), p / 500.0, dtype=np.float32)
            inp = torch.from_numpy(
                np.hstack([pts_norm, p_col]).astype(np.float32),
            ).to(device)
            pred = model(inp).cpu().numpy()
            mse = float(np.mean((pred - sol) ** 2))
            pred_train_p.append(p)
            pred_train_mse.append(mse)

    # ---- save all MSE curves to one .npz -------------------------------
    save_mse_npz(
        os.path.join(output_dirs['npy'], 'mse.npz'),
        deeponet=(ev['p'], ev['mse']),
        fem=(fem_p_list, fem_mse_list),
        pred_train=(pred_train_p, pred_train_mse),
    )

    # ---- plots ---------------------------------------------------------
    plot_training_history(history, f'6.3.1 {domain}',
                          os.path.join(output_dirs['plots'], 'training.png'))
    plot_mse_vs_p(ev['p'], ev['mse'], f'6.3.1 {domain}',
                  os.path.join(output_dirs['plots'], 'mse_vs_p.png'))

    for pv in (200, 500):
        _plot_2d(model, domain, device, output_dirs['plots'], pv)

    return model, history, ev


def _plot_2d(model, domain, device, out_dir, p_value, n=101):
    """Produce prediction-vs-exact comparison on a grid."""
    lin = torch.linspace(-1, 1, n)
    X, Y = torch.meshgrid(lin, lin, indexing='ij')
    mask = (X ** 2 + Y ** 2 <= 1) if domain == 'disc' else torch.ones_like(X, dtype=torch.bool)

    inp = torch.stack([
        X.flatten() / 2 + 0.5,
        Y.flatten() / 2 + 0.5,
        torch.full((n * n,), p_value / 500.0),
    ], dim=1)

    model.eval()
    with torch.no_grad():
        pred = model(inp.to(device)).cpu().squeeze().reshape(n, n).numpy()
    exact = exact_solution(X.flatten(), Y.flatten()).reshape(n, n).numpy()

    plot_prediction_2d(
        pred, exact, X, Y,
        title=f'{domain} p={p_value}',
        output_path=os.path.join(out_dir, f'pred_p{p_value}.png'),
        mask=mask,
    )


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description='Dist-to-Origin DeepONet')
    ap.add_argument('--domain', default='disc', choices=['disc', 'square', 'all'])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--trunk-layers', default='2,512,512,512,128')
    ap.add_argument('--branch-layers', default='1,128,128,128,128')
    ap.add_argument('--data-dir', default='./experiment_6_3_1')
    ap.add_argument('--output-dir', default='outputs')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--no-exact-inf', dest='add_exact_inf', action='store_false',
                    help='Do NOT append exact p=∞ point to mat-file data')
    ap.add_argument('--pinns-checkpoint', default=None,
                    help='Path to PINNs .pt checkpoint (from experiment_6_3.py). '
                         'Runs inference on FEM grid, caches to .npy.')
    ap.add_argument('--pinns-file', default=None,
                    help='Path to pre-saved PINNs prediction (.npy). '
                         'Skipped if --pinns-checkpoint is given.')
    ap.add_argument('--run-tag', default='auto',
                    help='Tag for MSE output files (auto-detected from flags '
                         'when "auto": no_inf / exact_inf / pinns_inf)')
    args = ap.parse_args()

    domains = ['disc', 'square'] if args.domain == 'all' else [args.domain]
    for d in domains:
        run_experiment(d, args)

    print("\nAll experiments completed!")


if __name__ == '__main__':
    main()
