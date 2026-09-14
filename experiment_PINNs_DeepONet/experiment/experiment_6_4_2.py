#!/usr/bin/env python3
"""
Section 6.4.2 — Distance to Boundary (DeepONet, 2-D & 3-D)

BVP:
    Δ_p u_p = −1  in Ω,   u_p = 0 on ∂Ω.

Exact limiting solution:  u_∞ = dist(x, ∂Ω)

2-D domains:  disc · ellipse1 · ellipse2 · ellipse3
3-D domains:  sphere · cube · cylinder · torus

Usage:
    python experiment/experiment_6_4_2.py --domain disc     --epochs 20
    python experiment/experiment_6_4_2.py --domain sphere   --epochs 20
    python experiment/experiment_6_4_2.py --domain all-2d   --epochs 20
    python experiment/experiment_6_4_2.py --domain all-3d   --epochs 20
    python experiment/experiment_6_4_2.py --domain all      --epochs 20
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
    distance_to_disc_boundary, dist_to_ellipse_boundary,
)

torch.set_default_dtype(torch.float32)


# =====================================================================
# Domain configurations
# =====================================================================

DOMAINS_2D = {
    'disc': {
        'dim': 2, 'name': 'Unit Disc',
        'equation': 'x² + y² ≤ 1',
        'a': 1.0, 'b': 1.0, 'theta': 0.0,
        'coord_scale': 2.0, 'coord_offset': 0.5,
        'data_subdir': 'Unit disc',
    },
    'ellipse1': {
        'dim': 2, 'name': 'Ellipse 1 (a=1, b=¼)',
        'equation': 'x² + 16y² ≤ 1',
        'a': 1.0, 'b': 0.25, 'theta': 0.0,
        'coord_scale': 2.0, 'coord_offset': 0.5,
        'data_subdir': 'Ellipse 1',
    },
    'ellipse2': {
        'dim': 2, 'name': 'Ellipse 2 (rotated −π/4)',
        'equation': '8.5x² + 8.5y² − 15xy ≤ 1',
        'a': 1.0, 'b': 0.25, 'theta': -np.pi / 4,
        'coord_scale': 2.0, 'coord_offset': 0.5,
        'data_subdir': 'Ellipse 2',
    },
    'ellipse3': {
        'dim': 2, 'name': 'Ellipse 3 (a=½, b=1)',
        'equation': '4x² + y² ≤ 1',
        'a': 0.5, 'b': 1.0, 'theta': 0.0,
        'coord_scale': 2.0, 'coord_offset': 0.5,
        'data_subdir': 'Ellipse 3',
    },
}

DOMAINS_3D = {
    'sphere': {
        'dim': 3, 'name': 'Unit Sphere',
        'shape_flag': 0,
        'coord_scale': 2.0, 'coord_offset': 0.5,
        'data_subdir': 'Sphere',
    },
    'cube': {
        'dim': 3, 'name': 'Unit Cube',
        'shape_flag': 1,
        'coord_scale': 2.0, 'coord_offset': 0.5,
        'data_subdir': 'Cube',
    },
    'cylinder': {
        'dim': 3, 'name': 'Cylinder',
        'shape_flag': 0.1,
        'coord_scale': 2.0, 'coord_offset': 0.5,
        'data_subdir': 'Cylinder',
    },
    'torus': {
        'dim': 3, 'name': 'Torus',
        'shape_flag': 0.2,
        'coord_scale': 6.0, 'coord_offset': 0.5,
        'data_subdir': 'Torus',
        'subsample': 10,
    },
}

ALL_DOMAINS = {**DOMAINS_2D, **DOMAINS_3D}


# =====================================================================
# Exact solutions
# =====================================================================

def exact_2d(x, y, cfg):
    """Distance to boundary for a 2-D domain."""
    if cfg.get('a') == 1.0 and cfg.get('b') == 1.0 and cfg.get('theta', 0) == 0:
        return distance_to_disc_boundary(x, y)
    theta = cfg.get('theta', 0.0)
    if theta != 0:
        xr = np.cos(theta) * x - np.sin(theta) * y
        yr = np.sin(theta) * x + np.cos(theta) * y
    else:
        xr, yr = x, y
    return dist_to_ellipse_boundary(xr, yr, cfg['a'], cfg['b'])


def exact_3d(x, y, z, cfg):
    """Distance to boundary for a 3-D domain."""
    sf = cfg['shape_flag']
    if sf == 0:       # sphere
        return 1.0 - torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    if sf == 1:       # cube [-1,1]^3
        return torch.min(torch.min(1 - torch.abs(x), 1 - torch.abs(y)), 1 - torch.abs(z))
    if sf == 0.1:     # cylinder  x∈[-1,1], y²+z²≤1
        return torch.min(1 - torch.abs(x), 1 - torch.sqrt(y ** 2 + z ** 2))
    if sf == 0.2:     # torus  (√(x²+z²)−2)²+y²≤1
        return 1.0 - torch.sqrt((torch.sqrt(x ** 2 + z ** 2) - 2) ** 2 + y ** 2)
    raise ValueError(f"Unknown shape_flag={sf}")


# =====================================================================
# Test-grid generation
# =====================================================================

def _inside_2d(x, y, domain):
    if domain == 'disc':
        return x ** 2 + y ** 2 <= 1
    if domain == 'ellipse1':
        return x ** 2 + 16 * y ** 2 <= 1
    if domain == 'ellipse2':
        return 8.5 * x ** 2 + 8.5 * y ** 2 - 15 * x * y <= 1
    if domain == 'ellipse3':
        return 4 * x ** 2 + y ** 2 <= 1
    return x ** 2 + y ** 2 <= 1


def _inside_3d(x, y, z, cfg):
    sf = cfg['shape_flag']
    if sf == 0:
        return x ** 2 + y ** 2 + z ** 2 <= 1
    if sf == 1:
        return torch.ones_like(x, dtype=torch.bool)
    if sf == 0.1:
        return (y ** 2 + z ** 2 <= 1) & (torch.abs(x) <= 1)
    if sf == 0.2:
        return (torch.sqrt(x ** 2 + z ** 2) - 2) ** 2 + y ** 2 <= 1
    return torch.ones_like(x, dtype=torch.bool)


def make_test_grid_2d(domain, cfg, n=101):
    lin = torch.linspace(-1, 1, n)
    X, Y = torch.meshgrid(lin, lin, indexing='ij')
    mask = _inside_2d(X.flatten(), Y.flatten(), domain)
    xf, yf = X.flatten()[mask], Y.flatten()[mask]
    sol = exact_2d(xf, yf, cfg).unsqueeze(1)
    coords = torch.stack([xf, yf], dim=1) / cfg['coord_scale'] + cfg['coord_offset']
    test_x = torch.cat([coords, torch.zeros(coords.shape[0], 1)], dim=1)
    return test_x, sol


def make_test_grid_3d(domain, cfg, n=51):
    lo, hi = -1.0, 1.0
    if cfg['shape_flag'] == 0.2:
        lo, hi = -3.0, 3.0
    lin = torch.linspace(lo, hi, n)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')
    xf, yf, zf = X.flatten(), Y.flatten(), Z.flatten()
    mask = _inside_3d(xf, yf, zf, cfg)
    xf, yf, zf = xf[mask], yf[mask], zf[mask]
    sol = exact_3d(xf, yf, zf, cfg).unsqueeze(1)
    s, o = cfg['coord_scale'], cfg['coord_offset']
    coords = torch.stack([xf, yf, zf], dim=1) / s + o
    test_x = torch.cat([coords, torch.zeros(coords.shape[0], 1)], dim=1)
    return test_x, sol


# =====================================================================
# Data loading / preparation  (single pass: load + FEM MSE + train)
# =====================================================================

def load_domain_data(domain, cfg, data_dir, add_exact_inf=True,
                     pinns_checkpoint=None, pinns_file=None, npy_dir=None):
    """
    Load FEM ``.mat`` files, compute FEM-vs-exact MSE per p, and
    build the training arrays — all in one pass.

    Returns:
        ``(X_train, Y_train, fem_p_list, fem_mse_list, per_p)``
        or ``(None, None, None, None, None)`` when no files are found.
    """
    mat_path = os.path.join(data_dir, cfg['data_subdir'])
    s, o = cfg['coord_scale'], cfg['coord_offset']
    sub = cfg.get('subsample')
    dim = cfg['dim']

    per_p = load_mat_files_per_p(mat_path, subsample=sub)
    if not per_p:
        return None, None, None, None, None

    X_list, Y_list = [], []
    fem_p_list, fem_mse_list = [], []

    for entry in per_p:
        p, pts, sol = entry['p'], entry['pts'], entry['sol']

        if dim == 2:
            sol_exact = exact_2d(
                torch.tensor(pts[:, 0], dtype=torch.float32),
                torch.tensor(pts[:, 1], dtype=torch.float32), cfg,
            ).numpy().reshape(-1, 1)
        else:
            sol_exact = exact_3d(
                torch.tensor(pts[:, 0], dtype=torch.float32),
                torch.tensor(pts[:, 1], dtype=torch.float32),
                torch.tensor(pts[:, 2], dtype=torch.float32), cfg,
            ).numpy().reshape(-1, 1)

        mse = float(np.mean((sol - sol_exact) ** 2))
        fem_p_list.append(p)
        fem_mse_list.append(mse)
        print(f"  Loaded p={p}: {pts.shape[0]} points, mse: {mse:.6e}")

        pts_norm = pts / s + o
        p_col = np.full((pts.shape[0], 1), p / 500.0, dtype=np.float32)
        X_list.append(np.hstack([pts_norm, p_col]).astype(np.float32))
        Y_list.append(sol.astype(np.float32))

    last_pts = per_p[-1]['pts']
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    if add_exact_inf and last_pts is not None:
        pts_norm = (last_pts / s + o).astype(np.float32)
        p_col = np.ones((pts_norm.shape[0], 1), dtype=np.float32)
        X_inf = np.hstack([pts_norm, p_col])

        if pinns_checkpoint:
            cache_dir = npy_dir or 'outputs/npy'
            os.makedirs(cache_dir, exist_ok=True)
            cache = os.path.join(cache_dir, f'pinns_inf_boundary_{domain}.npy')
            Y_inf = infer_pinns_on_grid(
                pinns_checkpoint, last_pts, cache_path=cache,
            )
        else:
            pf = pinns_file or os.path.join(mat_path, 'inf_pinns.npy')
            if pf and os.path.exists(pf):
                Y_inf = np.load(pf).reshape(-1, 1).astype(np.float32)
            elif dim == 2:
                Y_inf = exact_2d(
                    torch.tensor(last_pts[:, 0]),
                    torch.tensor(last_pts[:, 1]), cfg,
                ).numpy().reshape(-1, 1)
            else:
                Y_inf = exact_3d(
                    torch.tensor(last_pts[:, 0]),
                    torch.tensor(last_pts[:, 1]),
                    torch.tensor(last_pts[:, 2]), cfg,
                ).numpy().reshape(-1, 1)

        if dim == 2:
            sol_exact_inf = exact_2d(
                torch.tensor(last_pts[:, 0], dtype=torch.float32),
                torch.tensor(last_pts[:, 1], dtype=torch.float32), cfg,
            ).numpy().reshape(-1, 1)
        else:
            sol_exact_inf = exact_3d(
                torch.tensor(last_pts[:, 0], dtype=torch.float32),
                torch.tensor(last_pts[:, 1], dtype=torch.float32),
                torch.tensor(last_pts[:, 2], dtype=torch.float32), cfg,
            ).numpy().reshape(-1, 1)
        mse_inf = float(np.mean((Y_inf - sol_exact_inf) ** 2))
        print(f"  Added p=∞ point: {X_inf.shape[0]} pts, mse: {mse_inf:.6e}")

        X = np.vstack([X, X_inf])
        Y = np.vstack([Y, Y_inf])

    return X, Y, fem_p_list, fem_mse_list, per_p


# =====================================================================
# Single-domain runner
# =====================================================================

def run_experiment(domain, args):
    cfg = ALL_DOMAINS[domain]
    dim = cfg['dim']

    run_tag = determine_run_tag(args)
    base = os.path.join(args.output_dir, 'expr_6_4_2', domain, run_tag)
    setup_logging(base)
    output_dirs = create_output_dirs(base)

    print(f"\n{'=' * 60}")
    print(f"Dist-to-Boundary  —  {cfg['name']}  ({dim}-D)")
    print(f"{'=' * 60}")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ---- data ----------------------------------------------------------
    X_train, Y_train, fem_p_list, fem_mse_list, per_p = load_domain_data(
        domain, cfg, args.data_dir,
        add_exact_inf=args.add_exact_inf,
        pinns_checkpoint=args.pinns_checkpoint,
        pinns_file=args.pinns_file,
        npy_dir=output_dirs['npy'],
    )
    if X_train is None:
        print(f"  !! No data found for {domain} — skipping.")
        return None, None, None
    print(f"Training data: X={X_train.shape}  Y={Y_train.shape}")

    loader = DataLoader(
        DeepONetDataset(X_train, Y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    # ---- test data -----------------------------------------------------
    trunk = [int(x) for x in args.trunk_layers.split(',')]
    branch = [int(x) for x in args.branch_layers.split(',')]

    if dim == 2:
        if trunk[0] != 2:
            trunk[0] = 2
        test_x, test_y = make_test_grid_2d(domain, cfg)
        p_col_idx = 2
    else:
        if trunk[0] != 3:
            trunk[0] = 3
        test_x, test_y = make_test_grid_3d(domain, cfg)
        p_col_idx = 3

    test_data = []
    for pv, lbl in [(150, 'p150'), (200, 'p200'), (250, 'p250'), (500, 'p500')]:
        tx = test_x.clone()
        tx[:, p_col_idx] = pv / 500.0
        test_data.append((tx, test_y, lbl))

    # ---- model ---------------------------------------------------------
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
    test_x_500[:, p_col_idx] = 500.0 / 500.0
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
    ev = evaluate_over_p_range(
        model, test_x, test_y, device, p_col_idx=p_col_idx,
    )
    for i, (p, mse) in enumerate(zip(ev['p'], ev['mse'])):
        if i % 20 == 0:
            print(f"  p={p:3d}:  MSE={mse:.6e}")

    # ---- prediction vs FEM training data per p -------------------------
    s, o = cfg['coord_scale'], cfg['coord_offset']
    pred_train_p, pred_train_mse = [], []
    model.eval()
    with torch.no_grad():
        for entry in per_p:
            p, pts, sol = entry['p'], entry['pts'], entry['sol']
            pts_norm = pts / s + o
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
    plot_training_history(history, f'6.4.2 {cfg["name"]}',
                          os.path.join(output_dirs['plots'], 'training.png'))
    plot_mse_vs_p(ev['p'], ev['mse'], f'6.4.2 {cfg["name"]}',
                  os.path.join(output_dirs['plots'], 'mse_vs_p.png'))

    if dim == 2:
        for pv in (200, 500):
            _plot_2d_domain(model, domain, cfg, device, output_dirs['plots'], pv)

    return model, history, ev


def _plot_2d_domain(model, domain, cfg, device, out_dir, p_value, n=101):
    lin = torch.linspace(-1, 1, n)
    X, Y = torch.meshgrid(lin, lin, indexing='ij')
    mask = _inside_2d(X, Y, domain)

    s, o = cfg['coord_scale'], cfg['coord_offset']
    inp = torch.stack([
        X.flatten() / s + o,
        Y.flatten() / s + o,
        torch.full((n * n,), p_value / 500.0),
    ], dim=1)

    model.eval()
    with torch.no_grad():
        pred = model(inp.to(device)).cpu().squeeze().reshape(n, n).numpy()
    exact = exact_2d(X.flatten(), Y.flatten(), cfg).reshape(n, n).numpy()

    plot_prediction_2d(
        pred, exact, X, Y,
        title=f'{cfg["name"]} p={p_value}',
        output_path=os.path.join(out_dir, f'pred_p{p_value}.png'),
        mask=mask,
    )


# =====================================================================
# CLI
# =====================================================================

def main():
    all_keys = list(ALL_DOMAINS.keys())
    ap = argparse.ArgumentParser(description='Dist-to-Boundary DeepONet')
    ap.add_argument('--domain', default='disc',
                    choices=all_keys + ['all-2d', 'all-3d', 'all'])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--trunk-layers', default='2,512,512,512,128',
                    help='Default for 2-D; auto-adjusted to 3 for 3-D domains')
    ap.add_argument('--branch-layers', default='1,128,128,128,128')
    ap.add_argument('--data-dir', default='./data/distance_to_boundary')
    ap.add_argument('--output-dir', default='outputs')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--no-exact-inf', dest='add_exact_inf', action='store_false')
    ap.add_argument('--pinns-checkpoint', default=None,
                    help='Path to Eikonal PINN .pt checkpoint (from experiment_6_4_eikonal.py). '
                         'Runs inference on FEM grid, caches to .npy.')
    ap.add_argument('--pinns-file', default=None,
                    help='Path to pre-saved PINNs prediction (.npy). '
                         'Skipped if --pinns-checkpoint is given.')
    ap.add_argument('--run-tag', default='auto',
                    help='Tag for MSE output files (auto-detected from flags '
                         'when "auto": no_inf / exact_inf / pinns_inf)')
    args = ap.parse_args()

    if args.domain == 'all-2d':
        domains = list(DOMAINS_2D.keys())
    elif args.domain == 'all-3d':
        domains = list(DOMAINS_3D.keys())
    elif args.domain == 'all':
        domains = list(ALL_DOMAINS.keys())
    else:
        domains = [args.domain]

    for d in domains:
        run_experiment(d, args)

    print("\nAll experiments completed!")


if __name__ == '__main__':
    main()
