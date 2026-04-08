"""
Experiment 6.2: 2D p-Laplacian Equation — Aronsson Example (Homogeneous, Dirichlet)

    Δ_p u = div(|∇u|^(p-2) ∇u) = 0  in Ω
    u = g  on ∂Ω

Supports:
    - Simple mode (single p) and iterative mode (progressive p ramp)
    - Square [-1,1]² or unit disc domain
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt

from PINNs_code import (
    BasePINN, create_output_dirs, setup_logging, set_seed, get_device,
    InteriorSampler, split_data, train_pinn, train_pinn_iterative,
    plot_results, plot_pde_loss_distribution,
    generate_square_domain_data, generate_disc_domain_data,
    FixedAlphaSchedule, AlphaPlateauScheduler, ReLoBRaLo,
    parse_alpha_schedule,
)

torch.set_default_dtype(torch.float32)
SEED = 1234
set_seed(SEED)
device = get_device()
print(f"Using device: {device}")
print(f"Random seed: {SEED}")


# ============================================================================
# Exact solution
# ============================================================================

def aronsson_solution(x, y):
    return torch.abs(x) ** (4 / 3) - torch.abs(y) ** (4 / 3)


# ============================================================================
# PDE model
# ============================================================================

class PLaplacianPINN(BasePINN):
    """
    PINN for the p-Laplacian equation.

    Stores ``p`` as an attribute so that ``loss_pde(x)`` works without extra
    arguments (required by the shared training loop).
    """

    def __init__(self, hidden_layers=None, activation='tanh', eta=1e-5,
                 clip_residual=0.0, outlier_percentile=1.0, clip_grad_norm=0.0,
                 p=10):
        super().__init__(hidden_layers=hidden_layers, activation=activation,
                         clip_residual=clip_residual,
                         outlier_percentile=outlier_percentile)
        self.eta = eta
        self.clip_grad_norm = clip_grad_norm
        self.p = p

    def compute_pde_residual(self, x):
        """Regularised p-Laplacian residual using ``self.p``."""
        return self._compute_p_laplacian(x, self.p)

    def _compute_p_laplacian(self, x, p):
        g = x.clone()
        g.requires_grad = True
        u = self.forward(g)

        grad_u = autograd.grad(u, g, torch.ones_like(u),
                               retain_graph=True, create_graph=True)[0]
        u_x = grad_u[:, [0]]
        u_y = grad_u[:, [1]]

        grad_ux = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0]
        grad_uy = autograd.grad(u_y, g, torch.ones_like(u_y), create_graph=True)[0]

        u_xx = grad_ux[:, [0]]
        u_xy = grad_ux[:, [1]]
        u_yx = grad_uy[:, [0]]
        u_yy = grad_uy[:, [1]]

        grad_norm_sq = u_x ** 2 + u_y ** 2
        if self.clip_grad_norm > 0:
            grad_norm_sq = torch.clamp(grad_norm_sq, max=self.clip_grad_norm)
        gamma = (self.eta ** 2 + grad_norm_sq) ** ((p - 4) / 2)

        delta_p = gamma * (
            (p - 1) * u_x ** 2 * u_xx +
            (p - 1) * u_y ** 2 * u_yy +
            (p - 2) * u_x * u_y * (u_xy + u_yx) +
            u_x ** 2 * u_yy +
            u_y ** 2 * u_xx
        )
        return delta_p

    # -- diagnostic losses (raw, no eta / outlier) ----------------------------

    def compute_infinity_laplacian(self, x):
        g = x.clone()
        g.requires_grad = True
        u = self.forward(g)
        grad_u = autograd.grad(u, g, torch.ones_like(u),
                               retain_graph=True, create_graph=True)[0]
        u_x, u_y = grad_u[:, [0]], grad_u[:, [1]]
        grad_ux = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0]
        grad_uy = autograd.grad(u_y, g, torch.ones_like(u_y), create_graph=True)[0]
        u_xx, u_xy = grad_ux[:, [0]], grad_ux[:, [1]]
        u_yx, u_yy = grad_uy[:, [0]], grad_uy[:, [1]]
        return u_x ** 2 * u_xx + u_x * u_y * (u_xy + u_yx) + u_y ** 2 * u_yy

    def _compute_p_laplacian_raw(self, x, p):
        g = x.clone()
        g.requires_grad = True
        u = self.forward(g)
        grad_u = autograd.grad(u, g, torch.ones_like(u),
                               retain_graph=True, create_graph=True)[0]
        u_x, u_y = grad_u[:, [0]], grad_u[:, [1]]
        grad_ux = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0]
        grad_uy = autograd.grad(u_y, g, torch.ones_like(u_y), create_graph=True)[0]
        u_xx, u_xy = grad_ux[:, [0]], grad_ux[:, [1]]
        u_yx, u_yy = grad_uy[:, [0]], grad_uy[:, [1]]
        grad_norm_sq = u_x ** 2 + u_y ** 2
        gamma = grad_norm_sq ** ((p - 4) / 2)
        return gamma * (
            (p - 1) * u_x ** 2 * u_xx +
            (p - 1) * u_y ** 2 * u_yy +
            (p - 2) * u_x * u_y * (u_xy + u_yx) +
            u_x ** 2 * u_yy +
            u_y ** 2 * u_xx
        )

    def loss_pde_p_raw(self, x_pde, p):
        res = self._compute_p_laplacian_raw(x_pde, p)
        return self.loss_function(res, torch.zeros_like(res))

    def _compute_normalized_p_laplacian(self, x, p):
        """Normalized p-Laplacian: Δ_p^N u = (1/p) Δ_1 u + ((p-2)/p) Δ_∞^N u."""
        g = x.clone()
        g.requires_grad = True
        u = self.forward(g)
        grad_u = autograd.grad(u, g, torch.ones_like(u),
                               retain_graph=True, create_graph=True)[0]
        u_x, u_y = grad_u[:, [0]], grad_u[:, [1]]
        grad_ux = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0]
        grad_uy = autograd.grad(u_y, g, torch.ones_like(u_y), create_graph=True)[0]
        u_xx, u_xy = grad_ux[:, [0]], grad_ux[:, [1]]
        u_yx, u_yy = grad_uy[:, [0]], grad_uy[:, [1]]
        grad_norm_sq = u_x ** 2 + u_y ** 2 + 1e-12
        laplacian = u_xx + u_yy
        inf_lap_n = (u_x ** 2 * u_xx + u_x * u_y * (u_xy + u_yx)
                     + u_y ** 2 * u_yy) / grad_norm_sq
        return laplacian / p + (p - 2) / p * inf_lap_n

    def loss_pde_p_normalized(self, x_pde, p):
        res = self._compute_normalized_p_laplacian(x_pde, p)
        return self.loss_function(res, torch.zeros_like(res))

    def loss_pde_infinity(self, x_pde):
        res = self.compute_infinity_laplacian(x_pde)
        return self.loss_function(res, torch.zeros_like(res))

    def loss_pde_infinity_normalized(self, x_pde):
        """Normalized infinity-Laplacian: Δ_∞^N u = (u_x² u_xx + 2 u_x u_y u_xy + u_y² u_yy) / |∇u|²."""
        g = x_pde.clone()
        g.requires_grad = True
        u = self.forward(g)
        grad_u = autograd.grad(u, g, torch.ones_like(u),
                               retain_graph=True, create_graph=True)[0]
        u_x, u_y = grad_u[:, [0]], grad_u[:, [1]]
        grad_ux = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0]
        grad_uy = autograd.grad(u_y, g, torch.ones_like(u_y), create_graph=True)[0]
        u_xx, u_xy = grad_ux[:, [0]], grad_ux[:, [1]]
        u_yx, u_yy = grad_uy[:, [0]], grad_uy[:, [1]]
        grad_norm_sq = u_x ** 2 + u_y ** 2 + 1e-12
        res = (u_x ** 2 * u_xx + u_x * u_y * (u_xy + u_yx)
               + u_y ** 2 * u_yy) / grad_norm_sq
        return self.loss_function(res, torch.zeros_like(res))


# ============================================================================
# run_experiment
# ============================================================================

def run_experiment(domain='square', mode='simple', p=10,
                   p_start=2, p_end=100, p_steps=10, p_values_list=None,
                   epochs=100, epochs_per_p=50,
                   hidden_layers=None, activation='tanh', lr=1e-3,
                   alpha_scheduler=None, eta=1e-5,
                   n_interior=1000, n_boundary=10000,
                   batch_size_bc=400, batch_size_pde=1000,
                   output_dir='outputs', val_split=0.2, patience=None,
                   resample=True, scheduler_type='cosine', scheduler_end_factor=0.01,
                   clip_residual=10.0, clip_grad_norm=1.0, outlier_percentile=2.0,
                   outlier_off_epoch=None, plot_pde_every=0,
                   save_npy=False, bc_loss_threshold=None, pde_loss_threshold=None,
                   resume_checkpoint=None, run_tag=None):
    if hidden_layers is None:
        hidden_layers = [128, 128, 128, 128]

    if mode == 'simple':
        example_dir = f'p{p}_{domain}'
        expr_folder = 'expr_7_2_simple'
    else:
        if p_values_list is not None:
            example_dir = f'p{p_values_list[0]}to{p_values_list[-1]}_{domain}'
        else:
            example_dir = f'p{p_start}to{p_end}_{domain}'
        expr_folder = 'expr_7_2_iter'

    base = os.path.join(output_dir, expr_folder, example_dir)
    if run_tag:
        base = os.path.join(base, run_tag)
    setup_logging(base)

    print(f"\n{'=' * 60}")
    print(f"p-Laplacian Aronsson Example")
    print(f"{'=' * 60}\n")

    output_dirs = create_output_dirs(base)
    f_exact = aronsson_solution

    if domain == 'square':
        domain_type, domain_params = 'square', {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_square_domain_data(
            **domain_params, n_interior_grid=n_interior, n_boundary_grid=n_boundary,
            f_exact=f_exact, seed=SEED)
        domain_name = "Square [-1,1]²"
    else:
        domain_type, domain_params = 'disc', {'radius': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_disc_domain_data(
            radius=1.0, n_boundary=n_boundary, n_interior_grid=n_interior,
            f_exact=f_exact, seed=SEED)
        domain_name = "Unit Disc"

    interior_sampler = None
    if resample:
        interior_sampler = InteriorSampler(domain_type, domain_params, len(x_interior))
        print(f"Re-sampling enabled: ~{len(x_interior):,} interior points per epoch")

    x_bc_val, y_bc_val, x_interior_val = None, None, None
    if val_split > 0:
        print(f"\nSplitting data: {1 - val_split:.0%} train, {val_split:.0%} validation")
        x_bc, x_bc_val, y_bc, y_bc_val = split_data(x_bc, y_bc, train_ratio=1 - val_split, seed=SEED)
        x_interior, x_interior_val, _, _ = split_data(x_interior, None, train_ratio=1 - val_split, seed=SEED + 1)
        print(f"  BC: {len(x_bc):,} train, {len(x_bc_val):,} val")
        print(f"  Interior: {len(x_interior):,} train, {len(x_interior_val):,} val")

    model = PLaplacianPINN(hidden_layers=hidden_layers, activation=activation, eta=eta,
                            clip_residual=clip_residual, outlier_percentile=outlier_percentile,
                            clip_grad_norm=clip_grad_norm, p=p)
    model.to(device)

    resume_from_p = None
    if resume_checkpoint is not None:
        print(f"\nLoading checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'history' in ckpt and 'p' in ckpt['history']:
            completed_p = [v for v in ckpt['history']['p'] if v is not None]
            if completed_p:
                print(f"  Checkpoint trained up to p={completed_p[-1]}")

    print(f"\nModel: PLaplacianPINN on {domain_name}")
    print(f"  Hidden layers: {hidden_layers}, activation: {activation}")
    print(f"  η={eta}, clip_residual={clip_residual}, clip_grad_norm={clip_grad_norm}")
    if outlier_percentile > 0:
        print(f"  Outlier removal: top {outlier_percentile}%")
    print(f"  Parameters: {sum(pp.numel() for pp in model.parameters()):,}")

    if mode == 'simple':
        example_name = f'plaplacian_p{p}_{domain}'
        title = f"p-Laplacian (p={p}) — Aronsson on {domain_name}"

        history, training_time = train_pinn(
            model, x_bc, y_bc, x_interior, x_test, y_test,
            n_epochs=epochs, batch_size_bc=batch_size_bc, batch_size_pde=batch_size_pde,
            lr=lr, alpha_scheduler=alpha_scheduler,
            x_bc_val=x_bc_val, y_bc_val=y_bc_val, x_interior_val=x_interior_val,
            patience=patience, interior_sampler=interior_sampler,
            scheduler_type=scheduler_type, scheduler_end_factor=scheduler_end_factor,
            plot_pde_every=plot_pde_every, example_name=example_name,
            output_dirs=output_dirs, outlier_off_epoch=outlier_off_epoch,
            bc_loss_threshold=bc_loss_threshold, pde_loss_threshold=pde_loss_threshold,
            seed=SEED, device=device,
        )
    else:
        if p_values_list is not None:
            p_values = p_values_list
        elif p_steps == 1:
            p_values = [p_end]
        else:
            p_values = np.linspace(p_start, p_end, p_steps).astype(int).tolist()
            if p_values[-1] != p_end:
                p_values[-1] = p_end

        example_name = f'plaplacian_p{p_values[0]}to{p_values[-1]}_{domain}'
        title = f"p-Laplacian (p: {p_values[0]}→{p_values[-1]}) — Aronsson on {domain_name}"

        if resume_checkpoint is not None and 'history' in ckpt and 'p' in ckpt['history']:
            completed_p = [v for v in ckpt['history']['p'] if v is not None]
            if completed_p:
                last = completed_p[-1]
                cands = [pv for pv in p_values if pv > last]
                if cands:
                    resume_from_p = cands[0]

        history, training_time = train_pinn_iterative(
            model, x_bc, y_bc, x_interior, x_test, y_test,
            p_values=p_values, epochs_per_p=epochs_per_p,
            batch_size_bc=batch_size_bc, batch_size_pde=batch_size_pde,
            lr=lr, alpha_scheduler=alpha_scheduler,
            x_bc_val=x_bc_val, y_bc_val=y_bc_val, x_interior_val=x_interior_val,
            patience=patience, interior_sampler=interior_sampler,
            scheduler_type=scheduler_type, scheduler_end_factor=scheduler_end_factor,
            plot_pde_every=plot_pde_every, example_name=example_name,
            output_dirs=output_dirs, outlier_off_epoch=outlier_off_epoch,
            original_outlier_pct=outlier_percentile,
            bc_loss_threshold=bc_loss_threshold, pde_loss_threshold=pde_loss_threshold,
            resume_from_p=resume_from_p, seed=SEED, device=device,
        )

    print(f"\nFinal Test MSE: {history['test_loss'][-1]:.4e}")

    # Inference timing
    model.eval()
    x_test_dev = x_test.to(device)
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

    # PDE test losses
    final_p = p if mode == 'simple' else (p_values_list[-1] if p_values_list else p_end)
    model.eval()
    print(f"p-Lap (p={final_p}) PDE test (raw):        {model.loss_pde_p_raw(x_test_dev, final_p).item():.4e}")
    print(f"p-Lap (p={final_p}) PDE test (normalized): {model.loss_pde_p_normalized(x_test_dev, final_p).item():.4e}")
    print(f"Inf-Lap PDE test (raw):                    {model.loss_pde_infinity(x_test_dev).item():.4e}")
    print(f"Inf-Lap PDE test (normalized):             {model.loss_pde_infinity_normalized(x_test_dev).item():.4e}")

    if save_npy:
        npy_dir = output_dirs['npy']
        history_data = {}
        for k, vals in history.items():
            if k == 'p_schedule':
                history_data[k] = np.array(vals)
                continue
            history_data[k] = np.array([v if v is not None else np.nan for v in vals])
        np.savez(os.path.join(npy_dir, 'training_history.npz'), **history_data)
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test_dev).cpu()
        np.savez(os.path.join(npy_dir, 'exact_solution.npz'),
                 x=x_test[:, 0].numpy(), y=x_test[:, 1].numpy(),
                 u_exact=y_test.numpy().flatten())
        np.savez(os.path.join(npy_dir, 'predictions.npz'),
                 x=x_test[:, 0].numpy(), y=x_test[:, 1].numpy(),
                 u_pred=y_pred.numpy().flatten())
        print(f"Saved .npz files to {npy_dir}/")

    plot_path = os.path.join(output_dirs['plots'], 'results.png')
    fig = plot_results(model, x_test, y_test, history, title, output_path=plot_path, device=device)
    plt.show()

    return model, history


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='p-Laplacian Aronsson Example with PINNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--domain', type=str, default='square', choices=['square', 'disc'])
    parser.add_argument('--mode', type=str, default='simple', choices=['simple', 'iterative'])
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--p-start', type=int, default=2)
    parser.add_argument('--p-end', type=int, default=100)
    parser.add_argument('--p-steps', type=int, default=10)
    parser.add_argument('--p-values', type=str, default=None,
                        help='Comma-separated list of p values for iterative mode')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs-per-p', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta', type=float, default=1e-5)
    parser.add_argument('--hidden-layers', type=str, default='128,128,128,128')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'leaky_relu', 'elu', 'gelu', 'softplus', 'silu', 'sigmoid'])
    parser.add_argument('--interior-grid', type=int, default=1000)
    parser.add_argument('--boundary-grid', type=int, default=10000)
    parser.add_argument('--batch-bc', type=int, default=400)
    parser.add_argument('--batch-pde', type=int, default=1000)

    parser.add_argument('--clip-residual', type=float, default=100.0)
    parser.add_argument('--clip-grad-norm', type=float, default=1.0)
    parser.add_argument('--outlier-percentile', type=float, default=2.0)
    parser.add_argument('--outlier-off-epoch', type=int, default=None)

    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--alpha-schedule', type=str, default=None,
                        help='"auto", "relobralo", or "epoch:value,..."')
    parser.add_argument('--alpha-min', type=float, default=1e-5)
    parser.add_argument('--alpha-max', type=float, default=1e-1)
    parser.add_argument('--alpha-patience', type=int, default=5)
    parser.add_argument('--alpha-cooldown', type=int, default=3)
    parser.add_argument('--alpha-min-improvement', type=float, default=0.2)
    parser.add_argument('--relobralo-temperature', type=float, default=1.0)
    parser.add_argument('--relobralo-alpha', type=float, default=0.999)
    parser.add_argument('--relobralo-rho', type=float, default=0.99)

    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--bc-loss-threshold', type=float, default=None)
    parser.add_argument('--pde-loss-threshold', type=float, default=None)
    parser.add_argument('--no-resample', action='store_true')

    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine', 'step', 'exponential', 'none'])
    parser.add_argument('--scheduler-end-factor', type=float, default=0.01)
    parser.add_argument('--plot-pde-every', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='Ablation tag; creates a subdirectory under the example folder')
    parser.add_argument('--save-npy', action='store_true')
    parser.add_argument('--resume-checkpoint', type=str, default=None)

    args = parser.parse_args()

    if args.scheduler == 'none':
        args.scheduler = None

    alpha_scheduler = None
    if args.alpha_schedule is not None:
        s = args.alpha_schedule.strip().lower()
        if s == 'auto':
            alpha_scheduler = AlphaPlateauScheduler(
                alpha_init=args.alpha_min, alpha_max=args.alpha_max,
                patience=args.alpha_patience, cooldown=args.alpha_cooldown,
                min_relative_improvement=args.alpha_min_improvement)
        elif s == 'relobralo':
            alpha_scheduler = ReLoBRaLo(
                num_losses=2, alpha=args.relobralo_alpha,
                temperature=args.relobralo_temperature,
                rho=args.relobralo_rho, device=device)
        else:
            alpha_scheduler = FixedAlphaSchedule(parse_alpha_schedule(s))
    elif args.alpha is not None:
        alpha_scheduler = FixedAlphaSchedule({0: args.alpha})
    else:
        alpha_scheduler = AlphaPlateauScheduler(
            alpha_init=args.alpha_min, alpha_max=args.alpha_max,
            patience=args.alpha_patience, cooldown=args.alpha_cooldown,
            min_relative_improvement=args.alpha_min_improvement)

    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]
    p_values_list = None
    if args.p_values:
        p_values_list = [int(x.strip()) for x in args.p_values.split(',')]

    run_experiment(
        domain=args.domain, mode=args.mode, p=args.p,
        p_start=args.p_start, p_end=args.p_end, p_steps=args.p_steps,
        p_values_list=p_values_list,
        epochs=args.epochs, epochs_per_p=args.epochs_per_p,
        hidden_layers=hidden_layers, activation=args.activation,
        lr=args.lr, alpha_scheduler=alpha_scheduler, eta=args.eta,
        n_interior=args.interior_grid, n_boundary=args.boundary_grid,
        batch_size_bc=args.batch_bc, batch_size_pde=args.batch_pde,
        output_dir=args.output_dir, val_split=args.val_split,
        patience=args.patience, resample=not args.no_resample,
        scheduler_type=args.scheduler, scheduler_end_factor=args.scheduler_end_factor,
        clip_residual=args.clip_residual, clip_grad_norm=args.clip_grad_norm,
        outlier_percentile=args.outlier_percentile,
        outlier_off_epoch=args.outlier_off_epoch,
        plot_pde_every=args.plot_pde_every, save_npy=args.save_npy,
        bc_loss_threshold=args.bc_loss_threshold,
        pde_loss_threshold=args.pde_loss_threshold,
        resume_checkpoint=args.resume_checkpoint,
        run_tag=args.run_tag)
