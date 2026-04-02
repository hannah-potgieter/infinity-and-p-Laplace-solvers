"""
Experiment 6.1: 2D Infinity-Laplacian Equation with Dirichlet Boundary Conditions

    Δ_∞ u = u_x² u_xx + 2 u_x u_y u_xy + u_y² u_yy = 0  in Ω
    u = g  on ∂Ω

Examples:
    1. Arctan:   u(x,y) = arctan(y/x) on [0.01,1]²
    2. Absolute: u(x,y) = |x| - |y| on [-1,1]²
    3. Aronsson: u(x,y) = |x|^(4/3) - |y|^(4/3) on [-1,1]² or unit disc
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt

from PINNs_code import (
    BasePINN, create_output_dirs, setup_logging, set_seed, get_device,
    InteriorSampler, split_data, train_pinn,
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
# Exact solutions
# ============================================================================

def arctan_solution(x, y):
    return torch.atan(y / x)


def absolute_solution(x, y):
    return torch.abs(x) - torch.abs(y)


def aronsson_solution(x, y):
    return torch.abs(x) ** (4 / 3) - torch.abs(y) ** (4 / 3)


# ============================================================================
# PDE model
# ============================================================================

class InfinityLaplacianPINN(BasePINN):
    """PINN for the infinity-Laplacian equation."""

    def __init__(self, hidden_layers=None, activation='tanh', eta=0.0,
                 clip_residual=0.0, outlier_percentile=1.0):
        super().__init__(hidden_layers=hidden_layers, activation=activation,
                         clip_residual=clip_residual,
                         outlier_percentile=outlier_percentile)
        self.eta = eta

    def compute_pde_residual(self, x):
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

        delta_inf = u_x ** 2 * u_xx + u_x * u_y * (u_xy + u_yx) + u_y ** 2 * u_yy

        if self.eta > 0:
            grad_norm_sq = u_x ** 2 + u_y ** 2
            delta_inf = delta_inf / (self.eta ** 2 + grad_norm_sq)

        return delta_inf


# ============================================================================
# Domain decomposition (absolute value example)
# ============================================================================

class DomainDecompositionPINN(nn.Module):
    """
    4-quadrant decomposition for u(x,y) = |x| - |y| on [-1,1]².

    Each quadrant gets its own sub-network.  Interface continuity (C0)
    is enforced along x=0 and y=0.
    """

    def __init__(self, hidden_layers=None, activation='tanh', eta=0.0,
                 clip_residual=0.0, outlier_percentile=0.0,
                 n_interface_pts=200, interface_weight=1.0,
                 domain_bounds=(-1.0, 1.0, -1.0, 1.0)):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [32, 32, 32]
        self.subnets = nn.ModuleList([
            InfinityLaplacianPINN(hidden_layers, activation, eta,
                                  clip_residual, outlier_percentile)
            for _ in range(4)
        ])
        self.n_interface_pts = n_interface_pts
        self.interface_weight = interface_weight
        self.loss_function = nn.MSELoss(reduction='mean')
        self._outlier_percentile = outlier_percentile
        self.x_min, self.x_max, self.y_min, self.y_max = domain_bounds

    @property
    def outlier_percentile(self):
        return self._outlier_percentile

    @outlier_percentile.setter
    def outlier_percentile(self, value):
        self._outlier_percentile = value
        for subnet in self.subnets:
            subnet.outlier_percentile = value

    def _get_quadrant(self, x):
        qx_pos = x[:, 0] >= 0
        qy_pos = x[:, 1] >= 0
        q = torch.zeros(len(x), dtype=torch.long, device=x.device)
        q[qx_pos & qy_pos] = 0
        q[~qx_pos & qy_pos] = 1
        q[~qx_pos & ~qy_pos] = 2
        q[qx_pos & ~qy_pos] = 3
        return q

    def forward(self, x):
        q = self._get_quadrant(x)
        out = torch.zeros(len(x), 1, device=x.device, dtype=x.dtype)
        for i in range(4):
            mask = q == i
            if mask.any():
                out[mask] = self.subnets[i](x[mask])
        return out

    def loss_boundary(self, x_bc, y_bc):
        q = self._get_quadrant(x_bc)
        losses = []
        for i in range(4):
            mask = q == i
            if mask.any():
                losses.append(self.loss_function(self.subnets[i](x_bc[mask]), y_bc[mask]))
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=x_bc.device, requires_grad=True)

    def loss_pde(self, x_pde):
        q = self._get_quadrant(x_pde)
        losses = []
        for i in range(4):
            mask = q == i
            if mask.any():
                losses.append(self.subnets[i].loss_pde(x_pde[mask]))
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=x_pde.device, requires_grad=True)

    def loss_interface(self):
        n = self.n_interface_pts
        dev = next(self.parameters()).device
        total = torch.tensor(0.0, device=dev)

        y_vals = torch.linspace(0, self.y_max, n, device=dev)
        pts = torch.stack([torch.zeros(n, device=dev), y_vals], dim=1)
        total = total + self.loss_function(self.subnets[0](pts), self.subnets[1](pts))

        y_vals = torch.linspace(self.y_min, 0, n, device=dev)
        pts = torch.stack([torch.zeros(n, device=dev), y_vals], dim=1)
        total = total + self.loss_function(self.subnets[3](pts), self.subnets[2](pts))

        x_vals = torch.linspace(0, self.x_max, n, device=dev)
        pts = torch.stack([x_vals, torch.zeros(n, device=dev)], dim=1)
        total = total + self.loss_function(self.subnets[0](pts), self.subnets[3](pts))

        x_vals = torch.linspace(self.x_min, 0, n, device=dev)
        pts = torch.stack([x_vals, torch.zeros(n, device=dev)], dim=1)
        total = total + self.loss_function(self.subnets[1](pts), self.subnets[2](pts))

        return self.interface_weight * total / 4.0

    def get_pde_loss_with_outlier_info(self, x_pde):
        q = self._get_quadrant(x_pde)
        all_loss = torch.zeros(len(x_pde), device=x_pde.device)
        for i in range(4):
            mask = q == i
            if mask.any():
                with torch.enable_grad():
                    res = self.subnets[i].compute_pde_residual(x_pde[mask])
                res = res.detach()
                if self.subnets[i].clip_residual > 0:
                    res = torch.clamp(res, -self.subnets[i].clip_residual, self.subnets[i].clip_residual)
                all_loss[mask] = res.squeeze() ** 2

        ppl = all_loss.cpu().numpy()
        xc = x_pde[:, 0].detach().cpu().numpy()
        yc = x_pde[:, 1].detach().cpu().numpy()

        if self._outlier_percentile > 0:
            thr = np.percentile(ppl, 100.0 - self._outlier_percentile)
            is_out = ppl > thr
        else:
            thr = np.max(ppl) + 1
            is_out = np.zeros(len(ppl), dtype=bool)
        return xc, yc, ppl, is_out, thr


# ============================================================================
# run_example
# ============================================================================

def run_example(example_name, n_epochs=10, n_interior_grid=1000, n_boundary_grid=10000,
                batch_size_bc=400, batch_size_pde=1000, save_plots=True,
                val_split=0.0, patience=None, alpha_scheduler=None,
                resample=True, activation='tanh', hidden_layers=None, eta=0.0,
                scheduler_type=None, scheduler_end_factor=0.01, lr=1e-3,
                clip_residual=0.0, outlier_percentile=1.0, plot_pde_every=0,
                output_dir='outputs', outlier_off_epoch=None,
                domain_decomposition=False, n_interface_pts=200, interface_weight=1.0,
                interface_every=5, save_npy=False, run_tag=None):
    if hidden_layers is None:
        hidden_layers = [128, 128, 128, 128]

    label = example_name + ('_dd' if domain_decomposition else '')
    base = os.path.join(output_dir, 'expr_7_1', label)
    if run_tag:
        base = os.path.join(base, run_tag)
    setup_logging(base)

    print(f"\n{'=' * 60}")
    print(f"Running Example: {example_name}")
    print(f"{'=' * 60}\n")

    output_dirs = create_output_dirs(base)

    if example_name == 'arctan':
        f_exact = arctan_solution
        domain_type, domain_params = 'square', {'x_min': 0.01, 'x_max': 1.0, 'y_min': 0.01, 'y_max': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_square_domain_data(
            **domain_params, n_interior_grid=n_interior_grid, n_boundary_grid=n_boundary_grid,
            f_exact=f_exact, seed=SEED)
        title = "Arctan: u(x,y) = arctan(y/x)"
    elif example_name == 'absolute':
        f_exact = absolute_solution
        domain_type, domain_params = 'square', {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_square_domain_data(
            **domain_params, n_interior_grid=n_interior_grid, n_boundary_grid=n_boundary_grid,
            f_exact=f_exact, seed=SEED)
        title = "Absolute: u(x,y) = |x| - |y|"
    elif example_name == 'aronsson_square':
        f_exact = aronsson_solution
        domain_type, domain_params = 'square', {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_square_domain_data(
            **domain_params, n_interior_grid=n_interior_grid, n_boundary_grid=n_boundary_grid,
            f_exact=f_exact, seed=SEED)
        title = "Aronsson (Square): u = |x|^(4/3) - |y|^(4/3)"
    elif example_name == 'aronsson_disc':
        f_exact = aronsson_solution
        domain_type, domain_params = 'disc', {'radius': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_disc_domain_data(
            radius=1.0, n_boundary=n_boundary_grid, n_interior_grid=n_interior_grid,
            f_exact=f_exact, seed=SEED)
        title = "Aronsson (Disc): u = |x|^(4/3) - |y|^(4/3)"
    else:
        raise ValueError(f"Unknown example: {example_name}")

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

    if domain_decomposition:
        bounds = (domain_params.get('x_min', -1.0), domain_params.get('x_max', 1.0),
                  domain_params.get('y_min', -1.0), domain_params.get('y_max', 1.0))
        model = DomainDecompositionPINN(
            hidden_layers=hidden_layers, activation=activation, eta=eta,
            clip_residual=clip_residual, outlier_percentile=outlier_percentile,
            n_interface_pts=n_interface_pts, interface_weight=interface_weight,
            domain_bounds=bounds)
    else:
        model = InfinityLaplacianPINN(
            hidden_layers=hidden_layers, activation=activation, eta=eta,
            clip_residual=clip_residual, outlier_percentile=outlier_percentile)
    model.to(device)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Activation: {activation}")
    if eta > 0:
        print(f"  η: {eta} (normalized)")
    if clip_residual > 0:
        print(f"  Residual clipping: [-{clip_residual}, {clip_residual}]")
    if outlier_percentile > 0:
        print(f"  Outlier removal: top {outlier_percentile}%")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history, training_time = train_pinn(
        model, x_bc, y_bc, x_interior, x_test, y_test,
        n_epochs=n_epochs, batch_size_bc=batch_size_bc, batch_size_pde=batch_size_pde,
        lr=lr, alpha_scheduler=alpha_scheduler,
        x_bc_val=x_bc_val, y_bc_val=y_bc_val, x_interior_val=x_interior_val,
        patience=patience, interior_sampler=interior_sampler,
        scheduler_type=scheduler_type, scheduler_end_factor=scheduler_end_factor,
        plot_pde_every=plot_pde_every, example_name=example_name,
        output_dirs=output_dirs, outlier_off_epoch=outlier_off_epoch,
        interface_every=interface_every, seed=SEED, device=device,
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

    if save_npy:
        npy_dir = output_dirs['npy']
        history_data = {k: np.array([v if v is not None else np.nan for v in vals])
                        for k, vals in history.items()}
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

    fig = plot_results(model, x_test, y_test, history, title, device=device)
    if save_plots:
        path = os.path.join(output_dirs['plots'], 'results.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {path}")
    plt.show()

    return model, history


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='2D Infinity-Laplacian with PINNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--example', type=str, default='aronsson_square',
                        choices=['arctan', 'absolute', 'aronsson_square', 'aronsson_disc'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--interior-grid', type=int, default=1000)
    parser.add_argument('--boundary-grid', type=int, default=10000)
    parser.add_argument('--batch-bc', type=int, default=400)
    parser.add_argument('--batch-pde', type=int, default=1000)
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'leaky_relu', 'elu', 'gelu', 'softplus', 'silu', 'sigmoid'])
    parser.add_argument('--hidden-layers', type=str, default='128,128,128,128')
    parser.add_argument('--eta', type=float, default=1e-5)
    parser.add_argument('--clip-residual', type=float, default=10.0)
    parser.add_argument('--outlier-percentile', type=float, default=2.0)
    parser.add_argument('--outlier-off-epoch', type=int, default=None)
    parser.add_argument('--plot-pde-every', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='Ablation tag; creates a subdirectory under the example folder')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--save-npy', action='store_true')

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
    parser.add_argument('--no-resample', action='store_true')

    parser.add_argument('--domain-decomposition', action='store_true')
    parser.add_argument('--n-interface-pts', type=int, default=200)
    parser.add_argument('--interface-weight', type=float, default=1.0)
    parser.add_argument('--interface-every', type=int, default=5)

    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine', 'step', 'exponential', 'none'])
    parser.add_argument('--scheduler-end-factor', type=float, default=0.01)

    args = parser.parse_args()

    if args.scheduler == 'none':
        args.scheduler = None

    # Build alpha scheduler
    alpha_scheduler = None
    alpha_mode = None
    if args.alpha_schedule is not None:
        s = args.alpha_schedule.strip().lower()
        if s == 'auto':
            alpha_mode = 'auto'
            alpha_scheduler = AlphaPlateauScheduler(
                alpha_init=args.alpha_min, alpha_max=args.alpha_max,
                patience=args.alpha_patience, cooldown=args.alpha_cooldown,
                min_relative_improvement=args.alpha_min_improvement)
        elif s == 'relobralo':
            alpha_mode = 'relobralo'
            alpha_scheduler = ReLoBRaLo(
                num_losses=2, alpha=args.relobralo_alpha,
                temperature=args.relobralo_temperature,
                rho=args.relobralo_rho, device=device)
        else:
            alpha_scheduler = FixedAlphaSchedule(parse_alpha_schedule(s))
    elif args.alpha is not None:
        alpha_scheduler = FixedAlphaSchedule({0: args.alpha})
    else:
        alpha_mode = 'auto'
        alpha_scheduler = AlphaPlateauScheduler(
            alpha_init=args.alpha_min, alpha_max=args.alpha_max,
            patience=args.alpha_patience, cooldown=args.alpha_cooldown,
            min_relative_improvement=args.alpha_min_improvement)

    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]
    resample = not args.no_resample

    run_example(
        args.example, n_epochs=args.epochs,
        n_interior_grid=args.interior_grid, n_boundary_grid=args.boundary_grid,
        batch_size_bc=args.batch_bc, batch_size_pde=args.batch_pde,
        save_plots=not args.no_save, val_split=args.val_split,
        patience=args.patience, alpha_scheduler=alpha_scheduler,
        resample=resample, activation=args.activation,
        hidden_layers=hidden_layers, eta=args.eta,
        scheduler_type=args.scheduler, scheduler_end_factor=args.scheduler_end_factor,
        lr=args.lr, clip_residual=args.clip_residual,
        outlier_percentile=args.outlier_percentile,
        plot_pde_every=args.plot_pde_every, output_dir=args.output_dir,
        outlier_off_epoch=args.outlier_off_epoch,
        domain_decomposition=args.domain_decomposition,
        n_interface_pts=args.n_interface_pts,
        interface_weight=args.interface_weight,
        interface_every=args.interface_every,
        save_npy=args.save_npy,
        run_tag=args.run_tag)
