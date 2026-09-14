#!/usr/bin/env python3
"""Iterative PINN validation for the finite-p distance problem on a disk.

The continuum problem is

    -div(|grad u_p|^(p-2) grad u_p) = 1  in B_R(0),
    u_p = 0                                  on dB_R(0).

In two dimensions its exact unregularized solution is

    u_p(r) = (p-1)/p * 2^(-1/(p-1))
             * (R^(p/(p-1)) - r^(p/(p-1))).

Training uses the flux-based eta regularization from Section 3 of the paper.
The finite-p test target is recomputed at every continuation stage.
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PINNs_code import (
    AlphaPlateauScheduler,
    BasePINN,
    FixedAlphaSchedule,
    InteriorSampler,
    ReLoBRaLo,
    create_output_dirs,
    generate_disc_domain_data,
    get_device,
    parse_alpha_schedule,
    plot_results,
    set_seed,
    setup_logging,
    split_data,
    train_pinn_iterative,
)


torch.set_default_dtype(torch.float32)


def exact_disk_solution(x, y, p, radius=1.0, source=1.0):
    """Exact solution of the unregularized 2-D p-Poisson disk problem."""
    p = float(p)
    if p <= 1:
        raise ValueError(f'p must be greater than 1; got {p}')
    r = torch.sqrt(x ** 2 + y ** 2).clamp(max=radius)
    exponent = p / (p - 1.0)
    if source <= 0:
        raise ValueError(f'source must be positive; got {source}')
    coefficient = (
        (p - 1.0) / p * (source / 2.0) ** (1.0 / (p - 1.0))
    )
    return coefficient * (radius ** exponent - r ** exponent)


def exact_limit_solution(x, y, radius=1.0):
    """Distance to the boundary of the disk."""
    r = torch.sqrt(x ** 2 + y ** 2).clamp(max=radius)
    return radius - r


class DiskPpoissonPINN(BasePINN):
    """PINN for the flux-regularized p-Poisson equation on a disk."""

    def __init__(self, hidden_layers=None, activation='tanh', p=2,
                 eta=1e-5, source=1.0, gradient_cap=1.0,
                 clip_residual=100.0, outlier_percentile=2.0):
        super().__init__(
            hidden_layers=hidden_layers,
            activation=activation,
            clip_residual=clip_residual,
            outlier_percentile=outlier_percentile,
        )
        self.p = p
        self.eta = eta
        self.source = source
        self.gradient_cap = gradient_cap

    def _flux_divergence(self, x, p, eta=None, apply_gradient_cap=True):
        """Compute div((eta^2+|grad u|^2)^((p-2)/2) grad u)."""
        eta = self.eta if eta is None else eta
        g = x.clone()
        g.requires_grad_(True)
        u = self.forward(g)

        grad_u = autograd.grad(
            u, g, torch.ones_like(u), retain_graph=True, create_graph=True,
        )[0]
        u_x, u_y = grad_u[:, [0]], grad_u[:, [1]]
        grad_ux = autograd.grad(
            u_x, g, torch.ones_like(u_x), retain_graph=True,
            create_graph=True,
        )[0]
        grad_uy = autograd.grad(
            u_y, g, torch.ones_like(u_y), create_graph=True,
        )[0]
        u_xx, u_xy = grad_ux[:, [0]], grad_ux[:, [1]]
        u_yx, u_yy = grad_uy[:, [0]], grad_uy[:, [1]]

        grad_norm_sq = u_x ** 2 + u_y ** 2
        if apply_gradient_cap and self.gradient_cap > 0:
            grad_norm_sq = torch.clamp(
                grad_norm_sq, max=self.gradient_cap ** 2,
            )

        base = eta ** 2 + grad_norm_sq
        base = torch.clamp(base, min=torch.finfo(base.dtype).tiny)
        laplacian = u_xx + u_yy
        delta_infinity = (
            u_x ** 2 * u_xx
            + u_x * u_y * (u_xy + u_yx)
            + u_y ** 2 * u_yy
        )
        return base ** ((float(p) - 4.0) / 2.0) * (
            base * laplacian + (float(p) - 2.0) * delta_infinity
        )

    def residual_at_p(self, x, p, eta=None, apply_gradient_cap=True):
        divergence = self._flux_divergence(
            x, p, eta=eta, apply_gradient_cap=apply_gradient_cap,
        )
        return -divergence - self.source

    def compute_pde_residual(self, x):
        return self.residual_at_p(x, self.p)

    def regularized_residual_mse(self, x, p):
        residual = self.residual_at_p(x, p)
        return torch.mean(residual ** 2)


def _p_tag(p):
    value = float(p)
    return str(int(value)) if value.is_integer() else f'{value:g}'


def evaluate_stage_checkpoints(model, p_values, x_test, x_boundary,
                               checkpoint_dir, npy_dir, device,
                               radius, source, save_predictions):
    """Evaluate every saved continuation checkpoint against exact u_p."""
    metrics = {
        'p': [],
        'mse_exact': [],
        'relative_l2_exact': [],
        'max_abs_exact': [],
        'mse_limit': [],
        'boundary_mse': [],
        'regularized_pde_mse': [],
    }
    x_test_dev = x_test.to(device)
    x_boundary_dev = x_boundary.to(device)

    for p in p_values:
        tag = _p_tag(p)
        checkpoint = os.path.join(
            checkpoint_dir, f'final_model_p{tag}.pt',
        )
        if not os.path.exists(checkpoint):
            print(f'  Missing stage checkpoint for p={p}: {checkpoint}')
            continue

        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        model.p = p
        model.eval()

        exact = exact_disk_solution(
            x_test[:, 0], x_test[:, 1], p, radius, source,
        ).view(-1, 1)
        limit = exact_limit_solution(
            x_test[:, 0], x_test[:, 1], radius,
        ).view(-1, 1)
        with torch.no_grad():
            pred = model(x_test_dev).cpu()
            boundary_pred = model(x_boundary_dev).cpu()

        error = pred - exact
        mse_exact = torch.mean(error ** 2).item()
        relative_l2 = torch.sqrt(
            torch.sum(error ** 2) / torch.sum(exact ** 2)
        ).item()
        max_abs = torch.max(torch.abs(error)).item()
        mse_limit = torch.mean((pred - limit) ** 2).item()
        boundary_mse = torch.mean(boundary_pred ** 2).item()

        with torch.enable_grad():
            pde_mse = model.regularized_residual_mse(
                x_test_dev, p,
            ).detach().item()

        metrics['p'].append(float(p))
        metrics['mse_exact'].append(mse_exact)
        metrics['relative_l2_exact'].append(relative_l2)
        metrics['max_abs_exact'].append(max_abs)
        metrics['mse_limit'].append(mse_limit)
        metrics['boundary_mse'].append(boundary_mse)
        metrics['regularized_pde_mse'].append(pde_mse)

        print(
            f'  p={p:>4}: MSE(exact)={mse_exact:.6e}, '
            f'relL2={relative_l2:.6e}, PDE={pde_mse:.6e}'
        )

        if save_predictions:
            np.savez(
                os.path.join(npy_dir, f'predictions_p{tag}.npz'),
                x=x_test[:, 0].numpy(),
                y=x_test[:, 1].numpy(),
                u_pred=pred.numpy().flatten(),
                u_exact=exact.numpy().flatten(),
                u_infinity=limit.numpy().flatten(),
                error=error.numpy().flatten(),
            )

    arrays = {key: np.asarray(value) for key, value in metrics.items()}
    np.savez(os.path.join(npy_dir, 'stage_metrics.npz'), **arrays)

    columns = np.column_stack([arrays[key] for key in metrics])
    np.savetxt(
        os.path.join(npy_dir, 'stage_metrics.csv'), columns, delimiter=',',
        header=','.join(metrics.keys()), comments='',
    )
    return metrics


def plot_stage_metrics(metrics, output_path):
    if not metrics['p']:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(metrics['p'], metrics['mse_exact'], 'o-',
                label=r'$\mathrm{MSE}(u_{\theta,p},u_p)$')
    ax.semilogy(metrics['p'], metrics['mse_limit'], 's--',
                label=r'$\mathrm{MSE}(u_{\theta,p},u_\infty)$')
    ax.set_xlabel('$p$')
    ax.set_ylabel('Mean squared error')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_alpha_scheduler(args, device):
    if args.alpha_schedule:
        value = args.alpha_schedule.strip().lower()
        if value == 'auto':
            return AlphaPlateauScheduler(
                alpha_init=args.alpha_min,
                alpha_max=args.alpha_max,
                patience=args.alpha_patience,
                cooldown=args.alpha_cooldown,
                min_relative_improvement=args.alpha_min_improvement,
            )
        if value == 'relobralo':
            return ReLoBRaLo(
                num_losses=2, alpha=args.relobralo_alpha,
                temperature=args.relobralo_temperature,
                rho=args.relobralo_rho, device=device,
            )
        return FixedAlphaSchedule(parse_alpha_schedule(value))
    if args.alpha is not None:
        return FixedAlphaSchedule({0: args.alpha})
    return AlphaPlateauScheduler(
        alpha_init=args.alpha_min,
        alpha_max=args.alpha_max,
        patience=args.alpha_patience,
        cooldown=args.alpha_cooldown,
        min_relative_improvement=args.alpha_min_improvement,
    )


def run(args):
    set_seed(args.seed)
    device = get_device()
    p_values = [int(value.strip()) for value in args.p_values.split(',')]
    if any(p <= 1 for p in p_values):
        raise ValueError('Every continuation value must satisfy p > 1')

    example_dir = f'p{p_values[0]}to{p_values[-1]}_disc'
    base = os.path.join(
        args.output_dir, 'expr_6_3_iter_distance_boundary_disc', example_dir,
    )
    if args.run_tag:
        base = os.path.join(base, args.run_tag)
    setup_logging(base)
    output_dirs = create_output_dirs(base)

    print('\n' + '=' * 68)
    print('Finite-p distance-to-boundary problem on the unit disk')
    print('=' * 68)
    print(f'Device: {device}')
    print(f'p values: {p_values}')
    print(f'eta={args.eta}, source={args.source}, radius={args.radius}')

    final_p = p_values[-1]
    initial_exact = lambda x, y: exact_disk_solution(
        x, y, final_p, args.radius, args.source,
    )
    x_bc, y_bc, x_interior, x_test, _ = generate_disc_domain_data(
        radius=args.radius,
        n_boundary=args.boundary_grid,
        n_interior_grid=args.interior_grid,
        f_exact=initial_exact,
        seed=args.seed,
    )
    y_bc = torch.zeros_like(y_bc)
    x_bc_all = x_bc.clone()
    y_test_final = exact_disk_solution(
        x_test[:, 0], x_test[:, 1], final_p,
        args.radius, args.source,
    ).view(-1, 1)

    interior_sampler = None
    if not args.no_resample:
        interior_sampler = InteriorSampler(
            'disc', {'radius': args.radius}, len(x_interior),
        )

    x_bc_val = y_bc_val = x_interior_val = None
    if args.val_split > 0:
        x_bc, x_bc_val, y_bc, y_bc_val = split_data(
            x_bc, y_bc, train_ratio=1.0 - args.val_split, seed=args.seed,
        )
        x_interior, x_interior_val, _, _ = split_data(
            x_interior, None, train_ratio=1.0 - args.val_split,
            seed=args.seed + 1,
        )

    hidden_layers = [int(value.strip())
                     for value in args.hidden_layers.split(',')]
    model = DiskPpoissonPINN(
        hidden_layers=hidden_layers,
        activation=args.activation,
        p=p_values[0],
        eta=args.eta,
        source=args.source,
        gradient_cap=args.gradient_cap,
        clip_residual=args.clip_residual,
        outlier_percentile=args.outlier_percentile,
    ).to(device)

    resume_from_p = None
    if args.resume_checkpoint:
        state = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False,
        )
        model.load_state_dict(state['model_state_dict'])
        resume_from_p = p_values[0]
        print(f'Loaded continuation checkpoint: {args.resume_checkpoint}')

    alpha_scheduler = build_alpha_scheduler(args, device)
    scheduler_type = None if args.scheduler == 'none' else args.scheduler
    target_fn = lambda p, pts: exact_disk_solution(
        pts[:, 0], pts[:, 1], p, args.radius, args.source,
    ).view(-1, 1)

    history, training_time = train_pinn_iterative(
        model, x_bc, y_bc, x_interior, x_test, y_test_final,
        p_values=p_values,
        epochs_per_p=args.epochs_per_p,
        batch_size_bc=args.batch_bc,
        batch_size_pde=args.batch_pde,
        lr=args.lr,
        alpha_scheduler=alpha_scheduler,
        x_bc_val=x_bc_val,
        y_bc_val=y_bc_val,
        x_interior_val=x_interior_val,
        patience=args.patience,
        interior_sampler=interior_sampler,
        scheduler_type=scheduler_type,
        scheduler_end_factor=args.scheduler_end_factor,
        output_dirs=output_dirs,
        outlier_off_epoch=args.outlier_off_epoch,
        original_outlier_pct=args.outlier_percentile,
        bc_loss_threshold=args.bc_loss_threshold,
        pde_loss_threshold=args.pde_loss_threshold,
        resume_from_p=resume_from_p,
        test_target_fn=target_fn,
        seed=args.seed,
        device=device,
    )

    history_arrays = {}
    for key, values in history.items():
        if key == 'p_schedule':
            history_arrays[key] = np.asarray(values)
        else:
            history_arrays[key] = np.asarray([
                np.nan if value is None else value for value in values
            ])
    np.savez(
        os.path.join(output_dirs['npy'], 'training_history.npz'),
        **history_arrays,
    )

    print('\nExact finite-p checkpoint evaluation:')
    metrics = evaluate_stage_checkpoints(
        model, p_values, x_test, x_bc_all,
        output_dirs['checkpoints'], output_dirs['npy'], device,
        args.radius, args.source, args.save_predictions,
    )
    plot_stage_metrics(
        metrics, os.path.join(output_dirs['plots'], 'mse_vs_p.png'),
    )

    final_checkpoint = os.path.join(
        output_dirs['checkpoints'], f'final_model_p{_p_tag(final_p)}.pt',
    )
    final_state = torch.load(
        final_checkpoint, map_location=device, weights_only=False,
    )
    model.load_state_dict(final_state['model_state_dict'])
    model.p = final_p
    fig = plot_results(
        model, x_test, y_test_final, history,
        title=f'Disk distance problem, iterative PINN, p={final_p}',
        output_path=os.path.join(output_dirs['plots'], 'final_results.png'),
        device=device,
    )
    plt.close(fig)

    print(f'Completed in {training_time:.2f}s')
    print(f'Outputs: {base}')
    return model, history, metrics


def make_parser():
    parser = argparse.ArgumentParser(
        description='Iterative finite-p PINN on the unit disk',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--p-values', default='2,3,4,5,6,7,8,9,10,15,20',
        help='Comma-separated continuation values',
    )
    parser.add_argument('--epochs-per-p', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta', type=float, default=1e-5)
    parser.add_argument('--source', type=float, default=1.0)
    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--gradient-cap', type=float, default=1.0)
    parser.add_argument('--hidden-layers', default='128,128,128,128')
    parser.add_argument('--activation', default='tanh')
    parser.add_argument('--interior-grid', type=int, default=1000)
    parser.add_argument('--boundary-grid', type=int, default=10000)
    parser.add_argument('--batch-bc', type=int, default=400)
    parser.add_argument('--batch-pde', type=int, default=1000)
    parser.add_argument('--clip-residual', type=float, default=100.0)
    parser.add_argument('--outlier-percentile', type=float, default=2.0)
    parser.add_argument('--outlier-off-epoch', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--alpha-schedule', default=None)
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
    parser.add_argument(
        '--scheduler', default='cosine',
        choices=['linear', 'cosine', 'step', 'exponential', 'none'],
    )
    parser.add_argument('--scheduler-end-factor', type=float, default=0.01)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--run-tag', default=None)
    parser.add_argument('--resume-checkpoint', default=None)
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    return parser


if __name__ == '__main__':
    run(make_parser().parse_args())
