#!/usr/bin/env python3
"""Evaluate a disk distance-to-boundary DeepONet against exact finite-p data."""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DeepONet_code import DeepONet, get_device


def exact_disk_solution(x, y, p):
    """Solve -Delta_p u=1 in the unit disk with u=0 on its boundary."""
    p = float(p)
    radius = torch.sqrt(x ** 2 + y ** 2).clamp(max=1.0)
    exponent = p / (p - 1.0)
    coefficient = (p - 1.0) / p * 2.0 ** (-1.0 / (p - 1.0))
    return coefficient * (1.0 - radius ** exponent)


def make_disk_grid(size):
    axis = torch.linspace(-1.0, 1.0, size)
    x_grid, y_grid = torch.meshgrid(axis, axis, indexing='ij')
    mask = x_grid ** 2 + y_grid ** 2 <= 1.0
    return torch.stack([x_grid[mask], y_grid[mask]], dim=1)


def parse_p_values(args):
    if args.p_values:
        return [float(value.strip()) for value in args.p_values.split(',')]
    count = int(round((args.p_end - args.p_start) / args.p_step))
    return [args.p_start + index * args.p_step for index in range(count + 1)]


def run(args):
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    trunk_layers = [int(value) for value in args.trunk_layers.split(',')]
    branch_layers = [int(value) for value in args.branch_layers.split(',')]
    model = DeepONet(
        trunk_layers=trunk_layers,
        branch_layers=branch_layers,
    ).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False,
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    points = make_disk_grid(args.grid_size)
    exact_limit = 1.0 - torch.sqrt(
        points[:, 0] ** 2 + points[:, 1] ** 2,
    ).clamp(max=1.0)
    normalized_points = points / 2.0 + 0.5

    theta = torch.linspace(
        0.0, 2.0 * torch.pi, args.boundary_points + 1,
    )[:-1]
    boundary = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    normalized_boundary = boundary / 2.0 + 0.5

    p_values = parse_p_values(args)
    metrics = []
    for p in p_values:
        p_column = torch.full((len(points), 1), p / args.p_normalize)
        inputs = torch.cat([normalized_points, p_column], dim=1)
        boundary_p = torch.full(
            (len(boundary), 1), p / args.p_normalize,
        )
        boundary_inputs = torch.cat(
            [normalized_boundary, boundary_p], dim=1,
        )

        with torch.no_grad():
            prediction = model(inputs.to(device)).cpu().squeeze(1)
            boundary_prediction = model(
                boundary_inputs.to(device),
            ).cpu().squeeze(1)

        exact = exact_disk_solution(points[:, 0], points[:, 1], p)
        error = prediction - exact
        mse_exact = torch.mean(error ** 2).item()
        relative_l2 = torch.sqrt(
            torch.sum(error ** 2) / torch.sum(exact ** 2)
        ).item()
        is_training_p = (
            p <= args.training_p_max
            and abs(p / args.training_p_step
                    - round(p / args.training_p_step)) < 1e-9
        )
        if is_training_p:
            evaluation = 'Training'
        elif p <= args.training_p_max:
            evaluation = 'Interpolation'
        elif p < args.auxiliary_p:
            evaluation = 'Augmented interpolation'
        elif abs(p - args.auxiliary_p) < 1e-9:
            evaluation = 'Auxiliary input'
        else:
            evaluation = 'Extrapolation'

        metrics.append({
            'p': p,
            'evaluation': evaluation,
            'is_training_p': int(is_training_p),
            'mse_exact': mse_exact,
            'relative_l2_exact': relative_l2,
            'max_abs_exact': torch.max(torch.abs(error)).item(),
            'boundary_mse': torch.mean(boundary_prediction ** 2).item(),
            'mse_limit': torch.mean(
                (prediction - exact_limit) ** 2,
            ).item(),
            'exact_to_limit_mse': torch.mean(
                (exact - exact_limit) ** 2,
            ).item(),
        })

    fieldnames = list(metrics[0].keys())
    csv_path = os.path.join(args.output_dir, 'deeponet_exact_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    np.savez(
        os.path.join(args.output_dir, 'deeponet_exact_metrics.npz'),
        **{
            name: np.asarray([row[name] for row in metrics])
            for name in fieldnames
        },
    )

    p_array = np.asarray([row['p'] for row in metrics])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(
        p_array, [row['mse_exact'] for row in metrics],
        label=r'$\mathrm{MSE}(u_\theta,u_p)$',
    )
    ax.semilogy(
        p_array, [row['mse_limit'] for row in metrics],
        label=r'$\mathrm{MSE}(u_\theta,u_\infty)$',
    )
    ax.semilogy(
        p_array, [row['exact_to_limit_mse'] for row in metrics], '--',
        label=r'$\mathrm{MSE}(u_p,u_\infty)$',
    )
    ax.set_xlabel('$p$')
    ax.set_ylabel('Mean squared error')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'deeponet_exact_mse.png'), dpi=200)
    plt.close(fig)

    selected = {
        5, 7, 10, 12, 25, 37, 50, 77, 100, 137, 150, 177, 200,
        250, 300, 400, 500,
    }
    print(f'Device: {device}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Test points: {len(points)}; boundary points: {len(boundary)}')
    print('p, evaluation, mse_exact, relative_l2, max_abs, boundary_mse')
    for row in metrics:
        if any(abs(row['p'] - value) < 1e-9 for value in selected):
            print(
                f"{row['p']:g}, {row['evaluation']}, "
                f"{row['mse_exact']:.9e}, "
                f"{row['relative_l2_exact']:.9e}, "
                f"{row['max_abs_exact']:.9e}, "
                f"{row['boundary_mse']:.9e}"
            )
    print(f'Outputs: {args.output_dir}')


def make_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate unit-disc DeepONet against exact finite-p data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--p-values', default=None)
    parser.add_argument('--p-start', type=float, default=5.0)
    parser.add_argument('--p-end', type=float, default=500.0)
    parser.add_argument('--p-step', type=float, default=1.0)
    parser.add_argument('--p-normalize', type=float, default=500.0)
    parser.add_argument('--training-p-step', type=float, default=5.0)
    parser.add_argument('--training-p-max', type=float, default=200.0)
    parser.add_argument('--auxiliary-p', type=float, default=500.0)
    parser.add_argument('--grid-size', type=int, default=101)
    parser.add_argument('--boundary-points', type=int, default=10000)
    parser.add_argument('--trunk-layers', default='2,512,512,512,128')
    parser.add_argument('--branch-layers', default='1,128,128,128,128')
    return parser


if __name__ == '__main__':
    run(make_parser().parse_args())
