"""
Domain data generation for square and disc geometries.

Each generator produces boundary points with exact values, shuffled interior
collocation points, and a test grid -- ready for PINN training.
"""

from typing import Callable, Tuple

import numpy as np
import torch


def generate_square_domain_data(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    f_exact: Callable,
    n_interior_grid: int = 1000,
    n_boundary_grid: int = 10000,
    seed: int = 1234,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training data on a rectangular domain.

    Args:
        x_min, x_max, y_min, y_max: domain bounds
        f_exact: exact solution ``f(x, y) -> Tensor``
        n_interior_grid: grid size per axis for interior (N -> N*N points)
        n_boundary_grid: points per boundary edge
        seed: random seed for shuffling

    Returns:
        ``(x_bc, y_bc, x_interior, x_test, y_test)``
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    n_per_edge = n_boundary_grid

    left_x = torch.ones(n_per_edge, 1) * x_min
    left_y = torch.linspace(y_min, y_max, n_per_edge).view(-1, 1)

    right_x = torch.ones(n_per_edge, 1) * x_max
    right_y = torch.linspace(y_min, y_max, n_per_edge).view(-1, 1)

    bottom_x = torch.linspace(x_min, x_max, n_per_edge + 2)[1:-1].view(-1, 1)
    bottom_y = torch.ones(n_per_edge, 1) * y_min

    top_x = torch.linspace(x_min, x_max, n_per_edge + 2)[1:-1].view(-1, 1)
    top_y = torch.ones(n_per_edge, 1) * y_max

    x_bc = torch.cat([
        torch.cat([left_x, left_y], dim=1),
        torch.cat([right_x, right_y], dim=1),
        torch.cat([bottom_x, bottom_y], dim=1),
        torch.cat([top_x, top_y], dim=1),
    ], dim=0)

    y_bc = f_exact(x_bc[:, 0], x_bc[:, 1]).view(-1, 1)

    perm = torch.randperm(x_bc.shape[0], generator=generator)
    x_bc = x_bc[perm]
    y_bc = y_bc[perm]

    eps = 1e-6
    x_1d = torch.linspace(x_min + eps, x_max - eps, n_interior_grid)
    y_1d = torch.linspace(y_min + eps, y_max - eps, n_interior_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    x_interior = torch.stack([X.flatten(), Y.flatten()], dim=1)

    perm = torch.randperm(x_interior.shape[0], generator=generator)
    x_interior = x_interior[perm]

    n_test = 100
    xt = torch.linspace(x_min, x_max, n_test)
    yt = torch.linspace(y_min, y_max, n_test)
    X, Y = torch.meshgrid(xt, yt, indexing='ij')
    x_test = torch.stack([X.flatten(), Y.flatten()], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1]).view(-1, 1)

    print(f"Generated full domain data:")
    print(f"  Boundary points: {x_bc.shape[0]:,}")
    print(f"  Interior points: {x_interior.shape[0]:,}")
    print(f"  Test points: {x_test.shape[0]:,}")

    return x_bc, y_bc, x_interior, x_test, y_test


def generate_disc_domain_data(
    radius: float,
    f_exact: Callable,
    n_boundary: int = 10000,
    n_interior_grid: int = 1000,
    seed: int = 1234,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training data on a disc domain.

    Args:
        radius: disc radius
        f_exact: exact solution ``f(x, y) -> Tensor``
        n_boundary: number of boundary points on circle
        n_interior_grid: grid size for interior (grid filtered to disc)
        seed: random seed for shuffling

    Returns:
        ``(x_bc, y_bc, x_interior, x_test, y_test)``
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    theta = torch.linspace(0, 2 * np.pi, n_boundary + 1)[:-1]
    x_bc = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    y_bc = f_exact(x_bc[:, 0], x_bc[:, 1]).view(-1, 1)

    perm = torch.randperm(x_bc.shape[0], generator=generator)
    x_bc = x_bc[perm]
    y_bc = y_bc[perm]

    eps = 1e-6
    x_1d = torch.linspace(-radius + eps, radius - eps, n_interior_grid)
    y_1d = torch.linspace(-radius + eps, radius - eps, n_interior_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')

    mask = (X ** 2 + Y ** 2) < (radius - eps) ** 2
    x_interior = torch.stack([X[mask].flatten(), Y[mask].flatten()], dim=1)

    perm = torch.randperm(x_interior.shape[0], generator=generator)
    x_interior = x_interior[perm]

    n_test = 100
    xt = torch.linspace(-radius, radius, n_test)
    yt = torch.linspace(-radius, radius, n_test)
    X, Y = torch.meshgrid(xt, yt, indexing='ij')
    mask = X ** 2 + Y ** 2 <= radius ** 2
    x_test = torch.stack([X[mask].flatten(), Y[mask].flatten()], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1]).view(-1, 1)

    print(f"Generated full disc domain data:")
    print(f"  Boundary points: {x_bc.shape[0]:,}")
    print(f"  Interior points: {x_interior.shape[0]:,}")
    print(f"  Test points: {x_test.shape[0]:,}")

    return x_bc, y_bc, x_interior, x_test, y_test
