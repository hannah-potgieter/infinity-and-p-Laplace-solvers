"""
Experiment 6.3: Eikonal Equation |∇u|² = 1

    |∇u|² = 1  in Ω
    u = g      on Γ₁

2D domains:
    - disc, square: distance to origin (case='origin') or boundary (case='boundary')
    - ellipse1, ellipse2, ellipse3: distance to boundary only

3D domains (distance to boundary only):
    - sphere:   x² + y² + z² ≤ 1
    - cylinder: y² + z² ≤ 1, −1 ≤ x ≤ 1
    - torus:    (√(x²+z²) − 2)² + y² ≤ 1  (major R=2, minor r=1)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time

import numpy as np
import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt

from PINNs_code import (
    BasePINN, create_output_dirs, setup_logging, set_seed, get_device,
    InteriorSampler, split_data, train_pinn,
    plot_results, plot_pde_loss_distribution,
    FixedAlphaSchedule, AlphaPlateauScheduler, ReLoBRaLo,
    parse_alpha_schedule,
)

torch.set_default_dtype(torch.float32)
SEED = 1234
set_seed(SEED)
device = get_device()
print(f"Using device: {device}")
print(f"Random seed: {SEED}")

DOMAINS_2D = {'disc', 'square', 'ellipse1', 'ellipse2', 'ellipse3'}
DOMAINS_3D = {'sphere', 'cylinder', 'torus'}


# ============================================================================
# Exact solutions — 2D
# ============================================================================

def dist_to_origin(x, y):
    return torch.sqrt(x ** 2 + y ** 2)


def dist_to_disc_boundary(x, y):
    return 1.0 - torch.sqrt(x ** 2 + y ** 2)


def dist_to_square_boundary(x, y):
    return torch.minimum(1.0 - torch.abs(x), 1.0 - torch.abs(y))


# ============================================================================
# Ellipse support — 2D
# ============================================================================

ELLIPSE_CONFIGS = {
    'ellipse1': {'name': 'Ellipse 1', 'a': 1.0, 'b': 0.25, 'theta': 0.0},
    'ellipse2': {'name': 'Ellipse 2 (rotated)', 'a': 1.0, 'b': 0.25, 'theta': -np.pi / 4},
    'ellipse3': {'name': 'Ellipse 3', 'a': 0.5, 'b': 1.0, 'theta': 0.0},
}


def dist_to_ellipse_boundary(x_list, y_list, a, b, n_coarse=3600, chunk=2000):
    """Distance from points to ellipse x²/a² + y²/b² = 1 via grid search."""
    n = x_list.shape[0]
    theta_c = torch.linspace(0, 2 * np.pi, n_coarse + 1)[:-1]
    ex_c = a * torch.cos(theta_c)
    ey_c = b * torch.sin(theta_c)

    best_dist = torch.full((n,), float('inf'))
    best_theta = torch.zeros(n)
    d_theta = 2 * np.pi / n_coarse

    for i in range(0, n, chunk):
        xi, yi = x_list[i:i+chunk], y_list[i:i+chunk]
        dx = ex_c.unsqueeze(1) - xi.unsqueeze(0)
        dy = ey_c.unsqueeze(1) - yi.unsqueeze(0)
        dists = torch.sqrt(dx ** 2 + dy ** 2)
        min_d, min_idx = dists.min(dim=0)
        best_dist[i:i+chunk] = min_d
        best_theta[i:i+chunk] = theta_c[min_idx]

    n_fine = 200
    offsets = torch.linspace(-d_theta, d_theta, n_fine)
    for i in range(0, n, chunk):
        xi, yi = x_list[i:i+chunk], y_list[i:i+chunk]
        bt = best_theta[i:i+chunk]
        theta_f = bt.unsqueeze(0) + offsets.unsqueeze(1)
        ex_f = a * torch.cos(theta_f)
        ey_f = b * torch.sin(theta_f)
        dx = ex_f - xi.unsqueeze(0)
        dy = ey_f - yi.unsqueeze(0)
        dists = torch.sqrt(dx ** 2 + dy ** 2)
        min_d, _ = dists.min(dim=0)
        best_dist[i:i+chunk] = min_d

    return best_dist


def dist_to_ellipse(x, y, config):
    a, b, theta = config['a'], config['b'], config['theta']
    if theta != 0:
        x_rot = np.cos(theta) * x - np.sin(theta) * y
        y_rot = np.sin(theta) * x + np.cos(theta) * y
        return dist_to_ellipse_boundary(x_rot, y_rot, a, b)
    return dist_to_ellipse_boundary(x, y, a, b)


def is_inside_ellipse(x, y, config):
    a, b, theta = config['a'], config['b'], config['theta']
    if theta != 0:
        x_r = np.cos(theta) * x - np.sin(theta) * y
        y_r = np.sin(theta) * x + np.cos(theta) * y
    else:
        x_r, y_r = x, y
    return (x_r / a) ** 2 + (y_r / b) ** 2 <= 1


# ============================================================================
# Exact solutions — 3D
# ============================================================================

def dist_to_sphere_boundary(x, y, z):
    return 1.0 - torch.sqrt(x ** 2 + y ** 2 + z ** 2)


def dist_to_cylinder_boundary(x, y, z):
    d_side = 1.0 - torch.sqrt(y ** 2 + z ** 2)
    d_caps = 1.0 - torch.abs(x)
    return torch.minimum(d_side, d_caps)


def dist_to_torus_boundary(x, y, z, R=2.0, r=1.0):
    rho = torch.sqrt(x ** 2 + z ** 2)
    return r - torch.sqrt((rho - R) ** 2 + y ** 2)


# ============================================================================
# Inside-domain tests — 3D
# ============================================================================

def is_inside_sphere(x, y, z):
    return x ** 2 + y ** 2 + z ** 2 < 1.0


def is_inside_cylinder(x, y, z):
    return (y ** 2 + z ** 2 < 1.0) & (torch.abs(x) < 1.0)


def is_inside_torus(x, y, z, R=2.0, r=1.0):
    rho = torch.sqrt(x ** 2 + z ** 2)
    return (rho - R) ** 2 + y ** 2 < r ** 2


# ============================================================================
# Unified exact-solution lookup
# ============================================================================

def get_exact_solution(case, domain):
    if case == 'origin':
        return dist_to_origin
    if domain == 'disc':
        return dist_to_disc_boundary
    elif domain == 'square':
        return dist_to_square_boundary
    elif domain in ELLIPSE_CONFIGS:
        config = ELLIPSE_CONFIGS[domain]
        return lambda x, y: dist_to_ellipse(x, y, config)
    elif domain == 'sphere':
        return dist_to_sphere_boundary
    elif domain == 'cylinder':
        return dist_to_cylinder_boundary
    elif domain == 'torus':
        return dist_to_torus_boundary
    raise ValueError(f"Unknown domain: {domain}")


# ============================================================================
# PDE model (unified — works for any input dimension)
# ============================================================================

class EikonalPINN(BasePINN):
    """PINN for the eikonal equation |∇u|² = 1."""

    def __init__(self, *args, positivity_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.positivity_weight = positivity_weight

    def compute_pde_residual(self, x):
        g = x.clone()
        g.requires_grad = True
        u = self.forward(g)
        grad_u = autograd.grad(u, g, torch.ones_like(u),
                               retain_graph=True, create_graph=True)[0]
        eikonal = (grad_u ** 2).sum(dim=1, keepdim=True) - 1.0
        if self.positivity_weight > 0:
            positivity = self.positivity_weight * torch.relu(-u)
            return torch.cat([eikonal, positivity], dim=0)
        return eikonal


# ============================================================================
# Data generation — 2D
# ============================================================================

def generate_origin_bc_data(domain, n_ring_pts=1000, ring_radius=0.01, seed=1234):
    """Sample n_ring_pts points uniformly inside a disc of the given radius,
    with u = sqrt(x² + y²) (exact distance from origin)."""
    generator = torch.Generator().manual_seed(seed)
    n_origin = max(100, n_ring_pts // 10)
    x_origin = torch.zeros(n_origin, 2)
    y_origin = torch.zeros(n_origin, 1)

    angles = torch.rand(n_ring_pts, generator=generator) * 2 * np.pi
    rr = ring_radius * torch.sqrt(torch.rand(n_ring_pts, generator=generator))
    x_pts = torch.stack([rr * torch.cos(angles), rr * torch.sin(angles)], dim=1)
    y_pts = torch.sqrt(x_pts[:, 0] ** 2 + x_pts[:, 1] ** 2).unsqueeze(1)

    x_bc = torch.cat([x_origin, x_pts])
    y_bc = torch.cat([y_origin, y_pts])
    perm = torch.randperm(x_bc.shape[0], generator=generator)
    return x_bc[perm], y_bc[perm]


def generate_boundary_bc_data_disc(radius=1.0, n_boundary=10000, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    theta = torch.linspace(0, 2 * np.pi, n_boundary + 1)[:-1]
    x_bc = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    y_bc = torch.zeros(n_boundary, 1)
    perm = torch.randperm(x_bc.shape[0], generator=generator)
    return x_bc[perm], y_bc[perm]


def generate_boundary_bc_data_square(n_per_edge=10000, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    n = n_per_edge
    left = torch.stack([torch.full((n,), -1.0), torch.linspace(-1, 1, n)], dim=1)
    right = torch.stack([torch.full((n,), 1.0), torch.linspace(-1, 1, n)], dim=1)
    bottom = torch.stack([torch.linspace(-1, 1, n + 2)[1:-1], torch.full((n,), -1.0)], dim=1)
    top = torch.stack([torch.linspace(-1, 1, n + 2)[1:-1], torch.full((n,), 1.0)], dim=1)
    x_bc = torch.cat([left, right, bottom, top])
    y_bc = torch.zeros(x_bc.shape[0], 1)
    perm = torch.randperm(x_bc.shape[0], generator=generator)
    return x_bc[perm], y_bc[perm]


def generate_boundary_bc_data_ellipse(config, n_boundary=40000, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    a, b, theta_rot = config['a'], config['b'], config['theta']
    t = torch.linspace(0, 2 * np.pi, n_boundary + 1)[:-1]
    x_a = a * torch.cos(t)
    y_a = b * torch.sin(t)
    if theta_rot != 0:
        c, s = np.cos(-theta_rot), np.sin(-theta_rot)
        x_o, y_o = c * x_a - s * y_a, s * x_a + c * y_a
    else:
        x_o, y_o = x_a, y_a
    x_bc = torch.stack([x_o, y_o], dim=1)
    y_bc = torch.zeros(n_boundary, 1)
    perm = torch.randperm(x_bc.shape[0], generator=generator)
    return x_bc[perm], y_bc[perm]


def generate_interior_and_test_square(x_min, x_max, y_min, y_max,
                                      n_interior_grid=1000, f_exact=None, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    eps = 1e-6
    x_1d = torch.linspace(x_min + eps, x_max - eps, n_interior_grid)
    y_1d = torch.linspace(y_min + eps, y_max - eps, n_interior_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    x_int = torch.stack([X.flatten(), Y.flatten()], dim=1)
    x_int = x_int[torch.randperm(x_int.shape[0], generator=generator)]
    xt = torch.linspace(x_min, x_max, 100)
    yt = torch.linspace(y_min, y_max, 100)
    Xt, Yt = torch.meshgrid(xt, yt, indexing='ij')
    x_test = torch.stack([Xt.flatten(), Yt.flatten()], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1]).view(-1, 1)
    print(f"  Interior: {x_int.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    return x_int, x_test, y_test


def generate_interior_and_test_disc(radius=1.0, n_interior_grid=1000, f_exact=None, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    eps = 1e-6
    x_1d = torch.linspace(-radius + eps, radius - eps, n_interior_grid)
    y_1d = torch.linspace(-radius + eps, radius - eps, n_interior_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    mask = (X ** 2 + Y ** 2) < (radius - eps) ** 2
    x_int = torch.stack([X[mask].flatten(), Y[mask].flatten()], dim=1)
    x_int = x_int[torch.randperm(x_int.shape[0], generator=generator)]
    xt = torch.linspace(-radius, radius, 100)
    yt = torch.linspace(-radius, radius, 100)
    Xt, Yt = torch.meshgrid(xt, yt, indexing='ij')
    tmask = Xt ** 2 + Yt ** 2 <= radius ** 2
    x_test = torch.stack([Xt[tmask].flatten(), Yt[tmask].flatten()], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1]).view(-1, 1)
    print(f"  Interior: {x_int.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    return x_int, x_test, y_test


def generate_interior_and_test_ellipse(config, n_interior_grid=1000, f_exact=None, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    a, b = config['a'], config['b']
    bound = max(a, b) + 0.1
    eps = 1e-6
    box_area = (2 * bound) ** 2
    ellipse_area = np.pi * a * b
    accept_rate = ellipse_area / box_area
    adjusted_grid = int(n_interior_grid / np.sqrt(accept_rate))
    x_1d = torch.linspace(-bound + eps, bound - eps, adjusted_grid)
    y_1d = torch.linspace(-bound + eps, bound - eps, adjusted_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    mask = is_inside_ellipse(X, Y, config)
    x_int = torch.stack([X[mask].flatten(), Y[mask].flatten()], dim=1)
    x_int = x_int[torch.randperm(x_int.shape[0], generator=generator)]
    xt = torch.linspace(-bound, bound, 100)
    yt = torch.linspace(-bound, bound, 100)
    Xt, Yt = torch.meshgrid(xt, yt, indexing='ij')
    tmask = is_inside_ellipse(Xt, Yt, config)
    x_test = torch.stack([Xt[tmask].flatten(), Yt[tmask].flatten()], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1]).view(-1, 1)
    print(f"  Interior: {x_int.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    return x_int, x_test, y_test


def _sample_ellipse(generator, n_points, domain_params):
    """Custom sampler for ellipse domains, passed to InteriorSampler."""
    config = domain_params['config']
    a, b = config['a'], config['b']
    bound = max(a, b) + 0.1
    box_area = (2 * bound) ** 2
    ellipse_area = np.pi * a * b
    oversample = max(box_area / ellipse_area * 1.5, 2.0)
    n_sample = int(n_points * oversample)
    x = torch.rand(n_sample, generator=generator) * 2 * bound - bound
    y = torch.rand(n_sample, generator=generator) * 2 * bound - bound
    pts = torch.stack([x, y], dim=1)
    mask = is_inside_ellipse(pts[:, 0], pts[:, 1], config)
    pts = pts[mask]
    return pts[:n_points] if pts.shape[0] > n_points else pts


# ============================================================================
# Data generation — 3D
# ============================================================================

def generate_boundary_bc_sphere(n_boundary=20000, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    z_vals = torch.rand(n_boundary, generator=generator) * 2 - 1
    phi = torch.rand(n_boundary, generator=generator) * 2 * np.pi
    r_xy = torch.sqrt(1 - z_vals ** 2)
    x_bc = torch.stack([r_xy * torch.cos(phi), r_xy * torch.sin(phi), z_vals], dim=1)
    y_bc = torch.zeros(n_boundary, 1)
    return x_bc, y_bc


def generate_boundary_bc_cylinder(n_per_part=5000, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    n = n_per_part

    theta = torch.rand(n, generator=generator) * 2 * np.pi
    x_side = torch.rand(n, generator=generator) * 2 - 1
    side = torch.stack([x_side, torch.cos(theta), torch.sin(theta)], dim=1)

    r_cap = torch.sqrt(torch.rand(n, generator=generator))
    phi_cap = torch.rand(n, generator=generator) * 2 * np.pi
    cap_m = torch.stack([torch.full((n,), -1.0),
                         r_cap * torch.cos(phi_cap), r_cap * torch.sin(phi_cap)], dim=1)

    r_cap2 = torch.sqrt(torch.rand(n, generator=generator))
    phi_cap2 = torch.rand(n, generator=generator) * 2 * np.pi
    cap_p = torch.stack([torch.full((n,), 1.0),
                         r_cap2 * torch.cos(phi_cap2), r_cap2 * torch.sin(phi_cap2)], dim=1)

    x_bc = torch.cat([side, cap_m, cap_p])
    y_bc = torch.zeros(x_bc.shape[0], 1)
    perm = torch.randperm(x_bc.shape[0], generator=generator)
    return x_bc[perm], y_bc[perm]


def generate_boundary_bc_torus(R=2.0, r=1.0, n_boundary=20000, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    theta = torch.rand(n_boundary, generator=generator) * 2 * np.pi
    phi = torch.rand(n_boundary, generator=generator) * 2 * np.pi
    x_bc = torch.stack([
        (R + r * torch.cos(phi)) * torch.cos(theta),
        r * torch.sin(phi),
        (R + r * torch.cos(phi)) * torch.sin(theta),
    ], dim=1)
    y_bc = torch.zeros(n_boundary, 1)
    return x_bc, y_bc


def generate_interior_and_test_sphere(n_grid=100, f_exact=None, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    eps = 1e-6
    x_1d = torch.linspace(-1 + eps, 1 - eps, n_grid)
    X, Y, Z = torch.meshgrid(x_1d, x_1d, x_1d, indexing='ij')
    mask = is_inside_sphere(X, Y, Z)
    x_int = torch.stack([X[mask], Y[mask], Z[mask]], dim=1)
    x_int = x_int[torch.randperm(x_int.shape[0], generator=generator)]

    n_test = 50
    xt = torch.linspace(-1, 1, n_test)
    Xt, Yt, Zt = torch.meshgrid(xt, xt, xt, indexing='ij')
    tmask = is_inside_sphere(Xt, Yt, Zt)
    x_test = torch.stack([Xt[tmask], Yt[tmask], Zt[tmask]], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1], x_test[:, 2]).view(-1, 1)
    print(f"  Interior: {x_int.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    return x_int, x_test, y_test


def generate_interior_and_test_cylinder(n_grid=100, f_exact=None, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    eps = 1e-6
    x_1d = torch.linspace(-1 + eps, 1 - eps, n_grid)
    X, Y, Z = torch.meshgrid(x_1d, x_1d, x_1d, indexing='ij')
    mask = is_inside_cylinder(X, Y, Z)
    x_int = torch.stack([X[mask], Y[mask], Z[mask]], dim=1)
    x_int = x_int[torch.randperm(x_int.shape[0], generator=generator)]

    n_test = 50
    xt = torch.linspace(-1, 1, n_test)
    Xt, Yt, Zt = torch.meshgrid(xt, xt, xt, indexing='ij')
    tmask = is_inside_cylinder(Xt, Yt, Zt)
    x_test = torch.stack([Xt[tmask], Yt[tmask], Zt[tmask]], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1], x_test[:, 2]).view(-1, 1)
    print(f"  Interior: {x_int.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    return x_int, x_test, y_test


def generate_interior_and_test_torus(R=2.0, r=1.0, n_grid=80, f_exact=None, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    eps = 1e-6
    bound_xz = R + r + 0.1
    bound_y = r + 0.1
    x_1d = torch.linspace(-bound_xz + eps, bound_xz - eps, n_grid)
    y_1d = torch.linspace(-bound_y + eps, bound_y - eps, n_grid)
    z_1d = torch.linspace(-bound_xz + eps, bound_xz - eps, n_grid)
    X, Y, Z = torch.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    mask = is_inside_torus(X, Y, Z, R, r)
    x_int = torch.stack([X[mask], Y[mask], Z[mask]], dim=1)
    x_int = x_int[torch.randperm(x_int.shape[0], generator=generator)]

    n_test = 50
    xt_xz = torch.linspace(-bound_xz, bound_xz, n_test)
    xt_y = torch.linspace(-bound_y, bound_y, n_test)
    Xt, Yt, Zt = torch.meshgrid(xt_xz, xt_y, xt_xz, indexing='ij')
    tmask = is_inside_torus(Xt, Yt, Zt, R, r)
    x_test = torch.stack([Xt[tmask], Yt[tmask], Zt[tmask]], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1], x_test[:, 2]).view(-1, 1)
    print(f"  Interior: {x_int.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    return x_int, x_test, y_test


def _sample_sphere(generator, n_points, domain_params):
    n_sample = int(n_points * 2.0)
    x = torch.rand(n_sample, generator=generator) * 2 - 1
    y = torch.rand(n_sample, generator=generator) * 2 - 1
    z = torch.rand(n_sample, generator=generator) * 2 - 1
    pts = torch.stack([x, y, z], dim=1)
    mask = is_inside_sphere(pts[:, 0], pts[:, 1], pts[:, 2])
    pts = pts[mask]
    return pts[:n_points] if pts.shape[0] > n_points else pts


def _sample_cylinder(generator, n_points, domain_params):
    n_sample = int(n_points * 2.0)
    x = torch.rand(n_sample, generator=generator) * 2 - 1
    y = torch.rand(n_sample, generator=generator) * 2 - 1
    z = torch.rand(n_sample, generator=generator) * 2 - 1
    pts = torch.stack([x, y, z], dim=1)
    mask = is_inside_cylinder(pts[:, 0], pts[:, 1], pts[:, 2])
    pts = pts[mask]
    return pts[:n_points] if pts.shape[0] > n_points else pts


def _sample_torus(generator, n_points, domain_params):
    R, r = domain_params.get('R', 2.0), domain_params.get('r', 1.0)
    bound_xz = R + r + 0.1
    bound_y = r + 0.1
    box_vol = (2 * bound_xz) ** 2 * (2 * bound_y)
    torus_vol = 2 * np.pi ** 2 * R * r ** 2
    oversample = max(box_vol / torus_vol * 1.5, 3.0)
    n_sample = int(n_points * oversample)
    x = torch.rand(n_sample, generator=generator) * 2 * bound_xz - bound_xz
    y = torch.rand(n_sample, generator=generator) * 2 * bound_y - bound_y
    z = torch.rand(n_sample, generator=generator) * 2 * bound_xz - bound_xz
    pts = torch.stack([x, y, z], dim=1)
    mask = is_inside_torus(pts[:, 0], pts[:, 1], pts[:, 2], R, r)
    pts = pts[mask]
    return pts[:n_points] if pts.shape[0] > n_points else pts


# ============================================================================
# Plotting — 3D cross-sections
# ============================================================================

def plot_3d_results(model, x_test, y_test, history, title, device='cpu',
                    output_path=None):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test.to(device)).cpu()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    epochs = range(1, len(history['total_loss']) + 1)
    ax.semilogy(epochs, history['bc_loss'], label='BC')
    ax.semilogy(epochs, history['pde_loss'], label='PDE')
    ax.semilogy(epochs, history['test_loss'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training History')

    ax = axes[0, 1]
    error = (y_pred - y_test).abs().numpy().flatten()
    ax.hist(error, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('|u_pred - u_exact|')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (mean={error.mean():.4e})')

    ax = axes[0, 2]
    if 'alpha' in history and history['alpha']:
        ax.semilogy(epochs, history['alpha'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('α')
        ax.set_title('Alpha Schedule')
    else:
        ax.text(0.5, 0.5, 'No alpha data', ha='center', va='center',
                transform=ax.transAxes)

    z_vals = x_test[:, 2].numpy()
    z_threshold = np.percentile(np.abs(z_vals), 10)
    z_mask = np.abs(z_vals) < max(z_threshold, 0.1)

    if z_mask.sum() > 10:
        xs = x_test[z_mask, 0].numpy()
        ys = x_test[z_mask, 1].numpy()
        exact_slice = y_test[z_mask].numpy().flatten()
        pred_slice = y_pred[z_mask].numpy().flatten()
        err_slice = np.abs(pred_slice - exact_slice)

        ax = axes[1, 0]
        sc = ax.scatter(xs, ys, c=exact_slice, s=2, cmap='viridis')
        plt.colorbar(sc, ax=ax)
        ax.set_title('Exact (z≈0 slice)')
        ax.set_aspect('equal')

        ax = axes[1, 1]
        sc = ax.scatter(xs, ys, c=pred_slice, s=2, cmap='viridis')
        plt.colorbar(sc, ax=ax)
        ax.set_title('Predicted (z≈0 slice)')
        ax.set_aspect('equal')

        ax = axes[1, 2]
        sc = ax.scatter(xs, ys, c=err_slice, s=2, cmap='hot')
        plt.colorbar(sc, ax=ax)
        ax.set_title('|Error| (z≈0 slice)')
        ax.set_aspect('equal')
    else:
        for ax in axes[1, :]:
            ax.text(0.5, 0.5, 'Not enough z≈0 points', ha='center',
                    va='center', transform=ax.transAxes)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    return fig


# ============================================================================
# run_example (unified 2D / 3D)
# ============================================================================

def run_example(case='boundary', domain='disc', n_epochs=100, n_interior_grid=1000,
                n_boundary_grid=10000, batch_size_bc=400, batch_size_pde=1000,
                save_plots=True, alpha_scheduler=None,
                val_split=0.0, patience=None,
                resample=True, activation='tanh', hidden_layers=None,
                scheduler_type=None, scheduler_end_factor=0.01,
                lr=1e-3, clip_residual=0.0, outlier_percentile=1.0,
                plot_pde_every=0, output_dir='outputs', outlier_off_epoch=None,
                save_npy=False, ring_radius=0.01, n_ring_pts=1000,
                positivity_weight=10.0, run_tag=None):
    if hidden_layers is None:
        hidden_layers = [128, 128, 128, 128]

    dim = 3 if domain in DOMAINS_3D else 2

    if case == 'origin':
        example_name = f"eikonal_origin_{domain}"
    elif dim == 3:
        example_name = f"eikonal3d_{domain}"
    else:
        example_name = f"eikonal_{domain}"

    base = os.path.join(output_dir, 'expr_6_4_eikonal', example_name)
    if run_tag:
        base = os.path.join(base, run_tag)
    setup_logging(base)

    print(f"\n{'=' * 60}")
    print(f"Eikonal Equation |∇u|² = 1 ({'3D' if dim == 3 else '2D'})")
    print(f"Case: {case} | Domain: {domain}")
    print(f"{'=' * 60}\n")

    output_dirs = create_output_dirs(base)
    f_exact = get_exact_solution(case, domain)
    bc_resample_fn = None

    # ----- 2D data generation -----
    if dim == 2:
        if case == 'origin':
            x_bc, y_bc = generate_origin_bc_data(domain, n_ring_pts=n_ring_pts,
                                                  ring_radius=ring_radius, seed=SEED)
            if domain == 'disc':
                domain_type, domain_params = 'disc', {'radius': 1.0}
                x_interior, x_test, y_test = generate_interior_and_test_disc(
                    radius=1.0, n_interior_grid=n_interior_grid, f_exact=f_exact, seed=SEED)
                title = "Eikonal: Distance to Origin (Disc)"
            elif domain == 'square':
                domain_type = 'square'
                domain_params = {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0}
                x_interior, x_test, y_test = generate_interior_and_test_square(
                    -1, 1, -1, 1, n_interior_grid=n_interior_grid, f_exact=f_exact, seed=SEED)
                title = "Eikonal: Distance to Origin (Square)"
            else:
                raise ValueError(f"Case 'origin' only supports disc or square, got '{domain}'")

        elif case == 'boundary':
            if domain == 'disc':
                x_bc, y_bc = generate_boundary_bc_data_disc(radius=1.0, n_boundary=n_boundary_grid, seed=SEED)
                domain_type, domain_params = 'disc', {'radius': 1.0}
                x_interior, x_test, y_test = generate_interior_and_test_disc(
                    radius=1.0, n_interior_grid=n_interior_grid, f_exact=f_exact, seed=SEED)
                title = "Eikonal: Distance to Boundary (Disc)"
            elif domain == 'square':
                x_bc, y_bc = generate_boundary_bc_data_square(n_per_edge=n_boundary_grid, seed=SEED)
                domain_type = 'square'
                domain_params = {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0}
                x_interior, x_test, y_test = generate_interior_and_test_square(
                    -1, 1, -1, 1, n_interior_grid=n_interior_grid, f_exact=f_exact, seed=SEED)
                title = "Eikonal: Distance to Boundary (Square)"
            elif domain in ELLIPSE_CONFIGS:
                config = ELLIPSE_CONFIGS[domain]
                x_bc, y_bc = generate_boundary_bc_data_ellipse(config, n_boundary=n_boundary_grid, seed=SEED)
                domain_type = 'ellipse'
                domain_params = {'config': config}
                x_interior, x_test, y_test = generate_interior_and_test_ellipse(
                    config, n_interior_grid=n_interior_grid, f_exact=f_exact, seed=SEED)
                title = f"Eikonal: Distance to Boundary ({config['name']})"
            else:
                raise ValueError(f"Unknown 2D domain: {domain}")
        else:
            raise ValueError(f"Unknown case: {case}")

    # ----- 3D data generation -----
    else:
        if domain == 'sphere':
            x_bc, y_bc = generate_boundary_bc_sphere(n_boundary=n_boundary_grid, seed=SEED)
            x_interior, x_test, y_test = generate_interior_and_test_sphere(
                n_grid=n_interior_grid, f_exact=f_exact, seed=SEED)
            domain_type, domain_params = 'sphere', {}
            title = "Eikonal 3D: Distance to Boundary (Sphere)"
        elif domain == 'cylinder':
            x_bc, y_bc = generate_boundary_bc_cylinder(n_per_part=n_boundary_grid // 3, seed=SEED)
            x_interior, x_test, y_test = generate_interior_and_test_cylinder(
                n_grid=n_interior_grid, f_exact=f_exact, seed=SEED)
            domain_type, domain_params = 'cylinder', {}
            title = "Eikonal 3D: Distance to Boundary (Cylinder)"
        elif domain == 'torus':
            x_bc, y_bc = generate_boundary_bc_torus(n_boundary=n_boundary_grid, seed=SEED)
            x_interior, x_test, y_test = generate_interior_and_test_torus(
                n_grid=n_interior_grid, f_exact=f_exact, seed=SEED)
            domain_type, domain_params = 'torus', {'R': 2.0, 'r': 1.0}
            title = "Eikonal 3D: Distance to Boundary (Torus)"
        else:
            raise ValueError(f"Unknown 3D domain: {domain}")

    print(f"  Boundary points: {x_bc.shape[0]:,}")

    # ----- interior sampler -----
    interior_sampler = None
    if resample:
        if dim == 2:
            custom = _sample_ellipse if domain_type == 'ellipse' else None
            interior_sampler = InteriorSampler(domain_type, domain_params, len(x_interior),
                                               custom_sampler=custom)
        else:
            sampler_map = {'sphere': _sample_sphere, 'cylinder': _sample_cylinder,
                           'torus': _sample_torus}
            interior_sampler = InteriorSampler(domain_type, domain_params, x_interior.shape[0],
                                               custom_sampler=sampler_map[domain])
        print(f"  Re-sampling enabled: ~{len(x_interior):,} interior points per epoch")
        if case == 'origin':
            bc_resample_fn = lambda: generate_origin_bc_data(
                domain, n_ring_pts=n_ring_pts, ring_radius=ring_radius,
                seed=SEED + int(torch.randint(0, 100000, (1,)).item()))
            print(f"  BC re-sampling enabled: origin case with radius={ring_radius}")

    x_bc_val, y_bc_val, x_interior_val = None, None, None
    if val_split > 0:
        print(f"\n  Splitting: {1 - val_split:.0%} train, {val_split:.0%} val")
        x_bc, x_bc_val, y_bc, y_bc_val = split_data(x_bc, y_bc, train_ratio=1 - val_split, seed=SEED)
        x_interior, x_interior_val, _, _ = split_data(x_interior, None, train_ratio=1 - val_split, seed=SEED + 1)
        print(f"  BC: {len(x_bc):,} train, {len(x_bc_val):,} val")
        print(f"  Interior: {len(x_interior):,} train, {len(x_interior_val):,} val")

    model = EikonalPINN(hidden_layers=hidden_layers, activation=activation,
                        clip_residual=clip_residual, outlier_percentile=outlier_percentile,
                        positivity_weight=positivity_weight, input_dim=dim)
    model.to(device)

    print(f"\nModel: EikonalPINN (input_dim={dim})")
    print(f"  Hidden layers: {hidden_layers}, activation: {activation}")
    if clip_residual > 0:
        print(f"  Residual clipping: [-{clip_residual}, {clip_residual}]")
    if outlier_percentile > 0:
        print(f"  Outlier removal: top {outlier_percentile}%")
    print(f"  Positivity weight: {positivity_weight}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history, training_time = train_pinn(
        model, x_bc, y_bc, x_interior, x_test, y_test,
        n_epochs=n_epochs, batch_size_bc=batch_size_bc, batch_size_pde=batch_size_pde,
        lr=lr, alpha_scheduler=alpha_scheduler,
        x_bc_val=x_bc_val, y_bc_val=y_bc_val, x_interior_val=x_interior_val,
        patience=patience, interior_sampler=interior_sampler,
        bc_resample_fn=bc_resample_fn,
        scheduler_type=scheduler_type, scheduler_end_factor=scheduler_end_factor,
        plot_pde_every=plot_pde_every, example_name=example_name,
        output_dirs=output_dirs, outlier_off_epoch=outlier_off_epoch,
        seed=SEED, device=device,
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

    model.eval()
    with torch.no_grad():
        y_pred_best = model(x_test_dev).cpu()
    final_ckpt_path = os.path.join(output_dirs['checkpoints'], 'final_model.pt')
    if os.path.exists(final_ckpt_path):
        ckpt = torch.load(final_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        with torch.no_grad():
            y_pred_final = model(x_test_dev).cpu()
        best_path = os.path.join(output_dirs['checkpoints'], 'best_model.pt')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model_state_dict'])
            model.eval()
    else:
        y_pred_final = y_pred_best.clone()

    if save_npy:
        npy_dir = output_dirs['npy']
        history_data = {k: np.array([v if v is not None else np.nan for v in vals])
                        for k, vals in history.items()}
        np.savez(os.path.join(npy_dir, 'training_history.npz'), **history_data)
        coords = {'x': x_test[:, 0].numpy(), 'y': x_test[:, 1].numpy()}
        if dim == 3:
            coords['z'] = x_test[:, 2].numpy()
        np.savez(os.path.join(npy_dir, 'exact_solution.npz'),
                 **coords, u_exact=y_test.numpy().flatten())
        np.savez(os.path.join(npy_dir, 'predictions.npz'),
                 **coords,
                 u_pred_best=y_pred_best.numpy().flatten(),
                 u_pred_final=y_pred_final.numpy().flatten())
        print(f"Saved .npz files to {npy_dir}/")

    if dim == 3:
        plot_path = os.path.join(output_dirs['plots'], 'results.png') if save_plots else None
        fig = plot_3d_results(model, x_test, y_test, history, title, device=device,
                              output_path=plot_path)
    else:
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
    ALL_DOMAINS = sorted(DOMAINS_2D | DOMAINS_3D)

    parser = argparse.ArgumentParser(
        description='Eikonal Equation |nabla u|^2 = 1 (2D and 3D)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--case', type=str, default='boundary', choices=['origin', 'boundary'])
    parser.add_argument('--domain', type=str, default='disc', choices=ALL_DOMAINS)
    parser.add_argument('--ring-radius', type=float, default=0.01)
    parser.add_argument('--n-ring-pts', type=int, default=1000)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--interior-grid', type=int, default=None,
                        help='Grid points per axis (default: 1000 for 2D, 100 for 3D)')
    parser.add_argument('--boundary-grid', type=int, default=None,
                        help='Boundary points (default: 10000 for 2D, 20000 for 3D)')
    parser.add_argument('--batch-bc', type=int, default=400)
    parser.add_argument('--batch-pde', type=int, default=1000)
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'leaky_relu', 'elu', 'gelu', 'softplus', 'silu', 'sigmoid'])
    parser.add_argument('--hidden-layers', type=str, default='128,128,128,128')

    parser.add_argument('--clip-residual', type=float, default=10.0)
    parser.add_argument('--outlier-percentile', type=float, default=2.0)
    parser.add_argument('--outlier-off-epoch', type=int, default=None)
    parser.add_argument('--positivity-weight', type=float, default=1.0,
                        help='Weight for the u>=0 positivity penalty in PDE residual')
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

    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine', 'step', 'exponential', 'none'])
    parser.add_argument('--scheduler-end-factor', type=float, default=0.01)

    args = parser.parse_args()

    is_3d = args.domain in DOMAINS_3D

    if args.case == 'origin' and is_3d:
        parser.error(f"Case 'origin' is only supported for 2D domains, got '{args.domain}'")
    if args.case == 'origin' and args.domain not in ('disc', 'square'):
        parser.error(f"Case 'origin' only supports disc or square, got '{args.domain}'")

    if args.interior_grid is None:
        args.interior_grid = 100 if is_3d else 1000
    if args.boundary_grid is None:
        args.boundary_grid = 20000 if is_3d else 10000

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
    run_example(
        case=args.case, domain=args.domain,
        n_epochs=args.epochs, n_interior_grid=args.interior_grid,
        n_boundary_grid=args.boundary_grid,
        batch_size_bc=args.batch_bc, batch_size_pde=args.batch_pde,
        save_plots=not args.no_save,
        alpha_scheduler=alpha_scheduler,
        val_split=args.val_split, patience=args.patience,
        resample=not args.no_resample,
        activation=args.activation, hidden_layers=hidden_layers,
        scheduler_type=args.scheduler, scheduler_end_factor=args.scheduler_end_factor,
        lr=args.lr, clip_residual=args.clip_residual,
        outlier_percentile=args.outlier_percentile,
        plot_pde_every=args.plot_pde_every, output_dir=args.output_dir,
        outlier_off_epoch=args.outlier_off_epoch,
        save_npy=args.save_npy,
        ring_radius=args.ring_radius, n_ring_pts=args.n_ring_pts,
        positivity_weight=args.positivity_weight,
        run_tag=args.run_tag)
