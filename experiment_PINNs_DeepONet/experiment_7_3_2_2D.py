#!/usr/bin/env python3
"""
Experiment 6.3.2: Distance to Boundary DeepONet 2D Example

This script implements the DeepONet approach for learning the p-Laplacian operator
for the distance to boundary problem on 2D domains, as described in Section 6.3.2
of the paper "Solving p-Laplacian and infinity-Laplacian with Deep Learning".

The BVP is:
    Δ_p u_p = -1, in Ω
    u_p = 0, on ∂Ω

The limiting behavior as p → ∞ gives:
    u_∞ = dist(x, ∂Ω)

Four 2D domains:
    1. disc: x² + y² ≤ 1 (unit disc)
    2. ellipse1: x² + 16y² ≤ 1 (elongated ellipse, a=1, b=1/4)
    3. ellipse2: 8.5x² + 8.5y² - 15xy ≤ 1 (rotated ellipse, θ=-π/4)
    4. ellipse3: 4x² + y² ≤ 1 (compressed ellipse, a=1/2, b=1)

Usage:
    python experiment_6_3_2_2D.py --domain disc --epochs 20
    python experiment_6_3_2_2D.py --domain ellipse1 --epochs 20
    python experiment_6_3_2_2D.py --domain all --epochs 20
    python experiment_6_3_2_2D.py --domain all --epochs 20 --no-exact-inf
"""

import argparse
import os
import random
import glob
import re
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

# Optional: scipy for loading .mat files
try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Domain configurations
DOMAIN_CONFIGS = {
    'disc': {
        'name': 'Unit Disc',
        'equation': 'x² + y² ≤ 1',
        'is_rectangle': 0.0,
        'a': 1.0,
        'b': 1.0,
        'theta': 0.0,
        'trunk_layers': [2, 256, 256, 128],
        'data_subdir': 'Distance to boundary/2D examples/Unit disc',
        'mat_pattern': 'p{}discBoundary.mat',
        'max_training_p': 200,
    },
    'ellipse1': {
        'name': 'Ellipse 1',
        'equation': 'x² + 16y² ≤ 1',
        'is_rectangle': 0.1,
        'a': 1.0,
        'b': 0.25,
        'theta': 0.0,
        'trunk_layers': [2, 512, 512, 128],
        'data_subdir': 'Distance to boundary/2D examples/Ellipse 1',
        'mat_pattern': 'p{}ellipse1Boundary.mat',
        'max_training_p': 110,
    },
    'ellipse2': {
        'name': 'Ellipse 2 (rotated)',
        'equation': '8.5x² + 8.5y² - 15xy ≤ 1',
        'is_rectangle': 0.2,
        'a': 1.0,
        'b': 0.25,
        'theta': -np.pi / 4,
        'trunk_layers': [2, 512, 512, 128],
        'data_subdir': 'Distance to boundary/2D examples/Ellipse 2',
        'mat_pattern': 'p{}ellipse2Boundary.mat',
        'max_training_p': 110,
    },
    'ellipse3': {
        'name': 'Ellipse 3',
        'equation': '4x² + y² ≤ 1',
        'is_rectangle': 0.3,
        'a': 0.5,
        'b': 1.0,
        'theta': 0.0,
        'trunk_layers': [2, 512, 512, 128],
        'data_subdir': 'Distance to boundary/2D examples/Ellipse 3',
        'mat_pattern': 'p{}ellipse3Boundary.mat',
        'max_training_p': 200,
    },
}


def set_seed(seed: int = 1234):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def solve_4th_order(a_coef, b_coef, c_coef, d_coef, e_coef):
    """
    Solve 4th order polynomial equation for ellipse distance calculation.
    Uses Ferrari's solution for quartic equations.
    """
    p = (8 * a_coef * c_coef - 3 * b_coef**2) / (8 * a_coef**2)
    q = (b_coef**3 - 4 * a_coef * b_coef * c_coef + 8 * a_coef**2 * d_coef) / (8 * a_coef**3)
    delta0 = c_coef**2 - 3 * b_coef * d_coef + 12 * a_coef * e_coef
    delta1 = 2 * c_coef**3 - 9 * b_coef * c_coef * d_coef + 27 * b_coef**2 * e_coef + \
             27 * a_coef * d_coef**2 - 72 * a_coef * c_coef * e_coef
    
    delta0_0 = (torch.abs(delta0) < 0.001) * 1.0
    
    Q = torch.pow(
        (delta1 + ((1 - delta0_0) * torch.sqrt((delta1**2 - 4 * delta0**3).to(torch.complex64)) + 
         delta0_0 * delta1).to(torch.complex64)) / 2,
        1/3
    )
    
    S = torch.sqrt(-2/3 * p + (Q + delta0 / (Q)) / 3 / a_coef) / 2
    
    sol_list = []
    sol_list.append(torch.real(-b_coef / 4 / a_coef - S - torch.sqrt(-4 * S**2 - 2 * p + q / S) / 2))
    sol_list.append(torch.real(-b_coef / 4 / a_coef - S + torch.sqrt(-4 * S**2 - 2 * p + q / S) / 2))
    sol_list.append(torch.real(-b_coef / 4 / a_coef + S - torch.sqrt(-4 * S**2 - 2 * p - q / S) / 2))
    sol_list.append(torch.real(-b_coef / 4 / a_coef + S + torch.sqrt(-4 * S**2 - 2 * p - q / S) / 2))
    
    return torch.vstack(sol_list)


def distance_to_disc_boundary(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Exact solution for distance to boundary of unit disc.
    u_∞(x, y) = 1 - sqrt(x² + y²)
    """
    return 1.0 - torch.sqrt(x**2 + y**2)


def distance_to_ellipse_boundary(
    x_list: torch.Tensor, 
    y_list: torch.Tensor, 
    a: float, 
    b: float, 
    EPS: float = 1e-6
) -> torch.Tensor:
    """
    Compute distance to boundary of ellipse x²/a² + y²/b² = 1.
    Uses analytical solution via 4th order polynomial.
    """
    aa = torch.ones_like(x_list)
    bb = torch.ones_like(x_list) * (-2 * a**2 - 2 * b**2)
    cc = a**4 + b**4 + 4 * a**2 * b**2 - a**2 * x_list**2 - b**2 * y_list**2
    dd = -2 * a**2 * b**4 - 2 * a**4 * b**2 + 2 * a**2 * b**2 * x_list**2 + 2 * a**2 * b**2 * y_list**2
    ee = a**4 * b**4 - a**2 * b**4 * x_list**2 - a**4 * b**2 * y_list**2
    
    sol = solve_4th_order(aa, bb, cc, dd, ee)
    
    # Compute boundary points
    point_bound_X = (torch.abs(a**2 - sol) > EPS) * 1.0 * a**2 * x_list.repeat(4, 1) / (a**2 - sol + EPS) + \
                    (torch.abs(a**2 - sol) <= EPS) * 1.0 * a * ((x_list.repeat(4, 1) > 0) * 2 - 1)
    point_bound_X *= (torch.abs(torch.abs(x_list.repeat(4, 1))) >= 0.0001) * 1.0
    point_bound_X += (torch.abs(torch.abs(x_list.repeat(4, 1))) < 0.0001) * \
                     (torch.abs(y_list.repeat(4, 1)) < 0.1) * 1.0 * a * ((x_list.repeat(4, 1) > 0) * 2 - 1)
    
    point_bound_Y = (torch.abs(b**2 - sol) > EPS) * 1.0 * b**2 * y_list.repeat(4, 1) / (b**2 - sol + EPS) + \
                    (torch.abs(b**2 - sol) <= EPS) * 1.0 * b * ((y_list.repeat(4, 1) > 0) * 2 - 1)
    point_bound_Y *= (torch.abs(torch.abs(y_list.repeat(4, 1))) >= 0.0001) * 1.0
    point_bound_Y += (torch.abs(torch.abs(y_list.repeat(4, 1))) < 0.0001) * \
                     (torch.abs(x_list.repeat(4, 1)) < 0.1) * 1.0 * b * ((y_list.repeat(4, 1) > 0) * 2 - 1)
    
    # Compute distances
    distances = torch.sqrt((point_bound_X - x_list.repeat(4, 1))**2 + 
                          (point_bound_Y - y_list.repeat(4, 1))**2)
    distances = torch.nan_to_num(distances, nan=1000000.0)
    distances = torch.min(distances, axis=0)[0]
    distances[distances == 1000000.0] = float(min(a, b))
    distances[x_list**2 + y_list**2 < EPS] = float(min(a, b))
    
    return distances


def exact_solution(
    x: torch.Tensor, 
    y: torch.Tensor, 
    domain: str, 
    config: dict
) -> torch.Tensor:
    """
    Compute exact solution (distance to boundary) for the given domain.
    """
    if domain == 'disc':
        return distance_to_disc_boundary(x, y)
    else:
        # For rotated ellipse, transform coordinates first
        if config['theta'] != 0:
            theta = config['theta']
            x_rot = np.cos(theta) * x - np.sin(theta) * y
            y_rot = np.sin(theta) * x + np.cos(theta) * y
            return distance_to_ellipse_boundary(x_rot, y_rot, config['a'], config['b'])
        else:
            return distance_to_ellipse_boundary(x, y, config['a'], config['b'])


class DeepONetDataset(Dataset):
    """Dataset for DeepONet training."""
    
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DeepONet(nn.Module):
    """
    DeepONet architecture for learning the p-Laplacian operator.
    
    Architecture (matching Section 6.3.2 notebooks):
        - Trunk net: Takes spatial coordinates (x, y) as input
          Default layers: [2, 512, 512, 128] for ellipses, [2, 256, 256, 128] for disc
        - Branch net: Takes p value (normalized by 500) as input
          layers: [1, 128, 128, 128]
        - Output: Element-wise product of trunk and branch outputs (128-dim),
          followed by final linear layer (128 → 1)
    """
    
    def __init__(
        self,
        trunk_layers: list = [2, 512, 512, 128],
        branch_layers: list = [1, 128, 128, 128],
        activation: str = 'tanh',
        device: torch.device = None
    ):
        super().__init__()
        
        self.device = device if device else torch.device('cpu')
        
        # Activation function
        if activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()
        
        # Loss function
        self.loss_function = nn.MSELoss(reduction='mean')
        
        # Trunk network (spatial coordinates)
        self.trunk_layers = trunk_layers
        self.trunk_linears = nn.ModuleList([
            nn.Linear(trunk_layers[i], trunk_layers[i+1])
            for i in range(len(trunk_layers) - 1)
        ])
        
        # Branch network (p value)
        self.branch_layers = branch_layers
        self.branch_linears = nn.ModuleList([
            nn.Linear(branch_layers[i], branch_layers[i+1])
            for i in range(len(branch_layers) - 1)
        ])
        
        # Final combination layer (after element-wise multiplication)
        assert trunk_layers[-1] == branch_layers[-1], \
            "Trunk and branch output dimensions must match"
        self.output_layer = nn.Linear(trunk_layers[-1], 1)
        
        # Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier normal initialization."""
        for linear in self.trunk_linears:
            nn.init.xavier_normal_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
        for linear in self.branch_linears:
            nn.init.xavier_normal_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
        nn.init.xavier_normal_(self.output_layer.weight, gain=1.0)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (matching notebook implementation).
        
        Args:
            x: Input tensor of shape (N, 3) where columns are (x, y, p_normalized)
        
        Returns:
            Output tensor of shape (N, 1)
        """
        # Split input into spatial coordinates and p value
        a = x[:, 0:2]  # (x, y) - trunk input
        b = x[:, 2:3]  # p/500 - branch input
        
        # Trunk network
        for i in range(len(self.trunk_layers) - 2):
            z = self.trunk_linears[i](a)
            a = self.activation(z)
        a = self.trunk_linears[-1](a)  # Last layer, no activation
        
        # Branch network
        for i in range(len(self.branch_layers) - 2):
            z = self.branch_linears[i](b)
            b = self.activation(z)
        b = self.branch_linears[-1](b)  # Last layer, no activation
        
        # Combine: element-wise product
        out = a * b
        
        # Final output layer
        out = self.output_layer(out)
        
        return out
    
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between prediction and target."""
        pred = self.forward(x)
        return self.loss_function(pred, y)


def is_inside_domain(x: torch.Tensor, y: torch.Tensor, domain: str, config: dict) -> torch.Tensor:
    """Check if points are inside the domain."""
    if domain == 'disc':
        return x**2 + y**2 <= 1
    elif domain == 'ellipse1':
        return x**2 + 16 * y**2 <= 1
    elif domain == 'ellipse2':
        return 8.5 * x**2 + 8.5 * y**2 - 15 * x * y <= 1
    elif domain == 'ellipse3':
        return 4 * x**2 + y**2 <= 1
    else:
        return x**2 + y**2 <= 1


def generate_domain_points(domain: str, config: dict, n_points: int, normalize: bool = True) -> np.ndarray:
    """Generate random points inside the domain."""
    points = []
    while len(points) < n_points:
        x = np.random.uniform(-1, 1, n_points * 2)
        y = np.random.uniform(-1, 1, n_points * 2)
        
        if domain == 'disc':
            mask = x**2 + y**2 <= 1
        elif domain == 'ellipse1':
            mask = x**2 + 16 * y**2 <= 1
        elif domain == 'ellipse2':
            mask = 8.5 * x**2 + 8.5 * y**2 - 15 * x * y <= 1
        elif domain == 'ellipse3':
            mask = 4 * x**2 + y**2 <= 1
        else:
            mask = x**2 + y**2 <= 1
        
        valid_points = np.column_stack([x[mask], y[mask]])
        points.extend(valid_points.tolist())
    
    points = np.array(points[:n_points])
    
    if normalize:
        points = points / 2.0 + 0.5
    
    return points


def generate_test_grid(
    domain: str,
    config: dict,
    n_points: int = 101,
    normalize: bool = True
) -> tuple:
    """Generate a test grid for evaluation."""
    x = torch.linspace(-1, 1, n_points)
    y = torch.linspace(-1, 1, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Filter points inside domain
    mask = is_inside_domain(X_flat, Y_flat, domain, config)
    X_flat = X_flat[mask]
    Y_flat = Y_flat[mask]
    
    # Compute exact solution
    y_exact = exact_solution(X_flat, Y_flat, domain, config)
    
    # Create test points
    x_test = torch.stack([X_flat, Y_flat], dim=1)
    y_test = y_exact.unsqueeze(1)
    
    if normalize:
        x_test = x_test / 2.0 + 0.5
    
    return x_test, y_test


def load_mat_data(
    data_dir: str,
    domain: str,
    config: dict,
    p_normalize: float = 500.0,
    include_exact_inf: bool = True
) -> tuple:
    """Load training data from .mat files."""
    if not HAS_SCIPY:
        raise ImportError("scipy is required to load .mat files")
    
    mat_path = os.path.join(data_dir, config['data_subdir'])
    if not os.path.exists(mat_path):
        return None, None
    
    mat_pattern = config['mat_pattern']
    max_p = config['max_training_p']
    
    X_train_list = []
    Y_train_list = []
    last_pts = None
    
    # Try to load data for various p values
    for p in range(5, max_p + 1, 5):
        mat_file = os.path.join(mat_path, mat_pattern.format(p))
        if os.path.exists(mat_file):
            try:
                mat = scipy.io.loadmat(mat_file)
                pts = mat['pts'] / 2.0 + 0.5  # Normalize to [0, 1]
                solution = mat['solution'].flatten().reshape(-1, 1)
                p_value = np.ones((pts.shape[0], 1)) * p / p_normalize
                
                X = np.hstack([pts, p_value])
                Y = solution
                
                if X.shape[0] == Y.shape[0]:
                    X_train_list.append(X)
                    Y_train_list.append(Y)
                    last_pts = mat['pts']
                    print(f"  Loaded p={p}: {X.shape[0]} points")
            except Exception as e:
                print(f"  Warning: Could not load {mat_file}: {e}")
    
    if not X_train_list:
        return None, None
    
    # Add exact solution at p=500 (surrogate for p=∞)
    if include_exact_inf and last_pts is not None:
        # Try to load pre-computed infinity approximation
        inf_npy = os.path.join(mat_path, 'inf_pinns.npy')
        if os.path.exists(inf_npy):
            with open(inf_npy, 'rb') as f:
                Y_inf = np.load(f)
            print(f"  Loaded pre-computed p=∞ solution: {Y_inf.shape[0]} points")
        else:
            # Compute exact solution
            Y_inf = exact_solution(
                torch.tensor(last_pts[:, 0]),
                torch.tensor(last_pts[:, 1]),
                domain,
                config
            ).numpy().reshape(-1, 1)
            print(f"  Computed exact p=∞ solution: {Y_inf.shape[0]} points")
        
        pts_norm = last_pts / 2.0 + 0.5
        p_value = np.ones((pts_norm.shape[0], 1)) * 500.0 / p_normalize
        X_inf = np.hstack([pts_norm, p_value])
        
        X_train_list.append(X_inf)
        Y_train_list.append(Y_inf)
    
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)
    
    return X_train, Y_train


def generate_synthetic_data(
    domain: str,
    config: dict,
    p_values: list = None,
    n_points_per_p: int = 5000,
    p_normalize: float = 500.0,
    include_exact_inf: bool = True
) -> tuple:
    """Generate synthetic training data."""
    if p_values is None:
        max_p = config['max_training_p']
        p_values = list(range(5, min(max_p + 1, 201), 5))
    
    X_train_list = []
    Y_train_list = []
    
    for p in p_values:
        pts = generate_domain_points(domain, config, n_points_per_p, normalize=False)
        
        # Exact solution at infinity
        u_inf = exact_solution(
            torch.tensor(pts[:, 0]),
            torch.tensor(pts[:, 1]),
            domain,
            config
        ).numpy()
        
        # Approximate u_p based on 1/p convergence
        C = 0.1
        u_p = u_inf - C / p
        u_p = np.maximum(u_p, 0)
        
        # Normalize coordinates
        pts_norm = pts / 2.0 + 0.5
        p_value = np.ones((pts_norm.shape[0], 1)) * p / p_normalize
        
        X = np.hstack([pts_norm, p_value])
        Y = u_p.reshape(-1, 1)
        
        X_train_list.append(X)
        Y_train_list.append(Y)
        print(f"  Generated synthetic data for p={p}: {X.shape[0]} points")
    
    # Add exact solution at p=500
    if include_exact_inf:
        pts = generate_domain_points(domain, config, n_points_per_p, normalize=False)
        pts_norm = pts / 2.0 + 0.5
        p_value = np.ones((pts_norm.shape[0], 1)) * 500.0 / p_normalize
        u_inf = exact_solution(
            torch.tensor(pts[:, 0]),
            torch.tensor(pts[:, 1]),
            domain,
            config
        ).numpy()
        
        X_inf = np.hstack([pts_norm, p_value])
        Y_inf = u_inf.reshape(-1, 1)
        
        X_train_list.append(X_inf)
        Y_train_list.append(Y_inf)
        print(f"  Added exact solution at p=500 (∞): {X_inf.shape[0]} points")
    
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)
    
    return X_train, Y_train


def train_deeponet(
    model: DeepONet,
    train_loader: DataLoader,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    n_epochs: int,
    lr: float,
    device: torch.device,
    show_every: int = 1
) -> dict:
    """Train the DeepONet model."""
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    
    loss_history = {
        'train_loss': [],
        'test_loss_p200': [],
        'test_loss_p250': [],
        'test_loss_p500': []
    }
    
    print("Training Loss ----- Test Loss (p=200) ----- Test Loss (p=250) ----- Test Loss (p=500)")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(X_batch, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches
        
        # Evaluate on test set at different p values
        model.eval()
        with torch.no_grad():
            test_x_p200 = test_x.clone()
            test_x_p200[:, 2] = 200 / 500.0
            test_loss_p200 = model.compute_loss(test_x_p200.to(device), test_y.to(device))
            
            test_x_p250 = test_x.clone()
            test_x_p250[:, 2] = 250 / 500.0
            test_loss_p250 = model.compute_loss(test_x_p250.to(device), test_y.to(device))
            
            test_x_p500 = test_x.clone()
            test_x_p500[:, 2] = 500 / 500.0
            test_loss_p500 = model.compute_loss(test_x_p500.to(device), test_y.to(device))
        
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['test_loss_p200'].append(test_loss_p200.item())
        loss_history['test_loss_p250'].append(test_loss_p250.item())
        loss_history['test_loss_p500'].append(test_loss_p500.item())
        
        if epoch % show_every == 0:
            print(f"{epoch} --- {avg_train_loss:.6e} --- {test_loss_p200.item():.6e} --- "
                  f"{test_loss_p250.item():.6e} --- {test_loss_p500.item():.6e}")
    
    return loss_history


def evaluate_over_p_range(
    model: DeepONet,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    p_range: list = None
) -> dict:
    """Evaluate model over a range of p values."""
    if p_range is None:
        p_range = list(range(5, 501, 5))
    
    results = {'p': [], 'mse': []}
    
    model.eval()
    with torch.no_grad():
        for p in p_range:
            test_x_p = test_x.clone()
            test_x_p[:, 2] = p / 500.0
            test_loss = model.compute_loss(test_x_p.to(device), test_y.to(device))
            
            results['p'].append(p)
            results['mse'].append(test_loss.item())
    
    return results


def plot_results(
    loss_history: dict,
    eval_results: dict,
    domain: str,
    config: dict,
    output_dir: str
):
    """Plot training history and evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Training loss history
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(len(loss_history['train_loss']))
    ax.semilogy(epochs, loss_history['train_loss'], label='Train Loss')
    ax.semilogy(epochs, loss_history['test_loss_p200'], label='Test Loss (p=200)')
    ax.semilogy(epochs, loss_history['test_loss_p250'], label='Test Loss (p=250)')
    ax.semilogy(epochs, loss_history['test_loss_p500'], label='Test Loss (p=500/∞)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Training History - {config["name"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_history_{domain}.png'), dpi=150)
    plt.close()
    
    # Plot 2: MSE over p range
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(eval_results['p'], eval_results['mse'], 'b-o', markersize=3)
    ax.set_xlabel('p')
    ax.set_ylabel('MSE (vs exact solution at p=∞)')
    ax.set_title(f'DeepONet MSE vs p - {config["name"]}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mse_vs_p_{domain}.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


def plot_prediction_comparison(
    model: DeepONet,
    domain: str,
    config: dict,
    device: torch.device,
    output_dir: str,
    p_value: int = 200,
    n_points: int = 101
):
    """Plot comparison between model prediction and exact solution."""
    os.makedirs(output_dir, exist_ok=True)
    
    x = torch.linspace(-1, 1, n_points)
    y = torch.linspace(-1, 1, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create mask for domain
    mask = is_inside_domain(X, Y, domain, config)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Normalize and add p value
    X_norm = X_flat / 2.0 + 0.5
    Y_norm = Y_flat / 2.0 + 0.5
    p_norm = torch.full_like(X_flat, p_value / 500.0)
    
    test_input = torch.stack([X_norm, Y_norm, p_norm], dim=1)
    
    model.eval()
    with torch.no_grad():
        pred = model(test_input.to(device)).cpu()
    
    # Exact solution
    exact = exact_solution(X_flat, Y_flat, domain, config)
    
    # Reshape
    pred_grid = pred.squeeze().reshape(n_points, n_points).numpy()
    exact_grid = exact.reshape(n_points, n_points).numpy()
    
    # Apply mask
    pred_grid = np.where(mask.numpy(), pred_grid, np.nan)
    exact_grid = np.where(mask.numpy(), exact_grid, np.nan)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    X_np = X.numpy()
    Y_np = Y.numpy()
    
    im0 = axes[0].contourf(X_np, Y_np, pred_grid, levels=20, cmap='rainbow')
    axes[0].set_title(f'DeepONet Prediction (p={p_value})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(X_np, Y_np, exact_grid, levels=20, cmap='rainbow')
    axes[1].set_title('Exact Solution (p=∞)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1])
    
    error_grid = np.abs(pred_grid - exact_grid)
    im2 = axes[2].contourf(X_np, Y_np, error_grid, levels=20, cmap='hot')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(f'{config["name"]}: {config["equation"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prediction_comparison_{domain}_p{p_value}.png'), dpi=150)
    plt.close()


def run_experiment(
    domain: str,
    epochs: int,
    lr: float,
    batch_size: int,
    branch_layers: list,
    data_dir: str,
    output_dir: str,
    seed: int,
    use_synthetic: bool = False,
    include_exact_inf: bool = True
):
    """Run the full DeepONet experiment for a given domain."""
    config = DOMAIN_CONFIGS[domain]
    trunk_layers = config['trunk_layers']
    
    print(f"\n{'='*60}")
    print(f"Running Experiment 6.3.2: Distance to Boundary - {config['name']}")
    print(f"Domain: {config['equation']}")
    print(f"{'='*60}")
    
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load or generate training data
    print("\nLoading training data...")
    print(f"  Include exact p=∞ solution: {include_exact_inf}")
    if use_synthetic:
        X_train, Y_train = generate_synthetic_data(
            domain, config, include_exact_inf=include_exact_inf
        )
    else:
        X_train, Y_train = load_mat_data(
            data_dir, domain, config, include_exact_inf=include_exact_inf
        )
        if X_train is None:
            print(f"  No .mat files found in {data_dir}/{config['data_subdir']}/")
            print("  Generating synthetic training data instead...")
            X_train, Y_train = generate_synthetic_data(
                domain, config, include_exact_inf=include_exact_inf
            )
    
    print(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")
    
    # Create dataset and dataloader
    train_dataset = DeepONetDataset(X_train, Y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Generate test data
    print("\nGenerating test grid...")
    test_x, test_y = generate_test_grid(domain, config, n_points=101, normalize=True)
    test_x = torch.cat([test_x, torch.zeros(test_x.shape[0], 1)], dim=1)
    print(f"Test data shape: X={test_x.shape}, Y={test_y.shape}")
    
    # Create model
    print("\nCreating DeepONet model...")
    print(f"  Trunk layers: {trunk_layers}")
    print(f"  Branch layers: {branch_layers}")
    model = DeepONet(
        trunk_layers=trunk_layers,
        branch_layers=branch_layers,
        activation='tanh',
        device=device
    )
    model.to(device)
    print(model)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    loss_history = train_deeponet(
        model=model,
        train_loader=train_loader,
        test_x=test_x,
        test_y=test_y,
        n_epochs=epochs,
        lr=lr,
        device=device,
        show_every=1
    )
    
    # Evaluate over p range
    print("\nEvaluating over p range...")
    eval_results = evaluate_over_p_range(model, test_x, test_y, device)
    
    # Print results
    print("\nResults: MSE vs p")
    print("-" * 30)
    for i, (p, mse) in enumerate(zip(eval_results['p'], eval_results['mse'])):
        if i % 10 == 0:
            print(f"p={p:3d}: MSE={mse:.6e}")
    
    # Plot results
    domain_output_dir = os.path.join(output_dir, domain)
    plot_results(loss_history, eval_results, domain, config, domain_output_dir)
    plot_prediction_comparison(model, domain, config, device, domain_output_dir, p_value=200)
    plot_prediction_comparison(model, domain, config, device, domain_output_dir, p_value=500)
    
    # Save model
    suffix = '' if include_exact_inf else '_no_inf'
    model_path = os.path.join(domain_output_dir, f'deeponet_{domain}{suffix}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'trunk_layers': trunk_layers,
        'branch_layers': branch_layers,
        'loss_history': loss_history,
        'eval_results': eval_results,
        'seed': seed,
        'include_exact_inf': include_exact_inf,
        'domain_config': config
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, loss_history, eval_results


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 6.3.2: Distance to Boundary DeepONet 2D'
    )
    parser.add_argument(
        '--domain', type=str, default='disc',
        choices=['disc', 'ellipse1', 'ellipse2', 'ellipse3', 'all'],
        help='Domain type: disc, ellipse1, ellipse2, ellipse3, or all'
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size', type=int, default=2048,
        help='Batch size'
    )
    parser.add_argument(
        '--branch-layers', type=str, default='1,128,128,128',
        help='Branch network layers (comma-separated). Default: 1,128,128,128'
    )
    parser.add_argument(
        '--data-dir', type=str, default='./experiment_6_3_2',
        help='Directory containing training data (.mat files)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./outputs/experiment_6_3_2_2D',
        help='Output directory for plots and models'
    )
    parser.add_argument(
        '--seed', type=int, default=1234,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic data instead of .mat files'
    )
    parser.add_argument(
        '--no-exact-inf', action='store_true',
        help='Do NOT include exact solution at p=500 (p=∞) in training data'
    )
    
    args = parser.parse_args()
    
    # Parse branch layers
    branch_layers = [int(x) for x in args.branch_layers.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    if args.domain == 'all':
        domains = ['disc', 'ellipse1', 'ellipse2', 'ellipse3']
    else:
        domains = [args.domain]
    
    for domain in domains:
        run_experiment(
            domain=domain,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            branch_layers=branch_layers,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            use_synthetic=args.synthetic,
            include_exact_inf=not args.no_exact_inf
        )
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    main()
