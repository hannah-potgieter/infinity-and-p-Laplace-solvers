#!/usr/bin/env python3
"""
Experiment 6.3.2: Distance to Boundary DeepONet 3D Example

This script implements the DeepONet approach for learning the p-Laplacian operator
for the distance to boundary problem on 3D domains, as described in Section 6.3.2
of the paper "Solving p-Laplacian and infinity-Laplacian with Deep Learning".

The BVP is:
    Δ_p u_p = -1, in Ω
    u_p = 0, on ∂Ω

The limiting behavior as p → ∞ gives:
    u_∞ = dist(x, ∂Ω)

Three 3D domains:
    1. sphere: x² + y² + z² ≤ 1 (unit sphere)
    2. cylinder: y² + z² ≤ 1, -1 ≤ x ≤ 1 (radius 1, length 2)
    3. torus: (2 - sqrt(x² + z²))² + y² ≤ 1 (major radius 2, minor radius 1)

Usage:
    python experiment_6_3_2_3D.py --domain sphere --epochs 10
    python experiment_6_3_2_3D.py --domain cylinder --epochs 10
    python experiment_6_3_2_3D.py --domain torus --epochs 10
    python experiment_6_3_2_3D.py --domain all --epochs 10
    python experiment_6_3_2_3D.py --domain all --epochs 10 --no-exact-inf
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
from mpl_toolkits.mplot3d import Axes3D

# Optional: scipy for loading .mat files
try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Domain configurations
DOMAIN_CONFIGS = {
    'sphere': {
        'name': '3D Sphere',
        'equation': 'x² + y² + z² ≤ 1',
        'is_rectangle': 0.0,
        'x_min': -1, 'x_max': 1,
        'y_min': -1, 'y_max': 1,
        'z_min': -1, 'z_max': 1,
        'normalize_factor': 2.0,
        'trunk_layers': [3, 512, 512, 512, 128],
        'data_subdir': 'Distance to boundary/3D examples/Sphere',
        'mat_pattern': 'p{}sphere.mat',
        'max_training_p': 200,
    },
    'cylinder': {
        'name': '3D Cylinder',
        'equation': 'y² + z² ≤ 1, -1 ≤ x ≤ 1',
        'is_rectangle': 0.1,
        'x_min': -1, 'x_max': 1,
        'y_min': -1, 'y_max': 1,
        'z_min': -1, 'z_max': 1,
        'normalize_factor': 2.0,
        'trunk_layers': [3, 512, 512, 512, 128],
        'data_subdir': 'Distance to boundary/3D examples/Cylinder',
        'mat_pattern': 'p{}cylinder.mat',
        'max_training_p': 105,
    },
    'torus': {
        'name': '3D Torus',
        'equation': '(2 - √(x² + z²))² + y² ≤ 1',
        'is_rectangle': 0.2,
        'x_min': -3, 'x_max': 3,
        'y_min': -1, 'y_max': 1,
        'z_min': -3, 'z_max': 3,
        'normalize_factor': 6.0,
        'trunk_layers': [3, 512, 512, 512, 128],
        'data_subdir': 'Distance to boundary/3D examples/Torus',
        'mat_pattern': 'p{}torus.mat',
        'max_training_p': 165,
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


def distance_to_sphere_boundary(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Exact solution for distance to boundary of unit sphere."""
    return 1.0 - torch.sqrt(x**2 + y**2 + z**2)


def distance_to_cylinder_boundary(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Exact solution for distance to boundary of cylinder.
    Cylinder: y² + z² ≤ 1, -1 ≤ x ≤ 1
    """
    dist_to_caps = 1.0 - torch.abs(x)
    dist_to_lateral = 1.0 - torch.sqrt(y**2 + z**2)
    return torch.minimum(dist_to_caps, dist_to_lateral)


def distance_to_torus_boundary(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Exact solution for distance to boundary of torus.
    Torus: (R - sqrt(x² + z²))² + y² ≤ r² where R=2, r=1
    Distance to centerline of tube, then subtract tube radius.
    """
    R = 2.0  # Major radius
    r = 1.0  # Minor radius
    r1 = torch.sqrt(x**2 + z**2)
    r1 = torch.clamp(r1, min=1e-6)  # Avoid division by zero
    # Distance from point to nearest point on tube centerline
    center_x = x / r1 * R
    center_z = z / r1 * R
    dist_to_center = torch.sqrt((x - center_x)**2 + y**2 + (z - center_z)**2)
    return r - dist_to_center


def exact_solution_3d(
    x: torch.Tensor, 
    y: torch.Tensor, 
    z: torch.Tensor,
    domain: str
) -> torch.Tensor:
    """Compute exact solution (distance to boundary) for the given 3D domain."""
    if domain == 'sphere':
        return distance_to_sphere_boundary(x, y, z)
    elif domain == 'cylinder':
        return distance_to_cylinder_boundary(x, y, z)
    elif domain == 'torus':
        return distance_to_torus_boundary(x, y, z)
    else:
        raise ValueError(f"Unknown domain: {domain}")


class DeepONetDataset(Dataset):
    """Dataset for DeepONet training."""
    
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DeepONet3D(nn.Module):
    """
    DeepONet architecture for learning the 3D p-Laplacian operator.
    
    Architecture (matching Section 6.3.2 notebooks):
        - Trunk net: Takes 3D spatial coordinates (x, y, z) as input
          layers: [3, 512, 512, 512, 128] - three hidden layers with 512 neurons
        - Branch net: Takes p value (normalized by 500) as input
          layers: [1, 128, 128, 128]
        - Output: Element-wise product of trunk and branch outputs (128-dim),
          followed by final linear layer (128 → 1)
    """
    
    def __init__(
        self,
        trunk_layers: list = [3, 512, 512, 512, 128],
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
        
        # Trunk network (3D spatial coordinates)
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
        
        # Final combination layer
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
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 4) where columns are (x, y, z, p_normalized)
        
        Returns:
            Output tensor of shape (N, 1)
        """
        # Split input into spatial coordinates and p value
        a = x[:, 0:3]  # (x, y, z) - trunk input
        b = x[:, 3:4]  # p/500 - branch input
        
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


def is_inside_domain_3d(
    x: torch.Tensor, 
    y: torch.Tensor, 
    z: torch.Tensor, 
    domain: str, 
    config: dict
) -> torch.Tensor:
    """Check if 3D points are inside the domain."""
    if domain == 'sphere':
        return x**2 + y**2 + z**2 <= 1
    elif domain == 'cylinder':
        return (y**2 + z**2 <= 1) & (x >= -1) & (x <= 1)
    elif domain == 'torus':
        R = 2.0  # Major radius
        r = 1.0  # Minor radius
        r1 = torch.sqrt(x**2 + z**2)
        return (R - r1)**2 + y**2 <= r**2 + 0.00001
    else:
        return x**2 + y**2 + z**2 <= 1


def generate_domain_points_3d(
    domain: str, 
    config: dict, 
    n_points: int, 
    normalize: bool = True
) -> np.ndarray:
    """Generate random points inside the 3D domain."""
    x_min, x_max = config['x_min'], config['x_max']
    y_min, y_max = config['y_min'], config['y_max']
    z_min, z_max = config['z_min'], config['z_max']
    
    points = []
    while len(points) < n_points:
        x = np.random.uniform(x_min, x_max, n_points * 3)
        y = np.random.uniform(y_min, y_max, n_points * 3)
        z = np.random.uniform(z_min, z_max, n_points * 3)
        
        x_t = torch.tensor(x)
        y_t = torch.tensor(y)
        z_t = torch.tensor(z)
        
        mask = is_inside_domain_3d(x_t, y_t, z_t, domain, config).numpy()
        valid_points = np.column_stack([x[mask], y[mask], z[mask]])
        points.extend(valid_points.tolist())
    
    points = np.array(points[:n_points])
    
    if normalize:
        points = points / config['normalize_factor'] + 0.5
    
    return points


def generate_test_grid_3d(
    domain: str,
    config: dict,
    n_points: int = 51,
    normalize: bool = True
) -> tuple:
    """Generate a test grid for 3D evaluation."""
    x_min, x_max = config['x_min'], config['x_max']
    y_min, y_max = config['y_min'], config['y_max']
    z_min, z_max = config['z_min'], config['z_max']
    
    x = torch.linspace(x_min, x_max, n_points)
    y = torch.linspace(y_min, y_max, n_points)
    z = torch.linspace(z_min, z_max, n_points)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    
    # Filter points inside domain
    mask = is_inside_domain_3d(X_flat, Y_flat, Z_flat, domain, config)
    X_flat = X_flat[mask]
    Y_flat = Y_flat[mask]
    Z_flat = Z_flat[mask]
    
    # Compute exact solution
    y_exact = exact_solution_3d(X_flat, Y_flat, Z_flat, domain)
    
    # Create test points
    x_test = torch.stack([X_flat, Y_flat, Z_flat], dim=1)
    y_test = y_exact.unsqueeze(1)
    
    if normalize:
        x_test = x_test / config['normalize_factor'] + 0.5
    
    return x_test, y_test


def load_mat_data_3d(
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
    norm_factor = config['normalize_factor']
    
    X_train_list = []
    Y_train_list = []
    last_pts = None
    
    # Try to load data for various p values
    for p in range(5, max_p + 1, 5):
        mat_file = os.path.join(mat_path, mat_pattern.format(p))
        if os.path.exists(mat_file):
            try:
                mat = scipy.io.loadmat(mat_file)
                pts = mat['pts']
                
                # Subsample for torus (very large dataset)
                if domain == 'torus':
                    pts = pts[::10, ...]
                    solution = mat['solution'][:, ::10].flatten().reshape(-1, 1)
                else:
                    solution = mat['solution'].flatten().reshape(-1, 1)
                
                pts_norm = pts / norm_factor + 0.5
                p_value = np.ones((pts_norm.shape[0], 1)) * p / p_normalize
                
                X = np.hstack([pts_norm, p_value])
                Y = solution
                
                if X.shape[0] == Y.shape[0]:
                    X_train_list.append(X)
                    Y_train_list.append(Y)
                    last_pts = pts
                    print(f"  Loaded p={p}: {X.shape[0]} points")
            except Exception as e:
                print(f"  Warning: Could not load {mat_file}: {e}")
    
    if not X_train_list:
        return None, None
    
    # Add exact solution at p=500 (surrogate for p=∞)
    if include_exact_inf and last_pts is not None:
        pts_norm = last_pts / norm_factor + 0.5
        p_value = np.ones((pts_norm.shape[0], 1)) * 500.0 / p_normalize
        X_inf = np.hstack([pts_norm, p_value])
        Y_inf = exact_solution_3d(
            torch.tensor(last_pts[:, 0]),
            torch.tensor(last_pts[:, 1]),
            torch.tensor(last_pts[:, 2]),
            domain
        ).numpy().reshape(-1, 1)
        
        X_train_list.append(X_inf)
        Y_train_list.append(Y_inf)
        print(f"  Added exact p=∞ solution: {X_inf.shape[0]} points")
    
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)
    
    return X_train, Y_train


def generate_synthetic_data_3d(
    domain: str,
    config: dict,
    p_values: list = None,
    n_points_per_p: int = 10000,
    p_normalize: float = 500.0,
    include_exact_inf: bool = True
) -> tuple:
    """Generate synthetic training data for 3D domains."""
    if p_values is None:
        max_p = config['max_training_p']
        p_values = list(range(5, min(max_p + 1, 106), 5))
    
    X_train_list = []
    Y_train_list = []
    
    for p in p_values:
        pts = generate_domain_points_3d(domain, config, n_points_per_p, normalize=False)
        
        # Exact solution at infinity
        u_inf = exact_solution_3d(
            torch.tensor(pts[:, 0]),
            torch.tensor(pts[:, 1]),
            torch.tensor(pts[:, 2]),
            domain
        ).numpy()
        
        # Approximate u_p based on 1/p convergence
        C = 0.1
        u_p = u_inf - C / p
        u_p = np.maximum(u_p, 0)
        
        # Normalize coordinates
        pts_norm = pts / config['normalize_factor'] + 0.5
        p_value = np.ones((pts_norm.shape[0], 1)) * p / p_normalize
        
        X = np.hstack([pts_norm, p_value])
        Y = u_p.reshape(-1, 1)
        
        X_train_list.append(X)
        Y_train_list.append(Y)
        print(f"  Generated synthetic data for p={p}: {X.shape[0]} points")
    
    # Add exact solution at p=500
    if include_exact_inf:
        pts = generate_domain_points_3d(domain, config, n_points_per_p, normalize=False)
        pts_norm = pts / config['normalize_factor'] + 0.5
        p_value = np.ones((pts_norm.shape[0], 1)) * 500.0 / p_normalize
        u_inf = exact_solution_3d(
            torch.tensor(pts[:, 0]),
            torch.tensor(pts[:, 1]),
            torch.tensor(pts[:, 2]),
            domain
        ).numpy()
        
        X_inf = np.hstack([pts_norm, p_value])
        Y_inf = u_inf.reshape(-1, 1)
        
        X_train_list.append(X_inf)
        Y_train_list.append(Y_inf)
        print(f"  Added exact solution at p=500 (∞): {X_inf.shape[0]} points")
    
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)
    
    return X_train, Y_train


def train_deeponet_3d(
    model: DeepONet3D,
    train_loader: DataLoader,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    n_epochs: int,
    lr: float,
    device: torch.device,
    show_every: int = 1
) -> dict:
    """Train the 3D DeepONet model."""
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    
    loss_history = {
        'train_loss': [],
        'test_loss_p150': [],
        'test_loss_p200': [],
        'test_loss_p500': []
    }
    
    print("Training Loss ----- Test Loss (p=150) ----- Test Loss (p=200) ----- Test Loss (p=500)")
    
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
            test_x_p150 = test_x.clone()
            test_x_p150[:, 3] = 150 / 500.0
            test_loss_p150 = model.compute_loss(test_x_p150.to(device), test_y.to(device))
            
            test_x_p200 = test_x.clone()
            test_x_p200[:, 3] = 200 / 500.0
            test_loss_p200 = model.compute_loss(test_x_p200.to(device), test_y.to(device))
            
            test_x_p500 = test_x.clone()
            test_x_p500[:, 3] = 500 / 500.0
            test_loss_p500 = model.compute_loss(test_x_p500.to(device), test_y.to(device))
        
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['test_loss_p150'].append(test_loss_p150.item())
        loss_history['test_loss_p200'].append(test_loss_p200.item())
        loss_history['test_loss_p500'].append(test_loss_p500.item())
        
        if epoch % show_every == 0:
            print(f"{epoch} --- {avg_train_loss:.6e} --- {test_loss_p150.item():.6e} --- "
                  f"{test_loss_p200.item():.6e} --- {test_loss_p500.item():.6e}")
    
    return loss_history


def evaluate_over_p_range_3d(
    model: DeepONet3D,
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
            test_x_p[:, 3] = p / 500.0
            test_loss = model.compute_loss(test_x_p.to(device), test_y.to(device))
            
            results['p'].append(p)
            results['mse'].append(test_loss.item())
    
    return results


def plot_results_3d(
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
    ax.semilogy(epochs, loss_history['test_loss_p150'], label='Test Loss (p=150)')
    ax.semilogy(epochs, loss_history['test_loss_p200'], label='Test Loss (p=200)')
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


def run_experiment_3d(
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
    """Run the full 3D DeepONet experiment for a given domain."""
    config = DOMAIN_CONFIGS[domain]
    trunk_layers = config['trunk_layers']
    
    print(f"\n{'='*60}")
    print(f"Running Experiment 6.3.2: Distance to Boundary 3D - {config['name']}")
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
        X_train, Y_train = generate_synthetic_data_3d(
            domain, config, include_exact_inf=include_exact_inf
        )
    else:
        X_train, Y_train = load_mat_data_3d(
            data_dir, domain, config, include_exact_inf=include_exact_inf
        )
        if X_train is None:
            print(f"  No .mat files found in {data_dir}/{config['data_subdir']}/")
            print("  Generating synthetic training data instead...")
            X_train, Y_train = generate_synthetic_data_3d(
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
    
    # Generate test data (use smaller grid for 3D)
    print("\nGenerating test grid...")
    test_x, test_y = generate_test_grid_3d(domain, config, n_points=51, normalize=True)
    test_x = torch.cat([test_x, torch.zeros(test_x.shape[0], 1)], dim=1)
    print(f"Test data shape: X={test_x.shape}, Y={test_y.shape}")
    
    # Create model
    print("\nCreating 3D DeepONet model...")
    print(f"  Trunk layers: {trunk_layers}")
    print(f"  Branch layers: {branch_layers}")
    model = DeepONet3D(
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
    loss_history = train_deeponet_3d(
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
    eval_results = evaluate_over_p_range_3d(model, test_x, test_y, device)
    
    # Print results
    print("\nResults: MSE vs p")
    print("-" * 30)
    for i, (p, mse) in enumerate(zip(eval_results['p'], eval_results['mse'])):
        if i % 10 == 0:
            print(f"p={p:3d}: MSE={mse:.6e}")
    
    # Plot results
    domain_output_dir = os.path.join(output_dir, domain)
    plot_results_3d(loss_history, eval_results, domain, config, domain_output_dir)
    
    # Save model
    suffix = '' if include_exact_inf else '_no_inf'
    model_path = os.path.join(domain_output_dir, f'deeponet_3d_{domain}{suffix}.pt')
    os.makedirs(domain_output_dir, exist_ok=True)
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
        description='Experiment 6.3.2: Distance to Boundary DeepONet 3D'
    )
    parser.add_argument(
        '--domain', type=str, default='sphere',
        choices=['sphere', 'cylinder', 'torus', 'all'],
        help='Domain type: sphere, cylinder, torus, or all'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
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
        '--data-dir', type=str, default='./experiment_6_3_2_3D',
        help='Directory containing training data (.mat files)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./outputs/experiment_6_3_2_3D',
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
        domains = ['sphere', 'cylinder', 'torus']
    else:
        domains = [args.domain]
    
    for domain in domains:
        run_experiment_3d(
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
