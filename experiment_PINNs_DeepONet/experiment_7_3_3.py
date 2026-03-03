#!/usr/bin/env python3
"""
Experiment 6.3.3: Distance to Boundary DeepONet for All 2D Ellipses

This script implements the DeepONet approach for learning the p-Laplacian operator
for the distance to boundary problem on ALL 2D ellipses (parameterized by rotation
angle theta and semi-axes a, b), as described in Section 6.3.3 of the paper
"Solving p-Laplacian and infinity-Laplacian with Deep Learning".

The BVP is:
    Δ_p u_p = -1, in Ω
    u_p = 0, on ∂Ω

The limiting behavior as p → ∞ gives:
    u_∞ = dist(x, ∂Ω)

The DeepONet learns over a family of ellipses parameterized by:
    - theta: rotation angle ∈ [0, π/2]
    - a: semi-major axis parameter ∈ [0, 1] (mapped to ellipse semi-axis)
    - b: semi-minor axis parameter ∈ [0, 1] (mapped to ellipse semi-axis)
    - p: p-Laplacian parameter ∈ [5, 500]

Architecture:
    - Trunk: [2, 512, 512, 128] - spatial coordinates (x, y)
    - Branch: [4, 128, 128, 128] - parameters (p/500, theta/(π/2), a_param, b_param)

Usage:
    python experiment_6_3_3.py --epochs 20 --total-div 10
    python experiment_6_3_3.py --epochs 20 --total-div 10 --no-exact-inf
    python experiment_6_3_3.py --epochs 20 --total-div 2  # Smaller dataset
"""

import argparse
import os
import random
import glob
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Optional: scipy for loading .mat files
try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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
    Solve quartic equation ax⁴ + bx³ + cx² + dx + e = 0 using Ferrari's method.
    Returns all 4 roots stacked as tensor of shape (4, N).
    """
    p = (8 * a_coef * c_coef - 3 * b_coef**2) / (8 * a_coef**2)
    q = (b_coef**3 - 4 * a_coef * b_coef * c_coef + 8 * a_coef**2 * d_coef) / (8 * a_coef**3)
    delta0 = c_coef**2 - 3 * b_coef * d_coef + 12 * a_coef * e_coef
    delta1 = (2 * c_coef**3 - 9 * b_coef * c_coef * d_coef + 27 * b_coef**2 * e_coef 
              + 27 * a_coef * d_coef**2 - 72 * a_coef * c_coef * e_coef)
    
    # Handle near-zero delta0
    delta0_mask = (torch.abs(delta0) < 0.001).float()
    
    discriminant = (delta1**2 - 4 * delta0**3).to(torch.complex64)
    Q_inner = delta1 + ((1 - delta0_mask) * torch.sqrt(discriminant) + delta0_mask * delta1).to(torch.complex64)
    Q = torch.pow(Q_inner / 2, 1/3)
    
    S = torch.sqrt(-2/3 * p + (Q + delta0 / (Q + 1e-10)) / (3 * a_coef)) / 2
    
    sol_list = []
    sol_list.append(torch.real(-b_coef / (4 * a_coef) - S - torch.sqrt(-4 * S**2 - 2 * p + q / (S + 1e-10)) / 2))
    sol_list.append(torch.real(-b_coef / (4 * a_coef) - S + torch.sqrt(-4 * S**2 - 2 * p + q / (S + 1e-10)) / 2))
    sol_list.append(torch.real(-b_coef / (4 * a_coef) + S - torch.sqrt(-4 * S**2 - 2 * p - q / (S + 1e-10)) / 2))
    sol_list.append(torch.real(-b_coef / (4 * a_coef) + S + torch.sqrt(-4 * S**2 - 2 * p - q / (S + 1e-10)) / 2))
    
    return torch.vstack(sol_list)


def distance_to_ellipse_boundary(x: torch.Tensor, y: torch.Tensor, a: float, b: float, EPS: float = 1e-6):
    """
    Compute exact distance from point (x, y) to ellipse boundary x²/a² + y²/b² = 1.
    Uses quartic equation solver (Ferrari's method).
    
    Args:
        x, y: Coordinates of points (can be tensors)
        a: Semi-axis in x direction
        b: Semi-axis in y direction
        EPS: Small epsilon for numerical stability
    
    Returns:
        Distance to boundary for each point
    """
    # Coefficients of quartic equation for distance to ellipse
    aa = torch.ones_like(x)
    bb = torch.ones_like(x) * (-2 * a**2 - 2 * b**2)
    cc = a**4 + b**4 + 4 * a**2 * b**2 - a**2 * x**2 - b**2 * y**2
    dd = -2 * a**2 * b**4 - 2 * a**4 * b**2 + 2 * a**2 * b**2 * x**2 + 2 * a**2 * b**2 * y**2
    ee = a**4 * b**4 - a**2 * b**4 * x**2 - a**4 * b**2 * y**2
    
    sol = solve_4th_order(aa, bb, cc, dd, ee)
    
    # Compute boundary points for each root
    x_rep = x.repeat(4, 1)
    y_rep = y.repeat(4, 1)
    
    # Point on boundary: (a²x/(a²-λ), b²y/(b²-λ))
    mask_a = (torch.abs(a**2 - sol) > EPS).float()
    mask_b = (torch.abs(b**2 - sol) > EPS).float()
    
    point_bound_X = (mask_a * a**2 * x_rep / (a**2 - sol + EPS) + 
                     (1 - mask_a) * a * ((x_rep > 0).float() * 2 - 1))
    point_bound_X *= (torch.abs(x_rep) >= 0.0001).float()
    point_bound_X += ((torch.abs(x_rep) < 0.0001) * (torch.abs(y_rep) < 0.1)).float() * a * ((x_rep > 0).float() * 2 - 1)
    
    point_bound_Y = (mask_b * b**2 * y_rep / (b**2 - sol + EPS) + 
                     (1 - mask_b) * b * ((y_rep > 0).float() * 2 - 1))
    point_bound_Y *= (torch.abs(y_rep) >= 0.0001).float()
    point_bound_Y += ((torch.abs(y_rep) < 0.0001) * (torch.abs(x_rep) < 0.1)).float() * b * ((y_rep > 0).float() * 2 - 1)
    
    # Compute distance to each candidate boundary point
    dist = torch.sqrt((point_bound_X - x_rep)**2 + (point_bound_Y - y_rep)**2)
    dist = torch.nan_to_num(dist, nan=1e6)
    dist = torch.min(dist, dim=0)[0]
    
    # Handle special cases
    dist[dist == 1e6] = float(min(a, b))
    dist[x**2 + y**2 < EPS] = float(min(a, b))
    
    return dist


def exact_solution_ellipse(
    x: torch.Tensor, 
    y: torch.Tensor, 
    a_param: float, 
    b_param: float,
    theta: float = 0.0
) -> torch.Tensor:
    """
    Compute exact solution (distance to boundary) for rotated ellipse.
    
    Args:
        x, y: Coordinates
        a_param: Parameter in [0, 1] mapped to semi-axis a = a_param * 0.2 + 0.9
        b_param: Parameter in [0, 1] mapped to semi-axis b = b_param * 0.1 + 0.2
        theta: Rotation angle in radians
    
    Returns:
        Distance to boundary
    """
    # Map parameters to actual semi-axes (matching notebook convention)
    a = a_param * 0.2 + 0.9  # a ∈ [0.9, 1.1]
    b = b_param * 0.1 + 0.2  # b ∈ [0.2, 0.3]
    
    # Rotate coordinates to aligned ellipse frame
    x_rot = np.cos(theta) * x - np.sin(theta) * y
    y_rot = np.sin(theta) * x + np.cos(theta) * y
    
    return distance_to_ellipse_boundary(x_rot, y_rot, 1/a, 1/b)


class DeepONetDataset(Dataset):
    """Dataset for DeepONet training."""
    
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DeepONetAllEllipse(nn.Module):
    """
    DeepONet architecture for learning the p-Laplacian operator over ALL ellipses.
    
    Architecture (matching Section 6.3.3 notebooks):
        - Trunk net: Takes 2D spatial coordinates (x, y) as input
          layers: [2, 512, 512, 128]
        - Branch net: Takes 4 parameters (p/500, theta/(π/2), a_param, b_param)
          layers: [4, 128, 128, 128]
        - Output: Element-wise product of trunk and branch outputs (128-dim),
          followed by final linear layer (128 → 1)
    """
    
    def __init__(
        self,
        trunk_layers: list = [2, 512, 512, 128],
        branch_layers: list = [4, 128, 128, 128],
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
        
        # Trunk network (2D spatial coordinates)
        self.trunk_layers = trunk_layers
        self.trunk_linears = nn.ModuleList([
            nn.Linear(trunk_layers[i], trunk_layers[i+1])
            for i in range(len(trunk_layers) - 1)
        ])
        
        # Branch network (4 parameters: p, theta, a, b)
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
            x: Input tensor of shape (N, 6) where columns are:
               (x_coord, y_coord, p_norm, theta_norm, a_param, b_param)
        
        Returns:
            Output tensor of shape (N, 1)
        """
        # Split input into spatial coordinates and parameters
        spatial = x[:, 0:2]  # (x, y) - trunk input
        params = x[:, 2:6]   # (p/500, theta/(π/2), a_param, b_param) - branch input
        
        # Trunk network
        a = spatial
        for i in range(len(self.trunk_layers) - 2):
            z = self.trunk_linears[i](a)
            a = self.activation(z)
        a = self.trunk_linears[-1](a)  # Last layer, no activation
        
        # Branch network
        b = params
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


def generate_training_data(
    mat_path: str,
    total_div: int = 10,
    include_exact_inf: bool = True,
    p_normalize: float = 500.0
) -> tuple:
    """
    Generate training data for all ellipses experiment.
    
    The data generation follows the notebook pattern:
    1. Load base points from .mat file at p=500 (or highest available)
    2. Generate exact solution at p=∞ for base ellipse (a=0.5, b=0.5 params)
    3. Rotate base points by theta_i for various angles
    4. Vary (a_param, b_param) to create different ellipse shapes
    
    Args:
        mat_path: Path to directory containing .mat files
        total_div: Number of divisions for theta, a, b discretization
        include_exact_inf: Whether to include exact p=∞ solutions
        p_normalize: Normalization factor for p values
    
    Returns:
        X_train, Y_train: Training data arrays
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required to load .mat files")
    
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Data directory not found: {mat_path}")
    
    X_train_list = []
    Y_train_list = []
    
    # Load .mat files to get point locations
    mat_files = glob.glob(os.path.join(mat_path, '*.mat'))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {mat_path}")
    
    # Find highest p value file for reference points
    max_p = 0
    ref_mat = None
    for mf in mat_files:
        match = re.search(r'p\s*(\d+)', os.path.basename(mf))
        if match:
            p = int(match.group(1))
            if p > max_p:
                max_p = p
                ref_mat = mf
    
    if ref_mat is None:
        raise ValueError("Could not find reference .mat file")
    
    print(f"Using reference file: {ref_mat} (p={max_p})")
    mat = scipy.io.loadmat(ref_mat)
    pts = mat['pts']
    store_y = mat['solution'][0].reshape(-1, 1)
    
    # Base parameters (matching notebook)
    base_a_param = 0.5
    base_b_param = 0.5
    
    # Compute exact solution at p=∞ for base ellipse
    x_pts = torch.tensor(pts[:, 0])
    y_pts = torch.tensor(pts[:, 1])
    y_exact = exact_solution_ellipse(x_pts, y_pts, base_a_param, base_b_param, theta=0.0).numpy().reshape(-1, 1)
    
    # Filter points where FEM solution matches exact well
    index_valid = np.abs(y_exact[:, 0] - store_y[:, 0]) < 0.01
    
    # Normalize coordinates
    pts_norm = pts / 2.0 + 0.5
    
    # Create base training data at theta=0, base a, b
    if include_exact_inf:
        p_value = np.ones((pts.shape[0], 1)) * 500.0 / p_normalize
        theta_value = np.zeros((pts.shape[0], 1))
        a_value = np.ones((pts.shape[0], 1)) * base_a_param
        b_value = np.ones((pts.shape[0], 1)) * base_b_param
        
        X_base = np.hstack([pts_norm, p_value, theta_value, a_value, b_value])
        Y_base = y_exact
        
        # Filter and subsample
        X_base = X_base[index_valid][::10]
        Y_base = Y_base[index_valid][::10]
        
        X_train_list.append(X_base)
        Y_train_list.append(Y_base)
        print(f"  Added base ellipse (theta=0, a={base_a_param}, b={base_b_param}): {X_base.shape[0]} points")
        
        # Add rotations of base ellipse
        for theta_i in range(total_div):
            theta = (theta_i + 1) * np.pi / 2 / total_div
            
            # Rotate normalized points back to original, rotate, re-normalize
            x_orig = (pts_norm[:, 0] - 0.5) * 2.0
            y_orig = (pts_norm[:, 1] - 0.5) * 2.0
            x_rot = np.cos(theta) * x_orig - np.sin(theta) * y_orig
            y_rot = np.sin(theta) * x_orig + np.cos(theta) * y_orig
            x_rot_norm = x_rot / 2.0 + 0.5
            y_rot_norm = y_rot / 2.0 + 0.5
            
            theta_value = np.ones((pts.shape[0], 1)) * theta / (np.pi / 2)  # Normalize to [0, 1]
            
            X_rot = np.column_stack([
                x_rot_norm, y_rot_norm,
                np.ones(pts.shape[0]) * 500.0 / p_normalize,
                theta_value.flatten(),
                np.ones(pts.shape[0]) * base_a_param,
                np.ones(pts.shape[0]) * base_b_param
            ])
            
            X_rot = X_rot[index_valid][::10]
            Y_rot = Y_base.copy()
            
            X_train_list.append(X_rot)
            Y_train_list.append(Y_rot)
        
        print(f"  Added {total_div} rotations of base ellipse")
    
    # Add variations in (a_param, b_param)
    pts_filtered = pts[index_valid][::10]
    pts_norm_filtered = pts_norm[index_valid][::10]
    n_pts = pts_filtered.shape[0]
    
    for a_i in range(total_div):
        print(f"  Processing a_i: {a_i}/{total_div}")
        for b_i in range(total_div):
            # Compute a_param, b_param following notebook logic
            if a_i < total_div // 2:
                a_param = 0.5 - (a_i + 1) / total_div * 2 * 0.5
            else:
                a_param = 0.5 + (a_i - total_div // 2 + 1) / total_div * 2 * 0.5
            
            if b_i < total_div // 2:
                b_param = 0.5 - (b_i + 1) / total_div * 2 * 0.5
            else:
                b_param = 0.5 + (b_i - total_div // 2 + 1) / total_div * 2 * 0.5
            
            # Compute exact solution for this ellipse
            if include_exact_inf:
                y_ab = exact_solution_ellipse(
                    torch.tensor(pts_filtered[:, 0]),
                    torch.tensor(pts_filtered[:, 1]),
                    a_param, b_param, theta=0.0
                ).numpy().reshape(-1, 1)
                
                # Base (theta=0)
                X_ab = np.column_stack([
                    pts_norm_filtered,
                    np.ones(n_pts) * 500.0 / p_normalize,
                    np.zeros(n_pts),
                    np.ones(n_pts) * a_param,
                    np.ones(n_pts) * b_param
                ])
                
                X_train_list.append(X_ab)
                Y_train_list.append(y_ab)
                
                # Add rotations
                for theta_i in range(total_div):
                    theta = (theta_i + 1) * np.pi / 2 / total_div
                    
                    x_orig = (pts_norm_filtered[:, 0] - 0.5) * 2.0
                    y_orig = (pts_norm_filtered[:, 1] - 0.5) * 2.0
                    x_rot = np.cos(theta) * x_orig - np.sin(theta) * y_orig
                    y_rot = np.sin(theta) * x_orig + np.cos(theta) * y_orig
                    x_rot_norm = x_rot / 2.0 + 0.5
                    y_rot_norm = y_rot / 2.0 + 0.5
                    
                    X_rot = np.column_stack([
                        x_rot_norm, y_rot_norm,
                        np.ones(n_pts) * 500.0 / p_normalize,
                        np.ones(n_pts) * theta / (np.pi / 2),
                        np.ones(n_pts) * a_param,
                        np.ones(n_pts) * b_param
                    ])
                    
                    X_train_list.append(X_rot)
                    Y_train_list.append(y_ab.copy())
    
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)
    
    return X_train, Y_train


def generate_synthetic_data(
    total_div: int = 10,
    n_points: int = 5000,
    include_exact_inf: bool = True,
    p_normalize: float = 500.0
) -> tuple:
    """Generate synthetic training data when .mat files are not available."""
    X_train_list = []
    Y_train_list = []
    
    # Generate random points in [-1, 1]²
    np.random.seed(1234)
    
    for a_i in range(total_div):
        print(f"  Generating synthetic data: a_i={a_i}/{total_div}")
        for b_i in range(total_div):
            # Compute a_param, b_param
            if a_i < total_div // 2:
                a_param = 0.5 - (a_i + 1) / total_div * 2 * 0.5
            else:
                a_param = 0.5 + (a_i - total_div // 2 + 1) / total_div * 2 * 0.5
            
            if b_i < total_div // 2:
                b_param = 0.5 - (b_i + 1) / total_div * 2 * 0.5
            else:
                b_param = 0.5 + (b_i - total_div // 2 + 1) / total_div * 2 * 0.5
            
            # Map to actual semi-axes
            a = a_param * 0.2 + 0.9
            b = b_param * 0.1 + 0.2
            
            # Generate points inside ellipse
            pts_list = []
            while len(pts_list) < n_points // (total_div * total_div):
                x = np.random.uniform(-1, 1, n_points)
                y = np.random.uniform(-1, 1, n_points)
                inside = (x / (1/a))**2 + (y / (1/b))**2 < 1
                pts_list.extend(zip(x[inside], y[inside]))
            
            pts = np.array(pts_list[:n_points // (total_div * total_div)])
            pts_norm = pts / 2.0 + 0.5
            n_pts = pts.shape[0]
            
            if include_exact_inf:
                # Compute exact solution
                y_exact = exact_solution_ellipse(
                    torch.tensor(pts[:, 0]),
                    torch.tensor(pts[:, 1]),
                    a_param, b_param, theta=0.0
                ).numpy().reshape(-1, 1)
                
                # Add rotations
                for theta_i in range(total_div + 1):
                    theta = theta_i * np.pi / 2 / total_div
                    
                    x_rot = np.cos(theta) * pts[:, 0] - np.sin(theta) * pts[:, 1]
                    y_rot = np.sin(theta) * pts[:, 0] + np.cos(theta) * pts[:, 1]
                    x_rot_norm = x_rot / 2.0 + 0.5
                    y_rot_norm = y_rot / 2.0 + 0.5
                    
                    X = np.column_stack([
                        x_rot_norm, y_rot_norm,
                        np.ones(n_pts) * 500.0 / p_normalize,
                        np.ones(n_pts) * theta / (np.pi / 2),
                        np.ones(n_pts) * a_param,
                        np.ones(n_pts) * b_param
                    ])
                    
                    X_train_list.append(X)
                    Y_train_list.append(y_exact.copy())
    
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)
    
    return X_train, Y_train


def train_deeponet(
    model: DeepONetAllEllipse,
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
        'test_loss': []
    }
    
    print("Training Loss ----- Test Loss")
    
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
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_loss = model.compute_loss(test_x.to(device), test_y.to(device))
        
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['test_loss'].append(test_loss.item())
        
        if epoch % show_every == 0:
            print(f"{epoch} --- {avg_train_loss:.6e} --- {test_loss.item():.6e}")
    
    return loss_history


def evaluate_model(
    model: DeepONetAllEllipse,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    p_range: list = None
) -> dict:
    """Evaluate model over different p values."""
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
    total_div: int,
    output_dir: str
):
    """Plot training history and evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Training loss history
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(len(loss_history['train_loss']))
    ax.semilogy(epochs, loss_history['train_loss'], label='Train Loss')
    ax.semilogy(epochs, loss_history['test_loss'], label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Training History - All Ellipses (total_div={total_div})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_history_total{total_div}.png'), dpi=150)
    plt.close()
    
    # Plot 2: MSE over p range
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(eval_results['p'], eval_results['mse'], 'b-o', markersize=3)
    ax.set_xlabel('p')
    ax.set_ylabel('MSE (vs exact solution at p=∞)')
    ax.set_title(f'DeepONet MSE vs p - All Ellipses (total_div={total_div})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mse_vs_p_total{total_div}.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


def run_experiment(
    epochs: int,
    lr: float,
    batch_size: int,
    trunk_layers: list,
    branch_layers: list,
    total_div: int,
    data_dir: str,
    output_dir: str,
    seed: int,
    use_synthetic: bool = False,
    include_exact_inf: bool = True
):
    """Run the full experiment."""
    print(f"\n{'='*60}")
    print(f"Running Experiment 6.3.3: Distance to Boundary - All 2D Ellipses")
    print(f"total_div = {total_div}")
    print(f"{'='*60}")
    
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load or generate training data
    print("\nGenerating training data...")
    print(f"  Include exact p=∞ solution: {include_exact_inf}")
    print(f"  total_div = {total_div}")
    
    if use_synthetic:
        X_train, Y_train = generate_synthetic_data(
            total_div=total_div,
            include_exact_inf=include_exact_inf
        )
    else:
        try:
            mat_path = os.path.join(data_dir, 'Distance to boundary/2D examples/Ellipse 1')
            X_train, Y_train = generate_training_data(
                mat_path=mat_path,
                total_div=total_div,
                include_exact_inf=include_exact_inf
            )
        except (FileNotFoundError, ImportError) as e:
            print(f"  Warning: {e}")
            print("  Generating synthetic training data instead...")
            X_train, Y_train = generate_synthetic_data(
                total_div=total_div,
                include_exact_inf=include_exact_inf
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
    
    # Create test data (subset of training for quick evaluation)
    n_test = min(5000, len(X_train) // 10)
    test_indices = np.random.choice(len(X_train), n_test, replace=False)
    test_x = torch.from_numpy(X_train[test_indices]).float()
    test_y = torch.from_numpy(Y_train[test_indices]).float()
    print(f"Test data shape: X={test_x.shape}, Y={test_y.shape}")
    
    # Create model
    print("\nCreating DeepONet model...")
    print(f"  Trunk layers: {trunk_layers}")
    print(f"  Branch layers: {branch_layers}")
    model = DeepONetAllEllipse(
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
    eval_results = evaluate_model(model, test_x, test_y, device)
    
    # Print results
    print("\nResults: MSE vs p")
    print("-" * 30)
    for i, (p, mse) in enumerate(zip(eval_results['p'], eval_results['mse'])):
        if i % 10 == 0:
            print(f"p={p:3d}: MSE={mse:.6e}")
    
    # Plot results
    plot_results(loss_history, eval_results, total_div, output_dir)
    
    # Save model
    suffix = '' if include_exact_inf else '_no_inf'
    model_path = os.path.join(output_dir, f'deeponet_all_ellipse_total{total_div}{suffix}.pt')
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'trunk_layers': trunk_layers,
        'branch_layers': branch_layers,
        'total_div': total_div,
        'loss_history': loss_history,
        'eval_results': eval_results,
        'seed': seed,
        'include_exact_inf': include_exact_inf
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, loss_history, eval_results


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 6.3.3: Distance to Boundary DeepONet for All 2D Ellipses'
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
        '--trunk-layers', type=str, default='2,512,512,128',
        help='Trunk network layers (comma-separated). Default: 2,512,512,128'
    )
    parser.add_argument(
        '--branch-layers', type=str, default='4,128,128,128',
        help='Branch network layers (comma-separated). Default: 4,128,128,128'
    )
    parser.add_argument(
        '--total-div', type=int, default=10,
        help='Number of divisions for theta, a, b discretization. Default: 10'
    )
    parser.add_argument(
        '--data-dir', type=str, default='./experiment_6_3_3',
        help='Directory containing training data (.mat files)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./outputs/experiment_6_3_3',
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
    
    # Parse layer configurations
    trunk_layers = [int(x) for x in args.trunk_layers.split(',')]
    branch_layers = [int(x) for x in args.branch_layers.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    run_experiment(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        trunk_layers=trunk_layers,
        branch_layers=branch_layers,
        total_div=args.total_div,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        use_synthetic=args.synthetic,
        include_exact_inf=not args.no_exact_inf
    )
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == '__main__':
    main()
