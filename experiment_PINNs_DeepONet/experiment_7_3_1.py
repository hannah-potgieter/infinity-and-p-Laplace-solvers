#!/usr/bin/env python3
"""
Experiment 6.3.1: Distance to Origin DeepONet Example

This script implements the DeepONet approach for learning the p-Laplacian operator
for the distance to origin problem, as described in Section 6.3.1 of the paper
"Solving p-Laplacian and infinity-Laplacian with Deep Learning".

The BVP is:
    Δ_p u_p = -1, in Ω
    u_p = 0, on Γ_1 = (0, 0)  (origin)
    ∂u_p/∂n = 0, on Γ_2 = ∂Ω \ (0, 0)

The limiting behavior as p → ∞ gives:
    u_∞ = dist(x, (0, 0)) = sqrt(x² + y²)

Domains: Disc (x² + y² ≤ 1) and Square ([-1, 1] × [-1, 1])

Usage:
    python experiment_6_3_1.py --domain disc --epochs 20
    python experiment_6_3_1.py --domain square --epochs 20
    python experiment_6_3_1.py --domain all --epochs 20
"""

import argparse
import os
import random
import glob
import re
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.mplot3d import Axes3D

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


def exact_solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Exact solution for the distance to origin problem.
    u_∞(x, y) = sqrt(x² + y²)
    """
    return torch.sqrt(x**2 + y**2)


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
    
    Architecture (matching the paper and notebooks):
        - Trunk net: Takes spatial coordinates (x, y) as input
          layers: [2, 256, 256, 1] - two hidden layers with 256 neurons
        - Branch net: Takes p value (normalized by 500) as input
          layers: [1, 128, 128, 1] - two hidden layers with 128 neurons
        - Output: Element-wise product of trunk and branch outputs, 
          followed by final linear layer (1 → 1)
    
    Based on the paper's Section 5 and notebook implementation.
    """
    
    def __init__(
        self,
        trunk_layers: list = [2, 256, 256, 1],
        branch_layers: list = [1, 128, 128, 1],
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
        # Architecture: input → hidden layers with activation → output (no activation on last)
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
        # Output dimensions should match for element-wise product
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
        Forward pass (matching notebook implementation exactly).
        
        Original code pattern:
            for i in range(len(layers)-2):
                z = linears[i](a)
                a = activation(z)
            a = linears[-1](a)
        
        Args:
            x: Input tensor of shape (N, 3) where columns are (x, y, p_normalized)
               - x, y: spatial coordinates (normalized to [0, 1])
               - p_normalized: p value divided by 500
        
        Returns:
            Output tensor of shape (N, 1)
        """
        # Split input into spatial coordinates and p value
        a = x[:, 0:2]  # (x, y) - trunk input
        b = x[:, 2:3]  # p/500 - branch input
        
        # Trunk network: apply activation to all but the last layer
        # With trunk_layers = [2, 256, 256, 1], we have 3 linears:
        #   linears[0]: 2→256, linears[1]: 256→256, linears[2]: 256→1
        # Loop applies activation to first (len-2) layers, then last layer without activation
        for i in range(len(self.trunk_layers) - 2):
            z = self.trunk_linears[i](a)
            a = self.activation(z)
        a = self.trunk_linears[-1](a)  # Last layer, no activation
        
        # Branch network: same pattern
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


def generate_disc_points(n_points: int, normalize: bool = True) -> np.ndarray:
    """
    Generate random points inside the unit disc.
    
    Args:
        n_points: Number of points to generate
        normalize: If True, normalize to [0, 1] range (as in notebook)
    
    Returns:
        Array of shape (n_points, 2) with (x, y) coordinates
    """
    # Use rejection sampling for uniform distribution in disc
    points = []
    while len(points) < n_points:
        x = np.random.uniform(-1, 1, n_points * 2)
        y = np.random.uniform(-1, 1, n_points * 2)
        mask = x**2 + y**2 <= 1
        valid_points = np.column_stack([x[mask], y[mask]])
        points.extend(valid_points.tolist())
    
    points = np.array(points[:n_points])
    
    if normalize:
        points = points / 2.0 + 0.5
    
    return points


def generate_square_points(n_points: int, normalize: bool = True) -> np.ndarray:
    """
    Generate random points inside the unit square [-1, 1] × [-1, 1].
    
    Args:
        n_points: Number of points to generate
        normalize: If True, normalize to [0, 1] range
    
    Returns:
        Array of shape (n_points, 2) with (x, y) coordinates
    """
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    points = np.column_stack([x, y])
    
    if normalize:
        points = points / 2.0 + 0.5
    
    return points


def generate_test_grid(
    domain: str,
    n_points: int = 101,
    normalize: bool = True
) -> tuple:
    """
    Generate a test grid for evaluation.
    
    Args:
        domain: 'disc' or 'square'
        n_points: Number of points per dimension
        normalize: If True, normalize to [0, 1] range
    
    Returns:
        Tuple of (x_test, y_test) tensors
    """
    x = torch.linspace(-1, 1, n_points)
    y = torch.linspace(-1, 1, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    if domain == 'disc':
        mask = X**2 + Y**2 <= 1
        X_flat = X[mask]
        Y_flat = Y[mask]
    else:  # square
        X_flat = X.flatten()
        Y_flat = Y.flatten()
    
    # Compute exact solution
    y_exact = exact_solution(X_flat, Y_flat)
    
    # Create test points
    x_test = torch.stack([X_flat, Y_flat], dim=1)
    y_test = y_exact.unsqueeze(1)
    
    if normalize:
        x_test = x_test / 2.0 + 0.5
    
    return x_test, y_test


def load_mat_data(
    data_dir: str,
    domain: str,
    p_normalize: float = 500.0,
    include_exact_inf: bool = True
) -> tuple:
    """
    Load training data from .mat files generated by FEM solver.
    
    Args:
        data_dir: Directory containing .mat files
        domain: 'disc' or 'square' (subdirectory name)
        p_normalize: Normalization factor for p values
        include_exact_inf: If True, include exact solution at p=500 (as p=∞)
    
    Returns:
        Tuple of (X_train, Y_train) arrays
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required to load .mat files")
    
    mat_path = os.path.join(data_dir, domain.capitalize())
    if not os.path.exists(mat_path):
        return None, None
    
    mat_files = glob.glob(os.path.join(mat_path, '*.mat'))
    if not mat_files:
        return None, None
    
    X_train_list = []
    Y_train_list = []
    last_pts = None
    
    for mat_file in sorted(mat_files):
        try:
            mat = scipy.io.loadmat(mat_file)
            p = int(re.search(r'p\s*(\d+)', os.path.basename(mat_file)).group(1))
            
            pts = mat['pts'] / 2.0 + 0.5  # Normalize to [0, 1]
            solution = mat['solution'][0].reshape(-1, 1)
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
        pts_norm = last_pts / 2.0 + 0.5
        p_value = np.ones((pts_norm.shape[0], 1)) * 500.0 / p_normalize
        X_inf = np.hstack([pts_norm, p_value])
        Y_inf = exact_solution(
            torch.tensor(last_pts[:, 0]),
            torch.tensor(last_pts[:, 1])
        ).numpy().reshape(-1, 1)
        X_train_list.append(X_inf)
        Y_train_list.append(Y_inf)
        print(f"  Added exact solution at p=500 (∞): {X_inf.shape[0]} points")
    
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)
    
    return X_train, Y_train


def generate_synthetic_data(
    domain: str,
    p_values: list = None,
    n_points_per_p: int = 5000,
    p_normalize: float = 500.0,
    include_exact_inf: bool = True
) -> tuple:
    """
    Generate synthetic training data based on 1/p convergence rate approximation.
    
    For the distance to origin problem, the solution converges to sqrt(x²+y²) as p→∞
    at rate 1/p. We approximate u_p ≈ u_∞ - C/p for some constant.
    
    Args:
        domain: 'disc' or 'square'
        p_values: List of p values to generate data for. 
                  Default matches Table 5: [5, 25, 50, 75, 100, 150, 200]
        n_points_per_p: Number of points per p value
        p_normalize: Normalization factor for p values (500 as in paper)
        include_exact_inf: If True, include exact solution at p=500
    
    Returns:
        Tuple of (X_train, Y_train) arrays
    """
    # Default p values based on Table 5 in the paper
    if p_values is None:
        p_values = [5, 25, 50, 75, 100, 150, 200]
    
    X_train_list = []
    Y_train_list = []
    
    for p in p_values:
        if domain == 'disc':
            pts = generate_disc_points(n_points_per_p, normalize=False)
        else:
            pts = generate_square_points(n_points_per_p, normalize=False)
        
        # Exact solution at infinity
        u_inf = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        
        # Approximate u_p based on 1/p convergence (simplified model)
        # In practice, this should come from FEM solver
        C = 0.1  # Approximate correction constant
        u_p = u_inf - C / p
        u_p = np.maximum(u_p, 0)  # Ensure non-negative
        
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
        if domain == 'disc':
            pts = generate_disc_points(n_points_per_p, normalize=False)
        else:
            pts = generate_square_points(n_points_per_p, normalize=False)
        
        pts_norm = pts / 2.0 + 0.5
        p_value = np.ones((pts_norm.shape[0], 1)) * 500.0 / p_normalize
        u_inf = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        
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
    """
    Train the DeepONet model.
    
    Args:
        model: DeepONet model
        train_loader: DataLoader for training data
        test_x: Test input tensor
        test_y: Test target tensor
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Torch device
        show_every: Print progress every N epochs
    
    Returns:
        Dictionary of loss histories
    """
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
            # Test at p=200 (0.4 normalized)
            test_x_p200 = test_x.clone()
            test_x_p200[:, 2] = 200 / 500.0
            test_loss_p200 = model.compute_loss(test_x_p200.to(device), test_y.to(device))
            
            # Test at p=250 (0.5 normalized)
            test_x_p250 = test_x.clone()
            test_x_p250[:, 2] = 250 / 500.0
            test_loss_p250 = model.compute_loss(test_x_p250.to(device), test_y.to(device))
            
            # Test at p=500 (1.0 normalized, approximating p=∞)
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
    """
    Evaluate model over a range of p values.
    
    Args:
        model: Trained DeepONet model
        test_x: Test input tensor (without p value set)
        test_y: Test target tensor (exact solution at p=∞)
        device: Torch device
        p_range: List of p values to evaluate
    
    Returns:
        Dictionary with p values and corresponding MSE errors
    """
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
    output_dir: str
):
    """
    Plot training history and evaluation results.
    
    Args:
        loss_history: Training loss history
        eval_results: Evaluation results over p range
        domain: Domain name for plot title
        output_dir: Directory to save plots
    """
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
    ax.set_title(f'Training History - {domain.capitalize()} Domain')
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
    ax.set_title(f'DeepONet MSE vs p - {domain.capitalize()} Domain')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mse_vs_p_{domain}.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


def plot_prediction_comparison(
    model: DeepONet,
    domain: str,
    device: torch.device,
    output_dir: str,
    p_value: int = 200,
    n_points: int = 101
):
    """
    Plot comparison between model prediction and exact solution.
    
    Args:
        model: Trained DeepONet model
        domain: 'disc' or 'square'
        device: Torch device
        output_dir: Directory to save plots
        p_value: p value to evaluate at
        n_points: Grid resolution
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate grid
    x = torch.linspace(-1, 1, n_points)
    y = torch.linspace(-1, 1, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    if domain == 'disc':
        mask = X**2 + Y**2 <= 1
    else:
        mask = torch.ones_like(X, dtype=torch.bool)
    
    # Flatten for prediction
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Normalize and add p value
    X_norm = X_flat / 2.0 + 0.5
    Y_norm = Y_flat / 2.0 + 0.5
    p_norm = torch.full_like(X_flat, p_value / 500.0)
    
    test_input = torch.stack([X_norm, Y_norm, p_norm], dim=1)
    
    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(test_input.to(device)).cpu()
    
    # Exact solution
    exact = exact_solution(X_flat, Y_flat)
    
    # Reshape for plotting
    pred_grid = pred.squeeze().reshape(n_points, n_points).numpy()
    exact_grid = exact.reshape(n_points, n_points).numpy()
    
    # Apply mask for disc domain
    if domain == 'disc':
        pred_grid = np.where(mask.numpy(), pred_grid, np.nan)
        exact_grid = np.where(mask.numpy(), exact_grid, np.nan)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    X_np = X.numpy()
    Y_np = Y.numpy()
    
    # Prediction
    im0 = axes[0].contourf(X_np, Y_np, pred_grid, levels=20, cmap='rainbow')
    axes[0].set_title(f'DeepONet Prediction (p={p_value})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0])
    
    # Exact solution
    im1 = axes[1].contourf(X_np, Y_np, exact_grid, levels=20, cmap='rainbow')
    axes[1].set_title('Exact Solution (p=∞)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1])
    
    # Error
    error_grid = np.abs(pred_grid - exact_grid)
    im2 = axes[2].contourf(X_np, Y_np, error_grid, levels=20, cmap='hot')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prediction_comparison_{domain}_p{p_value}.png'), dpi=150)
    plt.close()
    
    print(f"Prediction comparison saved to {output_dir}/")


def run_experiment(
    domain: str,
    epochs: int,
    lr: float,
    batch_size: int,
    trunk_layers: list,
    branch_layers: list,
    data_dir: str,
    output_dir: str,
    seed: int,
    use_synthetic: bool = False,
    include_exact_inf: bool = True
):
    """
    Run the full DeepONet experiment for a given domain.
    
    Args:
        domain: 'disc' or 'square'
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        trunk_layers: Trunk network layer sizes
        branch_layers: Branch network layer sizes
        data_dir: Directory containing training data
        output_dir: Directory for output files
        seed: Random seed
        use_synthetic: If True, generate synthetic data instead of loading .mat files
        include_exact_inf: If True, include exact solution at p=500 (as p=∞) in training
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment 6.3.1: Distance to Origin - {domain.capitalize()} Domain")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load or generate training data
    print("\nLoading training data...")
    print(f"  Include exact p=∞ solution: {include_exact_inf}")
    if use_synthetic:
        X_train, Y_train = generate_synthetic_data(
            domain, include_exact_inf=include_exact_inf
        )
    else:
        X_train, Y_train = load_mat_data(
            data_dir, domain, include_exact_inf=include_exact_inf
        )
        if X_train is None:
            print(f"  No .mat files found in {data_dir}/{domain.capitalize()}/")
            print("  Generating synthetic training data instead...")
            X_train, Y_train = generate_synthetic_data(
                domain, include_exact_inf=include_exact_inf
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
    test_x, test_y = generate_test_grid(domain, n_points=101, normalize=True)
    # Add placeholder for p value
    test_x = torch.cat([test_x, torch.zeros(test_x.shape[0], 1)], dim=1)
    print(f"Test data shape: X={test_x.shape}, Y={test_y.shape}")
    
    # Create model
    print("\nCreating DeepONet model...")
    model = DeepONet(
        trunk_layers=trunk_layers,
        branch_layers=branch_layers,
        activation='tanh',
        device=device
    )
    model.to(device)
    print(model)
    
    # Count parameters
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
    
    # Print results table
    print("\nResults: MSE vs p")
    print("-" * 30)
    for i, (p, mse) in enumerate(zip(eval_results['p'], eval_results['mse'])):
        if i % 10 == 0:  # Print every 10th value
            print(f"p={p:3d}: MSE={mse:.6e}")
    
    # Plot results
    domain_output_dir = os.path.join(output_dir, domain)
    plot_results(loss_history, eval_results, domain, domain_output_dir)
    plot_prediction_comparison(model, domain, device, domain_output_dir, p_value=200)
    plot_prediction_comparison(model, domain, device, domain_output_dir, p_value=500)
    
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
        'include_exact_inf': include_exact_inf
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, loss_history, eval_results


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 6.3.1: Distance to Origin DeepONet'
    )
    parser.add_argument(
        '--domain', type=str, default='disc',
        choices=['disc', 'square', 'all'],
        help='Domain type: disc, square, or all'
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
        '--trunk-layers', type=str, default='2,256,256,1',
        help='Trunk network layers (comma-separated). Default: 2,256,256,1'
    )
    parser.add_argument(
        '--branch-layers', type=str, default='1,128,128,1',
        help='Branch network layers (comma-separated). Default: 1,128,128,1'
    )
    parser.add_argument(
        '--data-dir', type=str, default='./experiment_6_3_1',
        help='Directory containing training data (.mat files)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./outputs/experiment_6_3_1',
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
        help='Do NOT include exact solution at p=500 (p=∞) in training data. '
             'By default, exact p=∞ solution is included as p=500.'
    )
    
    args = parser.parse_args()
    
    # Parse layer configurations
    trunk_layers = [int(x) for x in args.trunk_layers.split(',')]
    branch_layers = [int(x) for x in args.branch_layers.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    if args.domain == 'all':
        domains = ['disc', 'square']
    else:
        domains = [args.domain]
    
    for domain in domains:
        run_experiment(
            domain=domain,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            trunk_layers=trunk_layers,
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
