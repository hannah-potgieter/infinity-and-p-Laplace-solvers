"""
Experiment 6.1: 2D Infinity-Laplacian Equation with Dirichlet Boundary Conditions

This script implements PINNs for solving the infinity-Laplacian equation:
    Δ_∞ u = u_x² u_xx + 2 u_x u_y u_xy + u_y² u_yy = 0  in Ω
    u = g  on ∂Ω

Includes three examples from Section 6.1 of the paper:
    1. Arctan Example: u(x,y) = arctan(y/x) on [0.01,1] × [0.01,1]
    2. Absolute Example: u(x,y) = |x| - |y| on [-1,1] × [-1,1]  
    3. Aronsson Example: u(x,y) = |x|^(4/3) - |y|^(4/3) on [-1,1]² or unit disc

Reference: "Solving p-Laplacian and infinity-Laplacian with Deep Learning"
Authors: Tak Shing Au Yeung, Ka Chun Cheung, Hannah Potgieter, Steven J. Ruuth, Simon See
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse
import random
import os
from typing import List, Optional, Tuple
import copy


def create_output_dirs(base_dir='outputs'):
    """
    Create output directories for plots and checkpoints.
    
    Args:
        base_dir: Base directory name (default: 'outputs')
        
    Returns:
        dict with 'plots' and 'checkpoints' paths
    """
    plots_dir = os.path.join(base_dir, 'plots')
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    return {
        'base': base_dir,
        'plots': plots_dir,
        'checkpoints': checkpoints_dir
    }


def set_seed(seed=1234):
    """
    Set all random seeds for reproducibility.
    
    This ensures that results are the same across runs when the seed is fixed.
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


# Set default dtype and random seeds
torch.set_default_dtype(torch.float32)
SEED = 1234
set_seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Random seed: {SEED}")


# ============================================================================
# ReLoBRaLo: Relative Loss Balancing with Random Lookback
# Based on: "Multi-Objective Loss Balancing for Physics-Informed Deep Learning"
# Paper: https://arxiv.org/abs/2110.09813
# ============================================================================

class ReLoBRaLo:
    """
    ReLoBRaLo (Relative Loss Balancing with Random Lookback)
    
    A self-adaptive loss balancing scheme for Physics-Informed Neural Networks.
    Dynamically adjusts weights of different loss components during training.
    
    Algorithm:
    1. Track loss history for each component
    2. Compute relative change in losses compared to a random lookback point
    3. Use softmax to normalize weights
    4. Apply exponential moving average for stability
    
    Reference: Bischof & Kraus (2021), arXiv:2110.09813
    """
    
    def __init__(
        self,
        num_losses: int = 2,
        alpha: float = 0.999,
        temperature: float = 1.0,
        rho: float = 0.99,
        random_lookback: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ReLoBRaLo.
        
        Args:
            num_losses: Number of loss components (e.g., 2 for BC + PDE)
            alpha: Exponential moving average decay for weights (default 0.999)
            temperature: Softmax temperature (higher = more uniform weights)
            rho: Probability of using random lookback vs initial loss (default 0.99)
            random_lookback: Whether to use random lookback (True) or fixed (False)
            device: Torch device
        """
        self.num_losses = num_losses
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.random_lookback = random_lookback
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tracking variables
        self.loss_history: List[List[float]] = [[] for _ in range(num_losses)]
        self.weights = torch.ones(num_losses, device=self.device) / num_losses
        self.initial_losses: Optional[torch.Tensor] = None
        self.step = 0
    
    def update(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Update loss weights based on current losses.
        
        Args:
            losses: List of loss tensors [loss_1, loss_2, ...]
        
        Returns:
            Updated weight tensor
        """
        assert len(losses) == self.num_losses, \
            f"Expected {self.num_losses} losses, got {len(losses)}"
        
        # Convert to tensor and detach
        current_losses = torch.tensor(
            [l.detach().item() if isinstance(l, torch.Tensor) else l for l in losses],
            device=self.device
        )
        
        # Store initial losses
        if self.initial_losses is None:
            self.initial_losses = current_losses.clone()
        
        # Store in history
        for i, loss in enumerate(current_losses):
            self.loss_history[i].append(loss.item())
        
        self.step += 1
        
        # Need at least 2 steps to compute relative change
        if self.step < 2:
            return self.weights
        
        # Get reference losses (random lookback or initial)
        if self.random_lookback and random.random() < self.rho and self.step > 2:
            # Random lookback: pick a random previous step
            lookback_idx = random.randint(0, self.step - 2)
            reference_losses = torch.tensor(
                [self.loss_history[i][lookback_idx] for i in range(self.num_losses)],
                device=self.device
            )
        else:
            # Use initial losses as reference
            reference_losses = self.initial_losses
        
        # Compute relative change (avoid division by zero)
        eps = 1e-8
        relative_change = current_losses / (reference_losses + eps)
        
        # Apply softmax with temperature to get new weights
        log_weights = torch.log(relative_change + eps) / self.temperature
        new_weights = torch.softmax(log_weights, dim=0)
        
        # Apply exponential moving average
        self.weights = self.alpha * self.weights + (1 - self.alpha) * new_weights
        
        # Normalize weights to sum to num_losses (so average weight is 1)
        self.weights = self.weights * self.num_losses / self.weights.sum()
        
        return self.weights
    
    def get_weights(self) -> torch.Tensor:
        """Get current weights."""
        return self.weights
    
    def reset(self):
        """Reset the loss balancer."""
        self.loss_history = [[] for _ in range(self.num_losses)]
        self.weights = torch.ones(self.num_losses, device=self.device) / self.num_losses
        self.initial_losses = None
        self.step = 0


# ============================================================================
# Activation Function Registry
# ============================================================================

ACTIVATIONS = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'softplus': nn.Softplus,
    'silu': nn.SiLU,  # Also known as Swish
    'sigmoid': nn.Sigmoid,
}


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()


class InfinityLaplacianPINN(nn.Module):
    """
    Physics-Informed Neural Network for solving the infinity-Laplacian equation.
    
    Architecture (from paper Section 4):
    - Configurable hidden layers
    - Configurable activation function
    - Single output for u(x,y)
    
    NOTE: The original notebook (PINN_plap.ipynb) uses a two-branch architecture 
    with smaller networks [2,32,32,32,1] and [2,16,16,1] that are multiplied.
    This file follows the paper's description of a simple feedforward network.
    """
    
    def __init__(self, hidden_layers=[512, 512, 512, 512], activation='tanh', eta=0.0,
                 clip_residual=0.0, outlier_percentile=1.0):
        super().__init__()
        
        self.activation = get_activation(activation)
        self.activation_name = activation
        self.loss_function = nn.MSELoss(reduction='mean')
        self.eta = eta  # Regularization parameter for normalized infinity-Laplacian
        self.clip_residual = clip_residual  # Clip PDE residual to [-clip, clip]
        self.outlier_percentile = outlier_percentile  # Exclude top X% of extreme PDE losses
        
        # Build network layers: input(2) -> hidden -> output(1)
        layers = []
        in_features = 2  # (x, y)
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, 1))  # Output layer
        
        self.linears = nn.ModuleList(layers)
        
        # Xavier Normal Initialization
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight.data, gain=1.0)
            nn.init.zeros_(layer.bias.data)
    
    def forward(self, x):
        """Forward pass through the network."""
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        
        a = x.float()
        for i in range(len(self.linears) - 1):
            a = self.activation(self.linears[i](a))
        a = self.linears[-1](a)  # No activation on output layer
        return a
    
    def compute_infinity_laplacian(self, x):
        """
        Compute the infinity-Laplacian of the network output.
        
        Standard form:
            Δ_∞ u = u_x² u_xx + 2 u_x u_y u_xy + u_y² u_yy
        
        Normalized form with regularization (when eta > 0):
            Δ_∞^N u = (u_x² u_xx + 2 u_x u_y u_xy + u_y² u_yy) / (η² + |∇u|²)
        
        The regularization η² prevents numerical instability when |∇u| → 0,
        similar to the paper's approach for p-Laplacian:
            γ(u) = (η² + |∇u|²)^{(p-2)/2}
        
        This is equivalent to: (Du)^T D²u Du where Du is gradient, D²u is Hessian.
        """
        g = x.clone()
        g.requires_grad = True
        
        u = self.forward(g)
        
        # First derivatives
        grad_u = autograd.grad(u, g, torch.ones_like(u), 
                               retain_graph=True, create_graph=True)[0]
        u_x = grad_u[:, [0]]
        u_y = grad_u[:, [1]]
        
        # Second derivatives
        grad_ux = autograd.grad(u_x, g, torch.ones_like(u_x), 
                                create_graph=True)[0]
        grad_uy = autograd.grad(u_y, g, torch.ones_like(u_y), 
                                create_graph=True)[0]
        
        u_xx = grad_ux[:, [0]]
        u_xy = grad_ux[:, [1]]
        u_yx = grad_uy[:, [0]]  # Should equal u_xy for smooth functions
        u_yy = grad_uy[:, [1]]
        
        # Infinity Laplacian: u_x² u_xx + 2 u_x u_y u_xy + u_y² u_yy
        # Note: Using u_xy + u_yx = 2*u_xy for numerical stability
        delta_inf = u_x**2 * u_xx + u_x * u_y * (u_xy + u_yx) + u_y**2 * u_yy
        
        # Apply normalization with regularization if eta > 0
        if self.eta > 0:
            grad_norm_sq = u_x**2 + u_y**2
            delta_inf = delta_inf / (self.eta**2 + grad_norm_sq)
        
        return delta_inf
    
    def loss_boundary(self, x_bc, y_bc):
        """Data-driven loss: MSE on boundary conditions."""
        return self.loss_function(self.forward(x_bc), y_bc)
    
    def loss_pde(self, x_pde):
        """
        PINNs loss: Residual of the infinity-Laplacian equation.
        
        Stabilization options:
        - clip_residual > 0: Clip residual to [-clip, clip]
        - outlier_percentile > 0: Exclude top X% of extreme losses
        """
        delta_inf = self.compute_infinity_laplacian(x_pde)
        
        # Clip residual to prevent extreme values from dominating
        if self.clip_residual > 0:
            delta_inf = torch.clamp(delta_inf, -self.clip_residual, self.clip_residual)
        
        # Outlier removal: exclude top X% of extreme losses
        if self.outlier_percentile > 0 and delta_inf.numel() > 10:
            # Compute per-point squared loss
            per_point_loss = delta_inf.squeeze() ** 2
            
            # Find threshold at (100 - outlier_percentile) percentile
            keep_percentile = 100.0 - self.outlier_percentile
            threshold = torch.quantile(per_point_loss, keep_percentile / 100.0)
            
            # Keep only points below threshold
            keep_mask = per_point_loss <= threshold
            
            if keep_mask.sum() > 0:
                # Compute mean loss from kept points only
                return per_point_loss[keep_mask].mean()
            # Fallback if all filtered out
        
        zeros = torch.zeros_like(delta_inf)
        return self.loss_function(delta_inf, zeros)
    
    def total_loss(self, x_bc, y_bc, x_pde, alpha):
        """
        Total loss = MSE_boundary + alpha * MSE_PDE
        
        From paper Section 4: Multi-step α method
        - Start with small α (10^-3) for stable training
        - Increase α when boundary loss reaches threshold (10^-4)
        """
        loss_bc = self.loss_boundary(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        return loss_bc + alpha * loss_pde, loss_bc, loss_pde
    
    def get_pde_loss_with_outlier_info(self, x_pde):
        """
        Compute PDE loss and return detailed info for visualization.
        
        Returns:
            x_coords: x coordinates of points
            y_coords: y coordinates of points  
            per_point_loss: squared residual for each point
            is_outlier: boolean mask indicating outlier points
            threshold: the outlier threshold value
        """
        # compute_infinity_laplacian handles requires_grad internally
        with torch.enable_grad():
            delta_inf = self.compute_infinity_laplacian(x_pde)
        
        # Detach from computation graph for visualization
        delta_inf = delta_inf.detach()
        
        if self.clip_residual > 0:
            delta_inf = torch.clamp(delta_inf, -self.clip_residual, self.clip_residual)
        
        per_point_loss = (delta_inf.squeeze() ** 2).cpu().numpy()
        x_coords = x_pde[:, 0].detach().cpu().numpy()
        y_coords = x_pde[:, 1].detach().cpu().numpy()
        
        if self.outlier_percentile > 0:
            keep_percentile = 100.0 - self.outlier_percentile
            threshold = np.percentile(per_point_loss, keep_percentile)
            is_outlier = per_point_loss > threshold
        else:
            threshold = np.max(per_point_loss) + 1
            is_outlier = np.zeros(len(per_point_loss), dtype=bool)
        
        return x_coords, y_coords, per_point_loss, is_outlier, threshold


def plot_pde_loss_distribution(model, x_interior, epoch, example_name, output_dir='.', outlier_color='red'):
    """
    Plot PDE loss distribution showing outliers in different color.
    
    Args:
        model: The PINN model
        x_interior: Interior points tensor
        epoch: Current epoch number
        example_name: Name of the example (for title)
        output_dir: Directory to save plots (default: current directory)
        outlier_color: Color for outlier points (default: red)
    """
    # Store current training mode and set to eval
    was_training = model.training
    model.eval()
    
    # Sample points if too many (for faster plotting)
    n_points = min(10000, len(x_interior))
    if len(x_interior) > n_points:
        indices = torch.randperm(len(x_interior))[:n_points]
        x_sample = x_interior[indices].to(device)
    else:
        x_sample = x_interior.to(device)
    
    x_coords, y_coords, per_point_loss, is_outlier, threshold = \
        model.get_pde_loss_with_outlier_info(x_sample)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Spatial distribution with outliers highlighted
    ax = axes[0]
    # Plot non-outliers first (blue)
    ax.scatter(x_coords[~is_outlier], y_coords[~is_outlier], 
               c='blue', s=1, alpha=0.5, label='Normal')
    # Plot outliers on top (red)
    n_outliers = is_outlier.sum()
    ax.scatter(x_coords[is_outlier], y_coords[is_outlier], 
               c=outlier_color, s=10, alpha=0.8, label=f'Outlier ({n_outliers})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Epoch {epoch}: Outlier Locations\n(Top {model.outlier_percentile}% = {n_outliers} points)')
    ax.legend()
    ax.set_aspect('equal')
    
    # Plot 2: Loss magnitude heatmap
    ax = axes[1]
    # Use log scale for loss values
    log_loss = np.log10(per_point_loss + 1e-10)
    sc = ax.scatter(x_coords, y_coords, c=log_loss, s=2, cmap='hot', alpha=0.7)
    plt.colorbar(sc, ax=ax, label='log10(loss)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Epoch {epoch}: PDE Loss Magnitude')
    ax.set_aspect('equal')
    
    # Plot 3: Loss histogram
    ax = axes[2]
    log_loss_all = np.log10(per_point_loss + 1e-10)
    ax.hist(log_loss_all[~is_outlier], bins=50, alpha=0.7, color='blue', label='Normal')
    ax.hist(log_loss_all[is_outlier], bins=50, alpha=0.7, color=outlier_color, label='Outlier')
    ax.axvline(np.log10(threshold + 1e-10), color='black', linestyle='--', 
               label=f'Threshold ({threshold:.2e})')
    ax.set_xlabel('log10(PDE loss)')
    ax.set_ylabel('Count')
    ax.set_title(f'Epoch {epoch}: Loss Distribution')
    ax.legend()
    
    plt.suptitle(f'{example_name} - PDE Loss Analysis (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    filename = f'pde_loss_epoch_{epoch:04d}_{example_name}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved PDE loss plot: {filepath}")
    
    # Restore training mode
    if was_training:
        model.train()
    
    return filename


def generate_square_domain_data_full(x_min, x_max, y_min, y_max, 
                                      n_interior_grid=1000, n_boundary_grid=10000, 
                                      f_exact=None, seed=1234):
    """
    Generate FULL domain training data for a square/rectangular domain.
    
    From paper Section 6.1:
    - Interior: n_interior_grid × n_interior_grid uniform grid (default 1000×1000 = 1M points)
    - Boundary: Finer n_boundary_grid × n_boundary_grid grid for boundary points
    
    Args:
        x_min, x_max, y_min, y_max: Domain bounds
        n_interior_grid: Grid size for interior (default 1000, gives 1M points)
        n_boundary_grid: Grid size for boundary (default 10000)
        f_exact: Exact solution function
        seed: Seed for reproducible shuffling (default: 1234)
    
    Returns:
        x_bc: All boundary points
        y_bc: Boundary values
        x_interior: All interior points
        x_test: Test grid points
        y_test: Test values
    """
    # Set generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # =========================================================================
    # BOUNDARY POINTS: From finer grid (4 edges)
    # =========================================================================
    n_per_edge = n_boundary_grid
    
    # Left edge (x = x_min)
    left_x = torch.ones(n_per_edge, 1) * x_min
    left_y = torch.linspace(y_min, y_max, n_per_edge).view(-1, 1)
    
    # Right edge (x = x_max)
    right_x = torch.ones(n_per_edge, 1) * x_max
    right_y = torch.linspace(y_min, y_max, n_per_edge).view(-1, 1)
    
    # Bottom edge (y = y_min), exclude corners
    bottom_x = torch.linspace(x_min, x_max, n_per_edge + 2)[1:-1].view(-1, 1)
    bottom_y = torch.ones(n_per_edge, 1) * y_min
    
    # Top edge (y = y_max), exclude corners
    top_x = torch.linspace(x_min, x_max, n_per_edge + 2)[1:-1].view(-1, 1)
    top_y = torch.ones(n_per_edge, 1) * y_max
    
    x_bc = torch.cat([
        torch.cat([left_x, left_y], dim=1),
        torch.cat([right_x, right_y], dim=1),
        torch.cat([bottom_x, bottom_y], dim=1),
        torch.cat([top_x, top_y], dim=1)
    ], dim=0)
    
    y_bc = f_exact(x_bc[:, 0], x_bc[:, 1]).view(-1, 1)
    
    # Shuffle boundary points for random batching
    perm = torch.randperm(x_bc.shape[0], generator=generator)
    x_bc = x_bc[perm]
    y_bc = y_bc[perm]
    
    # =========================================================================
    # INTERIOR POINTS: Full grid (1000×1000 = 1M points)
    # =========================================================================
    # Create grid excluding boundary
    eps = 1e-6  # Small offset to avoid boundary
    x_1d = torch.linspace(x_min + eps, x_max - eps, n_interior_grid)
    y_1d = torch.linspace(y_min + eps, y_max - eps, n_interior_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    x_interior = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Shuffle interior points for random batching
    perm = torch.randperm(x_interior.shape[0], generator=generator)
    x_interior = x_interior[perm]
    
    # =========================================================================
    # TEST GRID: Smaller grid for evaluation
    # =========================================================================
    n_test = 100
    x_test_1d = torch.linspace(x_min, x_max, n_test)
    y_test_1d = torch.linspace(y_min, y_max, n_test)
    X, Y = torch.meshgrid(x_test_1d, y_test_1d, indexing='ij')
    x_test = torch.stack([X.flatten(), Y.flatten()], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1]).view(-1, 1)
    
    print(f"Generated full domain data:")
    print(f"  Boundary points: {x_bc.shape[0]:,}")
    print(f"  Interior points: {x_interior.shape[0]:,}")
    print(f"  Test points: {x_test.shape[0]:,}")
    
    return x_bc, y_bc, x_interior, x_test, y_test


def generate_disc_domain_data_full(radius, n_boundary=10000, n_interior_grid=1000, 
                                    f_exact=None, seed=1234):
    """
    Generate FULL domain training data for a unit disc domain.
    
    Args:
        radius: Disc radius
        n_boundary: Number of boundary points on circle (default 10000)
        n_interior_grid: Grid size for interior sampling (default 1000)
        f_exact: Exact solution function
        seed: Seed for reproducible sampling (default: 1234)
    
    Returns:
        x_bc: All boundary points
        y_bc: Boundary values
        x_interior: All interior points
        x_test: Test grid points
        y_test: Test values
    """
    # Set generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # =========================================================================
    # BOUNDARY POINTS: Points on circle
    # =========================================================================
    theta = torch.linspace(0, 2 * np.pi, n_boundary + 1)[:-1]
    x_bc = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    y_bc = f_exact(x_bc[:, 0], x_bc[:, 1]).view(-1, 1)
    
    # Shuffle boundary points
    perm = torch.randperm(x_bc.shape[0], generator=generator)
    x_bc = x_bc[perm]
    y_bc = y_bc[perm]
    
    # =========================================================================
    # INTERIOR POINTS: Grid filtered to disc interior
    # =========================================================================
    eps = 1e-6  # Small offset from boundary
    x_1d = torch.linspace(-radius + eps, radius - eps, n_interior_grid)
    y_1d = torch.linspace(-radius + eps, radius - eps, n_interior_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    
    # Filter to interior of disc (r < radius)
    mask = (X**2 + Y**2) < (radius - eps)**2
    x_interior = torch.stack([X[mask].flatten(), Y[mask].flatten()], dim=1)
    
    # Shuffle interior points
    perm = torch.randperm(x_interior.shape[0], generator=generator)
    x_interior = x_interior[perm]
    
    # =========================================================================
    # TEST GRID: Smaller grid for evaluation
    # =========================================================================
    n_test = 100
    x_test_1d = torch.linspace(-radius, radius, n_test)
    y_test_1d = torch.linspace(-radius, radius, n_test)
    X, Y = torch.meshgrid(x_test_1d, y_test_1d, indexing='ij')
    mask = X**2 + Y**2 <= radius**2
    x_test = torch.stack([X[mask].flatten(), Y[mask].flatten()], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1]).view(-1, 1)
    
    print(f"Generated full disc domain data:")
    print(f"  Boundary points: {x_bc.shape[0]:,}")
    print(f"  Interior points: {x_interior.shape[0]:,}")
    print(f"  Test points: {x_test.shape[0]:,}")
    
    return x_bc, y_bc, x_interior, x_test, y_test


# ============================================================================
# Exact Solutions for the Three Examples
# ============================================================================

def arctan_solution(x, y):
    """Example 6.1.1: u(x,y) = arctan(y/x)"""
    return torch.atan(y / x)


def absolute_solution(x, y):
    """Example 6.1.2: u(x,y) = |x| - |y|"""
    return torch.abs(x) - torch.abs(y)


def aronsson_solution(x, y):
    """Example 6.1.3: u(x,y) = |x|^(4/3) - |y|^(4/3)"""
    return torch.abs(x)**(4/3) - torch.abs(y)**(4/3)


# ============================================================================
# Interior Point Sampler (for re-sampling each epoch)
# ============================================================================

class InteriorSampler:
    """
    Sampler for interior collocation points with soft exclusion near axes.
    Re-samples points each epoch for better domain coverage.
    """
    
    def __init__(self, domain_type: str, domain_params: dict, 
                 n_points: int):
        """
        Initialize the sampler.
        
        Args:
            domain_type: 'square' or 'disc'
            domain_params: For 'square': {'x_min', 'x_max', 'y_min', 'y_max'}
                          For 'disc': {'radius'}
            n_points: Number of points to sample
        """
        self.domain_type = domain_type
        self.domain_params = domain_params
        self.n_points = n_points
        self.epoch = 0
    
    def sample(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Sample interior points with soft exclusion near axes.
        
        Args:
            seed: Random seed (if None, uses epoch-based seed)
        
        Returns:
            x_interior: Tensor of shape (n_sampled, 2)
        """
        if seed is None:
            seed = 1234 + self.epoch
        self.epoch += 1
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        if self.domain_type == 'square':
            return self._sample_square(generator)
        elif self.domain_type == 'disc':
            return self._sample_disc(generator)
        else:
            raise ValueError(f"Unknown domain type: {self.domain_type}")
    
    def _sample_square(self, generator: torch.Generator) -> torch.Tensor:
        """Sample from square domain."""
        x_min = self.domain_params['x_min']
        x_max = self.domain_params['x_max']
        y_min = self.domain_params['y_min']
        y_max = self.domain_params['y_max']
        
        # Uniform random sampling
        x = torch.rand(self.n_points, generator=generator) * (x_max - x_min) + x_min
        y = torch.rand(self.n_points, generator=generator) * (y_max - y_min) + y_min
        x_interior = torch.stack([x, y], dim=1)
        
        return x_interior
    
    def _sample_disc(self, generator: torch.Generator) -> torch.Tensor:
        """Sample from disc domain using rejection sampling."""
        radius = self.domain_params['radius']
        
        # Over-sample to account for rejection
        oversample_factor = 1.3
        n_sample = int(self.n_points * oversample_factor)
        
        # Sample in square, reject outside disc
        x = torch.rand(n_sample, generator=generator) * 2 * radius - radius
        y = torch.rand(n_sample, generator=generator) * 2 * radius - radius
        x_interior = torch.stack([x, y], dim=1)
        
        # Keep only points inside disc
        r_squared = x_interior[:, 0]**2 + x_interior[:, 1]**2
        mask = r_squared < radius**2
        x_interior = x_interior[mask]
        
        # Limit to requested number
        if x_interior.shape[0] > self.n_points:
            x_interior = x_interior[:self.n_points]
        
        return x_interior


# ============================================================================
# Data Splitting for Validation
# ============================================================================

def split_data(x: torch.Tensor, y: Optional[torch.Tensor] = None, 
               train_ratio: float = 0.8, seed: int = 1234
               ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Split data into training and validation sets.
    
    Args:
        x: Input tensor
        y: Optional target tensor
        train_ratio: Fraction of data for training (default 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        x_train, x_val, y_train (or None), y_val (or None)
    """
    generator = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    n_train = int(n * train_ratio)
    
    # Random permutation
    perm = torch.randperm(n, generator=generator)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    x_train = x[train_idx]
    x_val = x[val_idx]
    
    if y is not None:
        y_train = y[train_idx]
        y_val = y[val_idx]
        return x_train, x_val, y_train, y_val
    else:
        return x_train, x_val, None, None


# ============================================================================
# Training Function
# ============================================================================

def train_pinn(model, x_bc, y_bc, x_interior, x_test, y_test, 
               n_epochs=10, batch_size_bc=400, batch_size_pde=1000,
               lr=1e-3, alpha_schedule=None, use_relobralo=False,
               relobralo_alpha=0.999, relobralo_temperature=1.0,
               x_bc_val=None, y_bc_val=None, x_interior_val=None,
               patience=None, interior_sampler=None,
               scheduler_type=None, scheduler_end_factor=0.01,
               plot_pde_every=0, example_name='', output_dirs=None):
    """
    Train the PINN model with proper epoch-based training.
    
    From paper Section 6.1:
    - 1 step = batch_size_bc boundary points + batch_size_pde PDE points
    - 1 epoch = full pass through ALL interior and boundary points
    
    Args:
        model: PINN model
        x_bc, y_bc: Training boundary points and values
        x_interior: Training interior/collocation points (used if interior_sampler is None)
        x_test, y_test: Test data (with exact solution for final evaluation)
        n_epochs: Number of epochs (full passes through data)
        batch_size_bc: Boundary points per step (default 400 from paper)
        batch_size_pde: PDE/collocation points per step (default 1000 from paper)
        lr: Learning rate
        alpha_schedule: Dict mapping epoch -> alpha value (ignored if use_relobralo=True)
        use_relobralo: Whether to use ReLoBRaLo for adaptive loss balancing
        relobralo_alpha: EMA decay for ReLoBRaLo (default 0.999)
        relobralo_temperature: Softmax temperature for ReLoBRaLo (default 1.0)
        x_bc_val, y_bc_val: Validation boundary points (for checkpoint selection)
        x_interior_val: Validation interior points (not used, kept for compatibility)
        patience: Early stopping patience (None = no early stopping)
        interior_sampler: InteriorSampler for re-sampling points each epoch (optional)
        scheduler_type: LR scheduler type ('linear', 'cosine', 'step', None for no scheduler)
        scheduler_end_factor: End LR factor for linear scheduler (default 0.01 = LR decays to 1%)
    
    Returns:
        history: Training history dict
        training_time: Total training time
    """
    # Determine if we have validation data
    use_validation = (x_bc_val is not None and y_bc_val is not None 
                      and x_interior_val is not None)
    
    # Determine if we're re-sampling interior points each epoch
    resample_interior = interior_sampler is not None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create learning rate scheduler if specified
    scheduler = None
    if scheduler_type == 'linear':
        # Linear decay from lr to lr * end_factor over all epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=scheduler_end_factor, 
            total_iters=n_epochs
        )
    elif scheduler_type == 'cosine':
        # Cosine annealing to near zero
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * scheduler_end_factor
        )
    elif scheduler_type == 'step':
        # Step decay: reduce LR by 10x every 1/3 of training
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, n_epochs // 3), gamma=0.1
        )
    elif scheduler_type == 'exponential':
        # Exponential decay to reach end_factor at the end
        gamma = scheduler_end_factor ** (1.0 / n_epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # Move test data to device
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Move validation data to device if available
    if use_validation:
        x_bc_val = x_bc_val.to(device)
        y_bc_val = y_bc_val.to(device)
        x_interior_val = x_interior_val.to(device)
    
    # Create DataLoaders for batching
    bc_dataset = TensorDataset(x_bc, y_bc)
    bc_loader = DataLoader(bc_dataset, batch_size=batch_size_bc, shuffle=True, 
                          drop_last=False, generator=torch.Generator().manual_seed(SEED))
    
    # Interior points: either fixed or re-sampled each epoch
    if resample_interior:
        # Will re-sample at the start of each epoch
        n_interior_points = interior_sampler.n_points
        interior_loader = None  # Created each epoch
    else:
        interior_dataset = TensorDataset(x_interior)
        interior_loader = DataLoader(interior_dataset, batch_size=batch_size_pde, shuffle=True,
                                     drop_last=False, generator=torch.Generator().manual_seed(SEED))
        n_interior_points = len(x_interior)
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Initialize ReLoBRaLo if enabled
    if use_relobralo:
        loss_balancer = ReLoBRaLo(
            num_losses=2,  # BC loss and PDE loss
            alpha=relobralo_alpha,
            temperature=relobralo_temperature,
            device=device
        )
        print(f"  Using ReLoBRaLo (α={relobralo_alpha}, T={relobralo_temperature})")
    else:
        loss_balancer = None
        # Default alpha schedule
        # Note: The notebook uses alpha=1.0 throughout (no ramping)
        # For large datasets, we can use a gentler schedule
        if alpha_schedule is None:
            alpha_schedule = {
                0: 1e-4,      # Start small
                100: 1e-3,    # Increase after boundary loss stabilizes
                500: 1e-2,
            }
    
    history = {
        'total_loss': [],
        'bc_loss': [],
        'pde_loss': [],
        'val_loss': [],
        'val_bc_loss': [],
        'val_pde_loss': [],
        'test_loss': [],
        'step': [],
        'weight_bc': [],
        'weight_pde': [],
        'lr': []
    }
    
    # Calculate steps per epoch
    n_bc_batches = len(bc_loader)
    n_pde_batches = (n_interior_points + batch_size_pde - 1) // batch_size_pde
    steps_per_epoch = max(n_bc_batches, n_pde_batches)
    total_steps = n_epochs * steps_per_epoch
    
    print(f"\nTraining configuration:")
    print(f"  Boundary points (train): {len(x_bc):,} -> {n_bc_batches} batches of {batch_size_bc}")
    print(f"  Interior points (train): {n_interior_points:,} -> {n_pde_batches} batches of {batch_size_pde}")
    if resample_interior:
        print(f"  Interior sampling: RE-SAMPLE each epoch")
    if use_validation:
        print(f"  Boundary points (val): {len(x_bc_val):,}")
        print(f"  Validation loss: BC only")
        if patience:
            print(f"  Early stopping patience: {patience}")
    else:
        print(f"  Validation: Disabled (using final model)")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Loss balancing: {'ReLoBRaLo' if use_relobralo else 'Fixed alpha schedule'}")
    if scheduler is not None:
        print(f"  LR scheduler: {scheduler_type} (LR: {lr:.1e} → {lr * scheduler_end_factor:.1e})")
    else:
        print(f"  LR scheduler: None (constant LR={lr:.1e})")
    print()
    
    start_time = time.time()
    current_alpha = 1.0 if use_relobralo else alpha_schedule.get(0, 1e-3)
    global_step = 0
    
    # Determine when the final alpha stage starts (for best model saving)
    # Only save best model during the final stage of alpha schedule
    if use_relobralo or len(alpha_schedule) <= 1:
        final_stage_epoch = 0  # No schedule or ReLoBRaLo, save from start
    else:
        final_stage_epoch = max(alpha_schedule.keys())
    
    entered_final_stage = False
    
    for epoch in range(n_epochs):
        # Update alpha based on schedule (only if not using ReLoBRaLo)
        if not use_relobralo and epoch in alpha_schedule:
            current_alpha = alpha_schedule[epoch]
            # Notify when entering final stage (for model selection)
            if epoch == final_stage_epoch and not entered_final_stage and final_stage_epoch > 0:
                print(f"\n>>> Entering final alpha stage (epoch {epoch + 1}): now tracking best model <<<\n")
                entered_final_stage = True
        
        # Re-sample interior points if using sampler
        if resample_interior:
            x_interior_epoch = interior_sampler.sample()
            interior_dataset = TensorDataset(x_interior_epoch)
            interior_loader = DataLoader(interior_dataset, batch_size=batch_size_pde, shuffle=True,
                                        drop_last=False, generator=torch.Generator().manual_seed(SEED + epoch))
        
        # Create iterators for this epoch
        bc_iter = iter(bc_loader)
        pde_iter = iter(interior_loader)
        
        epoch_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_pde_loss = 0.0
        epoch_weight_bc = 0.0
        epoch_weight_pde = 0.0
        n_steps_this_epoch = 0
        
        # Iterate through batches
        for step in range(steps_per_epoch):
            # Get boundary batch (cycle if needed)
            try:
                x_bc_batch, y_bc_batch = next(bc_iter)
            except StopIteration:
                bc_iter = iter(bc_loader)
                x_bc_batch, y_bc_batch = next(bc_iter)
            
            # Get PDE batch (cycle if needed)
            try:
                (x_pde_batch,) = next(pde_iter)
            except StopIteration:
                pde_iter = iter(interior_loader)
                (x_pde_batch,) = next(pde_iter)
            
            # Move to device
            x_bc_batch = x_bc_batch.to(device)
            y_bc_batch = y_bc_batch.to(device)
            x_pde_batch = x_pde_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Compute individual losses
            loss_bc = model.loss_boundary(x_bc_batch, y_bc_batch)
            loss_pde = model.loss_pde(x_pde_batch)
            
            # Compute weights
            if use_relobralo:
                # ReLoBRaLo: adaptive weights
                weights = loss_balancer.update([loss_bc, loss_pde])
                weight_bc = weights[0].item()
                weight_pde = weights[1].item()
                loss = weight_bc * loss_bc + weight_pde * loss_pde
            else:
                # Fixed schedule: weight_bc = 1, weight_pde = alpha
                weight_bc = 1.0
                weight_pde = current_alpha
                loss = loss_bc + current_alpha * loss_pde
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability (mentioned in paper Section 7)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_bc_loss += loss_bc.item()
            epoch_pde_loss += loss_pde.item()
            epoch_weight_bc += weight_bc
            epoch_weight_pde += weight_pde
            n_steps_this_epoch += 1
            global_step += 1
        
        # Average losses for this epoch
        avg_loss = epoch_loss / n_steps_this_epoch
        avg_bc_loss = epoch_bc_loss / n_steps_this_epoch
        avg_pde_loss = epoch_pde_loss / n_steps_this_epoch
        avg_weight_bc = epoch_weight_bc / n_steps_this_epoch
        avg_weight_pde = epoch_weight_pde / n_steps_this_epoch
        
        # Compute test loss (against exact solution)
        with torch.no_grad():
            test_loss = model.loss_boundary(x_test, y_test)
        
        # Compute validation loss (for checkpoint selection)
        # Using BC-only validation: faster and avoids gradient issues with PDE loss
        if use_validation:
            with torch.no_grad():
                val_bc_loss = model.loss_boundary(x_bc_val, y_bc_val)
                val_loss = val_bc_loss  # BC-only validation
            
            history['val_loss'].append(val_loss.item())
            history['val_bc_loss'].append(val_bc_loss.item())
            history['val_pde_loss'].append(None)  # Not computed
            
            # Check if this is the best model (only save during final alpha stage)
            in_final_stage = (epoch >= final_stage_epoch)
            
            if val_loss.item() < best_val_loss and in_final_stage:
                best_val_loss = val_loss.item()
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save checkpoint
                if output_dirs:
                    checkpoint_path = os.path.join(output_dirs['checkpoints'], f'best_model_{example_name}.pt')
                    torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': best_model_state,
                        'val_loss': best_val_loss,
                    }, checkpoint_path)
            elif in_final_stage:
                epochs_without_improvement += 1
            # Before final stage, don't count epochs without improvement
        else:
            history['val_loss'].append(None)
            history['val_bc_loss'].append(None)
            history['val_pde_loss'].append(None)
        
        history['total_loss'].append(avg_loss)
        history['bc_loss'].append(avg_bc_loss)
        history['pde_loss'].append(avg_pde_loss)
        history['test_loss'].append(test_loss.item())
        history['step'].append(global_step)
        history['weight_bc'].append(avg_weight_bc)
        history['weight_pde'].append(avg_weight_pde)
        
        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Print progress
        if use_validation:
            val_str = f"Val: {val_loss.item():.4e}"
            best_str = " *" if epoch + 1 == best_epoch else ""
        else:
            val_str = ""
            best_str = ""
        
        # Build LR string
        lr_str = f"LR={current_lr:.1e}" if scheduler is not None else ""
        
        if use_relobralo:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Steps: {global_step:6d} | "
                  f"w_bc={avg_weight_bc:.2f} w_pde={avg_weight_pde:.2f} | "
                  f"Loss: {avg_loss:.4e} | BC: {avg_bc_loss:.4e} | "
                  f"PDE: {avg_pde_loss:.4e} | {val_str} | Test: {test_loss.item():.4e} | {lr_str}{best_str}")
        else:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Steps: {global_step:6d} | α={current_alpha:.0e} | "
                  f"Loss: {avg_loss:.4e} | BC: {avg_bc_loss:.4e} | "
                  f"PDE: {avg_pde_loss:.4e} | {val_str} | Test: {test_loss.item():.4e} | {lr_str}{best_str}")
        
        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Plot PDE loss distribution with outliers
        if plot_pde_every > 0 and (epoch + 1) % plot_pde_every == 0:
            plots_dir = output_dirs['plots'] if output_dirs else '.'
            plot_pde_loss_distribution(model, x_interior_epoch, epoch + 1, example_name, output_dir=plots_dir)
        
        # Early stopping check
        if patience and use_validation and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
    
    training_time = time.time() - start_time
    
    # Restore best model if validation was used
    if use_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Compute test loss for the restored model
        model.eval()
        with torch.no_grad():
            y_pred_best = model(x_test.to(device))
            best_test_loss = torch.mean((y_pred_best - y_test.to(device))**2).item()
        model.train()
        print(f"\nRestored best model from epoch {best_epoch} (val_loss: {best_val_loss:.4e}, test_loss: {best_test_loss:.4e})")
        if output_dirs:
            print(f"Best checkpoint saved to: {os.path.join(output_dirs['checkpoints'], f'best_model_{example_name}.pt')}")
    
    # Save final model checkpoint
    if output_dirs:
        final_checkpoint_path = os.path.join(output_dirs['checkpoints'], f'final_model_{example_name}.pt')
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'history': history,
        }, final_checkpoint_path)
        print(f"Final checkpoint saved to: {final_checkpoint_path}")
    
    print(f"Training completed in {training_time:.2f}s ({global_step:,} total steps)")
    
    return history, training_time


def plot_results(model, x_test, y_test, history, title, n_grid=100):
    """Plot training history and solution comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Training curves
    ax = axes[0, 0]
    ax.semilogy(history['bc_loss'], label='Train BC', color='violet', alpha=0.7)
    ax.semilogy(history['pde_loss'], label='Train PDE', color='orange', alpha=0.7)
    
    # Plot validation loss if available
    if 'val_loss' in history and history['val_loss'][0] is not None:
        ax.semilogy(history['val_loss'], label='Val Loss', color='blue', linewidth=2)
        # Mark best epoch
        best_epoch = np.argmin(history['val_loss'])
        ax.axvline(x=best_epoch, color='blue', linestyle='--', alpha=0.5, label=f'Best (ep {best_epoch+1})')
    
    ax.semilogy(history['test_loss'], label='Test (exact)', color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # Add weight plot as inset if ReLoBRaLo was used (weights vary)
    if 'weight_bc' in history and len(set(history['weight_bc'])) > 1:
        ax_inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        ax_inset.plot(history['weight_bc'], label='w_BC', color='violet', alpha=0.7)
        ax_inset.plot(history['weight_pde'], label='w_PDE', color='orange', alpha=0.7)
        ax_inset.set_xlabel('Epoch', fontsize=8)
        ax_inset.set_ylabel('Weight', fontsize=8)
        ax_inset.legend(fontsize=7)
        ax_inset.tick_params(labelsize=7)
        ax_inset.set_title('ReLoBRaLo Weights', fontsize=8)
    
    # Add LR plot as inset if scheduler was used (LR varies)
    if 'lr' in history and len(history['lr']) > 0 and len(set(history['lr'])) > 1:
        # Position inset: bottom left if ReLoBRaLo inset is not present, else lower
        inset_pos = [0.55, 0.1, 0.4, 0.35] if 'weight_bc' in history and len(set(history['weight_bc'])) > 1 else [0.55, 0.55, 0.4, 0.4]
        ax_lr_inset = ax.inset_axes(inset_pos)
        ax_lr_inset.semilogy(history['lr'], label='LR', color='red', alpha=0.7)
        ax_lr_inset.set_xlabel('Epoch', fontsize=8)
        ax_lr_inset.set_ylabel('Learning Rate', fontsize=8)
        ax_lr_inset.tick_params(labelsize=7)
        ax_lr_inset.set_title('LR Schedule', fontsize=8)
        ax_lr_inset.grid(True, alpha=0.3)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test.to(device)).cpu()
    
    x_np = x_test.numpy()
    y_exact_np = y_test.numpy().flatten()
    y_pred_np = y_pred.numpy().flatten()
    
    # Determine domain shape for plotting
    n_pts = len(x_np)
    try:
        # Try to reshape for square domain
        side = int(np.sqrt(n_pts))
        if side * side == n_pts:
            X = x_np[:, 0].reshape(side, side)
            Y = x_np[:, 1].reshape(side, side)
            Z_exact = y_exact_np.reshape(side, side)
            Z_pred = y_pred_np.reshape(side, side)
            
            # Exact solution contour
            ax = axes[0, 1]
            c = ax.contourf(X, Y, Z_exact, levels=20, cmap='rainbow')
            plt.colorbar(c, ax=ax)
            ax.set_title('Exact Solution')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Predicted solution contour
            ax = axes[0, 2]
            c = ax.contourf(X, Y, Z_pred, levels=20, cmap='rainbow')
            plt.colorbar(c, ax=ax)
            ax.set_title('PINN Prediction')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Error contour
            ax = axes[1, 0]
            c = ax.contourf(X, Y, np.abs(Z_exact - Z_pred), levels=20, cmap='hot')
            plt.colorbar(c, ax=ax)
            ax.set_title('Absolute Error')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # 3D Exact
            ax = axes[1, 1]
            ax = fig.add_subplot(2, 3, 5, projection='3d')
            ax.plot_surface(X, Y, Z_exact, cmap='rainbow', alpha=0.8)
            ax.set_title('Exact (3D)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # 3D Predicted
            ax = fig.add_subplot(2, 3, 6, projection='3d')
            ax.plot_surface(X, Y, Z_pred, cmap='rainbow', alpha=0.8)
            ax.set_title('Predicted (3D)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            raise ValueError("Non-square grid")
    except:
        # Scatter plot for non-grid data (disc domain)
        ax = axes[0, 1]
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], c=y_exact_np, cmap='rainbow', s=1)
        plt.colorbar(sc, ax=ax)
        ax.set_title('Exact Solution')
        ax.set_aspect('equal')
        
        ax = axes[0, 2]
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], c=y_pred_np, cmap='rainbow', s=1)
        plt.colorbar(sc, ax=ax)
        ax.set_title('PINN Prediction')
        ax.set_aspect('equal')
        
        ax = axes[1, 0]
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], c=np.abs(y_exact_np - y_pred_np), 
                       cmap='hot', s=1)
        plt.colorbar(sc, ax=ax)
        ax.set_title('Absolute Error')
        ax.set_aspect('equal')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def run_example(example_name, n_epochs=10, n_interior_grid=1000, n_boundary_grid=10000,
                batch_size_bc=400, batch_size_pde=1000, save_plots=True,
                use_relobralo=False, relobralo_alpha=0.999, relobralo_temperature=1.0,
                val_split=0.0, patience=None, alpha_schedule=None,
                resample=True, activation='tanh', hidden_layers=[32, 32, 32], eta=0.0,
                scheduler_type=None, scheduler_end_factor=0.01, lr=1e-3,
                clip_residual=0.0, outlier_percentile=1.0, plot_pde_every=0,
                output_dir='outputs'):
    """
    Run one of the three examples from Section 6.1.
    
    Args:
        example_name: 'arctan', 'absolute', 'aronsson_square', or 'aronsson_disc'
        n_epochs: Number of epochs (full passes through data)
        n_interior_grid: Grid size for interior points (default 1000 -> 1M points)
        n_boundary_grid: Grid size for boundary points (default 10000 -> ~40K points)
        batch_size_bc: Boundary points per step (default 400)
        batch_size_pde: PDE points per step (default 1000)
        save_plots: Whether to save plots to files
        use_relobralo: Whether to use ReLoBRaLo for adaptive loss balancing
        relobralo_alpha: EMA decay for ReLoBRaLo (default 0.999)
        relobralo_temperature: Softmax temperature for ReLoBRaLo (default 1.0)
        val_split: Fraction of data to use for validation (0.0 = no validation)
        patience: Early stopping patience (None = no early stopping)
        alpha_schedule: Dict mapping epoch -> alpha value (None = use default)
        resample: Re-sample interior points each epoch (better coverage)
        activation: Activation function name ('tanh', 'relu', 'gelu', etc.)
        hidden_layers: List of hidden layer sizes
        eta: Regularization parameter for normalized infinity-Laplacian (0.0 = standard form)
        scheduler_type: LR scheduler type ('linear', 'cosine', 'step', 'exponential', None)
        scheduler_end_factor: End LR factor for scheduler (default 0.01)
    """
    
    print(f"\n{'='*60}")
    print(f"Running Example: {example_name}")
    print(f"{'='*60}\n")
    
    # Create output directories
    output_dirs = create_output_dirs(output_dir)
    print(f"Output directories:")
    print(f"  Plots: {output_dirs['plots']}")
    print(f"  Checkpoints: {output_dirs['checkpoints']}\n")
    
    # Configure based on example
    # Store domain info for potential re-sampling
    domain_type = None
    domain_params = None
    
    if example_name == 'arctan':
        # Section 6.1.1: Arctan Example
        f_exact = arctan_solution
        domain_type = 'square'
        domain_params = {'x_min': 0.01, 'x_max': 1.0, 'y_min': 0.01, 'y_max': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_square_domain_data_full(
            **domain_params, n_interior_grid=n_interior_grid, n_boundary_grid=n_boundary_grid,
            f_exact=f_exact, seed=SEED
        )
        title = "Arctan Example: u(x,y) = arctan(y/x)"
        
    elif example_name == 'absolute':
        # Section 6.1.2: Absolute Example - |x| - |y| is not differentiable at x=0, y=0
        f_exact = absolute_solution
        domain_type = 'square'
        domain_params = {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_square_domain_data_full(
            **domain_params, n_interior_grid=n_interior_grid, n_boundary_grid=n_boundary_grid,
            f_exact=f_exact, seed=SEED
        )
        title = "Absolute Example: u(x,y) = |x| - |y|"
        
    elif example_name == 'aronsson_square':
        # Section 6.1.3: Aronsson on square - |x|^(4/3) - |y|^(4/3) has singular derivatives at axes
        f_exact = aronsson_solution
        domain_type = 'square'
        domain_params = {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_square_domain_data_full(
            **domain_params, n_interior_grid=n_interior_grid, n_boundary_grid=n_boundary_grid,
            f_exact=f_exact, seed=SEED
        )
        title = "Aronsson Example (Square): u = |x|^(4/3) - |y|^(4/3)"
        
    elif example_name == 'aronsson_disc':
        # Section 6.1.3: Aronsson on unit disc - |x|^(4/3) - |y|^(4/3) has singular derivatives at axes
        f_exact = aronsson_solution
        domain_type = 'disc'
        domain_params = {'radius': 1.0}
        x_bc, y_bc, x_interior, x_test, y_test = generate_disc_domain_data_full(
            radius=1.0, n_boundary=n_boundary_grid, n_interior_grid=n_interior_grid,
            f_exact=f_exact, seed=SEED
        )
        title = "Aronsson Example (Disc): u = |x|^(4/3) - |y|^(4/3)"
        
    else:
        raise ValueError(f"Unknown example: {example_name}")
    
    # Create interior sampler if re-sampling is enabled
    interior_sampler = None
    if resample:
        # Number of points to sample each epoch (approximate, may vary with soft exclusion)
        n_interior_target = len(x_interior)  # Match initial count
        interior_sampler = InteriorSampler(
            domain_type=domain_type,
            domain_params=domain_params,
            n_points=n_interior_target
        )
        print(f"Re-sampling enabled: ~{n_interior_target:,} interior points per epoch")
    
    # =========================================================================
    # Split data into train/validation if val_split > 0
    # =========================================================================
    x_bc_val, y_bc_val, x_interior_val = None, None, None
    
    if val_split > 0:
        print(f"\nSplitting data: {1-val_split:.0%} train, {val_split:.0%} validation")
        train_ratio = 1 - val_split
        
        # Split boundary data
        x_bc_train, x_bc_val, y_bc_train, y_bc_val = split_data(
            x_bc, y_bc, train_ratio=train_ratio, seed=SEED
        )
        
        # Split interior data
        x_interior_train, x_interior_val, _, _ = split_data(
            x_interior, None, train_ratio=train_ratio, seed=SEED + 1
        )
        
        print(f"  BC: {len(x_bc_train):,} train, {len(x_bc_val):,} val")
        print(f"  Interior: {len(x_interior_train):,} train, {len(x_interior_val):,} val")
        
        # Use split data
        x_bc, y_bc = x_bc_train, y_bc_train
        x_interior = x_interior_train
    
    # =========================================================================
    # Create and train model
    # =========================================================================
    model = InfinityLaplacianPINN(hidden_layers=hidden_layers, activation=activation, eta=eta,
                                   clip_residual=clip_residual, outlier_percentile=outlier_percentile)
    model.to(device)
    print(f"\nModel architecture:")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Activation: {activation}")
    if eta > 0:
        print(f"  Regularization η: {eta} (normalized infinity-Laplacian)")
    else:
        print(f"  Regularization η: 0 (standard infinity-Laplacian)")
    if clip_residual > 0:
        print(f"  Residual clipping: [-{clip_residual}, {clip_residual}]")
    if outlier_percentile > 0:
        print(f"  Outlier removal: top {outlier_percentile}% excluded")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history, training_time = train_pinn(
        model, x_bc, y_bc, x_interior, x_test, y_test,
        n_epochs=n_epochs, batch_size_bc=batch_size_bc, batch_size_pde=batch_size_pde,
        lr=lr, alpha_schedule=alpha_schedule, use_relobralo=use_relobralo, 
        relobralo_alpha=relobralo_alpha, relobralo_temperature=relobralo_temperature,
        x_bc_val=x_bc_val, y_bc_val=y_bc_val, x_interior_val=x_interior_val,
        patience=patience, interior_sampler=interior_sampler,
        scheduler_type=scheduler_type, scheduler_end_factor=scheduler_end_factor,
        plot_pde_every=plot_pde_every, example_name=example_name,
        output_dirs=output_dirs
    )
    
    # Report final metrics
    final_test_mse = history['test_loss'][-1]
    print(f"\nFinal Test MSE: {final_test_mse:.4e}")
    
    # Inference time measurement
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(x_test.to(device))
        inference_time = (time.time() - start) / 100
    print(f"Inference time (full domain): {inference_time:.4e}s")
    
    # Plot results
    fig = plot_results(model, x_test, y_test, history, title)
    
    if save_plots:
        results_path = os.path.join(output_dirs['plots'], f'results_{example_name}.png')
        fig.savefig(results_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {results_path}")
    
    plt.show()
    
    return model, history


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment 6.1: 2D Infinity-Laplacian with PINNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--example', type=str, default='aronsson_square',
                        choices=['arctan', 'absolute', 'aronsson_square', 'aronsson_disc', 'all'],
                        help='Which example to run')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (full passes through data)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--interior-grid', type=int, default=1000,
                        help='Interior grid size (N -> N*N points, default 1000 -> 1M points)')
    parser.add_argument('--boundary-grid', type=int, default=10000,
                        help='Boundary grid size (default 10000 -> ~40K points)')
    parser.add_argument('--batch-bc', type=int, default=400,
                        help='Boundary batch size per step')
    parser.add_argument('--batch-pde', type=int, default=1000,
                        help='PDE/collocation batch size per step')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'leaky_relu', 'elu', 'gelu', 'softplus', 'silu', 'sigmoid'],
                        help='Activation function (default: tanh)')
    parser.add_argument('--hidden-layers', type=str, default='32,32,32',
                        help='Hidden layer sizes as comma-separated values (default: 32,32,32)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='Regularization η for normalized infinity-Laplacian: Δ_∞u/(η²+|∇u|²). Default 0.0 uses standard form. Paper uses η=1e-5 for p-Laplacian.')
    parser.add_argument('--clip-residual', type=float, default=0.0,
                        help='Clip PDE residual to [-clip, clip] before loss. Prevents extreme values from dominating. Default 0.0 = no clipping.')
    parser.add_argument('--outlier-percentile', type=float, default=1.0,
                        help='Exclude top X%% of extreme PDE losses. Default 1.0 = exclude top 1%%. Set to 0 to disable.')
    parser.add_argument('--plot-pde-every', type=int, default=0,
                        help='Plot PDE loss distribution every N epochs with outliers highlighted in red. Default 0 = disabled.')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Base directory for outputs (plots, checkpoints). Default: outputs')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots')
    
    # Alpha schedule options
    parser.add_argument('--alpha', type=float, default=None,
                        help='Fixed alpha value for PDE loss weight (overrides schedule)')
    parser.add_argument('--alpha-schedule', type=str, default=None,
                        help='Alpha schedule as "epoch:value,epoch:value,..." e.g. "0:1e-4,100:1e-3,500:1e-2"')
    
    # ReLoBRaLo options
    parser.add_argument('--relobralo', action='store_true',
                        help='Use ReLoBRaLo for adaptive loss balancing')
    parser.add_argument('--relobralo-alpha', type=float, default=0.999,
                        help='ReLoBRaLo EMA decay (default: 0.999)')
    parser.add_argument('--relobralo-temperature', type=float, default=1.0,
                        help='ReLoBRaLo softmax temperature (default: 1.0)')
    
    # Validation and early stopping options
    parser.add_argument('--val-split', type=float, default=0.0,
                        help='Fraction of data for validation (0.0 = no validation, 0.2 = 20%% val)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (epochs without improvement)')
    
    parser.add_argument('--no-resample', action='store_true',
                        help='Disable re-sampling of interior points (use fixed points from initial setup)')
    
    # Learning rate scheduler options
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['linear', 'cosine', 'step', 'exponential'],
                        help='LR scheduler type (default: none). linear=decay to end_factor, cosine=cosine annealing, step=decay every 1/3 epochs, exponential=exponential decay')
    parser.add_argument('--scheduler-end-factor', type=float, default=0.01,
                        help='End factor for LR scheduler: final_lr = initial_lr * end_factor (default: 0.01)')
    
    args = parser.parse_args()
    
    # Parse alpha schedule
    alpha_schedule = None
    if args.alpha is not None:
        # Fixed alpha value
        alpha_schedule = {0: args.alpha}
    elif args.alpha_schedule is not None:
        # Parse schedule string: "0:1e-4,100:1e-3,500:1e-2"
        alpha_schedule = {}
        for item in args.alpha_schedule.split(','):
            epoch_str, value_str = item.strip().split(':')
            alpha_schedule[int(epoch_str)] = float(value_str)
    # If neither specified, train_pinn will use its default schedule
    
    # Parse hidden layers
    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]
    
    print(f"Configuration:")
    print(f"  Interior grid: {args.interior_grid}x{args.interior_grid} = {args.interior_grid**2:,} points")
    print(f"  Boundary grid: ~{4*args.boundary_grid:,} points")
    print(f"  Batch sizes: BC={args.batch_bc}, PDE={args.batch_pde}")
    print(f"  Network: {hidden_layers} with {args.activation} activation")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr:.1e}")
    if args.relobralo:
        print(f"  Loss balancing: ReLoBRaLo (α={args.relobralo_alpha}, T={args.relobralo_temperature})")
    elif alpha_schedule:
        print(f"  Alpha schedule: {alpha_schedule}")
    else:
        print(f"  Alpha schedule: default (0:1e-4, 100:1e-3, 500:1e-2)")
    if args.val_split > 0:
        print(f"  Validation split: {args.val_split:.0%} (BC only)")
        if args.patience:
            print(f"  Early stopping patience: {args.patience}")
    else:
        print(f"  Validation: Disabled (using final model)")
    if args.no_resample:
        print(f"  Interior points: FIXED (no re-sampling)")
    else:
        print(f"  Interior points: RE-SAMPLE each epoch (default)")
    if args.scheduler:
        print(f"  LR scheduler: {args.scheduler} (end_factor={args.scheduler_end_factor})")
    else:
        print(f"  LR scheduler: None (constant LR)")
    
    resample = not args.no_resample
    
    if args.example == 'all':
        for ex in ['arctan', 'absolute', 'aronsson_square', 'aronsson_disc']:
            run_example(ex, n_epochs=args.epochs, 
                       n_interior_grid=args.interior_grid,
                       n_boundary_grid=args.boundary_grid,
                       batch_size_bc=args.batch_bc,
                       batch_size_pde=args.batch_pde,
                       save_plots=not args.no_save,
                       use_relobralo=args.relobralo,
                       relobralo_alpha=args.relobralo_alpha,
                       relobralo_temperature=args.relobralo_temperature,
                       val_split=args.val_split,
                       patience=args.patience,
                       alpha_schedule=alpha_schedule,
                       resample=resample,
                       activation=args.activation,
                       hidden_layers=hidden_layers,
                       eta=args.eta,
                       scheduler_type=args.scheduler,
                       scheduler_end_factor=args.scheduler_end_factor,
                       lr=args.lr,
                       clip_residual=args.clip_residual,
                       outlier_percentile=args.outlier_percentile)
    else:
        run_example(args.example, n_epochs=args.epochs,
                   n_interior_grid=args.interior_grid,
                   n_boundary_grid=args.boundary_grid,
                   batch_size_bc=args.batch_bc,
                   batch_size_pde=args.batch_pde,
                   save_plots=not args.no_save,
                   use_relobralo=args.relobralo,
                   relobralo_alpha=args.relobralo_alpha,
                   relobralo_temperature=args.relobralo_temperature,
                   val_split=args.val_split,
                   patience=args.patience,
                   alpha_schedule=alpha_schedule,
                   resample=resample,
                   activation=args.activation,
                   hidden_layers=hidden_layers,
                   eta=args.eta,
                   scheduler_type=args.scheduler,
                   scheduler_end_factor=args.scheduler_end_factor,
                   lr=args.lr,
                   clip_residual=args.clip_residual,
                   outlier_percentile=args.outlier_percentile,
                   plot_pde_every=args.plot_pde_every,
                   output_dir=args.output_dir)

