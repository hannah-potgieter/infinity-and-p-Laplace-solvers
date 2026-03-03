"""
Experiment 6.2: 2D p-Laplacian Equation - Aronsson Example (Homogeneous, Dirichlet)

This script implements PINNs for solving the p-Laplacian equation:
    Δ_p u = div(|∇u|^(p-2) ∇u) = 0  in Ω
    u = g  on ∂Ω

Expanded form:
    |∇u|^(p-4) * ((p-1)*u_x²*u_xx + (p-1)*u_y²*u_yy + (p-2)*u_x*u_y*(u_xy + u_yx) 
                   + u_x²*u_yy + u_y²*u_xx) = 0

Aronsson Example: u(x,y) = |x|^(4/3) - |y|^(4/3) on [-1,1]² or unit disc

Features:
    1. Simple training: Train for a single fixed p value
    2. Iterative training: Start from lower p, progressively increase to target p
       (This helps with numerical stability for large p values)

Reference: "Solving p-Laplacian and infinity-Laplacian with Deep Learning"
Authors: Tak Shing Au Yeung, Ka Chun Cheung, Hannah Potgieter, Steven J. Ruuth, Simon See

Usage:
    # Simple training with p=10
    python experiment_6_2.py --mode simple --p 10 --epochs 100
    
    # Iterative training from p=2 to p=100
    python experiment_6_2.py --mode iterative --p-start 2 --p-end 100 --p-steps 10 --epochs 50
    
    # Run on disc domain
    python experiment_6_2.py --domain disc --mode simple --p 20
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
from typing import List, Optional, Tuple, Dict
import copy


def create_output_dirs(base_dir='outputs'):
    """Create output directories for plots and checkpoints."""
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
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
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
# Activation Function Registry
# ============================================================================

ACTIVATIONS = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'softplus': nn.Softplus,
    'silu': nn.SiLU,
    'sigmoid': nn.Sigmoid,
}


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()


# ============================================================================
# p-Laplacian PINN Model
# ============================================================================

class PLaplacianPINN(nn.Module):
    """
    Physics-Informed Neural Network for solving the p-Laplacian equation.
    
    Architecture:
    - Configurable hidden layers
    - Configurable activation function
    - Single output for u(x,y)
    
    The p-Laplacian PDE:
        Δ_p u = div(|∇u|^(p-2) ∇u) = 0
        
    Expanded in 2D:
        |∇u|^(p-4) * ((p-1)*u_x²*u_xx + (p-1)*u_y²*u_yy + (p-2)*u_x*u_y*(u_xy + u_yx) 
                       + u_x²*u_yy + u_y²*u_xx) = 0
    """
    
    def __init__(self, hidden_layers=[32, 32, 32], activation='tanh', 
                 eta=1e-5, clip_residual=0.0):
        super().__init__()
        
        self.activation = get_activation(activation)
        self.activation_name = activation
        self.loss_function = nn.MSELoss(reduction='mean')
        self.eta = eta  # Regularization for gradient norm (prevent |∇u|^(p-4) blowup)
        self.clip_residual = clip_residual
        
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
    
    def compute_p_laplacian(self, x, p):
        """
        Compute the p-Laplacian of the network output.
        
        p-Laplacian:
            Δ_p u = |∇u|^(p-4) * ((p-1)*u_x²*u_xx + (p-1)*u_y²*u_yy 
                                  + (p-2)*u_x*u_y*(u_xy + u_yx) 
                                  + u_x²*u_yy + u_y²*u_xx)
        
        For p=2: reduces to standard Laplacian (u_xx + u_yy)
        For p→∞: reduces to infinity-Laplacian (u_x²*u_xx + 2*u_x*u_y*u_xy + u_y²*u_yy)
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
        u_yx = grad_uy[:, [0]]
        u_yy = grad_uy[:, [1]]
        
        # Gradient magnitude squared
        grad_norm_sq = u_x**2 + u_y**2
        
        # Regularized gradient norm for stability
        # gamma = (eta² + |∇u|²)^((p-4)/2)
        gamma = (self.eta**2 + grad_norm_sq)**((p - 4) / 2)
        
        # p-Laplacian residual
        # Δ_p u = gamma * ((p-1)*u_x²*u_xx + (p-1)*u_y²*u_yy 
        #                  + (p-2)*u_x*u_y*(u_xy + u_yx) + u_x²*u_yy + u_y²*u_xx)
        delta_p = gamma * (
            (p - 1) * u_x**2 * u_xx + 
            (p - 1) * u_y**2 * u_yy + 
            (p - 2) * u_x * u_y * (u_xy + u_yx) + 
            u_x**2 * u_yy + 
            u_y**2 * u_xx
        )
        
        return delta_p
    
    def loss_boundary(self, x_bc, y_bc):
        """Data-driven loss: MSE on boundary conditions."""
        return self.loss_function(self.forward(x_bc), y_bc)
    
    def loss_pde(self, x_pde, p):
        """PDE loss: Residual of the p-Laplacian equation."""
        delta_p = self.compute_p_laplacian(x_pde, p)
        
        if self.clip_residual > 0:
            delta_p = torch.clamp(delta_p, -self.clip_residual, self.clip_residual)
        
        zeros = torch.zeros_like(delta_p)
        return self.loss_function(delta_p, zeros)
    
    def total_loss(self, x_bc, y_bc, x_pde, p, alpha):
        """Total loss = MSE_boundary + alpha * MSE_PDE"""
        loss_bc = self.loss_boundary(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde, p)
        return loss_bc + alpha * loss_pde, loss_bc, loss_pde


# ============================================================================
# Exact Solution
# ============================================================================

def aronsson_solution(x, y):
    """Aronsson example: u(x,y) = |x|^(4/3) - |y|^(4/3)"""
    return torch.abs(x)**(4/3) - torch.abs(y)**(4/3)


# ============================================================================
# Data Generation
# ============================================================================

def generate_square_domain_data(x_min, x_max, y_min, y_max, 
                                 n_interior_grid=100, n_boundary_grid=1000, 
                                 f_exact=None, seed=1234):
    """Generate training data for a square domain."""
    generator = torch.Generator().manual_seed(seed)
    
    # Boundary points
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
        torch.cat([top_x, top_y], dim=1)
    ], dim=0)
    
    y_bc = f_exact(x_bc[:, 0], x_bc[:, 1]).view(-1, 1)
    
    # Shuffle boundary
    perm = torch.randperm(x_bc.shape[0], generator=generator)
    x_bc = x_bc[perm]
    y_bc = y_bc[perm]
    
    # Interior points (grid)
    eps = 1e-6
    x_1d = torch.linspace(x_min + eps, x_max - eps, n_interior_grid)
    y_1d = torch.linspace(y_min + eps, y_max - eps, n_interior_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    x_interior = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Shuffle interior
    perm = torch.randperm(x_interior.shape[0], generator=generator)
    x_interior = x_interior[perm]
    
    # Test grid
    n_test = 100
    x_test_1d = torch.linspace(x_min, x_max, n_test)
    y_test_1d = torch.linspace(y_min, y_max, n_test)
    X, Y = torch.meshgrid(x_test_1d, y_test_1d, indexing='ij')
    x_test = torch.stack([X.flatten(), Y.flatten()], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1]).view(-1, 1)
    
    print(f"Generated square domain data:")
    print(f"  Boundary points: {x_bc.shape[0]:,}")
    print(f"  Interior points: {x_interior.shape[0]:,}")
    print(f"  Test points: {x_test.shape[0]:,}")
    
    return x_bc, y_bc, x_interior, x_test, y_test


def generate_disc_domain_data(radius, n_boundary=1000, n_interior_grid=100, 
                               f_exact=None, seed=1234):
    """Generate training data for a disc domain."""
    generator = torch.Generator().manual_seed(seed)
    
    # Boundary points (circle)
    theta = torch.linspace(0, 2 * np.pi, n_boundary + 1)[:-1]
    x_bc = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    y_bc = f_exact(x_bc[:, 0], x_bc[:, 1]).view(-1, 1)
    
    # Shuffle boundary
    perm = torch.randperm(x_bc.shape[0], generator=generator)
    x_bc = x_bc[perm]
    y_bc = y_bc[perm]
    
    # Interior points (grid filtered to disc)
    eps = 1e-6
    x_1d = torch.linspace(-radius + eps, radius - eps, n_interior_grid)
    y_1d = torch.linspace(-radius + eps, radius - eps, n_interior_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    
    mask = (X**2 + Y**2) < (radius - eps)**2
    x_interior = torch.stack([X[mask].flatten(), Y[mask].flatten()], dim=1)
    
    # Shuffle interior
    perm = torch.randperm(x_interior.shape[0], generator=generator)
    x_interior = x_interior[perm]
    
    # Test grid
    n_test = 100
    x_test_1d = torch.linspace(-radius, radius, n_test)
    y_test_1d = torch.linspace(-radius, radius, n_test)
    X, Y = torch.meshgrid(x_test_1d, y_test_1d, indexing='ij')
    mask = X**2 + Y**2 <= radius**2
    x_test = torch.stack([X[mask].flatten(), Y[mask].flatten()], dim=1)
    y_test = f_exact(x_test[:, 0], x_test[:, 1]).view(-1, 1)
    
    print(f"Generated disc domain data:")
    print(f"  Boundary points: {x_bc.shape[0]:,}")
    print(f"  Interior points: {x_interior.shape[0]:,}")
    print(f"  Test points: {x_test.shape[0]:,}")
    
    return x_bc, y_bc, x_interior, x_test, y_test


# ============================================================================
# Training Functions
# ============================================================================

def train_pinn_simple(model, x_bc, y_bc, x_interior, x_test, y_test, 
                      p, n_epochs=100, batch_size_bc=400, batch_size_pde=1000,
                      lr=1e-3, alpha=1.0, output_dirs=None):
    """
    Simple training for a fixed p value.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Move data to device
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Create DataLoaders
    bc_dataset = TensorDataset(x_bc, y_bc)
    bc_loader = DataLoader(bc_dataset, batch_size=batch_size_bc, shuffle=True, 
                          drop_last=False, generator=torch.Generator().manual_seed(SEED))
    
    interior_dataset = TensorDataset(x_interior)
    interior_loader = DataLoader(interior_dataset, batch_size=batch_size_pde, shuffle=True,
                                 drop_last=False, generator=torch.Generator().manual_seed(SEED))
    
    history = {
        'total_loss': [],
        'bc_loss': [],
        'pde_loss': [],
        'test_loss': [],
        'p': []
    }
    
    n_bc_batches = len(bc_loader)
    n_pde_batches = len(interior_loader)
    steps_per_epoch = max(n_bc_batches, n_pde_batches)
    
    print(f"\nSimple Training (p={p}):")
    print(f"  Epochs: {n_epochs}, Steps/epoch: {steps_per_epoch}")
    print(f"  Alpha (PDE weight): {alpha}")
    print()
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        bc_iter = iter(bc_loader)
        pde_iter = iter(interior_loader)
        
        epoch_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_pde_loss = 0.0
        n_steps = 0
        
        for step in range(steps_per_epoch):
            try:
                x_bc_batch, y_bc_batch = next(bc_iter)
            except StopIteration:
                bc_iter = iter(bc_loader)
                x_bc_batch, y_bc_batch = next(bc_iter)
            
            try:
                (x_pde_batch,) = next(pde_iter)
            except StopIteration:
                pde_iter = iter(interior_loader)
                (x_pde_batch,) = next(pde_iter)
            
            x_bc_batch = x_bc_batch.to(device)
            y_bc_batch = y_bc_batch.to(device)
            x_pde_batch = x_pde_batch.to(device)
            
            optimizer.zero_grad()
            loss, loss_bc, loss_pde = model.total_loss(x_bc_batch, y_bc_batch, x_pde_batch, p, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_bc_loss += loss_bc.item()
            epoch_pde_loss += loss_pde.item()
            n_steps += 1
        
        # Compute test loss
        model.eval()
        with torch.no_grad():
            test_loss = model.loss_boundary(x_test, y_test)
        model.train()
        
        avg_loss = epoch_loss / n_steps
        avg_bc_loss = epoch_bc_loss / n_steps
        avg_pde_loss = epoch_pde_loss / n_steps
        
        history['total_loss'].append(avg_loss)
        history['bc_loss'].append(avg_bc_loss)
        history['pde_loss'].append(avg_pde_loss)
        history['test_loss'].append(test_loss.item())
        history['p'].append(p)
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch+1:4d}/{n_epochs} | p={p:3d} | "
                  f"Loss: {avg_loss:.4e} | BC: {avg_bc_loss:.4e} | "
                  f"PDE: {avg_pde_loss:.4e} | Test: {test_loss.item():.4e}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s")
    
    return history, training_time


def train_pinn_iterative(model, x_bc, y_bc, x_interior, x_test, y_test, 
                         p_start, p_end, p_steps, epochs_per_p=50,
                         batch_size_bc=400, batch_size_pde=1000,
                         lr=1e-3, alpha=1.0, output_dirs=None):
    """
    Iterative training: progressively increase p from p_start to p_end.
    
    This approach helps with numerical stability for large p values by
    starting with smaller p (where the PDE is more well-behaved) and
    gradually increasing.
    """
    # Generate p schedule
    if p_steps == 1:
        p_values = [p_end]
    else:
        p_values = np.linspace(p_start, p_end, p_steps).astype(int).tolist()
        # Ensure p_end is included
        if p_values[-1] != p_end:
            p_values[-1] = p_end
    
    print(f"\nIterative Training Schedule:")
    print(f"  p values: {p_values}")
    print(f"  Epochs per p: {epochs_per_p}")
    print()
    
    history = {
        'total_loss': [],
        'bc_loss': [],
        'pde_loss': [],
        'test_loss': [],
        'p': [],
        'p_schedule': p_values
    }
    
    # Move data to device
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Create DataLoaders
    bc_dataset = TensorDataset(x_bc, y_bc)
    bc_loader = DataLoader(bc_dataset, batch_size=batch_size_bc, shuffle=True, 
                          drop_last=False, generator=torch.Generator().manual_seed(SEED))
    
    interior_dataset = TensorDataset(x_interior)
    interior_loader = DataLoader(interior_dataset, batch_size=batch_size_pde, shuffle=True,
                                 drop_last=False, generator=torch.Generator().manual_seed(SEED))
    
    n_bc_batches = len(bc_loader)
    n_pde_batches = len(interior_loader)
    steps_per_epoch = max(n_bc_batches, n_pde_batches)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    global_epoch = 0
    
    for p_idx, p in enumerate(p_values):
        print(f"\n{'='*60}")
        print(f"Stage {p_idx + 1}/{len(p_values)}: Training with p={p}")
        print(f"{'='*60}")
        
        # Optionally reduce learning rate for later stages
        if p_idx > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (0.5 ** p_idx)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate adjusted to: {current_lr:.2e}")
        
        for epoch in range(epochs_per_p):
            bc_iter = iter(bc_loader)
            pde_iter = iter(interior_loader)
            
            epoch_loss = 0.0
            epoch_bc_loss = 0.0
            epoch_pde_loss = 0.0
            n_steps = 0
            
            for step in range(steps_per_epoch):
                try:
                    x_bc_batch, y_bc_batch = next(bc_iter)
                except StopIteration:
                    bc_iter = iter(bc_loader)
                    x_bc_batch, y_bc_batch = next(bc_iter)
                
                try:
                    (x_pde_batch,) = next(pde_iter)
                except StopIteration:
                    pde_iter = iter(interior_loader)
                    (x_pde_batch,) = next(pde_iter)
                
                x_bc_batch = x_bc_batch.to(device)
                y_bc_batch = y_bc_batch.to(device)
                x_pde_batch = x_pde_batch.to(device)
                
                optimizer.zero_grad()
                loss, loss_bc, loss_pde = model.total_loss(x_bc_batch, y_bc_batch, x_pde_batch, p, alpha)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_bc_loss += loss_bc.item()
                epoch_pde_loss += loss_pde.item()
                n_steps += 1
            
            # Compute test loss
            model.eval()
            with torch.no_grad():
                test_loss = model.loss_boundary(x_test, y_test)
            model.train()
            
            avg_loss = epoch_loss / n_steps
            avg_bc_loss = epoch_bc_loss / n_steps
            avg_pde_loss = epoch_pde_loss / n_steps
            
            history['total_loss'].append(avg_loss)
            history['bc_loss'].append(avg_bc_loss)
            history['pde_loss'].append(avg_pde_loss)
            history['test_loss'].append(test_loss.item())
            history['p'].append(p)
            
            if epoch % 10 == 0 or epoch == epochs_per_p - 1:
                print(f"Epoch {epoch+1:4d}/{epochs_per_p} | p={p:3d} | "
                      f"Loss: {avg_loss:.4e} | BC: {avg_bc_loss:.4e} | "
                      f"PDE: {avg_pde_loss:.4e} | Test: {test_loss.item():.4e}")
            
            global_epoch += 1
    
    training_time = time.time() - start_time
    print(f"\nIterative training completed in {training_time:.2f}s")
    print(f"Total epochs: {global_epoch}")
    
    return history, training_time


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_results(model, x_test, y_test, history, title, n_grid=100, output_path=None):
    """Plot training history and solution comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Training curves
    ax = axes[0, 0]
    ax.semilogy(history['bc_loss'], label='BC Loss', color='violet', alpha=0.7)
    ax.semilogy(history['pde_loss'], label='PDE Loss', color='orange', alpha=0.7)
    ax.semilogy(history['test_loss'], label='Test Loss', color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # p value over epochs (for iterative training)
    if 'p_schedule' in history and len(history['p_schedule']) > 1:
        ax_p = ax.twinx()
        ax_p.plot(history['p'], color='red', linestyle='--', alpha=0.5, label='p value')
        ax_p.set_ylabel('p', color='red')
        ax_p.tick_params(axis='y', labelcolor='red')
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test.to(device)).cpu()
    
    x_np = x_test.numpy()
    y_exact_np = y_test.numpy().flatten()
    y_pred_np = y_pred.numpy().flatten()
    
    # Try to reshape for grid plotting
    n_pts = len(x_np)
    try:
        side = int(np.sqrt(n_pts))
        if side * side == n_pts:
            X = x_np[:, 0].reshape(side, side)
            Y = x_np[:, 1].reshape(side, side)
            Z_exact = y_exact_np.reshape(side, side)
            Z_pred = y_pred_np.reshape(side, side)
            
            # Exact solution
            ax = axes[0, 1]
            c = ax.contourf(X, Y, Z_exact, levels=20, cmap='rainbow')
            plt.colorbar(c, ax=ax)
            ax.set_title('Exact Solution')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Predicted solution
            ax = axes[0, 2]
            c = ax.contourf(X, Y, Z_pred, levels=20, cmap='rainbow')
            plt.colorbar(c, ax=ax)
            ax.set_title('PINN Prediction')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Error
            ax = axes[1, 0]
            c = ax.contourf(X, Y, np.abs(Z_exact - Z_pred), levels=20, cmap='hot')
            plt.colorbar(c, ax=ax)
            ax.set_title('Absolute Error')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # 3D Exact
            ax = fig.add_subplot(2, 3, 5, projection='3d')
            ax.plot_surface(X, Y, Z_exact, cmap='rainbow', alpha=0.8)
            ax.set_title('Exact (3D)')
            
            # 3D Predicted
            ax = fig.add_subplot(2, 3, 6, projection='3d')
            ax.plot_surface(X, Y, Z_pred, cmap='rainbow', alpha=0.8)
            ax.set_title('Predicted (3D)')
        else:
            raise ValueError("Non-square grid")
    except:
        # Scatter plot for non-grid data
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
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    return fig


# ============================================================================
# Main Run Function
# ============================================================================

def run_experiment(domain='square', mode='simple', p=10, p_start=2, p_end=100, p_steps=10,
                   epochs=100, epochs_per_p=50, hidden_layers=[32, 32, 32], activation='tanh',
                   lr=1e-3, alpha=1.0, eta=1e-5, n_interior=100, n_boundary=1000,
                   batch_size_bc=400, batch_size_pde=1000, output_dir='outputs/experiment_6_2'):
    """Run the p-Laplacian experiment."""
    
    print(f"\n{'='*60}")
    print(f"Experiment 6.2: p-Laplacian Aronsson Example")
    print(f"{'='*60}\n")
    
    # Create output directories
    output_dirs = create_output_dirs(output_dir)
    
    # Generate data
    f_exact = aronsson_solution
    
    if domain == 'square':
        x_bc, y_bc, x_interior, x_test, y_test = generate_square_domain_data(
            -1.0, 1.0, -1.0, 1.0, n_interior_grid=n_interior, n_boundary_grid=n_boundary,
            f_exact=f_exact, seed=SEED
        )
        domain_name = "Square [-1,1]²"
    else:
        x_bc, y_bc, x_interior, x_test, y_test = generate_disc_domain_data(
            radius=1.0, n_boundary=n_boundary, n_interior_grid=n_interior,
            f_exact=f_exact, seed=SEED
        )
        domain_name = "Unit Disc"
    
    # Create model
    model = PLaplacianPINN(hidden_layers=hidden_layers, activation=activation, eta=eta)
    model.to(device)
    
    print(f"\nModel Configuration:")
    print(f"  Domain: {domain_name}")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Activation: {activation}")
    print(f"  Eta (regularization): {eta}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    if mode == 'simple':
        title = f"p-Laplacian (p={p}) - Aronsson on {domain_name}"
        history, training_time = train_pinn_simple(
            model, x_bc, y_bc, x_interior, x_test, y_test,
            p=p, n_epochs=epochs, batch_size_bc=batch_size_bc, batch_size_pde=batch_size_pde,
            lr=lr, alpha=alpha, output_dirs=output_dirs
        )
        model_name = f'plaplacian_p{p}_{domain}'
    else:
        title = f"p-Laplacian (p: {p_start}→{p_end}) - Aronsson on {domain_name}"
        history, training_time = train_pinn_iterative(
            model, x_bc, y_bc, x_interior, x_test, y_test,
            p_start=p_start, p_end=p_end, p_steps=p_steps, epochs_per_p=epochs_per_p,
            batch_size_bc=batch_size_bc, batch_size_pde=batch_size_pde,
            lr=lr, alpha=alpha, output_dirs=output_dirs
        )
        model_name = f'plaplacian_p{p_start}to{p_end}_{domain}'
    
    # Report final metrics
    final_test_mse = history['test_loss'][-1]
    print(f"\nFinal Test MSE: {final_test_mse:.4e}")
    
    # Plot results
    plot_path = os.path.join(output_dirs['plots'], f'results_{model_name}.png')
    fig = plot_results(model, x_test, y_test, history, title, output_path=plot_path)
    plt.show()
    
    # Save model
    model_path = os.path.join(output_dirs['checkpoints'], f'{model_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'hidden_layers': hidden_layers,
        'activation': activation,
        'eta': eta,
        'domain': domain,
        'mode': mode,
        'p': p if mode == 'simple' else (p_start, p_end, p_steps),
    }, model_path)
    print(f"Model saved to {model_path}")
    
    return model, history


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment 6.2: p-Laplacian Aronsson Example with PINNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Domain options
    parser.add_argument('--domain', type=str, default='square',
                        choices=['square', 'disc'],
                        help='Domain type')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='simple',
                        choices=['simple', 'iterative'],
                        help='Training mode: simple (fixed p) or iterative (progressive p)')
    
    # p-Laplacian parameter
    parser.add_argument('--p', type=int, default=10,
                        help='p value for simple mode')
    parser.add_argument('--p-start', type=int, default=2,
                        help='Starting p value for iterative mode')
    parser.add_argument('--p-end', type=int, default=100,
                        help='Ending p value for iterative mode')
    parser.add_argument('--p-steps', type=int, default=10,
                        help='Number of p steps for iterative mode')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (for simple mode)')
    parser.add_argument('--epochs-per-p', type=int, default=50,
                        help='Epochs per p value (for iterative mode)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='PDE loss weight')
    parser.add_argument('--eta', type=float, default=1e-5,
                        help='Gradient regularization parameter')
    
    # Network architecture
    parser.add_argument('--hidden-layers', type=str, default='32,32,32',
                        help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'gelu', 'silu', 'elu'],
                        help='Activation function')
    
    # Data parameters
    parser.add_argument('--n-interior', type=int, default=100,
                        help='Interior grid size (N -> N*N points)')
    parser.add_argument('--n-boundary', type=int, default=1000,
                        help='Number of boundary points per edge')
    parser.add_argument('--batch-bc', type=int, default=400,
                        help='Boundary batch size')
    parser.add_argument('--batch-pde', type=int, default=1000,
                        help='PDE batch size')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/experiment_6_2',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    SEED = args.seed
    set_seed(SEED)
    
    # Parse hidden layers
    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]
    
    print(f"Configuration:")
    print(f"  Domain: {args.domain}")
    print(f"  Mode: {args.mode}")
    if args.mode == 'simple':
        print(f"  p: {args.p}")
        print(f"  Epochs: {args.epochs}")
    else:
        print(f"  p range: {args.p_start} → {args.p_end} ({args.p_steps} steps)")
        print(f"  Epochs per p: {args.epochs_per_p}")
    print(f"  Network: {hidden_layers} with {args.activation}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Alpha (PDE weight): {args.alpha}")
    print(f"  Eta (regularization): {args.eta}")
    
    # Run experiment
    run_experiment(
        domain=args.domain,
        mode=args.mode,
        p=args.p,
        p_start=args.p_start,
        p_end=args.p_end,
        p_steps=args.p_steps,
        epochs=args.epochs,
        epochs_per_p=args.epochs_per_p,
        hidden_layers=hidden_layers,
        activation=args.activation,
        lr=args.lr,
        alpha=args.alpha,
        eta=args.eta,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        batch_size_bc=args.batch_bc,
        batch_size_pde=args.batch_pde,
        output_dir=args.output_dir
    )
