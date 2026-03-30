"""
BasePINN: shared feedforward architecture and loss logic for all PDE experiments.

Subclasses only need to override ``compute_pde_residual(self, x)``.
"""

import numpy as np
import torch
import torch.nn as nn

from .activations import get_activation


class BasePINN(nn.Module):
    """
    Base Physics-Informed Neural Network.

    Provides the common feedforward architecture, boundary loss, and
    outlier-aware PDE loss.  Subclasses implement the PDE-specific
    residual via :meth:`compute_pde_residual`.
    """

    def __init__(
        self,
        hidden_layers=None,
        activation='tanh',
        input_dim=2,
        output_dim=1,
        clip_residual=0.0,
        outlier_percentile=1.0,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 128]

        self.activation = get_activation(activation)
        self.activation_name = activation
        self.loss_function = nn.MSELoss(reduction='mean')
        self.clip_residual = clip_residual
        self.outlier_percentile = outlier_percentile

        layers = []
        in_features = input_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_dim))

        self.linears = nn.ModuleList(layers)

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight.data, gain=1.0)
            nn.init.zeros_(layer.bias.data)

    # -- forward -------------------------------------------------------------

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.linears) - 1):
            a = self.activation(self.linears[i](a))
        a = self.linears[-1](a)
        return a

    # -- PDE residual (override in subclass) ---------------------------------

    def compute_pde_residual(self, x):
        """
        Compute the PDE residual tensor for interior points *x*.

        Must return a tensor of shape ``(N, 1)`` or ``(N,)`` where the
        PDE is satisfied when the residual is zero.

        Subclasses **must** override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_pde_residual()"
        )

    # -- losses --------------------------------------------------------------

    def loss_boundary(self, x_bc, y_bc):
        """MSE loss on boundary conditions."""
        return self.loss_function(self.forward(x_bc), y_bc)

    def loss_pde(self, x_pde):
        """
        PDE residual loss with optional clipping and outlier removal.

        Calls :meth:`compute_pde_residual`, then applies:
        - ``clip_residual > 0``: clamp residual to ``[-clip, clip]``
        - ``outlier_percentile > 0``: exclude top X% of extreme per-point losses

        When the residual is longer than the input (e.g. concatenated PDE +
        constraint terms), outlier removal operates per-point on the first N
        entries (the PDE residual) and the remaining entries are kept in full.
        """
        residual = self.compute_pde_residual(x_pde)

        if self.clip_residual > 0:
            residual = torch.clamp(residual, -self.clip_residual, self.clip_residual)

        n_pts = x_pde.shape[0]
        r = residual.squeeze()

        if self.outlier_percentile > 0 and n_pts > 10:
            pde_r = r[:n_pts]
            extra_r = r[n_pts:] if r.shape[0] > n_pts else None

            per_point_loss = pde_r ** 2
            keep_pct = (100.0 - self.outlier_percentile) / 100.0
            threshold = torch.quantile(per_point_loss, keep_pct)
            keep_mask = per_point_loss <= threshold
            if keep_mask.sum() > 0:
                pde_loss = per_point_loss[keep_mask].mean()
            else:
                pde_loss = per_point_loss.mean()

            if extra_r is not None and extra_r.numel() > 0:
                return (pde_loss + (extra_r ** 2).mean()) / 2
            return pde_loss

        return self.loss_function(residual, torch.zeros_like(residual))

    # -- diagnostics ---------------------------------------------------------

    def get_pde_loss_with_outlier_info(self, x_pde):
        """
        Compute per-point PDE loss with outlier classification.

        Returns:
            x_coords, y_coords, per_point_loss, is_outlier, threshold
        """
        with torch.enable_grad():
            residual = self.compute_pde_residual(x_pde)

        residual = residual.detach()

        if self.clip_residual > 0:
            residual = torch.clamp(residual, -self.clip_residual, self.clip_residual)

        n_pts = x_pde.shape[0]
        r = residual.squeeze()
        if r.shape[0] > n_pts and r.shape[0] % n_pts == 0:
            r = r.view(-1, n_pts).pow(2).mean(dim=0)
            per_point_loss = r.cpu().numpy()
        else:
            per_point_loss = (r ** 2).cpu().numpy()
        x_coords = x_pde[:, 0].detach().cpu().numpy()
        y_coords = x_pde[:, 1].detach().cpu().numpy()

        if self.outlier_percentile > 0:
            keep_pct = 100.0 - self.outlier_percentile
            threshold = np.percentile(per_point_loss, keep_pct)
            is_outlier = per_point_loss > threshold
        else:
            threshold = np.max(per_point_loss) + 1
            is_outlier = np.zeros(len(per_point_loss), dtype=bool)

        return x_coords, y_coords, per_point_loss, is_outlier, threshold
