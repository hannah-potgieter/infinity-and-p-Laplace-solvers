"""
Unified DeepONet model for p-Laplacian operator learning.

Architecture (paper Section 5):
    - Trunk net: spatial coordinates -> K latent features
      Default: 3 hidden layers of 512 neurons, output K = 128
    - Branch net: PDE parameters -> K latent features
      Default: 3 hidden layers of 128 neurons, output K = 128
    - Output: element-wise product of trunk and branch outputs,
      followed by a linear layer (K -> 1)

The forward pass splits input at column ``trunk_layers[0]``::

    trunk_input  = x[:, :trunk_layers[0]]   # spatial coordinates
    branch_input = x[:, trunk_layers[0]:]   # PDE parameters
"""

import torch
import torch.nn as nn


class DeepONet(nn.Module):
    """
    Unified DeepONet that works for 2-D, 3-D, and parametric problems.

    The split between trunk and branch inputs is determined automatically
    by ``trunk_layers[0]`` (the spatial dimension).

    Examples::

        # 2-D, single parameter (p)
        net = DeepONet(trunk_layers=[2, 512, 512, 512, 128],
                       branch_layers=[1, 128, 128, 128, 128])

        # 3-D, single parameter (p)
        net = DeepONet(trunk_layers=[3, 512, 512, 512, 128],
                       branch_layers=[1, 128, 128, 128, 128])

        # 2-D, four parameters (p, theta, a, b)
        net = DeepONet(trunk_layers=[2, 512, 512, 512, 128],
                       branch_layers=[4, 128, 128, 128, 128])
    """

    def __init__(
        self,
        trunk_layers=None,
        branch_layers=None,
        activation='tanh',
    ):
        super().__init__()

        if trunk_layers is None:
            trunk_layers = [2, 512, 512, 512, 128]
        if branch_layers is None:
            branch_layers = [1, 128, 128, 128, 128]

        self.trunk_layers = list(trunk_layers)
        self.branch_layers = list(branch_layers)

        assert trunk_layers[-1] == branch_layers[-1], (
            f"Trunk output dim ({trunk_layers[-1]}) must equal "
            f"branch output dim ({branch_layers[-1]})"
        )

        _activations = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU}
        self.activation = _activations.get(activation.lower(), nn.Tanh)()

        self.loss_function = nn.MSELoss(reduction='mean')

        self.trunk_linears = nn.ModuleList([
            nn.Linear(trunk_layers[i], trunk_layers[i + 1])
            for i in range(len(trunk_layers) - 1)
        ])
        self.branch_linears = nn.ModuleList([
            nn.Linear(branch_layers[i], branch_layers[i + 1])
            for i in range(len(branch_layers) - 1)
        ])
        self.output_layer = nn.Linear(trunk_layers[-1], 1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Xavier-normal initialisation with zero biases."""
        for linear in (
            list(self.trunk_linears)
            + list(self.branch_linears)
            + [self.output_layer]
        ):
            nn.init.xavier_normal_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = self.trunk_layers[0]
        a = x[:, :split].float()
        b = x[:, split:].float()

        for i in range(len(self.trunk_layers) - 2):
            a = self.activation(self.trunk_linears[i](a))
        a = self.trunk_linears[-1](a)

        for i in range(len(self.branch_layers) - 2):
            b = self.activation(self.branch_linears[i](b))
        b = self.branch_linears[-1](b)

        return self.output_layer(a * b)

    # ------------------------------------------------------------------
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """MSE between prediction and target."""
        return self.loss_function(self.forward(x), y)
