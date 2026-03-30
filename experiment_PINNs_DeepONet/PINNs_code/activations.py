"""
Activation function registry for PINN models.
"""

import torch.nn as nn


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
        raise ValueError(
            f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]()
