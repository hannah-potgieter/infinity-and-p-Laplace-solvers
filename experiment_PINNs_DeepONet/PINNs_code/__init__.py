"""
PINNs_code: Shared library for Physics-Informed Neural Networks.

Re-exports every public symbol so callers can do:

    from PINNs_code import BasePINN, train_pinn, ...
"""

from .utils import create_output_dirs, set_seed, get_device
from .activations import ACTIVATIONS, get_activation
from .schedulers import (
    FixedAlphaSchedule,
    AlphaPlateauScheduler,
    ReLoBRaLo,
    parse_alpha_schedule,
)
from .models import BasePINN
from .samplers import InteriorSampler, split_data
from .training import train_pinn, train_pinn_iterative
from .plotting import plot_results, plot_pde_loss_distribution
from .data import generate_square_domain_data, generate_disc_domain_data
