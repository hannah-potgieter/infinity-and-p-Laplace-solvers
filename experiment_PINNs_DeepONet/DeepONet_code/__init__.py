"""
DeepONet_code: Shared library for DeepONet operator-learning experiments.

Re-exports every public symbol so callers can do::

    from DeepONet_code import DeepONet, train_deeponet, ...
"""

from .utils import create_output_dirs, set_seed, get_device, determine_run_tag, save_mse_npz
from .models import DeepONet
from .data import (
    DeepONetDataset,
    load_mat_files,
    load_mat_files_per_p,
    infer_pinns_on_grid,
    distance_to_disc_boundary,
    distance_to_ellipse_boundary,
    dist_to_ellipse_boundary,
    solve_4th_order,
)
from .training import train_deeponet, evaluate_over_p_range
from .plotting import plot_training_history, plot_mse_vs_p, plot_prediction_2d
