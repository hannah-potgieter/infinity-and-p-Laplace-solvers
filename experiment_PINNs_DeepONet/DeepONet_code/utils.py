"""
Shared utilities: output directories, reproducibility, and device selection.
"""

import os
import random
import numpy as np
import torch


def create_output_dirs(base_dir='outputs'):
    """
    Create output directories for plots and checkpoints.

    Returns:
        dict with 'base', 'plots', and 'checkpoints' paths
    """
    plots_dir = os.path.join(base_dir, 'plots')
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    return {
        'base': base_dir,
        'plots': plots_dir,
        'checkpoints': checkpoints_dir,
    }


def set_seed(seed=1234):
    """
    Set all random seeds for reproducibility.

    Covers Python built-in, NumPy, PyTorch CPU, and PyTorch CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device():
    """Return the best available torch device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def determine_run_tag(args):
    """
    Derive a short run tag from CLI flags to distinguish the three
    DeepONet training cases:

        * ``no_inf``    — trained without a p=∞ data point
        * ``exact_inf`` — p=∞ from the closed-form exact solution
        * ``pinns_inf`` — p=∞ from a PINNs prediction

    A user-supplied ``--run-tag`` overrides auto-detection.
    """
    if getattr(args, 'run_tag', 'auto') != 'auto':
        return args.run_tag
    if getattr(args, 'pinns_checkpoint', None) or getattr(args, 'pinns_file', None):
        return 'pinns_inf'
    if not getattr(args, 'add_exact_inf', True):
        return 'no_inf'
    return 'exact_inf'


def save_mse_npz(filepath, **curves):
    """
    Save multiple MSE curves into a single ``.npz`` file.

    Each keyword argument should be a ``(p_list, mse_list)`` tuple.
    They are stored as ``'<name>_p'`` and ``'<name>_mse'`` arrays.

    Example::

        save_mse_npz('outputs/npy/mse_pinns_inf_origin_disc.npz',
                     deeponet=(ev_p, ev_mse),
                     fem=(fem_p, fem_mse),
                     pred_train=(pt_p, pt_mse))
    """
    data = {}
    for name, (p_list, mse_list) in curves.items():
        data[f'{name}_p'] = np.asarray(p_list, dtype=np.float64)
        data[f'{name}_mse'] = np.asarray(mse_list, dtype=np.float64)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    np.savez(filepath, **data)
    print(f"  Saved MSE curves → {filepath}  ({', '.join(curves.keys())})")
