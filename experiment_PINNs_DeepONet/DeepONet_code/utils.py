"""
Shared utilities: output directories, reproducibility, device selection,
and logging.
"""

import os
import sys
import random
import numpy as np
import torch


class TeeWriter:
    """Write to both a file and the original stream (console)."""

    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def create_output_dirs(base_dir='outputs'):
    """
    Create output directories for plots, checkpoints, and npy.

    Returns:
        dict with 'base', 'plots', 'checkpoints', and 'npy' paths.
    """
    dirs = {'base': base_dir}
    for sub in ('plots', 'checkpoints', 'npy'):
        path = os.path.join(base_dir, sub)
        os.makedirs(path, exist_ok=True)
        dirs[sub] = path
    return dirs


def setup_logging(base_dir):
    """Tee stdout/stderr to both console and *base_dir*/train.log."""
    os.makedirs(base_dir, exist_ok=True)
    log_path = os.path.join(base_dir, 'train.log')
    log_file = open(log_path, 'w')
    sys.stdout = TeeWriter(log_file, sys.__stdout__)
    sys.stderr = TeeWriter(log_file, sys.__stderr__)
    return log_path


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
    if torch.cuda.is_available():
        # PhysicsNeMo enables TF32 by default.  Disable it so DeepONet
        # training and checkpoint evaluation reproduce the reported
        # full-FP32 results across supported CUDA environments.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
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
