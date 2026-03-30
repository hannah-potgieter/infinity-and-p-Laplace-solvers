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

    Args:
        base_dir: Base directory name (default: 'outputs')

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
