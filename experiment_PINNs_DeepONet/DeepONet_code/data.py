"""
Data utilities for DeepONet experiments.

Provides:
    - ``DeepONetDataset`` — PyTorch Dataset wrapping NumPy arrays
    - Geometry helpers — quartic solver, ellipse / disc distance functions
    - ``load_mat_files`` — generic loader for ``.mat`` training data
    - ``infer_pinns_on_grid`` — run a PINNs checkpoint on FEM grid points
"""

import os
import re
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ======================================================================
# Dataset
# ======================================================================

class DeepONetDataset(Dataset):
    """Simple dataset that wraps NumPy or Tensor arrays."""

    def __init__(self, X, Y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ======================================================================
# Geometry helpers
# ======================================================================

def distance_to_disc_boundary(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Distance from interior point to the unit-disc boundary."""
    return 1.0 - torch.sqrt(x ** 2 + y ** 2)


def solve_4th_order(a_c, b_c, c_c, d_c, e_c):
    """
    Solve quartic  a x⁴ + b x³ + c x² + d x + e = 0  (Ferrari's method).

    Returns tensor of shape ``(4, N)`` with all four roots.
    """
    p = (8 * a_c * c_c - 3 * b_c ** 2) / (8 * a_c ** 2)
    q = (b_c ** 3 - 4 * a_c * b_c * c_c + 8 * a_c ** 2 * d_c) / (8 * a_c ** 3)
    delta0 = c_c ** 2 - 3 * b_c * d_c + 12 * a_c * e_c
    delta1 = (
        2 * c_c ** 3 - 9 * b_c * c_c * d_c + 27 * b_c ** 2 * e_c
        + 27 * a_c * d_c ** 2 - 72 * a_c * c_c * e_c
    )

    d0_mask = (torch.abs(delta0) < 0.001).float()
    disc = (delta1 ** 2 - 4 * delta0 ** 3).to(torch.complex64)
    Q_inner = delta1 + (
        (1 - d0_mask) * torch.sqrt(disc) + d0_mask * delta1
    ).to(torch.complex64)
    Q = torch.pow(Q_inner / 2, 1 / 3)

    S = torch.sqrt(-2 / 3 * p + (Q + delta0 / (Q + 1e-10)) / (3 * a_c)) / 2
    b4a = -b_c / (4 * a_c)

    roots = []
    roots.append(torch.real(b4a - S - torch.sqrt(-4 * S ** 2 - 2 * p + q / (S + 1e-10)) / 2))
    roots.append(torch.real(b4a - S + torch.sqrt(-4 * S ** 2 - 2 * p + q / (S + 1e-10)) / 2))
    roots.append(torch.real(b4a + S - torch.sqrt(-4 * S ** 2 - 2 * p - q / (S + 1e-10)) / 2))
    roots.append(torch.real(b4a + S + torch.sqrt(-4 * S ** 2 - 2 * p - q / (S + 1e-10)) / 2))
    return torch.vstack(roots)


def distance_to_ellipse_boundary(
    x: torch.Tensor,
    y: torch.Tensor,
    a: float,
    b: float,
    EPS: float = 1e-6,
) -> torch.Tensor:
    """
    Distance from interior point ``(x, y)`` to the ellipse
    ``x²/a² + y²/b² = 1`` via the quartic solver.
    """
    ones = torch.ones_like(x)
    aa = ones
    bb = ones * (-2 * a ** 2 - 2 * b ** 2)
    cc = a ** 4 + b ** 4 + 4 * a ** 2 * b ** 2 - a ** 2 * x ** 2 - b ** 2 * y ** 2
    dd = (
        -2 * a ** 2 * b ** 4 - 2 * a ** 4 * b ** 2
        + 2 * a ** 2 * b ** 2 * x ** 2 + 2 * a ** 2 * b ** 2 * y ** 2
    )
    ee = a ** 4 * b ** 4 - a ** 2 * b ** 4 * x ** 2 - a ** 4 * b ** 2 * y ** 2

    sol = solve_4th_order(aa, bb, cc, dd, ee)

    x4 = x.repeat(4, 1)
    y4 = y.repeat(4, 1)

    mask_a = (torch.abs(a ** 2 - sol) > EPS).float()
    mask_b = (torch.abs(b ** 2 - sol) > EPS).float()

    bnd_x = mask_a * a ** 2 * x4 / (a ** 2 - sol + EPS) + (1 - mask_a) * a * ((x4 > 0).float() * 2 - 1)
    bnd_x *= (torch.abs(x4) >= 1e-4).float()
    bnd_x += ((torch.abs(x4) < 1e-4) * (torch.abs(y4) < 0.1)).float() * a * ((x4 > 0).float() * 2 - 1)

    bnd_y = mask_b * b ** 2 * y4 / (b ** 2 - sol + EPS) + (1 - mask_b) * b * ((y4 > 0).float() * 2 - 1)
    bnd_y *= (torch.abs(y4) >= 1e-4).float()
    bnd_y += ((torch.abs(y4) < 1e-4) * (torch.abs(x4) < 0.1)).float() * b * ((y4 > 0).float() * 2 - 1)

    dist = torch.sqrt((bnd_x - x4) ** 2 + (bnd_y - y4) ** 2)
    dist = torch.nan_to_num(dist, nan=1e6)
    dist = torch.min(dist, dim=0)[0]
    dist[dist == 1e6] = float(min(a, b))
    dist[x ** 2 + y ** 2 < EPS] = float(min(a, b))
    return dist


def dist_to_ellipse_boundary(x_list, y_list, a, b, n_coarse=3600, chunk=2000):
    """Distance from points to ellipse ``x²/a² + y²/b² = 1`` via grid search.

    Uses a coarse sweep of *n_coarse* boundary samples followed by a
    fine refinement of 200 samples around the best candidate.
    """
    n = x_list.shape[0]
    theta_c = torch.linspace(0, 2 * np.pi, n_coarse + 1)[:-1]
    ex_c = a * torch.cos(theta_c)
    ey_c = b * torch.sin(theta_c)

    best_dist = torch.full((n,), float('inf'))
    best_theta = torch.zeros(n)
    d_theta = 2 * np.pi / n_coarse

    for i in range(0, n, chunk):
        xi, yi = x_list[i:i + chunk], y_list[i:i + chunk]
        dx = ex_c.unsqueeze(1) - xi.unsqueeze(0)
        dy = ey_c.unsqueeze(1) - yi.unsqueeze(0)
        dists = torch.sqrt(dx ** 2 + dy ** 2)
        min_d, min_idx = dists.min(dim=0)
        best_dist[i:i + chunk] = min_d
        best_theta[i:i + chunk] = theta_c[min_idx]

    n_fine = 200
    offsets = torch.linspace(-d_theta, d_theta, n_fine)
    for i in range(0, n, chunk):
        xi, yi = x_list[i:i + chunk], y_list[i:i + chunk]
        bt = best_theta[i:i + chunk]
        theta_f = bt.unsqueeze(0) + offsets.unsqueeze(1)
        ex_f = a * torch.cos(theta_f)
        ey_f = b * torch.sin(theta_f)
        dx = ex_f - xi.unsqueeze(0)
        dy = ey_f - yi.unsqueeze(0)
        dists = torch.sqrt(dx ** 2 + dy ** 2)
        min_d, _ = dists.min(dim=0)
        best_dist[i:i + chunk] = min_d

    return best_dist


# ======================================================================
# .mat file loading
# ======================================================================

def load_mat_files(
    mat_dir,
    p_normalize=500.0,
    coord_scale=2.0,
    coord_offset=0.5,
    pts_key='pts',
    sol_key='solution',
    p_regex=r'p\s*(\d+)',
    subsample=None,
):
    """
    Load training data from ``.mat`` files in *mat_dir*.

    Each file must contain:
        - ``pts_key``: spatial coordinates  ``(N, D)``
        - ``sol_key``: solution values (flattened to ``(N,)`` or ``(1, N)``)

    The filename must match *p_regex* so the p-value can be extracted.

    Coordinates are normalised as ``pts / coord_scale + coord_offset``.

    Returns:
        ``(X_train, Y_train, last_pts_raw)``  where X columns are
        ``[coord_0 … coord_{D-1}, p / p_normalize]``.
        Returns ``(None, None, None)`` when no valid files are found.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required to load .mat files")
    if not os.path.isdir(mat_dir):
        return None, None, None

    all_mat = glob.glob(os.path.join(mat_dir, '*.mat'))
    if not all_mat:
        return None, None, None

    # extract (p, filepath) pairs and sort numerically
    pf_pairs = []
    for mf in all_mat:
        match = re.search(p_regex, os.path.basename(mf))
        if match is not None:
            pf_pairs.append((int(match.group(1)), mf))
    pf_pairs.sort(key=lambda t: t[0])

    X_list, Y_list = [], []
    last_pts = None

    for p, mf in pf_pairs:
        try:
            mat = scipy.io.loadmat(mf)
            pts = mat[pts_key].copy()
            sol = mat[sol_key].flatten().reshape(-1, 1)

            if subsample is not None:
                pts = pts[::subsample]
                sol = sol[::subsample]

            if pts.shape[0] != sol.shape[0]:
                print(f"  Skipping {mf}: shape mismatch pts={pts.shape} sol={sol.shape}")
                continue

            pts_norm = pts / coord_scale + coord_offset
            p_col = np.ones((pts_norm.shape[0], 1)) * p / p_normalize
            X_list.append(np.hstack([pts_norm, p_col]))
            Y_list.append(sol)
            last_pts = pts
            print(f"  Loaded p={p}: {pts.shape[0]} points")
        except Exception as e:
            print(f"  Warning: could not load {mf}: {e}")

    if not X_list:
        return None, None, None

    return np.vstack(X_list), np.vstack(Y_list), last_pts


def load_mat_files_per_p(
    mat_dir,
    pts_key='pts',
    sol_key='solution',
    p_regex=r'p\s*(\d+)',
    subsample=None,
):
    """
    Load ``.mat`` files and return data *per p-value* (un-concatenated).

    Returns:
        list of dicts ``[{'p': int, 'pts': ndarray, 'sol': ndarray}, ...]``
        sorted by ascending p.  Returns an empty list when no valid files
        are found.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required to load .mat files")
    if not os.path.isdir(mat_dir):
        return []

    all_mat = glob.glob(os.path.join(mat_dir, '*.mat'))
    if not all_mat:
        return []

    pf_pairs = []
    for mf in all_mat:
        match = re.search(p_regex, os.path.basename(mf))
        if match is not None:
            pf_pairs.append((int(match.group(1)), mf))
    pf_pairs.sort(key=lambda t: t[0])

    results = []
    for p, mf in pf_pairs:
        try:
            mat = scipy.io.loadmat(mf)
            pts = mat[pts_key].copy()
            sol = mat[sol_key].flatten().reshape(-1, 1).astype(np.float32)

            if subsample is not None:
                pts = pts[::subsample]
                sol = sol[::subsample]

            if pts.shape[0] != sol.shape[0]:
                continue

            results.append({'p': p, 'pts': pts, 'sol': sol})
        except Exception:
            continue

    return results


# ======================================================================
# PINNs checkpoint inference
# ======================================================================

def infer_pinns_on_grid(
    checkpoint_path,
    pts,
    cache_path=None,
    device=None,
    activation='tanh',
):
    """
    Load a PINNs ``.pt`` checkpoint and run inference on *pts*.

    The checkpoint must contain ``'model_state_dict'`` with keys
    ``linears.0.weight``, ``linears.0.bias``, etc. (the format saved by
    ``PINNs_code.training``).  The network architecture is inferred
    automatically from the weight shapes.

    Results are cached to *cache_path* (a ``.npy`` file) so that
    subsequent calls skip inference entirely.

    Args:
        checkpoint_path: path to the PINNs ``.pt`` checkpoint.
        pts: spatial coordinates, ``(N, D)`` NumPy array or Tensor
             (raw / un-normalised — same coordinate space as the FEM data).
        cache_path: if given, save / load the predictions here (``.npy``).
        device: torch device (auto-detected when ``None``).
        activation: activation function used by the PINNs model.

    Returns:
        ``(N, 1)`` float32 NumPy array of predictions.
    """
    # ---- check cache ---------------------------------------------------
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached PINNs prediction: {cache_path}")
        return np.load(cache_path).reshape(-1, 1).astype(np.float32)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- load checkpoint -----------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    # ---- infer architecture from weight shapes -------------------------
    weight_keys = sorted(
        [k for k in state_dict if k.startswith('linears.') and k.endswith('.weight')],
        key=lambda k: int(k.split('.')[1]),
    )
    hidden_layers = [state_dict[k].shape[0] for k in weight_keys[:-1]]
    input_dim = state_dict[weight_keys[0]].shape[1]
    output_dim = state_dict[weight_keys[-1]].shape[0]

    print(f"  PINNs architecture: input={input_dim} hidden={hidden_layers} output={output_dim}")

    # ---- build a lightweight feedforward net ---------------------------
    _acts = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU,
             'elu': nn.ELU, 'silu': nn.SiLU, 'softplus': nn.Softplus,
             'leaky_relu': nn.LeakyReLU, 'sigmoid': nn.Sigmoid}
    act = _acts.get(activation.lower(), nn.Tanh)()

    linears = nn.ModuleList()
    in_f = input_dim
    for h in hidden_layers:
        linears.append(nn.Linear(in_f, h))
        in_f = h
    linears.append(nn.Linear(in_f, output_dim))

    class _FeedForward(nn.Module):
        def __init__(self, linears, activation):
            super().__init__()
            self.linears = linears
            self.activation = activation

        def forward(self, x):
            a = x.float()
            for i in range(len(self.linears) - 1):
                a = self.activation(self.linears[i](a))
            return self.linears[-1](a)

    model = _FeedForward(linears, act)

    # load only the linears.* keys (ignore extras like positivity_weight)
    filtered = {k: v for k, v in state_dict.items() if k.startswith('linears.')}
    model.load_state_dict(filtered)
    model.to(device)
    model.eval()

    # ---- inference -----------------------------------------------------
    if isinstance(pts, np.ndarray):
        pts_t = torch.from_numpy(pts).float()
    else:
        pts_t = pts.float()

    with torch.no_grad():
        pred = model(pts_t.to(device)).cpu().numpy().astype(np.float32)
    pred = pred.reshape(-1, 1)

    # ---- cache ---------------------------------------------------------
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, pred)
        print(f"  Cached PINNs prediction → {cache_path}")

    return pred
