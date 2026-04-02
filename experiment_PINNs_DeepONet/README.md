# Deep-Learning Solvers for Infinity and p-Laplace Problems

Code repository for the paper *Deep-Learning Solvers for Infinity and p-Laplace Problems*, submitted to the Journal of Computational Physics.

**Authors:** Tak Shing Au Yeung, Ka Chun Cheung, Hannah Potgieter, Steven J. Ruuth, Simon See

## Abstract

We investigate the use of neural network solvers for infinity and p-Laplace problems, which are fundamental in nonlinear analysis and have practical applications. Our approach employs Physics-Informed Neural Networks (PINNs) with hard constraints and Deep Operator Networks (DeepONets) to address computational challenges associated with large p values, on various 2D and 3D domains. Our method appears to offer advantages over traditional physics-based solvers, particularly in handling high-dimensional problems. We demonstrate the effectiveness of these neural network solvers through numerical experiments and compare their performance with conventional methods.

## Repository Structure

```
experiment_PINNs_DeepONet/
├── PINNs_code/              # PINNs framework
│   ├── models.py            #   Network architectures (BasePINN, DomainDecompositionPINN)
│   ├── training.py          #   Training loops (single and iterative)
│   ├── data.py              #   Domain data generation (square, disc, ellipse, 3D)
│   ├── samplers.py          #   Interior point samplers
│   ├── schedulers.py        #   Alpha schedulers (auto, ReLoBRaLo, fixed)
│   ├── activations.py       #   Activation functions
│   ├── plotting.py          #   Visualization utilities
│   └── utils.py             #   Logging, checkpointing, seeding
│
├── DeepONet_code/           # DeepONet framework
│   ├── models.py            #   DeepONet architecture
│   ├── training.py          #   Training loop
│   ├── data.py              #   FEM data loading and PINNs inference caching
│   ├── plotting.py          #   Visualization utilities
│   └── utils.py             #   Logging, checkpointing, seeding
│
├── experiment/              # Experiment scripts
│   ├── experiment_7_1.py    #   Infinity Laplacian (PINNs)
│   ├── experiment_7_2.py    #   p-Laplacian (PINNs, simple + iterative)
│   ├── experiment_7_3.py    #   Eikonal equation (PINNs)
│   ├── experiment_7_3_1.py  #   Distance-to-origin (DeepONet)
│   ├── experiment_7_3_2.py  #   Distance-to-boundary (DeepONet, 2D + 3D)
│   └── experiment_7_3_3.py  #   All-ellipse generalization (DeepONet)
│
├── train_expr_7_1.sh              # Normal training for 7.1
├── train_expr_7_1_ablation.sh     # Ablation studies for 7.1
├── train_expr_7_2_simple.sh       # Simple p-Laplacian training
├── train_expr_7_2_iterative.sh    # Iterative p-Laplacian training
├── train_expr_7_3_pinns.sh        # Eikonal PINNs (prerequisite for DeepONet)
├── train_expr_7_3_1.sh            # Distance-to-origin DeepONet
├── train_expr_7_3_2.sh            # Distance-to-boundary DeepONet
├── train_expr_7_3_3.sh            # All-ellipse DeepONet
└── train_expr_7_3_ablation.sh     # DeepONet ablation (no_inf / exact_inf modes)
```

## Experiments

### 7.1 — Infinity Laplacian (PINNs)

Solves the homogeneous infinity Laplacian \(\Delta_\infty u = 0\) on 2D domains using PINNs with hard boundary constraints, eta-normalized residuals, residual clipping, outlier removal, and adaptive PDE loss weighting.

```bash
bash train_expr_7_1.sh            # normal training (5 examples)
```

| Example | Domain | PINNs MSE | Newton FEM MSE | Inference Time |
|---------|--------|-----------|----------------|----------------|
| Arctan | \([0.01,1]^2\) | **2.418e-07** | 5.892e-05 | 7.644e-05 s |
| Aronsson (square) | \([-1,1]^2\) | **5.735e-07** | 9.511e-05 | 1.124e-04 s |
| Aronsson (disc) | unit disc | **4.770e-07** | 2.522e-04 | 7.742e-05 s |
| Absolute | \([-1,1]^2\) | 8.524e-04 | **3.809e-05** | 7.632e-05 s |
| Absolute (DD-PINNs) | \([-1,1]^2\) | **1.908e-06** | 3.809e-05 | 1.288e-03 s |

Domain decomposition PINNs reduces MSE by over two orders of magnitude for the non-smooth absolute-value solution.

### 7.2 — p-Laplacian (PINNs)

Solves \(\Delta_p u = 0\) for \(p = 2\) to \(1000\) on \([-1,1]^2\) (Aronsson example). Two training strategies are compared.

```bash
bash train_expr_7_2_simple.sh     # independent training per p (13 values)
bash train_expr_7_2_iterative.sh  # 5-band continuation pipeline
```

**Simple training** (independent per p): becomes unstable beyond \(p \approx 10\).

**Iterative training** (5-band continuation): maintains stable convergence up to \(p = 1000\).

| p | MSE\_\(\infty\) (simple) | MSE\_\(\infty\) (iterative) |
|---|--------------------------|------------------------------|
| 10 | 1.615e-04 | 1.315e-04 |
| 100 | 5.071e-02 | 3.967e-06 |
| 1000 | 1.061e-01 | **1.548e-06** |

### 7.3 — Eikonal Equation (PINNs)

Solves \(|\nabla u|^2 = 1\) with PINNs to compute the distance-to-origin and distance-to-boundary functions on 2D and 3D domains. These PINNs solutions serve as the \(p = \infty\) training data for the downstream DeepONet experiments.

```bash
bash train_expr_7_3_pinns.sh  # 10 domains (prerequisite for 7.3.1 and 7.3.2)
```

**Domains:** square, disc, 3 ellipses (2D); sphere, cylinder, torus (3D).

### 7.3.1 — Distance-to-Origin (DeepONet)

Learns the operator mapping \(p \mapsto u_p\) for the distance-to-origin problem. Including the PINNs-predicted \(p = \infty\) solution in training dramatically improves extrapolation.

```bash
bash train_expr_7_3_1.sh          # pinns_inf + no_inf modes (4 runs)
```

| Domain | MSE at p=500 (without \(u_\infty\)) | MSE at p=500 (with \(u_\infty\)) |
|--------|--------------------------------------|-----------------------------------|
| Square | 5.527e-04 | **7.148e-07** |
| Disc | 3.549e-03 | **2.872e-06** |

### 7.3.2 — Distance-to-Boundary (DeepONet, 2D and 3D)

Learns the distance-to-boundary operator across 7 domains. A single DeepONet replaces per-domain Newton FEM solves.

```bash
bash train_expr_7_3_2.sh          # pinns_inf mode (7 runs)
```

| Domain | DeepONet MSE | Inference Time | Newton FEM Time |
|--------|-------------|----------------|-----------------|
| 2D disc | 5.452e-06 | 4.568e-04 s | 8,667 s |
| Ellipse 1 | 1.062e-05 | 3.505e-04 s | ~3.5 hours |
| 3D sphere | 6.037e-06 | 2.974e-03 s | ~5 days |
| 3D cylinder | 5.119e-05 | 4.559e-03 s | ~5 days |
| 3D torus | 3.640e-05 | 8.593e-04 s | ~5 days |

Total DeepONet training: ~4.86 hours, compared to ~5 days per 3D Newton FEM solve.

### 7.3.3 — All-Ellipse Generalization (DeepONet)

Learns a single model over a parametric family of ellipses (semi-axes \(a, b\) and rotation \(\theta\)), demonstrating generalization to unseen domain geometries with MSE on the order of 2e-05 at \(p = 500\).

```bash
bash train_expr_7_3_3.sh  # 5 total_div values
```

## Output Directory Structure

All experiments write outputs to an experiment-first hierarchy:

```
outputs/
├── expr_7_1/{example}/{run_tag}/
├── expr_7_2_simple/{example}/
├── expr_7_2_iter/{example}/
├── expr_7_3/{example}/
├── expr_7_3_1/{domain}/{run_tag}/
├── expr_7_3_2/{domain}/{run_tag}/
└── expr_7_3_3/{tag}/{run_tag}/
```

Each run directory contains:
- `train.log` — full training log (stdout + stderr)
- `checkpoints/` — `best_model.pt` and `final_model.pt`
- `plots/` — result visualizations
- `npy/` — numerical data (`.npz` files)

## Running

Training scripts are executed from the `experiment_PINNs_DeepONet/` directory.

```bash
# Normal training
bash train_expr_7_1.sh

# Ablation studies
bash train_expr_7_1_ablation.sh

```

The `--run-tag` argument controls the subdirectory name for ablation variants. All scripts use `set -e` to halt on first failure.

## Command-Line Arguments

### `experiment_7_1.py` — Infinity Laplacian (PINNs)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--example` | str | `aronsson_square` | Problem to solve. Choices: `arctan`, `absolute`, `aronsson_square`, `aronsson_disc` |
| `--epochs` | int | `100` | Number of training epochs |
| `--lr` | float | `1e-3` | Learning rate |
| `--hidden-layers` | str | `128,128,128,128` | Comma-separated layer widths |
| `--activation` | str | `tanh` | Choices: `tanh`, `relu`, `leaky_relu`, `elu`, `gelu`, `softplus`, `silu`, `sigmoid` |
| `--interior-grid` | int | `1000` | Interior collocation points per axis |
| `--boundary-grid` | int | `10000` | Boundary points |
| `--batch-bc` | int | `400` | Boundary batch size |
| `--batch-pde` | int | `1000` | PDE batch size |
| `--eta` | float | `1e-5` | Eta-normalization constant |
| `--clip-residual` | float | `10.0` | Residual clipping threshold (0 to disable) |
| `--outlier-percentile` | float | `2.0` | Top-percentile outlier removal (0 to disable) |
| `--outlier-off-epoch` | int | `None` | Epoch to stop outlier removal |
| `--alpha` | float | `None` | Fixed PDE loss weight (overrides schedule) |
| `--alpha-schedule` | str | `None` | `auto`, `relobralo`, or `epoch:value,...` |
| `--alpha-min` | float | `1e-5` | Initial alpha for auto schedule |
| `--alpha-max` | float | `1e-1` | Maximum alpha for auto schedule |
| `--alpha-patience` | int | `5` | Patience before alpha increase |
| `--alpha-cooldown` | int | `3` | Cooldown after alpha change |
| `--alpha-min-improvement` | float | `0.2` | Minimum relative improvement for auto schedule |
| `--relobralo-temperature` | float | `1.0` | ReLoBRaLo temperature |
| `--relobralo-alpha` | float | `0.999` | ReLoBRaLo exponential average alpha |
| `--relobralo-rho` | float | `0.99` | ReLoBRaLo rho |
| `--scheduler` | str | `cosine` | LR scheduler. Choices: `linear`, `cosine`, `step`, `exponential`, `none` |
| `--scheduler-end-factor` | float | `0.01` | Final LR as fraction of initial |
| `--val-split` | float | `0.2` | Validation split ratio |
| `--patience` | int | `None` | Early stopping patience |
| `--no-resample` | flag | `False` | Disable interior point resampling |
| `--domain-decomposition` | flag | `False` | Use domain decomposition PINNs |
| `--n-interface-pts` | int | `200` | Interface points for DD-PINNs |
| `--interface-weight` | float | `1.0` | Interface loss weight |
| `--interface-every` | int | `5` | Interface update frequency (epochs) |
| `--output-dir` | str | `outputs` | Root output directory |
| `--run-tag` | str | `None` | Ablation tag; creates a subdirectory under the example folder |
| `--save-npy` | flag | `False` | Save numerical data as `.npz` |
| `--no-save` | flag | `False` | Skip saving plots |
| `--plot-pde-every` | int | `0` | Plot PDE loss distribution every N epochs (0 to disable) |

### `experiment_7_2.py` — p-Laplacian (PINNs)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `simple` | Training mode. Choices: `simple`, `iterative` |
| `--domain` | str | `square` | Choices: `square`, `disc` |
| `--p` | int | `10` | Single p value (simple mode) |
| `--p-values` | str | `None` | Comma-separated p values (iterative mode) |
| `--p-start` | int | `2` | Start of p range (iterative, if `--p-values` not given) |
| `--p-end` | int | `100` | End of p range |
| `--p-steps` | int | `10` | Number of p steps |
| `--epochs` | int | `100` | Epochs (simple mode) |
| `--epochs-per-p` | int | `100` | Epochs per p value (iterative mode) |
| `--lr` | float | `1e-3` | Learning rate |
| `--hidden-layers` | str | `128,128,128,128` | Comma-separated layer widths |
| `--activation` | str | `tanh` | Activation function |
| `--eta` | float | `1e-5` | Eta-normalization constant |
| `--clip-residual` | float | `10.0` | Residual clipping threshold |
| `--clip-grad-norm` | float | `1.0` | Gradient norm clipping |
| `--outlier-percentile` | float | `2.0` | Outlier removal percentile |
| `--outlier-off-epoch` | int | `None` | Epoch to stop outlier removal |
| `--alpha` | float | `None` | Fixed PDE loss weight |
| `--alpha-schedule` | str | `auto` | Alpha schedule strategy |
| `--bc-loss-threshold` | float | `None` | Early stopping BC loss threshold (iterative) |
| `--pde-loss-threshold` | float | `None` | Early stopping PDE loss threshold (iterative) |
| `--resume-checkpoint` | str | `None` | Path to checkpoint for resuming iterative training |
| `--scheduler` | str | `cosine` | LR scheduler |
| `--output-dir` | str | `outputs` | Root output directory |
| `--run-tag` | str | `None` | Ablation subdirectory tag |
| `--save-npy` | flag | `False` | Save numerical data |

*Alpha schedule, validation, resampling, and batch size arguments are the same as `experiment_7_1.py`.*

### `experiment_7_3.py` — Eikonal Equation (PINNs)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--case` | str | `boundary` | Choices: `origin`, `boundary` |
| `--domain` | str | `disc` | Choices: `square`, `disc`, `ellipse1`, `ellipse2`, `ellipse3`, `sphere`, `cylinder`, `torus` |
| `--ring-radius` | float | `0.01` | Radius of ring source (origin case only) |
| `--n-ring-pts` | int | `1000` | Number of ring collocation points |
| `--positivity-weight` | float | `1.0` | Weight for the u >= 0 positivity penalty |
| `--interior-grid` | int | `None` | Points per axis (default: 1000 for 2D, 100 for 3D) |
| `--boundary-grid` | int | `None` | Boundary points (default: 10000 for 2D, 20000 for 3D) |

*Network, training, alpha schedule, and output arguments are the same as `experiment_7_1.py`.*

### `experiment_7_3_1.py` — Distance-to-Origin (DeepONet)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--domain` | str | `disc` | Choices: `disc`, `square`, `all` |
| `--epochs` | int | `20` | Training epochs |
| `--lr` | float | `1e-4` | Learning rate |
| `--batch-size` | int | `2048` | Batch size |
| `--trunk-layers` | str | `2,512,512,512,128` | Trunk network layers |
| `--branch-layers` | str | `1,128,128,128,128` | Branch network layers |
| `--data-dir` | str | `./experiment_6_3_1` | Directory containing FEM `.mat` files |
| `--pinns-checkpoint` | str | `None` | Path to PINNs `.pt` checkpoint; runs inference on FEM grid and caches to `.npy` |
| `--pinns-file` | str | `None` | Path to pre-saved PINNs prediction `.npy` (skipped if `--pinns-checkpoint` given) |
| `--no-exact-inf` | flag | — | Do not append exact p=infinity solution to training data |
| `--run-tag` | str | `auto` | Run tag (auto-detected: `no_inf`, `exact_inf`, or `pinns_inf`) |
| `--output-dir` | str | `outputs` | Root output directory |
| `--seed` | int | `1234` | Random seed |

### `experiment_7_3_2.py` — Distance-to-Boundary (DeepONet)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--domain` | str | `disc` | Choices: `disc`, `ellipse1`, `ellipse2`, `ellipse3`, `sphere`, `cube`, `cylinder`, `torus`, `all-2d`, `all-3d`, `all` |
| `--trunk-layers` | str | `2,512,512,512,128` | Trunk network (first dim auto-adjusted to 3 for 3D domains) |

*All other arguments are the same as `experiment_7_3_1.py`.*

### `experiment_7_3_3.py` — All-Ellipse Generalization (DeepONet)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--total-div` | int | `10` | Number of theta divisions for ellipse parametrization |
| `--trunk-layers` | str | `2,512,512,512,128` | Trunk network layers |
| `--branch-layers` | str | `4,128,128,128,128` | Branch network (4-dim input: x, y, p, theta) |

*Epochs, LR, batch size, data directory, seed, and output arguments are the same as `experiment_7_3_1.py`.*

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy

A ready-to-use environment can be set up with the official PyTorch Docker image:

```bash
docker pull pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
docker run --gpus all -it pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
pip install scipy matplotlib
```
