# Neural Experiments: PINNs and DeepONet

This directory contains the neural-network code for *Deep-Learning Solvers and Surrogates for Infinity and p-Laplace Problems*. It includes the PINN solvers, Eikonal PINNs used to supply limiting profiles, DeepONet surrogates, ablation launchers, the five-seed sensitivity study, direct finite-p validation, and manuscript plotting scripts.

File and output-directory names follow manuscript Sections 6.2--6.4.3. Figure numbers below also refer to the current manuscript.

## Directory structure

```text
experiment_PINNs_DeepONet/
├── PINNs_code/       PINN models, losses, schedulers, sampling, and training
├── DeepONet_code/    DeepONet models, FEM-data loading, training, and utilities
├── experiment/       training, checkpoint-evaluation, and result-summary programs
├── plot/             manuscript plotting scripts
│   └── data/         small plot-ready CSV files
├── outputs/images/   generated image files used by manuscript Figures 1--12
├── outputs/tables/   table-ready numerical results and provenance notes
├── train_expr_*.sh   experiment and ablation entry points
└── requirements-pytorch.txt
```

Large FEM datasets and model checkpoints are not committed. Store them outside the Git checkout and provide their locations using the options and environment variables below.

## Environment

The experiments use the PhysicsNeMo 25.06 environment documented in the [root README](../README.md). The verified image already contains PyTorch, NumPy, Matplotlib, and SciPy at the recorded versions; no package upgrade is required. `requirements-pytorch.txt` records the bundled NumPy, Matplotlib, and SciPy versions for reference and for plot-only environments outside the container.

Run commands from this `experiment_PINNs_DeepONet/` directory. The programs select a CUDA device when one is available and otherwise fall back to the CPU.

### Random seeds

All standard `train_expr_*.sh` launchers use seed `1234` by default. Run these scripts without a seed argument to reproduce the default manuscript configuration. The underlying Section 6.2 and DeepONet programs accept `--seed` when an override is needed; the Section 6.3 and Eikonal PINN programs currently define seed `1234` in their experiment programs. The dedicated five-seed launcher is the only workflow below that requires an explicit seed.

## Main experiments

| Manuscript section | Experiment | Command to run |
|---|---|---|
| Section 6.2 | Infinity-Laplace PINNs: Arctan and Aronsson examples | `bash train_expr_6_2.sh` |
| Section 6.2 | Five-seed sensitivity study for the same three examples | `bash experiment/run_section_6_2_seed_sensitivity.sh GPU_ID SEED RUN_ROOT` |
| Section 6.3.1 | Independently trained homogeneous $p$-Laplace PINNs | `bash train_expr_6_3_simple.sh` |
| Section 6.3.2 | Iterative $p$-continuation PINNs | `bash train_expr_6_3_iterative.sh` |
| Section 6.3.3 | Direct finite-$p$ PINN validation on the unit disc | `bash experiment/run_section_6_3_3_distance_boundary_bands.sh OUTPUT_DIR` |
| Section 6.4 (supporting models) | Eikonal PINNs that supply limiting profiles | `bash train_expr_6_4_eikonal.sh` |
| Section 6.4.1 | Distance-to-origin DeepONet | `bash train_expr_6_4_1.sh` |
| Section 6.4.2 | Distance-to-boundary DeepONet in 2D and 3D | `bash train_expr_6_4_2.sh` |
| Section 6.4.3 | Ellipse-family DeepONet | `bash train_expr_6_4_3.sh` |

### Section 6.2: Infinity-Laplace PINNs

The main manuscript examples are Arctan, Aronsson on the square, and Aronsson on the unit disc:

```bash
bash train_expr_6_2.sh
```

An individual run uses seed `1234` when `--seed` is omitted:

```bash
python experiment/experiment_6_2.py \
  --example arctan --run-tag normal --save-npy
```

To override the default, pass another seed explicitly, for example `--seed 3456`.

#### Five-seed sensitivity study

Table 4 uses the fixed seeds

```text
1234, 3456, 5678, 6789, 8765
```

for the Arctan, Aronsson-square, and Aronsson-disc PINNs. Each seed can be run on a selected GPU as follows:

```bash
for seed in 1234 3456 5678 6789 8765; do
  bash experiment/run_section_6_2_seed_sensitivity.sh \
    0 "$seed" /path/to/seed-study
done
```

Unlike the standard launchers, `run_section_6_2_seed_sensitivity.sh` requires the GPU identifier, seed, and output directory as explicit arguments.

Summarize the best-checkpoint test errors with:

```bash
python experiment/summarize_section_6_2_seed_sensitivity.py \
  /path/to/seed-study
```

The summary program extracts the `test_loss` from the model restored at the minimum validation boundary loss. It intentionally does not use the later `Final Test MSE` log line. The manuscript reports the arithmetic mean and sample standard deviation:

| Example | Test MSE over five seeds |
|---|---:|
| Arctan | $(3.118\pm2.327)\times10^{-7}$ |
| Aronsson, square | $(4.468\pm1.696)\times10^{-7}$ |
| Aronsson, disc | $(6.247\pm2.805)\times10^{-7}$ |

These runs were reproduced with the PhysicsNeMo 25.06 image recorded in the root README. The loss histories in the manuscript and the reported training and inference timings correspond to seed 1234; the mean and standard deviation concern test MSE only.

### Section 6.3.1: Independently trained homogeneous p-Laplace PINNs

```bash
bash train_expr_6_3_simple.sh
```

This launcher trains each $p$ independently.

### Section 6.3.2: Iterative p-Laplace PINNs

```bash
bash train_expr_6_3_iterative.sh
```

This launcher performs the continuation schedule for the homogeneous Aronsson problem.

### Section 6.3.3: Direct finite-p PINN validation on the unit disc

`experiment_6_3_3_distance_boundary_disc.py` solves

$$
-\Delta_p u_p=1\quad\text{in the unit disc},
\qquad u_p=0\quad\text{on its boundary},
$$

and evaluates every saved checkpoint against the exact finite-p radial solution. The reported parameter-only schedule through $p=1000$ is encoded in one portable launcher:

```bash
bash experiment/run_section_6_3_3_distance_boundary_bands.sh /path/to/run-directory
```

The optional second argument overrides the default seed of `1234`.

The launcher preserves the separate $p=11,\ldots,15$ and $p=16,\ldots,20$ stages used in the reported calculation, including the random-number reset between them. It writes `combined_metrics.csv`, containing total loss, boundary loss, PINN residual loss, error against $u_\infty$, and direct error against $u_p$.

Regenerate the Figure 7(b) panel with:

```bash
python plot/experiment_6_3_3_distance_boundary_disc_plot.py \
  --metrics-csv /path/to/run-directory/combined_metrics.csv
```

The committed `plot/data/experiment_6_3_3_distance_boundary_disc_all_p.csv` is the plot-ready table assembled from the selected Server 1 checkpoints through $p=1000$.

### Supporting Eikonal PINNs for Section 6.4

The Eikonal models provide the $p\to\infty$ profiles used by DeepONet when an analytic limiting solution is unavailable:

```bash
bash train_expr_6_4_eikonal.sh
```

### Section 6.4: DeepONet experiments

The DeepONet programs load FEM `.mat` files. Before using the launchers, set the relevant data locations:

```bash
export FEM_ORIGIN_DATA_DIR=/path/to/distance-to-origin-data
export FEM_BOUNDARY_2D_DATA_DIR=/path/to/2d-distance-to-boundary-data
export FEM_BOUNDARY_3D_DATA_DIR=/path/to/3d-distance-to-boundary-data
```

#### Section 6.4.1: Distance-to-origin DeepONet

```bash
bash train_expr_6_4_1.sh
```

#### Section 6.4.2: Distance-to-boundary DeepONet in 2D and 3D

```bash
bash train_expr_6_4_2.sh
```

These variables replace the original machine-specific `/mnt/ivan/...` paths. The expected filenames and coordinate normalization are implemented in `DeepONet_code/data.py` and the corresponding experiment programs.

##### Direct finite-p evaluation of the unit-disc DeepONet

No retraining is required for this evaluation. Given the saved unit-disc DeepONet checkpoint:

```bash
python experiment/evaluate_section_6_4_2_disk_exact.py \
  --checkpoint /path/to/deeponet_final_model.pt \
  --output-dir /path/to/evaluation-output \
  --p-start 5 --p-end 500 --p-step 1
```

The output distinguishes finite-p training inputs through $p=200$, unseen interpolation inputs in that range, PINN-augmented interpolation between $200$ and $500$, and the auxiliary input at $p=500$. It reports both the direct finite-p error `mse_exact` and the error `mse_limit` against the limiting distance profile.

The selected values reported in Table 9 are committed as `outputs/tables/table_9_deeponet_direct_finite_p.csv`; the evaluator outputs through the manuscript cutoff $p=500$ are retained in `outputs/tables/table_9_deeponet_source/`.

### Table-ready numerical results

`outputs/tables/` contains compact CSV files for values that are quoted in tables or prose but are not naturally represented by a plotting array. See `outputs/tables/README.md` for the source run, checkpoint selection rule, evaluation environment, and any distinction between a table value and a historical training output.

#### Section 6.4.3: Ellipse-family DeepONet

```bash
bash train_expr_6_4_3.sh
```

## Prepare every manuscript figure

The complete set of 29 image files used by manuscript Figures 1--12 can be assembled without retraining:

```bash
python plot/replot_manuscript_figures.py
```

The command retains or copies the committed static artwork for Figures 1--3, regenerates Figures 4--12 from the plot-ready arrays under `outputs/` and `plot/data/`, writes the PNG files to `outputs/images/`, and fails if any expected image is missing.

## Manuscript figure-to-script map

| Manuscript figure | Experiment/output | Plotting command or script |
|---|---|---|
| Figure 1 | Boundary partition diagram | Static asset: `outputs/images/dist_visual.png` |
| Figure 2 | PINN architecture and training workflow | Static asset: `outputs/images/pinns.png` |
| Figure 3 | DeepONet branch/trunk architecture | Static asset: `outputs/images/deeponet.png` |
| Figure 4 | Arctan PINN: loss, prediction, exact solution | `python plot/experiment_6_2_plot.py --example arctan` |
| Figure 5 | Aronsson PINN on square | `python plot/experiment_6_2_plot.py --example aronsson_square` |
| Figure 6 | Aronsson PINN on unit disc | `python plot/experiment_6_2_plot.py --example aronsson_disc` |
| Figure 7(a) | Homogeneous p-Laplace PINN continuation | `python plot/experiment_6_3_plot.py --domain square` |
| Figure 7(b) | Direct finite-p unit-disc PINN validation | `python plot/experiment_6_3_3_distance_boundary_disc_plot.py` |
| Figure 8 | Distance-to-origin DeepONet, disc and square | `python plot/experiment_6_4_plot.py --type origin` |
| Figure 9 | Distance-to-boundary DeepONet, 2D domains | `python plot/experiment_6_4_plot.py --type boundary` |
| Figure 10 | Distance-to-boundary DeepONet, 3D domains | `python plot/experiment_6_4_plot.py --type boundary` |
| Figure 11 | Ellipse-family errors versus $\theta$ and $(a,b)$ | `python plot/experiment_6_4_3_plot.py --total-div 10` |
| Figure 12 | Ellipse-family sampling-density study | `python plot/experiment_6_4_3_plot.py --total-div all` |

Unless overridden, plotting programs read from `outputs/` and write to `outputs/images/`. Use their `--help` options to point to archived run directories.

## Manuscript appendix-to-code map

Run these commands from `experiment_PINNs_DeepONet/` inside the environment described above.

| Appendix | Result reproduced | Command to run |
|---|---|---|
| Appendix A: training hyperparameters | Main PINN and DeepONet configurations | Run the corresponding Section 6.2--6.4 Bash launcher in the main experiment table above; each launcher passes the Appendix A settings to its experiment program. |
| Appendix B: infinity-Laplace PINN ablation | Arctan and Aronsson ablation results, including the reported width-256 runs | `bash train_expr_6_2_ablation.sh` |
| Appendix C: simple p-Laplace PINN ablation | Independent-training ablation results at $p=10$ and $p=200$ | `bash train_expr_6_3_simple_ablation.sh` |
| Appendix D: iterative p-Laplace schedules | Main continuation run | `bash train_expr_6_3_iterative.sh` |
| Appendix D: direct finite-$p$ schedule | Unit-disc finite-$p$ validation through $p=1000$ | `bash experiment/run_section_6_3_3_distance_boundary_bands.sh /path/to/run-directory` |
| Appendix E: Eikonal PINN data generation | Eikonal solutions used as limiting-profile inputs | `bash train_expr_6_4_eikonal.sh` |

## Output layout

The training programs create experiment-specific subdirectories under `outputs/`. A typical run contains:

```text
train.log
checkpoints/
plots/
npy/
```

Checkpoints can be large and are excluded by `.gitignore`. Preserve the complete run directory when archiving an experiment because the plotting and summary programs use the logs and `.npz`/CSV files as well as the checkpoint.
