# FEM and Deep-Learning Code for Infinity and p-Laplace Problems

This is the companion repository for *Deep-Learning Solvers and Surrogates for Infinity and p-Laplace Problems*. It contains both the finite-element data generators and the Python code used for the PINN and DeepONet experiments.

The repository contains the static artwork used in manuscript Figures 1--3 and the source code and plot-ready numerical arrays required to regenerate Figures 4--12. Large FEM training datasets and neural-network checkpoints are not committed; they are required for retraining, but not for replotting the reported numerical figures. The neural README explains how those external inputs are supplied.

## Repository contents

| Directory | Contents |
|---|---|
| `ellipses/` | deal.II generator for axis-aligned and rotated 2D ellipses |
| `ball-and-hypercube/` | deal.II generators for 2D and 3D reference domains |
| `cylinder/` | deal.II generator for the 3D cylinder |
| `torus/` | deal.II generator for the 3D torus |
| `experiment_PINNs_DeepONet/` | PINN, Eikonal PINN, DeepONet, ablation, seed-study, evaluation, and plotting code |

See [`experiment_PINNs_DeepONet/README.md`](experiment_PINNs_DeepONet/README.md) for the neural experiments, manuscript figure map, random seeds, and appendix-to-script map.

## Manuscript experiment-to-launcher map

Run the following Bash launchers from `experiment_PINNs_DeepONet/`. The five-seed study is listed with the other Section 6.2 experiments.

| Manuscript section | Experiment | Bash launcher |
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

The associated ablation launchers and direct checkpoint-evaluation commands are listed in the [neural README](experiment_PINNs_DeepONet/README.md).

## Mathematical problems

The FEM generators solve regularized p-Laplace problems of the form

$$
-\nabla\cdot\left((\eta^2+|\nabla u|^2)^{(p-2)/2}\nabla u\right)=f
\quad\text{in }\Omega,
$$

with the boundary conditions specified in each example directory. Continuation in $p$ provides data for studying the approach to the infinity-Laplace limit. The neural directory contains the PINN solvers and DeepONet surrogates evaluated in the manuscript.

## Reproducible environments

### Finite-element code

The manuscript computations used:

- deal.II 9.6.1;
- a C++17 compiler;
- CMake.

Each FEM directory is independent. A typical build is:

```bash
cd ellipses
cmake -DDEAL_II_DIR=/path/to/dealii .
cmake --build .
cmake --build . --target run
```

Consult the README in the selected FEM directory for its parameter file and output layout.

### PINN and DeepONet environment

The neural experiments, including the five-seed sensitivity study, were reproduced with:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.7.0a0+79aa17489c.nv25.04 |
| CUDA runtime | 12.9 |
| NumPy | 1.26.4 |
| Matplotlib | 3.10.1 |
| SciPy | 1.15.2 |

The Docker image is:

```text
nvcr.io/nvidia/physicsnemo/physicsnemo:25.06
```

One reproducible setup is:

```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.06
docker run --rm --gpus all -it \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/workspace" -w /workspace/experiment_PINNs_DeepONet \
  nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 \
  bash
```

The image already contains the listed Python packages; no package upgrade is required. NGC authentication may be required to pull the image. The exact seed procedure and reported statistics are documented in the neural README.

All standard PINN, Eikonal PINN, and DeepONet launchers use random seed `1234` by default, so no seed argument is required for the default manuscript runs. The Section 6.2 sensitivity study uses the five fixed seeds `1234`, `3456`, `5678`, `6789`, and `8765`; its launcher and summary commands are documented in the neural README. An explicit seed is needed only when overriding the default or running this five-seed study.

## Data and outputs

The FEM programs generate numerical p-Laplace solutions used by the DeepONet scripts. Because the complete datasets and checkpoints are large, they should be stored outside the Git checkout and passed through the documented `--data-dir` and `--checkpoint` options. Generated checkpoints (`*.pt`) and per-run plot directories are ignored by Git.

The exact PNG artwork for Figures 1--3 is committed under `experiment_PINNs_DeepONet/outputs/images/`. The numerical arrays under `experiment_PINNs_DeepONet/outputs/` and `experiment_PINNs_DeepONet/plot/data/` are sufficient to regenerate Figures 4--12. The table-ready CSV files under `experiment_PINNs_DeepONet/outputs/tables/` record the five-seed study, direct finite-$p$ validation, checkpoint reevaluations, reported solution norms, and Appendix B--C ablations. From `experiment_PINNs_DeepONet/`, run:

```bash
python plot/replot_manuscript_figures.py
```

The command retains or copies the three static artwork files, replots all numerical panels from the committed arrays, and verifies all 29 image files used by Figures 1--12. Generated images are written to `experiment_PINNs_DeepONet/outputs/images/`.

### Manuscript figure-to-script map

The following individual commands are run from `experiment_PINNs_DeepONet/`.

| Manuscript figure | Content | Plotting command |
|---|---|---|
| Figure 1 | Boundary partition diagram | Static asset: `outputs/images/dist_visual.png` |
| Figure 2 | PINN architecture and training workflow | Static asset: `outputs/images/pinns.png` |
| Figure 3 | DeepONet branch/trunk architecture | Static asset: `outputs/images/deeponet.png` |
| Figure 4 | Arctan infinity-Laplace PINN | `python plot/experiment_6_2_plot.py --example arctan` |
| Figure 5 | Aronsson infinity-Laplace PINN on the square | `python plot/experiment_6_2_plot.py --example aronsson_square` |
| Figure 6 | Aronsson infinity-Laplace PINN on the unit disc | `python plot/experiment_6_2_plot.py --example aronsson_disc` |
| Figure 7(a) | Homogeneous $p$-Laplace PINN continuation | `python plot/experiment_6_3_plot.py --domain square` |
| Figure 7(b) | Direct finite-$p$ PINN validation on the unit disc | `python plot/experiment_6_3_3_distance_boundary_disc_plot.py` |
| Figure 8 | Distance-to-origin DeepONet on the disc and square | `python plot/experiment_6_4_plot.py --type origin` |
| Figure 9 | Distance-to-boundary DeepONet on the 2D domains | `python plot/experiment_6_4_plot.py --type boundary` |
| Figure 10 | Distance-to-boundary DeepONet on the 3D domains | `python plot/experiment_6_4_plot.py --type boundary` |
| Figure 11 | Ellipse-family errors versus $\theta$ and $(a,b)$ | `python plot/experiment_6_4_3_plot.py --total-div 10` |
| Figure 12 | Ellipse-family sampling-density study | `python plot/experiment_6_4_3_plot.py --total-div all` |

## License

See [`LICENSE`](LICENSE).
