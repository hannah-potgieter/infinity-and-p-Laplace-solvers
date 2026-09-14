#!/bin/bash
set -e

data_2d="${FEM_BOUNDARY_2D_DATA_DIR:?Set FEM_BOUNDARY_2D_DATA_DIR to the 2D distance-to-boundary FEM data directory}"
data_3d="${FEM_BOUNDARY_3D_DATA_DIR:?Set FEM_BOUNDARY_3D_DATA_DIR to the 3D distance-to-boundary FEM data directory}"

# 2D
python experiment/experiment_6_4_2.py --domain disc     --data-dir "$data_2d" --run-tag pinns_inf --pinns-checkpoint outputs/expr_6_4_eikonal/eikonal_disc/checkpoints/best_model.pt
python experiment/experiment_6_4_2.py --domain ellipse1 --data-dir "$data_2d" --run-tag pinns_inf --pinns-checkpoint outputs/expr_6_4_eikonal/eikonal_ellipse1/checkpoints/best_model.pt
python experiment/experiment_6_4_2.py --domain ellipse2 --data-dir "$data_2d" --run-tag pinns_inf --pinns-checkpoint outputs/expr_6_4_eikonal/eikonal_ellipse2/checkpoints/best_model.pt
python experiment/experiment_6_4_2.py --domain ellipse3 --data-dir "$data_2d" --run-tag pinns_inf --pinns-checkpoint outputs/expr_6_4_eikonal/eikonal_ellipse3/checkpoints/best_model.pt

# 3D
python experiment/experiment_6_4_2.py --domain sphere   --data-dir "$data_3d" --run-tag pinns_inf --pinns-checkpoint outputs/expr_6_4_eikonal/eikonal3d_sphere/checkpoints/best_model.pt
python experiment/experiment_6_4_2.py --domain cylinder --data-dir "$data_3d" --run-tag pinns_inf --pinns-checkpoint outputs/expr_6_4_eikonal/eikonal3d_cylinder/checkpoints/best_model.pt
python experiment/experiment_6_4_2.py --domain torus    --data-dir "$data_3d" --run-tag pinns_inf --pinns-checkpoint outputs/expr_6_4_eikonal/eikonal3d_torus/checkpoints/best_model.pt
