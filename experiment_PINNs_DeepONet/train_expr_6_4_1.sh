#!/bin/bash
set -e

data_dir="${FEM_ORIGIN_DATA_DIR:?Set FEM_ORIGIN_DATA_DIR to the distance-to-origin FEM data directory}"

# square
python experiment/experiment_6_4_1.py --domain square --data-dir "$data_dir" --run-tag pinns_inf --pinns-checkpoint outputs/expr_6_4_eikonal/eikonal_origin_square/checkpoints/best_model.pt
python experiment/experiment_6_4_1.py --domain square --data-dir "$data_dir" --run-tag no_inf --no-exact-inf

# disc
python experiment/experiment_6_4_1.py --domain disc --data-dir "$data_dir" --run-tag pinns_inf --pinns-checkpoint outputs/expr_6_4_eikonal/eikonal_origin_disc/checkpoints/best_model.pt
python experiment/experiment_6_4_1.py --domain disc --data-dir "$data_dir" --run-tag no_inf --no-exact-inf
