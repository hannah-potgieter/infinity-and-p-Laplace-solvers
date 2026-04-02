#!/bin/bash
set -e

# square
python experiment/experiment_7_3_1.py --domain square --data-dir /mnt/ivan/PINNS_SVD_new/utils/ --run-tag pinns_inf --pinns-checkpoint outputs/expr_7_3/eikonal_origin_square/checkpoints/best_model.pt
python experiment/experiment_7_3_1.py --domain square --data-dir /mnt/ivan/PINNS_SVD_new/utils/ --run-tag no_inf --no-exact-inf

# disc
python experiment/experiment_7_3_1.py --domain disc --data-dir /mnt/ivan/PINNS_SVD_new/utils/ --run-tag pinns_inf --pinns-checkpoint outputs/expr_7_3/eikonal_origin_disc/checkpoints/best_model.pt
python experiment/experiment_7_3_1.py --domain disc --data-dir /mnt/ivan/PINNS_SVD_new/utils/ --run-tag no_inf --no-exact-inf
