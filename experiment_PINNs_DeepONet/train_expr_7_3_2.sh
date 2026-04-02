#!/bin/bash
set -e

# 2D
python experiment/experiment_7_3_2.py --domain disc     --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag pinns_inf --pinns-checkpoint outputs/expr_7_3/eikonal_disc/checkpoints/best_model.pt
python experiment/experiment_7_3_2.py --domain ellipse1 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag pinns_inf --pinns-checkpoint outputs/expr_7_3/eikonal_ellipse1/checkpoints/best_model.pt
python experiment/experiment_7_3_2.py --domain ellipse2 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag pinns_inf --pinns-checkpoint outputs/expr_7_3/eikonal_ellipse2/checkpoints/best_model.pt
python experiment/experiment_7_3_2.py --domain ellipse3 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag pinns_inf --pinns-checkpoint outputs/expr_7_3/eikonal_ellipse3/checkpoints/best_model.pt

# 3D
python experiment/experiment_7_3_2.py --domain sphere   --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/3D examples/" --run-tag pinns_inf --pinns-checkpoint outputs/expr_7_3/eikonal3d_sphere/checkpoints/best_model.pt
python experiment/experiment_7_3_2.py --domain cylinder --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/3D examples/" --run-tag pinns_inf --pinns-checkpoint outputs/expr_7_3/eikonal3d_cylinder/checkpoints/best_model.pt
python experiment/experiment_7_3_2.py --domain torus    --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/3D examples/" --run-tag pinns_inf --pinns-checkpoint outputs/expr_7_3/eikonal3d_torus/checkpoints/best_model.pt
