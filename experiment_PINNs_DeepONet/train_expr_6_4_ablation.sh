#!/bin/bash
set -e

origin_data="${FEM_ORIGIN_DATA_DIR:?Set FEM_ORIGIN_DATA_DIR to the distance-to-origin FEM data directory}"
boundary_2d_data="${FEM_BOUNDARY_2D_DATA_DIR:?Set FEM_BOUNDARY_2D_DATA_DIR to the 2D distance-to-boundary FEM data directory}"
boundary_3d_data="${FEM_BOUNDARY_3D_DATA_DIR:?Set FEM_BOUNDARY_3D_DATA_DIR to the 3D distance-to-boundary FEM data directory}"

# Section 6.4.1: exact_inf
python experiment/experiment_6_4_1.py --domain square --data-dir "$origin_data" --run-tag exact_inf
python experiment/experiment_6_4_1.py --domain disc   --data-dir "$origin_data" --run-tag exact_inf

# Section 6.4.2, 2D: no_inf + exact_inf
python experiment/experiment_6_4_2.py --domain disc     --data-dir "$boundary_2d_data" --run-tag no_inf --no-exact-inf
python experiment/experiment_6_4_2.py --domain disc     --data-dir "$boundary_2d_data" --run-tag exact_inf
python experiment/experiment_6_4_2.py --domain ellipse1 --data-dir "$boundary_2d_data" --run-tag no_inf --no-exact-inf
python experiment/experiment_6_4_2.py --domain ellipse1 --data-dir "$boundary_2d_data" --run-tag exact_inf
python experiment/experiment_6_4_2.py --domain ellipse2 --data-dir "$boundary_2d_data" --run-tag no_inf --no-exact-inf
python experiment/experiment_6_4_2.py --domain ellipse2 --data-dir "$boundary_2d_data" --run-tag exact_inf
python experiment/experiment_6_4_2.py --domain ellipse3 --data-dir "$boundary_2d_data" --run-tag no_inf --no-exact-inf
python experiment/experiment_6_4_2.py --domain ellipse3 --data-dir "$boundary_2d_data" --run-tag exact_inf

# Section 6.4.2, 3D: no_inf + exact_inf
python experiment/experiment_6_4_2.py --domain sphere   --data-dir "$boundary_3d_data" --run-tag no_inf --no-exact-inf
python experiment/experiment_6_4_2.py --domain sphere   --data-dir "$boundary_3d_data" --run-tag exact_inf
python experiment/experiment_6_4_2.py --domain cylinder --data-dir "$boundary_3d_data" --run-tag no_inf --no-exact-inf
python experiment/experiment_6_4_2.py --domain cylinder --data-dir "$boundary_3d_data" --run-tag exact_inf
python experiment/experiment_6_4_2.py --domain torus    --data-dir "$boundary_3d_data" --run-tag no_inf --no-exact-inf
python experiment/experiment_6_4_2.py --domain torus    --data-dir "$boundary_3d_data" --run-tag exact_inf
