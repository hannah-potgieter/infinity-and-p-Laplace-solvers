#!/bin/bash
set -e

data_dir="${FEM_BOUNDARY_2D_DATA_DIR:?Set FEM_BOUNDARY_2D_DATA_DIR to the 2D distance-to-boundary FEM data directory}"

python experiment/experiment_6_4_3.py --total-div 10 --data-dir "$data_dir"
python experiment/experiment_6_4_3.py --total-div 8  --data-dir "$data_dir"
python experiment/experiment_6_4_3.py --total-div 6  --data-dir "$data_dir"
python experiment/experiment_6_4_3.py --total-div 4  --data-dir "$data_dir"
python experiment/experiment_6_4_3.py --total-div 2  --data-dir "$data_dir"
