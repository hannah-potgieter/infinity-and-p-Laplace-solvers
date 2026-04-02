#!/bin/bash
set -e

python experiment/experiment_7_1.py --example arctan          --run-tag normal --save-npy
python experiment/experiment_7_1.py --example absolute        --run-tag normal --save-npy
python experiment/experiment_7_1.py --example aronsson_square --run-tag normal --save-npy
python experiment/experiment_7_1.py --example aronsson_disc   --run-tag normal --save-npy
python experiment/experiment_7_1.py --example absolute --domain-decomposition --hidden-layers 32,32,32,32 --run-tag normal --save-npy
