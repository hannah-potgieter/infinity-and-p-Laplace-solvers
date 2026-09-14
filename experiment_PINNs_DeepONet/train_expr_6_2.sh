#!/bin/bash
set -e

python experiment/experiment_6_2.py --example arctan          --run-tag normal --save-npy
python experiment/experiment_6_2.py --example aronsson_square --run-tag normal --save-npy
python experiment/experiment_6_2.py --example aronsson_disc   --run-tag normal --save-npy
