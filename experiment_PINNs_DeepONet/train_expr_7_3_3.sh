#!/bin/bash
set -e

python experiment/experiment_7_3_3.py --total-div 10 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/"
python experiment/experiment_7_3_3.py --total-div 8  --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/"
python experiment/experiment_7_3_3.py --total-div 6  --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/"
python experiment/experiment_7_3_3.py --total-div 4  --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/"
python experiment/experiment_7_3_3.py --total-div 2  --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/"
