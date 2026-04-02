#!/bin/bash
set -e

# 7.3.1: exact_inf
python experiment/experiment_7_3_1.py --domain square --data-dir /mnt/ivan/PINNS_SVD_new/utils/ --run-tag exact_inf
python experiment/experiment_7_3_1.py --domain disc   --data-dir /mnt/ivan/PINNS_SVD_new/utils/ --run-tag exact_inf

# 7.3.2 2D: no_inf + exact_inf
python experiment/experiment_7_3_2.py --domain disc     --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag no_inf --no-exact-inf
python experiment/experiment_7_3_2.py --domain disc     --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag exact_inf
python experiment/experiment_7_3_2.py --domain ellipse1 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag no_inf --no-exact-inf
python experiment/experiment_7_3_2.py --domain ellipse1 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag exact_inf
python experiment/experiment_7_3_2.py --domain ellipse2 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag no_inf --no-exact-inf
python experiment/experiment_7_3_2.py --domain ellipse2 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag exact_inf
python experiment/experiment_7_3_2.py --domain ellipse3 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag no_inf --no-exact-inf
python experiment/experiment_7_3_2.py --domain ellipse3 --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/2D examples/" --run-tag exact_inf

# 7.3.2 3D: no_inf + exact_inf
python experiment/experiment_7_3_2.py --domain sphere   --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/3D examples/" --run-tag no_inf --no-exact-inf
python experiment/experiment_7_3_2.py --domain sphere   --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/3D examples/" --run-tag exact_inf
python experiment/experiment_7_3_2.py --domain cylinder --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/3D examples/" --run-tag no_inf --no-exact-inf
python experiment/experiment_7_3_2.py --domain cylinder --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/3D examples/" --run-tag exact_inf
python experiment/experiment_7_3_2.py --domain torus    --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/3D examples/" --run-tag no_inf --no-exact-inf
python experiment/experiment_7_3_2.py --domain torus    --data-dir "/mnt/ivan/PINNS_SVD_new/utils/Distance to boundary/3D examples/" --run-tag exact_inf
