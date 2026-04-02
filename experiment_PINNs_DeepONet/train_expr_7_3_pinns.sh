#!/bin/bash
set -e

# Distance to origin (2D)
python experiment/experiment_7_3.py --case origin --domain square --save-npy --ring-radius 0.1 --alpha-min 1.0 --alpha-max 10.0 --positivity-weight 1.0
python experiment/experiment_7_3.py --case origin --domain disc   --save-npy --ring-radius 0.1 --alpha-min 1.0 --alpha-max 10.0 --positivity-weight 1.0

# Distance to boundary (2D)
python experiment/experiment_7_3.py --domain square   --save-npy --alpha-min 1.0 --alpha-max 10.0 --positivity-weight 1.0
python experiment/experiment_7_3.py --domain disc     --save-npy --alpha-min 1.0 --alpha-max 10.0 --positivity-weight 1.0
python experiment/experiment_7_3.py --domain ellipse1 --save-npy --alpha-min 0.1 --alpha-max 1.0  --positivity-weight 100.0
python experiment/experiment_7_3.py --domain ellipse2 --save-npy --alpha-min 0.1 --alpha-max 1.0  --positivity-weight 100.0
python experiment/experiment_7_3.py --domain ellipse3 --save-npy --alpha-min 1.0 --alpha-max 10.0 --positivity-weight 100.0

# Distance to boundary (3D)
python experiment/experiment_7_3.py --domain sphere   --save-npy --alpha-min 1.0 --alpha-max 10.0 --positivity-weight 0.5
python experiment/experiment_7_3.py --domain cylinder --save-npy --alpha-min 1.0 --alpha-max 10.0 --positivity-weight 0.5
python experiment/experiment_7_3.py --domain torus    --save-npy --alpha-min 0.1 --alpha-max 10.0 --positivity-weight 0.5
