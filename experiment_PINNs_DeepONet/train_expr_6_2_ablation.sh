#!/usr/bin/env bash
set -euo pipefail

# Appendix B: Arctan and Aronsson (square and disc) ablations.
for example in arctan aronsson_square aronsson_disc; do
    python experiment/experiment_6_2.py --example "$example" --run-tag alpha0      --alpha 0.0
    python experiment/experiment_6_2.py --example "$example" --run-tag alpha1e-5   --alpha 1e-5
    python experiment/experiment_6_2.py --example "$example" --run-tag alpha1e-1   --alpha 1e-1
    python experiment/experiment_6_2.py --example "$example" --run-tag relobralo   --alpha-schedule relobralo
    python experiment/experiment_6_2.py --example "$example" --run-tag eta0        --eta 0.0
    python experiment/experiment_6_2.py --example "$example" --run-tag clip0       --clip-residual 0.0
    python experiment/experiment_6_2.py --example "$example" --run-tag out0        --outlier-percentile 0.0
    python experiment/experiment_6_2.py --example "$example" --run-tag nosched     --scheduler none
    python experiment/experiment_6_2.py --example "$example" --run-tag clip0_out0      --clip-residual 0.0 --outlier-percentile 0.0
    python experiment/experiment_6_2.py --example "$example" --run-tag clip0_out0_eta0 --clip-residual 0.0 --outlier-percentile 0.0 --eta 0.0
    python experiment/experiment_6_2.py --example "$example" --run-tag h32 --hidden-layers 32,32,32,32
    python experiment/experiment_6_2.py --example "$example" --run-tag h64 --hidden-layers 64,64,64,64
    python experiment/experiment_6_2.py --example "$example" --run-tag l2  --hidden-layers 128,128
    python experiment/experiment_6_2.py --example "$example" --run-tag l3  --hidden-layers 128,128,128
    python experiment/experiment_6_2.py --example "$example" --run-tag l5  --hidden-layers 128,128,128,128,128
done
