#!/bin/bash
set -e

# ── arctan ──
python experiment/experiment_7_1.py --example arctan --run-tag alpha0      --alpha 0.0
python experiment/experiment_7_1.py --example arctan --run-tag alpha1e-5   --alpha 1e-5
python experiment/experiment_7_1.py --example arctan --run-tag alpha1e-1   --alpha 1e-1
python experiment/experiment_7_1.py --example arctan --run-tag relobralo   --alpha-schedule relobralo
python experiment/experiment_7_1.py --example arctan --run-tag eta0        --eta 0.0
python experiment/experiment_7_1.py --example arctan --run-tag clip0       --clip-residual 0.0
python experiment/experiment_7_1.py --example arctan --run-tag out0        --outlier-percentile 0.0
python experiment/experiment_7_1.py --example arctan --run-tag nosched     --scheduler none
python experiment/experiment_7_1.py --example arctan --run-tag clip0_out0      --clip-residual 0.0 --outlier-percentile 0.0
python experiment/experiment_7_1.py --example arctan --run-tag clip0_out0_eta0 --clip-residual 0.0 --outlier-percentile 0.0 --eta 0.0
python experiment/experiment_7_1.py --example arctan --run-tag h32  --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example arctan --run-tag h64  --hidden-layers 64,64,64,64
python experiment/experiment_7_1.py --example arctan --run-tag l2   --hidden-layers 128,128
python experiment/experiment_7_1.py --example arctan --run-tag l3   --hidden-layers 128,128,128
python experiment/experiment_7_1.py --example arctan --run-tag l5   --hidden-layers 128,128,128,128,128

# ── absolute ──
python experiment/experiment_7_1.py --example absolute --run-tag alpha0      --alpha 0.0
python experiment/experiment_7_1.py --example absolute --run-tag alpha1e-5   --alpha 1e-5
python experiment/experiment_7_1.py --example absolute --run-tag alpha1e-1   --alpha 1e-1
python experiment/experiment_7_1.py --example absolute --run-tag relobralo   --alpha-schedule relobralo
python experiment/experiment_7_1.py --example absolute --run-tag eta0        --eta 0.0
python experiment/experiment_7_1.py --example absolute --run-tag clip0       --clip-residual 0.0
python experiment/experiment_7_1.py --example absolute --run-tag out0        --outlier-percentile 0.0
python experiment/experiment_7_1.py --example absolute --run-tag nosched     --scheduler none
python experiment/experiment_7_1.py --example absolute --run-tag clip0_out0      --clip-residual 0.0 --outlier-percentile 0.0
python experiment/experiment_7_1.py --example absolute --run-tag clip0_out0_eta0 --clip-residual 0.0 --outlier-percentile 0.0 --eta 0.0
python experiment/experiment_7_1.py --example absolute --run-tag h32  --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --run-tag h64  --hidden-layers 64,64,64,64
python experiment/experiment_7_1.py --example absolute --run-tag l2   --hidden-layers 128,128
python experiment/experiment_7_1.py --example absolute --run-tag l3   --hidden-layers 128,128,128
python experiment/experiment_7_1.py --example absolute --run-tag l5   --hidden-layers 128,128,128,128,128

# ── aronsson_square ──
python experiment/experiment_7_1.py --example aronsson_square --run-tag alpha0      --alpha 0.0
python experiment/experiment_7_1.py --example aronsson_square --run-tag alpha1e-5   --alpha 1e-5
python experiment/experiment_7_1.py --example aronsson_square --run-tag alpha1e-1   --alpha 1e-1
python experiment/experiment_7_1.py --example aronsson_square --run-tag relobralo   --alpha-schedule relobralo
python experiment/experiment_7_1.py --example aronsson_square --run-tag eta0        --eta 0.0
python experiment/experiment_7_1.py --example aronsson_square --run-tag clip0       --clip-residual 0.0
python experiment/experiment_7_1.py --example aronsson_square --run-tag out0        --outlier-percentile 0.0
python experiment/experiment_7_1.py --example aronsson_square --run-tag nosched     --scheduler none
python experiment/experiment_7_1.py --example aronsson_square --run-tag clip0_out0      --clip-residual 0.0 --outlier-percentile 0.0
python experiment/experiment_7_1.py --example aronsson_square --run-tag clip0_out0_eta0 --clip-residual 0.0 --outlier-percentile 0.0 --eta 0.0
python experiment/experiment_7_1.py --example aronsson_square --run-tag h32  --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example aronsson_square --run-tag h64  --hidden-layers 64,64,64,64
python experiment/experiment_7_1.py --example aronsson_square --run-tag l2   --hidden-layers 128,128
python experiment/experiment_7_1.py --example aronsson_square --run-tag l3   --hidden-layers 128,128,128
python experiment/experiment_7_1.py --example aronsson_square --run-tag l5   --hidden-layers 128,128,128,128,128

# ── aronsson_disc ──
python experiment/experiment_7_1.py --example aronsson_disc --run-tag alpha0      --alpha 0.0
python experiment/experiment_7_1.py --example aronsson_disc --run-tag alpha1e-5   --alpha 1e-5
python experiment/experiment_7_1.py --example aronsson_disc --run-tag alpha1e-1   --alpha 1e-1
python experiment/experiment_7_1.py --example aronsson_disc --run-tag relobralo   --alpha-schedule relobralo
python experiment/experiment_7_1.py --example aronsson_disc --run-tag eta0        --eta 0.0
python experiment/experiment_7_1.py --example aronsson_disc --run-tag clip0       --clip-residual 0.0
python experiment/experiment_7_1.py --example aronsson_disc --run-tag out0        --outlier-percentile 0.0
python experiment/experiment_7_1.py --example aronsson_disc --run-tag nosched     --scheduler none
python experiment/experiment_7_1.py --example aronsson_disc --run-tag clip0_out0      --clip-residual 0.0 --outlier-percentile 0.0
python experiment/experiment_7_1.py --example aronsson_disc --run-tag clip0_out0_eta0 --clip-residual 0.0 --outlier-percentile 0.0 --eta 0.0
python experiment/experiment_7_1.py --example aronsson_disc --run-tag h32  --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example aronsson_disc --run-tag h64  --hidden-layers 64,64,64,64
python experiment/experiment_7_1.py --example aronsson_disc --run-tag l2   --hidden-layers 128,128
python experiment/experiment_7_1.py --example aronsson_disc --run-tag l3   --hidden-layers 128,128,128
python experiment/experiment_7_1.py --example aronsson_disc --run-tag l5   --hidden-layers 128,128,128,128,128

# ── absolute_dd (domain decomposition) ──
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag alpha0      --alpha 0.0          --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag alpha1e-5   --alpha 1e-5         --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag alpha1e-1   --alpha 1e-1         --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag relobralo   --alpha-schedule relobralo --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag eta0        --eta 0.0            --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag clip0       --clip-residual 0.0  --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag out0        --outlier-percentile 0.0 --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag nosched     --scheduler none     --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag clip0_out0      --clip-residual 0.0 --outlier-percentile 0.0 --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag clip0_out0_eta0 --clip-residual 0.0 --outlier-percentile 0.0 --eta 0.0 --hidden-layers 32,32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag h16  --hidden-layers 16,16,16,16
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag h64  --hidden-layers 64,64,64,64
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag l2   --hidden-layers 32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag l3   --hidden-layers 32,32,32
python experiment/experiment_7_1.py --example absolute --domain-decomposition --run-tag l5   --hidden-layers 32,32,32,32,32
