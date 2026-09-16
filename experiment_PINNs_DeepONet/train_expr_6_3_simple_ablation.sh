#!/bin/bash
set -e

# Ablation study for Section 6.3 — simple (independent) training mode.
# Tests stabilization techniques at p=10 and p=200.
# Defaults: eta=1e-5, clip-residual=100.0, clip-grad-norm=1.0, outlier-percentile=2.0

# ── p = 10 ──
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag eta0                      --eta 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag clip0                     --clip-residual 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag gradclip0                 --clip-grad-norm 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag out0                      --outlier-percentile 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag clip0_gradclip0           --clip-residual 0 --clip-grad-norm 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag gradclip0_out0            --clip-grad-norm 0 --outlier-percentile 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag gradclip0_eta0            --clip-grad-norm 0 --eta 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag clip0_gradclip0_out0      --clip-residual 0 --clip-grad-norm 0 --outlier-percentile 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag clip0_gradclip0_eta0      --clip-residual 0 --clip-grad-norm 0 --eta 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag gradclip0_out0_eta0       --clip-grad-norm 0 --outlier-percentile 0 --eta 0
python experiment/experiment_6_3.py --mode simple --p 10 --domain square --save-npy --run-tag clip0_gradclip0_out0_eta0 --clip-residual 0 --clip-grad-norm 0 --outlier-percentile 0 --eta 0

# ── p = 200 ──
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag eta0                      --eta 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag clip0                     --clip-residual 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag gradclip0                 --clip-grad-norm 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag out0                      --outlier-percentile 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag clip0_gradclip0           --clip-residual 0 --clip-grad-norm 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag gradclip0_out0            --clip-grad-norm 0 --outlier-percentile 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag gradclip0_eta0            --clip-grad-norm 0 --eta 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag clip0_gradclip0_out0      --clip-residual 0 --clip-grad-norm 0 --outlier-percentile 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag clip0_gradclip0_eta0      --clip-residual 0 --clip-grad-norm 0 --eta 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag gradclip0_out0_eta0       --clip-grad-norm 0 --outlier-percentile 0 --eta 0
python experiment/experiment_6_3.py --mode simple --p 200 --domain square --save-npy --run-tag clip0_gradclip0_out0_eta0 --clip-residual 0 --clip-grad-norm 0 --outlier-percentile 0 --eta 0
