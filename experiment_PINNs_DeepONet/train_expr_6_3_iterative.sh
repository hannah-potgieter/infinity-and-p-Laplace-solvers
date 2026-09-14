#!/bin/bash
set -e

# Stage 1: p = 2 → 20
python experiment/experiment_6_3.py --mode iterative --domain square --p-values 2,3,4,5,6,7,8,9,10,15,20 --save-npy --bc-loss-threshold 1e-4 --pde-loss-threshold 1e-3 --epochs-per-p 100

# Stage 2: p = 25 → 50 (resume from stage 1)
python experiment/experiment_6_3.py --mode iterative --domain square --p-values 25,30,40,50 --save-npy --bc-loss-threshold 1e-4 --pde-loss-threshold 1e-3 --epochs-per-p 20 --alpha 1e-2 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p2to20_square/checkpoints/best_model_p20.pt

# Stage 3: p = 60 → 100 (resume from stage 2)
python experiment/experiment_6_3.py --mode iterative --domain square --p-values 60,70,80,90,100 --save-npy --bc-loss-threshold 1e-4 --pde-loss-threshold 1e-3 --epochs-per-p 20 --alpha 1e-3 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p25to50_square/checkpoints/best_model_p50.pt

# Stage 4: p = 150 → 500 (resume from stage 3)
python experiment/experiment_6_3.py --mode iterative --domain square --p-values 150,200,300,400,500 --save-npy --bc-loss-threshold 1e-4 --pde-loss-threshold 1e-3 --epochs-per-p 20 --alpha 1e-4 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p60to100_square/checkpoints/best_model_p100.pt

# Stage 5: p = 600 → 1000 (resume from stage 4)
python experiment/experiment_6_3.py --mode iterative --domain square --p-values 600,700,800,900,1000 --save-npy --bc-loss-threshold 1e-4 --pde-loss-threshold 1e-3 --epochs-per-p 20 --alpha 1e-5 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p150to500_square/checkpoints/best_model_p500.pt
