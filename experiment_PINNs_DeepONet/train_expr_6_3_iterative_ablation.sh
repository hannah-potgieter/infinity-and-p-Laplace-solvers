#!/bin/bash
set -e

# Ablation study for Section 6.3 — iterative (continuation) training mode.
#
# Group 1: Stabilization (band 2 only, resume from normal band 1 checkpoint)
# Group 2: Continuation strategy (full pipeline variants)
#
# Normal band defaults for reference:
#   Band 1: p=2-20,  adaptive alpha, lr=1e-3, cosine sched, 100 epochs/p
#   Band 2: p=25-50, alpha=1e-2, lr=1e-5, no sched, 20 epochs/p
#   Band 3: p=60-100, alpha=1e-3, lr=1e-5, no sched, 20 epochs/p
#   Band 4: p=150-500, alpha=1e-4, lr=1e-5, no sched, 20 epochs/p
#   Band 5: p=600-1000, alpha=1e-5, lr=1e-5, no sched, 20 epochs/p

BAND1_CKPT="outputs/expr_6_3_iter/p2to20_square/checkpoints/best_model_p20.pt"
COMMON="--mode iterative --domain square --save-npy --bc-loss-threshold 1e-4 --pde-loss-threshold 1e-3"
BAND2_ARGS="--p-values 25,30,40,50 --epochs-per-p 20 --alpha 1e-2 --lr 1e-5 --scheduler none"

# =============================================================================
# Group 1 — Stabilization (band 2 only, resume from normal band 1 checkpoint)
# =============================================================================

python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag eta0                      --eta 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag clip0                     --clip-residual 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag gradclip0                 --clip-grad-norm 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag out0                      --outlier-percentile 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag clip0_gradclip0           --clip-residual 0 --clip-grad-norm 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag gradclip0_out0            --clip-grad-norm 0 --outlier-percentile 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag gradclip0_eta0            --clip-grad-norm 0 --eta 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag clip0_gradclip0_out0      --clip-residual 0 --clip-grad-norm 0 --outlier-percentile 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag clip0_gradclip0_eta0      --clip-residual 0 --clip-grad-norm 0 --eta 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag gradclip0_out0_eta0       --clip-grad-norm 0 --outlier-percentile 0 --eta 0
python experiment/experiment_6_3.py $COMMON $BAND2_ARGS --resume-checkpoint $BAND1_CKPT --run-tag clip0_gradclip0_out0_eta0 --clip-residual 0 --clip-grad-norm 0 --outlier-percentile 0 --eta 0

# =============================================================================
# Group 2 — Continuation strategy
# =============================================================================

# ── fix_alpha_1e-2: use alpha=1e-2 for all bands 2-5 ──
TAG=fix_alpha_1e-2
python experiment/experiment_6_3.py $COMMON --p-values 2,3,4,5,6,7,8,9,10,15,20 --epochs-per-p 100 --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 25,30,40,50       --epochs-per-p 20 --alpha 1e-2 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p2to20_square/$TAG/checkpoints/best_model_p20.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 60,70,80,90,100   --epochs-per-p 20 --alpha 1e-2 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p25to50_square/$TAG/checkpoints/best_model_p50.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 150,200,300,400,500     --epochs-per-p 20 --alpha 1e-2 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p60to100_square/$TAG/checkpoints/best_model_p100.pt --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 600,700,800,900,1000    --epochs-per-p 20 --alpha 1e-2 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p150to500_square/$TAG/checkpoints/best_model_p500.pt --run-tag $TAG

# ── fix_alpha_1e-3: use alpha=1e-3 for all bands 2-5 ──
TAG=fix_alpha_1e-3
python experiment/experiment_6_3.py $COMMON --p-values 2,3,4,5,6,7,8,9,10,15,20 --epochs-per-p 100 --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 25,30,40,50       --epochs-per-p 20 --alpha 1e-3 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p2to20_square/$TAG/checkpoints/best_model_p20.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 60,70,80,90,100   --epochs-per-p 20 --alpha 1e-3 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p25to50_square/$TAG/checkpoints/best_model_p50.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 150,200,300,400,500     --epochs-per-p 20 --alpha 1e-3 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p60to100_square/$TAG/checkpoints/best_model_p100.pt --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 600,700,800,900,1000    --epochs-per-p 20 --alpha 1e-3 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p150to500_square/$TAG/checkpoints/best_model_p500.pt --run-tag $TAG

# ── fix_alpha_1e-5: use alpha=1e-5 for all bands 2-5 ──
TAG=fix_alpha_1e-5
python experiment/experiment_6_3.py $COMMON --p-values 2,3,4,5,6,7,8,9,10,15,20 --epochs-per-p 100 --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 25,30,40,50       --epochs-per-p 20 --alpha 1e-5 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p2to20_square/$TAG/checkpoints/best_model_p20.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 60,70,80,90,100   --epochs-per-p 20 --alpha 1e-5 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p25to50_square/$TAG/checkpoints/best_model_p50.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 150,200,300,400,500     --epochs-per-p 20 --alpha 1e-5 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p60to100_square/$TAG/checkpoints/best_model_p100.pt --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 600,700,800,900,1000    --epochs-per-p 20 --alpha 1e-5 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p150to500_square/$TAG/checkpoints/best_model_p500.pt --run-tag $TAG

# ── fix_alpha_band1_1e-2: band 1 uses fixed alpha=1e-2, bands 2-5 keep normal defaults ──
TAG=fix_alpha_band1_1e-2
python experiment/experiment_6_3.py $COMMON --p-values 2,3,4,5,6,7,8,9,10,15,20 --epochs-per-p 100 --alpha 1e-2 --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 25,30,40,50       --epochs-per-p 20 --alpha 1e-2 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p2to20_square/$TAG/checkpoints/best_model_p20.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 60,70,80,90,100   --epochs-per-p 20 --alpha 1e-3 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p25to50_square/$TAG/checkpoints/best_model_p50.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 150,200,300,400,500     --epochs-per-p 20 --alpha 1e-4 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p60to100_square/$TAG/checkpoints/best_model_p100.pt --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 600,700,800,900,1000    --epochs-per-p 20 --alpha 1e-5 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p150to500_square/$TAG/checkpoints/best_model_p500.pt --run-tag $TAG

# ── fix_alpha_band1_1e-5: band 1 uses fixed alpha=1e-5, bands 2-5 keep normal defaults ──
TAG=fix_alpha_band1_1e-5
python experiment/experiment_6_3.py $COMMON --p-values 2,3,4,5,6,7,8,9,10,15,20 --epochs-per-p 100 --alpha 1e-5 --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 25,30,40,50       --epochs-per-p 20 --alpha 1e-2 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p2to20_square/$TAG/checkpoints/best_model_p20.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 60,70,80,90,100   --epochs-per-p 20 --alpha 1e-3 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p25to50_square/$TAG/checkpoints/best_model_p50.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 150,200,300,400,500     --epochs-per-p 20 --alpha 1e-4 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p60to100_square/$TAG/checkpoints/best_model_p100.pt --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 600,700,800,900,1000    --epochs-per-p 20 --alpha 1e-5 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p150to500_square/$TAG/checkpoints/best_model_p500.pt --run-tag $TAG

# ── alpha_schedule_band2: adaptive alpha-schedule for band 2 (band 2 only) ──
python experiment/experiment_6_3.py $COMMON --p-values 25,30,40,50 --epochs-per-p 20 --alpha-schedule auto --lr 1e-5 --scheduler none --resume-checkpoint $BAND1_CKPT --run-tag alpha_schedule_band2

# ── fewer_bands: combine bands 2-5 into 2 bands ──
TAG=fewer_bands
python experiment/experiment_6_3.py $COMMON --p-values 2,3,4,5,6,7,8,9,10,15,20 --epochs-per-p 100 --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 25,30,40,50,60,70,80,90,100           --epochs-per-p 20 --alpha 1e-2 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p2to20_square/$TAG/checkpoints/best_model_p20.pt    --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 150,200,300,400,500,600,700,800,900,1000 --epochs-per-p 20 --alpha 1e-4 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p25to100_square/$TAG/checkpoints/best_model_p100.pt --run-tag $TAG

# ── more_epochs: double epochs-per-p in bands 2-5 (40 instead of 20) ──
TAG=more_epochs
python experiment/experiment_6_3.py $COMMON --p-values 2,3,4,5,6,7,8,9,10,15,20 --epochs-per-p 100 --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 25,30,40,50       --epochs-per-p 40 --alpha 1e-2 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p2to20_square/$TAG/checkpoints/best_model_p20.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 60,70,80,90,100   --epochs-per-p 40 --alpha 1e-3 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p25to50_square/$TAG/checkpoints/best_model_p50.pt   --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 150,200,300,400,500     --epochs-per-p 40 --alpha 1e-4 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p60to100_square/$TAG/checkpoints/best_model_p100.pt --run-tag $TAG
python experiment/experiment_6_3.py $COMMON --p-values 600,700,800,900,1000    --epochs-per-p 40 --alpha 1e-5 --lr 1e-5 --scheduler none --resume-checkpoint outputs/expr_6_3_iter/p150to500_square/$TAG/checkpoints/best_model_p500.pt --run-tag $TAG
