#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    printf 'Usage: %s OUTPUT_DIR [SEED]\n' "$0" >&2
    exit 2
fi

output_dir="$1"
seed="${2:-1234}"
repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
script="$repo_root/experiment/experiment_6_3_3_distance_boundary_disc.py"

export MPLBACKEND=Agg
mkdir -p "$output_dir"

# Stage 1 is the validation/early-stopping stage. The p=2,...,10
# checkpoints are the values retained for the manuscript figure.
python -u "$script" \
    --p-values 2,3,4,5,6,7,8,9,10 \
    --epochs-per-p 100 \
    --lr 1e-3 \
    --scheduler cosine \
    --scheduler-end-factor 0.01 \
    --alpha-schedule auto \
    --bc-loss-threshold 1e-4 \
    --pde-loss-threshold 1e-3 \
    --interior-grid 1000 \
    --seed "$seed" \
    --output-dir "$output_dir" \
    --save-predictions

checkpoint="$output_dir/expr_6_3_iter_distance_boundary_disc/p2to10_disc/checkpoints/final_model_p10.pt"

# Stages 2a and 2b were run separately in the reported experiment. Keeping
# this split preserves the random-number reset and checkpoint sequence.
python -u "$script" \
    --p-values 11,12,13,14,15 \
    --epochs-per-p 50 \
    --lr 1e-4 \
    --scheduler cosine \
    --scheduler-end-factor 0.1 \
    --alpha 1e-2 \
    --val-split 0 \
    --interior-grid 500 \
    --seed "$seed" \
    --resume-checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --save-predictions

checkpoint="$output_dir/expr_6_3_iter_distance_boundary_disc/p11to15_disc/checkpoints/final_model_p15.pt"
python -u "$script" \
    --p-values 16,17,18,19,20 \
    --epochs-per-p 50 \
    --lr 1e-4 \
    --scheduler cosine \
    --scheduler-end-factor 0.1 \
    --alpha 1e-2 \
    --val-split 0 \
    --interior-grid 500 \
    --seed "$seed" \
    --resume-checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --save-predictions

# Full-grid p=20 refinement.
checkpoint="$output_dir/expr_6_3_iter_distance_boundary_disc/p16to20_disc/checkpoints/final_model_p20.pt"
python -u "$script" \
    --p-values 20 \
    --epochs-per-p 50 \
    --lr 1e-4 \
    --scheduler cosine \
    --scheduler-end-factor 0.1 \
    --alpha 1e-2 \
    --val-split 0 \
    --interior-grid 1000 \
    --seed "$seed" \
    --resume-checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --save-predictions

checkpoint="$output_dir/expr_6_3_iter_distance_boundary_disc/p20to20_disc/checkpoints/final_model_p20.pt"
python -u "$script" \
    --p-values 25,30,40,50 \
    --epochs-per-p 50 \
    --lr 1e-4 \
    --scheduler cosine \
    --scheduler-end-factor 0.1 \
    --alpha 1e-2 \
    --val-split 0 \
    --interior-grid 500 \
    --seed "$seed" \
    --resume-checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --save-predictions

checkpoint="$output_dir/expr_6_3_iter_distance_boundary_disc/p25to50_disc/checkpoints/final_model_p50.pt"
python -u "$script" \
    --p-values 60,70,80,90,100 \
    --epochs-per-p 50 \
    --lr 1e-4 \
    --scheduler cosine \
    --scheduler-end-factor 0.1 \
    --alpha 1e-2 \
    --val-split 0 \
    --interior-grid 500 \
    --seed "$seed" \
    --resume-checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --save-predictions

checkpoint="$output_dir/expr_6_3_iter_distance_boundary_disc/p60to100_disc/checkpoints/final_model_p100.pt"
python -u "$script" \
    --p-values 150,200,300,400,500 \
    --epochs-per-p 50 \
    --lr 1e-5 \
    --scheduler none \
    --alpha 1e-2 \
    --val-split 0 \
    --interior-grid 500 \
    --seed "$seed" \
    --resume-checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --save-predictions

python -u "$repo_root/experiment/collect_section_6_3_3_distance_boundary_metrics.py" \
    --run-root "$output_dir" \
    --output "$output_dir/combined_metrics.csv"
