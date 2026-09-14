#!/usr/bin/env bash
set -u

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 GPU_ID SEED RUN_ROOT" >&2
    exit 2
fi

gpu_id="$1"
seed="$2"
run_root="$3"
source_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="$gpu_id"
export MPLBACKEND=Agg

mkdir -p "$run_root"

for example in arctan aronsson_square aronsson_disc; do
    target="$run_root/outputs/expr_6_2/$example/seed$seed"
    if [ -e "$target" ]; then
        echo "Refusing to overwrite existing run: $target" >&2
        exit 3
    fi

    echo "Starting example=$example seed=$seed gpu=$gpu_id"
    python -u "$source_root/experiment/experiment_6_2.py" \
        --example "$example" \
        --seed "$seed" \
        --run-tag "seed$seed" \
        --save-npy \
        --no-save \
        --output-dir "$run_root/outputs"
    status=$?
    echo "Finished example=$example seed=$seed status=$status"
    if [ "$status" -ne 0 ]; then
        exit "$status"
    fi
done
