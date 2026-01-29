#!/bin/bash
# Single entry point: train GDO for all (walls, seed), then run exps with matching args.
# Edit the variables below, then: ./run_all.sh

set -e

# === Config (same semantics as exps + gdo) ===
ROWS=15
COLS=15
MAX_WALLS=50
STEP_WALLS=3
N_SEEDS=5
SKIP_TRAIN=1  # Set to 1 to skip GDO training (only run exps, use existing checkpoints)

# Experiment folder: all outputs (models, imgs, data.npz) go here
EXP_DIR="experiments/exp_${ROWS}x${COLS}_w${MAX_WALLS}_step${STEP_WALLS}_nseeds${N_SEEDS}"
mkdir -p "$EXP_DIR"

# exps uses range(0, n_walls, step_walls), so n_walls = MAX_WALLS+1 gives walls 0, STEP_WALLS, ..., MAX_WALLS
N_WALLS_ARG=$((MAX_WALLS + 1))

echo "=========================================="
echo "Experiment dir: $EXP_DIR"
echo "=========================================="

if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "Training GDO: grid ${ROWS}x${COLS}, walls 0..${MAX_WALLS} step ${STEP_WALLS}, seeds 0..$((N_SEEDS - 1))"
    for walls in $(seq 0 "$STEP_WALLS" "$MAX_WALLS"); do
        for seed in $(seq 0 $((N_SEEDS - 1))); do
            echo "GDO: walls=${walls}, seed=${seed}"
            python3 gdo.py \
                --n "$ROWS" \
                --m "$COLS" \
                --n_walls "$walls" \
                --seed "$seed" \
                --k 20 \
                --log_freq 1999 \
                --train-steps 10000 \
                --save-dir "$EXP_DIR"
        done
    done
else
    echo "Skipping GDO training (SKIP_TRAIN=1)"
fi

echo "=========================================="
echo "Running exps (plots + data.npz in $EXP_DIR)"
echo "=========================================="

python3 exps.py \
    --rows "$ROWS" \
    --cols "$COLS" \
    --n-walls "$N_WALLS_ARG" \
    --step-walls "$STEP_WALLS" \
    --n-seeds "$N_SEEDS" \
    --save-dir "$EXP_DIR"

