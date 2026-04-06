#!/bin/bash
# Single entry point: train GDO for all (walls, seed), then run exps with matching args.
# Edit the variables below, then: ./run_all.sh

set -e

# === Config (same semantics as exps + gdo) ===
ROWS=10
COLS=10
MAX_WALLS=20
STEP_WALLS=4
N_SEEDS=3
POLICY="random"
# Stochastic transitions (0 = deterministic, same as before). Non-zero → new subfolder so you do not overwrite prior exps.
TRANSITION_NOISE=0
NOISE_TAG=$(python3 -c "n=float('${TRANSITION_NOISE:-0}'); print('' if n==0 else '_eps'+str(n).replace('.','p'))")

# exps uses range(0, n_walls, step_walls), so n_walls = MAX_WALLS+1 gives walls 0, STEP_WALLS, ..., MAX_WALLS
N_WALLS_ARG=$((MAX_WALLS + 1))

# Working dir under experiments/ (same pattern as exps.py default --save-dir)
EXP_DIR="experiments/exp_${ROWS}x${COLS}_w${MAX_WALLS}_step${STEP_WALLS}_policy${POLICY}_nseeds${N_SEEDS}${NOISE_TAG}"
MODELS_DIR="${EXP_DIR}/models"
mkdir -p "$EXP_DIR" "$MODELS_DIR"

echo "=========================================="
echo "Experiment dir: $EXP_DIR"
echo "=========================================="

echo "GDO checkpoints (train only if missing) in $MODELS_DIR"
for walls in $(seq 0 "$STEP_WALLS" "$MAX_WALLS"); do
    for seed in $(seq 0 $((N_SEEDS - 1))); do
        CKPT="${MODELS_DIR}/gdo_n${ROWS}_m${COLS}_w${walls}_s${seed}${NOISE_TAG}.pt"
        if [ -f "$CKPT" ]; then
            echo "Skip (exists): $CKPT"
        else
            echo "Train: walls=${walls}, seed=${seed}"
            python3 gdo.py \
                --n "$ROWS" \
                --m "$COLS" \
                --n_walls "$walls" \
                --seed "$seed" \
                --k 20 \
                --log_freq 1999 \
                --train-steps 10000 \
                --save-dir "$MODELS_DIR" \
                --transition-noise "$TRANSITION_NOISE"
        fi
    done
done

echo "=========================================="
echo "Running exps (plots + data.npz in $EXP_DIR)"
echo "=========================================="

python3 exps.py \
    --rows "$ROWS" \
    --cols "$COLS" \
    --policy "$POLICY" \
    --n-walls "$N_WALLS_ARG" \
    --step-walls "$STEP_WALLS" \
    --n-seeds "$N_SEEDS" \
    --save-dir "$EXP_DIR" \
    --transition-noise "$TRANSITION_NOISE"

