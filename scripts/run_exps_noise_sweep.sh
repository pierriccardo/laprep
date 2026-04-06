#!/bin/bash
# Run the same exps.py sweep for several transition_noise values (does not touch deterministic runs).
# Edit the shared config below, then: chmod +x run_exps_noise_sweep.sh && ./run_exps_noise_sweep.sh

set -e

# === Same semantics as run_all.sh ===
ROWS=15
COLS=15
MAX_WALLS=50
STEP_WALLS=3
N_SEEDS=5
POLICY="uniform"

# Noise levels to sweep (stochastic transition mixing)
NOISE_LEVELS=(0.1 0.5 0.7)

N_WALLS_ARG=$((MAX_WALLS + 1))

for TRANSITION_NOISE in "${NOISE_LEVELS[@]}"; do
  NOISE_TAG=$(python3 -c "n=float('${TRANSITION_NOISE}'); print('' if n==0 else '_eps'+str(n).replace('.','p'))")
  EXP_DIR="experiments/exp_${ROWS}x${COLS}_w${MAX_WALLS}_step${STEP_WALLS}_policy${POLICY}_nseeds${N_SEEDS}${NOISE_TAG}"
  MODELS_DIR="${EXP_DIR}/models"
  mkdir -p "$EXP_DIR" "$MODELS_DIR"

  echo "=========================================="
  echo "transition_noise=${TRANSITION_NOISE}"
  echo "Experiment dir: $EXP_DIR"
  echo "=========================================="

  echo "GDO checkpoints (train only if missing): noise=${TRANSITION_NOISE} → $MODELS_DIR"
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

  echo "Running exps ..."
  python3 exps.py \
    --rows "$ROWS" \
    --cols "$COLS" \
    --policy "$POLICY" \
    --n-walls "$N_WALLS_ARG" \
    --step-walls "$STEP_WALLS" \
    --n-seeds "$N_SEEDS" \
    --save-dir "$EXP_DIR" \
    --transition-noise "$TRANSITION_NOISE"

  echo ""
done

ETA_DIR="experiments/exp_${ROWS}x${COLS}_w0_eta_nseeds${N_SEEDS}_k20"
echo "Wall/noise exps: experiments/exp_${ROWS}x${COLS}_w${MAX_WALLS}_step${STEP_WALLS}_policy${POLICY}_nseeds${N_SEEDS}_eps*"
echo "η sweep at w=0 (exp_eta_w0.py default): $ETA_DIR"
