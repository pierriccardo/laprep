# Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

# Scripts

- **`gdo.py`** — Runs the GDO algorithm to learn Laplacian representation eigenvectors. Use tyro CLI for options (e.g. grid size, walls, `k`, training steps).

- **`exps.py`** — Runs the experiments: sweeps over wall counts and seeds, compares analytical vs GDO eigenvectors, and saves error plots and eigenvector heatmaps. Options include grid size, number of walls, seeds, and output directories.

# run_all.sh

One entry point to train GDO and run experiments with the same config. Edit the variables at the top of the script, then run `./run_all.sh`.

- **Walls and step:** Set `MAX_WALLS` (e.g. `50`) and `STEP_WALLS` (e.g. `3`). The script sweeps wall counts **0, STEP_WALLS, 2×STEP_WALLS, …, MAX_WALLS** (e.g. 0, 3, 6, …, 50). Both GDO training and `exps.py` use these values.
- Other options: `ROWS`, `COLS`, `N_SEEDS`, and `SKIP_TRAIN` (set to `1` to skip training and only run exps with existing checkpoints).
