# Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

# Scripts

- **`gdo.py`** — Runs the GDO algorithm to learn Laplacian representation eigenvectors. Use tyro CLI for options (e.g. grid size, walls, `k`, training steps).

- **`exps.py`** — Sweeps walls/seeds; set **`--transition-noise`** (η). Default **`--save-dir`** is a subfolder of **`experiments/`** named from grid, wall sweep, seeds, and η; **`--models-dir`** defaults to **`<save_dir>/models`**.

- **`exp_eta_w0.py`** — **w = 0**, sweeps **`--etas`**. Default **`--save-dir`** is under **`experiments/`** (grid, ``w0``, seeds, ``k``); models default to **`<save_dir>/models`**.

- **`visualize_graph_env.py`** — Generates a multi-page PDF that sweeps wall counts and shows, for each environment, the gridworld with walls and the corresponding NetworkX graph. Example:

```bash
python3 visualize_graph_env.py --rows 30 --cols 30 --start-walls 5 --n-sweeps 10 --output graph_env_wall_sweep.pdf
```

# run_all.sh

One entry point to train GDO and run experiments with the same config. Edit the variables at the top of the script, then run `./run_all.sh`.

- **Walls and step:** Set `MAX_WALLS` (e.g. `50`) and `STEP_WALLS` (e.g. `3`). The script sweeps wall counts **0, STEP_WALLS, 2×STEP_WALLS, …, MAX_WALLS** (e.g. 0, 3, 6, …, 50). Both GDO training and `exps.py` use these values.
- Other options: `ROWS`, `COLS`, `N_SEEDS`. GDO is run only for missing checkpoints (`gdo_n…_w…_s….pt`); existing files are skipped.
- **Stochastic env / not overwriting past runs:** Set `TRANSITION_NOISE` in `run_all.sh` (e.g. `0.1`). That value is η in `P=(1-η)P_det + η·Unif(all states)` (global uniform teleport); `0` is deterministic. The script appends `_eps0p1` to the experiment folder name so plots and a **new** directory tree do not overwrite deterministic runs (`TRANSITION_NOISE=0` keeps the original path with no suffix). Checkpoints live under `experiments/.../models/`; figures and `data.npz` stay in the experiment root. For manual runs: `python exps.py --save-dir experiments/my_stoch_run --transition-noise 0.1 ...` (models default to `experiments/my_stoch_run/models/`). Override with `--models-dir` if needed.
