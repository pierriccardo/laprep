"""
Inspection script: eigengap analysis for exp_task_transfer experiments.

Loads task_transfer_data.npz from a given experiment folder and reports:
  - How many eigengaps (per k, per wall count, overall) are exactly zero or near-zero
  - Per-k breakdown: count of zeros, min, mean, fraction below threshold
  - Per-wall breakdown: how many k-slots are zero at each wall count

Usage:
    python inspect_task_transfer_eigengaps.py
    python inspect_task_transfer_eigengaps.py --exp-dir experiments/exp_task_transfer_15x15_...
    python inspect_task_transfer_eigengaps.py --threshold 1e-6
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect eigengaps from exp_task_transfer.")
    parser.add_argument(
        "--exp-dir",
        default="experiments/exp_task_transfer_10x10_w29_step1_nseeds5",
        help="Path to the experiment directory containing task_transfer_data.npz",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-8,
        help="Values below this are considered zero (default: 1e-8)",
    )
    args = parser.parse_args()

    npz_path = os.path.join(args.exp_dir, "task_transfer_data.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Data file not found: {npz_path}")

    d = np.load(npz_path)
    print(f"Loaded: {npz_path}")
    print(f"Keys: {list(d.files)}\n")

    if "eigengaps_vs_k" not in d.files:
        print("No eigengaps_vs_k array found. Re-run exp_task_transfer.py to regenerate data.")
        return

    egk = d["eigengaps_vs_k"]        # (n_seeds, W, k)
    wall_values = d["wall_values"]    # (W,)
    k = int(egk.shape[2])
    n_seeds, W = egk.shape[0], egk.shape[1]
    thr = args.threshold

    print(f"Array shape (n_seeds={n_seeds}, W={W}, k={k})")
    print(f"Zero threshold: {thr}\n")

    # Flatten for global stats
    flat = egk.reshape(-1)
    n_total = np.sum(np.isfinite(flat))
    n_exact_zero = int(np.sum(flat == 0.0))
    n_near_zero = int(np.sum(flat[np.isfinite(flat)] < thr))

    print("=" * 60)
    print("Global summary")
    print("=" * 60)
    print(f"  Total finite values  : {n_total}")
    print(f"  Exactly zero         : {n_exact_zero}  ({100*n_exact_zero/n_total:.1f}%)")
    print(f"  Below threshold      : {n_near_zero}  ({100*n_near_zero/n_total:.1f}%)")
    print(f"  Min                  : {_fmt(float(np.nanmin(flat)))}")
    print(f"  Mean                 : {_fmt(float(np.nanmean(flat)))}")
    print(f"  Max                  : {_fmt(float(np.nanmax(flat)))}")

    # Per-k breakdown
    print("\n" + "=" * 60)
    print(f"Per-k breakdown  (each k pooled over {n_seeds} seeds × {W} walls = {n_seeds*W} values)")
    print("=" * 60)
    header = f"{'k':>4}  {'min':>12}  {'mean':>12}  {'max':>12}  {'#zero':>6}  {'#<thr':>6}  {'%<thr':>7}"
    print(header)
    print("-" * len(header))
    for ki in range(k):
        vals = egk[:, :, ki].reshape(-1)
        finite = vals[np.isfinite(vals)]
        n_f = len(finite)
        if n_f == 0:
            continue
        n_z = int(np.sum(finite == 0.0))
        n_t = int(np.sum(finite < thr))
        print(
            f"{ki+1:>4}  {_fmt(float(np.min(finite))):>12}  {_fmt(float(np.mean(finite))):>12}"
            f"  {_fmt(float(np.max(finite))):>12}  {n_z:>6}  {n_t:>6}  {100*n_t/n_f:>6.1f}%"
        )

    # Per-wall breakdown: for each wall count, how many (seed, k) slots are near-zero
    print("\n" + "=" * 60)
    print(f"Per-wall breakdown  (each wall pooled over {n_seeds} seeds × {k} k-slots = {n_seeds*k} values)")
    print("=" * 60)
    header2 = f"{'walls':>6}  {'min gap':>12}  {'mean gap':>12}  {'#<thr':>6}  {'%<thr':>7}"
    print(header2)
    print("-" * len(header2))
    for wi, w in enumerate(wall_values):
        vals = egk[:, wi, :].reshape(-1)
        finite = vals[np.isfinite(vals)]
        n_f = len(finite)
        if n_f == 0:
            continue
        n_t = int(np.sum(finite < thr))
        print(
            f"{int(w):>6}  {_fmt(float(np.min(finite))):>12}  {_fmt(float(np.mean(finite))):>12}"
            f"  {n_t:>6}  {100*n_t/n_f:>6.1f}%"
        )

    # Locate the worst offenders: (seed, wall, k) triples with smallest eigengap
    print("\n" + "=" * 60)
    print("Top 10 smallest eigengaps (seed, wall, k, value)")
    print("=" * 60)
    flat_idx = np.argsort(egk.reshape(-1))
    for rank, idx in enumerate(flat_idx[:10]):
        si, wi, ki = np.unravel_index(idx, egk.shape)
        val = egk[si, wi, ki]
        if not np.isfinite(val):
            continue
        print(f"  #{rank+1:>2}  seed={si}  walls={int(wall_values[wi]):>3}  k={ki+1:>3}  gap={_fmt(val)}")


if __name__ == "__main__":
    main()
