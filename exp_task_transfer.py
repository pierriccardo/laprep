"""
Experiment: task transfer for policy evaluation using Laplacian representations.

The Laplacian representation encodes only the geometry of the state space — it is built from
the transition structure under a fixed policy and is independent of the reward function.
This experiment exploits that property:

  1. A GDO representation is trained on a default task (goal at the bottom-right corner).
     The same pre-trained Z is then reused for 5 different goal positions WITHOUT retraining.

  2. For each goal, the true differential value function v_true is computed analytically
     (average-reward Bellman equation, uniform policy), and the approximation error is
         ||v_true - Z (Z^T Phi Z)^{-1} Z^T Phi v_true||_Phi
     for both the analytical eigenvectors and the GDO learned eigenvectors.

  3. The sweep variable is the wall count: more walls -> less connected graph -> larger
     eigenvalues and potentially larger eigengap.

  4. The eigengap lambda_{k+1} - lambda_k (gap at the truncation point k) is tracked
     alongside errors to study whether a larger gap correlates with better approximation.

The uniform policy is used throughout: P_pi, phi, and the Laplacian are identical for all
goals (only the reward vector changes), so the representation needs to be computed only once
per (wall_count, seed) and reused across all goal evaluations.

Prerequisites:
  GDO checkpoints must already exist under models_dir (trained by run_all.sh or gdo.py
  with the default goal). This experiment does not retrain the representation.

Output (under save_dir):
  errors_per_goal_analytical.pdf  -- analytical error vs walls, one curve per goal
  errors_per_goal_gdo.pdf         -- GDO error vs walls, one curve per goal
  mean_error_vs_walls.pdf         -- mean over goals: analytical vs GDO
  eigengap_vs_walls.pdf           -- lambda_{k+1} - lambda_k vs walls
  error_vs_eigengap.pdf           -- error (mean over goals) vs eigengap, analytical + GDO
  task_transfer_data.npz          -- raw arrays for further analysis
"""

from __future__ import annotations

import dataclasses
import os
import random
import subprocess
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.ticker import ScalarFormatter
from scipy import stats
from tqdm import tqdm

from graph_env import GraphEnv
from gdo import Config as LapRepConfig, LapRep
from utils import (
    average_reward_value,
    compute_stationary_distribution,
    policy_expected_reward,
    policy_induced_transition_matrix,
    uniform_policy,
)


# ==================================================
# Helpers
# ==================================================

def _stoch_suffix(noise: float) -> str:
    return "" if noise <= 0 else f"_eps{str(noise).replace('.', 'p')}"


def _log_axis_exponent_once(ax, which: str = "y") -> None:
    fmt = ScalarFormatter()
    fmt.set_useOffset(True)
    fmt.set_scientific(True)
    fmt.set_useMathText(True)
    fmt.set_powerlimits((0, 0))
    if which == "y":
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_offset_position("left")
    else:
        ax.xaxis.set_major_formatter(fmt)


def _default_goals(n: int, m: int) -> List[Tuple[int, int]]:
    """5 goal positions spread across the grid: 4 corners + center."""
    return [
        (0, 0),
        (0, m - 1),
        (n - 1, 0),
        (n - 1, m - 1),   # default training goal
        (n // 2, m // 2),
    ]


def _goal_label(goal: Tuple[int, int]) -> str:
    return f"({goal[0]},{goal[1]})"


# ==================================================
# Config
# ==================================================

@dataclasses.dataclass
class Args:
    n_seeds: int = 5
    n_walls: int = 30
    """Sweep wall counts in range(0, n_walls, step_walls)."""
    step_walls: int = 1
    rows: int = 10
    cols: int = 10
    k: int = 20
    """Number of eigenvectors / subspace dimension."""
    seed: int = 42
    save_dir: Optional[str] = None
    """Output directory. Defaults to experiments/exp_task_transfer_{rows}x{cols}_..."""
    models_dir: Optional[str] = None
    """Directory with pre-trained GDO checkpoints and analytical caches.
    Defaults to <save_dir>/models. Point this to an existing experiment's models/
    directory to reuse trained checkpoints, e.g.:
      --models-dir experiments/exp_10x10_w29_step1_policyuniform_nseeds5/models
    """
    transition_noise: float = 0.0
    """η in P=(1-η)P_det + η Unif(all states). Must match the GDO checkpoints."""
    train_missing: bool = True
    """If True, automatically train GDO for any missing checkpoint using gdo.py."""
    train_steps: int = 10000
    log_freq: int = 1999


def _ensure_gdo_checkpoint(
    path: str,
    *,
    rows: int,
    cols: int,
    w: int,
    seed: int,
    eta: float,
    k: int,
    models_dir: str,
    train_missing: bool,
    train_steps: int,
    log_freq: int,
) -> None:
    if os.path.isfile(path):
        return
    if not train_missing:
        raise FileNotFoundError(
            f"GDO checkpoint not found: {path}\n"
            "Re-run with --train-missing to train automatically, or train manually:\n"
            f"  python gdo.py --n {rows} --m {cols} --n-walls {w} --seed {seed} --k {k} \\\n"
            f"    --train-steps 10000 --log-freq 1999 --save-dir {models_dir}"
            + (f" --transition-noise {eta}" if eta > 0 else "")
            + "\nOr point --models-dir to a directory with existing checkpoints."
        )
    repo_root = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(repo_root, "gdo.py"),
        "--n", str(rows),
        "--m", str(cols),
        "--n-walls", str(w),
        "--seed", str(seed),
        "--k", str(k),
        "--log-freq", str(log_freq),
        "--train-steps", str(train_steps),
        "--save-dir", models_dir,
        "--transition-noise", str(eta),
    ]
    print(f"Training GDO → {path}", flush=True)
    subprocess.run(cmd, check=True)
    if not os.path.isfile(path):
        raise RuntimeError(f"gdo.py finished but checkpoint not found: {path}")


def _write_task_transfer_markdown(
    path: str,
    *,
    args: Args,
    wall_values: list,
    goals: List[Tuple[int, int]],
    training_goal: Tuple[int, int],
    analytical_errors: np.ndarray,   # (n_seeds, W, n_goals)
    gdo_errors: np.ndarray,          # (n_seeds, W, n_goals)
    eigengaps: np.ndarray,           # (n_seeds, W)
    lambda2s: np.ndarray,            # (n_seeds, W)
    eigengap_stats: Optional[dict] = None,
    eigengaps_vs_k: Optional[np.ndarray] = None,  # (n_seeds, W, k)
    n_sample_walls: int = 8,
    n_sample_k: int = 10,
) -> None:
    """Write Markdown summary tables for the task-transfer experiment.

    Produces:
      - Parameters section
      - Per-wall table (sampled): eigengap, mean analytical/GDO error over goals
      - Per-goal table: mean analytical/GDO error over all walls and seeds

    Each cell uses mean, or mean ± (p90-p10)/2 when the band is nonzero.
    """

    def fmt(x: float) -> str:
        return f"{float(x):.4g}"

    def mean_pm_band(vals: np.ndarray) -> str:
        m = float(np.nanmean(vals))
        p10 = float(np.nanpercentile(vals, 10))
        p90 = float(np.nanpercentile(vals, 90))
        half = (p90 - p10) / 2.0
        if np.isclose(half, 0.0, rtol=0.0, atol=1e-15):
            return rf"${fmt(m)}$"
        return rf"${fmt(m)} \pm {fmt(half)}$"

    W = len(wall_values)
    n_sample = min(n_sample_walls, W)
    sample_wi = sorted({int(round(x)) for x in np.linspace(0, W - 1, n_sample)})

    eta = getattr(args, "transition_noise", 0.0)


    lines: List[str] = [
        "# Task Transfer — Results",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        rf"| Grid | ${args.rows} \times {args.cols}$ |",
        rf"| Wall sweep | $0, {args.step_walls}, \ldots, {args.n_walls - 1}$ |",
        rf"| Seeds | {args.n_seeds} |",
        rf"| Subspace dim $k$ | {args.k} |",
        rf"| Transition noise $\eta$ | {eta:g} |",
        rf"| Goals evaluated | {len(goals)} (4 corners + center) |",
        rf"| Training goal (GDO) | $({training_goal[0]},{training_goal[1]})$ |",
        "",
        r"Each cell is $\mathrm{mean}$, or $\mathrm{mean} \pm \tfrac{1}{2}(p_{90}-p_{10})$"
        r" when that band width is nonzero.",
        "",
        "## Per-wall results (sampled)",
        "",
        r"Errors are averaged over all goals and seeds.",
        "",
        r"| $w$ | eigengap $\lambda_{k+1}-\lambda_k$ | $\lambda_2$ | $\varepsilon_{\mathrm{ana}}$ | $\varepsilon_{\mathrm{GDO}}$ |",
        "|---|---:|---:|---:|---:|",
    ]

    # mean errors over goals for each (seed, wi)
    ae_mean_goals = np.nanmean(analytical_errors, axis=2)  # (n_seeds, W)
    ge_mean_goals = np.nanmean(gdo_errors, axis=2)

    for wi in sample_wi:
        w = wall_values[wi]
        lines.append(
            f"| {w} "
            f"| {mean_pm_band(eigengaps[:, wi])} "
            f"| {mean_pm_band(lambda2s[:, wi])} "
            f"| {mean_pm_band(ae_mean_goals[:, wi])} "
            f"| {mean_pm_band(ge_mean_goals[:, wi])} |"
        )

    lines += [
        "",
        "## Per-goal results (mean over all walls and seeds)",
        "",
        r"| Goal | $\varepsilon_{\mathrm{ana}}$ | $\varepsilon_{\mathrm{GDO}}$ |",
        "|---|---:|---:|",
    ]

    for gi, goal in enumerate(goals):
        label = rf"$({goal[0]},{goal[1]})$"
        if goal == training_goal:
            label += r" \*(train)"
        ae_vals = analytical_errors[:, :, gi].reshape(-1)
        ge_vals = gdo_errors[:, :, gi].reshape(-1)
        lines.append(f"| {label} | {mean_pm_band(ae_vals)} | {mean_pm_band(ge_vals)} |")

    if eigengap_stats is not None:
        s = eigengap_stats
        sw_note = "normal" if s["sw_p"] > 0.05 else "non-normal"
        t_note = "reject $H_0$" if s["t_p"] < 0.05 else "fail to reject $H_0$"
        lines += [
            "",
            "## Eigengap statistics",
            "",
            rf"Distribution of $\lambda_{{k+1}} - \lambda_k$ pooled over all seeds and wall counts ($n={s['n']}$ values).",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            rf"| $n$ | {s['n']} |",
            rf"| Min | ${fmt(s['min'])}$ |",
            rf"| Mean | ${fmt(s['mean'])}$ |",
            rf"| Std | ${fmt(s['std'])}$ |",
            rf"| Shapiro–Wilk $W$ | ${fmt(s['sw_stat'])}$ |",
            rf"| Shapiro–Wilk $p$ | ${fmt(s['sw_p'])}$ ({sw_note}) |",
            rf"| $t$-test statistic ($H_1: \mu > 0$, df$={s['t_df']}$) | ${fmt(s['t_stat'])}$ |",
            rf"| $t$-test $p$-value (one-sided) | ${fmt(s['t_p'])}$ ({t_note}) |",
        ]

    if eigengaps_vs_k is not None:
        k_dim = eigengaps_vs_k.shape[2]
        egk_flat = eigengaps_vs_k.reshape(-1, k_dim)
        egk_mean = np.nanmean(egk_flat, axis=0)
        egk_std = np.nanstd(egk_flat, axis=0)
        egk_min = np.nanmin(egk_flat, axis=0)
        k_sample = sorted({int(round(x)) for x in np.linspace(0, k_dim - 1, min(n_sample_k, k_dim))})
        lines += [
            "",
            "## Eigengap vs $k$ (analytical, mean over seeds and walls)",
            "",
            r"| $k$ | mean gap $\lambda_{k+1} - \lambda_k$ | std | min |",
            "|---:|---:|---:|---:|",
        ]
        for ki in k_sample:
            lines.append(rf"| {ki + 1} | ${fmt(egk_mean[ki])}$ | ${fmt(egk_std[ki])}$ | ${fmt(egk_min[ki])}$ |")

    lines += ["", "_Data saved in `task_transfer_data.npz`._", ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _default_save_dir(args: Args) -> str:
    noise_tag = _stoch_suffix(args.transition_noise)
    return (
        f"experiments/exp_task_transfer_{args.rows}x{args.cols}"
        f"_w{args.n_walls - 1}_step{args.step_walls}_nseeds{args.n_seeds}{noise_tag}"
    )


# ==================================================
# Main
# ==================================================

def main() -> None:
    args = tyro.cli(Args)
    if args.save_dir is None:
        args.save_dir = _default_save_dir(args)
    if args.models_dir is None:
        args.models_dir = os.path.join(args.save_dir, "models")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    plt.style.use("paper.mplstyle")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    goals = _default_goals(args.rows, args.cols)
    n_goals = len(goals)
    training_goal = (args.rows - 1, args.cols - 1)
    goal_labels = [_goal_label(g) for g in goals]

    wall_values = list(range(2, args.n_walls, args.step_walls))
    W = len(wall_values)

    # Result arrays: (n_seeds, W, n_goals) for errors, (n_seeds, W) for eigengap/lambda2
    analytical_errors = np.full((args.n_seeds, W, n_goals), np.nan)
    gdo_errors = np.full((args.n_seeds, W, n_goals), np.nan)
    eigengaps = np.full((args.n_seeds, W), np.nan)
    eigengaps_vs_k = np.full((args.n_seeds, W, args.k), np.nan)  # gap(j) = λ_{j+1} - λ_j, j=1..k
    lambda2s = np.full((args.n_seeds, W), np.nan)

    for wi, w in tqdm(list(enumerate(wall_values)), desc="walls"):
        for seed in range(args.n_seeds):

            # --------------------------------------------------
            # Base environment: default goal, used only for the
            # graph structure, policy, and Laplacian computation.
            # P_pi and phi are the same for all goals (uniform policy
            # + same graph => same transition dynamics).
            # --------------------------------------------------
            base_env = GraphEnv(
                n=args.rows,
                m=args.cols,
                n_walls=w,
                seed=seed,
                transition_noise=args.transition_noise,
            )
            policy = uniform_policy(base_env)
            P_pi = policy_induced_transition_matrix(base_env, policy)
            phi = compute_stationary_distribution(P_pi)
            Phi = np.diag(phi)

            # --------------------------------------------------
            # Analytical eigenvectors of the normalized Laplacian.
            # Cached to avoid recomputation across experiments.
            # eigvals has k+1 entries (0..k) so eigvals[k] = lambda_{k+1}.
            # --------------------------------------------------
            ana_path = os.path.join(
                args.models_dir,
                f"analytical_n{args.rows}_m{args.cols}_w{w}_s{seed}"
                f"{_stoch_suffix(args.transition_noise)}.npz",
            )
            if os.path.isfile(ana_path):
                d = np.load(ana_path)
                eigvals = d["eigvals"].real   # shape (k+1,)
                eigvecs = d["eigvecs"].real   # shape (n_states, k)
                Z_ana = eigvecs[:, : args.k]
            else:
                L = Phi - (Phi @ P_pi + P_pi.T @ Phi) / 2
                inv_phi = np.diag(1.0 / phi)
                L = inv_phi @ L
                eigvals_all, eigvecs_all = np.linalg.eig(L)
                idx = eigvals_all.argsort()
                eigvals_all = eigvals_all[idx].real
                eigvecs_all = eigvecs_all[:, idx].real
                Z_ana = eigvecs_all[:, : args.k]
                eigvals = eigvals_all[: args.k + 1]
                np.savez_compressed(ana_path, eigvals=eigvals, eigvecs=Z_ana)

            # Eigengap at chosen k: λ_{k+1} - λ_k
            eigengaps[seed, wi] = float(eigvals[args.k]) - float(eigvals[args.k - 1])
            # Eigengap for every j=1..k: gap(j) = λ_{j+1} - λ_j
            # eigvals has shape (k+1,): [λ_1, ..., λ_{k+1}]
            eigengaps_vs_k[seed, wi, :] = eigvals[1:] - eigvals[:-1]
            lambda2s[seed, wi] = float(eigvals[1])

            # --------------------------------------------------
            # GDO representation: pre-trained on default goal,
            # reused as-is for all goal positions (task transfer).
            # --------------------------------------------------
            gdo_path = os.path.join(
                args.models_dir,
                f"gdo_n{args.rows}_m{args.cols}_w{w}_s{seed}"
                f"{_stoch_suffix(args.transition_noise)}.pt",
            )
            _ensure_gdo_checkpoint(
                gdo_path,
                rows=args.rows,
                cols=args.cols,
                w=w,
                seed=seed,
                eta=args.transition_noise,
                k=args.k,
                models_dir=args.models_dir,
                train_missing=args.train_missing,
                train_steps=args.train_steps,
                log_freq=args.log_freq,
            )
            model = LapRep(LapRepConfig(n_states=base_env.n_states, k=args.k))
            model.load(gdo_path)
            Z_gdo = model.network(torch.arange(base_env.n_states)).detach().numpy()

            # --------------------------------------------------
            # Task transfer: evaluate on each goal position.
            # Only the reward (and therefore v_true) changes;
            # P_pi and phi are reused from the base environment.
            # --------------------------------------------------
            for gi, goal in enumerate(goals):
                goal_env = GraphEnv(
                    n=args.rows,
                    m=args.cols,
                    n_walls=w,
                    seed=seed,
                    goal=goal,
                    transition_noise=args.transition_noise,
                )
                r_pi = policy_expected_reward(goal_env, policy)
                v_true, _ = average_reward_value(P_pi, r_pi, phi)

                try:
                    v_pred_ana = Z_ana @ np.linalg.inv(Z_ana.T @ Phi @ Z_ana) @ (Z_ana.T @ Phi @ v_true)
                    v_pred_gdo = Z_gdo @ np.linalg.inv(Z_gdo.T @ Phi @ Z_gdo) @ (Z_gdo.T @ Phi @ v_true)

                    tmp_a = v_true - v_pred_ana
                    tmp_g = v_true - v_pred_gdo
                    analytical_errors[seed, wi, gi] = float(np.sqrt(tmp_a @ Phi @ tmp_a))
                    gdo_errors[seed, wi, gi] = float(np.sqrt(tmp_g @ Phi @ tmp_g))
                except np.linalg.LinAlgError as exc:
                    print(f"  LinAlgError at w={w}, seed={seed}, goal={goal}: {exc}")

    x = np.array(wall_values)

    # ==================================================
    # Plot 1: Analytical error vs walls, per goal
    # ==================================================
    plt.figure()
    for gi, label in enumerate(goal_labels):
        ls = "--" if goals[gi] == training_goal else "-"
        mean_e = np.nanmean(analytical_errors[:, :, gi], axis=0)
        p10 = np.nanpercentile(analytical_errors[:, :, gi], 10, axis=0)
        p90 = np.nanpercentile(analytical_errors[:, :, gi], 90, axis=0)
        star = " *train" if goals[gi] == training_goal else ""
        line, = plt.plot(x, mean_e, linestyle=ls, label=f"goal {label}{star}")
        plt.fill_between(x, p10, p90, alpha=0.12, color=line.get_color())
    plt.yscale("log")
    plt.xlabel("Number of Walls")
    plt.ylabel(r"Error $\|v - \hat{v}\|_\Phi$")
    plt.title(
        rf"Task Transfer — Analytical, ${args.rows}\times{args.cols}$, "
        rf"$k={args.k}$, {args.n_seeds} seeds",
        fontsize=9,
    )
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "errors_per_goal_analytical.pdf"), format="pdf")
    plt.close()

    # ==================================================
    # Plot 2: GDO error vs walls, per goal
    # ==================================================
    plt.figure()
    for gi, label in enumerate(goal_labels):
        ls = "--" if goals[gi] == training_goal else "-"
        mean_e = np.nanmean(gdo_errors[:, :, gi], axis=0)
        p10 = np.nanpercentile(gdo_errors[:, :, gi], 10, axis=0)
        p90 = np.nanpercentile(gdo_errors[:, :, gi], 90, axis=0)
        star = " *train" if goals[gi] == training_goal else ""
        line, = plt.plot(x, mean_e, linestyle=ls, label=f"goal {label}{star}")
        plt.fill_between(x, p10, p90, alpha=0.12, color=line.get_color())
    plt.yscale("log")
    plt.xlabel("Number of Walls")
    plt.ylabel(r"Error $\|v - \hat{v}\|_\Phi$")
    plt.title(
        rf"Task Transfer — GDO, ${args.rows}\times{args.cols}$, "
        rf"$k={args.k}$, {args.n_seeds} seeds",
        fontsize=9,
    )
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "errors_per_goal_gdo.pdf"), format="pdf")
    plt.close()

    # ==================================================
    # Plot 3: Mean error vs walls (analytical vs GDO, averaged over goals)
    # ==================================================
    mean_ana = np.nanmean(analytical_errors, axis=2)   # (n_seeds, W)
    mean_gdo = np.nanmean(gdo_errors, axis=2)

    plt.figure()
    line_a, = plt.plot(x, np.nanmean(mean_ana, axis=0), label="Analytical (mean over goals)")
    plt.fill_between(
        x,
        np.nanpercentile(mean_ana, 10, axis=0),
        np.nanpercentile(mean_ana, 90, axis=0),
        alpha=0.25, color=line_a.get_color(),
    )
    line_g, = plt.plot(x, np.nanmean(mean_gdo, axis=0), label="GDO (mean over goals)")
    plt.fill_between(
        x,
        np.nanpercentile(mean_gdo, 10, axis=0),
        np.nanpercentile(mean_gdo, 90, axis=0),
        alpha=0.25, color=line_g.get_color(),
    )
    plt.yscale("log")
    plt.xlabel("Number of Walls")
    plt.ylabel(r"Error $\|v - \hat{v}\|_\Phi$")
    plt.title(
        rf"Task Transfer — Mean over goals, ${args.rows}\times{args.cols}$, "
        rf"$k={args.k}$, {args.n_seeds} seeds",
        fontsize=9,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "mean_error_vs_walls.pdf"), format="pdf")
    plt.close()

    # ==================================================
    # Plot 4: Eigengap vs walls
    # ==================================================
    plt.figure()
    mean_gap = np.nanmean(eigengaps, axis=0)
    p10_gap = np.nanpercentile(eigengaps, 10, axis=0)
    p90_gap = np.nanpercentile(eigengaps, 90, axis=0)
    plt.plot(x, mean_gap, label="Mean")
    plt.fill_between(x, p10_gap, p90_gap, alpha=0.3, label=f"10-90% ({args.n_seeds} seeds)")
    ax = plt.gca()
    _log_axis_exponent_once(ax, "y")
    plt.xlabel("Number of Walls")
    plt.ylabel(rf"Eigengap $\lambda_{{k+1}} - \lambda_k$ ($k={args.k}$)")
    plt.title(
        rf"Eigengap vs Walls, ${args.rows}\times{args.cols}$, "
        rf"$k={args.k}$, {args.n_seeds} seeds",
        fontsize=9,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "eigengap_vs_walls.pdf"), format="pdf")
    plt.close()

    # ==================================================
    # Plot 4b: Eigengap vs k (analytical, averaged over seeds and walls)
    # ==================================================
    k_vals = np.arange(1, args.k + 1)
    # eigengaps_vs_k: (n_seeds, W, k) → flatten seeds×walls for statistics
    egk_flat = eigengaps_vs_k.reshape(-1, args.k)   # (n_seeds*W, k)
    egk_mean = np.nanmean(egk_flat, axis=0)
    egk_p10 = np.nanpercentile(egk_flat, 10, axis=0)
    egk_p90 = np.nanpercentile(egk_flat, 90, axis=0)

    plt.figure()
    plt.plot(k_vals, egk_mean, label="Mean")
    plt.fill_between(k_vals, egk_p10, egk_p90, alpha=0.3, label=f"10-90% ({args.n_seeds} seeds)")
    ax = plt.gca()
    _log_axis_exponent_once(ax, "y")
    plt.xlabel(r"$k$")
    plt.ylabel(r"Eigengap $\lambda_{k+1} - \lambda_k$")
    plt.title(
        rf"Eigengap vs $k$ (analytical), ${args.rows}\times{args.cols}$, {args.n_seeds} seeds",
        fontsize=9,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "eigengap_vs_k.pdf"), format="pdf")
    plt.close()

    # ==================================================
    # Plot 5: Error (mean over goals) vs eigengap
    # Scatter of all (seed, wall) points + binned median
    # ==================================================
    gap_flat = eigengaps.reshape(-1)
    ae_flat = mean_ana.reshape(-1)
    ge_flat = mean_gdo.reshape(-1)
    mask = np.isfinite(gap_flat) & np.isfinite(ae_flat) & np.isfinite(ge_flat)

    plt.figure()
    plt.scatter(gap_flat[mask], ae_flat[mask], s=8, alpha=0.3, label="Analytical")
    plt.scatter(gap_flat[mask], ge_flat[mask], s=8, alpha=0.3, label="GDO")

    # Binned median overlay
    n_bins = min(20, mask.sum() // 5)
    if n_bins >= 2:
        order = np.argsort(gap_flat[mask])
        gap_s = gap_flat[mask][order]
        ae_s = ae_flat[mask][order]
        ge_s = ge_flat[mask][order]
        bins = np.quantile(gap_s, np.linspace(0.0, 1.0, n_bins + 1))
        bin_idx = np.digitize(gap_s, bins[1:-1], right=True)
        for err_s, style, lbl in [(ae_s, "-", "Ana. median"), (ge_s, "--", "GDO median")]:
            xm, ym = [], []
            for b in range(n_bins):
                mask_b = bin_idx == b
                if mask_b.sum() >= 3:
                    xm.append(np.median(gap_s[mask_b]))
                    ym.append(np.median(err_s[mask_b]))
            if xm:
                plt.plot(xm, ym, linestyle=style, linewidth=2, label=lbl)

    ax = plt.gca()
    _log_axis_exponent_once(ax, "x")
    _log_axis_exponent_once(ax, "y")
    plt.xlabel(rf"Eigengap $\lambda_{{k+1}} - \lambda_k$ ($k={args.k}$)")
    plt.ylabel(r"Error $\|v - \hat{v}\|_\Phi$ (mean over goals)")
    plt.title(
        rf"Error vs Eigengap, ${args.rows}\times{args.cols}$, "
        rf"$k={args.k}$, {args.n_seeds} seeds",
        fontsize=9,
    )
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "error_vs_eigengap.pdf"), format="pdf")
    plt.close()

    # ==================================================
    # Plot 6: Eigengap boxplot + normality/Z tests
    # ==================================================
    gap_valid = eigengaps.reshape(-1)
    gap_valid = gap_valid[np.isfinite(gap_valid)]

    # Shapiro-Wilk normality test
    sw_stat, sw_p = stats.shapiro(gap_valid)

    # One-sample t-test (unknown variance): H0: mu=0, H1: mu>0 (one-sided)
    n_gap = len(gap_valid)
    gap_mean = float(np.mean(gap_valid))
    gap_std = float(np.std(gap_valid, ddof=1))
    t_result = stats.ttest_1samp(gap_valid, popmean=0.0, alternative="greater")
    t_stat = float(t_result.statistic)
    t_p = float(t_result.pvalue)
    t_df = n_gap - 1

    eigengap_stats = {
        "n": n_gap,
        "mean": gap_mean,
        "std": gap_std,
        "min": float(np.min(gap_valid)),
        "sw_stat": float(sw_stat),
        "sw_p": float(sw_p),
        "t_stat": t_stat,
        "t_p": t_p,
        "t_df": t_df,
    }
    print(
        f"Eigengap stats: n={n_gap}, mean={gap_mean:.4g}, std={gap_std:.4g}\n"
        f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4g}\n"
        f"  t-test (mu>0): t({t_df})={t_stat:.4f}, p={t_p:.4g}"
    )

    _, ax = plt.subplots()
    bp = ax.boxplot(gap_valid, vert=True, patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.6)
    ax.set_xticks([1])
    ax.set_xticklabels([rf"all (seed $\times$ walls)"])
    ax.set_ylabel(rf"Eigengap $\lambda_{{k+1}} - \lambda_k$ ($k={args.k}$)")
    ax.set_title(
        rf"Eigengap distribution, ${args.rows}\times{args.cols}$, $k={args.k}$, {args.n_seeds} seeds",
        fontsize=9,
    )
    sw_note = "normal" if sw_p > 0.05 else "non-normal"
    t_note = "reject $H_0$" if t_p < 0.05 else "fail to reject $H_0$"
    textstr = (
        f"$n={n_gap}$,  mean$={gap_mean:.4g}$,  std$={gap_std:.4g}$\n"
        f"Shapiro–Wilk: $W={sw_stat:.4f}$,  $p={sw_p:.4g}$ ({sw_note})\n"
        rf"$t$-test ($\mu>0$,  df$={t_df}$): $t={t_stat:.4f}$,  $p={t_p:.4g}$ ({t_note})"
    )
    ax.text(
        0.5, 0.97, textstr,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "eigengap_boxplot.pdf"), format="pdf")
    plt.close()

    # ==================================================
    # Save raw data
    # ==================================================
    np.savez_compressed(
        os.path.join(args.save_dir, "task_transfer_data.npz"),
        wall_values=np.array(wall_values),
        goals=np.array(goals),
        analytical_errors=analytical_errors,   # (n_seeds, W, n_goals)
        gdo_errors=gdo_errors,                 # (n_seeds, W, n_goals)
        eigengaps=eigengaps,                   # (n_seeds, W)
        eigengaps_vs_k=eigengaps_vs_k,         # (n_seeds, W, k)
        lambda2s=lambda2s,                     # (n_seeds, W)
        rows=np.array(args.rows),
        cols=np.array(args.cols),
        k=np.array(args.k),
        n_seeds=np.array(args.n_seeds),
        transition_noise=np.array(args.transition_noise),
    )
    _write_task_transfer_markdown(
        os.path.join(args.save_dir, "task_transfer_results.md"),
        args=args,
        wall_values=wall_values,
        goals=goals,
        training_goal=training_goal,
        analytical_errors=analytical_errors,
        gdo_errors=gdo_errors,
        eigengaps=eigengaps,
        lambda2s=lambda2s,
        eigengap_stats=eigengap_stats,
        eigengaps_vs_k=eigengaps_vs_k,
    )
    print(f"Done. Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
