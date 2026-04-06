"""
Experiment: wall count fixed at 0, sweep transition noise η, compare analytical vs GDO.

Writes under ``save_dir``: ``lambda2_vs_eta.pdf``, ``errors_vs_eta.pdf`` (linear y), ``errors_vs_eta_log.pdf`` (log y), ``eta_w0_data.npz``, ``eta_w0_results.md`` (compact table: mean, or mean ± half the (p90−p10) band when nonzero).
Loads GDO / analytical caches from ``models_dir`` (default: ``<save_dir>/models`` — same rule as ``exps.py``).

You do **not** need ``--save-dir`` or ``--models-dir`` unless you want custom paths; defaults are enough.

Default output: under ``experiments/``, a folder tagged by grid, ``w=0``, seeds, and ``k`` (see ``_default_save_dir``).

Requires trained GDO weights for **each** (η, seed) at w=0 under ``models_dir``, unless you pass ``--train-missing`` (runs ``gdo.py`` for missing checkpoints; first run can be slow).

Example reuse:
  python exp_eta_w0.py --rows 15 --cols 15 --models-dir experiments/exp_15x15_w50_step3_nseeds5/models
"""

from __future__ import annotations

import dataclasses
import os
import random
import subprocess
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from graph_env import GraphEnv
from gdo import LapRep, Config as LapRepConfig
from utils import (
    average_reward_value,
    compute_stationary_distribution,
    policy_expected_reward,
    policy_induced_transition_matrix,
    uniform_policy,
)


def _stoch_suffix(noise: float) -> str:
    return "" if noise <= 0 else f"_eps{str(noise).replace('.', 'p')}"


def _parse_etas(s: str) -> list[float]:
    out = [float(x.strip()) for x in s.replace(" ", "").split(",") if x.strip()]
    for eta in out:
        if eta < 0 or eta > 1:
            raise ValueError(f"each η must be in [0,1], got {eta}")
    return out


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


@dataclasses.dataclass
class Args:
    rows: int = 15
    cols: int = 15
    k: int = 20
    n_seeds: int = 5
    seed: int = 42
    #etas: str = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
    etas: str = "0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009"
    """Comma-separated η in [0, 1] for P=(1-η)P_det+η Unif(all states)."""
    save_dir: Optional[str] = None
    """If unset, ``experiments/exp_{rows}x{cols}_w0_eta_nseeds{n}_k{k}``."""
    models_dir: Optional[str] = None  # default after parse: <save_dir>/models (same as exps.py)
    train_missing: bool = False
    """If True, run ``gdo.py`` for any missing GDO checkpoint (same hyperparameters as ``run_all.sh``)."""
    train_steps: int = 10000
    log_freq: int = 1999


def _title(args: Args) -> str:
    return rf"${args.rows}\times{args.cols}$ grid, $w=0$, $k={args.k}$, {args.n_seeds} seeds, $\eta$ sweep"


def _default_save_dir(rows: int, cols: int, n_seeds: int, k: int) -> str:
    return f"experiments/exp_{rows}x{cols}_w0_eta_nseeds{n_seeds}_k{k}"


def _write_eta_sweep_markdown(
    path: str,
    *,
    args: Args,
    etas: np.ndarray,
    lambda2s: np.ndarray,
    analytical_errors: np.ndarray,
    gdo_errors: np.ndarray,
) -> None:
    """Write per-η summary: mean, or mean $\\pm$ $(p_{90}-p_{10})/2$ when that half-width is nonzero."""

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

    header = r"| $\eta$ | $\lambda_2$ | $\varepsilon_{\mathrm{ana}}$ | $\varepsilon_{\mathrm{GDO}}$ |"
    sep = "|---|---:|---:|---:|"
    lines = [
        r"# $\eta$ sweep results",
        "",
        rf"Grid: ${args.rows} \times {args.cols}$, walls $= 0$, $k = {args.k}$, {args.n_seeds} seeds.",
        "",
        r"Each cell is $\mathrm{mean}$, or $\mathrm{mean} \pm \tfrac{1}{2}(p_{90}-p_{10})$ when that band width is nonzero (same $10$–$90\%$ band as the PDFs).",
        "",
        header,
        sep,
    ]
    for i in range(len(etas)):
        eta = float(etas[i])
        l2 = lambda2s[:, i]
        ea = analytical_errors[:, i]
        eg = gdo_errors[:, i]
        lines.append(
            f"| ${fmt(eta)}$ | {mean_pm_band(l2)} | {mean_pm_band(ea)} | {mean_pm_band(eg)} |",
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _gdo_path(models_dir: str, rows: int, cols: int, w: int, seed: int, eta: float) -> str:
    return os.path.join(
        models_dir,
        f"gdo_n{rows}_m{cols}_w{w}_s{seed}{_stoch_suffix(eta)}.pt",
    )


def _require_gdo_checkpoint(
    path: str,
    *,
    rows: int,
    cols: int,
    w: int,
    seed: int,
    eta: float,
    k: int,
    models_dir: str,
) -> None:
    if os.path.isfile(path):
        return
    raise FileNotFoundError(
        f"Missing GDO checkpoint:\n  {path}\n\n"
        "Re-run with --train-missing to train missing checkpoints automatically, or train manually, e.g.\n"
        f"  python gdo.py --n {rows} --m {cols} --n-walls {w} --seed {seed} --k {k} \\\n"
        f"    --train-steps 10000 --log-freq 1999 --save-dir {models_dir} --transition-noise {eta}\n\n"
        "To reuse checkpoints from a wall-sweep experiment (only for η you actually trained), pass e.g.\n"
        "  --models-dir experiments/exp_15x15_w50_step3_nseeds5/models"
    )


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
        _require_gdo_checkpoint(
            path,
            rows=rows,
            cols=cols,
            w=w,
            seed=seed,
            eta=eta,
            k=k,
            models_dir=models_dir,
        )
    repo_root = os.path.dirname(os.path.abspath(__file__))
    gdo_script = os.path.join(repo_root, "gdo.py")
    cmd = [
        sys.executable,
        gdo_script,
        "--n",
        str(rows),
        "--m",
        str(cols),
        "--n-walls",
        str(w),
        "--seed",
        str(seed),
        "--k",
        str(k),
        "--log-freq",
        str(log_freq),
        "--train-steps",
        str(train_steps),
        "--save-dir",
        models_dir,
        "--transition-noise",
        str(eta),
    ]
    print(f"Training GDO → {path}", flush=True)
    subprocess.run(cmd, check=True)
    if not os.path.isfile(path):
        raise RuntimeError(f"gdo.py finished but checkpoint not found: {path}")


def main() -> None:
    args = tyro.cli(Args)
    if args.save_dir is None:
        args.save_dir = _default_save_dir(args.rows, args.cols, args.n_seeds, args.k)
    if args.models_dir is None:
        args.models_dir = os.path.join(args.save_dir, "models")

    eta_list = _parse_etas(args.etas)
    n_eta = len(eta_list)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    plt.style.use("paper.mplstyle")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    lambda2s = np.full((args.n_seeds, n_eta), np.nan)
    analytical_errors = np.full((args.n_seeds, n_eta), np.nan)
    gdo_errors = np.full((args.n_seeds, n_eta), np.nan)

    w = 0
    for ei, eta in enumerate(tqdm(eta_list, desc="η")):
        for seed in range(args.n_seeds):
            env = GraphEnv(
                n=args.rows,
                m=args.cols,
                n_walls=w,
                seed=seed,
                transition_noise=eta,
            )
            P = policy_induced_transition_matrix(env, uniform_policy(env))
            phi = compute_stationary_distribution(P)
            Phi = np.diag(phi)
            r_pi = policy_expected_reward(env, uniform_policy(env))
            v_true, _ = average_reward_value(P, r_pi, phi)

            ana_path = os.path.join(
                args.models_dir,
                f"analytical_n{args.rows}_m{args.cols}_w{w}_s{seed}{_stoch_suffix(eta)}.npz",
            )
            if os.path.isfile(ana_path):
                d = np.load(ana_path)
                eigvals, eigvecs = d["eigvals"], d["eigvecs"]
                Z_ana = eigvecs[:, : args.k]
            else:
                L = Phi - (Phi @ P + P.T @ Phi) / 2
                inv_phi = np.diag(1 / phi)
                L = inv_phi @ L
                eigvals, eigvecs = np.linalg.eig(L)
                idx = eigvals.argsort()
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                Z_ana = eigvecs[:, : args.k].real
                np.savez_compressed(ana_path, eigvals=eigvals[: args.k + 1], eigvecs=Z_ana)

            try:
                v_pred_analytical = Z_ana @ np.linalg.inv(Z_ana.T @ Phi @ Z_ana) @ (Z_ana.T @ Phi @ v_true)

                gdo_pt = _gdo_path(args.models_dir, args.rows, args.cols, w, seed, eta)
                _ensure_gdo_checkpoint(
                    gdo_pt,
                    rows=args.rows,
                    cols=args.cols,
                    w=w,
                    seed=seed,
                    eta=eta,
                    k=args.k,
                    models_dir=args.models_dir,
                    train_missing=args.train_missing,
                    train_steps=args.train_steps,
                    log_freq=args.log_freq,
                )
                model = LapRep(LapRepConfig(n_states=env.n_states, k=args.k))
                model.load(gdo_pt)
                Z_gdo = model.network(torch.arange(env.n_states))
                Z_gdo = Z_gdo.detach().numpy()
                v_pred_gdo = Z_gdo @ np.linalg.inv(Z_gdo.T @ Phi @ Z_gdo) @ (Z_gdo.T @ Phi @ v_true)

                tmp_a = v_true - v_pred_analytical
                err_a = np.sqrt(tmp_a.T @ Phi @ tmp_a)
                tmp_g = v_true - v_pred_gdo
                err_g = np.sqrt(tmp_g.T @ Phi @ tmp_g)
            except np.linalg.LinAlgError as e:
                print(f"Skipping eta={eta}, seed={seed}: {e}")
                continue

            analytical_errors[seed, ei] = err_a
            gdo_errors[seed, ei] = err_g
            lambda2s[seed, ei] = eigvals[1].real

    x = np.array(eta_list)

    plt.figure()
    plt.plot(x, np.nanmean(lambda2s, axis=0), label="Mean")
    plt.fill_between(
        x,
        np.nanpercentile(lambda2s, 10, axis=0),
        np.nanpercentile(lambda2s, 90, axis=0),
        alpha=0.3,
        label=f"10-90% ({args.n_seeds} seeds)",
    )
    ax = plt.gca()
    _log_axis_exponent_once(ax, "y")
    plt.xlabel(r"$\eta$ (global teleport weight)")
    plt.ylabel(r"$\lambda_2$")
    plt.title(_title(args), fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "lambda2_vs_eta.pdf"), format="pdf")
    plt.close()

    # Linear-scale error plot
    plt.figure()
    plt.plot(x, np.nanmean(analytical_errors, axis=0), label="Analytical (Mean)")
    plt.fill_between(
        x,
        np.nanpercentile(analytical_errors, 10, axis=0),
        np.nanpercentile(analytical_errors, 90, axis=0),
        alpha=0.3,
        label=f"Analytical 10-90% ({args.n_seeds} seeds)",
    )
    plt.plot(x, np.nanmean(gdo_errors, axis=0), label="GDO (Mean)")
    plt.fill_between(
        x,
        np.nanpercentile(gdo_errors, 10, axis=0),
        np.nanpercentile(gdo_errors, 90, axis=0),
        alpha=0.3,
        label=f"GDO 10-90% ({args.n_seeds} seeds)",
    )
    plt.xlabel(r"$\eta$ (global teleport weight)")
    plt.ylabel("Error")
    plt.title(_title(args), fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "errors_vs_eta.pdf"), format="pdf")
    plt.close()

    # Log-scale error plot (same curves and percentile bands)
    plt.figure()
    plt.plot(x, np.nanmean(analytical_errors, axis=0), label="Analytical (Mean)")
    plt.fill_between(
        x,
        np.nanpercentile(analytical_errors, 10, axis=0),
        np.nanpercentile(analytical_errors, 90, axis=0),
        alpha=0.3,
        label=f"Analytical 10-90% ({args.n_seeds} seeds)",
    )
    plt.plot(x, np.nanmean(gdo_errors, axis=0), label="GDO (Mean)")
    plt.fill_between(
        x,
        np.nanpercentile(gdo_errors, 10, axis=0),
        np.nanpercentile(gdo_errors, 90, axis=0),
        alpha=0.3,
        label=f"GDO 10-90% ({args.n_seeds} seeds)",
    )
    plt.yscale("log")
    plt.xlabel(r"$\eta$ (global teleport weight)")
    plt.ylabel("Error")
    plt.title(_title(args), fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "errors_vs_eta_log.pdf"), format="pdf")
    plt.close()

    np.savez_compressed(
        os.path.join(args.save_dir, "eta_w0_data.npz"),
        etas=x,
        lambda2s=lambda2s,
        analytical_errors=analytical_errors,
        gdo_errors=gdo_errors,
        rows=np.array(args.rows),
        cols=np.array(args.cols),
        k=np.array(args.k),
        n_seeds=np.array(args.n_seeds),
        n_walls=np.array(0),
    )
    _write_eta_sweep_markdown(
        os.path.join(args.save_dir, "eta_w0_results.md"),
        args=args,
        etas=x,
        lambda2s=lambda2s,
        analytical_errors=analytical_errors,
        gdo_errors=gdo_errors,
    )


if __name__ == "__main__":
    main()
