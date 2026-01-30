
import os
import tyro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import random
import torch

from tqdm import tqdm
import dataclasses
import matplotlib.pyplot as plt


from graph_env import GraphEnv
from gdo import LapRep, Config as LapRepConfig
from utils import (
    compute_stationary_distribution,
    uniform_policy,
    policy_induced_transition_matrix,
    policy_expected_reward,
    average_reward_value,
    diagonalize_in_subspace,
    generalized_eigen_residuals,
    save_eigenvector_heatmaps
)


@dataclasses.dataclass
class Args:
    n_seeds: int = 5
    n_walls: int = 30
    step_walls: int = 1
    rows: int = 10
    cols: int = 10
    k: int = 20  # Eigenvector cutting dims
    seed: int = 42
    save_dir: str = "imgs"
    models_dir: str = "models"  # Where to save/load analytical and gdo checkpoints
    n_eigenvec_heatmaps: int = 5  # Plot heatmaps for first n eigenvectors only (0 = skip)
    n_wall_heatmaps: int = 3  # Number of different wall counts to plot heatmaps for (spread over range)


def _log_axis_exponent_once(ax, which="y"):
    """Format axis so exponent (×10^n) appears once as offset text; y-axis: place it on the left."""
    fmt = ScalarFormatter()
    fmt.set_useOffset(True)
    fmt.set_scientific(True)
    fmt.set_useMathText(True)
    fmt.set_powerlimits((0, 0))  # force offset/scientific for all exponents so ×10^n always shows once
    if which == "y":
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_offset_position("left")
    else:
        ax.xaxis.set_major_formatter(fmt)


# ==================================================
# Setup
# ==================================================
args = tyro.cli(Args)

random.seed(args.seed)
plt.style.use("paper.mplstyle")
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.models_dir, exist_ok=True)

wall_values = np.arange(0, args.n_walls, args.step_walls)


wall_values = list(range(0, args.n_walls, args.step_walls))
W = len(wall_values)
# Wall indices for which to save eigenvector heatmaps (spread over range)
n_heatmap_walls = min(max(1, args.n_wall_heatmaps), W)
heatmap_wi_list = set(np.linspace(0, W - 1, n_heatmap_walls, dtype=int).tolist())

lambda2s = np.full((args.n_seeds, W), np.nan)
analytical_errors = np.full((args.n_seeds, W), np.nan)
gdo_errors = np.full((args.n_seeds, W), np.nan)


res_med_ana = np.full((args.n_seeds, W), np.nan)
res_med_gdo = np.full((args.n_seeds, W), np.nan)
res_max_ana = np.full((args.n_seeds, W), np.nan)
res_max_gdo = np.full((args.n_seeds, W), np.nan)
# Analytical error vs number of eigenvectors used (1 to args.k)
analytical_error_vs_k = np.full((args.n_seeds, W, args.k), np.nan)

for wi, w in tqdm(enumerate(wall_values)):
#for w in tqdm(range(0, args.n_walls, args.step_walls)):
    for seed in range(args.n_seeds):
        env = GraphEnv(
            n=args.rows,
            m=args.cols,
            n_walls=w,
            seed=seed
        )
        P = policy_induced_transition_matrix(env, uniform_policy(env))
        phi = compute_stationary_distribution(P)
        Phi = np.diag(phi)
        r_pi = policy_expected_reward(env, uniform_policy(env))
        v_true, _ = average_reward_value(P, r_pi, phi)

        # ==================================================
        # Analytical eigenvectors
        # ==================================================
        ana_path = os.path.join(args.models_dir, f"analytical_n{args.rows}_m{args.cols}_w{w}_s{seed}.npz")
        if os.path.isfile(ana_path):
            d = np.load(ana_path)
            eigvals, eigvecs = d["eigvals"], d["eigvecs"]
            Z_ana = eigvecs[:, :args.k]
        else:
            L = Phi - (Phi @ P + P.T @ Phi) / 2
            inv_phi = np.diag(1 / phi)
            L = inv_phi @ L
            eigvals, eigvecs = np.linalg.eig(L)
            idx = eigvals.argsort()
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            Z_ana = eigvecs[:, :args.k].real
            np.savez_compressed(ana_path, eigvals=eigvals[: args.k + 1], eigvecs=Z_ana)
        v_pred_analytical = Z_ana @ np.linalg.inv(Z_ana.T @ Phi @ Z_ana) @ (Z_ana.T @ Phi @ v_true)
        L = Phi - (Phi @ P + P.T @ Phi) / 2
        L = np.diag(1 / phi) @ L

        # ==================================================
        # GDO eigenvectors
        # ==================================================
        model = LapRep(LapRepConfig(n_states=env.n_states, k=args.k))
        model.load(os.path.join(args.models_dir, f"gdo_n{args.rows}_m{args.cols}_w{w}_s{seed}.pt"))
        Z_gdo = model.network(torch.arange(env.n_states))
        Z_gdo = Z_gdo.detach().numpy()
        v_pred_gdo = Z_gdo @ np.linalg.inv(Z_gdo.T @ Phi @ Z_gdo) @ (Z_gdo.T @ Phi @ v_true)

        tmp_error = v_true - v_pred_analytical
        err_analytical = np.sqrt(tmp_error.T @ Phi @ tmp_error)

        # Analytical error when using 1, 2, ..., args.k eigenvectors
        eigvecs_real = np.asarray(eigvecs).real
        for k in range(1, args.k + 1):
            Z_k = eigvecs_real[:, :k]
            v_pred_k = Z_k @ np.linalg.inv(Z_k.T @ Phi @ Z_k) @ (Z_k.T @ Phi @ v_true)
            tmp_k = v_true - v_pred_k
            analytical_error_vs_k[seed, wi, k - 1] = np.sqrt(tmp_k.T @ Phi @ tmp_k)

        tmp_error = v_true - v_pred_gdo
        err_gdo = np.sqrt(tmp_error.T @ Phi @ tmp_error)

        analytical_errors[seed, wi] = err_analytical
        gdo_errors[seed, wi] = err_gdo
        lambda2s[seed, wi] = eigvals[1].real


        # Saving data for residual analysis
        _, Z_gdo_diag = diagonalize_in_subspace(Z_gdo, L, Phi)
        _, _, summ_a = generalized_eigen_residuals(Z_ana, L, Phi)
        _, _, summ_g = generalized_eigen_residuals(Z_gdo_diag, L, Phi)
        res_med_ana[seed, wi] = summ_a["res_med"]
        res_med_gdo[seed, wi] = summ_g["res_med"]
        res_max_ana[seed, wi] = summ_a["res_max"]
        res_max_gdo[seed, wi] = summ_g["res_max"]

        # Save heatmaps for first n eigenvectors at selected wall counts (seed=0 only)
        if args.n_eigenvec_heatmaps > 0 and seed == 0 and wi in heatmap_wi_list:
            n_plot = min(args.n_eigenvec_heatmaps, args.k)
            w_label = wall_values[wi]
            subdir_a = os.path.join(args.save_dir, f"eigenvec_analytical_w{w_label}")
            subdir_g = os.path.join(args.save_dir, f"eigenvec_gdo_w{w_label}")
            save_eigenvector_heatmaps(env, eigvecs[:, :n_plot].real, subdir_a, prefix="analytical", cmap="RdBu_r")
            save_eigenvector_heatmaps(env, Z_gdo[:, :n_plot], subdir_g, prefix="gdo", cmap="RdBu_r")


# ==================================================
# Plot lambda2s
# ==================================================
lambda2s_p10 = np.percentile(lambda2s, 10, axis=0)
lambda2s_p90 = np.percentile(lambda2s, 90, axis=0)
x = np.array(wall_values)
plt.plot(x, lambda2s.mean(axis=0), label="Mean")
plt.fill_between(x, lambda2s_p10, lambda2s_p90, alpha=0.3, label=f"10-90% ({args.n_seeds} seeds)")
ax = plt.gca()
#ax.set_yscale("log")
_log_axis_exponent_once(ax, "y")
plt.xlabel("Number of Walls")
plt.ylabel(r"$\lambda_2$")
#plt.title("Second Smallest Eigenvalue vs Number of Walls")
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.save_dir}/lambda2.pdf", format="pdf")
plt.savefig(f"{args.save_dir}/lambda2.png", format="png")
plt.close()

# ==================================================
# Plot errors
# ==================================================
x = np.array(wall_values)
# Analytical errors
plt.plot(x, analytical_errors.mean(axis=0), label="Analytical (Mean)")
plt.fill_between(
    x,
    np.percentile(analytical_errors, 10, axis=0),
    np.percentile(analytical_errors, 90, axis=0),
    alpha=0.3,
    label=f"10-90% ({args.n_seeds} seeds)"
)
# GDO errors
plt.plot(x, gdo_errors.mean(axis=0), label="GDO (Mean)")
plt.fill_between(
    x,
    np.percentile(gdo_errors, 10, axis=0),
    np.percentile(gdo_errors, 90, axis=0),
    alpha=0.3,
    label=f"10-90% ({args.n_seeds} seeds)"
)
plt.yscale("log")
#ax = plt.gca()
#ax.set_yscale("log")
#_log_axis_exponent_once(ax, "y")
plt.xlabel("Number of Walls")
plt.ylabel("Error")
#plt.title("Error vs Number of Walls")
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.save_dir}/errors.pdf", format="pdf")
plt.savefig(f"{args.save_dir}/errors.png", format="png")
plt.close()

# ==================================================
# Plot: analytical error vs number of eigenvectors (1 to k)
# ==================================================
k_vals = np.arange(1, args.k + 1, dtype=float)
# Reshape to (n_seeds * W, args.k) and take mean/percentile over environments
err_vs_k_flat = analytical_error_vs_k.reshape(-1, args.k)
mean_err_vs_k = np.nanmean(err_vs_k_flat, axis=0)
p10_err_vs_k = np.nanpercentile(err_vs_k_flat, 10, axis=0)
p90_err_vs_k = np.nanpercentile(err_vs_k_flat, 90, axis=0)
plt.figure()
plt.plot(k_vals, mean_err_vs_k, label="Analytical (Mean)")
plt.fill_between(k_vals, p10_err_vs_k, p90_err_vs_k, alpha=0.3, label=f"10-90% ({args.n_seeds} seeds)")
ax = plt.gca()
#ax.set_yscale("log")
_log_axis_exponent_once(ax, "y")
plt.xlabel(r"Number of eigenvectors ($k$)")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.save_dir}/error_vs_n_eigenvec.pdf", format="pdf")
plt.savefig(f"{args.save_dir}/error_vs_n_eigenvec.png", format="png")
plt.close()

# ==================================================
# Plot: error vs lambda2 (x=lambda2, y=error)
# ==================================================
# Each point is one (seed, wall) environment.
x = lambda2s.reshape(-1)
y = analytical_errors.reshape(-1)

# Sort by x for optional binned trend
order = np.argsort(x)
x_sorted = x[order]
y_sorted = y[order]

plt.figure()
plt.scatter(x, y, s=10, alpha=0.35, label="All (seed,walls)")

# Optional: binned median + 10-90% band to show trend
n_bins = 20
bins = np.quantile(x_sorted, np.linspace(0.0, 1.0, n_bins + 1))
bin_idx = np.digitize(x_sorted, bins[1:-1], right=True)

x_med, y_med, y_p10, y_p90 = [], [], [], []
for b in range(n_bins):
    mask = (bin_idx == b)
    if mask.sum() < 5:
        continue
    xb = x_sorted[mask]
    yb = y_sorted[mask]
    x_med.append(np.median(xb))
    y_med.append(np.median(yb))
    y_p10.append(np.percentile(yb, 10))
    y_p90.append(np.percentile(yb, 90))

x_med = np.array(x_med)
y_med = np.array(y_med)
y_p10 = np.array(y_p10)
y_p90 = np.array(y_p90)

plt.plot(x_med, y_med, linewidth=2, label="Binned median")
plt.fill_between(x_med, y_p10, y_p90, alpha=0.25, label="Binned 10-90%")

ax = plt.gca()
#ax.set_xscale("log")
#ax.set_yscale("log")
_log_axis_exponent_once(ax, "x")
_log_axis_exponent_once(ax, "y")
plt.xlabel(r"$\lambda_2$")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.save_dir}/error_vs_lambda2.pdf", format="pdf")
plt.savefig(f"{args.save_dir}/error_vs_lambda2.png", format="png")
plt.close()

# ==================================================
# Plot: error vs lambda2 (x=lambda2, y=error)
# ==================================================
# Each point is one (seed, wall) environment.
x = lambda2s.reshape(-1)
y = gdo_errors.reshape(-1)

# Sort by x for optional binned trend
order = np.argsort(x)
x_sorted = x[order]
y_sorted = y[order]

plt.figure()
plt.scatter(x, y, s=10, alpha=0.35, label="All (seed,walls)")

# binned median + 10-90% band to show trend
n_bins = 20
bins = np.quantile(x_sorted, np.linspace(0.0, 1.0, n_bins + 1))
bin_idx = np.digitize(x_sorted, bins[1:-1], right=True)

x_med, y_med, y_p10, y_p90 = [], [], [], []
for b in range(n_bins):
    mask = (bin_idx == b)
    if mask.sum() < 5:
        continue
    xb = x_sorted[mask]
    yb = y_sorted[mask]
    x_med.append(np.median(xb))
    y_med.append(np.median(yb))
    y_p10.append(np.percentile(yb, 10))
    y_p90.append(np.percentile(yb, 90))

x_med = np.array(x_med)
y_med = np.array(y_med)
y_p10 = np.array(y_p10)
y_p90 = np.array(y_p90)

plt.plot(x_med, y_med, linewidth=2, label="Binned median")
plt.fill_between(x_med, y_p10, y_p90, alpha=0.25, label="Binned 10-90%")

ax = plt.gca()
#ax.set_xscale("log")
#ax.set_yscale("log")
_log_axis_exponent_once(ax, "x")
_log_axis_exponent_once(ax, "y")
plt.xlabel(r"$\lambda_2$")
plt.ylabel("GDO Error")
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.save_dir}/gdo_error_vs_lambda2.pdf", format="pdf")
plt.savefig(f"{args.save_dir}/gdo_error_vs_lambda2.png", format="png")
plt.close()


# ==================================================
# Plot: median and max generalized eigen residual vs number of walls
# ==================================================
x_walls = np.array(wall_values)

plt.figure()
plt.plot(x_walls, np.nanmean(res_med_ana, axis=0), label="Analytical (median residual)")
plt.plot(x_walls, np.nanmean(res_med_gdo, axis=0), label="GDO (median residual)")
ax = plt.gca()
#ax.set_yscale("log")
_log_axis_exponent_once(ax, "y")
plt.xlabel("Number of Walls")
plt.ylabel("Median generalized eigen residual")
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.save_dir}/residuals_median_vs_walls.png", dpi=200)
plt.close()

plt.figure()
plt.plot(x_walls, np.nanmean(res_max_ana, axis=0), label="Analytical (max residual)")
plt.plot(x_walls, np.nanmean(res_max_gdo, axis=0), label="GDO (max residual)")
ax = plt.gca()
#ax.set_yscale("log")
_log_axis_exponent_once(ax, "y")
plt.xlabel("Number of Walls")
plt.ylabel("Max generalized eigen residual")
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.save_dir}/residuals_max_vs_walls.png", dpi=200)
plt.close()

# ==================================================
# Plot: median generalized eigen residual vs lambda2
# ==================================================
eps = 1e-12
x = np.maximum(lambda2s.reshape(-1), eps)

y_a = res_med_ana.reshape(-1)
y_g = res_med_gdo.reshape(-1)

mask_a = np.isfinite(y_a)
mask_g = np.isfinite(y_g)

plt.figure()
plt.scatter(x[mask_a], y_a[mask_a], s=10, alpha=0.35, label="Analytical")
plt.scatter(x[mask_g], y_g[mask_g], s=10, alpha=0.35, label="GDO")
ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
_log_axis_exponent_once(ax, "x")
_log_axis_exponent_once(ax, "y")
plt.xlabel(r"$\lambda_2$")
plt.ylabel("Median generalized eigen residual")
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.save_dir}/residuals_vs_lambda2.png", dpi=200)
plt.close()

# ==================================================
# Save all computed data (errors, lambda2s, residuals, config)
# ==================================================
np.savez_compressed(
    os.path.join(args.save_dir, "data.npz"),
    wall_values=wall_values,
    lambda2s=lambda2s,
    analytical_errors=analytical_errors,
    gdo_errors=gdo_errors,
    res_med_ana=res_med_ana,
    res_med_gdo=res_med_gdo,
    res_max_ana=res_max_ana,
    res_max_gdo=res_max_gdo,
    rows=np.array(args.rows),
    cols=np.array(args.cols),
    n_walls=np.array(args.n_walls),
    step_walls=np.array(args.step_walls),
    n_seeds=np.array(args.n_seeds),
    k=np.array(args.k),
)