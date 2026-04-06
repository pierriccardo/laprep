import hashlib
import numpy as np
from graph_env import GraphEnv
import os
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Sequence, Union


def compute_stationary_distribution(
    P_pi: np.ndarray,
    max_iter: int = 10000,
    tol: float = 1e-10
) -> np.ndarray:
    """
    Compute the stationary distribution of a Markov chain.

    Uses eigenvalue decomposition to find the stationary distribution.
    Falls back to power iteration if needed.

    Args:
        P_pi: Transition matrix of shape (n_states, n_states).
        max_iter: Maximum number of iterations for power method fallback.
        tol: Convergence tolerance.

    Returns:
        Stationary distribution array of shape (n_states,).

    Raises:
        ValueError: If the algorithm doesn't converge.

    Example:
        >>> P = policy_induced_transition_matrix(env, policy)
        >>> mu = compute_stationary_distribution(P)
        >>> assert np.allclose(mu @ P, mu)
    """
    n_states = P_pi.shape[0]

    # Try eigenvalue method first (works for all ergodic chains)
    try:
        # Left eigenvectors of P correspond to right eigenvectors of P.T
        eigenvalues, eigenvectors = np.linalg.eig(P_pi.T)

        # Find eigenvector corresponding to eigenvalue 1
        # (there should be exactly one for an ergodic chain)
        idx = np.argmin(np.abs(eigenvalues - 1.0))

        if np.abs(eigenvalues[idx] - 1.0) < 1e-8:
            mu = np.real(eigenvectors[:, idx])
            # Normalize to get probability distribution
            mu = np.abs(mu)  # Ensure non-negative
            mu = mu / mu.sum()

            # Verify it's actually stationary
            if np.allclose(mu @ P_pi, mu, atol=1e-6):
                return mu
    except np.linalg.LinAlgError:
        pass

    # Fallback: power iteration with averaging (handles periodic chains)
    mu = np.ones(n_states) / n_states
    mu_avg = mu.copy()

    for i in range(1, max_iter + 1):
        mu_new = mu @ P_pi

        # Running average to handle periodicity
        mu_avg = (mu_avg * i + mu_new) / (i + 1)

        if i > 100 and np.max(np.abs(mu_avg @ P_pi - mu_avg)) < tol:
            return mu_avg / mu_avg.sum()

        mu = mu_new

    # If we haven't converged, return the running average anyway
    # (for periodic chains, the average is still valid)
    mu_avg = mu_avg / mu_avg.sum()
    if np.allclose(mu_avg @ P_pi, mu_avg, atol=1e-4):
        return mu_avg

    raise ValueError(f"Stationary distribution did not converge in {max_iter} iterations")



def uniform_policy(env) -> np.ndarray:
    """
    Create a uniform policy that assigns equal probability to all actions.

    For each state s, the policy assigns probability 1/n_actions to each action.

    Args:
        env: The GridWorld environment.

    Returns:
        Policy array of shape (n_states, n_actions) where each row sums to 1
        and all entries are 1/n_actions.

    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = uniform_policy(env)
        >>> print(pi[0])  # [0.25, 0.25, 0.25, 0.25]
    """
    policy = np.ones((env.n_states, env.n_actions), dtype=np.float64)
    policy /= env.n_actions
    return policy


def mixed_policy(env, k: int = 0.7) -> np.ndarray:
    """
    Create a "sparse" policy for each state: with probability k, take one designated action (e.g., action 0),
    and with the remaining probability (1-k), choose uniformly among the other actions.

    For each state s:
        - The agent takes action 0 with probability k.
        - The agent takes each of the other (n_actions - 1) actions with probability (1-k)/(n_actions-1).

    Args:
        env: The GridWorld environment.
        k: Probability of taking the designated action (default 0.9, can be adjusted).

    Returns:
        Policy array of shape (n_states, n_actions) where each row sums to 1,
        with prob k to action 0 and uniform for remaining actions.

    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = mixed_policy(env, k=0.9)
        >>> print(pi[0])  # [0.9, 0.025, 0.025, 0.025, 0.025] (if 5 actions)
    """
    n_states = env.n_states
    n_actions = env.n_actions
    if not (0 <= k <= 1):
        raise ValueError(f"k must be in [0,1], got {k}")

    policy = np.full((n_states, n_actions), (1 - k) / (n_actions - 1), dtype=np.float64)
    policy[:, 0] = k  # or set any specific action index here
    return policy


def goal_directed_policy(env, beta: float = 2.5, epsilon: float = 0.1) -> np.ndarray:
    """
    Stochastic policy biased toward actions that move closer to the goal.

    For each state, score each action by reduction in Manhattan distance to goal:
        score(s,a) = d(s, goal) - d(s_next, goal)
    Then apply softmax(beta * score) and mix with uniform exploration:
        pi = (1-epsilon) * softmax + epsilon * uniform

    This is non-uniform, state-aware, and less brittle than a fixed global action bias.
    """
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if not (0 <= epsilon <= 1):
        raise ValueError(f"epsilon must be in [0,1], got {epsilon}")

    n_states = env.n_states
    n_actions = env.n_actions
    policy = np.zeros((n_states, n_actions), dtype=np.float64)
    goal_r, goal_c = env.goal
    uniform = np.full(n_actions, 1.0 / n_actions, dtype=np.float64)

    for s in range(n_states):
        row, col = env.id2pos[s]
        d_curr = abs(row - goal_r) + abs(col - goal_c)
        scores = np.empty(n_actions, dtype=np.float64)
        for a_idx, (dr, dc) in env.actions.items():
            nxt = (row + dr, col + dc)
            if nxt in env.graph[(row, col)]:
                nr, nc = nxt
            else:
                nr, nc = row, col
            d_next = abs(nr - goal_r) + abs(nc - goal_c)
            scores[a_idx] = d_curr - d_next

        # Stable softmax
        z = beta * scores
        z -= np.max(z)
        probs = np.exp(z)
        probs /= np.sum(probs)
        policy[s] = (1.0 - epsilon) * probs + epsilon * uniform

    return policy


def random_policy(env) -> np.ndarray:
    """
    Stochastic policy with independent random action weights per state.

    Each row is drawn from a Dirichlet(1,...,1) distribution (uniform over the simplex),
    so every state has a different non-uniform distribution over actions.

    The RNG is seeded deterministically from ``env.seed``, grid size, wall count, and
    ``transition_noise`` so the same environment always yields the same policy.
    """
    key = f"{env.seed}_{env.n}_{env.m}_{env.n_walls}_{env.transition_noise}".encode()
    seed_int = int.from_bytes(hashlib.sha256(key).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed_int)
    n_states = env.n_states
    n_actions = env.n_actions
    policy = np.empty((n_states, n_actions), dtype=np.float64)
    alpha = np.ones(n_actions, dtype=np.float64)
    for s in range(n_states):
        policy[s] = rng.dirichlet(alpha)
    return policy


def policy_induced_transition_matrix(env, policy: np.ndarray) -> np.ndarray:
    """
    Compute the policy-induced transition matrix P_π.

    Given the environment's transition kernel P[s, a, s'] and a policy π[s, a],
    computes the state-to-state transition probabilities under the policy:

        P_π[s, s'] = Σ_a π[s, a] * P[s, a, s']

    This gives the probability of transitioning from state s to state s'
    when following policy π.

    Args:
        env: The GridWorld environment with transition kernel P.
        policy: Policy array of shape (n_states, n_actions).

    Returns:
        Transition matrix of shape (n_states, n_states) where P_π[s, s']
        is the probability of going from s to s' under policy π.
        Each row sums to 1.

    Raises:
        ValueError: If policy shape doesn't match environment dimensions.

    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = uniform_policy(env)
        >>> P_pi = policy_induced_transition_matrix(env, pi)
        >>> assert P_pi.shape == (env.n_states, env.n_states)
        >>> assert np.allclose(P_pi.sum(axis=1), 1.0)
    """
    if policy.shape != (env.n_states, env.n_actions):
        raise ValueError(
            f"Policy shape {policy.shape} doesn't match expected "
            f"({env.n_states}, {env.n_actions})"
        )

    # P_pi[s, s'] = sum_a pi[s, a] * P[s, a, s']
    # Using einsum for clarity and efficiency
    # s=current state, a=action, t=next state (s')
    P_pi = np.einsum('sa,sat->st', policy, env.P)

    return P_pi

def policy_expected_reward(
    env,
    policy: np.ndarray
) -> np.ndarray:
    """
    Compute the expected reward for each state under a given policy.

    For each state s, computes:
        r_π(s) = Σ_a π[s, a] * Σ_{s'} P[s, a, s'] * R(s, a, s')

    Args:
        env: The GridWorld environment.
        policy: Policy array of shape (n_states, n_actions).

    Returns:
        Array of shape (n_states,) with expected reward for each state.

    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = uniform_policy(env)
        >>> r_pi = policy_expected_reward(env, pi)
    """
    if policy.shape != (env.n_states, env.n_actions):
        raise ValueError(
            f"Policy shape {policy.shape} doesn't match expected "
            f"({env.n_states}, {env.n_actions})"
        )

    # Get expected reward for each (s, a) pair
    R_sa = env.get_expected_reward()  # shape (n_states, n_actions)

    # Compute r_pi(s) = sum_a pi[s, a] * R[s, a]
    r_pi = np.sum(policy * R_sa, axis=1)

    return r_pi

def average_reward_value(P, r, phi):
    """
    Solve for (v, rho) in the average-reward setting with normalization phi^T v = 0.

    Assumptions:
      - P is (n, n) transition matrix for a fixed policy.
      - r is (n,) reward vector.
      - phi is (n,) probability vector, typically the stationary distribution of P,
        so that phi^T P = phi^T and phi^T 1 = 1.

    Returns:
      v   : differential value function (n,)
      rho : average reward (scalar)
    """
    P = np.asarray(P, dtype=float)
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)

    n = r.shape[0]
    I = np.eye(n)
    ones = np.ones((n, 1))

    # Average reward: rho = phi^T r
    rho = np.dot(phi, r)

    # Centered reward: r - rho * 1
    rhs = r - rho  # broadcasts rho to all components

    # Solve (I - P + 1 phi^T) v = r - rho * 1
    A = I - P + ones @ phi.reshape(1, -1)
    v = np.linalg.solve(A, rhs)

    return v, rho


def diagonalize_in_subspace(Z, L, Phi, eps=1e-12):
    # Z: (n,k)
    A = Z.T @ L @ Z
    B = Z.T @ Phi @ Z
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)

    # Solve generalized eigenproblem via whitening B
    evalsB, U = np.linalg.eigh(B)
    evalsB = np.maximum(evalsB, eps)
    Binv2 = U @ np.diag(1.0 / np.sqrt(evalsB)) @ U.T

    M = Binv2 @ A @ Binv2
    M = 0.5 * (M + M.T)
    eigvals, W = np.linalg.eigh(M)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    W = W[:, idx]

    Usub = Binv2 @ W  # (k,k)
    Z_diag = Z @ Usub  # (n,k)
    return eigvals, Z_diag


def generalized_eigen_residuals(Z, L, Phi, normalize=True, eps=1e-12):
    """
    For each column z_j of Z, compute Rayleigh quotient lambda_j and residual:
        r_j = || L z_j - lambda_j Phi z_j ||_2

    Returns:
        lambdas   : (k,)
        residuals : (k,)
        summary   : dict with aggregate residual metrics
    """
    k = Z.shape[1]
    lambdas = np.full(k, np.nan)
    residuals = np.full(k, np.nan)

    for j in range(k):
        z = Z[:, j].astype(float)

        if normalize:
            denom = z.T @ Phi @ z
            if denom < eps:
                continue
            z = z / np.sqrt(denom)

        denom = z.T @ Phi @ z
        if denom < eps:
            continue

        lam = (z.T @ L @ z) / denom
        r = L @ z - lam * (Phi @ z)

        lambdas[j] = lam
        residuals[j] = np.linalg.norm(r)

    # Aggregates: median is robust, max catches worst-mode alignment
    summary = {
        "res_med": np.nanmedian(residuals),
        "res_mean": np.nanmean(residuals),
        "res_max": np.nanmax(residuals),
    }
    return lambdas, residuals, summary



def save_eigenvector_heatmaps(env: GraphEnv, Z: np.ndarray, save_dir: str, prefix: str = "eigenvec", cmap: str = "RdBu_r"):
    """
    Plot and save a heatmap for each column of Z (each eigenvector) using env.plot_heatmap().
    Z: (n_states, k) array.
    Saves to save_dir as {prefix}_0.pdf, ...
    """
    os.makedirs(save_dir, exist_ok=True)
    # Square figure size so heatmap isn't squashed; scale with grid for readability
    figsize = (max(5.0, env.n * 0.35), max(5.0, env.m * 0.35))
    k = Z.shape[1]
    for j in range(k):
        fig, ax = plt.subplots(figsize=figsize)
        env.plot_heatmap(values=Z[:, j], ax=ax, cmap=cmap, colorbar=False)
        eta_note = (
            rf", $\eta={env.transition_noise:g}$"
            if getattr(env, "transition_noise", 0) > 0
            else ""
        )
        ax.set_title(fr"{env.n}×{env.m}, {env.n_walls} walls{eta_note}, $\phi_{j+1}$", fontsize=10)
        plt.savefig(os.path.join(save_dir, f"{prefix}_{j}.pdf"), format="pdf", bbox_inches="tight")
        plt.close()


def _sample_indices(n: int, k: int = 5) -> List[int]:
    """Up to ``k`` indices spread across ``0 .. n-1`` (inclusive)."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({int(np.clip(round(x), 0, n - 1)) for x in np.linspace(0, n - 1, k)})


def write_experiment_summary(
    save_dir: str,
    args: Any,
    wall_values: Sequence[Union[int, float]],
    lambda2s: np.ndarray,
    analytical_errors: np.ndarray,
    gdo_errors: np.ndarray,
    analytical_error_vs_k: Optional[np.ndarray] = None,
    n_sample_points: int = 5,
    filename: str = "experiment_summary.txt",
) -> str:
    """
    Write a Markdown-formatted summary (tables): experiment parameters,
    LaTeX-style axis labels for main figures, per-plot sampled curve points, and aggregates.

    Saves UTF-8 text under ``save_dir`` (default ``experiment_summary.txt``).
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    ns = n_sample_points

    def _row(cells: List[str]) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    lines: List[str] = []
    lines.append("# Experiment summary (Markdown tables)")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.append(_row(["Parameter", "Value"]))
    lines.append(_row(["---", "---"]))

    param_rows = [
        ("Grid rows $n$ (``rows``)", getattr(args, "rows", "")),
        ("Grid cols $m$ (``cols``)", getattr(args, "cols", "")),
        ("Max wall index / sweep upper bound (``n_walls``)", getattr(args, "n_walls", "")),
        ("Wall step (``step_walls``)", getattr(args, "step_walls", "")),
        ("Number of random seeds (``n_seeds``)", getattr(args, "n_seeds", "")),
        ("Eigenvectors / subspace dim $k$ (``k``)", getattr(args, "k", "")),
        ("Base RNG seed (``seed``)", getattr(args, "seed", "")),
        (
            r"Teleport / uniform mix weight $\eta$ (``transition_noise``)",
            getattr(args, "transition_noise", 0.0),
        ),
        ("Heatmaps: eigenvectors plotted (``n_eigenvec_heatmaps``)", getattr(args, "n_eigenvec_heatmaps", "")),
        ("Heatmaps: wall counts sampled (``n_wall_heatmaps``)", getattr(args, "n_wall_heatmaps", "")),
    ]
    for name, val in param_rows:
        lines.append(_row([name, val]))

    wall_list = list(wall_values)
    lines.append(_row([r"Wall counts $w$ (actual)", ", ".join(str(int(w)) for w in wall_list)]))
    lines.append("")
    lines.append("## Figure filenames and axis labels (LaTeX)")
    lines.append("")
    lines.append(_row(["File", r"$x$-axis", r"$y$-axis"]))
    lines.append(_row(["---", "---", "---"]))
    fig_axes = [
        ("``lambda2.*``", r"Number of walls", r"$\lambda_2$ (2nd eigenvalue of normalized Laplacian)"),
        ("``errors.*``", r"Number of walls", r"Value function error $\|v - \hat v\|_\Phi$"),
        ("``error_vs_lambda2.*``", r"$\lambda_2$", r"Analytical error $\|v - \hat v\|_\Phi$"),
        ("``gdo_error_vs_lambda2.*``", r"$\lambda_2$", r"GDO error $\|v - \hat v\|_\Phi$"),
        ("``error_vs_n_eigenvec.*``", r"Number of eigenvectors $k$", r"Analytical error $\|v - \hat v\|_\Phi$"),
    ]
    for row in fig_axes:
        lines.append(_row(list(row)))

    # --- Sampled points per plot type (mean over seeds where applicable) ---
    ae = np.asarray(analytical_errors)
    ge = np.asarray(gdo_errors)
    lam = np.asarray(lambda2s)
    W = len(wall_list)
    wis = _sample_indices(W, ns)

    lines.append("")
    lines.append("## Sample values per figure")
    lines.append("")
    lines.append(
        f"_Each subsection matches one saved figure; values are a subset of "
        f"**{ns}** points along the sweep (wall index or sorted $\\lambda_2$). "
        r"Curve plots use **mean over seeds** where $w$ is the number of walls._"
    )
    lines.append("")

    # lambda2 vs walls
    lines.append("### ``lambda2.*`` — $\\lambda_2$ vs number of walls")
    lines.append("")
    lines.append(_row([r"$w$ (walls)", r"mean $\lambda_2$ over seeds"]))
    lines.append(_row(["---", "---"]))
    for wi in wis:
        lines.append(
            _row([str(int(wall_list[wi])), f"{np.nanmean(lam[:, wi]):.6g}"])
        )
    lines.append("")

    # errors vs walls
    lines.append("### ``errors.*`` — error vs number of walls")
    lines.append("")
    lines.append(
        _row(
            [
                r"$w$",
                r"mean analytical $\|v-\hat v\|_\Phi$",
                r"mean GDO $\|v-\hat v\|_\Phi$",
            ]
        )
    )
    lines.append(_row(["---", "---", "---"]))
    for wi in wis:
        lines.append(
            _row(
                [
                    str(int(wall_list[wi])),
                    f"{np.nanmean(ae[:, wi]):.6g}",
                    f"{np.nanmean(ge[:, wi]):.6g}",
                ]
            )
        )
    lines.append("")

    # error vs lambda2 (paired samples, sorted by lambda2)
    lines.append("### ``error_vs_lambda2.*`` / ``gdo_error_vs_lambda2.*`` — errors vs $\\lambda_2$")
    lines.append("")
    lam_f = lam.reshape(-1)
    ae_f = ae.reshape(-1)
    ge_f = ge.reshape(-1)
    mask = np.isfinite(lam_f) & np.isfinite(ae_f) & np.isfinite(ge_f)
    order = np.argsort(lam_f[mask])
    lam_s = lam_f[mask][order]
    ae_s = ae_f[mask][order]
    ge_s = ge_f[mask][order]
    n_s = len(lam_s)
    idxs = _sample_indices(n_s, ns) if n_s else []
    lines.append(
        _row(
            [
                r"$\lambda_2$",
                r"analytical error",
                r"GDO error",
            ]
        )
    )
    lines.append(_row(["---", "---", "---"]))
    for ii in idxs:
        lines.append(
            _row([f"{lam_s[ii]:.6g}", f"{ae_s[ii]:.6g}", f"{ge_s[ii]:.6g}"])
        )
    lines.append("")

    # error vs n eigenvectors
    if analytical_error_vs_k is not None and analytical_error_vs_k.size:
        ek = np.asarray(analytical_error_vs_k)
        k_dim = ek.shape[2]
        k_idx = _sample_indices(k_dim, ns)
        lines.append("### ``error_vs_n_eigenvec.*`` — analytical error vs $k$ eigenvectors")
        lines.append("")
        lines.append(
            _row(
                [
                    r"$k$ (eigenvectors used)",
                    r"mean error over $(\mathrm{seed}, w)$",
                ]
            )
        )
        lines.append(_row(["---", "---"]))
        for j in k_idx:
            kk = j + 1
            lines.append(
                _row([str(kk), f"{np.nanmean(ek[:, :, j]):.6g}"])
            )
        lines.append("")

    # Aggregate metrics (ae, ge, lam already defined above)

    lines.append("")
    lines.append("## Summary statistics (all $(\\mathrm{seed}, w)$ environments)")
    lines.append("")
    lines.append(_row(["Quantity", "Value"]))
    lines.append(_row(["---", "---"]))
    lines.append(
        _row(
            [
                r"Mean $\lambda_2$",
                f"{np.nanmean(lam):.6g}",
            ]
        )
    )
    lines.append(
        _row(
            [
                r"Mean analytical error $\|v-\hat v\|_\Phi$",
                f"{np.nanmean(ae):.6g}",
            ]
        )
    )
    lines.append(
        _row(
            [
                r"Std.\ analytical error",
                f"{np.nanstd(ae):.6g}",
            ]
        )
    )
    lines.append(
        _row(
            [
                r"Mean GDO error $\|v-\hat v\|_\Phi$",
                f"{np.nanmean(ge):.6g}",
            ]
        )
    )
    lines.append(
        _row(
            [
                r"Std.\ GDO error",
                f"{np.nanstd(ge):.6g}",
            ]
        )
    )

    if wall_list:
        wi0 = 0
        wi_last = len(wall_list) - 1
        lines.append(
            _row(
                [
                    r"Mean analytical error at $w = " + str(int(wall_list[wi0])) + r"$",
                    f"{np.nanmean(ae[:, wi0]):.6g}",
                ]
            )
        )
        lines.append(
            _row(
                [
                    r"Mean analytical error at $w = " + str(int(wall_list[wi_last])) + r"$",
                    f"{np.nanmean(ae[:, wi_last]):.6g}",
                ]
            )
        )

    if analytical_error_vs_k is not None and analytical_error_vs_k.size:
        ek = np.asarray(analytical_error_vs_k)
        k_dim = ek.shape[2]
        if k_dim >= 1:
            lines.append(
                _row(
                    [
                        r"Mean analytical error using $k=1$ eigenvector",
                        f"{np.nanmean(ek[:, :, 0]):.6g}",
                    ]
                )
            )
        if k_dim >= 2:
            lines.append(
                _row(
                    [
                        r"Mean analytical error using $k=" + str(k_dim) + r"$ eigenvectors",
                        f"{np.nanmean(ek[:, :, k_dim - 1]):.6g}",
                    ]
                )
            )

    lines.append("")
    lines.append(
        "_Data saved in ``data.npz`` in this folder; regenerate this file by re-running ``exps.py``._"
    )
    lines.append("")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def write_experiment_summary_from_npz(
    save_dir: str,
    args: Optional[Any] = None,
    filename: str = "experiment_summary.txt",
) -> str:
    """
    Rebuild ``experiment_summary.txt`` from ``data.npz`` if present (e.g. after adding this utility).
    Pass ``args`` if available so the parameter table is complete; otherwise only metrics from arrays are filled.
    """
    npz_path = os.path.join(save_dir, "data.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}")
    d = np.load(npz_path)
    wall_values = d["wall_values"].tolist()
    if args is None:
        class _NS:
            pass

        ns = _NS()
        ns.rows = int(d["rows"][0]) if "rows" in d.files else ""
        ns.cols = int(d["cols"][0]) if "cols" in d.files else ""
        ns.n_walls = int(d["n_walls"][0]) if "n_walls" in d.files else ""
        ns.step_walls = int(d["step_walls"][0]) if "step_walls" in d.files else ""
        ns.n_seeds = int(d["n_seeds"][0]) if "n_seeds" in d.files else ""
        ns.k = int(d["k"][0]) if "k" in d.files else ""
        ns.seed = ""
        ns.transition_noise = float(d["transition_noise"][0]) if "transition_noise" in d.files else 0.0
        ns.n_eigenvec_heatmaps = ""
        ns.n_wall_heatmaps = ""
        args = ns

    aevk = d["analytical_error_vs_k"] if "analytical_error_vs_k" in d.files else None
    return write_experiment_summary(
        save_dir,
        args,
        wall_values,
        d["lambda2s"],
        d["analytical_errors"],
        d["gdo_errors"],
        analytical_error_vs_k=aevk,
        filename=filename,
    )