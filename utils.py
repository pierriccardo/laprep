import numpy as np
from graph_env import GraphEnv
import os
import matplotlib.pyplot as plt


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
    Saves to save_dir as {prefix}_0.png, {prefix}_1.png, ...
    """
    os.makedirs(save_dir, exist_ok=True)
    # Square figure size so heatmap isn't squashed; scale with grid for readability
    figsize = (max(5.0, env.n * 0.35), max(5.0, env.m * 0.35))
    k = Z.shape[1]
    for j in range(k):
        fig, ax = plt.subplots(figsize=figsize)
        env.plot_heatmap(values=Z[:, j], ax=ax, cmap=cmap, colorbar=False)
        ax.set_title(fr"{env.n}×{env.m}, {env.n_walls} walls, $\phi_{j+1}$")
        plt.savefig(os.path.join(save_dir, f"{prefix}_{j}.png"), format="png", dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(save_dir, f"{prefix}_{j}.pdf"), format="pdf", bbox_inches="tight")
        plt.close()