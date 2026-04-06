from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tyro
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_env import GraphEnv
from utils import (
    compute_stationary_distribution,
    policy_induced_transition_matrix,
    uniform_policy,
)

from scipy.linalg import eigh


@dataclass
class Args:
    rows: int = 30
    cols: int = 30
    seed: int = 0
    start_walls: int = 5
    n_sweeps: int = 10
    end_walls: int | None = None
    end_fraction_of_max: float = 0.6
    output: str = "graph_env_wall_sweep.pdf"
    dpi: int = 200
    graph_node_size: int = 16
    graph_line_width: float = 0.7


def max_walls_for_connected_grid(rows: int, cols: int) -> int:
    n_states = rows * cols
    emax = rows * (cols - 1) + cols * (rows - 1)
    return emax - (n_states - 1)


def choose_wall_counts(args: Args) -> list[int]:
    wmax = max_walls_for_connected_grid(args.rows, args.cols)
    end_walls = args.end_walls
    if end_walls is None:
        end_walls = int(np.ceil(args.end_fraction_of_max * wmax))
    if args.start_walls < 0:
        raise ValueError(f"start_walls must be non-negative, got {args.start_walls}")
    if end_walls < args.start_walls:
        raise ValueError(
            f"end_walls must be >= start_walls, got start={args.start_walls}, end={end_walls}"
        )
    if end_walls > wmax:
        raise ValueError(f"end_walls must be <= max walls {wmax}, got {end_walls}")
    if args.n_sweeps <= 0:
        raise ValueError(f"n_sweeps must be positive, got {args.n_sweeps}")

    raw = np.linspace(args.start_walls, end_walls, num=args.n_sweeps)
    walls = np.rint(raw).astype(int).tolist()
    return sorted(set(walls))


def build_nx_graph(env: GraphEnv) -> nx.Graph:
    graph = nx.Graph()
    for node, neighbors in env.graph.items():
        graph.add_node(node)
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)
    return graph


def compute_representation_lambda2(env: GraphEnv) -> float:
    policy = uniform_policy(env)
    P_pi = policy_induced_transition_matrix(env, policy)
    phi = compute_stationary_distribution(P_pi)
    Phi = np.diag(phi)

    L = Phi - (Phi @ P_pi + P_pi.T @ Phi) / 2
    inv_phi = np.diag(1.0 / phi)
    normalized_laplacian = inv_phi @ L

    eigvals = np.linalg.eigvals(normalized_laplacian)
    eigvals = np.sort(eigvals.real)
    return float(eigvals[1])


def draw_gridworld(ax: plt.Axes, env: GraphEnv) -> None:
    rows, cols = env.n, env.m
    wall_right, wall_down = env.graph_to_edge_walls(rows, cols, env.graph)

    ax.set_facecolor("white")

    cell_edges = []
    for r in range(rows + 1):
        cell_edges.append([(0, r), (cols, r)])
    for c in range(cols + 1):
        cell_edges.append([(c, 0), (c, rows)])
    ax.add_collection(LineCollection(cell_edges, colors="#d9d9d9", linewidths=0.5))

    wall_segments = []
    for r in range(rows):
        for c in range(cols - 1):
            if wall_right[r, c]:
                x = c + 1
                wall_segments.append([(x, r), (x, r + 1)])
    for r in range(rows - 1):
        for c in range(cols):
            if wall_down[r, c]:
                y = r + 1
                wall_segments.append([(c, y), (c + 1, y)])
    if wall_segments:
        ax.add_collection(LineCollection(wall_segments, colors="black", linewidths=2.0))

    goal_r, goal_c = env.goal
    ax.scatter(
        [goal_c + 0.5],
        [goal_r + 0.5],
        s=28,
        c="#00a651",
        marker="s",
        edgecolors="none",
        zorder=3,
    )

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Gridworld", fontsize=12)

def koren_degree_normalized_layout(G):
    A = nx.to_numpy_array(G, dtype=float)
    d = A.sum(axis=1)
    D = np.diag(d)
    L = D - A

    # Solve L u = mu D u
    vals, vecs = eigh(L, D)

    # Skip the trivial first eigenvector
    x = vecs[:, 1]
    y = vecs[:, 2]

    return {node: (x[i], y[i]) for i, node in enumerate(G.nodes())}

def draw_graph(ax: plt.Axes, env: GraphEnv, node_size: int, line_width: float) -> None:
    graph = build_nx_graph(env)

    pos = {(r, c): (c, -r) for r in range(env.n) for c in range(env.m)}

    pos = koren_degree_normalized_layout(graph)

    """
        pos = nx.spring_layout(
            graph,
            seed=0,          # reproducible
            iterations=300,  # usually gives a cleaner result
        )
    """

    #pos = nx.kamada_kawai_layout(graph)

    #pos = nx.spectral_layout(graph)

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        width=line_width,
        edge_color="#555555",
    )
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        ax=ax,
        node_size=node_size,
        node_color="#4c78a8",
        linewidths=0,
    )
    goal = env.goal
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        nodelist=[goal],
        ax=ax,
        node_size=max(node_size * 2, node_size + 6),
        node_color="#00a651",
        linewidths=0,
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("NetworkX Graph", fontsize=12)


def make_page(env: GraphEnv, wall_count: int, max_walls: int, args: Args) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    draw_gridworld(axes[0], env)
    draw_graph(axes[1], env, node_size=args.graph_node_size, line_width=args.graph_line_width)
    lambda2 = compute_representation_lambda2(env)

    graph = build_nx_graph(env)
    title_text = (
        f"Grid {env.n}x{env.m} | walls={wall_count}/{max_walls} | "
        f"open edges={graph.number_of_edges()} | nodes={graph.number_of_nodes()} | "
        f"seed={env.seed} | "
        rf"$\mathbf{{\lambda_2 = {lambda2:.6f}}}$"
    )
    fig.suptitle(
        title_text,
        fontsize=14,
        color="#ff6b81",
    )
    return fig


def main(args: Args) -> None:
    max_walls = max_walls_for_connected_grid(args.rows, args.cols)
    wall_counts = choose_wall_counts(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for wall_count in wall_counts:
            env = GraphEnv(args.rows, args.cols, wall_count, args.seed)
            fig = make_page(env, wall_count, max_walls, args)
            pdf.savefig(fig, dpi=args.dpi)
            plt.close(fig)

    print(f"Saved {len(wall_counts)} pages to {output_path}")
    print(f"Wall counts: {wall_counts}")
    print(f"Maximum connected-wall count for {args.rows}x{args.cols}: {max_walls}")


if __name__ == "__main__":
    main(tyro.cli(Args))
