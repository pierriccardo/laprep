import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb


Coord = Tuple[int, int]
Edge = Tuple[Coord, Coord]


class GraphEnv():

    def __init__(self, n: int, m: int, n_walls: int, seed: int, goal: Coord | None = None, max_steps: int = 1000) -> None:
        self.n = n
        self.m = m
        self.n_walls = n_walls
        self.seed = seed

        # Max num of step to send truncation signal
        self.max_steps = max_steps

        self.graph = self._make_graph_by_walls()
        self.goal = (n - 1, m - 1) if goal is None else goal
        self.id2pos = {i: pos for i, pos in enumerate(self.graph.keys())}
        self.pos2id = {pos: i for i, pos in self.id2pos.items()}
        self.n_states = self.n * self.m
        self.goal_state = self.pos2id[self.goal]

        # Current state
        self.state = None
        self.steps = 0

        # Actions
        self.action_to_string: Dict[int, str] = {
            0: "up",
            1: "right",
            2: "down",
            3: "left"
        }
        self.actions: Dict[int, Tuple[int, int]] = {
            0: (-1, 0),   # up: decrease row
            1: (0, 1),    # right: increase col
            2: (1, 0),    # down: increase row
            3: (0, -1)    # left: decrease col
        }
        self.n_actions = len(self.actions.keys())
        self.P = self._transition_matrix()

    def get_reward_matrix(self) -> np.ndarray:
        """
        Get the reward matrix R(s, a, s').

        Returns:
            Numpy array of shape (n_states, n_actions, n_states) where
            R[s, a, s'] is the reward for transition (s, a) -> s'.
        """
        R = np.full((self.n_states, self.n_actions, self.n_states), -1.0)
        # Reward +1 for entering the goal state
        R[:, :, self.goal_state] = 1.0
        return R

    def get_expected_reward(self) -> np.ndarray:
        """
        Get the expected reward for each (state, action) pair.

        For deterministic environments, this is simply R(s, a, s') where
        s' is the unique next state.

        Returns:
            Numpy array of shape (n_states, n_actions).
        """
        R_full = self.get_reward_matrix()
        # Sum over s' weighted by P (for deterministic, equivalent to just picking the one s')
        return np.sum(self.P * R_full, axis=2)

    def _transition_matrix(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float64)

        for s in range(self.n_states):
            row, col = self.id2pos[s]

            for a in range(self.n_actions):
                delta_row, delta_col = self.actions[a]
                new_row = row + delta_row
                new_col = col + delta_col

                # Check if new position is valid (in bounds and not a wall)
                if (new_row, new_col) in self.graph[(row, col)]:
                    s_next = self.pos2id[(new_row, new_col)]
                else:
                    s_next = s  # Stay in place if move is invalid
                P[s, a, s_next] = 1.0
        return P

    def _make_graph_by_walls(self):
        n_states = self.n * self.m
        Emax = self.n * (self.m - 1) + self.m * (self.n - 1)
        Wmax = Emax - (n_states - 1)

        if self.n_walls < 0 or self.n_walls > Wmax:
            raise ValueError(f"n_walls must be in [0, {Wmax}] for connectivity.")

        open_edges = Emax - self.n_walls
        return self._make_fixed_state_connected_grid_graph(self.n, self.m, open_edges=open_edges, seed=self.seed)

    def _neighbors_4(self, r: int, c: int, n: int, m: int) -> List[Coord]:
        out: List[Coord] = []
        if r > 0:
            out.append((r - 1, c))
        if r + 1 < n:
            out.append((r + 1, c))
        if c > 0:
            out.append((r, c - 1))
        if c + 1 < m:
            out.append((r, c + 1))
        return out

    def _canon_edge(self, a: Coord, b: Coord) -> Edge:
        return (a, b) if a <= b else (b, a)

    def _make_fixed_state_connected_grid_graph(
        self,
        n: int,
        m: int,
        *,
        open_edges: Optional[int] = None,
        extra_edges: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[Coord, List[Coord]]:
        """
        Build a connected graph on exactly n*m states (all grid cells).
        Walls live on edges: blocking an edge removes that adjacency but does not remove states.

        You can control connectivity by:
        - open_edges: total number of undirected open edges to keep
        - or extra_edges: how many edges to add beyond a spanning tree (tree has n*m - 1 edges)

        Exactly one of (open_edges, extra_edges) can be given; if neither is given,
        defaults to extra_edges=0 (a random spanning tree: minimally connected).

        Returns:
        adjacency dict mapping each cell to list of neighboring cells reachable (open edges).
        """
        if n <= 0 or m <= 0:
            raise ValueError("n and m must be positive.")
        rng = np.random.default_rng(seed)

        nodes: List[Coord] = [(r, c) for r in range(n) for c in range(m)]
        num_nodes = n * m

        # All possible undirected edges in a 4-neighbor grid
        all_edges: List[Edge] = []
        for r in range(n):
            for c in range(m):
                a = (r, c)
                for b in self._neighbors_4(r, c, n, m):
                    e = self._canon_edge(a, b)
                    # only include once
                    if e not in all_edges:
                        all_edges.append(e)

        max_edges = len(all_edges)
        min_edges = num_nodes - 1  # any connected simple graph needs at least this many

        if open_edges is not None and extra_edges is not None:
            raise ValueError("Provide only one of open_edges or extra_edges.")

        if open_edges is None and extra_edges is None:
            extra_edges = 0

        if open_edges is None:
            open_edges = min_edges + int(extra_edges)

        if open_edges < min_edges:
            raise ValueError(f"open_edges too small for connectivity: need at least {min_edges}.")
        if open_edges > max_edges:
            raise ValueError(f"open_edges too large: at most {max_edges} in a 4-neighbor grid.")

        # Step 1: build a random spanning tree via randomized BFS/DFS-like growth.
        # This guarantees connectivity with exactly (num_nodes - 1) edges.
        remaining: Set[Coord] = set(nodes)
        start = nodes[int(rng.integers(num_nodes))]
        remaining.remove(start)

        tree_edges: Set[Edge] = set()
        frontier: List[Coord] = [start]

        while remaining:
            # pick a random node already in the grown component
            a = frontier[int(rng.integers(len(frontier)))]
            nbrs = self._neighbors_4(a[0], a[1], n, m)
            rng.shuffle(nbrs)

            connected = False
            for b in nbrs:
                if b in remaining:
                    remaining.remove(b)
                    tree_edges.add(self._canon_edge(a, b))
                    frontier.append(b)
                    connected = True
                    break

            if not connected:
                # a cannot expand anymore; drop it from frontier
                frontier.remove(a)
                if not frontier:
                    # Should not happen on a grid, but keep safe: restart frontier from any grown node
                    grown = set(nodes) - remaining
                    frontier = [next(iter(grown))]

        open_set: Set[Edge] = set(tree_edges)

        # Step 2: add extra random edges (remove "walls") until we reach open_edges.
        remaining_edges = [e for e in all_edges if e not in open_set]
        rng.shuffle(remaining_edges)

        need = open_edges - len(open_set)
        for i in range(need):
            open_set.add(remaining_edges[i])

        # Build adjacency
        adj: Dict[Coord, List[Coord]] = {v: [] for v in nodes}
        for (a, b) in open_set:
            adj[a].append(b)
            adj[b].append(a)

        return adj

    def graph_to_edge_walls(self, n: int, m: int, adj: Dict[Coord, List[Coord]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert adjacency into two wall arrays:
        - wall_right[r,c] == 1 means wall between (r,c) and (r,c+1)
        - wall_down[r,c]  == 1 means wall between (r,c) and (r+1,c)

        These are the "toggleable walls". States remain all N*M cells.
        """
        wall_right = np.ones((n, m - 1), dtype=int)
        wall_down = np.ones((n - 1, m), dtype=int)

        def is_open(a: Coord, b: Coord) -> bool:
            return b in adj.get(a, [])

        for r in range(n):
            for c in range(m - 1):
                if is_open((r, c), (r, c + 1)):
                    wall_right[r, c] = 0

        for r in range(n - 1):
            for c in range(m):
                if is_open((r, c), (r + 1, c)):
                    wall_down[r, c] = 0

        return wall_right, wall_down

    def render(self) -> np.ndarray:
        """
        Visual matrix of size (2n-1, 2m-1)

        Even-even indices   -> states
        Odd-even / even-odd -> walls (1) or open edges (0)
        """
        goal = self.goal
        adj = self.graph
        n = self.n
        m = self.m
        H = 2 * n + 1
        W = 2 * m + 1
        mat = np.full((H, W), "#", dtype="U1")

        # place states
        for r in range(n):
            for c in range(m):
                mat[2 * r + 1, 2 * c + 1] = " "

        if goal is not None:
            gr, gc = goal
            mat[2 * gr + 1, 2 * gc + 1] = "G"

        # open edges
        for (r, c), nbrs in adj.items():
            for (rr, cc) in nbrs:
                if rr == r and cc == c + 1:      # right
                    mat[2 * r + 1, 2 * c + 2] = " "
                elif rr == r and cc == c - 1:    # left
                    mat[2 * r + 1, 2 * c] = " "
                elif rr == r + 1 and cc == c:    # down
                    mat[2 * r + 2, 2 * c + 1] = " "
                elif rr == r - 1 and cc == c:    # up
                    mat[2 * r, 2 * c + 1] = " "

        # set the borders to '#'
        mat[0, :] = "#"
        mat[-1, :] = "#"
        mat[:, 0] = "#"
        mat[:, -1] = "#"

        # return as string visualization
        return "\n".join("".join(row) for row in mat)

    def plot_heatmap(
        self,
        values: Optional[Union[np.ndarray, List[float]]] = None,
        ax: Optional["plt.Axes"] = None,
        wall_color: str = "#808080",
        goal_color: str = "#00e676",
        cell_color: str = "white",
        cmap: Optional[str] = None,
        colorbar: bool = True,
        **imshow_kwargs,
    ):
        """
        Plot the graph env as a heatmap: gray walls, bright green goal, white cells (or heatmap from values).

        Uses a (2n+1) x (2m+1) grid: cells at odd indices, walls at even indices between them.
        Requires matplotlib.

        Args:
            values: Optional array of shape (n_states,) or (n, m) to color cells (heatmap).
                If None, cells are white and the goal is bright green.
            ax: Matplotlib axes. If None, uses current figure.
            wall_color: Color for walls (default gray).
            goal_color: Color for the goal cell (default bright green).
            cell_color: Default cell color when values is None (default white).
            cmap: Colormap name or instance when values is not None (e.g. 'viridis', 'plasma').
            colorbar: Whether to show colorbar when values is not None.
            **imshow_kwargs: Passed to imshow (e.g. vmin, vmax).
        """

        n, m = self.n, self.m
        wall_right, wall_down = self.graph_to_edge_walls(n, m, self.graph)
        gr, gc = self.goal

        # (2n+1) x (2m+1): cells at (2r+1, 2c+1), edges between them. Values:
        # WALL=0.5, PASSAGE=0 (or PASSAGE_SENTINEL when values used), cell = value or GOAL_SENTINEL
        H, W = 2 * n + 1, 2 * m + 1
        mat = np.full((H, W), 0.5)  # all wall by default (including corners)

        # Open edges: passage pixels (not wall). Use 0 when no values; with values use a sentinel
        # so cell value 0.0 is not confused with passage.
        PASSAGE_SENTINEL = -0.5
        for r in range(n):
            for c in range(m - 1):
                if not wall_right[r, c]:
                    mat[2 * r + 1, 2 * c + 2] = PASSAGE_SENTINEL if values is not None else 0.0
        for r in range(n - 1):
            for c in range(m):
                if not wall_down[r, c]:
                    mat[2 * r + 2, 2 * c + 1] = PASSAGE_SENTINEL if values is not None else 0.0

        # Cells
        for r in range(n):
            for c in range(m):
                mat[2 * r + 1, 2 * c + 1] = 1.0 if (r, c) == (gr, gc) else 0.0

        # If values provided, overwrite cell positions with normalized values (for heatmap)
        # Use GOAL_SENTINEL = 2.0 so only the actual goal gets goal_color; heatmap max is 1.0
        GOAL_SENTINEL = 2.0
        if values is not None:
            v = np.asarray(values, dtype=float)
            if v.size == self.n_states:
                v = v.reshape(n, m)
            elif v.shape != (n, m):
                raise ValueError(f"values must have size n_states={self.n_states} or shape (n,m)=({n},{m}), got {v.shape}")
            vmin = imshow_kwargs.pop("vmin", None)
            vmax = imshow_kwargs.pop("vmax", None)
            if vmin is None:
                vmin = np.nanmin(v)
            if vmax is None:
                vmax = np.nanmax(v)
            if vmax <= vmin:
                vmax = vmin + 1
            v_norm = (v - vmin) / (vmax - vmin)
            for r in range(n):
                for c in range(m):
                    if (r, c) == (gr, gc):
                        mat[2 * r + 1, 2 * c + 1] = GOAL_SENTINEL  # only goal gets goal color
                    else:
                        mat[2 * r + 1, 2 * c + 1] = v_norm[r, c]

        # Colormap: 0 -> cell_color, 0.5 -> wall_color, 1 -> goal_color (or cmap(1) when using values)
        # We need a custom discrete map: [0, 0.5, 1] -> [white, gray, goal_color] when values is None
        if values is None:
            custom = LinearSegmentedColormap.from_list(
                "graph_env",
                [cell_color, wall_color, goal_color],
                N=256,
            )
            if ax is None:
                ax = plt.gca()
            im = ax.imshow(mat, cmap=custom, vmin=0, vmax=1, aspect="equal", interpolation="nearest", **imshow_kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            # No colorbar when values is None (white/gray/goal only)
        else:
            cm = plt.get_cmap(cmap or "viridis")
            passage_rgba = (*to_rgb(cell_color), 1.0)  # passages = same as default cell (e.g. white)
            display = np.zeros((H, W, 4))
            for i in range(H):
                for j in range(W):
                    if mat[i, j] == 0.5:
                        display[i, j] = (*to_rgb(wall_color), 1.0)
                    elif mat[i, j] <= PASSAGE_SENTINEL:
                        display[i, j] = passage_rgba  # open corridor
                    elif mat[i, j] >= GOAL_SENTINEL:
                        display[i, j] = (*to_rgb(goal_color), 1.0)
                    else:
                        display[i, j] = cm(mat[i, j])
            if ax is None:
                ax = plt.gca()
            im = ax.imshow(display, aspect="equal", interpolation="nearest", **imshow_kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            if colorbar:
                sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                plt.colorbar(sm, ax=ax, shrink=0.6)
        return im

    def reward(self, s: int, a: int, s_next: int) -> float:
        if s_next == self.goal_state:
            return 1.0
        return -1.0

    def step(self, a: int):
        #s_next = self.rng.choice(self.n_states, p=self.P[s, a])
        pos = self.id2pos[self.state]
        action = self.actions[a]

        next_pos = (pos[0] + action[0], pos[1] + action[1])
        if next_pos in self.graph[pos]:
            s_next = self.pos2id[next_pos]
        else:
            s_next = self.state

        # Reward: +1 for entering goal, -1 otherwise (including leaving goal)
        r = self.reward(self.state, a, s_next)

        terminated = s_next == self.goal_state
        truncated = self.steps >= self.max_steps
        self.steps += 1

        self.state = s_next

        return s_next, r, terminated, truncated, {}

    def reset(self):
        self.steps = 0
        self.state = np.random.randint(self.n_states)
        while self.state == self.goal_state:
            self.state = np.random.randint(self.n_states)
        return self.state, {}

    def __repr__(self) -> str:
        g = ""
        for i, (k, v) in enumerate(self.graph.items()):
            g += f"state {i}, {k} -> {v}\n"
        return g

if __name__ == "__main__":

    env = GraphEnv(5, 10, 36, 2)
    print(env.render())
    print(env)
    env.plot_heatmap()
    plt.show()