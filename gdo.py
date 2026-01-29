import os
import sys
import tyro
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from graph_env import GraphEnv


@dataclass
class Config:
    # Algo
    k: int = 10
    n_states: int | None = None
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    learning_rate: float = 1e-3
    batch_size: int = 256
    train_steps: int = 10000
    beta_ortho: float = 1.0
    beta_neg: float = 0.0  # weight for negative-sample loss (0 = off)
    seed: int = 42
    log_interval: int = 100
    eval_interval: int = 0

    # Buffer
    buffer_size: int = 10**5

    # Env
    n: int = 10
    m: int = 10
    n_walls: int = 20
    max_steps: int = 1

    # Misc
    device: str = "cpu"
    save_dir: str = "models"
    log_freq: int = 10


class ReplayBuffer:

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity: int = capacity
        self.size: int = 0
        self._position: int = 0

        # Pre-allocate arrays for efficiency
        self.u: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.v: np.ndarray = np.zeros(capacity, dtype=np.int64)

    def add(self, u: int, v: int) -> None:
        """
        Add a single transition (u, v) to the buffer.

        Args:
            u: Source state index.
            v: Destination state index.
        """
        self.u[self._position] = u
        self.v[self._position] = v

        self._position = (self._position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, u: np.ndarray, v: np.ndarray) -> None:
        """
        Add a batch of transitions to the buffer.

        Args:
            u: Array of source state indices.
            v: Array of destination state indices.

        Raises:
            ValueError: If u and v have different lengths.
        """
        if len(u) != len(v):
            raise ValueError(f"u and v must have same length, got {len(u)} and {len(v)}")

        n = len(u)

        if n == 0:
            return

        # Handle case where batch fits in remaining space
        if self._position + n <= self.capacity:
            self.u[self._position:self._position + n] = u
            self.v[self._position:self._position + n] = v
            self._position = (self._position + n) % self.capacity
        else:
            # Batch wraps around
            first_chunk = self.capacity - self._position
            self.u[self._position:] = u[:first_chunk]
            self.v[self._position:] = v[:first_chunk]

            remaining = n - first_chunk
            if remaining > 0:
                # If remaining is larger than capacity, only keep the last capacity elements
                if remaining >= self.capacity:
                    self.u[:] = u[-self.capacity:]
                    self.v[:] = v[-self.capacity:]
                    self._position = 0
                else:
                    self.u[:remaining] = u[first_chunk:]
                    self.v[:remaining] = v[first_chunk:]
                    self._position = remaining

        self.size = min(self.size + n, self.capacity)

    def sample(
        self,
        batch_size: int,
        n_states: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample (u, v) positive pairs and v_neg random state indices (negative batch)."""
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with {self.size} elements"
            )
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.choice(self.size, size=batch_size, replace=False)
        u = self.u[indices].copy()
        v = self.v[indices].copy()
        v_neg = rng.integers(0, n_states, size=batch_size, dtype=np.int64)
        return u, v, v_neg

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.size = 0
        self._position = 0


# ==================================================
# Networks
# ==================================================
class LapNetwork(nn.Module):
    def __init__(self, n_states: int, k: int):
        super().__init__()

        # Embedding layer, equivalent to one-hot encoding
        # but more efficient
        self.embedding = nn.Embedding(n_states, k)

        # Optional: small init helps stability
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1e-2)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.embedding(states)


class LapRep():

    def __init__(self, config: Config) -> None:
        self.config = config
        self.network = LapNetwork(
            config.n_states,
            config.k,
        )
        self.network.to(torch.device(config.device))

        self.optimizer = optim.AdamW(self.network.parameters(), lr=config.learning_rate, eps = 1e-10)


    def neg_loss(self, x, c=1.0, reg=0.0):
        """
        x: n * d.
        sample based approximation for
        (E[x x^T] - c * I / d)^2
            = E[(x^T y)^2] - 2c E[x^T x] / d + c^2 / d
        #
        An optional regularization of
        reg * E[(x^T x - c)^2] / n
            = reg * E[(x^T x)^2 - 2c x^T x + c^2] / n
        for reg in [0, 1]
        """
        n = x.shape[0]
        d = x.shape[1]
        inprods = x @ x.T
        norms = inprods[torch.arange(n), torch.arange(n)]
        part1 = inprods.pow(2).sum() - norms.pow(2).sum()
        part1 = part1 / ((n - 1) * n)
        part2 = - 2 * c * norms.mean() / d
        part3 = c * c / d
        # regularization
        if reg > 0.0:
            reg_part1 = norms.pow(2).mean()
            reg_part2 = - 2 * c * norms.mean()
            reg_part3 = c * c
            reg_part = (reg_part1 + reg_part2 + reg_part3) / n
        else:
            reg_part = 0.0
        return part1 + part2 + part3 + reg * reg_part

    def update(self, batch):
        u, v, v_neg = batch
        u = u.to(self.config.device)
        v = v.to(self.config.device)
        v_neg = v_neg.to(self.config.device)

        phi_u = self.network(u)
        phi_v = self.network(v)
        loss_smooth = 0.5 * ((phi_u - phi_v) ** 2).sum(dim=1).mean()

        #all_states = torch.cat([u, v], dim=0)
        #phi_all = self.network(all_states)
        #phi_all = phi_all - phi_all.mean(dim=0, keepdim=True)

        #gram = (phi_all.T @ phi_all) / phi_all.shape[0]
        #I = torch.eye(self.config.k, device=self.config.device, dtype=phi_all.dtype)
        #loss_ortho = ((gram - I) ** 2).sum()

        phi_v_neg = self.network(v_neg)
        loss_neg = self.neg_loss(phi_v_neg, c=1.0, reg=0.0)
        total_loss = loss_smooth + self.config.beta_neg * loss_neg

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss_smooth": loss_smooth.item(),
            #"loss_ortho": loss_ortho.item(),
            "loss_neg": loss_neg.item(),
            "total_loss": total_loss.item(),
        }

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.network(x)

    def save(self, filename: Optional[str] = None) -> str:
        """Save the model weights. Filename defaults to gdo_n{n}_m{m}_w{n_walls}_s{seed}.pt."""
        if filename is None:
            filename = f"gdo_n{self.config.n}_m{self.config.m}_w{self.config.n_walls}_s{self.config.seed}.pt"
        os.makedirs(self.config.save_dir, exist_ok=True)
        filepath = os.path.join(self.config.save_dir, filename)
        torch.save(self.network.state_dict(), filepath)
        return filepath

    def load(self, filepath: str) -> None:
        """Load model weights from a .pt file. Uses current config (no config stored in file)."""
        device = torch.device(self.config.device)
        raw = torch.load(filepath, map_location=device, weights_only=True)
        state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        self.network.load_state_dict(state_dict)
        self.network.to(device)
        self.network.eval()




if __name__ == "__main__":

    config = tyro.cli(Config)

    # Seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    env = GraphEnv(config.n, config.m, config.n_walls, config.seed, max_steps=config.max_steps)
    config.n_states = env.n_states

    save_path = os.path.join(config.save_dir, f"gdo_n{config.n}_m{config.m}_w{config.n_walls}_s{config.seed}.pt")
    if os.path.isfile(save_path):
        print(f"Already trained: {save_path}")
        sys.exit(0)

    buffer = ReplayBuffer(config.buffer_size)

    # Fill the buffer with uniform policy
    s, _ = env.reset()
    while len(buffer) < config.buffer_size:

        action = np.random.randint(env.n_actions)
        s_next, reward, terminated, truncated, _ = env.step(action)

        buffer.add(s, s_next)
        s = s_next

        if terminated or truncated:
            s, _ = env.reset()

    model = LapRep(config)

    for step in range(config.train_steps):
        u, v, v_neg = buffer.sample(config.batch_size, config.n_states, rng=rng)
        metrics = model.update((torch.from_numpy(u), torch.from_numpy(v), torch.from_numpy(v_neg)))
        if step % config.log_freq == 0:
            print(f"step {step}" + "".join([f"{k}: {v} " for k, v in metrics.items()]))

    model.save(f"gdo_n{config.n}_m{config.m}_w{config.n_walls}_s{config.seed}.pt")


