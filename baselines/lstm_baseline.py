"""
lstm_baseline.py — LSTM load-prediction baseline.

A per-server LSTM observes a sliding window of the last WINDOW_SIZE steps
of [load, hazard_rate, session_ratio] for each server and predicts load
HORIZON steps ahead.  If predicted load > MIGRATE_THRESHOLD, it triggers
migration of top-20 sessions — identical migration action as all other
policies.

This baseline has TEMPORAL MEMORY but NO TOPOLOGY AWARENESS.
GNN has TOPOLOGY but only sees the current step (no history).

The comparison directly answers: "does topology matter more than history?"

Design:
  - Offline supervised pre-training on simulator trajectories
  - Online inference: maintain rolling window per server, predict, decide
  - Identical migration logic to ThresholdPolicy
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import deque

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import SessionStatus


WINDOW_SIZE = 10       # steps of history fed to LSTM
HORIZON = 5            # predict load this many steps ahead
FEAT_DIM = 3           # per-server features: load, hazard_rate, session_ratio
HIDDEN_DIM = 64
N_LAYERS = 2
MIGRATE_THRESHOLD = 0.65   # predicted load threshold to trigger migration
SESSIONS_PER_SERVER = 20   # same as all other policies


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMLoadPredictor(nn.Module):
    """
    Per-server LSTM: maps (window_size, feat_dim) → scalar predicted load.
    Applied independently to each server (no cross-server information).
    """

    def __init__(self, feat_dim: int = FEAT_DIM, hidden_dim: int = HIDDEN_DIM,
                 n_layers: int = N_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),   # load is in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window_size, feat_dim)
        returns: (batch,) predicted load
        """
        out, _ = self.lstm(x)          # (batch, window, hidden)
        last = out[:, -1, :]           # take last timestep
        return self.head(last).squeeze(-1)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_training_data(n_episodes: int = 100, steps: int = 200,
                          seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Run simulator and collect (window, future_load) pairs for supervised training.
    Returns:
      X: (N_samples, window_size, feat_dim)
      y: (N_samples,) load at step t+HORIZON
    """
    cfg = SimConfig(
        topology="fat_tree", n_servers=16, seed=seed,
        session_arrival_rate=10.0, session_duration_mean=60.0,
        base_load=0.55, spike_probability=0.08, spike_magnitude=0.35,
        enable_failures=True, max_steps=steps,
    )

    X_list, y_list = [], []

    for ep in range(n_episodes):
        cfg_ep = SimConfig(
            topology=cfg.topology, n_servers=cfg.n_servers, seed=seed + ep,
            session_arrival_rate=cfg.session_arrival_rate,
            session_duration_mean=cfg.session_duration_mean,
            base_load=cfg.base_load, spike_probability=cfg.spike_probability,
            spike_magnitude=cfg.spike_magnitude,
            enable_failures=cfg.enable_failures, max_steps=cfg.max_steps,
        )
        sim = ClusterSimulator(cfg_ep)
        n = sim.n
        obs = sim.reset()

        # per-server rolling buffer
        history = [deque(maxlen=WINDOW_SIZE + HORIZON) for _ in range(n)]

        step_feats = []   # (steps, n, feat_dim)
        for _ in range(steps):
            feats = []
            for sv in range(n):
                node = sim.topo.get_node(sv)
                f = [
                    float(node.load),
                    float(node.hazard_rate),                    float(min(len(sim.server_sessions[sv]) / 200.0, 1.0)),
                ]
                feats.append(f)
            step_feats.append(feats)
            obs, _, done, _ = sim.step({})
            if done:
                break

        T = len(step_feats)
        # Build (window, target) pairs
        for t in range(WINDOW_SIZE, T - HORIZON):
            for sv in range(n):
                window = np.array([step_feats[t2][sv] for t2 in range(t - WINDOW_SIZE, t)],
                                  dtype=np.float32)   # (window_size, feat_dim)
                target = float(step_feats[t + HORIZON][sv][0])   # future load
                X_list.append(window)
                y_list.append(target)

    X = np.stack(X_list)                    # (N, window_size, feat_dim)
    y = np.array(y_list, dtype=np.float32)  # (N,)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(save_path: str = "results/lstm_predictor.pt",
               n_episodes: int = 100, epochs: int = 20,
               batch_size: int = 512, lr: float = 1e-3) -> LSTMLoadPredictor:

    print(f"Collecting training data ({n_episodes} episodes)...")
    X, y = collect_training_data(n_episodes=n_episodes)
    print(f"  Collected {len(X):,} samples")

    model = LSTMLoadPredictor()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True)

    print(f"Training LSTM for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg = total_loss / len(X)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs}  MSE={avg:.5f}  "
                  f"RMSE={avg**0.5:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  Saved to {save_path}")
    return model


# ---------------------------------------------------------------------------
# Inference-time policy
# ---------------------------------------------------------------------------

class LSTMPolicy:
    """
    Drop-in replacement for ThresholdPolicy.
    Maintains a per-server rolling feature window; on each call to
    select_action(obs, sim) predicts load HORIZON steps ahead and
    migrates if predicted load > MIGRATE_THRESHOLD.
    """

    def __init__(self, model_path: str = "results/lstm_predictor.pt",
                 threshold: float = MIGRATE_THRESHOLD):
        self.threshold = threshold
        self.model = LSTMLoadPredictor()
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self._windows: dict[int, deque] = {}

    def _get_feats(self, sim, sv: int) -> list[float]:
        node = sim.topo.get_node(sv)
        n_sess = len(sim.server_sessions[sv])
        return [
            float(node.load),
            float(node.hazard_rate),
            float(min(n_sess / 200.0, 1.0)),   # session ratio proxy
        ]

    def select_action(self, obs: dict, sim) -> dict:
        n = sim.n
        decisions = {}

        # Update windows
        for sv in range(n):
            if sv not in self._windows:
                self._windows[sv] = deque(maxlen=WINDOW_SIZE)
                # pad with current feature repeated
                for _ in range(WINDOW_SIZE):
                    self._windows[sv].append(self._get_feats(sim, sv))
            self._windows[sv].append(self._get_feats(sim, sv))

        # Build batch tensor
        seqs = []
        for sv in range(n):
            seq = list(self._windows[sv])
            if len(seq) < WINDOW_SIZE:
                # pad left
                pad = [seq[0]] * (WINDOW_SIZE - len(seq))
                seq = pad + seq
            seqs.append(seq[-WINDOW_SIZE:])

        x = torch.tensor(seqs, dtype=torch.float32)   # (n, window, feat)
        with torch.no_grad():
            pred_loads = self.model(x).numpy()         # (n,)

        for sv in range(n):
            if pred_loads[sv] < self.threshold:
                continue
            node = sim.topo.get_node(sv)
            if not node.is_alive:
                continue
            neighbors = sim.topo.neighbors(sv)
            live = [nb for nb in neighbors
                    if sim.topo.get_node(nb).is_alive
                    and sim.topo.get_node(nb).load < 0.7]
            if not live:
                continue
            target = min(live, key=lambda nb: sim.topo.get_node(nb).load)
            for sid in sorted(sim.server_sessions[sv])[:SESSIONS_PER_SERVER]:
                if sid in sim.sessions and sim.sessions[sid].status == SessionStatus.ACTIVE:
                    decisions[sid] = target

        return decisions

    def reset(self):
        """Call between episodes to clear history."""
        self._windows.clear()


# ---------------------------------------------------------------------------
# Entry point: train the predictor
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_lstm(save_path="results/lstm_predictor.pt", n_episodes=100, epochs=20)
    print("Done. Use LSTMPolicy(model_path='results/lstm_predictor.pt') in experiments.")
