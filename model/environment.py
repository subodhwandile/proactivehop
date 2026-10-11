"""
environment.py — Gymnasium-compatible RL environment wrapping ClusterSimulator.

Each "agent" step corresponds to one simulation timestep.
The agent observes the full cluster graph state and decides which sessions
to speculatively pre-migrate.

Action space:
  For tractability, we operate at the server level:
  For each server with load > threshold, decide: do nothing OR pre-migrate
  the top-K most at-risk sessions to the best available neighbor.

  Simplified action: vector of N binary decisions (one per server).
  Action[i] = 1 → trigger speculative pre-migration for sessions on server i.

Observation space:
  - node_features: (N, 7)
  - flattened for MLP policy, or passed as graph for GNN policy
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import SessionStatus


class ServerMigrationEnv(gym.Env):
    """
    RL environment for predictive session pre-migration.

    Observation: flattened node feature matrix (N * 7,)
    Action: binary vector (N,) — for each server, trigger pre-migration of
            its top-risk sessions to the lowest-load neighbor.

    The GNN policy wraps this env and uses the raw graph observation
    (node_features, edge_index, edge_features) directly.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: SimConfig = None, use_gnn_obs: bool = True):
        super().__init__()
        self.cfg = config or SimConfig()
        self.sim = ClusterSimulator(self.cfg)
        self.n = self.sim.n
        self.use_gnn_obs = use_gnn_obs

        # Action space: binary per server
        self.action_space = spaces.MultiBinary(self.n)

        # Observation space: node features flattened
        obs_dim = self.n * 7
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._last_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.sim.reset()
        flat_obs = obs_dict["node_features"].flatten()
        self._last_obs = obs_dict
        return flat_obs, {}

    def step(self, action: np.ndarray):
        """
        action: (N,) binary array.
        For each server i where action[i]==1:
            find its top-K at-risk sessions
            decide best neighbor target
            submit speculative migration decisions
        """
        decisions = self._action_to_decisions(action)
        obs_dict, reward, done, info = self.sim.step(decisions)
        flat_obs = obs_dict["node_features"].flatten()
        self._last_obs = obs_dict

        truncated = False
        return flat_obs, reward, done, truncated, info

    def _action_to_decisions(self, action: np.ndarray) -> dict[str, int]:
        """
        Translate binary server-level action to per-session migration decisions.
        For each activated server: pick top-3 at-risk sessions → best neighbor.
        """
        decisions = {}
        for server_id in range(self.n):
            if action[server_id] == 0:
                continue

            node = self.sim.topo.get_node(server_id)
            if node.load < 0.5 or not node.is_alive:
                continue

            # Find best neighbor (lowest load, alive)
            neighbors = self.sim.topo.neighbors(server_id)
            live_neighbors = [
                nb for nb in neighbors
                if self.sim.topo.get_node(nb).is_alive
                and self.sim.topo.get_node(nb).load < 0.7
            ]
            if not live_neighbors:
                continue
            best_target = min(live_neighbors,
                              key=lambda nb: self.sim.topo.get_node(nb).load)

            # Pick top-3 at-risk sessions (highest session load ratio)
            sessions_on_server = [
                self.sim.sessions[sid]
                for sid in self.sim.server_sessions[server_id]
                if sid in self.sim.sessions
                and self.sim.sessions[sid].status == SessionStatus.ACTIVE
            ]
            # Sort by state size (larger = more expensive to migrate later)
            sessions_on_server.sort(
                key=lambda s: s.state.state_payload_kb, reverse=True
            )
            for s in sessions_on_server[:20]:   # migrate top-20 at-risk sessions
                decisions[s.session_id] = best_target

        return decisions

    def get_graph_obs(self) -> dict:
        """Returns raw graph observation for GNN-based policies."""
        return self._last_obs

    def render(self):
        pass

    def close(self):
        pass
