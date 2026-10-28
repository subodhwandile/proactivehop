"""
ppo_agent.py — PPO agent with GNN backbone for session migration policy.

Architecture:
  Observation → GNN Encoder → per-node embeddings + graph embedding
  → MigrationScorer → per-server action logits
  → PPO (clipped surrogate objective)

The GNN encoder provides the topology-aware representation.
The PPO update uses the standard clipped objective with entropy bonus.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli

from model.gnn_encoder import GNNEncoder


class GNNActorCritic(nn.Module):
    """
    Combined actor-critic network with GNN backbone.

    Actor: outputs per-server migration probability
    Critic: outputs state value estimate
    """

    def __init__(
        self,
        n_servers: int,
        node_feat_dim: int = 7,
        edge_feat_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_servers = n_servers
        self.hidden_dim = hidden_dim

        # Shared GNN encoder
        self.gnn = GNNEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
        )

        # Actor head: per-server migration probability
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # node_embed + graph_embed
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Critic head: scalar value
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        node_feat: torch.Tensor,    # (N, 7)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr: torch.Tensor,    # (E, 4)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_logits: (N,) — per-server migration probability logit
            value: scalar
        """
        node_embed, graph_embed = self.gnn(node_feat, edge_index, edge_attr)
        # (N, hidden_dim), (hidden_dim,)

        # Actor: concatenate per-node embed with global graph embed
        N = node_embed.shape[0]
        graph_exp = graph_embed.unsqueeze(0).expand(N, -1)  # (N, hidden_dim)
        actor_input = torch.cat([node_embed, graph_exp], dim=1)  # (N, 2*hidden_dim)
        action_logits = self.actor_head(actor_input).squeeze(-1)  # (N,)

        # Critic uses global embedding
        value = self.critic_head(graph_embed).squeeze(-1)  # scalar

        return action_logits, value

    def get_action_and_value(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        action: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and compute log-probs, entropy, value.
        """
        logits, value = self.forward(node_feat, edge_index, edge_attr)
        dist = Bernoulli(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum()   # sum over servers
        entropy = dist.entropy().sum()

        return action, log_prob, entropy, value


class PPOAgent:
    """
    PPO training loop for the GNN migration policy.

    Hyperparameters follow CleanRL PPO defaults, tuned for discrete
    multi-binary action space.
    """

    def __init__(
        self,
        n_servers: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.model = GNNActorCritic(n_servers=n_servers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Rollout buffer
        self._buffer_clear()

    def _buffer_clear(self):
        self.buf_node_feats = []
        self.buf_edge_index = []
        self.buf_edge_attr = []
        self.buf_actions = []
        self.buf_log_probs = []
        self.buf_values = []
        self.buf_rewards = []
        self.buf_dones = []

    def select_action(self, obs: dict, greedy: bool = False) -> np.ndarray:
        """
        Given raw graph observation dict, run GNN + sample action.
        If greedy=True, use argmax (deterministic) instead of sampling.
        Returns numpy action array (N,).
        """
        node_feat = torch.tensor(obs["node_features"], dtype=torch.float32,
                                 device=self.device)
        ei = obs["edge_index"]
        ea = obs["edge_features"]

        if ei.shape[1] == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
        else:
            edge_index = torch.tensor(ei, dtype=torch.long, device=self.device)
            edge_attr = torch.tensor(ea, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if greedy:
                logits, _ = self.model(node_feat, edge_index, edge_attr)
                from torch.distributions import Bernoulli as _B
                action = (_B(logits=logits).probs >= 0.5).long()
                return action.cpu().numpy()
            action, log_prob, entropy, value = self.model.get_action_and_value(
                node_feat, edge_index, edge_attr
            )

        # Store for training
        self.buf_node_feats.append(node_feat)
        self.buf_edge_index.append(edge_index)
        self.buf_edge_attr.append(edge_attr)
        self.buf_actions.append(action)
        self.buf_log_probs.append(log_prob)
        self.buf_values.append(value)

        return action.cpu().numpy().astype(np.int32)

    def store_reward(self, reward: float, done: bool):
        self.buf_rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.buf_dones.append(torch.tensor(float(done), dtype=torch.float32))

    def update(self, last_obs: dict) -> dict:
        """
        Run PPO update using collected rollout.
        Returns loss metrics.
        """
        if len(self.buf_rewards) < 2:
            return {}

        # Compute GAE returns
        returns, advantages = self._compute_gae(last_obs)

        # Flatten buffer
        T = len(self.buf_rewards)

        # PPO epochs
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent = 0.0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]

                for i in batch_idx:
                    nf = self.buf_node_feats[i]
                    ei = self.buf_edge_index[i]
                    ea = self.buf_edge_attr[i]
                    old_action = self.buf_actions[i]
                    old_log_prob = self.buf_log_probs[i]
                    ret = returns[i].to(self.device)
                    adv = advantages[i].to(self.device)

                    _, new_log_prob, entropy, new_value = \
                        self.model.get_action_and_value(nf, ei, ea, old_action)

                    ratio = torch.exp(new_log_prob - old_log_prob.detach())
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    pg_loss = torch.max(pg_loss1, pg_loss2)

                    vf_loss = 0.5 * (new_value - ret) ** 2

                    loss = pg_loss + self.value_coef * vf_loss - self.entropy_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_pg_loss += pg_loss.item()
                    total_vf_loss += vf_loss.item()
                    total_ent += entropy.item()

        self._buffer_clear()
        return {
            "pg_loss": total_pg_loss / (T * self.n_epochs),
            "vf_loss": total_vf_loss / (T * self.n_epochs),
            "entropy": total_ent / (T * self.n_epochs),
        }

    def _compute_gae(self, last_obs: dict) -> tuple[list, list]:
        T = len(self.buf_rewards)

        # Bootstrap value from last state
        nf = torch.tensor(last_obs["node_features"], dtype=torch.float32,
                          device=self.device)
        ei_raw = last_obs["edge_index"]
        ea_raw = last_obs["edge_features"]
        if ei_raw.shape[1] == 0:
            ei = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            ea = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
        else:
            ei = torch.tensor(ei_raw, dtype=torch.long, device=self.device)
            ea = torch.tensor(ea_raw, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, last_value = self.model(nf, ei, ea)

        advantages = [None] * T
        returns = [None] * T
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_value.item()
                next_done = self.buf_dones[t].item()
            else:
                next_val = self.buf_values[t + 1].item()
                next_done = self.buf_dones[t].item()

            delta = (self.buf_rewards[t].item()
                     + self.gamma * next_val * (1 - next_done)
                     - self.buf_values[t].item())
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = torch.tensor(gae, dtype=torch.float32)
            returns[t] = advantages[t] + self.buf_values[t]

        return returns, advantages

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
