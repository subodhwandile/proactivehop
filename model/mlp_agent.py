"""
mlp_agent.py — Ablation baseline: MLP policy with NO graph structure.

Receives flattened node features (N × 7) without any graph convolution.
Has the same parameter budget as GNNActorCritic.

Purpose: prove that topology-aware GNN encoding is what drives
catch-rate improvement over the flat MLP, not just the RL framework.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli


class MLPActorCritic(nn.Module):
    """
    Flat MLP — sees all node features concatenated, NO graph structure.
    Same hidden width (64) as GNNActorCritic for fair comparison.
    """

    def __init__(self, n_servers: int, node_feat_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.n_servers = n_servers
        input_dim = n_servers * node_feat_dim   # flat: all nodes concatenated

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: per-server logit
        self.actor_head = nn.Linear(hidden_dim, n_servers)

        # Critic head
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_flat: torch.Tensor):
        h = self.backbone(x_flat)
        action_logits = self.actor_head(h)   # (N,)
        value = self.critic_head(h).squeeze(-1)
        return action_logits, value

    def get_action_and_value(self, x_flat, action=None):
        logits, value = self.forward(x_flat)
        dist = Bernoulli(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        entropy = dist.entropy().sum()
        return action, log_prob, entropy, value


class MLPAgent:
    """
    PPO agent using flat MLP — ablation baseline for GNN comparison.
    Identical PPO hyperparameters as PPOAgent.
    """

    def __init__(
        self,
        n_servers: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        device: str = "cpu",
    ):
        self.n_servers = n_servers
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.device = torch.device(device)

        self.model = MLPActorCritic(n_servers=n_servers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self._buffer_clear()

    def _buffer_clear(self):
        self.buf_x = []
        self.buf_actions = []
        self.buf_log_probs = []
        self.buf_values = []
        self.buf_rewards = []
        self.buf_dones = []

    def select_action(self, obs: dict, greedy: bool = False) -> np.ndarray:
        x = torch.tensor(obs["node_features"].flatten(),
                         dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if greedy:
                logits, _ = self.model(x)
                action = (Bernoulli(logits=logits).probs >= 0.5).long()
                return action.cpu().numpy().astype(np.int32)
            action, log_prob, entropy, value = self.model.get_action_and_value(x)
        self.buf_x.append(x)
        self.buf_actions.append(action)
        self.buf_log_probs.append(log_prob)
        self.buf_values.append(value)
        return action.cpu().numpy().astype(np.int32)

    def store_reward(self, reward: float, done: bool):
        self.buf_rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.buf_dones.append(torch.tensor(float(done), dtype=torch.float32))

    def update(self, last_obs: dict) -> dict:
        if len(self.buf_rewards) < 2:
            return {}

        last_x = torch.tensor(last_obs["node_features"].flatten(),
                               dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, last_value = self.model(last_x)

        T = len(self.buf_rewards)
        advantages = [None] * T
        returns = [None] * T
        gae = 0.0

        for t in reversed(range(T)):
            next_val = last_value.item() if t == T - 1 else self.buf_values[t+1].item()
            next_done = self.buf_dones[t].item()
            delta = (self.buf_rewards[t].item()
                     + self.gamma * next_val * (1 - next_done)
                     - self.buf_values[t].item())
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = torch.tensor(gae, dtype=torch.float32)
            returns[t] = advantages[t] + self.buf_values[t]

        total_pg = total_vf = total_ent = 0.0
        for _ in range(self.n_epochs):
            indices = np.random.permutation(T)
            for i in indices:
                _, new_lp, ent, new_val = self.model.get_action_and_value(
                    self.buf_x[i], self.buf_actions[i])
                ratio = torch.exp(new_lp - self.buf_log_probs[i].detach())
                adv = advantages[i].to(self.device)
                ret = returns[i].to(self.device)
                pg = torch.max(-adv * ratio,
                               -adv * torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps))
                vf = 0.5 * (new_val - ret) ** 2
                loss = pg + self.value_coef * vf - self.entropy_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_pg += pg.item()
                total_vf += vf.item()
                total_ent += ent.item()

        self._buffer_clear()
        return {"pg_loss": total_pg/(T*self.n_epochs),
                "vf_loss": total_vf/(T*self.n_epochs),
                "entropy": total_ent/(T*self.n_epochs)}

    def save(self, path: str):
        torch.save({"model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
