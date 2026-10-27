"""
ablation_multiseed.py — Ablation with 5 independent training seeds.

For each agent type (GNN, MLP):
  - Train 5 independent models (different torch.manual_seed for init)
  - Evaluate each frozen model on 20 held-out seeds (no gradient updates)
  - Report mean ± std across training seeds + Welch t-test p-value

This separates training variance from evaluation variance and gives a
statistically valid comparison of GNN vs MLP.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import torch
from pathlib import Path
from scipy import stats

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import SessionStatus
from model.ppo_agent import PPOAgent
from model.mlp_agent import MLPAgent

N_TRAIN_EPISODES = 500
N_EVAL_EPISODES  = 20
TRAIN_SEEDS      = [42, 43, 44, 45, 46]   # episode seeds during training
EVAL_SEEDS       = list(range(100, 120))   # held-out evaluation seeds
TRAIN_INIT_SEEDS = [0, 1, 2, 3, 4]        # torch.manual_seed for model init

BASE_CFG = dict(
    topology="fat_tree", n_servers=16, session_arrival_rate=10.0,
    session_duration_mean=60.0, base_load=0.55, spike_probability=0.08,
    spike_magnitude=0.35, enable_failures=True, max_steps=100,
)
CHECKPOINT_DIR = "results/ablation_multiseed"


def run_episode_train(agent, seed: int) -> dict:
    cfg = SimConfig(**BASE_CFG, seed=seed)
    sim = ClusterSimulator(cfg)
    obs = sim.reset()
    for _ in range(cfg.max_steps):
        action = agent.select_action(obs)
        decisions = {}
        for sv in range(sim.n):
            if action[sv] == 0:
                continue
            node = sim.topo.get_node(sv)
            if node.load < 0.5 or not node.is_alive:
                continue
            neighbors = sim.topo.neighbors(sv)
            live = [nb for nb in neighbors
                    if sim.topo.get_node(nb).is_alive
                    and sim.topo.get_node(nb).load < 0.7]
            if not live:
                continue
            target = min(live, key=lambda nb: sim.topo.get_node(nb).load)
            for sid in list(sim.server_sessions[sv])[:20]:
                if sid in sim.sessions and sim.sessions[sid].status == SessionStatus.ACTIVE:
                    decisions[sid] = target
        obs, reward, done, info = sim.step(decisions)
        agent.store_reward(reward, done)
        if done:
            break
    agent.update(obs)
    return sim.summary()


def run_episode_eval(agent, seed: int) -> dict:
    """Inference only — no gradient updates."""
    cfg = SimConfig(**BASE_CFG, seed=seed)
    sim = ClusterSimulator(cfg)
    obs = sim.reset()
    for _ in range(cfg.max_steps):
        action = agent.select_action(obs)
        decisions = {}
        for sv in range(sim.n):
            if action[sv] == 0:
                continue
            node = sim.topo.get_node(sv)
            if node.load < 0.5 or not node.is_alive:
                continue
            neighbors = sim.topo.neighbors(sv)
            live = [nb for nb in neighbors
                    if sim.topo.get_node(nb).is_alive
                    and sim.topo.get_node(nb).load < 0.7]
            if not live:
                continue
            target = min(live, key=lambda nb: sim.topo.get_node(nb).load)
            for sid in list(sim.server_sessions[sv])[:20]:
                if sid in sim.sessions and sim.sessions[sid].status == SessionStatus.ACTIVE:
                    decisions[sid] = target
        obs, reward, done, info = sim.step(decisions)
        if done:
            break
    agent._buffer_clear()
    return sim.summary()


def train_one_model(AgentClass, init_seed: int, agent_name: str,
                    n_episodes: int = N_TRAIN_EPISODES) -> object:
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    agent = AgentClass(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4)
    best_reward = -np.inf
    for ep in range(n_episodes):
        seed = TRAIN_SEEDS[ep % len(TRAIN_SEEDS)]
        summary = run_episode_train(agent, seed)
        reward_proxy = summary.get("catch_rate", 0)
        if reward_proxy > best_reward:
            best_reward = reward_proxy
            agent.save(f"{CHECKPOINT_DIR}/{agent_name}_init{init_seed}.pt")
    agent.load(f"{CHECKPOINT_DIR}/{agent_name}_init{init_seed}.pt")
    agent.model.eval()
    return agent


def evaluate_model(agent, n_eval: int = N_EVAL_EPISODES) -> dict:
    cr_list, fp_list = [], []
    for seed in EVAL_SEEDS[:n_eval]:
        s = run_episode_eval(agent, seed)
        cr_list.append(s.get("catch_rate", 0))
        fp_list.append(s.get("false_positive_rate", 0))
    return {
        "catch_rates": cr_list,
        "catch_rate_mean": float(np.mean(cr_list)),
        "catch_rate_std":  float(np.std(cr_list)),
        "fp_rate_mean":    float(np.mean(fp_list)),
    }


def run_multiseed_ablation():
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    gnn_per_seed_means = []
    mlp_per_seed_means = []
    gnn_results = []
    mlp_results = []

    for i, init_seed in enumerate(TRAIN_INIT_SEEDS):
        print(f"\n--- Init seed {init_seed} ({i+1}/{len(TRAIN_INIT_SEEDS)}) ---")

        print(f"  Training GNN (init_seed={init_seed})...")
        gnn = train_one_model(PPOAgent, init_seed, "GNN")
        gnn_eval = evaluate_model(gnn)
        gnn_results.append(gnn_eval)
        gnn_per_seed_means.append(gnn_eval["catch_rate_mean"])
        print(f"  GNN catch_rate={gnn_eval['catch_rate_mean']:.3f}±{gnn_eval['catch_rate_std']:.3f}")

        print(f"  Training MLP (init_seed={init_seed})...")
        mlp = train_one_model(MLPAgent, init_seed, "MLP")
        mlp_eval = evaluate_model(mlp)
        mlp_results.append(mlp_eval)
        mlp_per_seed_means.append(mlp_eval["catch_rate_mean"])
        print(f"  MLP catch_rate={mlp_eval['catch_rate_mean']:.3f}±{mlp_eval['catch_rate_std']:.3f}")

    # Aggregate across training seeds
    gnn_means = np.array(gnn_per_seed_means)
    mlp_means = np.array(mlp_per_seed_means)

    # Welch's t-test (unequal variance)
    t_stat, p_value = stats.ttest_ind(gnn_means, mlp_means, equal_var=False)
    # Mann-Whitney U (non-parametric, better for small N)
    u_stat, p_mw = stats.mannwhitneyu(gnn_means, mlp_means, alternative='greater')

    improvement = gnn_means.mean() / max(mlp_means.mean(), 1e-6)

    print("\n" + "=" * 70)
    print("MULTI-SEED ABLATION: GNN vs MLP (5 training seeds × 20 eval seeds)")
    print("=" * 70)
    print(f"{'Metric':<35} {'GNN':>15} {'MLP':>15}")
    print("-" * 70)
    print(f"{'Catch Rate (mean across train seeds)':<35} "
          f"{gnn_means.mean()*100:>13.2f}%  {mlp_means.mean()*100:>13.2f}%")
    print(f"{'Catch Rate std (across train seeds)':<35} "
          f"{gnn_means.std()*100:>13.2f}%  {mlp_means.std()*100:>13.2f}%")
    print(f"{'Individual seed means (GNN)':<35} "
          f"{[f'{v*100:.2f}' for v in gnn_means]}")
    print(f"{'Individual seed means (MLP)':<35} "
          f"{[f'{v*100:.2f}' for v in mlp_means]}")
    print(f"\nImprovement factor: {improvement:.2f}×")
    print(f"Welch t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")
    if p_value < 0.05:
        print("→ SIGNIFICANT (p < 0.05)")
    elif p_value < 0.10:
        print("→ MARGINAL (p < 0.10)")
    else:
        print("→ NOT SIGNIFICANT (p ≥ 0.10)")
    print("=" * 70)

    result = {
        "gnn": {
            "per_seed_means": list(gnn_means),
            "mean": float(gnn_means.mean()),
            "std":  float(gnn_means.std()),
        },
        "mlp": {
            "per_seed_means": list(mlp_means),
            "mean": float(mlp_means.mean()),
            "std":  float(mlp_means.std()),
        },
        "improvement_factor": float(improvement),
        "welch_t": float(t_stat),
        "welch_p": float(p_value),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(p_mw),
    }
    out = f"{CHECKPOINT_DIR}/ablation_multiseed_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out}")
    return result


if __name__ == "__main__":
    run_multiseed_ablation()
