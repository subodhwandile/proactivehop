"""
ablation.py — Ablation study: GNN encoder vs flat MLP encoder.

Trains both agents for the same number of episodes on the same seeds.
Produces a head-to-head comparison proving topology-aware GNN
encoding drives the catch-rate improvement, not just the RL framework.

Key claim: GNN sees graph structure (which servers are neighbors,
link latencies, load propagation paths). MLP sees the same numbers
but as an unstructured vector — cannot reason about topology.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
from pathlib import Path

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import SessionStatus
from model.ppo_agent import PPOAgent
from model.mlp_agent import MLPAgent

ABLATION_CFG = SimConfig(
    topology="fat_tree",
    n_servers=16,
    seed=42,
    session_arrival_rate=10.0,
    session_duration_mean=60.0,
    base_load=0.55,
    spike_probability=0.08,
    spike_magnitude=0.35,
    enable_failures=True,
    max_steps=100,
)

N_EPISODES = 100
CHECKPOINT_DIR = "results/ablation"
SEEDS = [42, 43, 44]


def run_episode(agent, cfg_seed: int, steps: int = 100) -> tuple[float, dict]:
    cfg = SimConfig(
        topology=ABLATION_CFG.topology,
        n_servers=ABLATION_CFG.n_servers,
        seed=cfg_seed,
        session_arrival_rate=ABLATION_CFG.session_arrival_rate,
        session_duration_mean=ABLATION_CFG.session_duration_mean,
        base_load=ABLATION_CFG.base_load,
        spike_probability=ABLATION_CFG.spike_probability,
        spike_magnitude=ABLATION_CFG.spike_magnitude,
        enable_failures=ABLATION_CFG.enable_failures,
        max_steps=steps,
    )
    sim = ClusterSimulator(cfg)
    obs = sim.reset()
    ep_reward = 0.0

    for _ in range(steps):
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
        ep_reward += reward
        if done:
            break

    loss = agent.update(obs)
    summary = sim.summary()
    return ep_reward, summary


def train_agent(agent, agent_name: str, n_episodes: int, log_path: str):
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    log = []
    best_reward = -np.inf

    print(f"\nTraining {agent_name} for {n_episodes} episodes...")
    for ep in range(n_episodes):
        seed = SEEDS[ep % len(SEEDS)]
        reward, summary = run_episode(agent, seed)
        entry = {
            "episode": ep,
            "agent": agent_name,
            "reward": reward,
            "catch_rate": summary.get("catch_rate", 0),
            "p95_ms": summary.get("p95_reconnect_ms", 0),
            "fp_rate": summary.get("false_positive_rate", 0),
        }
        log.append(entry)

        if ep % 10 == 0:
            print(f"  Ep {ep:3d} | reward={reward:8.0f} | "
                  f"catch={summary.get('catch_rate',0):.2%} | "
                  f"p95={summary.get('p95_reconnect_ms',0):.0f}ms")

        if reward > best_reward:
            best_reward = reward
            agent.save(f"{CHECKPOINT_DIR}/best_{agent_name}.pt")

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Best reward: {best_reward:.0f} | Log: {log_path}")
    return log


def evaluate_agent(agent, n_eval_episodes: int = 20) -> dict:
    """Evaluate trained agent on held-out seeds."""
    all_cr, all_p95, all_fp = [], [], []
    for ep in range(n_eval_episodes):
        seed = 100 + ep  # held-out seeds
        _, summary = run_episode(agent, seed)
        all_cr.append(summary.get("catch_rate", 0))
        all_p95.append(summary.get("p95_reconnect_ms", 0))
        all_fp.append(summary.get("false_positive_rate", 0))
    return {
        "catch_rate_mean": float(np.mean(all_cr)),
        "catch_rate_std": float(np.std(all_cr)),
        "p95_mean": float(np.mean(all_p95)),
        "p95_std": float(np.std(all_p95)),
        "fp_rate_mean": float(np.mean(all_fp)),
    }


def run_ablation():
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # --- GNN Agent ---
    gnn_agent = PPOAgent(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4)
    gnn_log = train_agent(gnn_agent, "GNN", N_EPISODES,
                          f"{CHECKPOINT_DIR}/gnn_training.json")

    # --- MLP Agent (ablation) ---
    mlp_agent = MLPAgent(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4)
    mlp_log = train_agent(mlp_agent, "MLP", N_EPISODES,
                          f"{CHECKPOINT_DIR}/mlp_training.json")

    # --- Evaluation on held-out seeds ---
    print("\nEvaluating on held-out seeds (seed 100–119)...")

    gnn_agent.load(f"{CHECKPOINT_DIR}/best_GNN.pt")
    gnn_agent.model.eval()
    gnn_eval = evaluate_agent(gnn_agent)

    mlp_agent.load(f"{CHECKPOINT_DIR}/best_MLP.pt")
    mlp_agent.model.eval()
    mlp_eval = evaluate_agent(mlp_agent)

    # --- Print comparison table ---
    print("\n" + "=" * 70)
    print("ABLATION STUDY: GNN Encoder vs Flat MLP Encoder")
    print("=" * 70)
    print(f"{'Metric':<30} {'GNN (ours)':>18} {'MLP (no graph)':>18}")
    print("-" * 70)
    print(f"{'Catch Rate (mean±std)':<30} "
          f"{gnn_eval['catch_rate_mean']:>16.1%}±{gnn_eval['catch_rate_std']:.1%} "
          f"{mlp_eval['catch_rate_mean']:>16.1%}±{mlp_eval['catch_rate_std']:.1%}")
    print(f"{'P95 Latency ms (mean±std)':<30} "
          f"{gnn_eval['p95_mean']:>15.0f}±{gnn_eval['p95_std']:.0f}ms "
          f"{mlp_eval['p95_mean']:>15.0f}±{mlp_eval['p95_std']:.0f}ms")
    print(f"{'False Positive Rate':<30} "
          f"{gnn_eval['fp_rate_mean']:>18.1%} "
          f"{mlp_eval['fp_rate_mean']:>18.1%}")

    improvement = (gnn_eval['catch_rate_mean'] /
                   max(mlp_eval['catch_rate_mean'], 0.001))
    print(f"\nGNN catch-rate improvement over MLP: {improvement:.1f}x")
    print("=" * 70)

    result = {
        "gnn": gnn_eval,
        "mlp": mlp_eval,
        "improvement_factor": improvement,
    }
    with open(f"{CHECKPOINT_DIR}/ablation_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {CHECKPOINT_DIR}/ablation_results.json")
    return result


if __name__ == "__main__":
    run_ablation()
