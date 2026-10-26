"""
sensitivity.py — Hyperparameter sensitivity analysis.

Sweeps entropy_coef and learning_rate independently (one-at-a-time).
Each configuration trains for N_EPISODES and evaluates on 10 held-out seeds.
Produces results/sensitivity.json and results/plots/fig_sensitivity.{pdf,png}.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import SessionStatus
from model.ppo_agent import PPOAgent

N_TRAIN   = 150   # episodes per config (fast sweep)
N_EVAL    = 10    # held-out evaluation episodes
TRAIN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SEEDS  = list(range(100, 110))

BASE_CFG = dict(
    topology="fat_tree", n_servers=16, session_arrival_rate=10.0,
    session_duration_mean=60.0, base_load=0.55, spike_probability=0.08,
    spike_magnitude=0.35, enable_failures=True, max_steps=100,
)

# Default hyperparameters
DEFAULT_LR      = 3e-4
DEFAULT_ENTROPY = 0.05
DEFAULT_CBW     = 5e-5

# Sweep grids (one-at-a-time, others fixed to default)
ENTROPY_SWEEP = [0.01, 0.02, 0.05, 0.10, 0.20]
LR_SWEEP      = [1e-4, 3e-4, 1e-3, 3e-3]
CBW_SWEEP     = [1e-5, 5e-5, 2e-4, 5e-4]


def run_episode(agent, seed: int, cbw: float = DEFAULT_CBW) -> dict:
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


def eval_episode(agent, seed: int) -> dict:
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


def train_and_eval(lr: float, entropy: float, label: str) -> dict:
    agent = PPOAgent(n_servers=16, lr=lr, entropy_coef=entropy, n_epochs=4)
    for ep in range(N_TRAIN):
        seed = TRAIN_SEEDS[ep % len(TRAIN_SEEDS)]
        run_episode(agent, seed)

    agent.model.eval()
    cr_list, fp_list = [], []
    for seed in EVAL_SEEDS:
        s = eval_episode(agent, seed)
        cr_list.append(s.get("catch_rate", 0))
        fp_list.append(s.get("false_positive_rate", 0))

    result = {
        "label": label,
        "lr": lr,
        "entropy": entropy,
        "catch_rate_mean": float(np.mean(cr_list)),
        "catch_rate_std":  float(np.std(cr_list)),
        "fp_rate_mean":    float(np.mean(fp_list)),
    }
    print(f"  {label:<30}  catch={result['catch_rate_mean']:.3f}±{result['catch_rate_std']:.3f}"
          f"  fp={result['fp_rate_mean']:.3f}")
    return result


def run_sensitivity():
    Path("results").mkdir(exist_ok=True)
    results = {"entropy_sweep": [], "lr_sweep": []}

    print("\n=== Entropy coefficient sweep (lr fixed at 3e-4) ===")
    for e in ENTROPY_SWEEP:
        r = train_and_eval(DEFAULT_LR, e, f"entropy={e}")
        results["entropy_sweep"].append(r)

    print("\n=== Learning rate sweep (entropy fixed at 0.05) ===")
    for lr in LR_SWEEP:
        r = train_and_eval(lr, DEFAULT_ENTROPY, f"lr={lr:.0e}")
        results["lr_sweep"].append(r)

    with open("results/sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/sensitivity.json")

    _plot(results)
    return results


def _plot(results: dict):
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, xlabel in [
        (axes[0], "entropy_sweep", "Entropy coefficient"),
        (axes[1], "lr_sweep",      "Learning rate"),
    ]:
        rows = results[key]
        x_vals = [r["entropy"] if "entropy" in key else r["lr"] for r in rows]
        means  = [r["catch_rate_mean"] * 100 for r in rows]
        stds   = [r["catch_rate_std"]  * 100 for r in rows]
        default_idx = 2 if "entropy" in key else 1  # default is 3rd / 2nd entry

        ax.errorbar(range(len(rows)), means, yerr=stds,
                    fmt='o-', color='steelblue', linewidth=2,
                    capsize=4, markersize=6)
        ax.axvline(default_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(default_idx + 0.1, max(means) * 0.95, "default", color='red', fontsize=8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([str(v) for v in x_vals], rotation=20, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Catch Rate (%)", fontsize=10)
        ax.set_title(f"Sensitivity: {xlabel}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(means) * 1.5 + 0.5)

    plt.tight_layout()
    plt.savefig("results/plots/fig_sensitivity.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("results/plots/fig_sensitivity.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved results/plots/fig_sensitivity.{pdf,png}")


if __name__ == "__main__":
    run_sensitivity()
