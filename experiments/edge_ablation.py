"""
edge_ablation.py — GNN with edges removed vs full GNN.

Tests whether topology (edge_index) is necessary for the GNN's advantage.
An edge-removed GNN sees the same node features but no inter-server links,
degrading to an independent per-node MLP (though with more parameters than
the MLPAgent, making this a conservative test).

Two conditions:
  GNN-Full   : standard 2-layer GraphSAGE with real edges
  GNN-NoEdge : same architecture but edge_index forced to empty at inference

Both models use the same 5 init seeds from ablation_multiseed.
GNN-Full checkpoints are reused from results/ablation_multiseed/.
GNN-NoEdge is trained from scratch (5 seeds × 500 episodes).
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


TRAIN_INIT_SEEDS = [0, 1, 2, 3, 4]
EVAL_SEEDS       = list(range(100, 120))
N_TRAIN_EPISODES = 500
CHECKPOINT_DIR   = "results/edge_ablation"

BASE_CFG = dict(
    topology="fat_tree", n_servers=16, session_arrival_rate=10.0,
    session_duration_mean=60.0, base_load=0.55, spike_probability=0.08,
    spike_magnitude=0.35, enable_failures=True, max_steps=100,
)
TRAIN_SEEDS = [42, 43, 44, 45, 46]


def make_empty_edge_obs(obs: dict) -> dict:
    """Return a copy of obs with edge_index zeroed out (no topology)."""
    out = dict(obs)
    out["edge_index"]    = np.zeros((2, 0), dtype=np.int64)
    out["edge_features"] = np.zeros((0, 4),  dtype=np.float32)
    return out


def run_episode(agent, seed: int, strip_edges: bool = False,
                training: bool = True) -> dict:
    cfg = SimConfig(**BASE_CFG, seed=seed)
    sim = ClusterSimulator(cfg)
    obs = sim.reset()
    if strip_edges:
        obs = make_empty_edge_obs(obs)

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
        obs, reward, done, _ = sim.step(decisions)
        if strip_edges:
            obs = make_empty_edge_obs(obs)
        if training:
            agent.store_reward(reward, done)
        if done:
            break

    if training:
        agent.update(obs)
    else:
        agent._buffer_clear()
    return sim.summary()


def train_noedge(init_seed: int) -> str:
    """Train GNN-NoEdge from scratch with empty edges."""
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    agent = PPOAgent(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4)
    save_path = f"{CHECKPOINT_DIR}/GNN_NoEdge_init{init_seed}.pt"
    best_cr = -np.inf
    print(f"  Training GNN-NoEdge (init_seed={init_seed}, {N_TRAIN_EPISODES} eps)...")
    for ep in range(N_TRAIN_EPISODES):
        seed = TRAIN_SEEDS[ep % len(TRAIN_SEEDS)]
        s = run_episode(agent, seed, strip_edges=True, training=True)
        cr = s.get("catch_rate", 0)
        if cr > best_cr:
            best_cr = cr
            agent.save(save_path)
        if (ep + 1) % 100 == 0:
            print(f"    ep {ep+1}/{N_TRAIN_EPISODES}  best_cr={best_cr:.3f}")
    agent.load(save_path)
    agent.model.eval()
    return save_path


def evaluate(agent, strip_edges: bool) -> dict:
    cr_list, fp_list = [], []
    for seed in EVAL_SEEDS:
        s = run_episode(agent, seed, strip_edges=strip_edges, training=False)
        cr_list.append(s.get("catch_rate", 0))
        fp_list.append(s.get("false_positive_rate", 0))
    return {
        "catch_rates":     cr_list,
        "catch_rate_mean": float(np.mean(cr_list)),
        "catch_rate_std":  float(np.std(cr_list)),
        "fp_rate_mean":    float(np.mean(fp_list)),
    }


def run_edge_ablation():
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    full_means, noedge_means = [], []

    for i, init_seed in enumerate(TRAIN_INIT_SEEDS):
        print(f"\n--- Init seed {init_seed} ({i+1}/{len(TRAIN_INIT_SEEDS)}) ---")

        # GNN-Full: load from ablation_multiseed
        full_path = f"results/ablation_multiseed/GNN_init{init_seed}.pt"
        full_agent = PPOAgent(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4)
        if Path(full_path).exists():
            full_agent.load(full_path)
            full_agent.model.eval()
        else:
            print(f"  WARNING: {full_path} not found, using untrained")
        full_eval = evaluate(full_agent, strip_edges=False)
        full_means.append(full_eval["catch_rate_mean"])
        print(f"  GNN-Full    catch_rate={full_eval['catch_rate_mean']:.3f}±{full_eval['catch_rate_std']:.3f}")

        # GNN-NoEdge: train from scratch
        noedge_path = train_noedge(init_seed)
        noedge_agent = PPOAgent(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4)
        noedge_agent.load(noedge_path)
        noedge_agent.model.eval()
        noedge_eval = evaluate(noedge_agent, strip_edges=True)
        noedge_means.append(noedge_eval["catch_rate_mean"])
        print(f"  GNN-NoEdge  catch_rate={noedge_eval['catch_rate_mean']:.3f}±{noedge_eval['catch_rate_std']:.3f}")

    full_arr    = np.array(full_means)
    noedge_arr  = np.array(noedge_means)
    t_stat, p_value = stats.ttest_ind(full_arr, noedge_arr, equal_var=False)

    print("\n" + "=" * 60)
    print("EDGE-REMOVED GNN ABLATION (5 seeds × 20 eval seeds)")
    print("=" * 60)
    print(f"GNN-Full   : {full_arr.mean()*100:.2f}% ± {full_arr.std()*100:.2f}%")
    print(f"GNN-NoEdge : {noedge_arr.mean()*100:.2f}% ± {noedge_arr.std()*100:.2f}%")
    print(f"Welch t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print("=" * 60)

    result = {
        "gnn_full": {
            "per_seed_means": list(full_means),
            "mean": float(full_arr.mean()),
            "std":  float(full_arr.std()),
        },
        "gnn_noedge": {
            "per_seed_means": list(noedge_means),
            "mean": float(noedge_arr.mean()),
            "std":  float(noedge_arr.std()),
        },
        "welch_t": float(t_stat),
        "welch_p": float(p_value),
    }
    out = f"{CHECKPOINT_DIR}/edge_ablation_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out}")
    return result


if __name__ == "__main__":
    run_edge_ablation()
