"""
edge_ablation.py — Topology necessity test: GNN with edges removed.

Two evaluation conditions for each of the 5 trained GNN_initX checkpoints:
  GNN-Full   : standard inference with real edge_index + edge_features
  GNN-NoEdge : same checkpoint, edge_index forced empty at inference time

If the GNN is truly exploiting topology (not just ignoring edges), stripping
them should degrade catch rate.  If performance is unchanged, the GNN learned
a purely per-node policy despite having access to topology.

This is a zero-shot topology-removal test — no retraining required,
runs in <5 minutes for all 5 seeds × 20 eval episodes.

Additionally trains 5 GNN-NoEdge models from scratch (200 episodes each)
to check whether an edge-free GNN can learn a competitive policy.
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
N_TRAIN_EPISODES = 200       # reduced: edge-free GNN converges fast (no topo)
CHECKPOINT_DIR   = "results/edge_ablation"

BASE_CFG = dict(
    topology="fat_tree", n_servers=16, session_arrival_rate=10.0,
    session_duration_mean=60.0, base_load=0.55, spike_probability=0.08,
    spike_magnitude=0.35, enable_failures=True, max_steps=100,
)
TRAIN_SEEDS = [42, 43, 44, 45, 46]


def make_empty_edge_obs(obs: dict) -> dict:
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


def train_noedge(init_seed: int) -> str:
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    agent = PPOAgent(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4)
    save_path = f"{CHECKPOINT_DIR}/GNN_NoEdge_init{init_seed}.pt"
    best_cr = -np.inf
    for ep in range(N_TRAIN_EPISODES):
        seed = TRAIN_SEEDS[ep % len(TRAIN_SEEDS)]
        s = run_episode(agent, seed, strip_edges=True, training=True)
        cr = s.get("catch_rate", 0)
        if cr > best_cr:
            best_cr = cr
            agent.save(save_path)
        if (ep + 1) % 50 == 0:
            print(f"    ep {ep+1}/{N_TRAIN_EPISODES}  best_cr={best_cr:.3f}")
    agent.load(save_path)
    agent.model.eval()
    return save_path


def run_edge_ablation():
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    full_means, noedge_eval_means, noedge_train_means = [], [], []

    for i, init_seed in enumerate(TRAIN_INIT_SEEDS):
        print(f"\n--- Init seed {init_seed} ({i+1}/{len(TRAIN_INIT_SEEDS)}) ---")

        # GNN-Full: load from ablation_multiseed, evaluate with full edges
        full_path = f"results/ablation_multiseed/GNN_init{init_seed}.pt"
        agent = PPOAgent(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4)
        if Path(full_path).exists():
            agent.load(full_path)
        agent.model.eval()

        full_eval = evaluate(agent, strip_edges=False)
        full_means.append(full_eval["catch_rate_mean"])
        print(f"  GNN-Full       catch={full_eval['catch_rate_mean']:.3f}±{full_eval['catch_rate_std']:.3f}")

        # GNN-NoEdge-eval: same checkpoint, edges stripped at inference
        noedge_eval = evaluate(agent, strip_edges=True)
        noedge_eval_means.append(noedge_eval["catch_rate_mean"])
        print(f"  GNN-NoEdge-eval catch={noedge_eval['catch_rate_mean']:.3f}±{noedge_eval['catch_rate_std']:.3f}")

        # GNN-NoEdge-train: retrained from scratch without edges (200 eps)
        print(f"  Training GNN-NoEdge from scratch ({N_TRAIN_EPISODES} eps)...")
        noedge_path = train_noedge(init_seed)
        noedge_train_agent = PPOAgent(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4)
        noedge_train_agent.load(noedge_path)
        noedge_train_agent.model.eval()
        noedge_train_eval = evaluate(noedge_train_agent, strip_edges=True)
        noedge_train_means.append(noedge_train_eval["catch_rate_mean"])
        print(f"  GNN-NoEdge-train catch={noedge_train_eval['catch_rate_mean']:.3f}±{noedge_train_eval['catch_rate_std']:.3f}")

    full_arr         = np.array(full_means)
    noedge_eval_arr  = np.array(noedge_eval_means)
    noedge_train_arr = np.array(noedge_train_means)

    t_eval,  p_eval  = stats.ttest_ind(full_arr, noedge_eval_arr,  equal_var=False)
    t_train, p_train = stats.ttest_ind(full_arr, noedge_train_arr, equal_var=False)

    print("\n" + "=" * 65)
    print("EDGE-REMOVED GNN ABLATION (5 seeds × 20 eval seeds)")
    print("=" * 65)
    print(f"GNN-Full         : {full_arr.mean()*100:.2f}% ± {full_arr.std()*100:.2f}%")
    print(f"GNN-NoEdge-eval  : {noedge_eval_arr.mean()*100:.2f}% ± {noedge_eval_arr.std()*100:.2f}%  (p={p_eval:.4f})")
    print(f"GNN-NoEdge-train : {noedge_train_arr.mean()*100:.2f}% ± {noedge_train_arr.std()*100:.2f}%  (p={p_train:.4f})")
    print("=" * 65)

    result = {
        "gnn_full":         {"per_seed_means": list(full_means),
                             "mean": float(full_arr.mean()),
                             "std":  float(full_arr.std())},
        "gnn_noedge_eval":  {"per_seed_means": list(noedge_eval_means),
                             "mean": float(noedge_eval_arr.mean()),
                             "std":  float(noedge_eval_arr.std()),
                             "welch_p": float(p_eval)},
        "gnn_noedge_train": {"per_seed_means": list(noedge_train_means),
                             "mean": float(noedge_train_arr.mean()),
                             "std":  float(noedge_train_arr.std()),
                             "welch_p": float(p_train)},
    }
    out = f"{CHECKPOINT_DIR}/edge_ablation_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out}")
    return result


if __name__ == "__main__":
    run_edge_ablation()
