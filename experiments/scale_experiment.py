"""
scale_experiment.py — Scalability test: 16 → 32 → 64 servers.

Tests two properties:
  1. Catch rate holds up as cluster grows (policy generalizes to unseen sizes)
  2. GNN inference time scales O(degree²) not O(N²) — the key architectural claim

Compares GNN policy (trained on 16 nodes) vs Threshold (size-agnostic heuristic).
The GNN policy is tested zero-shot on 32 and 64-node clusters.

Paper claim: "Our GNN encoder processes only the 2-hop neighborhood of each node,
keeping per-step inference at O(degree²) regardless of cluster size N."
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
from baselines.baselines import ThresholdPolicy, OraclePolicy, NoMigrationPolicy
from baselines.lstm_baseline import LSTMPolicy


SCALE_SIZES = [16, 32, 64]
EVAL_STEPS = 400
SEEDS = [42, 43, 44, 45, 46]


def run_ppo_at_scale(agent, n_servers: int, steps: int, seed: int) -> tuple[dict, float]:
    cfg = SimConfig(
        topology="fat_tree",
        n_servers=n_servers,
        seed=seed,
        session_arrival_rate=10.0,
        session_duration_mean=60.0,
        base_load=0.55,
        spike_probability=0.08,
        spike_magnitude=0.35,
        enable_failures=True,
        max_steps=steps,
    )
    sim = ClusterSimulator(cfg)
    n = sim.n  # may differ from n_servers (fat-tree rounds to k²)
    obs = sim.reset()

    inference_times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        action = agent.select_action(obs)
        inference_times.append(time.perf_counter() - t0)

        decisions = {}
        for sv in range(n):
            if action[sv % len(action)] == 0:
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

        obs, _, done, _ = sim.step(decisions)
        if done:
            break

    agent._buffer_clear()
    summary = sim.summary()
    median_inference_ms = float(np.median(inference_times) * 1000)
    return summary, median_inference_ms


def run_baseline_at_scale(policy, n_servers: int, steps: int, seed: int) -> tuple[dict, float]:
    cfg = SimConfig(
        topology="fat_tree",
        n_servers=n_servers,
        seed=seed,
        session_arrival_rate=10.0,
        session_duration_mean=60.0,
        base_load=0.55,
        spike_probability=0.08,
        spike_magnitude=0.35,
        enable_failures=True,
        max_steps=steps,
    )
    sim = ClusterSimulator(cfg)
    obs = sim.reset()

    inference_times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        decisions = policy.select_action(obs, sim)
        inference_times.append(time.perf_counter() - t0)
        obs, _, done, _ = sim.step(decisions)
        if done:
            break

    summary = sim.summary()
    median_ms = float(np.median(inference_times) * 1000)
    return summary, median_ms


def run_scale_experiment(model_path: str = "results/checkpoints/best_model.pt"):
    Path("results").mkdir(exist_ok=True)

    # Load PPO trained on 16-node cluster
    ppo = PPOAgent(n_servers=16, device="cpu")
    if Path(model_path).exists():
        ppo.load(model_path)
        ppo.model.eval()
        print(f"Loaded PPO from {model_path}")
    else:
        print(f"Warning: {model_path} not found — using untrained policy")

    threshold = ThresholdPolicy(0.70)
    oracle = OraclePolicy()
    no_mig = NoMigrationPolicy()
    lstm = LSTMPolicy(model_path="results/lstm_predictor.pt")

    results = {}

    print("\n" + "=" * 90)
    print("SCALE EXPERIMENT: Catch Rate and Inference Time vs Cluster Size")
    print("=" * 90)
    print(f"{'Policy':<22} {'N=16':>20} {'N=36':>20} {'N=64':>20}")
    print(f"{'':22} {'catch / ms':>20} {'catch / ms':>20} {'catch / ms':>20}")
    print("-" * 90)

    policies = [
        ("No Migration",    lambda obs, sim: no_mig.select_action(obs, sim),    False),
        ("Threshold 70%",   lambda obs, sim: threshold.select_action(obs, sim), False),
        ("LSTM Predict",    lambda obs, sim: lstm.select_action(obs, sim),      False),
        ("Oracle (UB)",     lambda obs, sim: oracle.select_action(obs, sim),    False),
        ("PPO+GNN (ours)",  None,                                               True),
    ]

    for policy_name, policy_fn, is_ppo in policies:
        row_results = {}
        row_str = f"{policy_name:<22}"

        for n_servers in SCALE_SIZES:
            all_cr, all_ms = [], []
            for seed in SEEDS:
                if is_ppo:
                    s, ms = run_ppo_at_scale(ppo, n_servers, EVAL_STEPS, seed)
                else:
                    # wrap for baseline
                    cfg = SimConfig(
                        topology="fat_tree", n_servers=n_servers, seed=seed,
                        session_arrival_rate=10.0, session_duration_mean=60.0,
                        base_load=0.55, spike_probability=0.08, spike_magnitude=0.35,
                        enable_failures=True, max_steps=EVAL_STEPS,
                    )
                    sim = ClusterSimulator(cfg)
                    obs = sim.reset()
                    t_inf = []
                    for _ in range(EVAL_STEPS):
                        t0 = time.perf_counter()
                        d = policy_fn(obs, sim)
                        t_inf.append(time.perf_counter() - t0)
                        obs, _, done, _ = sim.step(d)
                        if done:
                            break
                    s = sim.summary()
                    ms = float(np.median(t_inf) * 1000)

                all_cr.append(s.get("catch_rate", 0))
                all_ms.append(ms)

            actual_n = ClusterSimulator(SimConfig(topology="fat_tree", n_servers=n_servers,
                                                  seed=42)).n
            cr_m = np.mean(all_cr)
            ms_m = np.mean(all_ms)
            row_results[actual_n] = {"catch_rate": cr_m, "inference_ms": ms_m}
            row_str += f"  {cr_m:>8.1%}/{ms_m:>6.3f}ms"

        results[policy_name] = row_results
        marker = " ←" if is_ppo else ""
        print(row_str + marker)

    print("=" * 90)

    # Scaling analysis: print inference time ratio
    if "PPO+GNN (ours)" in results:
        gnn_times = {n: v["inference_ms"] for n, v in results["PPO+GNN (ours)"].items()}
        sizes = sorted(gnn_times.keys())
        print("\nGNN Inference Time Scaling:")
        for i, n in enumerate(sizes):
            if i == 0:
                print(f"  N={n:3d}: {gnn_times[n]:.3f}ms (baseline)")
            else:
                ratio = gnn_times[n] / gnn_times[sizes[0]]
                print(f"  N={n:3d}: {gnn_times[n]:.3f}ms  ({ratio:.1f}x vs N={sizes[0]})")
        if len(sizes) >= 2:
            size_ratio = sizes[-1] / sizes[0]
            time_ratio = gnn_times[sizes[-1]] / gnn_times[sizes[0]]
            print(f"\n  Cluster grew {size_ratio:.1f}x, inference grew {time_ratio:.1f}x")
            if time_ratio < size_ratio:
                print(f"  → Sub-linear scaling confirmed ✓")

    # Save results
    out = "results/scale_experiment.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
    return results


if __name__ == "__main__":
    run_scale_experiment()
