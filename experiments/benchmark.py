"""
benchmark.py — Full benchmark suite comparing all policies.

Runs each policy on 3 topology types × 3 load scenarios = 9 conditions.
Generates paper-quality tables and plots.

Usage: python experiments/benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict

from sim.cluster import ClusterSimulator, SimConfig
from model.ppo_agent import PPOAgent
from baselines.baselines import (
    NoMigrationPolicy,
    ThresholdPolicy,
    RandomPolicy,
    OraclePolicy,
)


TOPOLOGIES = ["fat_tree", "random_geo", "scale_free"]

LOAD_SCENARIOS = {
    "low":    {"base_load": 0.25, "spike_probability": 0.02, "enable_failures": False},
    "medium": {"base_load": 0.45, "spike_probability": 0.05, "enable_failures": True},
    "high":   {"base_load": 0.65, "spike_probability": 0.10, "enable_failures": True},
}

EVAL_STEPS = 600     # 10 min simulated per condition
N_SEEDS = 3          # repeat per seed for confidence intervals


def run_policy(policy, sim: ClusterSimulator, steps: int) -> dict:
    """Run a policy for `steps` steps, return summary metrics."""
    obs = sim.reset()
    for _ in range(steps):
        decisions = policy.select_action(obs, sim)
        obs, _, done, _ = sim.step(decisions)
        if done:
            break
    return sim.summary()


def run_ppo_policy(agent: PPOAgent, sim: ClusterSimulator, steps: int) -> dict:
    """Run trained PPO agent (inference only, no training)."""
    from sim.session import SessionStatus
    obs = sim.reset()
    for _ in range(steps):
        action = agent.select_action(obs)  # stores in buffer — we'll discard
        decisions = {}
        for server_id in range(sim.n):
            if action[server_id] == 0:
                continue
            node = sim.topo.get_node(server_id)
            if node.load < 0.5 or not node.is_alive:
                continue
            neighbors = sim.topo.neighbors(server_id)
            live_nb = [nb for nb in neighbors
                       if sim.topo.get_node(nb).is_alive
                       and sim.topo.get_node(nb).load < 0.7]
            if not live_nb:
                continue
            target = min(live_nb, key=lambda nb: sim.topo.get_node(nb).load)
            for sid in list(sim.server_sessions[server_id])[:3]:
                if sid in sim.sessions and sim.sessions[sid].status == SessionStatus.ACTIVE:
                    decisions[sid] = target
        obs, _, done, _ = sim.step(decisions)
        if done:
            break
    # Discard PPO buffer (inference mode)
    agent._buffer_clear()
    return sim.summary()


def run_benchmark(
    model_path: str = "results/checkpoints/best_model.pt",
    output_dir: str = "results",
    seeds: list = None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = list(range(N_SEEDS))

    # Load trained PPO model
    ppo_available = False
    ppo_agent = None
    if Path(model_path).exists():
        cfg_tmp = SimConfig(n_servers=16, seed=0)
        sim_tmp = ClusterSimulator(cfg_tmp)
        ppo_agent = PPOAgent(n_servers=sim_tmp.n, device="cpu")
        ppo_agent.load(model_path)
        ppo_agent.model.eval()
        ppo_available = True
        print(f"Loaded PPO model from {model_path}")
    else:
        print(f"Warning: no PPO model at {model_path} — skipping PPO baseline")

    policies = {
        "No Migration": NoMigrationPolicy(),
        "Threshold (80%)": ThresholdPolicy(load_threshold=0.80),
        "Threshold (70%)": ThresholdPolicy(load_threshold=0.70),
        "Random": RandomPolicy(trigger_prob=0.10),
        "Oracle": OraclePolicy(),
    }
    if ppo_available:
        policies["PPO+GNN (ours)"] = None  # handled separately

    results = defaultdict(lambda: defaultdict(list))

    total_runs = len(TOPOLOGIES) * len(LOAD_SCENARIOS) * len(policies) * len(seeds)
    run_idx = 0

    for topology in TOPOLOGIES:
        for scenario_name, scenario_cfg in LOAD_SCENARIOS.items():
            condition = f"{topology}/{scenario_name}"
            print(f"\n--- Condition: {condition} ---")

            for policy_name, policy in policies.items():
                seed_results = []
                for seed in seeds:
                    cfg = SimConfig(
                        topology=topology,
                        n_servers=16,
                        seed=seed,
                        max_steps=EVAL_STEPS,
                        **scenario_cfg,
                    )
                    sim = ClusterSimulator(cfg)

                    t0 = time.time()
                    if policy_name == "PPO+GNN (ours)":
                        summary = run_ppo_policy(ppo_agent, sim, EVAL_STEPS)
                    else:
                        summary = run_policy(policy, sim, EVAL_STEPS)
                    elapsed = time.time() - t0

                    seed_results.append(summary)
                    run_idx += 1
                    print(f"  [{run_idx}/{total_runs}] {policy_name:25s} seed={seed} "
                          f"catch={summary.get('catch_rate',0):.2%} "
                          f"p95={summary.get('p95_reconnect_ms',0):.0f}ms "
                          f"({elapsed:.1f}s)")

                results[condition][policy_name] = seed_results

    # Save raw results
    raw_path = f"{output_dir}/benchmark_raw.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {raw_path}")

    # Build summary table
    table = build_summary_table(results)
    table_path = f"{output_dir}/benchmark_table.json"
    with open(table_path, "w") as f:
        json.dump(table, f, indent=2)
    print_table(table)

    return results, table


def build_summary_table(results: dict) -> dict:
    """
    Compute mean ± std across seeds for each metric.
    """
    metrics = [
        "catch_rate", "false_positive_rate",
        "p50_reconnect_ms", "p95_reconnect_ms", "p99_reconnect_ms",
        "unhandled_drops", "total_migration_bw_mb",
    ]
    table = {}
    for condition, policy_results in results.items():
        table[condition] = {}
        for policy_name, seed_list in policy_results.items():
            row = {}
            for m in metrics:
                vals = [s.get(m, 0) for s in seed_list]
                row[m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
            table[condition][policy_name] = row
    return table


def print_table(table: dict):
    """Print a paper-style comparison table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    for condition, policy_results in table.items():
        print(f"\nCondition: {condition}")
        print(f"{'Policy':<28} {'Catch Rate':>12} {'P95 Latency':>14} "
              f"{'FP Rate':>10} {'Unhandled':>12} {'BW (MB)':>10}")
        print("-" * 90)

        for policy_name, row in sorted(policy_results.items()):
            cr = row["catch_rate"]["mean"]
            p95 = row["p95_reconnect_ms"]["mean"]
            fpr = row["false_positive_rate"]["mean"]
            drops = row["unhandled_drops"]["mean"]
            bw = row["total_migration_bw_mb"]["mean"]
            cr_std = row["catch_rate"]["std"]
            p95_std = row["p95_reconnect_ms"]["std"]

            print(f"{policy_name:<28} "
                  f"{cr:>10.1%}±{cr_std:.1%} "
                  f"{p95:>10.0f}±{p95_std:.0f}ms "
                  f"{fpr:>10.1%} "
                  f"{drops:>12.0f} "
                  f"{bw:>10.2f}")


if __name__ == "__main__":
    run_benchmark()
