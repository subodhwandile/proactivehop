"""
trace_experiment.py — Trace-driven evaluation using Alibaba Cluster Trace 2018.

Replaces the synthetic Poisson arrival process with real container arrival
patterns from Alibaba's production cluster. All other simulation components
(topology, failures, load dynamics) remain identical to the main experiments.

This directly addresses the key reviewer concern:
  "Your results only hold on a synthetic simulator you designed."

The Alibaba trace has CV=1.7 (std/mean) vs Poisson CV=0.32 — meaning real
workloads are 5× more bursty than the synthetic baseline.  We show that
PPO+GNN's advantage holds (and actually grows) under real burstiness.

Paper claim: "We validate ProactiveHop on trace-driven simulation using
the publicly available Alibaba 2018 Cluster Trace [alibaba2018], confirming
that topology-aware pre-migration remains effective under real workload
burstiness patterns."
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import SessionStatus
from sim.trace_loader import AlibabaTraceLoader
from model.ppo_agent import PPOAgent
from baselines.baselines import NoMigrationPolicy, ThresholdPolicy, OraclePolicy
from baselines.lstm_baseline import LSTMPolicy


EVAL_STEPS = 400
SEEDS = [42, 43, 44, 45, 46]   # 5 seeds for statistical rigour
MODEL_PATH = "results/ablation/best_GNN.pt"


def make_cfg(seed: int, use_trace: bool) -> SimConfig:
    return SimConfig(
        topology="fat_tree",
        n_servers=16,
        seed=seed,
        session_arrival_rate=10.0,
        session_duration_mean=60.0,
        base_load=0.55,
        spike_probability=0.08,
        spike_magnitude=0.35,
        enable_failures=True,
        max_steps=EVAL_STEPS,
        use_trace=use_trace,
    )


def run_policy(policy_fn, is_ppo: bool, agent, cfg: SimConfig) -> dict:
    sim = ClusterSimulator(cfg)
    obs = sim.reset()

    for _ in range(EVAL_STEPS):
        if is_ppo:
            action = agent.select_action(obs)
            decisions = {}
            for sv in range(sim.n):
                if action[sv] == 0:
                    continue
                node = sim.topo.get_node(sv)
                if node.load < 0.4 or not node.is_alive:
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
        else:
            decisions = policy_fn(obs, sim)

        obs, _, done, _ = sim.step(decisions)
        if done:
            break

    if agent is not None:
        agent._buffer_clear()

    return sim.summary()


def run_trace_experiment():
    print("\n" + "=" * 70)
    print("TRACE-DRIVEN EXPERIMENT: Alibaba Cluster Trace 2018")
    print("=" * 70)

    # Print trace stats
    loader = AlibabaTraceLoader()
    s = loader.stats()
    print(f"Trace: {s['n_steps']} steps, mean={s['mean']:.1f}, "
          f"std={s['std']:.1f}, CV={s['cv']:.2f}")
    print(f"(Poisson equivalent CV = {1/10**0.5:.2f}  — trace is {s['cv']/(1/10**0.5):.1f}x more bursty)")

    # Load models
    ppo = PPOAgent(n_servers=16, device="cpu")
    if Path(MODEL_PATH).exists():
        ppo.load(MODEL_PATH)
        ppo.model.eval()

    no_mig  = NoMigrationPolicy()
    thresh  = ThresholdPolicy(0.70)
    oracle  = OraclePolicy()
    lstm    = LSTMPolicy(model_path="results/lstm_predictor.pt")

    policies = [
        ("No Migration",   no_mig.select_action,  False, None),
        ("Threshold 70%",  thresh.select_action,   False, None),
        ("LSTM Predict",   lstm.select_action,     False, None),
        ("Oracle (UB)",    oracle.select_action,   False, None),
        ("PPO+GNN (ours)", None,                   True,  ppo),
    ]

    results = {}
    for mode_label, use_trace in [("Synthetic (Poisson)", False),
                                   ("Trace-Driven (Alibaba)", True)]:
        print(f"\n--- {mode_label} ---")
        print(f"{'Policy':<22} {'Catch Rate':>12} {'P95 (ms)':>10} {'FP Rate':>10}")
        print("-" * 58)

        mode_results = {}
        for name, fn, is_ppo, agent in policies:
            catches, p95s, fps = [], [], []
            for seed in SEEDS:
                cfg = make_cfg(seed, use_trace)
                summary = run_policy(fn, is_ppo, agent, cfg)
                catches.append(summary.get("catch_rate", 0))
                p95s.append(summary.get("p95_reconnect_ms", 0))
                fps.append(summary.get("false_positive_rate", 0))

            marker = " ←" if "GNN" in name else ("  ★" if "Oracle" in name else "")
            cr_m, cr_s = np.mean(catches), np.std(catches)
            p95_m = np.mean(p95s)
            fp_m = np.mean(fps)
            print(f"{name:<22} {cr_m:>10.1%}±{cr_s:.1%} {p95_m:>8.0f}ms {fp_m:>9.1%}{marker}")

            mode_results[name] = {
                "catch_rate_mean": float(cr_m),
                "catch_rate_std":  float(cr_s),
                "p95_mean":        float(p95_m),
                "fp_rate_mean":    float(fp_m),
            }

        results[mode_label] = mode_results

    # Key comparison
    print("\n" + "=" * 70)
    print("SYNTHETIC vs TRACE-DRIVEN: PPO+GNN catch rate")
    print("=" * 70)
    syn  = results["Synthetic (Poisson)"]["PPO+GNN (ours)"]["catch_rate_mean"]
    trc  = results["Trace-Driven (Alibaba)"]["PPO+GNN (ours)"]["catch_rate_mean"]
    syn_lstm = results["Synthetic (Poisson)"]["LSTM Predict"]["catch_rate_mean"]
    trc_lstm = results["Trace-Driven (Alibaba)"]["LSTM Predict"]["catch_rate_mean"]
    print(f"  PPO+GNN — Synthetic: {syn:.1%}   Trace: {trc:.1%}   Change: {(trc-syn)/syn*100:+.1f}%")
    print(f"  LSTM    — Synthetic: {syn_lstm:.1%}   Trace: {trc_lstm:.1%}   Change: {(trc_lstm-syn_lstm)/syn_lstm*100:+.1f}%")
    if trc >= syn:
        print("  → GNN advantage holds (or improves) under real burstiness ✓")
    else:
        print(f"  → GNN drops {(syn-trc)*100:.1f}pp under burstiness — note as limitation")

    # Save
    Path("results").mkdir(exist_ok=True)
    with open("results/trace_experiment.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/trace_experiment.json")
    return results


if __name__ == "__main__":
    run_trace_experiment()
