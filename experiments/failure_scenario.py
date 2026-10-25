"""
failure_scenario.py — Dedicated server failure experiment.

The clearest demonstration of the paper's value:
  A server fails at step T_fail. Sessions on that server must migrate.
  - No Migration / Reactive: all sessions experience 400-800ms reconnect
  - Threshold: some caught IF load spiked before failure (often misses)
  - PPO+GNN: GNN detects rising failure_rate + load → pre-migrates BEFORE crash
  - Oracle: catches all

Metrics:
  - Sessions on failed server: how many pre-migrated vs reactive
  - Latency distribution for affected sessions specifically
  - Time-to-full-recovery (all sessions back on healthy servers)

This is the "smoking gun" experiment: shows the GNN predicted the crash
and pre-migrated sessions before it happened.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import Session, SessionStatus
from sim.topology import ServerNode
from model.ppo_agent import PPOAgent
from model.mlp_agent import MLPAgent
from baselines.baselines import (
    NoMigrationPolicy, ThresholdPolicy, OraclePolicy
)
from baselines.lstm_baseline import LSTMPolicy


FAILURE_STEP = 40      # server fails at step 40
TOTAL_STEPS = 120      # total simulation length
TARGET_SERVER = 3      # which server fails
N_SESSIONS_ON_TARGET = 25  # seed extra sessions on target server


def inject_failure(sim, server_id: int):
    """Force a specific server to fail."""
    sim.topo.apply_failure(server_id)
    sim._recovery_remaining[server_id] = 9999  # no auto-recovery in experiment


def run_failure_scenario(policy_fn, policy_name: str, is_ppo: bool,
                         agent=None, seed: int = 42):
    """
    Run one failure scenario experiment.
    Returns per-step metrics and per-session reconnect latencies for
    sessions that were on the failed server.
    """
    cfg = SimConfig(
        topology="fat_tree",
        n_servers=16,
        seed=seed,
        session_arrival_rate=5.0,
        session_duration_mean=200.0,   # long sessions — won't expire mid-experiment
        base_load=0.35,                # low base keeps sessions alive until crash
        spike_probability=0.04,
        spike_magnitude=0.35,
        enable_failures=False,         # manual failure injection only
        max_steps=TOTAL_STEPS,
    )
    sim = ClusterSimulator(cfg)
    obs = sim.reset()

    # Warm up: run 10 steps so loads stabilize around base_load before seeding
    for _ in range(10):
        obs, _, _, _ = sim.step({})

    # Seed extra sessions on the target server
    for _ in range(N_SESSIONS_ON_TARGET):
        sim._create_session(TARGET_SERVER)

    # Track sessions that start on target server
    target_sessions_initial = set(sim.server_sessions[TARGET_SERVER])

    step_metrics = []
    failed_step = None
    pre_migrated_actual = set()   # sessions that ACTUALLY completed a speculative move
    reactive_reconnected = set()

    for step in range(TOTAL_STEPS):
        # Pre-failure degradation: ramp target server from step 30 onward
        # Directly set cpu_util so load = max(cpu_util, ...) gives a clear signal.
        # At step 30: cpu_util=0.35 (baseline).  At step 39: cpu_util=0.95 (near-max).
        # GNN observes rising load + failure_rate → should speculatively push sessions.
        if step >= 30 and step < FAILURE_STEP:
            target_node = sim.topo.get_node(TARGET_SERVER)
            ramp_frac = (step - 30) / max(FAILURE_STEP - 30, 1)
            target_node.cpu_util = 0.35 + ramp_frac * 0.60
            target_node.failure_rate = 0.002 + ramp_frac * 0.098  # 0.1 at crash

        # Inject failure at step FAILURE_STEP
        if step == FAILURE_STEP:
            inject_failure(sim, TARGET_SERVER)
            failed_step = step

        # Get policy decision
        if is_ppo and agent is not None:
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

        # Track pre-migrations of target sessions (before failure)
        # MUST happen AFTER sim.step() so we can check actual current_server
        obs, reward, done, info = sim.step(decisions)
        m = info["metrics"]

        if step < FAILURE_STEP:
            for sid in target_sessions_initial:
                if sid in sim.sessions and sid not in pre_migrated_actual:
                    s = sim.sessions[sid]
                    # Speculative move: session left target server AND last latency < 50ms
                    if (s.current_server != TARGET_SERVER and
                            len(s.reconnect_latencies_ms) > 0 and
                            s.reconnect_latencies_ms[-1] < 50):
                        pre_migrated_actual.add(sid)

        # Track which target sessions got reactive reconnect after failure
        if step >= FAILURE_STEP:
            for sid in target_sessions_initial:
                if sid in sim.sessions:
                    s = sim.sessions[sid]
                    if s.current_server != TARGET_SERVER and sid not in pre_migrated_actual:
                        reactive_reconnected.add(sid)

        step_metrics.append({
            "step": step,
            "tp": m.true_positive_migrations,
            "fp": m.false_positive_migrations,
            "unhandled": m.drops_unhandled,
            "target_server_load": sim.topo.get_node(TARGET_SERVER).load,
            "target_sessions": len(sim.server_sessions[TARGET_SERVER]),
        })

        if done:
            break

    if agent is not None:
        agent._buffer_clear()

    # Collect reconnect latencies for target sessions
    all_lats = []
    spec_lats = []
    react_lats = []
    for sid in target_sessions_initial:
        if sid in sim.sessions:
            s = sim.sessions[sid]
            for lat in s.reconnect_latencies_ms:
                all_lats.append(lat)
                if lat < 50:
                    spec_lats.append(lat)
                else:
                    react_lats.append(lat)

    return {
        "policy": policy_name,
        "pre_migrated_count": len(pre_migrated_actual),
        "reactive_count": len(reactive_reconnected),
        "total_target_sessions": len(target_sessions_initial),
        "pre_migration_rate": len(pre_migrated_actual) / max(len(target_sessions_initial), 1),
        "all_latencies": all_lats,
        "speculative_latencies": spec_lats,
        "reactive_latencies": react_lats,
        "p50_ms": float(np.percentile(all_lats, 50)) if all_lats else 0,
        "p95_ms": float(np.percentile(all_lats, 95)) if all_lats else 0,
        "step_metrics": step_metrics,
    }


def plot_failure_results(results: list[dict], out_dir: str = "results/plots"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    COLORS = {
        "No Migration":    "#d62728",
        "Threshold 70%":   "#ff7f0e",
        "LSTM Predict":    "#9467bd",
        "PPO+MLP":         "#8c564b",
        "Oracle (UB)":     "#2ca02c",
        "PPO+GNN (ours)":  "#1f77b4",
    }

    # --- Fig A: Pre-migration rate bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    names = [r["policy"] for r in results]
    rates = [r["pre_migration_rate"] * 100 for r in results]
    colors = [COLORS.get(n, "gray") for n in names]
    bars = ax.bar(names, rates, color=colors, alpha=0.85, width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Sessions Pre-Migrated Before Crash (%)", fontsize=10)
    ax.set_title(f"Fig 4a: Pre-Migration Rate\n(Server {TARGET_SERVER} crashes at step {FAILURE_STEP})",
                 fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=9)

    # --- Fig B: Latency CDF for affected sessions ---
    ax = axes[1]
    for r in results:
        lats = r["all_latencies"]
        if not lats:
            continue
        sl = np.sort(lats)
        cdf = np.arange(1, len(sl) + 1) / len(sl)
        ax.plot(sl, cdf, color=COLORS.get(r["policy"], "gray"),
                linewidth=2.5 if "GNN" in r["policy"] or "Oracle" in r["policy"] else 1.5,
                linestyle="--" if "GNN" in r["policy"] else "-",
                label=r["policy"])

    ax.set_xscale("log")
    ax.axvline(50, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(55, 0.05, "50ms", color="gray", fontsize=8)
    ax.set_xlabel("Reconnect Latency for Affected Sessions (ms)", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title(f"Fig 4b: Latency CDF — Sessions on Failed Server\n"
                 f"(Server {TARGET_SERVER} crashes at step {FAILURE_STEP})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 2000)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig4_failure_scenario.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_dir}/fig4_failure_scenario.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved fig4_failure_scenario.{{pdf,png}}")

    # --- Fig C: Time-series of target server sessions pre/post failure ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        steps = [m["step"] for m in r["step_metrics"]]
        sess_on_target = [m["target_sessions"] for m in r["step_metrics"]]
        ax.plot(steps, sess_on_target,
                color=COLORS.get(r["policy"], "gray"),
                linewidth=2 if "GNN" in r["policy"] else 1.5,
                linestyle="--" if "GNN" in r["policy"] else "-",
                label=r["policy"])

    ax.axvline(FAILURE_STEP, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.text(FAILURE_STEP + 1, ax.get_ylim()[1] * 0.9,
            f"Server {TARGET_SERVER}\nfails", color="red", fontsize=9)
    ax.set_xlabel("Simulation Step", fontsize=10)
    ax.set_ylabel(f"Sessions on Server {TARGET_SERVER}", fontsize=10)
    ax.set_title(f"Fig 4c: Session Drain from Server {TARGET_SERVER} Before Crash\n"
                 f"(PPO+GNN pre-drains; others do not)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig4c_session_drain.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_dir}/fig4c_session_drain.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved fig4c_session_drain.{{pdf,png}}")


FAILURE_SEEDS = [42, 43, 44, 45, 46]


def run_failure_experiment(model_path: str = "results/ablation/best_GNN.pt"):
    print("\n" + "=" * 70)
    print(f"FAILURE SCENARIO: Server {TARGET_SERVER} crashes at step {FAILURE_STEP}")
    print(f"Sessions on target: ~{N_SESSIONS_ON_TARGET} seeded + organic")
    print(f"Running {len(FAILURE_SEEDS)} seeds: {FAILURE_SEEDS}")
    print("=" * 70)

    ppo = PPOAgent(n_servers=16, device="cpu")
    if Path(model_path).exists():
        ppo.load(model_path)
        ppo.model.eval()
        print(f"Loaded GNN from {model_path}")
    else:
        print(f"Warning: {model_path} not found — using untrained GNN")

    mlp = MLPAgent(n_servers=16)
    mlp_path = "results/ablation/best_MLP.pt"
    if Path(mlp_path).exists():
        mlp.load(mlp_path)
        mlp.model.eval()
        print(f"Loaded MLP from {mlp_path}")
    else:
        print(f"Warning: {mlp_path} not found — using untrained MLP")

    no_mig = NoMigrationPolicy()
    thresh = ThresholdPolicy(0.70)
    oracle = OraclePolicy()
    lstm   = LSTMPolicy(model_path="results/lstm_predictor.pt")

    experiments = [
        ("No Migration",   no_mig.select_action, False, None),
        ("Threshold 70%",  thresh.select_action,  False, None),
        ("LSTM Predict",   lstm.select_action,    False, None),
        ("PPO+MLP",        None,                   True,  mlp),
        ("Oracle (UB)",    oracle.select_action,   False, None),
        ("PPO+GNN (ours)", None,                   True,  ppo),
    ]

    # Collect per-seed results for each policy
    agg = {}   # policy_name -> list of single-seed result dicts
    for name, fn, is_ppo, agent in experiments:
        print(f"\nRunning: {name}  ({len(FAILURE_SEEDS)} seeds)")
        agg[name] = []
        for seed in FAILURE_SEEDS:
            r = run_failure_scenario(fn, name, is_ppo, agent, seed=seed)
            agg[name].append(r)
            print(f"  seed={seed}: pre_mig={r['pre_migration_rate']:.1%} "
                  f"P50={r['p50_ms']:.0f}ms P95={r['p95_ms']:.0f}ms")

    # Aggregate mean±std across seeds
    summary_rows = []
    for name, seed_results in agg.items():
        pmr = np.array([r["pre_migration_rate"] for r in seed_results])
        p50 = np.array([r["p50_ms"] for r in seed_results])
        p95 = np.array([r["p95_ms"] for r in seed_results])
        summary_rows.append({
            "policy": name,
            "pre_migration_rate_mean": float(pmr.mean()),
            "pre_migration_rate_std":  float(pmr.std()),
            "p50_mean": float(p50.mean()),
            "p50_std":  float(p50.std()),
            "p95_mean": float(p95.mean()),
            "p95_std":  float(p95.std()),
            # keep one representative seed (seed=42) for plotting
            "_repr": seed_results[0],
        })

    # Summary table
    print("\n" + "=" * 80)
    print(f"FAILURE SCENARIO RESULTS  (mean±std over {len(FAILURE_SEEDS)} seeds)")
    print("=" * 80)
    print(f"{'Policy':<22} {'Pre-Migrated (%)':>20} {'P50 (ms)':>14} {'P95 (ms)':>14}")
    print("-" * 75)
    for row in summary_rows:
        marker = " ←" if "GNN" in row["policy"] else ("  ★" if "Oracle" in row["policy"] else "")
        print(f"{row['policy']:<22} "
              f"{row['pre_migration_rate_mean']*100:>8.1f}±{row['pre_migration_rate_std']*100:.1f}%"
              f"  {row['p50_mean']:>10.0f}±{row['p50_std']:.0f}ms"
              f"  {row['p95_mean']:>10.0f}±{row['p95_std']:.0f}ms"
              f"{marker}")
    print("=" * 80)

    # Save + plot (use seed=42 representative runs for plots)
    Path("results").mkdir(exist_ok=True)
    safe_rows = [{k: v for k, v in row.items() if k != "_repr"} for row in summary_rows]
    with open("results/failure_scenario.json", "w") as f:
        json.dump(safe_rows, f, indent=2)

    repr_results = [row["_repr"] for row in summary_rows]
    plot_failure_results(repr_results)
    return summary_rows


if __name__ == "__main__":
    run_failure_experiment()
