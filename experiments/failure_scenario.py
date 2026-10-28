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
RAMP_START = 30        # degradation ramp begins (= seeding step)
SEED_STEP = RAMP_START # seed sessions at ramp start so denominator is ramp-window specific


def inject_failure(sim, server_id: int):
    """Force a specific server to fail."""
    sim.topo.apply_failure(server_id)
    sim._recovery_remaining[server_id] = 9999  # no auto-recovery in experiment


def run_failure_scenario(policy_fn, policy_name: str, is_ppo: bool,
                         agent=None, seed: int = 42):
    """
    Run one failure scenario experiment.

    Sessions are seeded on Server TARGET_SERVER at the start of the degradation
    ramp (step RAMP_START).  Server 3's cpu_util and mem_util are reset to
    base_load before seeding so warm-up spikes do not pre-drain sessions.
    The pre-migration denominator is the set of sessions on Server TARGET_SERVER
    at seeding time (target_sessions_initial).
    Pre-migration count = sessions from that set that were moved OFF Server 3
    by the policy during the ramp window (steps RAMP_START .. FAILURE_STEP),
    tracked via sim._migrated_this_step.
    This measures ramp-driven proactive migration, not warm-up or reactive behaviour.

    Returns per-step metrics and per-session reconnect latencies for
    sessions that were on the failed server.
    """
    # Seed the simulator RNG and numpy so that environment randomness is
    # reproducible.  PPO agents use greedy (argmax) action selection, making
    # policy decisions fully deterministic given the checkpoint.
    # set/dict iteration over server_sessions uses sorted() throughout, so
    # PYTHONHASHSEED is not required for reproducibility.
    import torch as _torch
    import numpy as _np
    _np.random.seed(seed)
    _torch.manual_seed(seed)

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

    step_metrics = []
    failed_step = None
    seeded_sids: set = set()            # exactly the N_SESSIONS_ON_TARGET created at SEED_STEP
    target_sessions_initial: set = set()  # alias for seeded_sids (set after seeding)
    pre_migrated_actual = set()        # seeded sessions moved off Server 3 by policy during ramp
    reactive_reconnected = set()
    ramp_migrations_total = 0          # total cluster-wide migrations during ramp window (cost)

    for step in range(TOTAL_STEPS):
        # Seed extra sessions on the target server just before the ramp begins.
        # Capturing the denominator here ensures we measure ramp-driven pre-migration,
        # not early-step load-spike behaviour.
        if step == SEED_STEP:
            # Reset Server 3 to base load so warm-up spikes don't cause policies
            # to immediately drain the seeded sessions before the ramp signal appears.
            # load = max(cpu_util, mem_util, sessions/max) so both must be reset.
            target_node = sim.topo.get_node(TARGET_SERVER)
            target_node.cpu_util = cfg.base_load         # = 0.35
            target_node.mem_util = cfg.base_load * 0.8   # = 0.28
            sim._spike_remaining[TARGET_SERVER] = 0.0    # clear any active spike
            # Capture ONLY the explicitly seeded IDs so the denominator is identical
            # across all policies (independent of what each policy left on Server 3
            # during the warm-up steps 0–29).
            for _ in range(N_SESSIONS_ON_TARGET):
                s = sim._create_session(TARGET_SERVER)
                seeded_sids.add(s.session_id)
            target_sessions_initial = seeded_sids

        # Pre-failure degradation: ramp target server from RAMP_START onward.
        # At step 30: cpu_util=0.35 (baseline).  At step 39: cpu_util=0.95 (near-max).
        # GNN observes rising load + failure_rate → should speculatively push sessions.
        if step >= RAMP_START and step < FAILURE_STEP:
            target_node = sim.topo.get_node(TARGET_SERVER)
            ramp_frac = (step - RAMP_START) / max(FAILURE_STEP - RAMP_START, 1)
            target_node.cpu_util = 0.35 + ramp_frac * 0.60
            target_node.failure_rate = 0.002 + ramp_frac * 0.098  # 0.1 at crash

        # Inject failure at step FAILURE_STEP
        if step == FAILURE_STEP:
            inject_failure(sim, TARGET_SERVER)
            failed_step = step

        # Get policy decision (PPO agents use greedy/argmax for deterministic eval)
        if is_ppo and agent is not None:
            action = agent.select_action(obs, greedy=True)
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
                for sid in sorted(sim.server_sessions[sv]):
                    if sid in sim.sessions and sim.sessions[sid].status == SessionStatus.ACTIVE:
                        decisions[sid] = target
        else:
            decisions = policy_fn(obs, sim)

        # Execute simulation step.  All tracking happens AFTER this call so we
        # observe the post-step server assignments.
        obs, reward, done, info = sim.step(decisions)
        m = info["metrics"]

        # Track pre-migrations ONLY during ramp window (steps 30–39).
        # A seeded session is pre-migrated if it was moved OFF Server 3 by
        # the policy (appears in _migrated_this_step with source=TARGET_SERVER).
        # The denominator is exactly the N_SESSIONS_ON_TARGET seeded IDs,
        # identical for every policy.
        # Also accumulate ramp_migrations_total as a cost/churn metric.
        if RAMP_START <= step < FAILURE_STEP:
            ramp_migrations_total += len(sim._migrated_this_step)
            if seeded_sids:
                for sid in seeded_sids:
                    if sid not in pre_migrated_actual:
                        if (sid in sim._migrated_this_step
                                and sim._migrated_this_step[sid] == TARGET_SERVER):
                            pre_migrated_actual.add(sid)

        # Track which target sessions got reactive reconnect after failure
        if seeded_sids and step >= FAILURE_STEP:
            for sid in seeded_sids:
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

    # Collect reconnect latencies for the 25 seeded target sessions
    all_lats = []
    spec_lats = []
    react_lats = []
    n_target = len(seeded_sids)  # always N_SESSIONS_ON_TARGET (25), same for every policy
    for sid in seeded_sids:
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
        "total_target_sessions": n_target,   # always N_SESSIONS_ON_TARGET
        "ramp_migrations_total": ramp_migrations_total,  # cost: all policy moves during ramp
        "pre_migration_rate": len(pre_migrated_actual) / max(n_target, 1),
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
        "No Migration":       "#d62728",
        "Threshold 70%":      "#ff7f0e",
        "LSTM Predict":       "#9467bd",
        "PPO+MLP":            "#8c564b",
        "p_drop heuristic":   "#2ca02c",
        "PPO+GNN":            "#1f77b4",
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
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig4c_session_drain.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_dir}/fig4c_session_drain.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved fig4c_session_drain.{{pdf,png}}")


FAILURE_SEEDS = list(range(42, 62))   # 20 seeds for stable estimates


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
        ("No Migration",      no_mig.select_action, False, None),
        ("Threshold 70%",     thresh.select_action,  False, None),
        ("LSTM Predict",      lstm.select_action,    False, None),
        ("PPO+MLP",           None,                   True,  mlp),
        ("p_drop heuristic",  oracle.select_action,   False, None),
        ("PPO+GNN",           None,                   True,  ppo),
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
                  f"ramp_migs={r['ramp_migrations_total']} "
                  f"P50={r['p50_ms']:.0f}ms P95={r['p95_ms']:.0f}ms")

    # Aggregate mean±std across seeds
    summary_rows = []
    for name, seed_results in agg.items():
        pmr  = np.array([r["pre_migration_rate"] for r in seed_results])
        p50  = np.array([r["p50_ms"] for r in seed_results])
        p95  = np.array([r["p95_ms"] for r in seed_results])
        rmig = np.array([r["ramp_migrations_total"] for r in seed_results])
        summary_rows.append({
            "policy": name,
            "pre_migration_rate_mean": float(pmr.mean()),
            "pre_migration_rate_std":  float(pmr.std()),
            "pre_migration_rate_per_seed": pmr.tolist(),
            "ramp_migrations_mean": float(rmig.mean()),
            "ramp_migrations_std":  float(rmig.std()),
            "seeds": FAILURE_SEEDS,
            "p50_mean": float(p50.mean()),
            "p50_std":  float(p50.std()),
            "p95_mean": float(p95.mean()),
            "p95_std":  float(p95.std()),
            "_repr": seed_results[0],
        })

    # Summary table
    print("\n" + "=" * 90)
    print(f"FAILURE SCENARIO RESULTS  (mean±std over {len(FAILURE_SEEDS)} seeds)")
    print("=" * 90)
    print(f"{'Policy':<22} {'Pre-Migrated (%)':>20} {'RampMigs':>12} {'P50 (ms)':>14} {'P95 (ms)':>14}")
    print("-" * 85)
    for row in summary_rows:
        marker = " ←" if "GNN" in row["policy"] else ""
        print(f"{row['policy']:<22} "
              f"{row['pre_migration_rate_mean']*100:>8.1f}±{row['pre_migration_rate_std']*100:.1f}%"
              f"  {row['ramp_migrations_mean']:>8.0f}±{row['ramp_migrations_std']:.0f}"
              f"  {row['p50_mean']:>10.0f}±{row['p50_std']:.0f}ms"
              f"  {row['p95_mean']:>10.0f}±{row['p95_std']:.0f}ms"
              f"{marker}")
    print("=" * 90)

    # Save summary + per-step data for the seed=42 representative run
    Path("results").mkdir(exist_ok=True)
    safe_rows = [{k: v for k, v in row.items() if k != "_repr"} for row in summary_rows]
    with open("results/failure_scenario.json", "w") as f:
        json.dump(safe_rows, f, indent=2)

    # Save per-step drain data so the plot can be regenerated without re-running
    repr_results = [row["_repr"] for row in summary_rows]
    repr_serialisable = [
        {k: v for k, v in r.items() if k != "latencies"}
        for r in repr_results
    ]
    with open("results/failure_scenario_step_data.json", "w") as f:
        json.dump(repr_serialisable, f, indent=2)

    plot_failure_results(repr_results)
    return summary_rows


if __name__ == "__main__":
    run_failure_experiment()
