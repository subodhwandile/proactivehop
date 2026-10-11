"""
plot_results.py — Generate paper-quality figures from benchmark results.

Produces:
  Fig 1: Reconnect latency CDF (all policies, medium load, fat_tree)
  Fig 2: Catch rate vs load scenario (bar chart, 3 topologies)
  Fig 3: Training curve (reward + catch rate over episodes)
  Fig 4: Bandwidth overhead vs catch rate tradeoff scatter
  Fig 5: Policy comparison heatmap across all 9 conditions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

from sim.cluster import ClusterSimulator, SimConfig
from baselines.baselines import (
    NoMigrationPolicy, ThresholdPolicy, RandomPolicy, OraclePolicy,
)


PLOT_DIR = "results/plots"
POLICY_COLORS = {
    "No Migration":       "#d62728",
    "Threshold (80%)":    "#ff7f0e",
    "Threshold (70%)":    "#ffbb78",
    "Random":             "#9467bd",
    "Oracle":             "#2ca02c",
    "PPO+GNN (ours)":     "#1f77b4",
}
POLICY_MARKERS = {
    "No Migration":       "x",
    "Threshold (80%)":    "s",
    "Threshold (70%)":    "^",
    "Random":             "D",
    "Oracle":             "*",
    "PPO+GNN (ours)":     "o",
}


def collect_reconnect_latencies(policy, sim, steps):
    """Collect per-session reconnect latencies during simulation."""
    obs = sim.reset()
    for _ in range(steps):
        decisions = policy.select_action(obs, sim)
        obs, _, done, _ = sim.step(decisions)
        if done:
            break
    latencies = []
    for s in sim.sessions.values():
        latencies.extend(s.reconnect_latencies_ms)
    return latencies


def fig1_latency_cdf(output_dir: str):
    """Figure 1: CDF of reconnect latencies for all baselines."""
    print("Generating Fig 1: Latency CDF...")
    cfg = SimConfig(
        topology="fat_tree", n_servers=16, seed=42,
        base_load=0.45, spike_probability=0.05,
        enable_failures=True, max_steps=400,
    )

    policies = {
        "No Migration": NoMigrationPolicy(),
        "Threshold (80%)": ThresholdPolicy(0.80),
        "Threshold (70%)": ThresholdPolicy(0.70),
        "Random": RandomPolicy(0.10),
        "Oracle": OraclePolicy(),
    }

    # Try loading PPO
    ppo_path = "results/checkpoints/best_model.pt"
    ppo_latencies = None
    if Path(ppo_path).exists():
        from model.ppo_agent import PPOAgent
        from sim.session import SessionStatus
        cfg_tmp = SimConfig(n_servers=16, seed=0)
        sim_tmp = ClusterSimulator(cfg_tmp)
        ppo = PPOAgent(n_servers=sim_tmp.n, device="cpu")
        ppo.load(ppo_path)
        ppo.model.eval()

        sim = ClusterSimulator(cfg)
        obs = sim.reset()
        for _ in range(400):
            action = ppo.select_action(obs)
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
        ppo_latencies = []
        for s in sim.sessions.values():
            ppo_latencies.extend(s.reconnect_latencies_ms)
        ppo.model.train()

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, policy in policies.items():
        sim = ClusterSimulator(cfg)
        lats = collect_reconnect_latencies(policy, sim, 400)
        if lats:
            sorted_lats = np.sort(lats)
            cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
            ax.plot(sorted_lats, cdf,
                    color=POLICY_COLORS.get(name, "gray"),
                    label=name,
                    linewidth=2,
                    marker=POLICY_MARKERS.get(name, "o"),
                    markevery=max(1, len(sorted_lats)//10),
                    markersize=5)

    if ppo_latencies:
        sorted_lats = np.sort(ppo_latencies)
        cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
        ax.plot(sorted_lats, cdf,
                color=POLICY_COLORS["PPO+GNN (ours)"],
                label="PPO+GNN (ours)",
                linewidth=2.5,
                linestyle="--",
                marker="o",
                markevery=max(1, len(sorted_lats)//10),
                markersize=6)

    ax.set_xscale("log")
    ax.set_xlabel("Reconnect Latency (ms)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Fig 1: Session Reconnect Latency CDF\n"
                 "(Fat-Tree topology, Medium Load, 16 servers)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5, label='100ms SLA')
    ax.set_xlim(left=1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/fig1_latency_cdf.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig1_latency_cdf.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig1_latency_cdf.{{pdf,png}}")


def fig2_catch_rate_bars(output_dir: str):
    """Figure 2: Catch rate across load scenarios and topologies."""
    print("Generating Fig 2: Catch rate bar chart...")
    topologies = ["fat_tree", "random_geo", "scale_free"]
    loads = {
        "Low\n(base=0.25)":    {"base_load": 0.25, "spike_probability": 0.02, "enable_failures": False},
        "Medium\n(base=0.45)": {"base_load": 0.45, "spike_probability": 0.05, "enable_failures": True},
        "High\n(base=0.65)":   {"base_load": 0.65, "spike_probability": 0.10, "enable_failures": True},
    }

    policies_to_plot = {
        "No Migration": NoMigrationPolicy(),
        "Threshold (80%)": ThresholdPolicy(0.80),
        "Oracle": OraclePolicy(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    x = np.arange(len(loads))
    bar_width = 0.25

    for ax_idx, topology in enumerate(topologies):
        ax = axes[ax_idx]
        for p_idx, (p_name, policy) in enumerate(policies_to_plot.items()):
            catch_rates = []
            for load_cfg in loads.values():
                cfg = SimConfig(
                    topology=topology, n_servers=16, seed=42,
                    max_steps=300, **load_cfg,
                )
                sim = ClusterSimulator(cfg)
                summary = {}
                obs = sim.reset()
                for _ in range(300):
                    decisions = policy.select_action(obs, sim)
                    obs, _, done, _ = sim.step(decisions)
                    if done:
                        break
                summary = sim.summary()
                catch_rates.append(summary.get("catch_rate", 0))

            offset = (p_idx - len(policies_to_plot) / 2 + 0.5) * bar_width
            ax.bar(x + offset, catch_rates,
                   width=bar_width,
                   label=p_name,
                   color=POLICY_COLORS.get(p_name, "gray"),
                   alpha=0.85)

        ax.set_title(f"Topology: {topology.replace('_',' ').title()}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(list(loads.keys()), fontsize=8)
        ax.set_xlabel("Load Scenario", fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("Speculative Catch Rate", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle("Fig 2: Speculative Migration Catch Rate\nby Topology and Load Scenario",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_catch_rate_bars.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig2_catch_rate_bars.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig2_catch_rate_bars.{{pdf,png}}")


def fig3_training_curve(output_dir: str, log_file: str = "results/training_log.json"):
    """Figure 3: Training curve from training log."""
    print("Generating Fig 3: Training curve...")
    if not Path(log_file).exists():
        print(f"  Warning: {log_file} not found. Run train.py first.")
        return

    with open(log_file) as f:
        log = json.load(f)

    episodes = [e["episode"] for e in log]
    rewards = [e["total_reward"] for e in log]
    catch_rates = [e["catch_rate"] for e in log]
    p95_latencies = [e["p95_reconnect_ms"] for e in log]

    def smooth(vals, w=5):
        return np.convolve(vals, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Reward
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.3, color='steelblue')
    ep_s = episodes[4:]
    ax.plot(ep_s, smooth(rewards), color='steelblue', linewidth=2, label='Smoothed reward')
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title("Fig 3: PPO+GNN Training Convergence", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Catch rate + P95
    ax2 = axes[1]
    ax2.plot(episodes, catch_rates, alpha=0.3, color='green')
    ax2.plot(ep_s, smooth(catch_rates), color='green', linewidth=2, label='Catch Rate')
    ax2.set_ylabel("Catch Rate", color='green', fontsize=11)
    ax2.set_ylim(0, 1)

    ax3 = ax2.twinx()
    ax3.plot(episodes, p95_latencies, alpha=0.3, color='red')
    ax3.plot(ep_s, smooth(p95_latencies), color='red', linewidth=2, linestyle='--',
             label='P95 Latency')
    ax3.set_ylabel("P95 Reconnect Latency (ms)", color='red', fontsize=11)

    ax2.set_xlabel("Training Episode", fontsize=11)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_training_curve.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig3_training_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig3_training_curve.{{pdf,png}}")


def fig4_bw_tradeoff(output_dir: str):
    """Figure 4: Migration bandwidth overhead vs catch rate (Pareto frontier)."""
    print("Generating Fig 4: BW-Catch tradeoff...")
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    random_probs = [0.05, 0.10, 0.15, 0.20, 0.30]

    cfg = SimConfig(
        topology="fat_tree", n_servers=16, seed=42,
        base_load=0.45, spike_probability=0.05,
        enable_failures=True, max_steps=300,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Threshold policies at various thresholds
    th_bw, th_cr = [], []
    for t in thresholds:
        sim = ClusterSimulator(cfg)
        policy = ThresholdPolicy(t)
        obs = sim.reset()
        for _ in range(300):
            obs, _, done, _ = sim.step(policy.select_action(obs, sim))
            if done:
                break
        s = sim.summary()
        th_bw.append(s.get("total_migration_bw_mb", 0))
        th_cr.append(s.get("catch_rate", 0))
    ax.plot(th_bw, th_cr, 'o-', color=POLICY_COLORS["Threshold (80%)"],
            label="Threshold (varies)", linewidth=1.5)
    for t, bw, cr in zip(thresholds, th_bw, th_cr):
        ax.annotate(f"t={t:.0%}", (bw, cr), textcoords="offset points",
                    xytext=(4, 2), fontsize=7)

    # Random at various probs
    rnd_bw, rnd_cr = [], []
    for p in random_probs:
        sim = ClusterSimulator(cfg)
        policy = RandomPolicy(p)
        obs = sim.reset()
        for _ in range(300):
            obs, _, done, _ = sim.step(policy.select_action(obs, sim))
            if done:
                break
        s = sim.summary()
        rnd_bw.append(s.get("total_migration_bw_mb", 0))
        rnd_cr.append(s.get("catch_rate", 0))
    ax.plot(rnd_bw, rnd_cr, 's--', color=POLICY_COLORS["Random"],
            label="Random (varies)", linewidth=1.5)

    # Oracle (optimal)
    sim = ClusterSimulator(cfg)
    obs = sim.reset()
    oracle = OraclePolicy()
    for _ in range(300):
        obs, _, done, _ = sim.step(oracle.select_action(obs, sim))
        if done:
            break
    s = sim.summary()
    ax.scatter([s.get("total_migration_bw_mb", 0)], [s.get("catch_rate", 0)],
               s=200, marker="*", color=POLICY_COLORS["Oracle"],
               label="Oracle (upper bound)", zorder=5)

    # No migration
    ax.scatter([0], [0], s=100, marker="x", color=POLICY_COLORS["No Migration"],
               label="No Migration", zorder=5)

    ax.set_xlabel("Total Migration Bandwidth (MB)", fontsize=11)
    ax.set_ylabel("Speculative Catch Rate", fontsize=11)
    ax.set_title("Fig 4: Bandwidth Overhead vs Catch Rate Tradeoff\n"
                 "(Fat-Tree, Medium Load)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_bw_tradeoff.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig4_bw_tradeoff.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig4_bw_tradeoff.{{pdf,png}}")


def fig5_heatmap(output_dir: str, table_file: str = "results/benchmark_table.json"):
    """Figure 5: P95 latency heatmap across all 9 conditions."""
    print("Generating Fig 5: Heatmap...")
    if not Path(table_file).exists():
        print(f"  Warning: {table_file} not found. Run benchmark.py first.")
        return

    with open(table_file) as f:
        table = json.load(f)

    topologies = ["fat_tree", "random_geo", "scale_free"]
    loads = ["low", "medium", "high"]
    conditions = [f"{t}/{l}" for t in topologies for l in loads]

    all_policies = set()
    for c in conditions:
        if c in table:
            all_policies.update(table[c].keys())
    all_policies = sorted(all_policies)

    data = np.zeros((len(all_policies), len(conditions)))
    for ci, cond in enumerate(conditions):
        if cond not in table:
            continue
        for pi, pol in enumerate(all_policies):
            if pol in table[cond]:
                data[pi, ci] = table[cond][pol]["p95_reconnect_ms"]["mean"]

    fig, ax = plt.subplots(figsize=(12, max(4, len(all_policies) * 0.7)))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=max(500, data.max()))

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace("/", "\n") for c in conditions],
                       fontsize=8, rotation=0)
    ax.set_yticks(range(len(all_policies)))
    ax.set_yticklabels(all_policies, fontsize=9)

    for pi in range(len(all_policies)):
        for ci in range(len(conditions)):
            val = data[pi, ci]
            ax.text(ci, pi, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 300 else "white")

    plt.colorbar(im, ax=ax, label="P95 Reconnect Latency (ms)")
    ax.set_title("Fig 5: P95 Reconnect Latency (ms) — All Policies × All Conditions\n"
                 "(lower = better)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_heatmap.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig5_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig5_heatmap.{{pdf,png}}")


if __name__ == "__main__":
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    fig1_latency_cdf(PLOT_DIR)
    fig2_catch_rate_bars(PLOT_DIR)
    fig3_training_curve(PLOT_DIR)
    fig4_bw_tradeoff(PLOT_DIR)
    fig5_heatmap(PLOT_DIR)
    print(f"\nAll figures saved to {PLOT_DIR}/")
