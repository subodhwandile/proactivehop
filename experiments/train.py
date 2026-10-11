"""
train.py — Training script for the PPO + GNN migration policy.

Trains for N episodes, logs metrics, saves checkpoints.
Usage: python experiments/train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
from pathlib import Path

from sim.cluster import ClusterSimulator, SimConfig
from model.ppo_agent import PPOAgent


def train(
    n_episodes: int = 150,
    steps_per_episode: int = 100,
    seed: int = 42,
    checkpoint_dir: str = "results/checkpoints",
    log_file: str = "results/training_log.json",
):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(
        topology="fat_tree",
        n_servers=16,
        seed=seed,
        session_arrival_rate=10.0,
        session_duration_mean=60.0,
        base_load=0.55,
        spike_probability=0.08,
        spike_magnitude=0.35,
        enable_failures=True,
        max_steps=steps_per_episode,
    )

    sim = ClusterSimulator(cfg)
    n = sim.n

    agent = PPOAgent(
        n_servers=n,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.05,   # higher entropy encourages exploration
        n_epochs=4,
        device="cpu",
    )

    training_log = []
    best_reward = -np.inf

    print(f"Training PPO+GNN on {n}-server cluster for {n_episodes} episodes...")
    print(f"Steps per episode: {steps_per_episode}")

    for ep in range(n_episodes):
        obs = sim.reset()
        ep_reward = 0.0
        ep_start = time.time()

        for step in range(steps_per_episode):
            # Agent selects action from graph observation
            action = agent.select_action(obs)

            # Translate binary server action to session decisions
            decisions = {}
            for server_id in range(n):
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
                for sid in list(sim.server_sessions[server_id])[:20]:
                    if sid in sim.sessions:
                        from sim.session import SessionStatus
                        if sim.sessions[sid].status == SessionStatus.ACTIVE:
                            decisions[sid] = target

            obs, reward, done, info = sim.step(decisions)
            agent.store_reward(reward, done)
            ep_reward += reward

            if done:
                break

        # PPO update
        loss_info = agent.update(obs)

        # Collect episode summary
        summary = sim.summary()
        elapsed = time.time() - ep_start

        log_entry = {
            "episode": ep,
            "total_reward": ep_reward,
            "catch_rate": summary.get("catch_rate", 0),
            "unhandled_drops": summary.get("unhandled_drops", 0),
            "false_positive_rate": summary.get("false_positive_rate", 0),
            "p50_reconnect_ms": summary.get("p50_reconnect_ms", 0),
            "p95_reconnect_ms": summary.get("p95_reconnect_ms", 0),
            "p99_reconnect_ms": summary.get("p99_reconnect_ms", 0),
            "migration_bw_mb": summary.get("total_migration_bw_mb", 0),
            "pg_loss": loss_info.get("pg_loss", 0),
            "vf_loss": loss_info.get("vf_loss", 0),
            "entropy": loss_info.get("entropy", 0),
            "elapsed_sec": elapsed,
        }
        training_log.append(log_entry)

        if ep % 5 == 0:
            print(f"Ep {ep:3d} | reward={ep_reward:7.1f} | "
                  f"catch={summary.get('catch_rate',0):.2%} | "
                  f"p95={summary.get('p95_reconnect_ms',0):.0f}ms | "
                  f"fp={summary.get('false_positive_rate',0):.2%}")

        # Save best checkpoint
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(f"{checkpoint_dir}/best_model.pt")

        # Periodic checkpoint
        if ep % 10 == 0:
            agent.save(f"{checkpoint_dir}/ep_{ep:04d}.pt")

    # Save training log
    with open(log_file, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nTraining complete. Best reward: {best_reward:.1f}")
    print(f"Log saved to {log_file}")
    return training_log


if __name__ == "__main__":
    train()
