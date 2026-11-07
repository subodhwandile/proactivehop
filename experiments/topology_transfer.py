"""
topology_transfer.py — Zero-shot topology generalisation.

Take the checkpoints trained on fat_tree (results/ablation_multiseed or
results/ablation) and evaluate them on *unseen* graph topologies
(random_geo, scale_free) with zero retraining.

Claim under test: a graph-structured policy is conditioned on structure, not
on node identities, so the same weights should transfer across cluster
architectures; a flat-MLP policy has no such structural prior.

    python experiments/topology_transfer.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import SessionStatus
from model.ppo_agent import PPOAgent
from model.mlp_agent import MLPAgent

EVAL_SEEDS = list(range(200, 220))   # held-out seeds, disjoint from prior evals
BASE_CFG = dict(
    n_servers=16, session_arrival_rate=10.0, session_duration_mean=60.0,
    base_load=0.55, spike_probability=0.08, spike_magnitude=0.35,
    enable_failures=True, max_steps=100,
)
CKPT_DIR_CANDIDATES = ["results/ablation_multiseed", "results/ablation"]


def run_episode_eval(agent, seed: int, topology: str) -> dict:
    cfg = SimConfig(**BASE_CFG, topology=topology, seed=seed)
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
            live = [nb for nb in sim.topo.neighbors(sv)
                    if sim.topo.get_node(nb).is_alive
                    and sim.topo.get_node(nb).load < 0.7]
            if not live:
                continue
            target = min(live, key=lambda nb: sim.topo.get_node(nb).load)
            for sid in list(sim.server_sessions[sv])[:20]:
                if sid in sim.sessions and sim.sessions[sid].status == SessionStatus.ACTIVE:
                    decisions[sid] = target
        obs, reward, done, _ = sim.step(decisions)
        if done:
            break
    agent._buffer_clear()
    return sim.summary()


def eval_agent(agent, topology: str) -> dict:
    cr = [run_episode_eval(agent, s, topology).get("catch_rate", 0)
          for s in EVAL_SEEDS]
    return {"catch_rates": cr,
            "catch_rate_mean": float(np.mean(cr)),
            "catch_rate_std": float(np.std(cr))}


def main():
    import torch
    torch.set_num_threads(4)
    results = {}
    for agent_name, cls, stem in [("GNN", PPOAgent, "GNN"),
                                  ("MLP", MLPAgent, "MLP")]:
        results[agent_name] = {}
        for init_seed in range(5):
            ckpt = None
            for d in CKPT_DIR_CANDIDATES:
                p = Path(d) / f"{stem}_init{init_seed}.pt"
                if p.exists():
                    ckpt = str(p); break
            if ckpt is None and init_seed == 0:
                p = Path("results/ablation") / f"best_{stem}.pt"
                if p.exists():
                    ckpt = str(p)
            if ckpt is None:
                continue
            agent = cls(n_servers=16, device="cpu")
            agent.load(ckpt); agent.model.eval()
            for topo in ["fat_tree", "random_geo", "scale_free"]:
                ev = eval_agent(agent, topo)
                results[agent_name].setdefault(topo, []).append(
                    {"init_seed": init_seed, **ev})
            print(f"evaluated {agent_name} init{init_seed} ({ckpt})", flush=True)

    print("\n" + "=" * 78)
    print("ZERO-SHOT TOPOLOGY TRANSFER (catch rate, 20 eval seeds, no retraining)")
    print("=" * 78)
    out = {}
    for agent_name, topos in results.items():
        out[agent_name] = {}
        for topo, runs in topos.items():
            m = np.array([r["catch_rate_mean"] for r in runs])
            out[agent_name][topo] = {
                "mean": float(m.mean()), "std": float(m.std()),
                "n_models": len(runs)}
            print(f"{agent_name} on {topo:11s}: {m.mean()*100:.2f}% ± {m.std()*100:.2f}%  (n={len(runs)} models)")
        # transfer retention = unseen-topo mean / fat_tree mean
        if "fat_tree" in out[agent_name] and out[agent_name]["fat_tree"]["mean"] > 0:
            base = out[agent_name]["fat_tree"]["mean"]
            for topo in out[agent_name]:
                if topo != "fat_tree":
                    ret = out[agent_name][topo]["mean"] / base
                    out[agent_name][topo]["retention"] = float(ret)
                    print(f"   {agent_name} retention {topo}: {ret:.2f}x")
    print("=" * 78)
    Path("results").mkdir(exist_ok=True)
    with open("results/topology_transfer.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/topology_transfer.json")


if __name__ == "__main__":
    main()
