"""
cascade_ablation.py — Multi-seed ablation under topology-correlated failures.

Trains GNN, MLP and NoEdge-GNN under the cascade failure model
(cascade_failures=True: a failure raises a hidden common-cause hazard at
graph neighbours) and evaluates catch rate on held-out seeds.

This is the experiment the original evaluation lacked: a scenario where the
failure signal is genuinely relational, so topology-aware observation has a
mechanism to exploit.

Run one worker per (agent, init_seed) for parallelism:
    python experiments/cascade_ablation.py --agent GNN    --init-seed 0
    python experiments/cascade_ablation.py --agent MLP    --init-seed 0
    python experiments/cascade_ablation.py --agent NoEdge --init-seed 0
Aggregate afterwards:
    python experiments/cascade_ablation.py --aggregate
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import SessionStatus
from model.ppo_agent import PPOAgent
from model.mlp_agent import MLPAgent

N_TRAIN_EPISODES = 500
EVAL_SEEDS       = list(range(100, 120))   # 20 held-out seeds
TRAIN_SEEDS      = [42, 43, 44, 45, 46]
CHECKPOINT_DIR   = "results/cascade_ablation"

# Cascade regime — overridable via CLI for the strong-cascade variant.
CASCADE_PARAMS = dict(
    cascade_boost=0.03,
    cascade_duration=15,
    cascade_hops=1,
    cascade_hop_decay=0.5,
)

BASE_CFG = dict(
    topology="fat_tree", n_servers=16, session_arrival_rate=10.0,
    session_duration_mean=60.0, base_load=0.55, spike_probability=0.08,
    spike_magnitude=0.35, enable_failures=True, max_steps=100,
    server_mtbf=150,
    cascade_failures=True,            # <-- the new failure mode
)


def make_empty_edge_obs(obs: dict) -> dict:
    out = dict(obs)
    out["edge_index"]    = np.zeros((2, 0), dtype=np.int64)
    out["edge_features"] = np.zeros((0, 4), dtype=np.float32)
    return out


EVAC_WINDOW = 15   # steps before death over which evacuation is credited


def run_episode(agent, seed: int, strip_edges: bool, training: bool) -> dict:
    cfg = SimConfig(**BASE_CFG, **CASCADE_PARAMS, seed=seed)
    sim = ClusterSimulator(cfg)
    obs = sim.reset()
    if strip_edges:
        obs = make_empty_edge_obs(obs)
    # Snapshot history for evacuation metric: step -> {server: {sids}}
    sess_hist: list[dict] = []
    evac_rates: list[float] = []
    for _ in range(cfg.max_steps):
        sess_hist.append({i: set(sim.server_sessions[i]) for i in range(sim.n)})
        if len(sess_hist) > EVAC_WINDOW + 1:
            sess_hist.pop(0)
        alive_before = {i for i in range(sim.n) if sim.topo.get_node(i).is_alive}
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
        if strip_edges:
            obs = make_empty_edge_obs(obs)
        # Evacuation metric: for each node that died this step, what fraction
        # of its sessions (as of EVAC_WINDOW steps ago) are no longer on it?
        alive_after = {i for i in range(sim.n) if sim.topo.get_node(i).is_alive}
        for dead in alive_before - alive_after:
            base = sess_hist[0].get(dead, set())
            if base:
                evacuated = sum(1 for sid in base
                                if sid in sim.sessions
                                and sim.sessions[sid].current_server != dead)
                evac_rates.append(evacuated / len(base))
        if training:
            agent.store_reward(reward, done)
        if done:
            break
    if training:
        agent.update(obs)
    else:
        agent._buffer_clear()
    s = sim.summary()
    s["evac_rate"] = float(np.mean(evac_rates)) if evac_rates else 0.0
    s["n_server_deaths"] = len(evac_rates)
    return s


def evaluate(agent, strip_edges: bool) -> dict:
    cr_list, fp_list, drop_list, evac_list, deaths_list = [], [], [], [], []
    for seed in EVAL_SEEDS:
        s = run_episode(agent, seed, strip_edges=strip_edges, training=False)
        cr_list.append(s.get("catch_rate", 0))
        fp_list.append(s.get("false_positive_rate", 0))
        drop_list.append(s.get("total_drops", 0))
        evac_list.append(s.get("evac_rate", 0))
        deaths_list.append(s.get("n_server_deaths", 0))
    return {
        "catch_rates": cr_list,
        "catch_rate_mean": float(np.mean(cr_list)),
        "catch_rate_std": float(np.std(cr_list)),
        "fp_rate_mean": float(np.mean(fp_list)),
        "total_drops_mean": float(np.mean(drop_list)),
        "evac_rate_mean": float(np.mean(evac_list)),
        "evac_rate_std": float(np.std(evac_list)),
        "deaths_mean": float(np.mean(deaths_list)),
    }


def train_worker(agent_name: str, init_seed: int, out_dir: str):
    torch.set_num_threads(1)
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    strip = (agent_name == "NoEdge")
    cls = MLPAgent if agent_name == "MLP" else PPOAgent
    agent = cls(n_servers=16, lr=3e-4, entropy_coef=0.05, n_epochs=4,
                device="cpu")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{out_dir}/{agent_name}_init{init_seed}.pt"
    best_cr = -np.inf
    for ep in range(N_TRAIN_EPISODES):
        seed = TRAIN_SEEDS[ep % len(TRAIN_SEEDS)]
        s = run_episode(agent, seed, strip_edges=strip, training=True)
        cr = s.get("catch_rate", 0)
        if cr > best_cr:
            best_cr = cr
            agent.save(save_path)
        if (ep + 1) % 100 == 0:
            print(f"  [{agent_name}/init{init_seed}] ep {ep+1}/{N_TRAIN_EPISODES} best_cr={best_cr:.4f}", flush=True)

    agent.load(save_path)
    agent.model.eval()
    ev = evaluate(agent, strip_edges=strip)
    out = {"agent": agent_name, "init_seed": init_seed,
           "best_train_cr": float(best_cr), **ev}
    with open(f"{out_dir}/{agent_name}_init{init_seed}_eval.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"DONE {agent_name} init{init_seed}: catch={ev['catch_rate_mean']:.4f}±{ev['catch_rate_std']:.4f}", flush=True)


def aggregate(out_dir: str):
    rows = {}
    for p in sorted(Path(out_dir).glob("*_eval.json")):
        d = json.load(open(p))
        rows.setdefault(d["agent"], []).append(d)
    print("\n" + "=" * 78)
    print("CASCADE-MODE ABLATION (catch rate + evac rate, 20 held-out eval seeds)")
    print("=" * 78)
    means, evac = {}, {}
    for name, runs in sorted(rows.items()):
        m = np.array([r["catch_rate_mean"] for r in runs])
        e = np.array([r["evac_rate_mean"] for r in runs])
        means[name] = m; evac[name] = e
        print(f"{name:8s}: catch={m.mean()*100:.2f}%±{m.std()*100:.2f}%  evac={e.mean()*100:.1f}%±{e.std()*100:.1f}%  deaths={np.mean([r.get('deaths_mean',0) for r in runs]):.1f}")
    print("-" * 78)
    for a, b in [("GNN","MLP"),("GNN","NoEdge")]:
        if a in means and b in means:
            t1, p1 = stats.ttest_ind(means[a], means[b], equal_var=False)
            t2, p2 = stats.ttest_ind(evac[a], evac[b], equal_var=False)
            print(f"{a} vs {b:6s}: catch Welch t={t1:.2f}, p={p1:.4f} | evac t={t2:.2f}, p={p2:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["GNN", "MLP", "NoEdge"])
    ap.add_argument("--init-seed", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--boost", type=float, default=None)
    ap.add_argument("--duration", type=int, default=None)
    ap.add_argument("--hops", type=int, default=None)
    ap.add_argument("--hop-decay", type=float, default=None)
    args = ap.parse_args()
    if args.boost is not None:
        CASCADE_PARAMS["cascade_boost"] = args.boost
    if args.duration is not None:
        CASCADE_PARAMS["cascade_duration"] = args.duration
    if args.hops is not None:
        CASCADE_PARAMS["cascade_hops"] = args.hops
    if args.hop_decay is not None:
        CASCADE_PARAMS["cascade_hop_decay"] = args.hop_decay
    if args.aggregate:
        aggregate(args.out_dir)
    else:
        train_worker(args.agent, args.init_seed, args.out_dir)
