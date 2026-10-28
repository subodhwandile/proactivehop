# ProactiveHop

**Proactive Session Migration via Graph Neural Networks and Reinforcement Learning**

Paper: `paper/paper.pdf`

## Overview

ProactiveHop pre-migrates stateful real-time sessions *before* server failures using a PPO agent whose observation is a GNN encoding of the live cluster topology. In a controlled crash scenario, PPO+GNN achieves **86.0% pre-migration rate** (median 100%, 13/20 seeds) vs 26.6% for threshold heuristics (3.2×, Wilcoxon p<0.001, Holm-corrected); PPO+MLP achieves the highest crash-scenario rate (98.0%), though the difference from GNN is not significant (p=0.14). The primary proven GNN advantage is in i.i.d. ablation (1.8× higher catch rate vs MLP, p=0.0006) and near-linear inference scaling.

## Key Results

| Experiment | Result |
|---|---|
| Pre-migration vs Threshold (failure scenario) | **86.0% vs 26.6% (3.2×, Wilcoxon p<0.001, Holm-corrected)** |
| GNN vs flat MLP catch rate | **1.8× improvement** |
| Inference scaling (N=16 → N=64) | **1.1× overhead for 4× cluster** |

## Structure

```
sim/          — Discrete-event cluster simulator (fat-tree, failures, sessions)
model/        — GNN+PPO agent (ppo_agent.py) and MLP ablation (mlp_agent.py)
baselines/    — No-Migration, Threshold-70%, p_drop heuristic, Random policies
experiments/  — train.py, ablation.py, scale_experiment.py, failure_scenario.py
results/      — Plots, JSONs, trained checkpoints
paper/        — LaTeX source + compiled PDF
```

## Quickstart

```bash
pip install torch torch-geometric numpy matplotlib

# Train
python experiments/train.py

# Failure scenario (smoking-gun experiment)
python experiments/failure_scenario.py

# Ablation: GNN vs MLP
python experiments/ablation.py

# Scale experiment: N=16/32/64
python experiments/scale_experiment.py
```

## Reproducibility — Commit provenance

Each results file was produced at a specific commit. Commits `13c3c56`, `1b1033d`, and `60f02bb` predate the `_create_session` RNG fix (`8d8d384`) and the `hash()` fix (`f80a1b5`), so re-running at those commits will not give bit-identical results. To reproduce Tables 3–5 with the current bit-deterministic simulator, run the corresponding experiment scripts at HEAD.

| File | Commit | Table | Notes |
|------|--------|-------|-------|
| `results/failure_scenario.json` | `f80a1b5` | Table 2 | Fixed 25-session denominator, Wilcoxon+Holm |
| `results/ablation_multiseed/ablation_multiseed_results.json` | `13c3c56` | Table 3 rows 1–2 | Before RNG fix |
| `results/lstm_ablation/`, `results/edge_ablation/` | `1b1033d` | Table 3 rows 3–5 | Before RNG fix |
| `results/scale_experiment.json` | `60f02bb` | Table 4 | Before RNG fix |
| `results/sensitivity.json` | `13c3c56` | Table 5 | Before RNG fix |



```
@article{wandile2025proactivehop,
  title   = {ProactiveHop: Topology-Aware Proactive Session Migration via GNN and PPO},
  author  = {Wandile, Subodh},
  year    = {2025}
}
```
