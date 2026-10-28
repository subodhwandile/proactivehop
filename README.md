# ProactiveHop

**Proactive Session Migration via Graph Neural Networks and Reinforcement Learning**

Paper: `paper/paper.pdf`

## Overview

ProactiveHop pre-migrates stateful real-time sessions *before* server failures using a PPO agent whose observation is a GNN encoding of the live cluster topology. The GNN detects rising failure signals across the 2-hop neighborhood and drains sessions proactively, achieving **55.6% pre-migration rate** vs 19.3% for threshold heuristics in a controlled crash scenario.

## Key Results

| Experiment | Result |
|---|---|
| Pre-migration vs Threshold (failure scenario) | **55.6% vs 19.3% (2.9×)** |
| GNN vs flat MLP catch rate | **2.3× improvement** |
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

## Citation

```
@article{wandile2025proactivehop,
  title   = {ProactiveHop: Topology-Aware Proactive Session Migration via GNN and PPO},
  author  = {Wandile, Subodh},
  year    = {2025}
}
```
