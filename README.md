# ProactiveHop

**Proactive Session Migration via Graph Neural Networks and Reinforcement Learning**

Paper: `paper/paper.pdf`

## Overview

ProactiveHop pre-migrates stateful real-time sessions *before* server failures using a PPO agent whose observation is a GNN encoding of the live cluster topology. The GNN detects rising load signals across the 2-hop neighborhood and drains sessions proactively, achieving **86.0% pre-migration rate** (median 100%, 13/20 seeds) vs 26.6% for threshold heuristics in a controlled crash scenario (3.2×, Wilcoxon p<0.001, Holm-corrected). PPO+MLP achieves the highest crash-scenario rate (98.0%), though the difference from GNN is not significant (p=0.14). The primary proven GNN advantage is in i.i.d. ablation (1.8× higher catch rate vs MLP, p=0.0006) and near-linear inference scaling.

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

Each results file was produced at a specific commit:

| File | Commit | Notes |
|------|--------|-------|
| `results/failure_scenario.json` (Table 2) | `8144338` | Fixed 25-session denominator, Wilcoxon+Holm |
| `results/ablation_multiseed_speculative/ablation_multiseed_results.json` (Table 3) | `2974d8c` | After RNG fix (`8d8d384`) |
| `results/scale_experiment.json` (Table 4) | `60f02bb` | Before `_create_session` RNG fix |
| `results/sensitivity.json` (Table 5) | `b840ad9` | Before `_create_session` RNG fix |

To exactly reproduce Tables 4–5, check out the respective commit before running `experiments/scale_experiment.py` or `experiments/sensitivity.py`.



```
@article{wandile2025proactivehop,
  title   = {ProactiveHop: Topology-Aware Proactive Session Migration via GNN and PPO},
  author  = {Wandile, Subodh},
  year    = {2025}
}
```
