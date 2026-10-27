"""
lstm_ablation_multiseed.py — LSTM catch-rate ablation across 5 training seeds.

Trains 5 independent LSTMs (different numpy/torch seeds for both data collection
and model initialization), evaluates each on the same 20 held-out seeds used
by the GNN/MLP ablation, and reports mean±std comparable to Table 3.

Design mirrors ablation_multiseed.py but uses supervised LSTM training.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import torch
from pathlib import Path
from scipy import stats

from sim.cluster import ClusterSimulator, SimConfig
from sim.session import SessionStatus
from baselines.lstm_baseline import (
    LSTMLoadPredictor, LSTMPolicy,
    collect_training_data, MIGRATE_THRESHOLD,
    WINDOW_SIZE, FEAT_DIM, HIDDEN_DIM, N_LAYERS,
)
import torch.nn as nn
import torch.optim as optim


TRAIN_INIT_SEEDS = [0, 1, 2, 3, 4]
EVAL_SEEDS       = list(range(100, 120))
N_COLLECT_EPS    = 50       # supervised data collection episodes per seed (fast)
TRAIN_EPOCHS     = 15
BATCH_SIZE       = 256
CHECKPOINT_DIR   = "results/lstm_ablation"

BASE_CFG = dict(
    topology="fat_tree", n_servers=16, session_arrival_rate=10.0,
    session_duration_mean=60.0, base_load=0.55, spike_probability=0.08,
    spike_magnitude=0.35, enable_failures=True, max_steps=100,
)


def train_lstm_seed(init_seed: int) -> str:
    """Train one LSTM from scratch with given seed. Returns saved model path."""
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    save_path = f"{CHECKPOINT_DIR}/lstm_init{init_seed}.pt"
    print(f"  Collecting data (seed={init_seed}, {N_COLLECT_EPS} eps)...")
    X, y = collect_training_data(n_episodes=N_COLLECT_EPS, seed=init_seed)
    print(f"  {len(X):,} samples. Training {TRAIN_EPOCHS} epochs...")

    model = LSTMLoadPredictor(FEAT_DIM, HIDDEN_DIM, N_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()
    dataset   = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    loader    = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                            shuffle=True)

    for epoch in range(TRAIN_EPOCHS):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % 5 == 0:
            avg = total / len(X)
            print(f"    epoch {epoch+1}/{TRAIN_EPOCHS}  MSE={avg:.5f}")

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path


def evaluate_lstm(model_path: str) -> dict:
    """Evaluate saved LSTM on all EVAL_SEEDS, return catch rates."""
    policy = LSTMPolicy(model_path=model_path, threshold=MIGRATE_THRESHOLD)
    cr_list, fp_list = [], []
    for seed in EVAL_SEEDS:
        cfg = SimConfig(**BASE_CFG, seed=seed)
        sim = ClusterSimulator(cfg)
        obs = sim.reset()
        policy.reset()
        for _ in range(cfg.max_steps):
            decisions = policy.select_action(obs, sim)
            obs, _, done, _ = sim.step(decisions)
            if done:
                break
        s = sim.summary()
        cr_list.append(s.get("catch_rate", 0))
        fp_list.append(s.get("false_positive_rate", 0))
    return {
        "catch_rates":     cr_list,
        "catch_rate_mean": float(np.mean(cr_list)),
        "catch_rate_std":  float(np.std(cr_list)),
        "fp_rate_mean":    float(np.mean(fp_list)),
    }


def run_lstm_ablation():
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    per_seed_means = []
    per_seed_results = []

    for i, init_seed in enumerate(TRAIN_INIT_SEEDS):
        print(f"\n--- LSTM init seed {init_seed} ({i+1}/{len(TRAIN_INIT_SEEDS)}) ---")
        model_path = train_lstm_seed(init_seed)
        eval_res = evaluate_lstm(model_path)
        per_seed_means.append(eval_res["catch_rate_mean"])
        per_seed_results.append(eval_res)
        print(f"  catch_rate={eval_res['catch_rate_mean']:.3f}±{eval_res['catch_rate_std']:.3f}")

    means = np.array(per_seed_means)
    print("\n" + "=" * 60)
    print("LSTM 5-SEED ABLATION RESULTS")
    print("=" * 60)
    print(f"Per-seed means: {[f'{v*100:.2f}%' for v in means]}")
    print(f"Overall mean:   {means.mean()*100:.2f}% ± {means.std()*100:.2f}%")
    print("=" * 60)

    result = {
        "lstm": {
            "per_seed_means": list(means),
            "mean": float(means.mean()),
            "std":  float(means.std()),
        },
        "init_seeds": TRAIN_INIT_SEEDS,
        "eval_seeds": EVAL_SEEDS,
        "n_collect_episodes": N_COLLECT_EPS,
    }
    out = f"{CHECKPOINT_DIR}/lstm_ablation_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out}")
    return result


if __name__ == "__main__":
    run_lstm_ablation()
