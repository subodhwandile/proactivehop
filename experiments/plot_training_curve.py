"""
plot_training_curve.py — Regenerate Fig. 2 (training curve) from persistent-mechanic log.

Reads results/training_log_persistent.json and saves:
  results/plots/fig2_training_curve.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def smooth(vals, w=10):
    kernel = np.ones(w) / w
    return np.convolve(vals, kernel, mode='valid')


def plot_training_curve(log_file: str = "results/training_log_persistent.json",
                        output_dir: str = "results/plots"):
    if not Path(log_file).exists():
        print(f"Log not found: {log_file}")
        return

    with open(log_file) as f:
        log = json.load(f)

    episodes  = [e["episode"]      for e in log]
    rewards   = [e["total_reward"] for e in log]
    catches   = [e["catch_rate"]   for e in log]

    best_catch = max(catches)
    best_ep    = episodes[catches.index(best_catch)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- Reward ---
    ax1.plot(episodes, rewards, alpha=0.25, color="#1f77b4", linewidth=1)
    w = min(10, len(rewards) // 5)
    if w > 1:
        ep_s = episodes[w - 1:]
        ax1.plot(ep_s, smooth(rewards, w), color="#1f77b4", linewidth=2,
                 label=f"Smoothed reward (w={w})")
    ax1.set_ylabel("Episode Reward", fontsize=11)
    ax1.set_title("PPO+GNN Training Convergence (persistent migration mechanic)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e5:.1f}×10⁵"))

    # --- Catch rate ---
    ax2.plot(episodes, catches, alpha=0.25, color="#2ca02c", linewidth=1)
    if w > 1:
        ax2.plot(ep_s, smooth(catches, w), color="#2ca02c", linewidth=2,
                 label="Smoothed catch rate")
    ax2.axhline(np.mean(catches[-50:]) if len(catches) >= 50 else np.mean(catches),
                color="#2ca02c", linestyle="--", alpha=0.5, linewidth=1,
                label=f"Final mean (last 50 ep)")
    ax2.scatter([best_ep], [best_catch], color="red", zorder=5, s=40,
                label=f"Best: {best_catch:.1%} @ ep {best_ep}")
    ax2.set_xlabel("Training Episode", fontsize=11)
    ax2.set_ylabel("Catch Rate", fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{output_dir}/fig2_training_curve.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close()

    n = len(log)
    print(f"\nSummary ({n} episodes):")
    print(f"  Best catch: {best_catch:.2%} at ep {best_ep}")
    print(f"  Final reward (last 10 mean): {np.mean(rewards[-10:]):.0f}")
    return best_catch, best_ep


if __name__ == "__main__":
    import sys
    log = sys.argv[1] if len(sys.argv) > 1 else "results/training_log_persistent.json"
    plot_training_curve(log)
