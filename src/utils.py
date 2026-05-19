"""
utils.py
--------
Plotting and display utilities.
"""

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np


MODEL_COLORS = {
    "MLP":     "#e41a1c",
    "GCN+Dot": "#377eb8",
    "GCN+MLP": "#4daf4a",
    "GIN+Dot": "#ff7f00",
    "GIN+MLP": "#984ea3",
}


def plot_training_curves(all_results: dict, save_path: str = "results/training_curves.png"):
    """
    Plot validation AUC curves for all model configurations on a single figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, res in all_results.items():
        color = MODEL_COLORS.get(name, "gray")
        hist  = res["history"]
        epochs = range(1, len(hist["val_auc"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], label=name, color=color, linewidth=1.8)
        axes[1].plot(epochs, hist["val_auc"],    label=name, color=color, linewidth=1.8)

    axes[0].set_title("Training Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation ROC-AUC", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.suptitle(
        "GNN for Drug–Drug Interaction Prediction\n"
        "BioSNAP ChCh-Miner Dataset · Morgan Fingerprint Features",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {save_path}")


def print_results_table(all_results: dict):
    """Print a formatted results table to stdout."""
    header = f"{'Model':<12} {'Val AUC':>9} {'Test AUC':>10} {'Test AP':>9}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for name, res in all_results.items():
        print(
            f"{name:<12} "
            f"{res['best_val_auc']:>9.4f} "
            f"{res['test_auc']:>10.4f} "
            f"{res['test_ap']:>9.4f}"
        )
    print(sep)

    best = max(all_results.items(), key=lambda kv: kv[1]["test_auc"])
    print(f"\nBest model: {best[0]}  (Test AUC = {best[1]['test_auc']:.4f})\n")


def save_results_csv(all_results: dict, save_path: str = "results/results.csv"):
    """Save the results table as a CSV."""
    import csv
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "best_val_auc", "test_auc", "test_ap"]
        )
        writer.writeheader()
        for name, res in all_results.items():
            writer.writerow({
                "model":        name,
                "best_val_auc": round(res["best_val_auc"], 4),
                "test_auc":     round(res["test_auc"],     4),
                "test_ap":      round(res["test_ap"],      4),
            })
    print(f"Results CSV saved → {save_path}")
