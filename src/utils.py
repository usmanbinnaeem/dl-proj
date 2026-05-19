"""
utils.py
--------
Plotting, display, and evaluation utilities.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import torch


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


# ── Calibration ───────────────────────────────────────────────────────────────

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Partitions [0,1] into n_bins equal-width bins; for each bin computes the
    gap between mean predicted confidence and empirical accuracy, weighted by
    bin fraction.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ── MC Dropout uncertainty ─────────────────────────────────────────────────────

def mc_dropout_predict(model, data, device, n_passes: int = 20):
    """
    Monte Carlo Dropout inference.

    Runs ``n_passes`` stochastic forward passes with dropout kept active
    (model.train() mode) to sample from the approximate posterior.

    Returns
    -------
    mean_probs : np.ndarray  shape [E]  — mean predicted probability
    std_probs  : np.ndarray  shape [E]  — epistemic uncertainty (std)
    """
    x                = data.x.to(device)
    edge_index       = data.edge_index.to(device)
    edge_label_index = data.edge_label_index.to(device)

    model.to(device)
    model.train()          # keep dropout active during inference

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(x, edge_index, edge_label_index)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    model.eval()

    stacked = np.stack(all_probs, axis=0)   # [n_passes, E]
    return stacked.mean(axis=0), stacked.std(axis=0)


# ── Ablation plots ────────────────────────────────────────────────────────────

def plot_ablation_results(ablations: dict, save_dir: str = "results"):
    """
    Plot 4 ablation groups (embed_dim, num_layers, features, neg_ratio)
    as a 2×2 bar-chart grid.

    ablations : dict mapping group_name → list of {config, test_auc, test_ap}
    """
    os.makedirs(save_dir, exist_ok=True)

    group_titles = {
        "embed_dim":  "A. Embedding Dimension",
        "num_layers": "B. Number of GCN Layers",
        "features":   "C. Node Feature Type",
        "neg_ratio":  "D. Negative Sampling Ratio",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (group, results) in zip(axes, ablations.items()):
        configs = [r["config"] for r in results]
        aucs    = [r["test_auc"] for r in results]

        # Human-readable axis labels
        labels = []
        for c in configs:
            c = (c.replace("embed_", "d=")
                  .replace("layers_", "L=")
                  .replace("features_", "")
                  .replace("neg_1:1", "1:1")
                  .replace("neg_5:1", "1:5"))
            labels.append(c)

        bars = ax.bar(labels, aucs, color="#4daf4a", edgecolor="black",
                      alpha=0.85, zorder=3)
        y_lo = max(0.50, min(aucs) - 0.05)
        ax.set_ylim(y_lo, min(1.01, max(aucs) + 0.06))
        ax.set_ylabel("Test ROC-AUC", fontsize=10)
        ax.set_title(group_titles.get(group, group), fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, zorder=0)

        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{auc:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Ablation Studies — GCN+MLP on BioSNAP DDI",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation plot saved → {path}")


def plot_ood_comparison(ood_results: dict,
                        save_path: str = "results/ood_comparison.png"):
    """
    Grouped bar chart: IID vs OOD performance for MLP and GCN+MLP.

    ood_results keys: 'MLP_iid', 'MLP_ood', 'GCN+MLP_iid', 'GCN+MLP_ood'
    """
    models   = ["MLP", "GCN+MLP"]
    iid_aucs = [ood_results[f"{m}_iid"] for m in models]
    ood_aucs = [ood_results[f"{m}_ood"] for m in models]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - width / 2, iid_aucs, width, label="IID (random split)",
                color="#4daf4a", edgecolor="black", alpha=0.85, zorder=3)
    b2 = ax.bar(x + width / 2, ood_aucs, width, label="OOD (cold-start)",
                color="#e41a1c", edgecolor="black", alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Test ROC-AUC", fontsize=11)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("IID vs. OOD Performance — Cold-Start Drug Pairs", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"OOD comparison plot saved → {save_path}")


def save_ablation_csv(ablations: dict, save_dir: str = "results"):
    """Save each ablation group to its own CSV file."""
    os.makedirs(save_dir, exist_ok=True)
    for group, rows in ablations.items():
        path = os.path.join(save_dir, f"ablation_{group}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["config", "val_auc", "test_auc", "test_ap"]
            )
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in r.items()
                })
        print(f"Ablation CSV saved → {path}")
