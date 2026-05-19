"""
run_experiments.py
------------------
Entry point. Loads data, trains all 5 models, prints results table,
and saves training curves + CSV.

Usage:
    /path/to/.venv/bin/python run_experiments.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_dataset
from train import run_all_experiments
from utils import plot_training_curves, print_results_table, save_results_csv


def main():
    print("=" * 60)
    print("  GNN for Drug–Drug Interaction Prediction")
    print("  AI-600 Deep Learning — Project #10")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────────────
    data, train_data, val_data, test_data, drug_ids = load_dataset(
        data_dir="data", seed=42
    )
    in_channels = data.x.shape[1]   # 2048 Morgan fingerprint bits
    print(f"\nNode feature dim: {in_channels}")

    # ── 2. Train all 5 models ──────────────────────────────────────────────
    all_results = run_all_experiments(
        train_data, val_data, test_data,
        in_channels=in_channels,
        epochs=150,
    )

    # ── 3. Report ──────────────────────────────────────────────────────────
    print_results_table(all_results)
    plot_training_curves(all_results, save_path="results/training_curves.png")
    save_results_csv(all_results, save_path="results/results.csv")

    print("\nDone. Outputs saved to results/")


if __name__ == "__main__":
    main()
