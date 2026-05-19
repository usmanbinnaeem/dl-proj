"""
run_experiments.py
------------------
Full experiment pipeline for the final submission:
  1. Train all 5 main model configurations (IID, random split)
  2. Compute ECE calibration + MC Dropout uncertainty for best model
  3. OOD (cold-start) evaluation for MLP and GCN+MLP
  4. Ablation studies for GCN+MLP (embed dim, layers, features, neg ratio)

Usage:
    /path/to/.venv/bin/python run_experiments.py [--skip-ablations] [--skip-ood]
"""

import argparse
import csv
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from data_loader import load_dataset
from train import run_all_experiments
from utils import (
    plot_training_curves, print_results_table, save_results_csv,
    expected_calibration_error, mc_dropout_predict,
    plot_ablation_results, plot_ood_comparison, save_ablation_csv,
)
from ablation import run_all_ablations
from ood_eval import run_ood_evaluation, save_ood_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Skip ablation studies (saves ~60 min)")
    parser.add_argument("--skip-ood", action="store_true",
                        help="Skip OOD evaluation (saves ~10 min)")
    args = parser.parse_args()

    print("=" * 60)
    print("  GNN for Drug–Drug Interaction Prediction")
    print("  AI-600 Deep Learning — Project #10  (Final)")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────
    data, train_data, val_data, test_data, drug_ids = load_dataset(
        data_dir="data", seed=42
    )
    in_channels = data.x.shape[1]    # 2048 Morgan fingerprint bits
    num_nodes   = data.num_nodes      # 1338 drugs
    print(f"\nNode feature dim : {in_channels}")
    print(f"Number of nodes  : {num_nodes}")

    # ── 2. Train all 5 models (IID) ────────────────────────────────────────
    all_results = run_all_experiments(
        train_data, val_data, test_data,
        in_channels=in_channels,
        epochs=150,
    )
    print_results_table(all_results)
    plot_training_curves(all_results, save_path="results/training_curves.png")
    save_results_csv(all_results, save_path="results/results.csv")

    # ── 3. ECE + MC Dropout for best model (GCN+MLP) ──────────────────────
    print("\n" + "=" * 60)
    print("CALIBRATION & UNCERTAINTY  (GCN+MLP, best model)")
    print("=" * 60)

    import torch
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    best_model = all_results["GCN+MLP"]["trained_model"].to(device)

    # Deterministic predictions for ECE
    best_model.eval()
    with torch.no_grad():
        logits = best_model(
            test_data.x.to(device),
            test_data.edge_index.to(device),
            test_data.edge_label_index.to(device),
        )
    det_probs  = torch.sigmoid(logits).cpu().numpy()
    test_labels = test_data.edge_label.cpu().numpy()

    ece = expected_calibration_error(det_probs, test_labels)
    print(f"  ECE (GCN+MLP, deterministic) : {ece:.4f}")

    # MC Dropout uncertainty (20 stochastic passes)
    mean_probs, std_probs = mc_dropout_predict(
        best_model, test_data, device, n_passes=20
    )
    avg_uncertainty = std_probs.mean()
    print(f"  MC Dropout mean uncertainty  : {avg_uncertainty:.4f}")

    # Save calibration summary
    calib_path = "results/calibration.csv"
    with open(calib_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["ECE", round(float(ece), 4)])
        writer.writerow(["MC_mean_uncertainty", round(float(avg_uncertainty), 4)])
    print(f"  Calibration summary saved → {calib_path}")

    # ── 4. OOD evaluation ─────────────────────────────────────────────────
    if not args.skip_ood:
        ood_results = run_ood_evaluation(
            data, all_results, in_channels,
            data_dir="data", ood_fraction=0.15, epochs=150,
        )
        save_ood_csv(ood_results, save_path="results/ood_results.csv")
        plot_ood_comparison(ood_results, save_path="results/ood_comparison.png")
    else:
        print("\n[Skipped OOD evaluation]")

    # ── 5. Ablation studies ───────────────────────────────────────────────
    if not args.skip_ablations:
        ablations = run_all_ablations(
            train_data, val_data, test_data,
            in_channels=in_channels,
            num_nodes=num_nodes,
            data_dir="data",
            epochs=100,
        )
        save_ablation_csv(ablations, save_dir="results")
        plot_ablation_results(ablations, save_dir="results")
    else:
        print("\n[Skipped ablation studies]")

    print("\n" + "=" * 60)
    print("All done. Outputs saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
