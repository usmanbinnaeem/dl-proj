"""
run_remaining.py
----------------
Resumes from saved results.csv (5 main models already trained).
Runs: ECE calibration, MC Dropout, OOD evaluation, and ablation studies.

Usage:
    .venv/bin/python3 run_remaining.py [--skip-ablations]
"""

import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from data_loader import load_dataset
from models import build_model
from train import run_all_experiments, get_device
from utils import (
    expected_calibration_error, mc_dropout_predict,
    plot_ablation_results, plot_ood_comparison,
    save_ablation_csv,
)
from ablation import run_all_ablations
from ood_eval import run_ood_evaluation, save_ood_csv


def retrain_gcn_mlp(train_data, val_data, test_data, in_channels, device):
    """Retrain GCN+MLP to get a live model object."""
    from train import run_experiment
    res = run_experiment(
        "GCN+MLP", train_data, val_data, test_data,
        in_channels=in_channels, epochs=150, device=device,
    )
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading dataset …")
    data, train_data, val_data, test_data, drug_ids = load_dataset(
        data_dir="data", seed=42
    )
    in_channels = data.x.shape[1]
    num_nodes   = data.num_nodes
    print(f"Nodes: {num_nodes}, Features: {in_channels}")

    # ── Load known IID results from CSV (already computed) ────────────────
    iid_results_stub = {}
    with open("results/results.csv") as f:
        for row in csv.DictReader(f):
            iid_results_stub[row["model"]] = {
                "best_val_auc": float(row["best_val_auc"]),
                "test_auc":     float(row["test_auc"]),
                "test_ap":      float(row["test_ap"]),
            }
    print("\nLoaded IID results:")
    for m, r in iid_results_stub.items():
        print(f"  {m:12s}  Test AUC={r['test_auc']:.4f}  Test AP={r['test_ap']:.4f}")

    # ── Retrain GCN+MLP to get a live model (needed for ECE + MC Dropout) ─
    print("\n" + "="*60)
    print("Retraining GCN+MLP for calibration/uncertainty analysis …")
    print("="*60)
    gcn_mlp_res = retrain_gcn_mlp(train_data, val_data, test_data,
                                   in_channels, device)
    best_model  = gcn_mlp_res["trained_model"].to(device)

    # ── ECE ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CALIBRATION & UNCERTAINTY  (GCN+MLP)")
    print("="*60)

    best_model.eval()
    with torch.no_grad():
        logits = best_model(
            test_data.x.to(device),
            test_data.edge_index.to(device),
            test_data.edge_label_index.to(device),
        )
    det_probs   = torch.sigmoid(logits).cpu().numpy()
    test_labels = test_data.edge_label.cpu().numpy()

    ece = expected_calibration_error(det_probs, test_labels)
    print(f"  ECE (deterministic)  : {ece:.4f}")

    mean_probs, std_probs = mc_dropout_predict(
        best_model, test_data, device, n_passes=20
    )
    avg_unc = std_probs.mean()
    print(f"  MC Dropout mean std  : {avg_unc:.4f}")

    calib_path = "results/calibration.csv"
    with open(calib_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["ECE", round(float(ece), 4)])
        writer.writerow(["MC_mean_uncertainty", round(float(avg_unc), 4)])
    print(f"  Saved → {calib_path}")

    # ── OOD evaluation ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("OOD EVALUATION")
    print("="*60)
    ood_results = run_ood_evaluation(
        data, iid_results_stub, in_channels,
        data_dir="data", ood_fraction=0.15, epochs=150,
    )
    save_ood_csv(ood_results, save_path="results/ood_results.csv")
    plot_ood_comparison(ood_results, save_path="results/ood_comparison.png")

    # ── Ablation studies ───────────────────────────────────────────────────
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

    print("\n" + "="*60)
    print("Done. All outputs saved to results/")
    print("="*60)


if __name__ == "__main__":
    main()
