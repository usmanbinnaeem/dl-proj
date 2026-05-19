"""
ablation.py
-----------
Ablation studies for GCN+MLP (best model) on BioSNAP ChCh-Miner DDI.

Groups
------
  A. Embedding dimension  : {32, 64, 128, 256}
  B. Number of GCN layers : {1, 2, 3}
  C. Node feature type    : Morgan fingerprints  vs.  learnable embeddings
  D. Negative sampling    : 1:1  vs.  1:5
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
from models import FlexGCNEncoder, MLPDecoder, GNNModel, LearnableGCNMLP
from train import get_device
from data_loader import load_dataset


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_gcn_mlp(in_channels: int, hidden: int = 256,
                   embed: int = 64, num_layers: int = 2) -> torch.nn.Module:
    enc = FlexGCNEncoder(in_channels, hidden=hidden, out=embed,
                         num_layers=num_layers)
    dec = MLPDecoder(embed_dim=embed)
    return GNNModel(enc, dec)


def _train_and_eval(model, label: str, train_data, val_data, test_data,
                    epochs: int = 100, patience: int = 20) -> dict:
    """Train one model configuration and return a results dict."""
    device    = get_device()
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_val_auc = 0.0
    best_state   = None
    no_improve   = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── train step ─────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        x          = train_data.x.to(device)
        edge_index = train_data.edge_index.to(device)
        eli        = train_data.edge_label_index.to(device)
        labels     = train_data.edge_label.float().to(device)
        logits     = model(x, edge_index, eli)
        loss       = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        # ── validation step ────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            v_logits = model(
                val_data.x.to(device),
                val_data.edge_index.to(device),
                val_data.edge_label_index.to(device),
            )
            v_probs  = torch.sigmoid(v_logits).cpu().numpy()
        val_auc = roc_auc_score(val_data.edge_label.cpu().numpy(), v_probs)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # ── test evaluation ────────────────────────────────────────────────────
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        t_logits = model(
            test_data.x.to(device),
            test_data.edge_index.to(device),
            test_data.edge_label_index.to(device),
        )
        t_probs = torch.sigmoid(t_logits).cpu().numpy()
    t_labels = test_data.edge_label.cpu().numpy()
    test_auc = roc_auc_score(t_labels, t_probs)
    test_ap  = average_precision_score(t_labels, t_probs)

    elapsed = time.time() - t0
    print(f"    {label:30s}  AUC={test_auc:.4f}  AP={test_ap:.4f}  ({elapsed:.0f}s)")
    return {"config": label, "val_auc": best_val_auc,
            "test_auc": test_auc, "test_ap": test_ap}


# ── Ablation groups ────────────────────────────────────────────────────────────

def run_ablation_embed_dim(train_data, val_data, test_data, in_channels: int,
                           dims=(32, 64, 128, 256), epochs: int = 100) -> list:
    """Ablation A: vary output embedding dimension."""
    print("\n  Group A — Embedding Dimension")
    return [
        _train_and_eval(
            _build_gcn_mlp(in_channels, hidden=256, embed=d),
            f"embed_{d}", train_data, val_data, test_data, epochs,
        )
        for d in dims
    ]


def run_ablation_num_layers(train_data, val_data, test_data, in_channels: int,
                            layer_counts=(1, 2, 3), epochs: int = 100) -> list:
    """Ablation B: vary number of GCN message-passing layers."""
    print("\n  Group B — Number of GCN Layers")
    return [
        _train_and_eval(
            _build_gcn_mlp(in_channels, hidden=256, embed=64, num_layers=n),
            f"layers_{n}", train_data, val_data, test_data, epochs,
        )
        for n in layer_counts
    ]


def run_ablation_features(train_data, val_data, test_data, in_channels: int,
                          num_nodes: int, epochs: int = 100) -> list:
    """Ablation C: Morgan fingerprints vs. learnable node embeddings."""
    print("\n  Group C — Node Feature Type")
    results = []

    # Morgan fingerprints (standard GCN+MLP, embed=64)
    results.append(_train_and_eval(
        _build_gcn_mlp(in_channels, hidden=256, embed=64),
        "features_Morgan", train_data, val_data, test_data, epochs,
    ))

    # Learnable position embeddings (no molecular information)
    results.append(_train_and_eval(
        LearnableGCNMLP(num_nodes=num_nodes, hidden=256, out=64),
        "features_Learnable", train_data, val_data, test_data, epochs,
    ))
    return results


def run_ablation_neg_ratio(data_dir: str, in_channels: int,
                           ratios=(1.0, 5.0), epochs: int = 100) -> list:
    """Ablation D: negative sampling ratio (1:1 vs 1:5)."""
    print("\n  Group D — Negative Sampling Ratio")
    results = []
    for r in ratios:
        label = f"neg_{int(r)}:1"
        print(f"    Loading data with neg_sampling_ratio={r} …")
        _, tr, va, te, _ = load_dataset(data_dir=data_dir,
                                        neg_sampling_ratio=r)
        results.append(_train_and_eval(
            _build_gcn_mlp(in_channels, hidden=256, embed=64),
            label, tr, va, te, epochs,
        ))
    return results


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def run_all_ablations(train_data, val_data, test_data,
                      in_channels: int, num_nodes: int,
                      data_dir: str = "data", epochs: int = 100) -> dict:
    """Run all four ablation groups and return results dict."""
    print("\n" + "=" * 60)
    print("ABLATION STUDIES  (GCN+MLP, baseline: embed=64, L=2, Morgan, neg=1:1)")
    print("=" * 60)

    return {
        "embed_dim":  run_ablation_embed_dim(
            train_data, val_data, test_data, in_channels, epochs=epochs),
        "num_layers": run_ablation_num_layers(
            train_data, val_data, test_data, in_channels, epochs=epochs),
        "features":   run_ablation_features(
            train_data, val_data, test_data, in_channels, num_nodes, epochs=epochs),
        "neg_ratio":  run_ablation_neg_ratio(
            data_dir, in_channels, epochs=epochs),
    }
