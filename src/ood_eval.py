"""
ood_eval.py
-----------
Out-of-distribution (cold-start / emerging drugs) evaluation.

Protocol
--------
1. Randomly hold out 15% of drug nodes (≈ 200 drugs).
2. Training graph  : only edges whose BOTH endpoints are seen drugs.
3. Context graph   : seen–seen edges + seen–held edges (used for
                     inductive message passing at test time).
4. OOD test edges  : edges whose BOTH endpoints are held-out drugs
                     (positives drawn directly from the dataset;
                      negatives sampled from held-out × held-out pairs).
5. Inference       : GNN runs on the context graph so held-out nodes
                     receive one-hop messages from seen neighbours —
                     simulating the clinical scenario of a new drug
                     entering an existing interaction network.

Hypothesis: MLP performance drops less on OOD (it uses only drug-intrinsic
Morgan fingerprints) while GNN models that rely heavily on graph topology
will show a larger degradation.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model
from train import get_device
from utils import mc_dropout_predict


# ── OOD split ─────────────────────────────────────────────────────────────────

def make_ood_split(data: Data, ood_fraction: float = 0.15, seed: int = 42):
    """
    Create a cold-start node split.

    Returns
    -------
    train_data      : Data  (link-prediction split on seen–seen edges)
    val_data        : Data  (link-prediction split on seen–seen edges)
    ood_test_data   : Data  (context graph + held-out test labels)
    held_out_mask   : BoolTensor [N]  — True for held-out nodes
    """
    torch.manual_seed(seed)
    N     = data.num_nodes
    perm  = torch.randperm(N)
    n_ood = int(N * ood_fraction)

    held_out_set  = set(perm[:n_ood].tolist())
    held_out_mask = torch.zeros(N, dtype=torch.bool)
    held_out_mask[list(held_out_set)] = True

    src, dst = data.edge_index[0], data.edge_index[1]
    src_held = held_out_mask[src]
    dst_held = held_out_mask[dst]

    # ── Training graph: only seen–seen edges ─────────────────────────────
    seen_seen_mask      = (~src_held) & (~dst_held)
    seen_seen_edge_idx  = data.edge_index[:, seen_seen_mask]

    seen_data           = Data(x=data.x, edge_index=seen_seen_edge_idx)
    seen_data.num_nodes = N   # keep full node count for consistent indexing

    torch.manual_seed(seed)
    splitter = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, _ = splitter(seen_data)

    # ── Context graph: seen–seen + seen–held (for inductive embedding) ───
    held_held_mask     = src_held & dst_held
    context_edge_idx   = data.edge_index[:, ~held_held_mask]  # exclude test targets

    # ── OOD test positives: held–held edges (undirected: keep a→b, a<b) ─
    test_pos = data.edge_index[:, held_held_mask]
    keep_dir = test_pos[0] < test_pos[1]
    test_pos = test_pos[:, keep_dir]
    n_pos    = test_pos.shape[1]

    if n_pos == 0:
        raise RuntimeError(
            "No held–held test edges found. "
            "Try increasing ood_fraction or using a denser dataset."
        )

    # ── OOD test negatives: random held–held pairs not in test_pos set ──
    held_out_tensor = perm[:n_ood]
    pos_set         = set(map(tuple, test_pos.t().tolist()))
    neg_src_list, neg_dst_list = [], []
    rng = torch.Generator()
    rng.manual_seed(seed + 1)
    attempts = 0
    while len(neg_src_list) < n_pos and attempts < n_pos * 20:
        a = held_out_tensor[torch.randint(n_ood, (1,), generator=rng).item()].item()
        b = held_out_tensor[torch.randint(n_ood, (1,), generator=rng).item()].item()
        attempts += 1
        if a == b:
            continue
        if (min(a, b), max(a, b)) in pos_set:
            continue
        neg_src_list.append(a)
        neg_dst_list.append(b)

    # Pad with what we have if we ran out of unique pairs
    while len(neg_src_list) < n_pos:
        idx = len(neg_src_list) % len(neg_src_list) if neg_src_list else 0
        neg_src_list.append(neg_src_list[idx] if neg_src_list else held_out_tensor[0].item())
        neg_dst_list.append(neg_dst_list[idx] if neg_dst_list else held_out_tensor[1 % n_ood].item())

    test_neg = torch.tensor([neg_src_list[:n_pos], neg_dst_list[:n_pos]])

    test_eli    = torch.cat([test_pos, test_neg], dim=1)
    test_labels = torch.cat([torch.ones(n_pos), torch.zeros(n_pos)])

    ood_test_data                   = Data(x=data.x, edge_index=context_edge_idx)
    ood_test_data.num_nodes         = N
    ood_test_data.edge_label_index  = test_eli
    ood_test_data.edge_label        = test_labels

    print(
        f"OOD split: {N - n_ood} seen drugs, {n_ood} held-out drugs, "
        f"{n_pos} OOD test pairs (pos+neg each)"
    )
    return train_data, val_data, ood_test_data, held_out_mask


# ── Training helper ────────────────────────────────────────────────────────────

def _train_model(model_name: str, train_data, val_data, in_channels: int,
                 epochs: int = 150, patience: int = 20) -> tuple:
    """Train a model on OOD train split; returns (model, device)."""
    device    = get_device()
    model     = build_model(model_name, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_val_auc = 0.0
    best_state   = None
    no_improve   = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(
            train_data.x.to(device),
            train_data.edge_index.to(device),
            train_data.edge_label_index.to(device),
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, train_data.edge_label.float().to(device)
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(
                val_data.x.to(device),
                val_data.edge_index.to(device),
                val_data.edge_label_index.to(device),
            )
        val_auc = roc_auc_score(
            val_data.edge_label.cpu().numpy(),
            torch.sigmoid(v_logits).cpu().numpy(),
        )
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    [{model_name}] Early stop at epoch {epoch} "
                      f"({time.time()-t0:.0f}s)")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, device


# ── OOD evaluation ────────────────────────────────────────────────────────────

def run_ood_evaluation(data: Data, iid_results: dict,
                       in_channels: int, data_dir: str = "data",
                       ood_fraction: float = 0.15,
                       epochs: int = 150,
                       iid_test_data=None) -> dict:
    """
    Compare IID (random split) vs OOD (cold-start) for MLP and GCN+MLP.
    Also computes MC Dropout uncertainty on both IID and OOD test pairs
    for GCN+MLP, exposing whether uncertainty is higher on OOD pairs.

    Parameters
    ----------
    data          : full PyG Data object
    iid_results   : dict from run_all_experiments (has IID AUCs)
    in_channels   : node feature dim
    iid_test_data : PyG Data for IID test set (needed for uncertainty comparison)

    Returns
    -------
    dict with keys per model + 'uncertainty' sub-dict
    """
    print("\n" + "=" * 60)
    print("OOD EVALUATION  (cold-start, 15% held-out nodes)")
    print("=" * 60)

    train_data, val_data, ood_test_data, held_mask = make_ood_split(
        data, ood_fraction=ood_fraction
    )

    results = {}
    details = {}

    for model_name in ("MLP", "GCN+MLP"):
        iid_auc = iid_results[model_name]["test_auc"]

        print(f"\n  Training {model_name} on OOD split …")
        model, device = _train_model(
            model_name, train_data, val_data, in_channels, epochs=epochs
        )

        model.eval()
        with torch.no_grad():
            logits = model(
                ood_test_data.x.to(device),
                ood_test_data.edge_index.to(device),
                ood_test_data.edge_label_index.to(device),
            )
        ood_probs  = torch.sigmoid(logits).cpu().numpy()
        ood_labels = ood_test_data.edge_label.cpu().numpy()
        ood_auc    = roc_auc_score(ood_labels, ood_probs)
        ood_ap     = average_precision_score(ood_labels, ood_probs)

        print(f"  {model_name:10s}  IID AUC={iid_auc:.4f}  "
              f"OOD AUC={ood_auc:.4f}  Δ={ood_auc - iid_auc:+.4f}")

        results[f"{model_name}_iid"] = iid_auc
        results[f"{model_name}_ood"] = float(ood_auc)
        details[model_name] = {
            "iid_auc": iid_auc,
            "ood_auc": float(ood_auc),
            "ood_ap":  float(ood_ap),
            "delta":   float(ood_auc - iid_auc),
        }

        # ── MC Dropout uncertainty: IID vs OOD (GCN+MLP only) ────────────
        if model_name == "GCN+MLP" and iid_test_data is not None:
            device = get_device()
            model.to(device)
            print(f"\n  Computing MC Dropout uncertainty (20 passes) …")
            _, iid_stds = mc_dropout_predict(model, iid_test_data, device, n_passes=20)
            _, ood_stds = mc_dropout_predict(model, ood_test_data,  device, n_passes=20)
            print(f"    IID mean uncertainty : {iid_stds.mean():.4f}")
            print(f"    OOD mean uncertainty : {ood_stds.mean():.4f}")
            print(f"    Ratio OOD/IID        : {ood_stds.mean() / (iid_stds.mean() + 1e-9):.2f}x")
            results["uncertainty"] = {
                "iid_mean_std": float(iid_stds.mean()),
                "ood_mean_std": float(ood_stds.mean()),
                "iid_stds":     iid_stds,
                "ood_stds":     ood_stds,
            }

    results["details"] = details
    return results


def save_ood_csv(ood_results: dict, save_path: str = "results/ood_results.csv"):
    """Save OOD evaluation summary to CSV."""
    import csv
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    details = ood_results.get("details", {})
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "iid_auc", "ood_auc", "ood_ap", "delta"]
        )
        writer.writeheader()
        for model_name, d in details.items():
            writer.writerow({"model": model_name, **{
                k: round(v, 4) for k, v in d.items()
            }})
    print(f"OOD results saved → {save_path}")
