"""
train.py
--------
Training loop, evaluation (ROC-AUC + Average Precision),
and experiment runner for all five DDI model configurations.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling

sys.path.insert(0, os.path.dirname(__file__))
from models import ALL_MODELS, build_model

# ── device ─────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── training helpers ───────────────────────────────────────────────────────────

def train_epoch(model, train_data, optimizer, device):
    model.train()
    optimizer.zero_grad()

    x          = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)

    # Use the pre-split edge labels (positives + negatives from RandomLinkSplit)
    edge_label_index = train_data.edge_label_index.to(device)
    edge_label       = train_data.edge_label.float().to(device)

    logits = model(x, edge_index, edge_label_index)
    loss   = F.binary_cross_entropy_with_logits(logits, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device):
    """
    Evaluate on data.edge_label_index / data.edge_label.
    Returns (roc_auc, average_precision).
    """
    model.eval()
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_label_index = data.edge_label_index.to(device)

    logits = model(x, edge_index, edge_label_index)
    probs  = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.cpu().numpy()

    auc = roc_auc_score(labels, probs)
    ap  = average_precision_score(labels, probs)
    return auc, ap


# ── experiment runner ──────────────────────────────────────────────────────────

def run_experiment(
    model_name: str,
    train_data,
    val_data,
    test_data,
    in_channels: int = 2048,
    epochs: int = 150,
    lr: float = 1e-3,
    patience: int = 20,
    device=None,
):
    """
    Train one model configuration and return a results dict.
    Includes early stopping on val AUC.
    """
    if device is None:
        device = get_device()

    model = build_model(model_name, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_val_auc = 0.0
    best_state   = None
    no_improve   = 0
    history      = {"train_loss": [], "val_auc": [], "val_ap": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_data, optimizer, device)
        val_auc, val_ap = evaluate(model, val_data, device)
        scheduler.step(val_auc)

        history["train_loss"].append(loss)
        history["val_auc"].append(val_auc)
        history["val_ap"].append(val_ap)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{model_name}] Early stop at epoch {epoch}")
                break

        if epoch % 25 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  [{model_name}] Epoch {epoch:3d} | "
                f"Loss {loss:.4f} | Val AUC {val_auc:.4f} | Val AP {val_ap:.4f} | "
                f"{elapsed:.1f}s"
            )

    # Restore best weights and evaluate on test set
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_auc, test_ap = evaluate(model, test_data, device)

    return {
        "model":     model_name,
        "best_val_auc": best_val_auc,
        "test_auc":  test_auc,
        "test_ap":   test_ap,
        "history":   history,
        "trained_model": model.cpu(),
    }


def run_all_experiments(train_data, val_data, test_data, in_channels=2048, epochs=150):
    device = get_device()
    print(f"\nUsing device: {device}\n{'='*60}")

    all_results = {}
    for name in ALL_MODELS:
        print(f"\nTraining: {name}")
        print("-" * 40)
        res = run_experiment(
            name, train_data, val_data, test_data,
            in_channels=in_channels, epochs=epochs, device=device
        )
        all_results[name] = res
        print(
            f"  ✓ {name} → Test AUC: {res['test_auc']:.4f} | Test AP: {res['test_ap']:.4f}"
        )

    return all_results
