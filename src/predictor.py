"""
predictor.py
------------
Standalone DDI predictor.  Loads the saved model bundle and predicts
whether two drugs interact given their SMILES strings or DrugBank IDs.

Usage (Python API)
------------------
    from src.predictor import DDIPredictor

    pred = DDIPredictor("results/model_bundle.pt")
    result = pred.predict_smiles(
        "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    )
    print(result)
    # {'probability': 0.73, 'uncertainty': 0.08,
    #  'confidence': 'MEDIUM', 'ood_warning': True}

Usage (CLI)
-----------
    .venv/bin/python3 src/predictor.py \\
        --smiles-a "CC(=O)Oc1ccccc1C(=O)O" \\
        --smiles-b "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"

    .venv/bin/python3 src/predictor.py --id-a DB00945 --id-b DB00067
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model


# ── Morgan FP helper (mirrors data_loader.py) ─────────────────────────────────

def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert a SMILES string to a Morgan fingerprint vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


# ── Predictor class ────────────────────────────────────────────────────────────

class DDIPredictor:
    """
    Load a saved model bundle and predict drug–drug interactions.

    Parameters
    ----------
    bundle_path : path to results/model_bundle.pt
    n_passes    : number of MC Dropout forward passes for uncertainty
    device      : torch device (auto-detected if None)
    """

    CONFIDENCE_THRESHOLDS = {
        "HIGH":   (0.0,  0.05),   # std < 0.05
        "MEDIUM": (0.05, 0.10),   # 0.05 ≤ std < 0.10
        "LOW":    (0.10, 1.0),    # std ≥ 0.10
    }
    OOD_STD_THRESHOLD = 0.08      # flag pair as OOD if std exceeds this

    def __init__(self, bundle_path: str = "results/model_bundle.pt",
                 n_passes: int = 20, device=None):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path!r}.\n"
                "Run run_remaining.py first to train and save the model."
            )

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.in_channels      = bundle["in_channels"]
        self.drug_ids         = bundle["drug_ids"]          # list[str]
        self.drug_id_to_idx   = {d: i for i, d in enumerate(self.drug_ids)}
        self.graph_x          = bundle["graph_x"]           # [N, 2048]
        self.graph_edge_index = bundle["graph_edge_index"]  # [2, E]
        self.n_passes         = n_passes

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        self.model = build_model("GCN+MLP", in_channels=self.in_channels)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"DDIPredictor loaded  ({len(self.drug_ids)} training drugs, "
              f"device={self.device})")

    # ── Core inference ─────────────────────────────────────────────────────────

    def _get_node_index(self, smiles: str) -> tuple:
        """
        Return (node_idx, x_matrix, edge_index, is_ood).

        If the drug is already in the training graph we reuse its index.
        If it's a new (OOD) drug we append it as an isolated node — the GCN
        will compute its embedding from its own Morgan FP alone (no neighbours).
        """
        # Check if SMILES matches a known drug by fingerprint similarity
        fp = smiles_to_fp(smiles)
        fp_tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)  # [1, 2048]

        # Cosine similarity against all training drugs
        norms_train = self.graph_x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        norm_query  = fp_tensor.norm().clamp(min=1e-8)
        sims = (self.graph_x @ fp_tensor.T).squeeze() / (norms_train.squeeze() * norm_query)
        best_sim, best_idx = sims.max(dim=0)

        if best_sim.item() > 0.999:
            # Exact match — use training graph node
            return int(best_idx.item()), self.graph_x, self.graph_edge_index, False
        else:
            # New drug — append as isolated node
            new_x = torch.cat([self.graph_x, fp_tensor], dim=0)
            new_idx = new_x.shape[0] - 1
            return new_idx, new_x, self.graph_edge_index, True

    def _run_mc_dropout(self, x, edge_index, src_idx: int, dst_idx: int) -> tuple:
        """Run n_passes stochastic forward passes; return (mean_prob, std_prob)."""
        x          = x.to(self.device)
        edge_index = edge_index.to(self.device)
        eli        = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long).to(self.device)

        self.model.train()   # keep dropout active
        probs = []
        with torch.no_grad():
            for _ in range(self.n_passes):
                logits = self.model(x, edge_index, eli)
                probs.append(torch.sigmoid(logits).item())
        self.model.eval()

        arr = np.array(probs)
        return float(arr.mean()), float(arr.std())

    def _confidence_label(self, std: float) -> str:
        for label, (lo, hi) in self.CONFIDENCE_THRESHOLDS.items():
            if lo <= std < hi:
                return label
        return "LOW"

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict_smiles(self, smiles_a: str, smiles_b: str) -> dict:
        """
        Predict DDI probability for two drugs given their SMILES strings.

        Returns
        -------
        dict with keys:
          probability  : float  [0, 1]
          uncertainty  : float  MC Dropout std
          confidence   : str    HIGH / MEDIUM / LOW
          ood_warning  : bool   True if either drug is not in training set
          drug_a_known : bool
          drug_b_known : bool
        """
        idx_a, x_a, ei_a, ood_a = self._get_node_index(smiles_a)
        idx_b, x_b, ei_b, ood_b = self._get_node_index(smiles_b)

        # Merge node sets if both are OOD (or one of them is)
        if ood_a and ood_b:
            # Append both as isolated nodes
            fp_a = smiles_to_fp(smiles_a)
            fp_b = smiles_to_fp(smiles_b)
            x = torch.cat([
                self.graph_x,
                torch.tensor(fp_a).unsqueeze(0),
                torch.tensor(fp_b).unsqueeze(0),
            ], dim=0)
            idx_a = x.shape[0] - 2
            idx_b = x.shape[0] - 1
            edge_index = self.graph_edge_index
        elif ood_a:
            x, edge_index = x_a, ei_a
            idx_b_local = idx_b   # b is in training graph at its original index
            idx_b = idx_b_local
        elif ood_b:
            x, edge_index = x_b, ei_b
        else:
            x, edge_index = self.graph_x, self.graph_edge_index

        mean_prob, std_prob = self._run_mc_dropout(x, edge_index, idx_a, idx_b)

        return {
            "probability":  round(mean_prob, 4),
            "uncertainty":  round(std_prob,  4),
            "confidence":   self._confidence_label(std_prob),
            "ood_warning":  ood_a or ood_b,
            "drug_a_known": not ood_a,
            "drug_b_known": not ood_b,
        }

    def predict_drugbank_ids(self, id_a: str, id_b: str,
                              smiles_cache_path: str = "data/drug_smiles.csv") -> dict:
        """
        Predict DDI for two DrugBank IDs.
        Looks up their SMILES from the cache CSV.
        """
        import csv as _csv
        smiles_map = {}
        with open(smiles_cache_path) as f:
            for row in _csv.DictReader(f):
                if row["smiles"]:
                    smiles_map[row["drug_id"]] = row["smiles"]

        if id_a not in smiles_map:
            raise ValueError(f"DrugBank ID {id_a!r} not found in SMILES cache.")
        if id_b not in smiles_map:
            raise ValueError(f"DrugBank ID {id_b!r} not found in SMILES cache.")

        result = self.predict_smiles(smiles_map[id_a], smiles_map[id_b])
        result["drug_a_id"] = id_a
        result["drug_b_id"] = id_b
        return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Predict drug–drug interaction probability."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles-a", help="SMILES for drug A")
    parser.add_argument("--smiles-b", help="SMILES for drug B (required with --smiles-a)")
    group.add_argument("--id-a", help="DrugBank ID for drug A (e.g. DB00945)")
    parser.add_argument("--id-b", help="DrugBank ID for drug B (required with --id-a)")
    parser.add_argument("--bundle", default="results/model_bundle.pt",
                        help="Path to saved model bundle")
    parser.add_argument("--passes", type=int, default=20,
                        help="MC Dropout forward passes (default: 20)")
    args = parser.parse_args()

    pred = DDIPredictor(bundle_path=args.bundle, n_passes=args.passes)

    if args.smiles_a:
        if not args.smiles_b:
            parser.error("--smiles-b is required when using --smiles-a")
        result = pred.predict_smiles(args.smiles_a, args.smiles_b)
    else:
        if not args.id_b:
            parser.error("--id-b is required when using --id-a")
        result = pred.predict_drugbank_ids(args.id_a, args.id_b)

    print("\n" + "=" * 45)
    print("  DDI Prediction Result")
    print("=" * 45)
    if "drug_a_id" in result:
        print(f"  Drug A            : {result['drug_a_id']}")
        print(f"  Drug B            : {result['drug_b_id']}")
    print(f"  Probability       : {result['probability']:.4f}  "
          f"({'INTERACT' if result['probability'] >= 0.5 else 'NO INTERACTION'})")
    print(f"  Uncertainty (std) : {result['uncertainty']:.4f}")
    print(f"  Confidence        : {result['confidence']}")
    if result["ood_warning"]:
        print("  ⚠  OOD WARNING: one or both drugs were not in the training set.")
        print("     Prediction is based on molecular features alone — treat with caution.")
    print("=" * 45)


if __name__ == "__main__":
    _cli()
