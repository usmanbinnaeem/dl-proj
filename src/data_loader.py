"""
data_loader.py
--------------
Downloads BioSNAP ChCh-Miner DDI edges, fetches SMILES from PubChem,
generates 2048-bit Morgan fingerprints, and builds PyG Data objects
with train / val / test link-prediction splits.
"""

import gzip
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm

# ── constants ──────────────────────────────────────────────────────────────────
SNAP_DDI_URL = (
    "https://snap.stanford.edu/biodata/datasets/10001/files/"
    "ChCh-Miner_durgbank-chem-chem.tsv.gz"
)
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/"
    "RegistryID/{drug_id}/property/IsomericSMILES/JSON"
)
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MAX_WORKERS = 5          # PubChem rate-limit: ≤5 req/s
REQUEST_DELAY = 0.25     # seconds between worker bursts


# ── helpers ────────────────────────────────────────────────────────────────────

def _download_ddi_edges(data_dir: str) -> str:
    gz_path = os.path.join(data_dir, "ChCh-Miner.tsv.gz")
    if not os.path.exists(gz_path):
        print("Downloading BioSNAP ChCh-Miner DDI edges …")
        req = urllib.request.Request(
            SNAP_DDI_URL, headers={"User-Agent": "DDI-GNN-Project/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            with open(gz_path, "wb") as f:
                f.write(r.read())
        print(f"  Saved to {gz_path}")
    return gz_path


def _load_edges(gz_path: str):
    pairs = []
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            a, b = line.strip().split()
            pairs.append((a, b))
    return pairs


def _fetch_single_smiles(drug_id: str):
    url = PUBCHEM_URL.format(drug_id=drug_id)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "DDI-GNN-Project/1.0"}
        )
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
            props = data["PropertyTable"]["Properties"][0]
            smiles = props.get("IsomericSMILES") or props.get("SMILES") or props.get("CanonicalSMILES")
            return drug_id, smiles
    except Exception:
        return drug_id, None


def _fetch_smiles_parallel(drug_ids: list, cache_path: str) -> dict:
    """Fetch SMILES for all drug_ids, using disk cache to avoid re-fetching."""
    # Load existing cache (including previously-tried-but-failed entries)
    cached_all = {}   # drug_id -> smiles_or_None  (tracks all attempted drugs)
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        for _, row in df.iterrows():
            v = row["smiles"]
            cached_all[row["drug_id"]] = v if isinstance(v, str) and v else None

    # Only fetch drugs we have never attempted before
    missing = [d for d in drug_ids if d not in cached_all]
    if missing:
        print(f"Fetching SMILES for {len(missing)} drugs from PubChem …")
        print("  (This is a one-time operation; results will be cached.)")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_fetch_single_smiles, d): d for d in missing}
            for i, fut in enumerate(
                tqdm(as_completed(futures), total=len(missing), desc="PubChem")
            ):
                drug_id, smiles = fut.result()
                cached_all[drug_id] = smiles  # store None for failures too
                if i % 5 == 4:
                    time.sleep(REQUEST_DELAY)

        # Persist all entries (including None) so failures aren't retried
        df = pd.DataFrame(
            [{"drug_id": k, "smiles": v if v else ""} for k, v in cached_all.items()]
        )
        df.to_csv(cache_path, index=False)
        valid_count = sum(1 for v in cached_all.values() if v)
        print(f"  Cached {valid_count} SMILES (+ {len(cached_all)-valid_count} failures) to {cache_path}")

    # Return only drugs with valid SMILES
    cached = {k: v for k, v in cached_all.items() if v}

    return cached


def _morgan_fp(smiles: str) -> np.ndarray:
    """Convert a SMILES string to a 2048-bit Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = AllChem.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(MORGAN_BITS, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


# ── public API ─────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str = "data", seed: int = 42,
                 neg_sampling_ratio: float = 1.0):
    """
    Load and preprocess the BioSNAP DDI dataset.

    Returns
    -------
    data        : PyG Data  (full graph with node features)
    train_data  : PyG Data  (link-prediction split, train)
    val_data    : PyG Data  (link-prediction split, val)
    test_data   : PyG Data  (link-prediction split, test)
    drug_ids    : list[str] (DrugBank IDs, index → id)
    """
    os.makedirs(data_dir, exist_ok=True)

    # 1. DDI edges ────────────────────────────────────────────────────────────
    gz_path = _download_ddi_edges(data_dir)
    pairs = _load_edges(gz_path)
    all_drugs = sorted({d for pair in pairs for d in pair})
    print(f"Dataset: {len(all_drugs)} drugs, {len(pairs)} DDI edges")

    # 2. SMILES → Morgan fingerprints ────────────────────────────────────────
    smiles_cache = os.path.join(data_dir, "drug_smiles.csv")
    smiles_map = _fetch_smiles_parallel(all_drugs, smiles_cache)

    # Build valid drug list (only drugs with successful SMILES)
    fps = {}
    for d in all_drugs:
        smi = smiles_map.get(d)
        if smi:
            arr = _morgan_fp(smi)
            if arr is not None:
                fps[d] = arr

    valid_drugs = [d for d in all_drugs if d in fps]
    valid_set = set(valid_drugs)
    drug_to_idx = {d: i for i, d in enumerate(valid_drugs)}
    print(
        f"Valid drugs (with SMILES): {len(valid_drugs)} / {len(all_drugs)}"
    )

    # 3. Filter edges to valid drugs only ────────────────────────────────────
    valid_pairs = [(a, b) for a, b in pairs if a in valid_set and b in valid_set]
    print(f"Valid DDI edges (both drugs have SMILES): {len(valid_pairs)}")

    # 4. Build node feature matrix ────────────────────────────────────────────
    x = torch.tensor(
        np.stack([fps[d] for d in valid_drugs]), dtype=torch.float
    )  # shape [N, 2048]

    # 5. Build edge_index (undirected: add both directions) ───────────────────
    src = [drug_to_idx[a] for a, b in valid_pairs]
    dst = [drug_to_idx[b] for a, b in valid_pairs]
    # Undirected: add both (a→b) and (b→a)
    edge_index = torch.tensor(
        [src + dst, dst + src], dtype=torch.long
    )

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = len(valid_drugs)

    # 6. Train / val / test split (80 / 10 / 10) ─────────────────────────────
    torch.manual_seed(seed)
    splitter = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=neg_sampling_ratio,
    )
    train_data, val_data, test_data = splitter(data)

    print(
        f"Split → train pos edges: {train_data.edge_label.sum().int().item()}, "
        f"val pos: {val_data.edge_label.sum().int().item()}, "
        f"test pos: {test_data.edge_label.sum().int().item()}"
    )

    return data, train_data, val_data, test_data, valid_drugs


if __name__ == "__main__":
    data, train_data, val_data, test_data, drugs = load_dataset()
    print("\nFull graph:", data)
    print("Train data:", train_data)
    print("Val data:  ", val_data)
    print("Test data: ", test_data)
