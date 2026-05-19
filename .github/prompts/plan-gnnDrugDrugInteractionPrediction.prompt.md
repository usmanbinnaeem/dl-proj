# Plan: GNN for Drug-Drug Interaction Prediction — From Scratch

**TL;DR:** Build a proper GNN-based DDI prediction system on a real benchmark dataset (DrugBank-DDI with molecular fingerprint features), compare multiple architectures, and deliver a well-grounded report with real preliminary results. The mid-report is due **tomorrow night** so the plan is split into what happens today vs. what improves for the final.

---

## Critical Timeline

| Block | When | Duration |
|---|---|---|
| Phase 0 — Setup | Today, now | ~1 hr |
| Phase 1 — Code + Results | Today afternoon/evening | ~6 hrs |
| Phase 2 — Mid-Report Writing | Tonight/tomorrow morning | ~3-4 hrs |
| Phase 3 — Final Report Extensions | May 9–21 | ~2 weeks |

---

## Phase 0 — Environment & Dataset Setup
*Do this first, nothing else can proceed without it.*

**Step 1 — Python environment**
Create a conda env with Python 3.10, install:
- `torch` (2.x)
- `torch_geometric` + its dependencies (`torch-scatter`, `torch-sparse`, `torch-cluster`)
- `rdkit` (for Morgan fingerprints from SMILES)
- `scikit-learn` (AUC, AP metrics)
- `matplotlib`, `pandas`, `numpy`, `tqdm`

**Step 2 — Dataset selection and download**
Use **DrugBank-DDI** (the well-known binary DDI benchmark):
- ~572 drugs, ~37,264 DDI pairs, each drug has a SMILES string attached
- Available in preprocessed form from GitHub repos like `kexinhuang12345/CASTER` or `zzeqii/DDI-prediction`
- Positive pairs = known interactions; negative pairs = randomly sampled non-interacting pairs
- Why this dataset: it has SMILES → you can generate real molecular features, AUC is the standard metric, multiple 2024-2025 papers benchmark on it

---

## Phase 1 — Code Sprint (Preliminary Results for Mid-Report)

**Step 3 — Data pipeline** (~1.5 hrs)
- Load drug SMILES + DDI edge list
- For each drug, generate a **2048-bit Morgan fingerprint** (radius=2) using RDKit — this becomes the node feature vector
- Build a PyTorch Geometric `Data` object:
  - `x` = drug feature matrix [572 × 2048] (Morgan fingerprints)
  - `edge_index` = known DDI edges
- Split edges into **train / val / test (80/10/10)** using random split with fixed seed
- Generate **negative samples** (random non-edges) in equal ratio for training

**Step 4 — Model implementations** (~2.5 hrs)
Build 4 models, all in one `models.py` file:

| Model | Architecture | Role |
|---|---|---|
| **MLP Baseline** | Feature concat → 3-layer MLP → sigmoid | No graph structure — upper bound on "memorization" |
| **GCN + Dot** | `GCNConv` × 2 layers → node embeddings → dot product decoder | Standard GNN baseline |
| **GCN + MLP** | Same GCN encoder → concatenate pair embeddings → MLP decoder | More expressive decoder |
| **GIN + Dot** | `GINConv` × 2 layers → node embeddings → dot product | Theoretically more expressive encoder |
| **GIN + MLP** | Same GIN encoder → MLP decoder | Best expected model |

Training: Adam optimizer, Binary Cross-Entropy loss, 100 epochs, early stopping on val AUC.

**Step 5 — Training loop + evaluation** (~1 hr)
- Track **ROC-AUC** and **Average Precision (AP)** on val and test set per model
- Save best model checkpoint per architecture
- Output a clean results table (this goes directly into the report)

Expected results range (based on literature):
- MLP: ~0.70-0.75 AUC (will overfit without structure)
- GCN/GIN: ~0.82-0.88 AUC (competitive with published baselines)
- Best model: targeting ~0.88+ AUC to be competitive with 2024-2025 papers

**Step 6 — One plot** (~30 min)
Training AUC curves for all models on one figure — shows learning dynamics and convergence.

---

## Phase 2 — Mid-Report Writing

Mid-report does NOT need ICML format (that is only for final). It just needs to be a clear document. Structure it as:

**Section 1 — Introduction (½ page)**
What DDIs are, why predicting them matters, how you frame it as graph link prediction.

**Section 2 — Literature Review (1 page)**
Cover these papers in this order:
1. *Zitnik et al. (2018)* — Decagon, ISMB 2018 — the foundational GNN-for-DDI paper; introduced polypharmacy graph formulation, arXiv:1802.00543
2. *Kipf & Welling (2017)* — GCN, ICLR 2017 — foundational graph convolution, arXiv:1609.02907
3. *Gilmer et al. (2017)* — MPNN, ICML 2017 — message passing unifying framework, arXiv:1704.01212
4. *Xu et al. (2019)* — GIN, ICLR 2019 — expressiveness analysis, GIN vs GCN, arXiv:1810.00826
5. **One 2025 paper** — search Google Scholar for "drug drug interaction graph neural network 2025 site:arxiv.org" — pick one with AUC numbers to use as your SOTA comparison anchor

For the 2025 paper, TA will check this. It must be real. Search terms: *"drug-drug interaction prediction 2025 GNN"* on Google Scholar, filter by 2025. Note down the exact title, authors, venue, and AUC score reported.

**Section 3 — Methodology (1-1.5 pages)**
- Problem formulation: graph G=(V,E), link prediction task, binary BCE loss
- Dataset description: DrugBank-DDI, 572 drugs, SMILES → Morgan fingerprints → 2048-dim node features
- Architecture description: encoder (GCN vs GIN, 2 layers, hidden dim 64 or 128), decoder (dot vs MLP)
- Training setup: Adam, lr=0.001, 100 epochs, 80/10/10 split, negative sampling

**Section 4 — Preliminary Results (1 page)**
- Table with AUC and AP per model
- Training curves figure
- 2-3 sentences of interpretation ("GIN outperforms GCN, which outperforms MLP, consistent with expressiveness theory from Xu et al.")

**Section 5 — Planned Extensions (½ page)**
- OOD evaluation (hold out entire drugs from training)
- MC Dropout for epistemic uncertainty estimation
- Ablation on hidden dimension and number of GNN layers
- Molecular graph representation (atom-bond graph per drug, not just fingerprints)

---

## Phase 3 — Final Report Extensions (May 9–22)

These make your report "go beyond baseline" per the professor's requirements:

**Extension A — OOD Evaluation**
Hold out 15% of drug nodes entirely from training. Evaluate on edges between those held-out drugs. Shows whether the model generalizes to new drugs vs. memorizing the training graph topology.

**Extension B — MC Dropout Uncertainty**
Run inference with dropout active (N=20 passes) → compute variance per prediction → high variance = uncertain → flag these pairs. Shows a medically meaningful "safety" signal.

**Extension C — Ablation Studies** (required by grading rubric)
- Hidden dimension: 32 vs 64 vs 128 vs 256
- Number of GNN layers: 1 vs 2 vs 3
- With vs without node features (learnable embeddings vs Morgan fingerprints)
- Dot decoder vs MLP decoder (already covered)

**Extension D — SOTA Comparison**
Report your best model's AUC side-by-side with the 2025 paper you cited. The goal is to come within 0.03-0.05 AUC of it (SOTA baseline) or beat it.

**Extension E — GitHub + ICML Format**
Clean code on GitHub, 4-5 page ICML-format LaTeX report.

---

## File Structure (will be created)

```
dl-proj/
├── src/
│   ├── data_loader.py    — dataset download, feature generation, PyG Data object
│   ├── models.py         — MLP, GCNEncoder, GINEncoder, DotDecoder, MLPDecoder
│   ├── train.py          — training loop, evaluation, checkpointing
│   └── utils.py          — negative sampling, AUC/AP computation, plotting
├── data/                 — raw and processed dataset files
├── results/              — saved model checkpoints and output tables
├── notebooks/
│   └── exploration.ipynb — data exploration (optional)
├── report/
│   └── mid_report.pdf    — submission artifact
└── README.md             — GitHub project description (for final)
```

---

## What Makes This Approach Meaningful (TA Requirement)

| TA Expectation | How This Plan Addresses It |
|---|---|
| "Read 2025-2026 papers, check their datasets" | Using DrugBank-DDI benchmark + citing a real 2025 paper with AUC numbers |
| "What result parameters do they use" | ROC-AUC and Average Precision — the exact metrics used in DDI literature |
| "Train a model close to baseline results" | GIN+MLP targeting 0.88+ AUC, which is the 2025 baseline range |
| "How well you understand prior work" | Literature review covers the full chain from GCN → MPNN → GIN → Decagon |
| "What makes your approach meaningful" | OOD evaluation + MC Dropout uncertainty = beyond just accuracy |
| "Proper experimentation, ablations" | 5 model variants + 4 ablation dimensions in final report |

---

## Key Decisions

- **Dataset**: DrugBank-DDI binary (not synthetic, not 50 drugs — a real benchmark)
- **Node features**: Morgan fingerprints (2048-bit, RDKit) — better than random learnable embeddings
- **Primary metric**: ROC-AUC (standard in DDI literature, interpretable, comparable to papers)
- **Mid-report scope**: 4 models, preliminary AUC/AP table, no OOD yet (that's for final)
- **ICML format**: Only required for final report — mid-report is simpler
