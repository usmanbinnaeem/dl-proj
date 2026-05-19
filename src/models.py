"""
models.py
---------
Five model configurations for DDI link-prediction:
  1. MLPBaseline    — no graph structure (feature concat → MLP)
  2. GCN + Dot      — GCN encoder + dot-product decoder
  3. GCN + MLP      — GCN encoder + MLP decoder
  4. GIN + Dot      — GIN encoder + dot-product decoder
  5. GIN + MLP      — GIN encoder + MLP decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv


# ── Encoders ───────────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """Two-layer GCN that maps node features to d-dim embeddings."""

    def __init__(self, in_channels: int, hidden: int = 256, out: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, out)
        self.bn2   = nn.BatchNorm1d(out)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        return h  # [N, out]


class GINEncoder(nn.Module):
    """Two-layer GIN that maps node features to d-dim embeddings."""

    def __init__(self, in_channels: int, hidden: int = 256, out: int = 64, dropout: float = 0.3):
        super().__init__()

        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.bn1   = nn.BatchNorm1d(hidden)

        mlp2 = nn.Sequential(
            nn.Linear(hidden, out),
            nn.BatchNorm1d(out),
            nn.ReLU(),
            nn.Linear(out, out),
        )
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.bn2   = nn.BatchNorm1d(out)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        return h  # [N, out]


# ── Decoders ───────────────────────────────────────────────────────────────────

class DotDecoder(nn.Module):
    """Score an edge by the dot product of the two node embeddings."""

    def forward(self, z, edge_index):
        # edge_index: [2, E]
        src = z[edge_index[0]]  # [E, d]
        dst = z[edge_index[1]]  # [E, d]
        return (src * dst).sum(dim=-1)  # [E]  (raw logits)


class MLPDecoder(nn.Module):
    """Score an edge by feeding the concatenated embeddings through an MLP."""

    def __init__(self, embed_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, z, edge_index):
        src = z[edge_index[0]]  # [E, d]
        dst = z[edge_index[1]]  # [E, d]
        pair = torch.cat([src, dst], dim=-1)  # [E, 2d]
        return self.net(pair).squeeze(-1)  # [E]  (raw logits)


# ── Full models ────────────────────────────────────────────────────────────────

class GNNModel(nn.Module):
    """GNN encoder + choice of decoder."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        return self.decoder(z, edge_label_index)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class MLPBaseline(nn.Module):
    """
    Non-relational baseline: no message passing.
    Concatenates the raw Morgan fingerprints of both drugs
    and feeds them through an MLP.
    """

    def __init__(self, in_channels: int = 2048, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, edge_label_index):
        # edge_index is unused — no message passing
        src = x[edge_label_index[0]]   # [E, in_channels]
        dst = x[edge_label_index[1]]   # [E, in_channels]
        pair = torch.cat([src, dst], dim=-1)  # [E, 2*in_channels]
        return self.net(pair).squeeze(-1)  # [E]  (raw logits)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_model(name: str, in_channels: int = 2048) -> nn.Module:
    """
    Build one of the five model configurations by name.

    Names
    -----
    'MLP'       : MLPBaseline (no graph)
    'GCN+Dot'   : GCN encoder + dot decoder
    'GCN+MLP'   : GCN encoder + MLP decoder
    'GIN+Dot'   : GIN encoder + dot decoder
    'GIN+MLP'   : GIN encoder + MLP decoder
    """
    hidden, embed = 256, 64

    if name == "MLP":
        return MLPBaseline(in_channels=in_channels)

    if name == "GCN+Dot":
        enc = GCNEncoder(in_channels, hidden, embed)
        dec = DotDecoder()
        return GNNModel(enc, dec)

    if name == "GCN+MLP":
        enc = GCNEncoder(in_channels, hidden, embed)
        dec = MLPDecoder(embed)
        return GNNModel(enc, dec)

    if name == "GIN+Dot":
        enc = GINEncoder(in_channels, hidden, embed)
        dec = DotDecoder()
        return GNNModel(enc, dec)

    if name == "GIN+MLP":
        enc = GINEncoder(in_channels, hidden, embed)
        dec = MLPDecoder(embed)
        return GNNModel(enc, dec)

    raise ValueError(f"Unknown model name: {name}")


ALL_MODELS = ["MLP", "GCN+Dot", "GCN+MLP", "GIN+Dot", "GIN+MLP"]


# ── Ablation models ────────────────────────────────────────────────────────────

class FlexGCNEncoder(nn.Module):
    """
    GCN encoder with configurable layer depth and embedding dimension.
    Used for Ablation A (embed dim) and Ablation B (num layers).
    """

    def __init__(self, in_channels: int, hidden: int = 256, out: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 1
        if num_layers == 1:
            dims = [in_channels, out]
        else:
            dims = [in_channels] + [hidden] * (num_layers - 1) + [out]

        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(dims[i + 1]) for i in range(num_layers)]
        )
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(h, edge_index)
            h = bn(h)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h  # [N, out]


class LearnableGCNMLP(nn.Module):
    """
    GCN+MLP using learnable node embeddings instead of Morgan fingerprints.
    Ablation C: tests whether molecular features are necessary.
    Nodes are identified by position index; no chemistry is encoded.
    """

    def __init__(self, num_nodes: int, hidden: int = 256, out: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, hidden)
        self.conv      = GCNConv(hidden, out)
        self.bn        = nn.BatchNorm1d(out)
        self.dropout   = dropout
        self.decoder   = MLPDecoder(embed_dim=out, dropout=dropout)

    def forward(self, x, edge_index, edge_label_index):
        # x is only used to retrieve device and num_nodes;
        # actual features come from the learnable embedding table.
        node_ids = torch.arange(x.shape[0], device=x.device)
        h = self.embedding(node_ids)                               # [N, hidden]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv(h, edge_index)                               # [N, out]
        h = self.bn(h)
        return self.decoder(h, edge_label_index)
