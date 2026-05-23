"""
pocket_encoder.py — Multi-resolution SE(3)-EGNN Pocket Encoder.

4-layer SE(3)-equivariant GNN that processes pocket atoms (from .pdb or .mol2)
and produces per-atom embeddings h_P ∈ ℝ^{N_P × hidden_dim} used for
cross-attention at every layer of the main EGNN.

Key design:
  - k-NN graph (k=16) rebuilt from pocket coordinates
  - RBF distance edge features
  - Pocket coordinates (x_P) are NEVER updated — only features (h_P)
  - Global mean-pool produces a pocket-level embedding for size prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.featurizer import build_knn_graph, rbf_encode
from .utils import scatter_mean


class EGNNPocketLayer(nn.Module):
    """Single SE(3)-equivariant message-passing layer for pocket encoding.

    Only updates node features h — coordinates x are frozen.
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int = 16):
        super().__init__()
        # Edge model: φ_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Attention gate: φ_att
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Node update: φ_h
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # (N, hidden_dim)
        pos: torch.Tensor,         # (N, 3)
        edge_index: torch.Tensor,  # (2, E)
        edge_feat: torch.Tensor,   # (E, edge_feat_dim)
    ) -> torch.Tensor:
        src, dst = edge_index  # source, destination node indices

        # Squared distance as additional edge input
        d_sq = ((pos[src] - pos[dst]) ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Edge messages
        edge_input = torch.cat([h[src], h[dst], edge_feat, d_sq], dim=-1)
        m_ij = self.edge_mlp(edge_input)           # (E, hidden_dim)
        att_ij = self.att_mlp(m_ij)                 # (E, 1)
        m_ij = att_ij * m_ij                        # gated messages

        # Aggregate messages per node (match dtype for bf16 compatibility)
        agg = torch.zeros_like(h).to(m_ij.dtype)
        agg.index_add_(0, dst, m_ij)

        # Update node features (residual)
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.layer_norm(h + h_new)

        return h


class PocketEncoder(nn.Module):
    """4-layer SE(3)-EGNN pocket encoder.

    Parameters
    ----------
    in_dim     : dimension of raw pocket atom features
    hidden_dim : hidden dimension (default 128)
    num_layers : number of message-passing layers (default 4)
    knn_k      : k for k-NN graph (default 16)
    num_rbf    : number of RBF distance features (default 16)
    """

    def __init__(
        self,
        in_dim: int = 40,
        hidden_dim: int = 128,
        num_layers: int = 4,
        knn_k: int = 16,
        num_rbf: int = 16,
    ):
        super().__init__()
        self.knn_k = knn_k
        self.num_rbf = num_rbf

        # Project raw features to hidden dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNPocketLayer(hidden_dim, edge_feat_dim=num_rbf)
            for _ in range(num_layers)
        ])

        # Global pooling projection (for size prediction)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        pos: torch.Tensor,              # (N_P, 3) — pocket atom coordinates
        feat: torch.Tensor,             # (N_P, in_dim) — pocket atom features
        batch_P: torch.Tensor = None,   # (N_P,) long — graph assignment
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
            h_P    : (N_P, hidden_dim) — per-atom pocket embeddings
            h_glob : (B, hidden_dim) or (hidden_dim,) — global pocket embedding
        """
        # Infinite shift trick: offset each graph's coords so k-NN
        # never connects atoms from different graphs
        if batch_P is not None:
            pos_shifted = pos + batch_P.unsqueeze(-1).float() * 10000.0
        else:
            pos_shifted = pos

        # Build k-NN graph (on shifted coords to isolate graphs)
        edge_index, _ = build_knn_graph(pos_shifted, k=self.knn_k)

        # RBF edge features (on REAL distances, not shifted)
        diff_real = pos[edge_index[0]] - pos[edge_index[1]]
        real_dist = torch.sqrt((diff_real ** 2).sum(dim=-1) + 1e-8)
        edge_feat = rbf_encode(real_dist, num_rbf=self.num_rbf)

        # Project input features
        h = self.input_proj(feat)  # (N_P, hidden_dim)

        # Message passing (coordinates frozen)
        for layer in self.layers:
            h = layer(h, pos, edge_index, edge_feat)

        # Global mean-pool
        if batch_P is not None:
            B = batch_P.max().item() + 1
            h_glob = scatter_mean(h, batch_P, B)       # (B, hidden_dim)
            h_glob = self.global_proj(h_glob)           # (B, hidden_dim)
        else:
            h_glob = self.global_proj(h.mean(dim=0))    # (hidden_dim,)

        return {"h_P": h, "h_glob": h_glob}
