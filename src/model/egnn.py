"""
egnn.py — 9-Layer SE(3)-Equivariant GNN with Cross-Attention and Value Head.

Core architecture:
  1. Cross-attention from ligand atoms to pocket context h_P at every layer
  2. EGNN message passing with attention-gated messages
  3. SE(3) coordinate update with cross-product term (breaks E(3) → SE(3))
  4. Affinity value head (critic) for RL proxy reward

Key equations per layer:
  // Cross-attend to pocket
  α_i = softmax(Q(h_i)·K(h_P)^T / √d)
  c_i = V(h_P)·α_i
  h_i = h_i + LayerNorm(c_i)

  // EGNN message passing
  m_ij = φ_e(h_i, h_j, d²_ij, e_ij)
  ẽ_ij = φ_att(m_ij)  (attention gate)
  h_i' = φ_h(h_i, Σ_j ẽ_ij·m_ij)

  // SE(3) coordinate update (ligand only)
  Δx_i = Σ_j [(x_i−x_j)/d_ij]·φ_d(h_i,h_j,d²,e)
       + [(x_i−x̄)×(x_j−x̄)] / (||...||+1)·φ_×(...)
  x_i' = x_i + Δx_i
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.featurizer import build_knn_graph, rbf_encode
from .utils import SinusoidalTimeEmbedding


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention Module
# ──────────────────────────────────────────────────────────────────────────────

class PocketCrossAttention(nn.Module):
    """Multi-head cross-attention: ligand queries attend to pocket keys/values.

    Q from ligand h_L, K/V from pocket h_P.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 16, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # Spec: scale by 1/√hidden_dim (not 1/√head_dim) for softer pocket attention
        self.scale = hidden_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_L: torch.Tensor,  # (N_L, D) ligand features
        h_P: torch.Tensor,  # (N_P, D) pocket features
    ) -> torch.Tensor:
        """Returns updated ligand features with pocket context."""
        N_L = h_L.size(0)
        N_P = h_P.size(0)

        Q = self.q_proj(h_L).view(N_L, self.num_heads, self.head_dim)  # (N_L, H, d)
        K = self.k_proj(h_P).view(N_P, self.num_heads, self.head_dim)  # (N_P, H, d)
        V = self.v_proj(h_P).view(N_P, self.num_heads, self.head_dim)  # (N_P, H, d)

        # (N_L, H, N_P)
        attn = torch.einsum("lhd,phd->lhp", Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # (N_L, H, d) → (N_L, D)
        context = torch.einsum("lhp,phd->lhd", attn, V)
        context = context.reshape(N_L, -1)
        context = self.out_proj(context)

        # Residual + LayerNorm
        return self.layer_norm(h_L + context)


# ──────────────────────────────────────────────────────────────────────────────
# Single EGNN Layer with Cross-Attention and SE(3) Coord Update
# ──────────────────────────────────────────────────────────────────────────────

class EGNNLayerWithCrossAttn(nn.Module):
    """One layer of the 9-layer SE(3)-EGNN.

    1. Cross-attend to pocket
    2. Message passing on ligand graph
    3. SE(3) coordinate update with cross-product reflection term
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        edge_feat_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cross_attn = PocketCrossAttention(hidden_dim, num_heads, dropout)

        # Edge model: φ_e(h_i, h_j, d², e_ij)
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

        # Node update: φ_h(h_i, agg)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

        # ── Coordinate update MLPs ──
        # Distance-based: φ_d(h_i, h_j, d², e_ij) → scalar weight
        self.coord_dist_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Cross-product term: φ_×(h_i, h_j, d², e_ij) → scalar weight
        # This breaks E(3) → SE(3), making the model reflection-sensitive
        self.coord_cross_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_L: torch.Tensor,          # (N_L, D) ligand features
        x_L: torch.Tensor,          # (N_L, 3) ligand coordinates
        h_P: torch.Tensor,          # (N_P, D) pocket features
        edge_index: torch.Tensor,   # (2, E) ligand graph edges
        edge_feat: torch.Tensor,    # (E, edge_feat_dim) RBF features
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_L_new : (N_L, D) updated ligand features
        x_L_new : (N_L, 3) updated ligand coordinates
        """
        # 1. Cross-attend to pocket
        h_L = self.cross_attn(h_L, h_P)

        # 2. Message passing
        src, dst = edge_index
        d_sq = ((x_L[src] - x_L[dst]) ** 2).sum(dim=-1, keepdim=True)

        edge_input = torch.cat([h_L[src], h_L[dst], edge_feat, d_sq], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        att_ij = self.att_mlp(m_ij)
        m_ij = att_ij * m_ij

        agg = torch.zeros_like(h_L)
        agg.index_add_(0, dst, m_ij)

        h_L_new = self.node_mlp(torch.cat([h_L, agg], dim=-1))
        h_L = self.node_norm(h_L + h_L_new)

        # 3. SE(3) coordinate update
        diff = x_L[src] - x_L[dst]                             # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
        direction = diff / dist                                  # unit vector

        # Distance-based displacement
        w_dist = self.coord_dist_mlp(edge_input)                # (E, 1)
        delta_dist = direction * w_dist                          # (E, 3)

        # Cross-product term (SE(3) not E(3))
        x_mean = x_L.mean(dim=0, keepdim=True)
        v_i = x_L[src] - x_mean                                 # (E, 3)
        v_j = x_L[dst] - x_mean                                 # (E, 3)
        cross = torch.cross(v_i, v_j, dim=-1)                   # (E, 3)
        cross_norm = torch.norm(cross, dim=-1, keepdim=True).clamp(min=1e-8)
        cross_dir = cross / (cross_norm + 1.0)

        w_cross = self.coord_cross_mlp(edge_input)              # (E, 1)
        delta_cross = cross_dir * w_cross                        # (E, 3)

        # Aggregate coordinate updates
        delta_x = torch.zeros_like(x_L)
        delta_x.index_add_(0, dst, delta_dist + delta_cross)

        x_L = x_L + delta_x

        return h_L, x_L


# ──────────────────────────────────────────────────────────────────────────────
# Full 9-Layer SE(3)-EGNN Model
# ──────────────────────────────────────────────────────────────────────────────

class SBDDEGNN(nn.Module):
    """9-layer SE(3)-EGNN with cross-attention, flow velocity head, and
    affinity value head.

    Parameters
    ----------
    ligand_in_dim : raw ligand feature dimension
    pocket_dim    : pocket embedding dimension (from PocketEncoder)
    hidden_dim    : hidden dimension (128)
    num_layers    : number of EGNN layers (9)
    num_heads     : cross-attention heads (16)
    num_atom_types : number of categorical atom types (10)
    time_emb_dim  : sinusoidal time embedding dimension
    """

    def __init__(
        self,
        ligand_in_dim: int = 20,
        pocket_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 9,
        num_heads: int = 16,
        num_atom_types: int = 10,
        time_emb_dim: int = 128,
        knn_k: int = 16,
        num_rbf: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.knn_k = knn_k
        self.num_rbf = num_rbf
        self.num_atom_types = num_atom_types

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # Input projection for ligand features
        self.ligand_proj = nn.Linear(ligand_in_dim + num_atom_types, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayerWithCrossAttn(hidden_dim, num_heads, num_rbf, dropout)
            for _ in range(num_layers)
        ])

        # ── Flow velocity head ──
        # Predicts v_θ(z_t, t, P) — the velocity field for coordinates
        self.vel_coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Predicts velocity for atom types (categorical flow)
        self.vel_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atom_types),
        )

        # ── Affinity value head (critic) ──
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_L: torch.Tensor,           # (N_L, 3) ligand coords at time t
        h_L_raw: torch.Tensor,        # (N_L, ligand_in_dim) raw features
        atom_types_onehot: torch.Tensor,  # (N_L, num_atom_types) noised one-hot
        t: torch.Tensor,              # (1,) or scalar — time in [0, 1]
        h_P: torch.Tensor,            # (N_P, pocket_dim) pocket embeddings
    ) -> dict:
        """
        Returns
        -------
        dict with:
            vel_coord : (N_L, 3) predicted coordinate velocity
            vel_type  : (N_L, num_atom_types) predicted type velocity
            pK_pred   : scalar — predicted affinity proxy
            h_L       : (N_L, hidden_dim) final ligand embeddings
        """
        N_L = x_L.size(0)

        # Time conditioning
        t_emb = self.time_proj(self.time_emb(t))  # (1, hidden_dim)
        if t_emb.dim() == 2:
            t_emb = t_emb.squeeze(0)  # (hidden_dim,)

        # Ligand input: concatenate raw features + noised atom type one-hot
        h_L_input = torch.cat([h_L_raw, atom_types_onehot], dim=-1)
        h_L = self.ligand_proj(h_L_input)         # (N_L, hidden_dim)

        # Add time embedding
        h_L = h_L + t_emb.unsqueeze(0).expand(N_L, -1)

        # Build ligand k-NN graph
        edge_index, edge_dist = build_knn_graph(x_L, k=self.knn_k)
        edge_feat = rbf_encode(edge_dist, num_rbf=self.num_rbf)

        # 9 layers of EGNN with cross-attention
        for layer in self.layers:
            h_L, x_L = layer(h_L, x_L, h_P, edge_index, edge_feat)

            # Rebuild graph after coordinate update (dynamic graph)
            edge_index, edge_dist = build_knn_graph(x_L, k=self.knn_k)
            edge_feat = rbf_encode(edge_dist, num_rbf=self.num_rbf)

        # Velocity heads
        vel_coord = self.vel_coord_head(h_L)   # (N_L, 3)
        vel_type = self.vel_type_head(h_L)      # (N_L, num_atom_types)

        # Affinity value head: mean-pool over ligand → scalar
        h_pool = h_L.mean(dim=0)                # (hidden_dim,)
        pK_pred = self.value_head(h_pool)        # (1,)

        return {
            "vel_coord": vel_coord,
            "vel_type": vel_type,
            "pK_pred": pK_pred.squeeze(-1),
            "h_L": h_L,
            "x_L_updated": x_L,
        }
