"""
utils.py — Shared model utilities: CoM subtraction, graph ops, bond inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def subtract_com(
    pos: torch.Tensor,          # (N, 3)
    batch: torch.Tensor = None,  # (N,) optional batch indices
) -> torch.Tensor:
    """Subtract the centre of mass so that CoM = 0.

    If ``batch`` is provided, each graph in the batch is centred independently.
    """
    if batch is None:
        return pos - pos.mean(dim=0, keepdim=True)

    # Per-graph CoM
    unique_batches = batch.unique()
    centred = pos.clone()
    for b in unique_batches:
        mask = batch == b
        centred[mask] -= pos[mask].mean(dim=0, keepdim=True)
    return centred


def pairwise_distances(pos: torch.Tensor) -> torch.Tensor:
    """Compute all-pairs squared distances.  (N, N)"""
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    return (diff ** 2).sum(dim=-1)


class CosineBetaSchedule:
    """Cosine annealing for the KL penalty β in DDPO.

    β(t) = β_end + 0.5*(β_start − β_end)*(1 + cos(π*t/T))
    """

    def __init__(self, beta_start: float, beta_end: float, total_steps: int):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        import math
        t = min(step / max(self.total_steps, 1), 1.0)
        return self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (
            1.0 + math.cos(math.pi * t)
        )


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the diffusion time step t ∈ [0,1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) or scalar — normalised time in [0, 1]

        Returns
        -------
        emb : (B, dim) sinusoidal embedding
        """
        import math

        if t.dim() == 0:
            t = t.unsqueeze(0)

        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.unsqueeze(-1) * freq.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb
