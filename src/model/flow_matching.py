"""
flow_matching.py — Rectified Flow Matching backbone.

Implements the continuous normalising flow for both coordinates (continuous)
and atom types (categorical).

Forward process (training):
    z_t = (1 − t)·z_data + t·z_noise,   t ∈ [0, 1]
    Target velocity:  u_t = z_noise − z_data
    Loss: E_t[ ||v_θ(z_t, t, P) − u_t||² ]

Reverse process (sampling):
    Start:  z_0 ~ N(0, I)
    Euler:  z_{t+Δt} = z_t + v_θ(z_t, t, P)·Δt
    50 steps sufficient (20× faster than DDPM)

Categorical flow for atom types:
    Uniform noise mixing at t=0, one-hot at t=1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .egnn import SBDDEGNN
from .pocket_encoder import PocketEncoder
from .utils import subtract_com


class FlowMatching(nn.Module):
    """Full Flow Matching model wrapping PocketEncoder + SBDDEGNN.

    Parameters
    ----------
    pocket_encoder : PocketEncoder instance
    egnn           : SBDDEGNN instance
    num_steps      : Euler integration steps for sampling (default 50)
    sigma_min      : minimum noise scale (default 1e-5)
    """

    def __init__(
        self,
        pocket_encoder: PocketEncoder,
        egnn: SBDDEGNN,
        num_steps: int = 50,
        sigma_min: float = 1e-5,
    ):
        super().__init__()
        self.pocket_encoder = pocket_encoder
        self.egnn = egnn
        self.num_steps = num_steps
        self.sigma_min = sigma_min

        # Size predictor: pocket global embedding → predicted N_L
        self.size_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # ensure positive
        )

    # ──────────────────────────────────────────────────────────────────────
    # Forward interpolation (training)
    # ──────────────────────────────────────────────────────────────────────

    def forward_interpolation(
        self,
        x_data: torch.Tensor,          # (N_L, 3) ground-truth coords
        type_data: torch.Tensor,       # (N_L,) int — ground-truth atom types
        t: torch.Tensor,               # scalar in [0, 1]
        num_atom_types: int = 10,
    ) -> dict:
        """Compute the noised state z_t and the target velocity u_t.

        Returns dict with z_t_coord, z_t_type, u_t_coord, u_t_type.
        """
        N_L = x_data.size(0)

        # ── Coordinate flow ──
        z_noise_coord = torch.randn_like(x_data)
        # Subtract CoM from noise
        z_noise_coord = z_noise_coord - z_noise_coord.mean(dim=0, keepdim=True)

        z_t_coord = (1 - t) * x_data + t * z_noise_coord
        u_t_coord = z_noise_coord - x_data  # target velocity

        # ── Categorical atom type flow ──
        # One-hot ground truth
        type_onehot = F.one_hot(type_data, num_atom_types).float()
        # Uniform noise
        uniform = torch.ones_like(type_onehot) / num_atom_types
        # Interpolate
        z_t_type = (1 - t) * type_onehot + t * uniform
        u_t_type = uniform - type_onehot  # target velocity

        return {
            "z_t_coord": z_t_coord,
            "z_t_type": z_t_type,
            "u_t_coord": u_t_coord,
            "u_t_type": u_t_type,
            "z_noise_coord": z_noise_coord,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        pocket_pos: torch.Tensor,      # (N_P, 3)
        pocket_feat: torch.Tensor,     # (N_P, F_P)
        ligand_pos: torch.Tensor,      # (N_L, 3)
        ligand_feat: torch.Tensor,     # (N_L, F_L)
        ligand_atom_types: torch.Tensor,  # (N_L,) int
        affinity: torch.Tensor,        # scalar — ground-truth affinity
        weight: torch.Tensor = None,   # scalar — sample weight
        affinity_lambda: float = 0.1,
        reward_offset: float = 6.0,
        reward_scale: float = 7.0,
    ) -> dict:
        """Compute the flow matching loss + affinity head loss.

        Returns dict with total_loss, flow_loss, affinity_loss.
        """
        device = pocket_pos.device

        # Subtract CoM of pocket coords
        pocket_pos = subtract_com(pocket_pos)
        ligand_pos = subtract_com(ligand_pos)

        # Encode pocket
        pocket_out = self.pocket_encoder(pocket_pos, pocket_feat)
        h_P = pocket_out["h_P"]

        # Sample random time
        t = torch.rand(1, device=device).clamp(min=self.sigma_min, max=1.0 - self.sigma_min)

        # Forward interpolation
        interp = self.forward_interpolation(
            ligand_pos, ligand_atom_types, t,
            num_atom_types=self.egnn.num_atom_types,
        )

        # Predict velocity
        model_out = self.egnn(
            x_L=interp["z_t_coord"],
            h_L_raw=ligand_feat,
            atom_types_onehot=interp["z_t_type"],
            t=t,
            h_P=h_P,
        )

        # ── Flow matching loss ──
        loss_coord = F.mse_loss(model_out["vel_coord"], interp["u_t_coord"])
        loss_type = F.mse_loss(model_out["vel_type"], interp["u_t_type"])
        flow_loss = loss_coord + loss_type

        # ── Affinity head loss ──
        # Target: normalised reward r = (|aff| - offset) / scale
        target_reward = (abs(affinity.item()) - reward_offset) / reward_scale
        target_reward = torch.tensor(target_reward, device=device)
        affinity_loss = F.mse_loss(
            torch.sigmoid(model_out["pK_pred"]), target_reward
        )

        # Total loss
        total_loss = flow_loss + affinity_lambda * affinity_loss

        # Apply sample weight
        if weight is not None:
            total_loss = total_loss * weight

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "loss_coord": loss_coord,
            "loss_type": loss_type,
            "affinity_loss": affinity_loss,
            "pK_pred": model_out["pK_pred"].detach(),
        }

    def compute_contrastive_loss(
        self,
        pocket_pos: torch.Tensor,       # (N_P, 3)
        pocket_feat: torch.Tensor,      # (N_P, F_P)
        ligand_pos_a: torch.Tensor,     # (N_A, 3) — stronger binder
        ligand_feat_a: torch.Tensor,    # (N_A, F_L)
        ligand_types_a: torch.Tensor,   # (N_A,) int
        affinity_a: float,              # stronger (more negative)
        ligand_pos_b: torch.Tensor,     # (N_B, 3) — weaker binder
        ligand_feat_b: torch.Tensor,    # (N_B, F_L)
        ligand_types_b: torch.Tensor,   # (N_B,) int
        affinity_b: float,              # weaker (less negative)
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Contrastive ranking loss for same-pocket ligand pairs.

        Enforces pK_pred(A) > pK_pred(B) + margin when A binds stronger.
        Uses MarginRankingLoss with target = +1.
        """
        device = pocket_pos.device

        pocket_pos = subtract_com(pocket_pos)

        # Encode pocket once (shared for both ligands)
        pocket_out = self.pocket_encoder(pocket_pos, pocket_feat)
        h_P = pocket_out["h_P"]

        # Sample a shared time for both
        t = torch.rand(1, device=device).clamp(min=self.sigma_min, max=1.0 - self.sigma_min)

        # Forward pass for ligand A
        interp_a = self.forward_interpolation(
            subtract_com(ligand_pos_a), ligand_types_a, t,
            num_atom_types=self.egnn.num_atom_types,
        )
        out_a = self.egnn(
            x_L=interp_a["z_t_coord"], h_L_raw=ligand_feat_a,
            atom_types_onehot=interp_a["z_t_type"], t=t, h_P=h_P,
        )

        # Forward pass for ligand B
        interp_b = self.forward_interpolation(
            subtract_com(ligand_pos_b), ligand_types_b, t,
            num_atom_types=self.egnn.num_atom_types,
        )
        out_b = self.egnn(
            x_L=interp_b["z_t_coord"], h_L_raw=ligand_feat_b,
            atom_types_onehot=interp_b["z_t_type"], t=t, h_P=h_P,
        )

        # MarginRankingLoss: pK_pred(A) should be > pK_pred(B)
        # target = +1 means x1 should be ranked higher than x2
        target = torch.tensor([1.0], device=device)
        ranking_loss = F.margin_ranking_loss(
            out_a["pK_pred"].unsqueeze(0),
            out_b["pK_pred"].unsqueeze(0),
            target,
            margin=margin,
        )
        return ranking_loss

    # ──────────────────────────────────────────────────────────────────────
    # Reverse sampling (generation)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        pocket_pos: torch.Tensor,     # (N_P, 3)
        pocket_feat: torch.Tensor,    # (N_P, F_P)
        num_atoms: int = None,        # override number of ligand atoms
        ligand_feat_dim: int = 20,    # raw ligand feature dim
    ) -> dict:
        """Generate a molecule via 50-step Euler integration.

        If ``num_atoms`` is None, predict it from the pocket.

        Returns dict with pos (N_L, 3), atom_types (N_L,), pK_pred.
        """
        device = pocket_pos.device

        # Subtract pocket CoM
        pocket_pos = subtract_com(pocket_pos)

        # Encode pocket
        pocket_out = self.pocket_encoder(pocket_pos, pocket_feat)
        h_P = pocket_out["h_P"]

        # Predict number of ligand atoms if not given
        if num_atoms is None:
            size_pred = self.size_predictor(pocket_out["h_glob"])
            num_atoms = max(int(size_pred.item() + 0.5), 4)  # at least 4 atoms

        N_L = num_atoms

        # Start from pure noise
        z_coord = torch.randn(N_L, 3, device=device)
        z_coord = z_coord - z_coord.mean(dim=0, keepdim=True)  # zero CoM
        z_type = torch.ones(N_L, self.egnn.num_atom_types, device=device) / self.egnn.num_atom_types

        # Dummy ligand features (will be refined during sampling)
        h_L_raw = torch.zeros(N_L, ligand_feat_dim, device=device)

        dt = 1.0 / self.num_steps

        for step in range(self.num_steps):
            t_val = step * dt
            t = torch.tensor([t_val], device=device)

            out = self.egnn(
                x_L=z_coord,
                h_L_raw=h_L_raw,
                atom_types_onehot=z_type,
                t=t,
                h_P=h_P,
            )

            # Euler step
            z_coord = z_coord + out["vel_coord"] * dt
            z_type = z_type + out["vel_type"] * dt

            # Re-centre CoM
            z_coord = z_coord - z_coord.mean(dim=0, keepdim=True)

        # Decode atom types from final type vector
        atom_types = z_type.argmax(dim=-1)  # (N_L,)

        # Get final affinity prediction
        pK_pred = out["pK_pred"]

        return {
            "pos": z_coord,
            "atom_types": atom_types,
            "type_probs": F.softmax(z_type, dim=-1),
            "pK_pred": pK_pred,
            "num_atoms": N_L,
        }
