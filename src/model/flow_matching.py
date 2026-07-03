"""
flow_matching.py -- Rectified Flow Matching backbone (SOTA v2).

Coordinate flow: continuous, MSE velocity loss.
Atom type flow:  DISCRETE flow matching -- x1 prediction + cross-entropy loss.

Training:
    Coordinates: z_t = (1-t)*z_noise + t*x_data,  u_t = x_data - z_noise
    Types:       z_t = (1-t)*marginal + t*one_hot,  loss: CE(type_logits, true_idx)
    Self-conditioning applied 50% of training steps (FlowMol3).

Inference:
    Start from N(0,I) coords and marginal type distribution.
    Type velocity: vel_type = (softmax(type_logits) - z_t) / (1-t+eps)
    50 Euler steps with self-conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .egnn import SE3EGNN
from .pocket_encoder import PocketEncoder
from .utils import subtract_com


class FlowMatching(nn.Module):
    """Full Flow Matching model wrapping PocketEncoder + SE3EGNN."""

    def __init__(
        self,
        pocket_encoder: PocketEncoder,
        egnn: SE3EGNN,
        num_steps: int = 50,
        sigma_min: float = 1e-5,
    ):
        super().__init__()
        self.pocket_encoder = pocket_encoder
        self.egnn = egnn
        self.num_steps = num_steps
        self.sigma_min = sigma_min

        self.size_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward_interpolation(
        self,
        x_data: torch.Tensor,
        type_data: torch.Tensor,
        t: torch.Tensor,
        num_atom_types: int = 6,
        marginal: torch.Tensor = None,
    ) -> dict:
        """Noised state z_t and training targets.

        Coordinates: continuous (MSE velocity loss).
        Types: discrete x1-prediction (CE loss) with marginal prior.
        """
        device = x_data.device

        z_noise_coord = torch.randn_like(x_data)
        z_noise_coord = z_noise_coord - z_noise_coord.mean(dim=0, keepdim=True)
        z_t_coord = (1 - t) * z_noise_coord + t * x_data
        u_t_coord = x_data - z_noise_coord

        type_onehot = F.one_hot(type_data, num_atom_types).float()
        if marginal is not None:
            N_L = x_data.size(0)
            prior = marginal.to(device).unsqueeze(0).expand(N_L, -1)
        else:
            prior = torch.ones_like(type_onehot) / num_atom_types
        z_t_type = (1 - t) * prior + t * type_onehot

        return {
            "z_t_coord": z_t_coord,
            "z_t_type": z_t_type,
            "u_t_coord": u_t_coord,
            "z_noise_coord": z_noise_coord,
            "type_data": type_data,
        }

    def forward(self, *args, mode="flow", **kwargs):
        if mode == "flow":
            return self.compute_loss(*args, **kwargs)
        elif mode == "contrastive":
            return self.compute_contrastive_loss(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_loss(
        self,
        pocket_pos: torch.Tensor,
        pocket_feat: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_feat: torch.Tensor,
        ligand_atom_types: torch.Tensor,
        affinity: torch.Tensor,
        weight: torch.Tensor = None,
        affinity_lambda: float = 0.1,
        reward_offset: float = 6.0,
        reward_scale: float = 7.0,
        type_loss_weight: float = 1.0,
        ligand_bonds: torch.Tensor = None,
        bond_dropout: float = 0.5,
        batch_P: torch.Tensor = None,
        batch_L: torch.Tensor = None,
        marginal: torch.Tensor = None,
        sc_prob: float = 0.5,
    ) -> dict:
        """CE loss on x1 prediction + MSE coord loss + self-conditioning."""
        device = pocket_pos.device

        pocket_pos  = torch.nan_to_num(pocket_pos,  nan=0.0)
        pocket_feat = torch.nan_to_num(pocket_feat, nan=0.0)
        ligand_pos  = torch.nan_to_num(ligand_pos,  nan=0.0)
        ligand_feat = torch.nan_to_num(ligand_feat, nan=0.0)

        pocket_pos = subtract_com(pocket_pos, batch_P)
        ligand_pos = subtract_com(ligand_pos, batch_L)

        t = torch.rand(1, device=device).clamp(min=self.sigma_min, max=1.0 - self.sigma_min)

        z_noise_pocket = torch.randn_like(pocket_pos) * 0.1
        z_t_pocket = pocket_pos + z_noise_pocket
        u_t_pocket = -z_noise_pocket

        pocket_out = self.pocket_encoder(z_t_pocket, pocket_feat, batch_P=batch_P)
        h_P = pocket_out["h_P"]

        interp = self.forward_interpolation(
            ligand_pos, ligand_atom_types, t,
            num_atom_types=self.egnn.num_atom_types,
            marginal=marginal,
        )

        if self.training and ligand_bonds is not None and torch.rand(1).item() < bond_dropout:
            ligand_bonds = torch.zeros_like(ligand_bonds)

        ligand_feat_clean = ligand_feat[:, 16:]
        if self.training and torch.rand(1).item() < 0.5:
            ligand_feat_clean = torch.zeros_like(ligand_feat_clean)

        # Self-Conditioning (FlowMol3): 50% of steps run no-grad forward,
        # feed prediction back as sc_prior for the main pass.
        N_L = interp["z_t_coord"].size(0)
        use_sc = self.training and (torch.rand(1).item() < sc_prob)
        if use_sc:
            with torch.no_grad():
                sc_out = self.egnn(
                    x_L=interp["z_t_coord"],
                    h_L_raw=ligand_feat_clean,
                    atom_types_onehot=interp["z_t_type"],
                    t=t, h_P=h_P, ligand_bonds=ligand_bonds,
                    batch_L=batch_L, batch_P=batch_P,
                    sc_prior=torch.zeros(N_L, self.egnn.num_atom_types, device=device),
                )
                sc_prior = F.softmax(sc_out["type_logits"], dim=-1).detach()
        else:
            sc_prior = torch.zeros(N_L, self.egnn.num_atom_types, device=device)

        model_out = self.egnn(
            x_L=interp["z_t_coord"],
            h_L_raw=ligand_feat_clean,
            atom_types_onehot=interp["z_t_type"],
            t=t, h_P=h_P, ligand_bonds=ligand_bonds,
            batch_L=batch_L, batch_P=batch_P,
            sc_prior=sc_prior,
        )

        loss_coord = F.mse_loss(model_out["vel_coord"], interp["u_t_coord"])
        # Discrete flow: CE on x1 prediction
        loss_type = F.cross_entropy(model_out["type_logits"], interp["type_data"])
        loss_pocket_coord = F.mse_loss(model_out["vel_pocket"], u_t_pocket)
        flow_loss = loss_coord + type_loss_weight * loss_type + loss_pocket_coord

        if affinity.dim() == 0:
            target_reward = torch.tensor(
                (abs(affinity.item()) - reward_offset) / reward_scale, device=device)
        else:
            target_reward = (affinity.abs() - reward_offset) / reward_scale

        pK_pred = model_out["pK_pred"]
        affinity_loss = F.mse_loss(torch.sigmoid(pK_pred), target_reward)
        total_loss = flow_loss + affinity_lambda * affinity_loss

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
        pocket_pos, pocket_feat,
        ligand_pos_a, ligand_feat_a, ligand_types_a, affinity_a,
        ligand_pos_b, ligand_feat_b, ligand_types_b, affinity_b,
        margin=1.0, ligand_bonds_a=None, ligand_bonds_b=None,
    ) -> torch.Tensor:
        """Contrastive ranking loss: pK_pred(A) > pK_pred(B) + margin."""
        device = pocket_pos.device
        pocket_pos = subtract_com(pocket_pos)
        pocket_out = self.pocket_encoder(pocket_pos, pocket_feat)
        h_P = pocket_out["h_P"]
        t = torch.rand(1, device=device).clamp(min=self.sigma_min, max=1.0 - self.sigma_min)

        def _fwd(lpos, lfeat, ltypes, lbonds):
            interp = self.forward_interpolation(
                subtract_com(lpos), ltypes, t,
                num_atom_types=self.egnn.num_atom_types)
            return self.egnn(
                x_L=interp["z_t_coord"], h_L_raw=lfeat[:, 16:],
                atom_types_onehot=interp["z_t_type"], t=t, h_P=h_P,
                ligand_bonds=lbonds)

        out_a = _fwd(ligand_pos_a, ligand_feat_a, ligand_types_a, ligand_bonds_a)
        out_b = _fwd(ligand_pos_b, ligand_feat_b, ligand_types_b, ligand_bonds_b)

        target = torch.tensor([1.0], device=device)
        return F.margin_ranking_loss(
            out_a["pK_pred"].unsqueeze(0),
            out_b["pK_pred"].unsqueeze(0),
            target, margin=margin)

    @torch.no_grad()
    def sample(
        self,
        pocket_pos: torch.Tensor,
        pocket_feat: torch.Tensor,
        num_atoms: int = None,
        ligand_feat_dim: int = 4,
        temperature: float = 0.8,
        marginal: torch.Tensor = None,
    ) -> dict:
        """Generate a molecule via 50-step Euler ODE (SOTA v2).

        Uses: marginal prior init, x1-prediction velocity, self-conditioning.
        """
        device = pocket_pos.device
        pocket_pos = subtract_com(pocket_pos)
        pocket_out = self.pocket_encoder(pocket_pos, pocket_feat)
        h_P = pocket_out["h_P"]

        if num_atoms is None:
            num_atoms = int(torch.randint(20, 35, (1,)).item())
        N_L = num_atoms

        z_coord = torch.randn(N_L, 3, device=device)
        z_coord = z_coord - z_coord.mean(dim=0, keepdim=True)

        if marginal is not None:
            z_type = marginal.to(device).unsqueeze(0).expand(N_L, -1).clone().float()
        else:
            z_type = torch.ones(N_L, self.egnn.num_atom_types, device=device) / self.egnn.num_atom_types

        h_L_raw = torch.zeros(N_L, ligand_feat_dim, device=device)
        dt = 1.0 / self.num_steps
        sc_prior = torch.zeros(N_L, self.egnn.num_atom_types, device=device)

        for step in range(self.num_steps):
            t_val = step * dt
            t = torch.tensor([t_val], device=device)

            out = self.egnn(
                x_L=z_coord, h_L_raw=h_L_raw,
                atom_types_onehot=z_type, t=t, h_P=h_P,
                ligand_bonds=None, sc_prior=sc_prior,
            )

            z_coord = z_coord + out["vel_coord"] * dt
            pocket_pos = pocket_pos + out["vel_pocket"] * dt
            z_coord = z_coord - z_coord.mean(dim=0, keepdim=True)

            # Discrete flow: x1 prediction velocity
            pred_x1 = F.softmax(out["type_logits"], dim=-1)
            vel_type = (pred_x1 - z_type) / (1.0 - t_val + 1e-8)
            z_type = z_type + vel_type * dt
            sc_prior = pred_x1

        type_probs = F.softmax(z_type / temperature, dim=-1)
        atom_types = torch.multinomial(type_probs, num_samples=1).squeeze(-1)

        return {
            "pos": z_coord,
            "atom_types": atom_types,
            "type_probs": type_probs,
            "pK_pred": out["pK_pred"],
            "num_atoms": N_L,
            "pocket_pos_updated": pocket_pos.detach(),
        }
