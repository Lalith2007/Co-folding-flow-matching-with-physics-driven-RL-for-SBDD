"""
rl_inpaint.py — Inpainting RL: scaffold-constrained molecule generation.

Given a known scaffold (e.g., a core ring system), we mask its atoms and
only allow the flow model to generate/optimise the functional groups.
This is done by zeroing the velocity field on masked (scaffold) atoms
during both the generation and the DDPO gradient-tracking passes.

Usage:
    Fix the core scaffold → generate R-groups → score with reward oracle
    → backprop through the unmasked flow steps only.

This is a variant of Phase B RL that alternates with de novo generation.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import get_rl_subset
from ..model.flow_matching import FlowMatching
from ..model.reward import RewardOracle
from ..model.utils import CosineBetaSchedule

logger = logging.getLogger(__name__)


def rl_inpaint(
    model: FlowMatching,
    pretrained_checkpoint: str,
    train_pairs: list,
    base_dir: str,
    max_steps: int = 20_000,
    lr: float = 1e-5,
    batch_pockets: int = 16,
    mols_per_pocket: int = 50,
    top_k: int = 5,
    kl_beta_start: float = 0.01,
    kl_beta_end: float = 0.001,
    scaffold_ratio: float = 0.5,
    save_every: int = 5000,
    save_dir: str = "checkpoints",
    device: str = "cuda",
):
    """Run Inpainting RL fine-tuning.

    Parameters
    ----------
    model              : FlowMatching model
    pretrained_checkpoint : θ₀ for KL penalty
    train_pairs        : training pairs
    base_dir           : server data directory
    scaffold_ratio     : fraction of atoms to fix as scaffold (0.5 = 50%)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Frozen reference
    model_ref = copy.deepcopy(model)
    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model_ref.load_state_dict(ckpt["model_state_dict"])
    model_ref = model_ref.to(device)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    beta_schedule = CosineBetaSchedule(kl_beta_start, kl_beta_end, max_steps)
    reward_oracle = RewardOracle()

    rl_pairs = get_rl_subset(train_pairs, threshold=-9.0)

    step = 0
    t_start = time.time()

    logger.info(
        f"Starting Inpainting RL: {max_steps} steps, "
        f"scaffold_ratio={scaffold_ratio}"
    )

    from tqdm import tqdm
    pbar = tqdm(total=max_steps, initial=0, desc="RL Phase C (Inpaint)")

    while step < max_steps:
        optimizer.zero_grad()
        beta = beta_schedule(step)

        pocket_sample = random.sample(
            rl_pairs, min(batch_pockets, len(rl_pairs))
        )

        total_reward = 0.0
        total_loss = 0.0
        n_mols = 0

        for pair in pocket_sample:
            pocket_pos = pair.get("_pocket_pos")
            pocket_feat = pair.get("_pocket_feat")
            if pocket_pos is None:
                continue

            pocket_pos = pocket_pos.to(device)
            pocket_feat = pocket_feat.to(device)

            # ── Generate with scaffold mask ──
            with torch.no_grad():
                candidates = []
                for _ in range(mols_per_pocket):
                    gen = _sample_with_scaffold_mask(
                        model, pocket_pos, pocket_feat,
                        scaffold_ratio=scaffold_ratio,
                    )
                    candidates.append(gen)

            # Score and select top-k
            rewards = [
                reward_oracle.compute_proxy_reward(gen["pK_pred"])
                for gen in candidates
            ]
            reward_tensor = torch.tensor(rewards)
            _, top_indices = reward_tensor.topk(min(top_k, len(rewards)))

            # ── Re-run with gradients (only on unmasked atoms) ──
            for idx in top_indices:
                gen = candidates[idx.item()]
                r = rewards[idx.item()]
                scaffold_mask = gen["scaffold_mask"]  # (N_L,) bool

                model.train()
                pocket_enc = model.pocket_encoder(pocket_pos, pocket_feat)
                h_P = pocket_enc["h_P"]

                N_L = gen["num_atoms"]
                z_coord = torch.randn(N_L, 3, device=device)
                z_coord = z_coord - z_coord.mean(0, keepdim=True)
                z_type = torch.ones(
                    N_L, model.egnn.num_atom_types, device=device
                ) / model.egnn.num_atom_types
                h_L_raw = torch.zeros(N_L, 20, device=device)

                dt = 1.0 / model.num_steps
                log_prob = torch.tensor(0.0, device=device)

                for s in range(model.num_steps):
                    t_val = s * dt
                    t = torch.tensor([t_val], device=device)

                    out = model.egnn(
                        x_L=z_coord, h_L_raw=h_L_raw,
                        atom_types_onehot=z_type, t=t, h_P=h_P,
                    )

                    vel = out["vel_coord"]

                    # ── Zero velocity on scaffold atoms ──
                    vel = vel.clone()
                    vel[scaffold_mask] = 0.0

                    vel_type = out["vel_type"].clone()
                    vel_type[scaffold_mask] = 0.0

                    log_prob = log_prob - 0.5 * (vel ** 2).sum() * dt

                    z_coord = z_coord + vel * dt
                    z_type = z_type + vel_type * dt
                    z_coord = z_coord - z_coord.mean(0, keepdim=True)

                # KL penalty
                with torch.no_grad():
                    ref_enc = model_ref.pocket_encoder(pocket_pos, pocket_feat)
                    t_mid = torch.tensor([0.5], device=device)
                    ref_out = model_ref.egnn(
                        x_L=z_coord.detach(), h_L_raw=h_L_raw,
                        atom_types_onehot=z_type.detach(),
                        t=t_mid, h_P=ref_enc["h_P"],
                    )
                    cur_out = model.egnn(
                        x_L=z_coord.detach(), h_L_raw=h_L_raw,
                        atom_types_onehot=z_type.detach(),
                        t=t_mid, h_P=h_P,
                    )

                kl_loss = F.mse_loss(cur_out["vel_coord"], ref_out["vel_coord"])
                rl_loss = -log_prob * r + beta * kl_loss

                rl_loss.backward()
                total_reward += r
                total_loss += rl_loss.item()
                n_mols += 1

        if n_mols > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= n_mols
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1
        pbar.update(1)

        if step % 10 == 0 and n_mols > 0:
            avg_r = total_reward / n_mols
            avg_loss = total_loss / n_mols
            
            pbar.set_postfix({
                "R": f"{avg_r:.3f}",
                "loss": f"{avg_loss:.4f}",
                "β": f"{beta:.4f}"
            })

            logger.info(
                f"Inpaint RL Step {step}/{max_steps} | "
                f"R={avg_r:.3f} loss={avg_loss:.4f} "
                f"β={beta:.4f}"
            )

        if step % save_every == 0:
            ckpt_path = save_path / f"inpaint_step{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
            }, ckpt_path)
            logger.info(f"  Saved: {ckpt_path}")

    pbar.close()
    return model


def _sample_with_scaffold_mask(
    model: FlowMatching,
    pocket_pos: torch.Tensor,
    pocket_feat: torch.Tensor,
    scaffold_ratio: float = 0.5,
    ligand_feat_dim: int = 20,
) -> dict:
    """Generate a molecule with a random scaffold mask.

    scaffold_ratio of the atoms are designated as 'scaffold' and their
    velocity is zeroed during integration (they stay near their initial
    positions, simulating a fixed core).
    """
    device = pocket_pos.device

    pocket_out = model.pocket_encoder(pocket_pos, pocket_feat)
    h_P = pocket_out["h_P"]

    size_pred = model.size_predictor(pocket_out["h_glob"])
    N_L = int(torch.randint(20, 35, (1,)).item())

    # Create scaffold mask: randomly select scaffold_ratio of atoms
    n_scaffold = max(1, int(N_L * scaffold_ratio))
    scaffold_indices = torch.randperm(N_L)[:n_scaffold]
    scaffold_mask = torch.zeros(N_L, dtype=torch.bool, device=device)
    scaffold_mask[scaffold_indices] = True

    z_coord = torch.randn(N_L, 3, device=device)
    z_coord = z_coord - z_coord.mean(0, keepdim=True)
    z_type = torch.ones(
        N_L, model.egnn.num_atom_types, device=device
    ) / model.egnn.num_atom_types
    h_L_raw = torch.zeros(N_L, ligand_feat_dim, device=device)

    dt = 1.0 / model.num_steps

    for s in range(model.num_steps):
        t_val = s * dt
        t = torch.tensor([t_val], device=device)

        out = model.egnn(
            x_L=z_coord, h_L_raw=h_L_raw,
            atom_types_onehot=z_type, t=t, h_P=h_P,
        )

        vel = out["vel_coord"]
        vel_type = out["vel_type"]

        # Zero velocity on scaffold atoms
        vel[scaffold_mask] = 0.0
        vel_type[scaffold_mask] = 0.0

        z_coord = z_coord + vel * dt
        z_type = z_type + vel_type * dt
        z_coord = z_coord - z_coord.mean(0, keepdim=True)

    return {
        "pos": z_coord,
        "atom_types": z_type.argmax(dim=-1),
        "pK_pred": out["pK_pred"],
        "num_atoms": N_L,
        "scaffold_mask": scaffold_mask,
    }
