"""
rl_finetune.py — Phase B: DDPO RL Fine-tuning.

Denoising Diffusion Policy Optimization through the full flow chain.

Training loop:
  1. Sample B=32 pockets from RL subset
  2. Generate 100 molecules per pocket (50-step flow, no_grad for speed)
  3. Score all with R(m, pocket):
     - Proxy r_proxy: every round (fast)
     - Vina r_vina: every 10 rounds (slow)
  4. Select top-10 per pocket as seeds
  5. Re-run flow with gradient tracking on top-k
  6. Compute DDPO loss + KL penalty
  7. Update θ with Adam lr=1e-5
  8. Curriculum: increase pocket difficulty every 500 rounds

Key: ∇_θ J = E[ Σ_t ∇_θ log p_θ(z_{t+1}|z_t, P) · R ]
     L = L_RL − β · KL(θ || θ₀)
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import SBDDDataset, get_rl_subset
from ..model.flow_matching import FlowMatching
from ..model.reward import RewardOracle
from ..model.utils import CosineBetaSchedule

logger = logging.getLogger(__name__)


def rl_finetune(
    model: FlowMatching,
    pretrained_checkpoint: str,
    train_pairs: list,
    base_dir: str,
    max_steps: int = 50_000,
    lr: float = 1e-5,
    batch_pockets: int = 32,
    mols_per_pocket: int = 100,
    top_k: int = 10,
    kl_beta_start: float = 0.01,
    kl_beta_end: float = 0.001,
    vina_every_n: int = 10,
    curriculum_every: int = 500,
    save_every: int = 5000,
    save_dir: str = "checkpoints",
    device: str = "cuda",
    reward_offset: float = 6.0,
    reward_scale: float = 7.0,
):
    """Run Phase B DDPO RL fine-tuning.

    Parameters
    ----------
    model                : FlowMatching model (initialized from pretrained)
    pretrained_checkpoint: path to θ₀ checkpoint for KL penalty
    train_pairs          : training pairs from dataset
    base_dir             : server base directory for file access
    max_steps            : total RL steps (50K)
    lr                   : learning rate (1e-5, 10× smaller than pretrain)
    batch_pockets        : pockets per RL round
    mols_per_pocket      : molecules generated per pocket
    top_k                : top-k molecules kept for gradient update
    kl_beta_start/end    : KL penalty β annealing range
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load pretrained model as frozen reference (θ₀) for KL penalty
    model_ref = copy.deepcopy(model)
    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model_ref.load_state_dict(ckpt["model_state_dict"], strict=False)
    model_ref = model_ref.to(device)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # KL β schedule
    beta_schedule = CosineBetaSchedule(kl_beta_start, kl_beta_end, max_steps)

    # Reward oracle — pharma-grade safety gates
    reward_oracle = RewardOracle(
        vina_every_n=vina_every_n,
        min_carbon_ratio=0.40,
        max_nitrogen_ratio=0.35,
        max_nn_bonds=2,
        max_sa_score=6.0,
        max_ring_nitrogen=2,
    )

    # RL curriculum: start with easy pockets, increase difficulty
    rl_pairs = get_rl_subset(train_pairs, threshold=-11.0)

    # Difficulty levels (affinity thresholds)
    difficulty_levels = [-7.0, -8.0, -9.0, -10.0, -11.0]
    current_difficulty = 0

    step = 0
    t_start = time.time()

    logger.info(
        f"Starting RL fine-tuning: {max_steps} steps, lr={lr}, "
        f"β={kl_beta_start}→{kl_beta_end}"
    )

    from tqdm import tqdm
    pbar = tqdm(total=max_steps, initial=0, desc="RL Phase B")

    # Need featurizer to load pocket data on the fly
    from ..data.featurizer import PocketFeaturizer
    pocket_featurizer = PocketFeaturizer()
    base_dir_path = Path(base_dir)

    while step < max_steps:
        optimizer.zero_grad()

        # Current β for KL penalty
        beta = beta_schedule(step)

        # Sample pockets for this round (PDBBind weighted 2x as step progresses)
        import random
        progress = min(step / max(max_steps, 1), 1.0)
        # PDBBind weight linearly increases from 1x to 2x
        pdbbind_weight = 1.0 + progress  # 1.0 → 2.0
        weights = [
            pdbbind_weight if p.get("dataset") == "pdbbind" else 1.0
            for p in rl_pairs
        ]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        sample_size = min(batch_pockets, len(rl_pairs))
        pocket_sample = random.choices(rl_pairs, weights=probs, k=sample_size)

        total_reward = 0.0
        total_rl_loss = 0.0
        total_kl_loss = 0.0
        n_mols = 0

        for pair in pocket_sample:
            # Load pocket data on the fly
            pocket_path = base_dir_path / pair["pocket_path"]
            try:
                pocket_data = pocket_featurizer.featurize(str(pocket_path))
                if pocket_data["pos"] is None or pocket_data["pos"].shape[0] == 0:
                    continue
                pocket_pos = pocket_data["pos"].to(device)
                pocket_feat = pocket_data["feat"].to(device)
            except Exception:
                continue

            # ── Step 1: Generate molecules (no grad for speed) ──
            with torch.no_grad():
                candidates = []
                for _ in range(mols_per_pocket):
                    gen = model.sample(pocket_pos, pocket_feat)
                    candidates.append(gen)

            # ── Step 2: Score with FULL multi-objective reward ──
            # This prevents reward hacking: the model can't maximize
            # proxy affinity at the expense of drug-likeness.
            rewards = []
            for gen in candidates:
                # Reconstruct RDKit molecule for chemical metrics
                try:
                    from rdkit import Chem
                    from rdkit.Geometry import Point3D
                    from ..data.featurizer import LIGAND_ATOM_TYPES

                    pos_np = gen["pos"].cpu().numpy()
                    types_np = gen["atom_types"].cpu().numpy()

                    mol = Chem.RWMol()
                    conf = Chem.Conformer(len(pos_np))
                    for i, (p, t) in enumerate(zip(pos_np, types_np)):
                        elem = LIGAND_ATOM_TYPES[t] if t < len(LIGAND_ATOM_TYPES) else "C"
                        atom_num = Chem.GetPeriodicTable().GetAtomicNumber(elem)
                        mol.AddAtom(Chem.Atom(atom_num))
                        conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
                    mol.AddConformer(conf, assignId=True)

                    try:
                        from rdkit.Chem import rdDetermineBonds
                        rdDetermineBonds.DetermineBonds(mol.GetMol())
                        mol = Chem.RWMol(mol.GetMol())
                    except Exception:
                        pass

                    Chem.SanitizeMol(mol)
                    reward_dict = reward_oracle.compute_rl_reward(
                        mol=mol.GetMol(),
                        pK_pred=gen["pK_pred"],
                        pocket_path=str(pocket_path),
                        pocket_pos_updated=gen.get("pocket_pos_updated"),
                        rl_round=step,
                    )
                    r = reward_dict["total_reward"]
                except Exception:
                    # Hard penalty for molecules that can't be reconstructed
                    # (prevents reward hacking with invalid chemistry)
                    # MUST BE >= 0.0 to prevent unbaselined REINFORCE from exploding
                    r = 0.0
                rewards.append(r)

            # ── Step 3: Select top-k ──
            reward_tensor = torch.tensor(rewards)
            _, top_indices = reward_tensor.topk(min(top_k, len(rewards)))

            # ── Step 4: Re-run with gradients for top-k ──
            for idx in top_indices:
                gen = candidates[idx.item()]
                r = rewards[idx.item()]

                # Re-run the flow chain WITH gradients
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

                # Use fewer ODE steps for RL (20 vs 50) — faster, still good enough
                rl_num_steps = 20
                dt = 1.0 / rl_num_steps
                log_prob = torch.tensor(0.0, device=device)

                for s in range(rl_num_steps):
                    t_val = s * dt
                    t = torch.tensor([t_val], device=device)

                    out = model.egnn(
                        x_L=z_coord,
                        h_L_raw=h_L_raw,
                        atom_types_onehot=z_type,
                        t=t,
                        h_P=h_P,
                    )

                    # Approximate log p: ||v_θ||² proxy
                    # (Full change-of-variables trace is expensive;
                    #  using velocity norm as proxy for policy gradient)
                    vel = out["vel_coord"]
                    log_prob = log_prob - 0.5 * (vel ** 2).sum() * dt

                    z_coord = z_coord + vel * dt
                    z_type = z_type + out["vel_type"] * dt
                    z_coord = z_coord - z_coord.mean(0, keepdim=True)

                # ── Entropy reward: r_entropy = -H(softmax(v_type)) ──
                # Lower entropy = more confident atom types = better
                type_probs = F.softmax(z_type, dim=-1).clamp(min=1e-8)
                entropy = -(type_probs * type_probs.log()).sum(dim=-1).mean()
                # Normalise to [0, 1]: max entropy = log(10) ≈ 2.3
                import math
                r_entropy = max(0.0, 1.0 - entropy.item() / math.log(model.egnn.num_atom_types))
                # Add entropy to reward (weighted 0.1) — only for valid molecules
                if r > 0:
                    r = r + 0.1 * r_entropy

                # ── KL penalty against θ₀ ──
                with torch.no_grad():
                    ref_enc = model_ref.pocket_encoder(pocket_pos, pocket_feat)
                    h_P_ref = ref_enc["h_P"]

                    z_coord_ref = z_coord.detach().clone()
                    t_mid = torch.tensor([0.5], device=device)

                    ref_out = model_ref.egnn(
                        x_L=z_coord_ref,
                        h_L_raw=h_L_raw,
                        atom_types_onehot=z_type.detach(),
                        t=t_mid,
                        h_P=h_P_ref,
                    )
                    cur_out = model.egnn(
                        x_L=z_coord_ref,
                        h_L_raw=h_L_raw,
                        atom_types_onehot=z_type.detach(),
                        t=t_mid,
                        h_P=h_P,
                    )

                kl_loss = F.mse_loss(cur_out["vel_coord"], ref_out["vel_coord"])

                # ── DDPO loss: -log_p * R + β * KL ──
                rl_loss = -log_prob * r + beta * kl_loss

                rl_loss.backward()

                total_reward += r
                total_rl_loss += rl_loss.item()
                total_kl_loss += kl_loss.item()
                n_mols += 1

        if n_mols > 0:
            # Average gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= n_mols

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1
        pbar.update(1)

        # ── Logging ──
        if step % 10 == 0 and n_mols > 0:
            avg_r = total_reward / n_mols
            avg_rl = total_rl_loss / n_mols
            avg_kl = total_kl_loss / n_mols
            elapsed = time.time() - t_start

            pbar.set_postfix({
                "R": f"{avg_r:.3f}",
                "rl": f"{avg_rl:.4f}",
                "kl": f"{avg_kl:.4f}"
            })

            logger.info(
                f"RL Step {step}/{max_steps} | "
                f"R={avg_r:.3f} rl_loss={avg_rl:.4f} kl={avg_kl:.4f} "
                f"β={beta:.4f} | {elapsed:.0f}s"
            )

        # ── Curriculum ──
        if step % curriculum_every == 0:
            current_difficulty = min(
                current_difficulty + 1, len(difficulty_levels) - 1
            )
            threshold = difficulty_levels[current_difficulty]
            rl_pairs = get_rl_subset(train_pairs, threshold=threshold)
            logger.info(
                f"  Curriculum update: difficulty={current_difficulty}, "
                f"threshold={threshold}, pairs={len(rl_pairs)}"
            )

        # ── Checkpointing ──
        if step % save_every == 0:
            ckpt_path = save_path / f"rl_step{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"  Saved RL checkpoint: {ckpt_path}")

    # Final save
    final_path = save_path / "rl_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
    }, final_path)
    logger.info(f"RL fine-tuning complete. Final checkpoint: {final_path}")
    pbar.close()

    return model
