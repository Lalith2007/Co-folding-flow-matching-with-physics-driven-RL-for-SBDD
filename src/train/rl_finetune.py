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

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppress noisy C++ valence errors

logger = logging.getLogger(__name__)


def validate_rl(
    model: FlowMatching,
    val_pairs: list,
    base_dir: str,
    device: str,
    step: int,
    reward_oracle: RewardOracle,
    val_pockets: int = 5,
    mols_per_pocket: int = 10,
    log_path: str = "rl_validation_log.jsonl",
) -> dict:
    """Lightweight validation: generate molecules for a few pockets, report QED/SA/reward."""
    import json, random
    from generate import coords_to_rdkit_mol
    from rdkit.Chem import QED
    from ..data.featurizer import PocketFeaturizer

    pocket_featurizer = PocketFeaturizer()
    base_dir_path = Path(base_dir)

    model.eval()
    sample_pairs = random.sample(val_pairs, min(val_pockets, len(val_pairs)))

    all_qed, all_sa, all_reward, all_valid = [], [], [], 0
    total_mols = 0

    with torch.no_grad():
        for pair in sample_pairs:
            pocket_path = base_dir_path / pair["pocket_path"]
            try:
                pocket_data = pocket_featurizer.featurize(str(pocket_path))
                if pocket_data["pos"] is None or pocket_data["pos"].shape[0] == 0:
                    continue
                pocket_pos  = pocket_data["pos"].to(device)
                pocket_feat = pocket_data["feat"].to(device)
            except Exception:
                continue

            for _ in range(mols_per_pocket):
                total_mols += 1
                try:
                    gen = model.sample(pocket_pos, pocket_feat, temperature=1.0, num_steps=20)
                    mol, sanitized = coords_to_rdkit_mol(
                        gen["pos"].cpu().numpy(),
                        gen["atom_types"].cpu().numpy(),
                    )
                    if not sanitized or mol is None:
                        continue
                    all_valid += 1
                    all_qed.append(RewardOracle.compute_qed(mol))
                    all_sa.append(RewardOracle.compute_sa_raw(mol))
                    rd = reward_oracle.compute_rl_reward(
                        mol=mol,
                        pK_pred=gen["pK_pred"],
                        pocket_path=str(pocket_path),
                        pocket_pos_updated=gen.get("pocket_pos_updated"),
                        rl_round=step,
                    )
                    all_reward.append(rd["total_reward"])
                except Exception:
                    pass

    model.train()

    stats = {
        "step": step,
        "val_validity":    all_valid / max(total_mols, 1),
        "val_qed_mean":    float(sum(all_qed)   / len(all_qed))   if all_qed   else 0.0,
        "val_sa_mean":     float(sum(all_sa)    / len(all_sa))    if all_sa    else 0.0,
        "val_reward_mean": float(sum(all_reward) / len(all_reward)) if all_reward else 0.0,
        "val_n_valid":     all_valid,
        "val_n_total":     total_mols,
    }

    logger.info(
        f"[VAL step={step}] "
        f"Validity={stats['val_validity']*100:.1f}% ({all_valid}/{total_mols}) | "
        f"QED={stats['val_qed_mean']:.4f} | "
        f"SA={stats['val_sa_mean']:.4f} | "
        f"Reward={stats['val_reward_mean']:.4f}"
    )

    # Append to JSONL log for later plotting
    with open(log_path, "a") as f:
        f.write(json.dumps(stats) + "\n")

    return stats


def rl_finetune(
    model: FlowMatching,
    pretrained_checkpoint: str,
    train_pairs: list,
    base_dir: str,
    val_pairs: list = None,
    resume_checkpoint: str = None,
    max_steps: int = 50_000,
    lr: float = 1e-5,
    batch_pockets: int = 32,
    mols_per_pocket: int = 100,
    top_k: int = 10,
    kl_beta_start: float = 0.01,
    kl_beta_end: float = 0.001,
    vina_every_n: int = 10,
    curriculum_every: int = 500,
    val_every: int = 500,
    save_every: int = 1000,
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
    val_pairs            : held-out validation pairs (optional, subset of val split)
    resume_checkpoint    : path to intermediate rl_step*.pt to resume optimizer & step
    max_steps            : total RL steps
    lr                   : learning rate (1e-5, 10× smaller than pretrain)
    batch_pockets        : pockets per RL round
    mols_per_pocket      : molecules generated per pocket
    top_k                : top-k molecules kept for gradient update
    kl_beta_start/end    : KL penalty β annealing range
    val_every            : run validation every N steps (default: 500)
    save_every           : save checkpoint every N steps (default: 1000)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load pretrained model as frozen reference (θ₀) for KL penalty
    model_ref = copy.deepcopy(model)
    ref_ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model_ref.load_state_dict(ref_ckpt["model_state_dict"], strict=False)
    model_ref = model_ref.to(device)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Restore step and optimizer if resuming
    step = 0
    if resume_checkpoint and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location=device)
        step = ckpt.get("step", 0)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass
        logger.info(f"Resumed RL training at step {step} from {resume_checkpoint}")

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
    difficulty_levels = [-7.0, -8.0, -9.0, -10.0, -11.0]
    current_difficulty = min(step // curriculum_every, len(difficulty_levels) - 1)
    threshold = difficulty_levels[current_difficulty]
    rl_pairs = get_rl_subset(train_pairs, threshold=threshold)

    t_start = time.time()

    logger.info(
        f"RL fine-tuning step {step}/{max_steps}, lr={lr}, "
        f"β={kl_beta_start}→{kl_beta_end}, difficulty={current_difficulty} ({threshold} kcal/mol)"
    )

    from tqdm import tqdm
    is_main_process = True
    if torch.distributed.is_initialized():
        is_main_process = (torch.distributed.get_rank() == 0)

    pbar = tqdm(total=max_steps, initial=step, desc="RL Phase B", disable=not is_main_process)

    # Need featurizer to load pocket data on the fly (with in-memory cache)
    from ..data.featurizer import PocketFeaturizer
    pocket_featurizer = PocketFeaturizer()
    pocket_cache = {}
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
            # Load pocket data on the fly (cached in RAM)
            pocket_path = base_dir_path / pair["pocket_path"]
            pocket_key = str(pocket_path)
            if pocket_key not in pocket_cache:
                try:
                    pocket_data = pocket_featurizer.featurize(pocket_key)
                    if pocket_data["pos"] is None or pocket_data["pos"].shape[0] == 0:
                        continue
                    pocket_cache[pocket_key] = (
                        pocket_data["pos"].to(device),
                        pocket_data["feat"].to(device),
                    )
                except Exception:
                    continue
            pocket_pos, pocket_feat = pocket_cache[pocket_key]

            # ── Step 1: Generate molecules (no grad, 20-step Euler rollout for high speed) ──
            # Temperature annealing: 1.2 (explore) → 0.8 (exploit drug-like molecules)
            # This matches RLHF best practice: high temperature early, low temperature late
            progress = min(step / max_steps, 1.0)
            temperature = 1.2 - 0.4 * progress  # 1.2 → 0.8 linearly
            with torch.no_grad():
                candidates = []
                for _ in range(mols_per_pocket):
                    gen = model.sample(pocket_pos, pocket_feat, temperature=temperature, num_steps=20)
                    candidates.append(gen)

            # ── Step 2: Score with FULL multi-objective reward ──
            # This prevents reward hacking: the model can't maximize
            # proxy affinity at the expense of drug-likeness.
            rewards = []
            _log_this_batch = (step % 10 == 0)  # log every 10 steps for diagnosis
            for gen_idx, gen in enumerate(candidates):
                # Reconstruct RDKit molecule for chemical metrics
                try:
                    from generate import coords_to_rdkit_mol

                    pos_np = gen["pos"].cpu().numpy()
                    types_np = gen["atom_types"].cpu().numpy()

                    mol, sanitized = coords_to_rdkit_mol(pos_np, types_np)
                    if not sanitized or mol is None:
                        if _log_this_batch and gen_idx == 0:
                            logger.warning(f"[REWARD DEBUG] Step {step}: mol reconstruction FAILED (sanitized={sanitized})")
                        rewards.append(0.0)
                        continue

                    reward_dict = reward_oracle.compute_rl_reward(
                        mol=mol,
                        pK_pred=gen["pK_pred"],
                        pocket_path=str(pocket_path),
                        pocket_pos_updated=gen.get("pocket_pos_updated"),
                        rl_round=step,
                    )
                    r = reward_dict["total_reward"]

                    # ── Diagnostic log: print component rewards to find R=0 root cause ──
                    if _log_this_batch and gen_idx == 0:
                        gate = reward_dict.get('gate_reason', '?')
                        cq   = reward_dict.get('chem_quality', -1)
                        rq   = reward_dict.get('r_qed', -1)
                        rp   = reward_dict.get('r_proxy', -1)
                        cr   = reward_dict.get('carbon_ratio', -1)
                        nr   = reward_dict.get('nitrogen_ratio', -1)
                        logger.info(
                            f"[REWARD DEBUG] step={step} R={r:.4f} "
                            f"chem_q={cq:.3f} r_qed={rq:.3f} r_proxy={rp:.3f} "
                            f"C_ratio={cr:.2f} N_ratio={nr:.2f} gate={gate}"
                        )

                except Exception as exc:
                    # Log first error per batch so we can diagnose the root cause
                    if _log_this_batch:
                        logger.warning(f"[REWARD DEBUG] Step {step}: exception in reward: {type(exc).__name__}: {exc}")
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
                h_L_raw = torch.zeros(N_L, 4, device=device)  # 4 non-element features: aromatic, degree, charge, ring

                # Use 10 ODE steps for RL policy gradient backprop (linear flow enables
                # fast 10-step BPTT with lower gradient variance and 2x faster wall-clock speed)
                rl_num_steps = 10
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

                    # Approximate log p: ||v_coord||² proxy for coordinate policy
                    # (Full change-of-variables trace is expensive;
                    #  using velocity norm as proxy for policy gradient)
                    vel = out["vel_coord"]
                    log_prob = log_prob - 0.5 * (vel ** 2).sum() * dt

                    z_coord = z_coord + vel * dt
                    z_type = z_type + out["vel_type"] * dt
                    z_coord = z_coord - z_coord.mean(0, keepdim=True)

                # ── Atom type REINFORCE via re-run z_type (with live gradient) ──
                # Use z_type from the re-run ODE — NOT z_type_final from gen[].
                #
                # z_type_final comes from the no-grad rollout, so requires_grad=False.
                # Using it makes the entire atom_type_log_prob a constant in the
                # computation graph, giving vel_type_head ZERO REINFORCE gradient.
                #
                # z_type from the re-run ODE accumulates through vel_type_head
                # (trainable), so requires_grad=True → gradient flows.
                #
                # Yes, sampled_types come from the original no-grad trajectory
                # (a different random seed), creating a trajectory mismatch.
                # But a biased gradient >> zero gradient, and this is no worse
                # than the coord log_prob approximation (which also uses different
                # initial noise and is acknowledged as a proxy).
                norm_probs = F.softmax(z_type, dim=-1)  # z_type from re-run: HAS grad
                atom_type_log_prob = torch.log(norm_probs + 1e-8)
                sampled_types = gen["atom_types"].to(device)
                log_prob = log_prob + 0.3 * atom_type_log_prob[
                    torch.arange(N_L, device=device), sampled_types
                ].sum()

                # ── Entropy reward: r_entropy = -H(p_type) ──
                entropy = -(norm_probs * (norm_probs + 1e-8).log()).sum(dim=-1).mean()
                import math
                r_entropy = max(0.0, 1.0 - entropy.item() / math.log(model.egnn.num_atom_types))
                if r > 0:
                    r = r + 0.1 * r_entropy

                # ── KL penalty against θ₀ (Coordinate geometry only) ──
                # Anchors 3D pocket conformation to preserve 51.5% PoseBusters validity,
                # while allowing atom types to freely explore Nitrogen and Oxygen.
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

                kl_coord = F.mse_loss(cur_out["vel_coord"], ref_out["vel_coord"])
                kl_loss  = kl_coord

                # ── Continuous Pharma Stoichiometry Loss ──
                # Provides direct, unconditional gradient toward 18% N, 10% O, 70% C
                # on EVERY step regardless of what atom types were sampled.
                #
                # CRITICAL: Use z_type from the re-run ODE loop (has requires_grad=True
                # because it accumulated via trainable vel_type_head).
                # DO NOT use z_type_final from gen[] — that came from the no-grad
                # rollout, so requires_grad=False and backward() would be a no-op.
                #
                # BUG FIX (reduction): Use 'sum' NOT 'batchmean'.
                # F.kl_div batchmean divides by first dim. For shape (6,), it divides
                # by 6, making stoich 6× weaker than intended.
                #
                # BUG FIX (weight): 1.5 (was 0.15).
                # rl_loss = -log_prob * r ≈ 30–50. 0.15 * 0.02 = 0.003 → 0.006% rel.
                # 1.5 gives ~5% relative gradient — enough to emerge Nitrogen.
                p_target = torch.tensor([0.70, 0.18, 0.10, 0.01, 0.005, 0.005], device=device)
                mean_pred_p = F.softmax(z_type, dim=-1).mean(dim=0)  # z_type from re-run: HAS grad
                stoich_loss = F.kl_div(
                    torch.log(mean_pred_p + 1e-8),
                    p_target,
                    reduction='sum'   # NOT batchmean — batchmean divides by 6 for shape (6,)
                )

                # ── DDPO loss: -log_p * R + β * KL_coord + λ * Stoich ──
                rl_loss = -log_prob * r + beta * kl_loss + 1.5 * stoich_loss

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
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                        p.grad /= torch.distributed.get_world_size()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1
        pbar.update(1)

        # ── Logging (rank 0 only) ──
        if is_main_process and step % 10 == 0 and n_mols > 0:
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
            if is_main_process:
                logger.info(
                    f"  Curriculum update: difficulty={current_difficulty}, "
                    f"threshold={threshold}, pairs={len(rl_pairs)}"
                )

        # ── Validation every val_every steps (rank 0 only) ──
        if is_main_process and step % val_every == 0 and val_pairs:
            val_log_path = str(Path(save_dir) / "rl_validation_log.jsonl")
            validate_rl(
                model=model,
                val_pairs=val_pairs,
                base_dir=base_dir,
                device=device,
                step=step,
                reward_oracle=reward_oracle,
                val_pockets=5,
                mols_per_pocket=10,
                log_path=val_log_path,
            )

        # ── Checkpointing (rank 0 only) ──
        if is_main_process and step % save_every == 0:
            ckpt_path = save_path / f"rl_step{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"  Saved RL checkpoint: {ckpt_path}")

    # Final save (rank 0 only)
    if is_main_process:
        final_path = save_path / "rl_final.pt"
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
        }, final_path)
        logger.info(f"RL fine-tuning complete. Final checkpoint: {final_path}")
    pbar.close()

    return model
