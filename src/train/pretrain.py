"""
pretrain.py — Phase A: Pretraining loop for the Flow Matching SBDD model.

200K steps on all filtered pairs (~54K) with:
  - Flow matching velocity loss (MSE)
  - Affinity head loss (MSE, λ=0.1)
  - Affinity-weighted sampling: w_i = softmax(|aff_i| / T)
  - AdamW lr=3e-4, gradient clipping at 8.0
  - Evaluation every 2K steps on validation set
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import SBDDDataset, collate_skip_none
from ..model.flow_matching import FlowMatching

logger = logging.getLogger(__name__)


def pretrain(
    model: FlowMatching,
    train_dataset: SBDDDataset,
    val_dataset: Optional[SBDDDataset] = None,
    max_steps: int = 200_000,
    batch_size: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    betas: tuple = (0.9, 0.999),
    grad_clip: float = 8.0,
    affinity_lambda: float = 0.1,
    eval_every: int = 2000,
    save_every: int = 10000,
    save_dir: str = "checkpoints",
    device: str = "cuda",
    reward_offset: float = 6.0,
    reward_scale: float = 7.0,
    start_step: int = 0,
    optimizer_state: Optional[dict] = None,
):
    """Run Phase A pretraining.

    Parameters
    ----------
    model          : FlowMatching model (pocket encoder + EGNN)
    train_dataset  : SBDDDataset for training
    val_dataset    : SBDDDataset for validation (optional)
    max_steps      : total training steps (200K default)
    batch_size     : complexes per step (4 default, ~16GB VRAM)
    lr             : learning rate (3e-4)
    grad_clip      : gradient clipping norm (8.0)
    affinity_lambda: weight on affinity MSE loss (0.1)
    eval_every     : evaluate every N steps
    save_every     : save checkpoint every N steps
    save_dir       : directory for checkpoints
    device         : "cuda" or "cpu"
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Custom collate — each sample has variable-size tensors
    # For simplicity we process one complex at a time (batch_size=1 effective)
    # and accumulate gradients over `batch_size` samples.
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4,
        pin_memory=True, collate_fn=collate_skip_none,
    )
    train_iter = iter(train_loader)

    step = start_step
    accum_loss = 0.0
    accum_flow = 0.0
    accum_aff = 0.0
    t_start = time.time()

    logger.info(f"Starting pretraining: {max_steps} steps, lr={lr}, device={device}")

    while step < max_steps:
        optimizer.zero_grad()
        batch_loss = 0.0

        for _ in range(batch_size):
            try:
                sample = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                sample = next(train_iter)

            # Skip bad samples (corrupt/missing files)
            if sample is None:
                continue

            # Move to device
            pocket_pos = sample["pocket_pos"].to(device)
            pocket_feat = sample["pocket_feat"].to(device)
            ligand_pos = sample["ligand_pos"].to(device)
            ligand_feat = sample["ligand_feat"].to(device)
            ligand_types = sample["ligand_atom_types"].to(device)
            affinity = sample["affinity"].to(device)
            weight = sample["weight"].to(device)

            losses = model.compute_loss(
                pocket_pos=pocket_pos,
                pocket_feat=pocket_feat,
                ligand_pos=ligand_pos,
                ligand_feat=ligand_feat,
                ligand_atom_types=ligand_types,
                affinity=affinity,
                weight=weight,
                affinity_lambda=affinity_lambda,
                reward_offset=reward_offset,
                reward_scale=reward_scale,
            )

            loss = losses["total_loss"] / batch_size
            loss.backward()
            batch_loss += losses["total_loss"].item()
            accum_flow += losses["flow_loss"].item()
            accum_aff += losses["affinity_loss"].item()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        step += 1
        accum_loss += batch_loss

        # ── Contrastive ranking loss (every 4th step) ──
        if step % 4 == 0 and hasattr(train_dataset, 'contrastive_pairs'):
            optimizer.zero_grad()
            pair = train_dataset.sample_contrastive_pair()
            if pair is not None:
                c_loss = model.compute_contrastive_loss(**{
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in pair.items()
                })
                (0.05 * c_loss).backward()  # λ_contrastive = 0.05
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        # ── Logging ──
        if step % 100 == 0:
            avg_loss = accum_loss / 100
            avg_flow = accum_flow / (100 * batch_size)
            avg_aff = accum_aff / (100 * batch_size)
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed

            logger.info(
                f"Step {step}/{max_steps} | "
                f"loss={avg_loss:.4f} flow={avg_flow:.4f} aff={avg_aff:.4f} | "
                f"{steps_per_sec:.1f} steps/s"
            )
            accum_loss = 0.0
            accum_flow = 0.0
            accum_aff = 0.0

        # ── Evaluation ──
        if step % eval_every == 0 and val_dataset is not None:
            val_loss = evaluate(model, val_dataset, device, reward_offset, reward_scale)
            logger.info(f"  [VAL] Step {step} | val_loss={val_loss:.4f}")
            model.train()

        # ── Checkpointing ──
        if step % save_every == 0:
            ckpt_path = save_path / f"pretrain_step{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

    # Save final checkpoint (θ₀ — frozen reference for RL)
    final_path = save_path / "pretrain_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)
    logger.info(f"Pretraining complete. Final checkpoint: {final_path}")

    return model


@torch.no_grad()
def evaluate(
    model: FlowMatching,
    val_dataset: SBDDDataset,
    device: str = "cuda",
    reward_offset: float = 6.0,
    reward_scale: float = 7.0,
    max_samples: int = 500,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n = min(len(val_dataset), max_samples)

    n_valid = 0
    for i in range(n):
        sample = val_dataset[i]
        if sample is None:
            continue

        pocket_pos = sample["pocket_pos"].to(device)
        pocket_feat = sample["pocket_feat"].to(device)
        ligand_pos = sample["ligand_pos"].to(device)
        ligand_feat = sample["ligand_feat"].to(device)
        ligand_types = sample["ligand_atom_types"].to(device)
        affinity = sample["affinity"].to(device)

        losses = model.compute_loss(
            pocket_pos=pocket_pos,
            pocket_feat=pocket_feat,
            ligand_pos=ligand_pos,
            ligand_feat=ligand_feat,
            ligand_atom_types=ligand_types,
            affinity=affinity,
            reward_offset=reward_offset,
            reward_scale=reward_scale,
        )
        total_loss += losses["total_loss"].item()
        n_valid += 1

    return total_loss / max(n_valid, 1)
