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
    grad_clip: float = 1.0,
    affinity_lambda: float = 0.1,
    type_loss_weight: float = 5.0,
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

    # Mixed precision: bf16 on A100 for ~2x speedup
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    if use_amp:
        logger.info(f"Mixed precision enabled: {amp_dtype}")

    # ── DDP Setup ──
    import os
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in [-1, 0]

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler = None

    # Disjoint union collation — DataLoader handles true batching now
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(sampler is None), num_workers=2,
        pin_memory=True, collate_fn=collate_skip_none, drop_last=True, sampler=sampler
    )
    train_iter = iter(train_loader)

    step = start_step
    accum_loss = 0.0
    accum_flow = 0.0
    accum_aff = 0.0
    t_start = time.time()

    logger.info(f"Starting pretraining: {max_steps} steps, lr={lr}, batch_size={batch_size}, device={device}")

    # Learning rate warmup scheduler: linearly ramp from lr/100 to lr
    # over the first 1000 steps to prevent gradient explosion
    warmup_steps = 1000
    def get_lr_scale(current_step):
        if current_step < warmup_steps:
            return 0.01 + 0.99 * (current_step / warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

    if is_main_process:
        from tqdm import tqdm
        pbar = tqdm(total=max_steps, initial=start_step, desc="Pretraining Phase A")
    else:
        pbar = None

    while step < max_steps:
        try:
            sample = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            sample = next(train_iter)

        # Skip bad batches (all samples were None)
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
        ligand_bonds = sample["ligand_bonds"].to(device)
        batch_P = sample["batch_P"].to(device)
        batch_L = sample["batch_L"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            losses = model(
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
                type_loss_weight=type_loss_weight,
                ligand_bonds=ligand_bonds,
                batch_P=batch_P,
                batch_L=batch_L,
            )

        # Guard: skip step if loss is NaN to prevent poisoning model weights
        if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
            if is_main_process:
                logger.warning(f"NaN/Inf loss at step {step+1}, skipping batch")
            continue

        # bf16 doesn't need GradScaler, but fp16 does
        if use_amp and amp_dtype == torch.float16:
            scaler.scale(losses["total_loss"]).backward()
        else:
            losses["total_loss"].backward()

        # Check for NaN gradients — clip_grad_norm_ does NOT handle NaN,
        # it just propagates it, which permanently corrupts model weights
        grad_ok = True
        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                grad_ok = False
                break

        if not grad_ok:
            if is_main_process:
                logger.warning(f"NaN/Inf gradient at step {step+1}, skipping update")
            optimizer.zero_grad()
            continue

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_amp and amp_dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        step += 1
        if is_main_process:
            pbar.update(1)
        accum_loss += losses["total_loss"].item()
        accum_flow += losses["flow_loss"].item()
        accum_aff += losses["affinity_loss"].item()

        # ── Contrastive ranking loss (every 4th step) ──
        if step % 4 == 0 and hasattr(train_dataset, 'contrastive_pairs'):
            optimizer.zero_grad()
            pair = train_dataset.sample_contrastive_pair()
            if pair is not None:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    c_loss = model.compute_contrastive_loss(**{
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in pair.items()
                    })
                (0.05 * c_loss).backward()  # λ_contrastive = 0.05
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        # ── Logging ──
        if step % 100 == 0 and is_main_process:
            avg_loss = accum_loss / 100
            avg_flow = accum_flow / 100
            avg_aff = accum_aff / 100
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "flow": f"{avg_flow:.4f}",
                "aff": f"{avg_aff:.4f}",
                "it/s": f"{steps_per_sec:.1f}"
            })

            logger.info(
                f"Step {step}/{max_steps} | "
                f"loss={avg_loss:.4f} flow={avg_flow:.4f} aff={avg_aff:.4f} | "
                f"{steps_per_sec:.1f} steps/s"
            )
            accum_loss = 0.0
            accum_flow = 0.0
            accum_aff = 0.0

        # ── Evaluation ──
        if step % eval_every == 0 and val_dataset is not None and is_main_process:
            # We must unwrap model for evaluation because DDP forward is train-only
            eval_model = model.module if local_rank != -1 else model
            val_loss = evaluate(eval_model, val_dataset, device, reward_offset, reward_scale)
            logger.info(f"  [VAL] Step {step} | val_loss={val_loss:.4f}")
            model.train()

        # ── Checkpointing ──
        if step % save_every == 0 and is_main_process:
            ckpt_path = save_path / f"pretrain_step{step}.pt"
            save_model = model.module if local_rank != -1 else model
            torch.save({
                "step": step,
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

    # Save final checkpoint (θ₀ — frozen reference for RL)
    if is_main_process:
        final_path = save_path / "pretrain_final.pt"
        save_model = model.module if local_rank != -1 else model
        torch.save({
            "step": step,
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, final_path)
        logger.info(f"Pretraining complete. Final checkpoint: {final_path}")
        if pbar: pbar.close()

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
        ligand_bonds = sample["ligand_bonds"].to(device)

        losses = model.compute_loss(
            pocket_pos=pocket_pos,
            pocket_feat=pocket_feat,
            ligand_pos=ligand_pos,
            ligand_feat=ligand_feat,
            ligand_atom_types=ligand_types,
            affinity=affinity,
            reward_offset=reward_offset,
            reward_scale=reward_scale,
            ligand_bonds=ligand_bonds,
        )
        total_loss += losses["total_loss"].item()
        n_valid += 1

    return total_loss / max(n_valid, 1)
