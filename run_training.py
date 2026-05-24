#!/usr/bin/env python3
"""
run_training.py — Master launcher for the SBDD training pipeline.

Reads configs/default.yaml, constructs all modules, and runs the
training phases in order:

  Phase A: Pretraining     (200K steps, flow matching + affinity head)
  Phase B: RL Fine-tuning  (50K steps, DDPO with entropy reward)
  Phase C: Inpainting RL   (20K steps, scaffold-constrained, optional)

Usage:
  # Run everything (A → B → C):
  python run_training.py

  # Run only Phase A:
  python run_training.py --phase A

  # Run only Phase B (requires pretrained checkpoint):
  python run_training.py --phase B --checkpoint checkpoints/pretrain_final.pt

  # Resume Phase A from a checkpoint:
  python run_training.py --phase A --checkpoint checkpoints/pretrain_step100000.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
import glob
import re

# ── Setup logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger("run_training")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load and return the YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def build_model(cfg: dict, device: str) -> "FlowMatching":
    """Construct the full FlowMatching model from config."""
    from src.model.pocket_encoder import PocketEncoder
    from src.model.egnn import SE3EGNN
    from src.model.flow_matching import FlowMatching

    pocket_encoder = PocketEncoder(
        in_dim=40,  # 16 elem + 21 aa + 1 backbone + 1 bfactor + 1 centroid_dist
        hidden_dim=cfg["pocket_encoder"]["hidden_dim"],
        num_layers=cfg["pocket_encoder"]["num_layers"],
        knn_k=cfg["pocket"]["knn_k"],
    )

    egnn = SE3EGNN(
        ligand_in_dim=20,  # 16 elem + 1 aromatic + 1 degree + 1 charge + 1 ring
        pocket_dim=cfg["egnn"]["hidden_dim"],
        hidden_dim=cfg["egnn"]["hidden_dim"],
        num_layers=cfg["egnn"]["num_layers"],
        num_heads=cfg["egnn"]["num_heads"],
        num_atom_types=cfg["ligand"]["num_atom_types"],
        knn_k=cfg["pocket"]["knn_k"],
        dropout=cfg["egnn"]["dropout"],
    )

    model = FlowMatching(
        pocket_encoder=pocket_encoder,
        egnn=egnn,
        num_steps=cfg["flow"]["num_steps_sample"],
        sigma_min=cfg["flow"]["sigma_min"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built: {n_params:,} parameters")

    return model.to(device)


def build_datasets(cfg: dict):
    """Load data, filter, split, and return datasets."""
    from src.data.dataset import (
        load_and_filter_dataset,
        split_by_protein,
        SBDDDataset,
    )

    # Load and filter
    proteins, flat_pairs = load_and_filter_dataset(
        json_path=cfg["data"]["dataset_json"],
        aff_min=cfg["affinity"]["min"],
        aff_max=cfg["affinity"]["max"],
    )

    # Protein-level split
    train_pairs, val_pairs, test_pairs = split_by_protein(
        flat_pairs,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        seed=cfg["split"]["seed"],
    )

    # Build PyTorch datasets
    base_dir = cfg["data"]["base_dir"]
    reward_offset = cfg["affinity"]["reward_offset"]
    reward_scale = cfg["affinity"]["reward_scale"]

    train_dataset = SBDDDataset(
        train_pairs, base_dir,
        reward_offset=reward_offset,
        reward_scale=reward_scale,
    )
    val_dataset = SBDDDataset(
        val_pairs, base_dir,
        reward_offset=reward_offset,
        reward_scale=reward_scale,
    )

    logger.info(
        f"Datasets ready: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_pairs)}"
    )
    logger.info(
        f"Contrastive pairs available: "
        f"{len(train_dataset.contrastive_pairs)} proteins with ≥2 ligands"
    )

    return train_dataset, val_dataset, train_pairs


def get_latest_checkpoint(prefix: str, save_dir: str = "checkpoints") -> str | None:
    """Find the latest checkpoint file matching a prefix (e.g. 'pretrain_step')."""
    path = Path(save_dir)
    if not path.exists():
        return None
    
    # E.g. find all checkpoints/pretrain_step*.pt
    files = list(path.glob(f"{prefix}*.pt"))
    if not files:
        return None
        
    # Extract the step number using regex to find the highest one
    latest_file = None
    max_step = -1
    
    for f in files:
        # Check if it's the final checkpoint first
        if f.name == f"{prefix.split('_')[0]}_final.pt":
            return str(f)
            
        match = re.search(r"step(\d+)\.pt", f.name)
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                latest_file = str(f)
                
    return latest_file


def run_phase_a(
    model, cfg, train_dataset, val_dataset, device, checkpoint=None
):
    """Phase A: Pretraining with flow matching + affinity head."""
    from src.train.pretrain import pretrain

    logger.info("=" * 60)
    logger.info("PHASE A: PRETRAINING")
    logger.info("=" * 60)

    start_step = 0
    opt_state = None

    # Resume from checkpoint if provided
    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_step = ckpt.get("step", 0)
        opt_state = ckpt.get("optimizer_state_dict", None)

    pt_cfg = cfg["pretrain"]

    model = pretrain(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_steps=pt_cfg["max_steps"],
        batch_size=pt_cfg["batch_size"],
        lr=pt_cfg["lr"],
        weight_decay=pt_cfg["weight_decay"],
        betas=tuple(pt_cfg["betas"]),
        grad_clip=pt_cfg["grad_clip"],
        affinity_lambda=pt_cfg["affinity_loss_lambda"],
        type_loss_weight=pt_cfg.get("type_loss_weight", 5.0),
        eval_every=pt_cfg["eval_every"],
        save_every=pt_cfg["save_every"],
        save_dir="checkpoints",
        device=device,
        reward_offset=cfg["affinity"]["reward_offset"],
        reward_scale=cfg["affinity"]["reward_scale"],
        start_step=start_step,
        optimizer_state=opt_state,
    )

    logger.info("Phase A complete. Checkpoint: checkpoints/pretrain_final.pt")
    return model


def run_phase_b(model, cfg, train_pairs, device, checkpoint=None):
    """Phase B: DDPO RL fine-tuning."""
    from src.train.rl_finetune import rl_finetune

    logger.info("=" * 60)
    logger.info("PHASE B: RL FINE-TUNING (DDPO)")
    logger.info("=" * 60)

    pretrained_ckpt = checkpoint or "checkpoints/pretrain_final.pt"
    if not Path(pretrained_ckpt).exists():
        logger.error(
            f"Pretrained checkpoint not found: {pretrained_ckpt}. "
            f"Run Phase A first."
        )
        return model

    # Load the pretrained weights into the model
    ckpt = torch.load(pretrained_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    rl_cfg = cfg["rl"]

    model = rl_finetune(
        model=model,
        pretrained_checkpoint=pretrained_ckpt,
        train_pairs=train_pairs,
        base_dir=cfg["data"]["base_dir"],
        max_steps=rl_cfg["max_steps"],
        lr=rl_cfg["lr"],
        batch_pockets=rl_cfg["batch_pockets"],
        mols_per_pocket=rl_cfg["mols_per_pocket"],
        top_k=rl_cfg["top_k"],
        kl_beta_start=rl_cfg["kl_beta_start"],
        kl_beta_end=rl_cfg["kl_beta_end"],
        save_dir="checkpoints",
        device=device,
        reward_offset=cfg["affinity"]["reward_offset"],
        reward_scale=cfg["affinity"]["reward_scale"],
    )

    logger.info("Phase B complete. Checkpoint: checkpoints/rl_final.pt")
    return model


def run_phase_c(model, cfg, train_pairs, device, checkpoint=None):
    """Phase C: Inpainting RL (optional)."""
    from src.train.rl_inpaint import rl_inpaint

    logger.info("=" * 60)
    logger.info("PHASE C: INPAINTING RL (SCAFFOLD-CONSTRAINED)")
    logger.info("=" * 60)

    rl_ckpt = checkpoint or "checkpoints/rl_final.pt"
    if not Path(rl_ckpt).exists():
        # Fall back to pretrained
        rl_ckpt = "checkpoints/pretrain_final.pt"
    if not Path(rl_ckpt).exists():
        logger.error("No checkpoint found. Run Phase A or B first.")
        return model

    ckpt = torch.load(rl_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model = rl_inpaint(
        model=model,
        pretrained_checkpoint=rl_ckpt,
        train_pairs=train_pairs,
        base_dir=cfg["data"]["base_dir"],
        max_steps=20_000,
        lr=1e-5,
        save_dir="checkpoints",
        device=device,
    )

    logger.info("Phase C complete.")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="SBDD Training Pipeline Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py                          # Run A → B → C
  python run_training.py --phase A                # Phase A only
  python run_training.py --phase B --checkpoint checkpoints/pretrain_final.pt
  python run_training.py --phase AB               # Phase A then B
  python run_training.py --config configs/custom.yaml
        """,
    )
    parser.add_argument(
        "--phase", default="ABC",
        help="Which phases to run: 'A', 'B', 'C', 'AB', 'BC', or 'ABC' (default: ABC)"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", default="auto",
        help="Path to checkpoint to resume from. Set to 'auto' to auto-detect the latest, or 'none' to start fresh."
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device ('cuda' or 'cpu'). Default: from config."
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Override max training steps for Phase A. Useful for warm-up runs."
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # DDP Initialization
    import os
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        logger.info(f"DDP Initialized. Rank: {local_rank}, Device: {device}")
    else:
        # Device
        device = args.device or cfg["hardware"]["device"]
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Build model
    model = build_model(cfg, device)

    # Build datasets
    train_dataset, val_dataset, train_pairs = build_datasets(cfg)

    phases = args.phase.upper()

    # ── Resolve Auto Checkpoint ──
    start_ckpt = args.checkpoint
    if start_ckpt is not None and start_ckpt.lower() == "none":
        start_ckpt = None
    elif start_ckpt == "auto":
        # Guess the prefix based on the first phase being run
        first_phase = phases[0]
        prefix = "pretrain_step" if first_phase == "A" else "rl_step"
        start_ckpt = get_latest_checkpoint(prefix)
        if start_ckpt:
            logger.info(f"Auto-detected latest checkpoint: {start_ckpt}")
        else:
            logger.info(f"No previous checkpoints found for Phase {first_phase}. Starting fresh.")

    # ── Execute phases ──
    if "A" in phases:
        if args.max_steps is not None:
            cfg["pretrain"]["max_steps"] = args.max_steps
        model = run_phase_a(
            model, cfg, train_dataset, val_dataset, device,
            checkpoint=start_ckpt if "A" == phases[0] else None,
        )

    if "B" in phases:
        model = run_phase_b(
            model, cfg, train_pairs, device,
            checkpoint=start_ckpt if "B" == phases[0] else None,
        )

    if "C" in phases:
        model = run_phase_c(
            model, cfg, train_pairs, device,
            checkpoint=start_ckpt if "C" == phases[0] else None,
        )

    logger.info("=" * 60)
    logger.info("ALL REQUESTED PHASES COMPLETE")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Start backend:  uvicorn api.main:app --port 8000")
    logger.info("  2. Start frontend: cd frontend && npm run dev")
    logger.info("  3. Or use CLI:     python -m src.inference.pipeline --pdb protein.pdb --checkpoint checkpoints/rl_final.pt")


if __name__ == "__main__":
    main()
