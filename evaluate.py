#!/usr/bin/env python3
"""
evaluate.py — Evaluation script for the SBDD Flow Matching model.

Runs the trained model on the held-out test set and computes:
  1. Flow matching reconstruction loss (how well the model predicts velocity)
  2. Generation quality metrics (validity, QED, Lipinski, diversity)
  3. Predicted binding affinity distribution

Usage:
    # Full evaluation on test set:
    python evaluate.py

    # Evaluate a specific checkpoint:
    python evaluate.py --checkpoint checkpoints/pretrain_step100000.pt

    # Quick evaluation (fewer samples):
    python evaluate.py --max_test_samples 100 --num_gen_mols 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


# Reuse utilities from generate.py
from generate import (
    load_model,
    featurize_pocket,
    coords_to_rdkit_mol,
    compute_mol_metrics,
    LIGAND_ATOM_TYPES,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Test-set flow matching loss evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_loss(model, test_dataset, device, max_samples=500):
    """Compute average flow matching loss on the test set.

    This measures how well the model has learned to predict the denoising
    velocity field — the core objective of Phase A pretraining.
    """
    model.eval()
    total_loss = 0.0
    total_flow = 0.0
    total_aff = 0.0
    n_valid = 0

    n = min(len(test_dataset), max_samples)
    logger.info(f"Evaluating flow matching loss on {n} test samples...")

    for i in range(n):
        sample = test_dataset[i]
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
            ligand_bonds=ligand_bonds,
        )

        loss_val = losses["total_loss"].item()
        if not (np.isnan(loss_val) or np.isinf(loss_val)):
            total_loss += loss_val
            total_flow += losses["flow_loss"].item()
            total_aff += losses["affinity_loss"].item()
            n_valid += 1

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{n} samples...")

    avg_loss = total_loss / max(n_valid, 1)
    avg_flow = total_flow / max(n_valid, 1)
    avg_aff = total_aff / max(n_valid, 1)

    return {
        "test_loss": avg_loss,
        "test_flow_loss": avg_flow,
        "test_affinity_loss": avg_aff,
        "n_evaluated": n_valid,
        "n_skipped": n - n_valid,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Generation quality evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_generation(
    model,
    test_dataset,
    device,
    num_pockets=20,
    num_mols_per_pocket=10,
):
    """Generate molecules for test pockets and evaluate quality metrics.

    For each test pocket, generates molecules and computes:
    - Validity rate (can RDKit parse it?)
    - QED distribution
    - Lipinski pass rate
    - Molecular weight distribution
    - Atom type distribution
    - Diversity (unique SMILES ratio)
    """
    model.eval()

    all_metrics = []
    all_smiles = []
    pK_preds = []
    gen_times = []

    # Collect unique pockets from test set
    seen_pockets = set()
    pocket_indices = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        if sample is None:
            continue
        pdb_id = sample.get("pdb_id", f"pocket_{i}")
        if pdb_id not in seen_pockets:
            seen_pockets.add(pdb_id)
            pocket_indices.append(i)
        if len(pocket_indices) >= num_pockets:
            break

    logger.info(f"Generating molecules for {len(pocket_indices)} test pockets "
                f"({num_mols_per_pocket} mols each)...")

    for pi, idx in enumerate(pocket_indices):
        sample = test_dataset[idx]
        if sample is None:
            continue

        pocket_pos = sample["pocket_pos"].to(device)
        pocket_feat = sample["pocket_feat"].to(device)
        pdb_id = sample.get("pdb_id", f"pocket_{idx}")

        for mi in range(num_mols_per_pocket):
            t_start = time.time()

            result = model.sample(
                pocket_pos=pocket_pos,
                pocket_feat=pocket_feat,
            )

            gen_time = time.time() - t_start
            gen_times.append(gen_time)

            pos_np = result["pos"].cpu().numpy()
            types_np = result["atom_types"].cpu().numpy()
            pK = result["pK_pred"].cpu().item()
            pK_preds.append(pK)

            # Reconstruct molecule
            mol, sanitized = coords_to_rdkit_mol(pos_np, types_np)
            metrics = compute_mol_metrics(mol, sanitized)
            metrics["pocket"] = pdb_id
            metrics["pK_pred"] = pK
            metrics["gen_time_s"] = gen_time
            all_metrics.append(metrics)

            if metrics.get("valid", False) and metrics.get("smiles", ""):
                all_smiles.append(metrics["smiles"])

        if (pi + 1) % 5 == 0:
            logger.info(f"  Completed {pi+1}/{len(pocket_indices)} pockets...")

    # ── Aggregate statistics ──
    total = len(all_metrics)
    valid = sum(1 for m in all_metrics if m.get("valid", False))
    valid_metrics = [m for m in all_metrics if m.get("valid", False)]

    stats = {
        "total_generated": total,
        "valid_count": valid,
        "validity_rate": valid / max(total, 1),
        "avg_gen_time_s": np.mean(gen_times) if gen_times else 0,
    }

    if valid_metrics:
        qeds = [m["qed"] for m in valid_metrics]
        mws = [m.get("mw", 0) for m in valid_metrics]
        logps = [m.get("logp", 0) for m in valid_metrics]
        lipinski_pass = sum(m.get("lipinski", 0) for m in valid_metrics)

        stats["qed_mean"] = np.mean(qeds)
        stats["qed_std"] = np.std(qeds)
        stats["qed_median"] = np.median(qeds)
        stats["mw_mean"] = np.mean(mws)
        stats["mw_std"] = np.std(mws)
        stats["logp_mean"] = np.mean(logps)
        stats["logp_std"] = np.std(logps)
        stats["lipinski_pass_rate"] = lipinski_pass / len(valid_metrics)
        stats["pK_pred_mean"] = np.mean(pK_preds)
        stats["pK_pred_std"] = np.std(pK_preds)

        # Diversity: fraction of unique SMILES
        unique_smiles = set(all_smiles)
        stats["unique_smiles"] = len(unique_smiles)
        stats["diversity"] = len(unique_smiles) / max(len(all_smiles), 1)

        # Atom type distribution
        atom_counts = Counter()
        for m in valid_metrics:
            smiles = m.get("smiles", "")
            for c in smiles:
                if c in "CNOS":
                    atom_counts[c] += 1
        stats["atom_distribution"] = dict(atom_counts.most_common())

    return stats, all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# 3. Print evaluation report
# ──────────────────────────────────────────────────────────────────────────────

def print_evaluation_report(test_loss_stats, gen_stats, checkpoint_path):
    """Print a comprehensive evaluation report."""
    print("\n" + "=" * 80)
    print("  SBDD FLOW MATCHING MODEL — EVALUATION REPORT")
    print("=" * 80)
    print(f"\n  Checkpoint: {checkpoint_path}")

    # Test loss
    print(f"\n  {'─' * 40}")
    print(f"  PHASE A — TEST SET LOSS")
    print(f"  {'─' * 40}")
    print(f"  Samples evaluated : {test_loss_stats['n_evaluated']}")
    print(f"  Test Loss (total) : {test_loss_stats['test_loss']:.4f}")
    print(f"  Flow Loss         : {test_loss_stats['test_flow_loss']:.4f}")
    print(f"  Affinity Loss     : {test_loss_stats['test_affinity_loss']:.4f}")

    # Generation quality
    print(f"\n  {'─' * 40}")
    print(f"  GENERATION QUALITY METRICS")
    print(f"  {'─' * 40}")
    print(f"  Total generated   : {gen_stats['total_generated']}")
    print(f"  Valid molecules    : {gen_stats['valid_count']}/{gen_stats['total_generated']} "
          f"({gen_stats['validity_rate']*100:.1f}%)")
    print(f"  Avg gen time      : {gen_stats.get('avg_gen_time_s', 0):.2f}s per molecule")

    if gen_stats.get("qed_mean") is not None:
        print(f"\n  {'Metric':<20} {'Mean':>10} {'Std':>10} {'Median':>10}")
        print(f"  {'─' * 50}")
        print(f"  {'QED':<20} {gen_stats['qed_mean']:>10.4f} {gen_stats['qed_std']:>10.4f} "
              f"{gen_stats.get('qed_median', 0):>10.4f}")
        print(f"  {'Mol Weight':<20} {gen_stats['mw_mean']:>10.1f} {gen_stats['mw_std']:>10.1f}")
        print(f"  {'LogP':<20} {gen_stats['logp_mean']:>10.2f} {gen_stats['logp_std']:>10.2f}")
        print(f"  {'pK_pred':<20} {gen_stats['pK_pred_mean']:>10.4f} {gen_stats['pK_pred_std']:>10.4f}")
        print(f"\n  Lipinski Pass Rate : {gen_stats.get('lipinski_pass_rate', 0)*100:.1f}%")
        print(f"  Unique SMILES      : {gen_stats.get('unique_smiles', 0)}")
        print(f"  Diversity          : {gen_stats.get('diversity', 0)*100:.1f}%")

        if gen_stats.get("atom_distribution"):
            print(f"\n  Atom Distribution:")
            for atom, count in gen_stats["atom_distribution"].items():
                print(f"    {atom}: {count}")

    print(f"\n{'=' * 80}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained SBDD Flow Matching model."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/rl_final.pt",
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML."
    )
    parser.add_argument(
        "--max_test_samples", type=int, default=500,
        help="Max test samples for loss evaluation (default: 500)."
    )
    parser.add_argument(
        "--num_pockets", type=int, default=20,
        help="Number of test pockets for generation evaluation (default: 20)."
    )
    parser.add_argument(
        "--num_gen_mols", type=int, default=10,
        help="Molecules to generate per pocket (default: 10)."
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results",
        help="Output directory for evaluation reports."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu."
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Output dir
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, cfg = load_model(args.config, args.checkpoint, args.device)

    # Build test dataset
    from src.data.dataset import load_and_filter_dataset, split_by_protein, SBDDDataset

    proteins, flat_pairs = load_and_filter_dataset(
        json_path=cfg["data"]["dataset_json"],
        aff_min=cfg["affinity"]["min"],
        aff_max=cfg["affinity"]["max"],
    )
    _, _, test_pairs = split_by_protein(
        flat_pairs,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        seed=cfg["split"]["seed"],
    )
    test_dataset = SBDDDataset(
        test_pairs, cfg["data"]["base_dir"],
        reward_offset=cfg["affinity"]["reward_offset"],
        reward_scale=cfg["affinity"]["reward_scale"],
    )
    logger.info(f"Test dataset: {len(test_dataset)} pairs")

    # ── Step 1: Test set loss ──
    logger.info("=" * 60)
    logger.info("STEP 1: Evaluating flow matching loss on test set...")
    logger.info("=" * 60)
    test_loss_stats = evaluate_test_loss(
        model, test_dataset, args.device,
        max_samples=args.max_test_samples,
    )

    # ── Step 2: Generation quality ──
    logger.info("=" * 60)
    logger.info("STEP 2: Evaluating generation quality...")
    logger.info("=" * 60)
    gen_stats, all_metrics = evaluate_generation(
        model, test_dataset, args.device,
        num_pockets=args.num_pockets,
        num_mols_per_pocket=args.num_gen_mols,
    )

    # ── Print report ──
    print_evaluation_report(test_loss_stats, gen_stats, args.checkpoint)

    # ── Save results to JSON ──
    results_json = {
        "checkpoint": args.checkpoint,
        "test_loss": test_loss_stats,
        "generation": {k: v for k, v in gen_stats.items()
                      if not isinstance(v, np.floating)},
    }
    # Convert numpy types for JSON serialisation
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list) or isinstance(obj, tuple):
            return [convert_numpy(v) for v in obj]
        # Catch RDKit SWIG C++ objects (like _vectSt6vectorIiSaIiEE)
        if hasattr(obj, '__class__') and obj.__class__.__name__.startswith('_vect'):
            try:
                return [convert_numpy(x) for x in obj]
            except:
                return str(obj)
        return obj

    results_path = output_path / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(convert_numpy(results_json), f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save per-molecule details
    details_path = output_path / "per_molecule_details.json"
    serialisable_metrics = []
    for m in all_metrics:
        entry = {k: v for k, v in m.items() if k != "mol"}
        serialisable_metrics.append(convert_numpy(entry))
    with open(details_path, "w") as f:
        json.dump(serialisable_metrics, f, indent=2)
    logger.info(f"Per-molecule details saved to {details_path}")


if __name__ == "__main__":
    main()
