"""
compute_marginal.py — Compute empirical atom type marginal distribution.

Run once before training:
    python scripts/compute_marginal.py

Outputs:
    marginal_prior.npy  — shape (6,) float32, sums to 1.0
    [C, N, O, S, F, Cl]
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.data.dataset import SBDDDataset

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl"]
NUM_TYPES = len(ATOM_TYPES)


def compute_marginal(dataset_json: str, base_dir: str, max_samples: int = None):
    dataset = SBDDDataset(dataset_json, base_dir=base_dir, split="train")
    counts = np.zeros(NUM_TYPES, dtype=np.float64)
    total = 0

    limit = max_samples or len(dataset)
    for i in range(limit):
        sample = dataset[i]
        if sample is None:
            continue
        types = sample["ligand_atom_types"].numpy()  # (N_L,)
        for t in types:
            if 0 <= t < NUM_TYPES:
                counts[t] += 1
                total += 1
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{limit} samples...")

    if total == 0:
        raise RuntimeError("No valid samples found")

    marginal = (counts / total).astype(np.float32)
    print(f"\nAtom type marginal distribution over {total:,} atoms:")
    for i, atype in enumerate(ATOM_TYPES):
        print(f"  {atype:3s}: {marginal[i]*100:.2f}%")

    out_path = "marginal_prior.npy"
    np.save(out_path, marginal)
    print(f"\nSaved to {out_path}")
    return marginal


if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    marginal = compute_marginal(
        dataset_json=cfg["data"]["dataset_json"],
        base_dir=cfg["data"]["base_dir"],
    )
