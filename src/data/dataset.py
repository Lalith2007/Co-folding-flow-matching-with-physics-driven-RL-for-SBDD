"""
dataset.py — Data loading, affinity filtering, and protein-level splitting
for the RL-Guided Flow Diffusion SBDD pipeline.

The JSON file (final_dataset.json) contains only *paths* to pocket and ligand
files on the server.  This module loads those paths, filters by affinity range
[-13, -6] kcal/mol, performs protein-level train/val/test splitting, and wraps
everything in a PyTorch Dataset that reads the actual molecular files on the
fly via the featurizer.
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .featurizer import PocketFeaturizer, LigandFeaturizer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. JSON loading & affinity filtering
# ──────────────────────────────────────────────────────────────────────────────

def load_and_filter_dataset(
    json_path: str | Path,
    aff_min: float = -13.0,
    aff_max: float = -6.0,
) -> Tuple[Dict[str, dict], List[dict]]:
    """Load the master JSON and filter pocket-ligand pairs by affinity.

    Parameters
    ----------
    json_path : path to ``final_dataset.json``
    aff_min   : lower bound of affinity range (kcal/mol, inclusive)
    aff_max   : upper bound of affinity range (kcal/mol, inclusive)

    Returns
    -------
    proteins : dict  keyed by pdb_id, each with filtered sources
    flat_pairs : list of dicts, one per (pocket, ligand, affinity) triple
    """
    with open(json_path, "r") as f:
        raw: Dict[str, dict] = json.load(f)

    proteins: Dict[str, dict] = {}
    flat_pairs: List[dict] = []
    n_total = 0
    n_kept = 0

    for pdb_id, entry in raw.items():
        filtered_sources = []
        for src in entry["sources"]:
            n_total += 1
            aff = src["affinity"]
            if aff_min <= aff <= aff_max:
                n_kept += 1
                pair = {
                    "pdb_id": pdb_id,
                    "protein_path": entry["protein_path"],
                    "pocket_path": src["pocket_path"],
                    "ligand_path": src["ligand_path"],
                    "dataset": src["dataset"],
                    "affinity": aff,
                    "original_id": src["original_id"],
                }
                flat_pairs.append(pair)
                filtered_sources.append(src)

        if filtered_sources:
            proteins[pdb_id] = {
                "pdb_id": pdb_id,
                "protein_path": entry["protein_path"],
                "sources": filtered_sources,
            }

    logger.info(
        f"Affinity filter [{aff_min}, {aff_max}]: "
        f"kept {n_kept}/{n_total} pairs "
        f"across {len(proteins)} proteins "
        f"(removed {n_total - n_kept})"
    )
    return proteins, flat_pairs


# ──────────────────────────────────────────────────────────────────────────────
# 2. Protein-level train / val / test split (no data leakage)
# ──────────────────────────────────────────────────────────────────────────────

def split_by_protein(
    flat_pairs: List[dict],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split pairs by *protein* to prevent data leakage.

    Returns train_pairs, val_pairs, test_pairs.
    """
    # Gather unique pdb_ids
    pdb_ids = sorted({p["pdb_id"] for p in flat_pairs})
    rng = np.random.RandomState(seed)
    rng.shuffle(pdb_ids)

    n = len(pdb_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(pdb_ids[:n_train])
    val_ids = set(pdb_ids[n_train : n_train + n_val])
    test_ids = set(pdb_ids[n_train + n_val :])

    train_pairs = [p for p in flat_pairs if p["pdb_id"] in train_ids]
    val_pairs = [p for p in flat_pairs if p["pdb_id"] in val_ids]
    test_pairs = [p for p in flat_pairs if p["pdb_id"] in test_ids]

    logger.info(
        f"Split: train={len(train_pairs)} pairs ({len(train_ids)} proteins), "
        f"val={len(val_pairs)} pairs ({len(val_ids)} proteins), "
        f"test={len(test_pairs)} pairs ({len(test_ids)} proteins)"
    )
    return train_pairs, val_pairs, test_pairs


def get_rl_subset(
    pairs: List[dict],
    threshold: float = -11.0,
) -> List[dict]:
    """Select pockets for RL curriculum based on mean affinity per protein.

    A protein is included if the *mean* affinity of all its known ligands
    is below the threshold (i.e., the pocket is a 'hard' target on average).
    All pairs for qualifying proteins are returned.
    """
    from collections import defaultdict

    # Group pairs by protein
    protein_pairs: Dict[str, List[dict]] = defaultdict(list)
    for p in pairs:
        protein_pairs[p["pdb_id"]].append(p)

    # Compute mean affinity per protein
    qualifying_ids = set()
    for pdb_id, plist in protein_pairs.items():
        mean_aff = sum(p["affinity"] for p in plist) / len(plist)
        if mean_aff <= threshold:
            qualifying_ids.add(pdb_id)

    rl_pairs = [p for p in pairs if p["pdb_id"] in qualifying_ids]
    logger.info(
        f"RL subset: {len(rl_pairs)} pairs from {len(qualifying_ids)} proteins "
        f"with mean affinity <= {threshold}"
    )
    return rl_pairs


# ──────────────────────────────────────────────────────────────────────────────
# 3. Reward normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise_reward(
    affinity: float,
    offset: float = 6.0,
    scale: float = 7.0,
) -> float:
    """Map affinity from [-13, -6] to [1, 0].

    r = (|affinity| - offset) / scale
    -13 → (13 - 6) / 7 = 1.0   (strong binder, high reward)
    -6  → (6 - 6) / 7  = 0.0   (weak binder, low reward)
    """
    return (abs(affinity) - offset) / scale


# ──────────────────────────────────────────────────────────────────────────────
# 4. Contrastive pair utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_intra_protein_spread(
    proteins: Dict[str, dict],
) -> Dict[str, float]:
    """Compute affinity spread per protein for contrastive ranking."""
    spreads = {}
    for pdb_id, entry in proteins.items():
        affs = [s["affinity"] for s in entry["sources"]]
        if len(affs) >= 2:
            spreads[pdb_id] = max(affs) - min(affs)
    return spreads


# ──────────────────────────────────────────────────────────────────────────────
# 5. PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SBDDDataset(Dataset):
    """PyTorch Dataset for SBDD pocket-ligand pairs.

    Each __getitem__ call reads the pocket and ligand files from the server
    (via ``base_dir / pair["pocket_path"]`` etc.) and featurizes them into
    tensors suitable for the SE(3)-EGNN.

    Parameters
    ----------
    pairs     : list of dicts from ``load_and_filter_dataset``
    base_dir  : root directory on the server containing all sub-datasets
    pocket_featurizer : PocketFeaturizer instance
    ligand_featurizer : LigandFeaturizer instance
    reward_offset / reward_scale : for reward normalisation
    """

    def __init__(
        self,
        pairs: List[dict],
        base_dir: str | Path,
        pocket_featurizer: Optional[PocketFeaturizer] = None,
        ligand_featurizer: Optional[LigandFeaturizer] = None,
        reward_offset: float = 6.0,
        reward_scale: float = 7.0,
    ):
        self.pairs = pairs
        self.base_dir = Path(base_dir)
        self.pocket_feat = pocket_featurizer or PocketFeaturizer()
        self.ligand_feat = ligand_featurizer or LigandFeaturizer()
        self.reward_offset = reward_offset
        self.reward_scale = reward_scale

        # Pre-compute affinity weights for the training loss
        # w_i = softmax(|aff_i| / T)  with T = 2.0
        affs = np.array([abs(p["affinity"]) for p in pairs])
        temp = 2.0
        exp_w = np.exp(affs / temp)
        self._weights = exp_w / exp_w.sum()

        # Build contrastive pair index: proteins with >= 2 ligands
        from collections import defaultdict
        self._protein_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(pairs):
            self._protein_to_indices[p["pdb_id"]].append(i)
        self.contrastive_pairs = [
            pdb_id for pdb_id, idxs in self._protein_to_indices.items()
            if len(idxs) >= 2
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def sample_contrastive_pair(self) -> Optional[dict]:
        """Sample a random (strong, weak) ligand pair for the same pocket.

        Returns a dict ready for FlowMatching.compute_contrastive_loss(),
        or None if no contrastive pairs are available.
        """
        import random
        if not self.contrastive_pairs:
            return None

        pdb_id = random.choice(self.contrastive_pairs)
        indices = self._protein_to_indices[pdb_id]

        # Pick two random ligands for this protein
        idx_a, idx_b = random.sample(indices, 2)
        pair_a = self.pairs[idx_a]
        pair_b = self.pairs[idx_b]

        # Ensure A is the stronger binder (more negative affinity)
        if pair_a["affinity"] > pair_b["affinity"]:
            pair_a, pair_b = pair_b, pair_a

        # Featurize pocket (shared — use pocket from pair_a)
        pocket_path = self.base_dir / pair_a["pocket_path"]
        pocket_data = self.pocket_feat.featurize(str(pocket_path))

        # Featurize both ligands
        lig_a = self.ligand_feat.featurize(str(self.base_dir / pair_a["ligand_path"]))
        lig_b = self.ligand_feat.featurize(str(self.base_dir / pair_b["ligand_path"]))

        return {
            "pocket_pos": pocket_data["pos"],
            "pocket_feat": pocket_data["feat"],
            "ligand_pos_a": lig_a["pos"],
            "ligand_feat_a": lig_a["feat"],
            "ligand_types_a": lig_a["atom_types"],
            "affinity_a": pair_a["affinity"],
            "ligand_pos_b": lig_b["pos"],
            "ligand_feat_b": lig_b["feat"],
            "ligand_types_b": lig_b["atom_types"],
            "affinity_b": pair_b["affinity"],
        }

    def __getitem__(self, idx: int) -> dict | None:
        pair = self.pairs[idx]

        pocket_path = self.base_dir / pair["pocket_path"]
        ligand_path = self.base_dir / pair["ligand_path"]

        try:
            # Featurize pocket (handles .pdb and .mol2)
            pocket_data = self.pocket_feat.featurize(str(pocket_path))

            # Featurize ligand (.sdf)
            ligand_data = self.ligand_feat.featurize(str(ligand_path))

            # Validate — ensure we got real tensors
            if pocket_data["pos"] is None or ligand_data["pos"] is None:
                return None
            if pocket_data["pos"].shape[0] == 0 or ligand_data["pos"].shape[0] == 0:
                return None

        except Exception as e:
            logger.debug(f"Skipping {pair['pdb_id']}: {e}")
            return None

        # Reward
        reward = normalise_reward(
            pair["affinity"], self.reward_offset, self.reward_scale
        )

        return {
            "pdb_id": pair["pdb_id"],
            "dataset": pair["dataset"],
            "affinity": torch.tensor(pair["affinity"], dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "weight": torch.tensor(self._weights[idx], dtype=torch.float32),
            # Pocket tensors
            "pocket_pos": pocket_data["pos"],          # (N_P, 3)
            "pocket_feat": pocket_data["feat"],        # (N_P, F_pocket)
            # Ligand tensors
            "ligand_pos": ligand_data["pos"],           # (N_L, 3)
            "ligand_feat": ligand_data["feat"],         # (N_L, F_ligand)
            "ligand_atom_types": ligand_data["atom_types"],  # (N_L,) int
        }

    def get_sample_weight(self, idx: int) -> float:
        """Return the affinity-based importance weight for this sample."""
        return float(self._weights[idx])


def collate_skip_none(batch):
    """Custom collate function that filters out None samples.

    When __getitem__ returns None (corrupt/missing files), this collate
    skips those entries so the DataLoader never crashes.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return batch[0]  # batch_size=1, so just return the single dict

