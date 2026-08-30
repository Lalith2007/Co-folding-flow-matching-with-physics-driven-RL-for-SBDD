"""
pipeline.py — Master inference pipeline.

Orchestrates the full end-to-end inference:
  1. Detect pockets from raw PDB (P2Rank with geometric fallback)
  2. Featurize the top pocket
  3. Load trained model and generate molecules (3D coords + atom types)
  4. Reconstruct bonds → produce valid SMILES with individual properties
  5. Validate and return results
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference: PDB → Pocket → Model → SMILES.

    Parameters
    ----------
    checkpoint_path : path to trained model checkpoint (.pt)
    device          : 'cuda' or 'cpu'
    p2rank_home     : path to p2rank installation (auto-download if None)
    num_samples     : molecules to generate per pocket (best SMILES returned)
    num_steps       : Euler integration steps (default 50)
    top_pocket      : which pocket rank to use (default 1 = best)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        p2rank_home: Optional[str] = None,
        num_samples: int = 10,
        num_steps: int = 50,
        top_pocket: int = 1,
    ):
        self.device = device
        self.p2rank_home = p2rank_home
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.top_pocket = top_pocket

        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        logger.info(f"Model loaded on {device}")

    def _load_model(self, checkpoint_path: str):
        """Load the FlowMatching model from a checkpoint."""
        from ..model.pocket_encoder import PocketEncoder
        from ..model.egnn import SE3EGNN
        from ..model.flow_matching import FlowMatching

        pocket_encoder = PocketEncoder(
            in_dim=40, hidden_dim=128, num_layers=4, knn_k=16,
        )
        egnn = SE3EGNN(
            ligand_in_dim=4, hidden_dim=128, num_layers=9,
            num_heads=16, num_atom_types=6, knn_k=16,
        )
        model = FlowMatching(
            pocket_encoder=pocket_encoder,
            egnn=egnn,
            num_steps=self.num_steps,
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)

        return model

    def run(
        self,
        pdb_path: str,
        pocket_index: Optional[int] = None,
    ) -> Dict:
        """Execute the full inference pipeline."""
        timings = {}
        pocket_idx = (pocket_index or self.top_pocket) - 1

        # ── Step 1: Pocket Detection ──
        t0 = time.time()
        from .p2rank_wrapper import run_p2rank, extract_pocket_pdb

        p2rank_result = run_p2rank(
            pdb_path,
            p2rank_home=self.p2rank_home,
        )
        pockets = p2rank_result["pockets"]
        timings["p2rank"] = round(time.time() - t0, 2)

        if not pockets:
            return self._fail("No binding cavities found in the uploaded PDB.")

        if pocket_idx >= len(pockets):
            pocket_idx = 0

        pocket = pockets[pocket_idx]
        logger.info(
            f"Using pocket {pocket_idx + 1}/{len(pockets)}: "
            f"score={pocket['score']:.2f}, {len(pocket['residues'])} residues"
        )

        # Extract pocket PDB
        pocket_pdb_path = extract_pocket_pdb(pdb_path, pocket)

        # ── Step 2: Featurize Pocket ──
        t1 = time.time()
        from ..data.featurizer import PocketFeaturizer

        featurizer = PocketFeaturizer()
        pocket_data = featurizer.featurize(pocket_pdb_path)
        pocket_pos = pocket_data["pos"].to(self.device)
        pocket_feat = pocket_data["feat"].to(self.device)
        timings["featurize"] = round(time.time() - t1, 2)

        logger.info(f"Pocket featurized: {pocket_pos.shape[0]} atoms, feat_dim={pocket_feat.shape[1]}")

        # ── Step 3: Generate Molecules ──
        t2 = time.time()
        generated = []
        with torch.no_grad():
            for i in range(self.num_samples):
                gen = self.model.sample(pocket_pos, pocket_feat)
                generated.append({
                    "coords": gen["pos"].cpu().numpy(),
                    "atom_type_indices": gen["atom_types"].cpu().numpy(),
                    "pK_pred": float(gen["pK_pred"].item()) if hasattr(gen["pK_pred"], "item") else float(gen["pK_pred"]),
                })
        timings["generation"] = round(time.time() - t2, 2)

        logger.info(f"Generated {len(generated)} molecules in {timings['generation']}s")

        # ── Step 4: Bond Inference → SMILES ──
        t3 = time.time()
        from .bond_inference import coords_to_smiles, validate_smiles, IDX_TO_ELEMENT

        all_results = []
        for gen in generated:
            result = coords_to_smiles(
                gen["coords"],
                gen["atom_type_indices"],
            )
            if result["success"]:
                props = validate_smiles(result["smiles"])
                all_results.append({
                    "smiles": result["smiles"],
                    "coords": gen["coords"],
                    "atom_types": [
                        IDX_TO_ELEMENT[int(i)] if i < len(IDX_TO_ELEMENT) else "C"
                        for i in gen["atom_type_indices"]
                    ],
                    "pK_pred": gen["pK_pred"],
                    "properties": props,
                    "num_bonds": result["num_bonds"],
                })

        timings["bond_inference"] = round(time.time() - t3, 2)

        if not all_results:
            return self._fail(
                f"Generated {self.num_samples} molecules but none produced valid SMILES."
            )

        # Select best by QED
        best = max(
            all_results,
            key=lambda r: r["properties"].get("qed", 0.0),
        )

        all_candidates = [
            {
                "smiles": r["smiles"],
                "properties": r["properties"],
                "coords": r["coords"].tolist() if hasattr(r["coords"], "tolist") else r["coords"],
                "atom_types": r["atom_types"],
                "pK_pred": r["pK_pred"],
            }
            for r in all_results
        ]

        logger.info(
            f"Best SMILES: {best['smiles']} | "
            f"QED={best['properties'].get('qed', 'N/A')} | "
            f"SA={best['properties'].get('sa_score', 'N/A')}"
        )

        return {
            "success": True,
            "error": None,
            "smiles": best["smiles"],
            "all_smiles": [r["smiles"] for r in all_results],
            "all_candidates": all_candidates,
            "coords_3d": best["coords"].tolist() if hasattr(best["coords"], "tolist") else best["coords"],
            "atom_types": best["atom_types"],
            "pocket_info": {
                "rank": pocket["rank"],
                "score": pocket["score"],
                "center": pocket["center"],
                "num_residues": len(pocket["residues"]),
                "total_pockets_found": len(pockets),
            },
            "properties": best["properties"],
            "num_valid": len(all_results),
            "num_generated": self.num_samples,
            "timings": timings,
        }

    @staticmethod
    def _fail(error: str) -> Dict:
        return {
            "success": False,
            "error": error,
            "smiles": "",
            "all_smiles": [],
            "all_candidates": [],
            "coords_3d": [],
            "atom_types": [],
            "pocket_info": {},
            "properties": {},
            "num_valid": 0,
            "num_generated": 0,
            "timings": {},
        }
