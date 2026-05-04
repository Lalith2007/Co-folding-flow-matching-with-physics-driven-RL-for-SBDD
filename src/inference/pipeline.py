"""
pipeline.py — Master inference pipeline.

Orchestrates the full end-to-end inference:
  1. Run P2Rank on user-uploaded PDB → detect pockets
  2. Featurize the top pocket
  3. Load trained model and generate a molecule (3D coords + atom types)
  4. Reconstruct bonds → produce valid SMILES
  5. Validate and return results

Usage (CLI):
    python -m src.inference.pipeline --pdb protein.pdb --checkpoint pretrain_final.pt

Usage (Python):
    from src.inference.pipeline import InferencePipeline
    pipe = InferencePipeline(checkpoint_path="checkpoints/pretrain_final.pt")
    result = pipe.run("protein.pdb")
    print(result["smiles"])
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
        from ..model.egnn import SBDDEGNN
        from ..model.flow_matching import FlowMatching

        # Build model with default architecture
        pocket_encoder = PocketEncoder(
            in_dim=40, hidden_dim=128, num_layers=4, knn_k=16,
        )
        egnn = SBDDEGNN(
            ligand_in_dim=20, hidden_dim=128, num_layers=9,
            num_heads=16, num_atom_types=10, knn_k=16,
        )
        model = FlowMatching(
            pocket_encoder=pocket_encoder,
            egnn=egnn,
            num_steps=self.num_steps,
        )

        # Load weights
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model = model.to(self.device)

        return model

    def run(
        self,
        pdb_path: str,
        pocket_index: Optional[int] = None,
    ) -> Dict:
        """Execute the full inference pipeline.

        Parameters
        ----------
        pdb_path     : path to the user-uploaded PDB file
        pocket_index : override which pocket to use (1-indexed). Default: top-ranked.

        Returns
        -------
        dict with:
            smiles         : str — best generated SMILES
            all_smiles     : list of all valid SMILES generated
            coords_3d      : (N, 3) array of atom positions
            atom_types     : (N,) array of element symbols
            pocket_info    : pocket metadata from P2Rank
            properties     : molecular property dict (MW, QED, logP, etc.)
            timings        : dict of step durations
            success        : bool
            error          : str or None
        """
        timings = {}
        pocket_idx = (pocket_index or self.top_pocket) - 1  # 0-indexed

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
            return self._fail("P2Rank found no pockets in the uploaded PDB.")

        if pocket_idx >= len(pockets):
            pocket_idx = 0
            logger.warning(f"Requested pocket index out of range, using top pocket")

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
                    "pK_pred": gen["pK_pred"].item(),
                })
        timings["generation"] = round(time.time() - t2, 2)

        logger.info(f"Generated {len(generated)} molecules")

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
                f"Generated {self.num_samples} molecules but none produced valid SMILES. "
                "Model may need more training."
            )

        # Select best by QED (or pK_pred if QED unavailable)
        best = max(
            all_results,
            key=lambda r: r["properties"].get("qed", r["pK_pred"]),
        )

        logger.info(
            f"Best SMILES: {best['smiles']} | "
            f"QED={best['properties'].get('qed', 'N/A')} | "
            f"pK={best['pK_pred']:.3f}"
        )

        return {
            "success": True,
            "error": None,
            "smiles": best["smiles"],
            "all_smiles": [r["smiles"] for r in all_results],
            "coords_3d": best["coords"].tolist(),
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
            "coords_3d": [],
            "atom_types": [],
            "pocket_info": {},
            "properties": {},
            "num_valid": 0,
            "num_generated": 0,
            "timings": {},
        }


# ── CLI entry point ──
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SBDD Inference: PDB → Pocket → SMILES"
    )
    parser.add_argument("--pdb", required=True, help="Path to input PDB file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--pocket", type=int, default=1, help="Which pocket rank to use")
    parser.add_argument("--p2rank_home", default=None, help="Path to P2Rank installation")
    parser.add_argument("--output_json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint,
        device=device,
        p2rank_home=args.p2rank_home,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        top_pocket=args.pocket,
    )

    result = pipeline.run(args.pdb)

    if result["success"]:
        print(f"\n{'='*60}")
        print(f"  GENERATED SMILES: {result['smiles']}")
        print(f"  Valid molecules: {result['num_valid']}/{result['num_generated']}")
        print(f"  QED: {result['properties'].get('qed', 'N/A')}")
        print(f"  MW:  {result['properties'].get('molecular_weight', 'N/A')}")
        print(f"  Pocket score: {result['pocket_info'].get('score', 'N/A')}")
        print(f"  Timings: {result['timings']}")
        print(f"{'='*60}\n")
    else:
        print(f"\n  ERROR: {result['error']}\n")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
