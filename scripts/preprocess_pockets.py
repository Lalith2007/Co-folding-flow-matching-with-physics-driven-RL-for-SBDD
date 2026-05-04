#!/usr/bin/env python3
"""
preprocess_pockets.py — One-time preprocessing of pocket PDB files.

Computes expensive features that would bottleneck on-the-fly training:
  - Secondary structure (DSSP: helix/sheet/coil → 3-dim one-hot)
  - Solvent Accessible Surface Area (SASA, normalised)
  - Gasteiger partial charges
  - H-bond donor/acceptor flags
  - Hydrophobicity (Kyte-Doolittle scale)

Outputs are saved as .pt files (one per PDB) to a specified output directory,
keyed by the original pocket file path.

Usage:
    python scripts/preprocess_pockets.py \\
        --dataset_json final_dataset.json \\
        --base_dir /path/to/server/data \\
        --output_dir preprocessed_pockets \\
        --num_workers 8

Each .pt file contains:
    {
        "pos":  (N, 3) float32 — atom coordinates,
        "feat": (N, F) float32 — full feature vector (40 + 8 = 48 dims)
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Kyte-Doolittle hydrophobicity scale ──
HYDROPHOBICITY = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}

# Known H-bond donor/acceptor amino acids
HBOND_DONORS = {"ARG", "LYS", "HIS", "ASN", "GLN", "SER", "THR", "TRP", "TYR", "CYS"}
HBOND_ACCEPTORS = {"ASP", "GLU", "ASN", "GLN", "SER", "THR", "HIS", "TYR", "CYS"}


def process_single_pocket(
    pocket_path: str,
    output_path: str,
) -> str:
    """Process a single .pdb pocket file and save enriched features.

    Returns the pocket_path on success, or an error string.
    """
    try:
        from Bio.PDB import PDBParser, DSSP
        from Bio.PDB.SASA import ShrakeRupley

        # ── Element / AA vocabularies (must match featurizer.py) ──
        ELEMENT_LIST = [
            "C", "N", "O", "S", "H", "F", "Cl", "Br", "P", "I",
            "B", "Se", "Si", "Fe", "Zn",
        ]
        ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENT_LIST)}
        NUM_ELEMENTS = len(ELEMENT_LIST) + 1

        AA_LIST = [
            "ALA", "ARG", "ASN", "ASP", "CYS",
            "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL",
        ]
        AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
        NUM_AA = len(AA_LIST) + 1

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pocket", pocket_path)
        model = structure[0]

        # ── DSSP (secondary structure) ──
        # Returns per-residue: (dssp_index, aa, ss, ...)
        dssp_dict = {}
        try:
            dssp = DSSP(model, pocket_path, dssp="mkdssp")
            for key in dssp.keys():
                chain_id, res_id = key
                ss = dssp[key][2]  # secondary structure letter
                # Map to 3 classes: H(helix), E(sheet), C(coil/other)
                if ss in ("H", "G", "I"):
                    ss_class = 0  # helix
                elif ss in ("E", "B"):
                    ss_class = 1  # sheet
                else:
                    ss_class = 2  # coil
                dssp_dict[(chain_id, res_id)] = ss_class
        except Exception as e:
            logger.debug(f"DSSP failed for {pocket_path}: {e} — using coil default")

        # ── SASA ──
        sasa_calc = ShrakeRupley()
        sasa_calc.compute(structure, level="A")  # atom-level SASA

        # ── Collect features ──
        coords = []
        features = []

        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().strip()
                aa_idx = AA_TO_IDX.get(res_name, NUM_AA - 1)
                res_key = (chain.get_id(), residue.get_id())
                ss_class = dssp_dict.get(res_key, 2)  # default coil

                # Hydrophobicity (normalised to [-1, 1])
                hydro = HYDROPHOBICITY.get(res_name, 0.0) / 4.5

                # H-bond flags
                is_hbd = 1.0 if res_name in HBOND_DONORS else 0.0
                is_hba = 1.0 if res_name in HBOND_ACCEPTORS else 0.0

                for atom in residue:
                    element = atom.element.strip()
                    elem_idx = ELEMENT_TO_IDX.get(element, NUM_ELEMENTS - 1)

                    # One-hot element (16-dim)
                    elem_oh = np.zeros(NUM_ELEMENTS, dtype=np.float32)
                    elem_oh[elem_idx] = 1.0

                    # One-hot amino acid (21-dim)
                    aa_oh = np.zeros(NUM_AA, dtype=np.float32)
                    aa_oh[aa_idx] = 1.0

                    # Backbone flag
                    is_backbone = 1.0 if atom.get_name() in (
                        "N", "CA", "C", "O"
                    ) else 0.0

                    # B-factor (normalised)
                    bfactor = atom.get_bfactor() / 100.0

                    # SASA (normalised, max ~250 Å²)
                    sasa_val = getattr(atom, "sasa", 0.0) / 250.0

                    # Secondary structure (3-dim one-hot)
                    ss_oh = np.zeros(3, dtype=np.float32)
                    ss_oh[ss_class] = 1.0

                    # Partial charge placeholder (0.0 — set below if possible)
                    partial_charge = 0.0

                    feat = np.concatenate([
                        elem_oh,              # 16
                        aa_oh,                # 21
                        [is_backbone],        # 1
                        [bfactor],            # 1
                        [0.0],                # 1  dist_to_centroid (filled below)
                        ss_oh,                # 3  secondary structure
                        [sasa_val],           # 1  SASA
                        [partial_charge],     # 1  partial charge
                        [is_hbd],             # 1  H-bond donor
                        [is_hba],             # 1  H-bond acceptor
                        [hydro],              # 1  hydrophobicity
                    ])  # total = 48

                    coords.append(atom.get_vector().get_array())
                    features.append(feat)

        if not coords:
            return f"EMPTY: {pocket_path}"

        pos = torch.tensor(np.array(coords), dtype=torch.float32)
        feat = torch.tensor(np.array(features), dtype=torch.float32)

        # Distance to centroid
        centroid = pos.mean(dim=0, keepdim=True)
        dist_to_centroid = torch.norm(pos - centroid, dim=-1)
        max_dist = dist_to_centroid.max().clamp(min=1e-8)
        feat[:, 40] = dist_to_centroid / max_dist  # column index 40

        # ── Gasteiger charges via RDKit (if possible) ──
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromPDBFile(pocket_path, sanitize=False, removeHs=True)
            if mol is not None:
                AllChem.ComputeGasteigerCharges(mol)
                for i, atom in enumerate(mol.GetAtoms()):
                    if i < feat.size(0):
                        charge = float(atom.GetDoubleProp("_GasteigerCharge"))
                        if not np.isnan(charge):
                            feat[i, 45] = np.clip(charge, -1.0, 1.0)
        except Exception:
            pass  # charges remain 0.0

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({"pos": pos, "feat": feat}, output_path)
        return f"OK: {pocket_path}"

    except Exception as e:
        return f"ERROR: {pocket_path} — {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess pocket PDB files with DSSP, SASA, and charges."
    )
    parser.add_argument("--dataset_json", required=True, help="Path to final_dataset.json")
    parser.add_argument("--base_dir", required=True, help="Server base directory for data files")
    parser.add_argument("--output_dir", default="preprocessed_pockets", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # Load dataset and gather unique pocket paths
    with open(args.dataset_json) as f:
        dataset = json.load(f)

    pocket_paths = set()
    for pdb_id, entry in dataset.items():
        for src in entry["sources"]:
            pocket_path = src["pocket_path"]
            if pocket_path.endswith(".pdb"):
                pocket_paths.add(pocket_path)

    logger.info(f"Found {len(pocket_paths)} unique .pdb pocket files to preprocess")

    # Process in parallel
    results = {"ok": 0, "error": 0, "empty": 0}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for pocket_rel in pocket_paths:
            full_path = os.path.join(args.base_dir, pocket_rel)
            out_path = os.path.join(
                args.output_dir,
                pocket_rel.replace(".pdb", ".pt"),
            )
            futures[executor.submit(process_single_pocket, full_path, out_path)] = pocket_rel

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result.startswith("OK"):
                results["ok"] += 1
            elif result.startswith("EMPTY"):
                results["empty"] += 1
            else:
                results["error"] += 1
                if results["error"] <= 10:
                    logger.warning(result)

            if (i + 1) % 500 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(pocket_paths)} — "
                    f"ok={results['ok']} error={results['error']} empty={results['empty']}"
                )

    logger.info(
        f"Done. ok={results['ok']}, error={results['error']}, empty={results['empty']}"
    )


if __name__ == "__main__":
    main()
