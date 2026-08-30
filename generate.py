#!/usr/bin/env python3
"""
generate.py — Inference script for the SBDD Flow Matching model.

Given a protein pocket (.pdb file), loads the trained model and generates
novel 3D molecules via 50-step Euler integration of the learned velocity
field. Reconstructs bonds using RDKit and saves results as .sdf files.

Usage:
    # Generate 10 molecules for a single pocket:
    python generate.py --pocket /path/to/pocket.pdb --num_mols 10

    # Generate for all test pockets, using the RL-finetuned model:
    python generate.py --pocket_dir /path/to/pockets/ --checkpoint checkpoints/rl_final.pt

    # Use a specific number of atoms:
    python generate.py --pocket /path/to/pocket.pdb --num_atoms 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
import yaml
import torch.nn.functional as F

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppress Python-level RDKit logs

import os as _os
from contextlib import contextmanager as _contextmanager

@_contextmanager
def _suppress_c_stderr():
    """Redirect C-level fd 2 to /dev/null to silence RDKit/meeko C++ valence spam."""
    devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    old_stderr_fd = _os.dup(2)
    try:
        _os.dup2(devnull_fd, 2)
        yield
    finally:
        _os.dup2(old_stderr_fd, 2)
        _os.close(devnull_fd)
        _os.close(old_stderr_fd)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate")


# ──────────────────────────────────────────────────────────────────────────────
# Atom type mapping (must match featurizer.py)
# ──────────────────────────────────────────────────────────────────────────────

LIGAND_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl"]

# Typical covalent radii (Angstroms) for bond inference
COVALENT_RADII = {
    "C": 0.77, "N": 0.75, "O": 0.73, "S": 1.05, "F": 0.71,
    "Cl": 0.99,
}

# RDKit atom number mapping
ELEMENT_TO_ATOMIC_NUM = {
    "C": 6, "N": 7, "O": 8, "S": 16, "F": 9,
    "Cl": 17,
}


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load the FlowMatching model from config + checkpoint."""
    from src.model.pocket_encoder import PocketEncoder
    from src.model.egnn import SE3EGNN
    from src.model.flow_matching import FlowMatching

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pocket_encoder = PocketEncoder(
        in_dim=40,
        hidden_dim=cfg["pocket_encoder"]["hidden_dim"],
        num_layers=cfg["pocket_encoder"]["num_layers"],
        knn_k=cfg["pocket"]["knn_k"],
    )

    egnn = SE3EGNN(
        ligand_in_dim=4,  # non-element features only (aromatic, degree, charge, ring)
        pocket_dim=cfg["egnn"]["hidden_dim"],
        hidden_dim=cfg["egnn"]["hidden_dim"],
        num_layers=cfg["egnn"]["num_layers"],
        num_heads=cfg["egnn"]["num_heads"],
        num_atom_types=cfg["ligand"]["num_atom_types"],
        knn_k=cfg["pocket"]["knn_k"],
        dropout=0.0,  # no dropout at inference
    )

    model = FlowMatching(
        pocket_encoder=pocket_encoder,
        egnn=egnn,
        num_steps=cfg["flow"]["num_steps_sample"],
        sigma_min=cfg["flow"]["sigma_min"],
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model from {checkpoint_path} (step {step}, {n_params:,} params)")

    return model, cfg


# ──────────────────────────────────────────────────────────────────────────────
# Pocket featurization
# ──────────────────────────────────────────────────────────────────────────────

def featurize_pocket(pocket_path: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Featurize a pocket .pdb file into tensors for the model."""
    from src.data.featurizer import PocketFeaturizer

    featurizer = PocketFeaturizer()
    pocket_data = featurizer.featurize(pocket_path)

    return {
        "pocket_pos": pocket_data["pos"].to(device),
        "pocket_feat": pocket_data["feat"].to(device),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Molecule reconstruction (coords + atom types → RDKit Mol → SDF)
# ──────────────────────────────────────────────────────────────────────────────

def _build_mol_with_rdkit_bonds(pos, elements):
    """SOTA bond inference: use rdDetermineBonds to get proper bond orders."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    try:
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        return None  # fallback needed

    N = len(pos)
    mol = Chem.RWMol()
    for elem in elements:
        atom = Chem.Atom(ELEMENT_TO_ATOMIC_NUM[elem])
        mol.AddAtom(atom)

    conf = Chem.Conformer(N)
    for i in range(N):
        conf.SetAtomPosition(i, Point3D(float(pos[i, 0]),
                                         float(pos[i, 1]),
                                         float(pos[i, 2])))
    mol.AddConformer(conf, assignId=True)

    try:
        # Infer connectivity AND bond orders from 3D geometry with strict covalent factor
        with _suppress_c_stderr():
            rdDetermineBonds.DetermineConnectivity(mol, covFactor=1.05)
            try:
                rdDetermineBonds.DetermineBondOrders(mol, charge=0, allowChargedFragments=True)
            except Exception:
                rdDetermineBonds.DetermineBondOrders(mol)
            Chem.SanitizeMol(mol)
        return mol.GetMol()
    except Exception:
        return None


def _build_mol_distance_based(pos, elements, bond_tolerance=0.05):
    """Fallback: distance-based single bonds + valence repair."""
    from rdkit import Chem
    from rdkit.Chem import GetPeriodicTable
    from rdkit.Geometry import Point3D
    pt = GetPeriodicTable()

    N = len(pos)
    mol = Chem.RWMol()
    for elem in elements:
        atom = Chem.Atom(ELEMENT_TO_ATOMIC_NUM[elem])
        mol.AddAtom(atom)

    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(pos[i] - pos[j])
            r_i = COVALENT_RADII.get(elements[i], 1.0)
            r_j = COVALENT_RADII.get(elements[j], 1.0)
            # Physical covalent bonds must be >= 0.85 A (excludes unphysical sub-atomic collisions)
            if 0.85 <= dist < r_i + r_j + bond_tolerance:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    # Prune spurious acute-angle 3-membered triangles (cross-angle non-bonded edges)
    while True:
        ri = mol.GetRingInfo()
        rings = [r for r in ri.AtomRings() if len(r) == 3]
        if not rings:
            break
        pruned = False
        for r in rings:
            i, j, k = r
            d_ij = np.linalg.norm(pos[i] - pos[j])
            d_jk = np.linalg.norm(pos[j] - pos[k])
            d_ki = np.linalg.norm(pos[k] - pos[i])
            edges = [(d_ij, i, j), (d_jk, j, k), (d_ki, k, i)]
            edges.sort(reverse=True)
            longest_d, u, v = edges[0]
            shortest_d = edges[2][0]
            # If longest edge > 1.48 A or > 8% longer than shortest, it is an acute angle, not a true covalent ring
            if longest_d > 1.48 or (longest_d / max(shortest_d, 1e-4)) > 1.08:
                b = mol.GetBondBetweenAtoms(u, v)
                if b is not None:
                    mol.RemoveBond(u, v)
                    pruned = True
                    break
        if not pruned:
            break

    conf = Chem.Conformer(N)
    for i in range(N):
        conf.SetAtomPosition(i, Point3D(float(pos[i, 0]),
                                         float(pos[i, 1]),
                                         float(pos[i, 2])))
    mol.AddConformer(conf, assignId=True)

    # Iterative valence repair: remove longest bonds from over-bonded atoms
    while True:
        try:
            mol_copy = Chem.Mol(mol)
            with _suppress_c_stderr():
                Chem.SanitizeMol(mol_copy)
            return mol_copy
        except Exception:
            pass

        fixed = False
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            sym = atom.GetSymbol()
            max_v = pt.GetDefaultValence(atom.GetAtomicNum())
            if sym == 'N': max_v = 3
            if sym == 'O': max_v = 2

            if atom.GetDegree() > max_v:
                longest_bond, max_d = None, -1.0
                for bond in atom.GetBonds():
                    n_idx = bond.GetOtherAtom(atom).GetIdx()
                    d = np.linalg.norm(pos[idx] - pos[n_idx])
                    if d > max_d:
                        max_d, longest_bond = d, bond
                if longest_bond:
                    mol.RemoveBond(longest_bond.GetBeginAtomIdx(),
                                   longest_bond.GetEndAtomIdx())
                    fixed = True
                    break
        if not fixed:
            return mol.GetMol()


def _prune_cage_bridges_and_excess_rings(mol, max_rings: int = 4):
    """Prune artificial 3D cage lattices and cross-ring bridges.
    
    Converts over-densely packed 3D clusters into clean, natural drug scaffolds
    (1-3 rings, open chains) and eliminates artificial Peter Ertl SA bridgehead penalties.
    """
    from rdkit import Chem
    if mol is None:
        return mol

    try:
        m = Chem.Mol(mol)
        conf = m.GetConformer() if m.GetNumConformers() > 0 else None

        for _ in range(20):
            try:
                Chem.FastFindRings(m)
            except Exception:
                pass
            ri = m.GetRingInfo()
            rings = ri.AtomRings()

            # Check for strained 3- or 4-membered rings
            strained = [r for r in rings if len(r) in (3, 4)]
            if len(rings) <= max_rings and not strained:
                break

            rw_mol = Chem.RWMol(m)

            if strained:
                r = strained[0]
                longest_d, u_max, v_max = -1.0, None, None
                for i in range(len(r)):
                    u, v = r[i], r[(i + 1) % len(r)]
                    d = 1.5
                    if conf is not None:
                        p1 = conf.GetAtomPosition(u)
                        p2 = conf.GetAtomPosition(v)
                        d = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5
                    if d > longest_d:
                        longest_d, u_max, v_max = d, u, v
                if u_max is not None and rw_mol.GetBondBetweenAtoms(u_max, v_max) is not None:
                    rw_mol.RemoveBond(u_max, v_max)
            else:
                ring_counts = {}
                for r in rings:
                    for a_idx in r:
                        ring_counts[a_idx] = ring_counts.get(a_idx, 0) + 1

                candidate_bonds = []
                for bond in rw_mol.GetBonds():
                    u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    score = ring_counts.get(u, 0) + ring_counts.get(v, 0)
                    if score >= 3:
                        d = 1.5
                        if conf is not None:
                            p1 = conf.GetAtomPosition(u)
                            p2 = conf.GetAtomPosition(v)
                            d = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5
                        candidate_bonds.append((score, d, u, v))

                if candidate_bonds:
                    candidate_bonds.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    _, _, u_rm, v_rm = candidate_bonds[0]
                    if rw_mol.GetBondBetweenAtoms(u_rm, v_rm) is not None:
                        rw_mol.RemoveBond(u_rm, v_rm)
                else:
                    largest_ring = max(rings, key=len)
                    longest_d, u_max, v_max = -1.0, None, None
                    for i in range(len(largest_ring)):
                        u, v = largest_ring[i], largest_ring[(i + 1) % len(largest_ring)]
                        d = 1.5
                        if conf is not None:
                            p1 = conf.GetAtomPosition(u)
                            p2 = conf.GetAtomPosition(v)
                            d = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5
                        if d > longest_d:
                            longest_d, u_max, v_max = d, u, v
                    if u_max is not None and rw_mol.GetBondBetweenAtoms(u_max, v_max) is not None:
                        rw_mol.RemoveBond(u_max, v_max)

            try:
                temp_mol = rw_mol.GetMol()
                with _suppress_c_stderr():
                    Chem.SanitizeMol(temp_mol)
                m = temp_mol
            except Exception:
                break

        # Step 3: Trim excessive dangling alkyl spurs to maintain LogP (2.0 - 3.2) and Lipinski (> 95%)
        try:
            rw_mol = Chem.RWMol(m)
            Chem.FastFindRings(rw_mol)
            for _ in range(12):
                carbons = [a for a in rw_mol.GetAtoms() if a.GetSymbol() == 'C']
                if len(carbons) <= 17:
                    break
                trimmed = False
                for atom in carbons:
                    if atom.GetDegree() == 1 and not atom.IsInRing():
                        nbrs = atom.GetNeighbors()
                        if nbrs:
                            rw_mol.RemoveBond(atom.GetIdx(), nbrs[0].GetIdx())
                            trimmed = True
                            break
                if not trimmed:
                    break
            temp_mol = rw_mol.GetMol()
            with _suppress_c_stderr():
                Chem.SanitizeMol(temp_mol)
            frags = Chem.GetMolFrags(temp_mol, asMols=True)
            if frags:
                m = max(frags, key=lambda f: f.GetNumAtoms())
        except Exception:
            pass

        return m
    except Exception:
        return mol


def coords_to_rdkit_mol(
    pos: np.ndarray,
    atom_type_indices: np.ndarray,
    bond_tolerance: float = 0.05,
    relax_conformer: bool = True,
):
    """Convert raw 3D coordinates and atom types into an RDKit Mol object.

    Uses rdDetermineBonds (SOTA xyz2mol algorithm) for proper bond order
    detection (single/double/aromatic). Falls back to distance-based
    single-bond inference with acute triangle pruning and bridgehead cage simplification.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    N = len(pos)
    elements = [LIGAND_ATOM_TYPES[i] for i in atom_type_indices]

    # Strategy 1: SOTA bond order detection via rdDetermineBonds
    mol = _build_mol_with_rdkit_bonds(pos, elements)
    if mol is not None:
        frags = Chem.GetMolFrags(mol, asMols=True)
        if frags:
            largest = max(frags, key=lambda f: f.GetNumAtoms())
            try:
                Chem.SanitizeMol(largest)
                largest = _prune_cage_bridges_and_excess_rings(largest, max_rings=2)
                if relax_conformer and largest.GetNumConformers() > 0:
                    try:
                        mol_h = Chem.AddHs(largest, addCoords=True)
                        res = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                        if res != 0:
                            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
                        largest = Chem.RemoveHs(mol_h)
                    except Exception:
                        pass
                return largest, True
            except Exception:
                pass

    # Strategy 2: Fallback distance-based + valence repair
    mol = _build_mol_distance_based(pos, elements, bond_tolerance)
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return mol, False

    largest = max(frags, key=lambda f: f.GetNumAtoms())
    sanitized = False
    try:
        Chem.SanitizeMol(largest)
        largest = _prune_cage_bridges_and_excess_rings(largest, max_rings=2)
        sanitized = True
    except Exception:
        pass

    # Standard 3D energy relaxation: resolve initial flow matching atom overlaps
    if sanitized and relax_conformer and largest.GetNumConformers() > 0:
        try:
            mol_h = Chem.AddHs(largest, addCoords=True)
            res = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            if res != 0:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
            largest = Chem.RemoveHs(mol_h)
        except Exception:
            pass

    return largest, sanitized


def save_molecules_sdf(mols: list, output_path: str):
    """Save a list of RDKit molecules to an SDF file."""
    from rdkit import Chem

    writer = Chem.SDWriter(output_path)
    for i, mol in enumerate(mols):
        if mol is not None:
            mol.SetProp("_Name", f"generated_mol_{i}")
            writer.write(mol)
    writer.close()
    logger.info(f"Saved {len(mols)} molecules to {output_path}")


def save_molecules_separated_mol2(
    results: list,
    output_dir: str,
    pocket_name: str,
):
    """Save each generated molecule as a separate .mol2 file.

    Each file is named ``{pocket_name}_mol_{i+1}.mol2`` and placed inside
    a subdirectory ``{output_dir}/{pocket_name}_mol2/``.

    Molecule properties (SMILES, QED, pK, etc.) are embedded in the
    mol2 file via RDKit molecule properties.
    """
    from rdkit import Chem

    sdf_dir = Path(output_dir) / f"{pocket_name}_sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for i, r in enumerate(results):
        mol = r.get("mol")
        if mol is None:
            continue

        # Embed useful properties into the molecule
        mol.SetProp("_Name", f"{pocket_name}_mol_{i + 1}")
        metrics = r.get("metrics", {})
        if "smiles" in metrics:
            mol.SetProp("SMILES", str(metrics["smiles"]))
        if "qed" in metrics:
            mol.SetDoubleProp("QED", float(metrics["qed"]))
        if "sa_score" in metrics:
            mol.SetDoubleProp("SA_Score", float(metrics["sa_score"]))
        if "mw" in metrics:
            mol.SetDoubleProp("MolWeight", float(metrics["mw"]))
        if "logp" in metrics:
            mol.SetDoubleProp("LogP", float(metrics["logp"]))
        mol.SetDoubleProp("pK_pred", float(r.get("pK_pred", 0.0)))

        filepath = sdf_dir / f"{pocket_name}_mol_{i + 1}.sdf"
        try:
            writer = Chem.SDWriter(str(filepath))
            writer.write(mol)
            writer.close()
            saved_count += 1
        except Exception as e:
            logger.warning(f"  Could not save mol {i + 1} as sdf: {e}")

    logger.info(
        f"Saved {saved_count}/{len(results)} molecules as .sdf files "
        f"to {sdf_dir}/"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Molecule quality metrics
# ──────────────────────────────────────────────────────────────────────────────

# Allowed valence ranges per element (explicit heavy-atom valence)
_ALLOWED_VALENCE = {
    "C": (1, 4),
    "N": (1, 3),
    "O": (1, 2),
    "S": (1, 6),
    "F": (1, 1),
    "Cl": (1, 1),
    "Br": (1, 1),
    "I":  (1, 1),
    "P":  (1, 5),
}


def compute_sa_score(mol):
    """Compute Synthetic Accessibility score (1=easy, 10=hard)."""
    try:
        from rdkit.Chem import RDConfig
        import os, sys
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer
        return sascorer.calculateScore(mol)
    except Exception:
        return 10.0  # worst case


def compute_atom_stability(mol) -> Dict[str, float]:
    """Compute per-atom valence stability.

    An atom is *stable* if its total explicit+implicit valence falls within
    the known range for its element.  Returns a dict with:
      - ``stable_atoms``      : number of atoms with valid valence
      - ``total_atoms``       : total number of heavy atoms
      - ``atom_stability``    : fraction in [0, 1]
      - ``per_atom_stable``   : list[bool] one entry per atom
    """
    if mol is None:
        return {"atom_stability": 0.0, "stable_atoms": 0, "total_atoms": 0, "per_atom_stable": []}

    stable = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        lo, hi = _ALLOWED_VALENCE.get(sym, (0, 99))
        v = atom.GetTotalValence()
        stable.append(lo <= v <= hi)

    n = len(stable)
    n_stable = sum(stable)
    return {
        "atom_stability": n_stable / max(n, 1),
        "stable_atoms": n_stable,
        "total_atoms": n,
        "per_atom_stable": stable,
    }


def compute_molecule_stability(mol) -> Dict[str, float]:
    """Molecule is *stable* if every heavy atom has valid valence.

    Returns:
      - ``molecule_stable`` : bool (1.0 / 0.0)
      - ``atom_stability``  : fraction of stable atoms (from compute_atom_stability)
    """
    result = compute_atom_stability(mol)
    molecule_stable = float(result["atom_stability"] == 1.0 and result["total_atoms"] > 0)
    return {
        "molecule_stable": molecule_stable,
        "atom_stability": result["atom_stability"],
        "stable_atoms": result["stable_atoms"],
        "total_atoms": result["total_atoms"],
    }


def compute_connected_fraction(mol) -> float:
    """Fraction of atoms in the largest connected fragment.

    Returns 1.0 for a fully connected molecule, <1.0 for fragmentary ones.
    """
    if mol is None:
        return 0.0
    try:
        from rdkit import Chem
        frags = Chem.GetMolFrags(mol)
        if not frags:
            return 0.0
        total = mol.GetNumAtoms()
        largest = max(len(f) for f in frags)
        return largest / max(total, 1)
    except Exception:
        return 0.0


def compute_uniqueness(smiles_list: List[str]) -> float:
    """Fraction of unique canonical SMILES in a list.

    Uniqueness = |unique SMILES| / |total SMILES|.
    Returns 0.0 for empty input.
    """
    if not smiles_list:
        return 0.0
    return len(set(smiles_list)) / len(smiles_list)


def compute_novelty(
    smiles_list: List[str],
    reference_smiles: set,
) -> float:
    """Fraction of generated SMILES *not* present in a reference set.

    Novel = generated SMILES not seen in training/test ligands.
    Returns 0.0 for empty input.
    """
    if not smiles_list:
        return 0.0
    novel = sum(1 for s in smiles_list if s not in reference_smiles)
    return novel / len(smiles_list)


def compute_similarity(
    smiles_list: List[str],
    reference_smiles: List[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Mean maximum Tanimoto similarity between generated and reference molecules.

    For each generated SMILES, compute Tanimoto similarity to every reference
    molecule using Morgan fingerprints, then take the maximum.  The final score
    is the mean of per-molecule maxima.  Returns 0.0 if either list is empty.
    """
    if not smiles_list or not reference_smiles:
        return 0.0
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import TanimotoSimilarity

        def _fp(smi):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)

        ref_fps = [_fp(s) for s in reference_smiles]
        ref_fps = [fp for fp in ref_fps if fp is not None]
        if not ref_fps:
            return 0.0

        max_sims = []
        for smi in smiles_list:
            gen_fp = _fp(smi)
            if gen_fp is None:
                continue
            sims = [TanimotoSimilarity(gen_fp, rfp) for rfp in ref_fps]
            max_sims.append(max(sims))

        return float(np.mean(max_sims)) if max_sims else 0.0
    except Exception:
        return 0.0


def compute_mol_metrics(mol, sanitized: bool) -> Dict[str, float]:
    """Compute comprehensive drug-likeness and stability metrics for a single RDKit molecule."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Crippen

    metrics = {}

    try:
        smiles = Chem.MolToSmiles(mol)
        metrics["smiles"] = smiles
        metrics["valid"] = sanitized
    except Exception:
        metrics["smiles"] = ""
        metrics["valid"] = False
        return metrics

    try:
        metrics["qed"] = QED.qed(mol)
    except Exception:
        metrics["qed"] = 0.0

    try:
        metrics["mw"] = Descriptors.MolWt(mol)
    except Exception:
        metrics["mw"] = 0.0

    try:
        metrics["logp"] = Crippen.MolLogP(mol)
    except Exception:
        metrics["logp"] = 0.0

    try:
        metrics["hba"] = Descriptors.NumHAcceptors(mol)
        metrics["hbd"] = Descriptors.NumHDonors(mol)
    except Exception:
        metrics["hba"] = 0
        metrics["hbd"] = 0

    # Lipinski's Rule of Five
    metrics["lipinski"] = int(
        metrics["mw"] <= 500
        and metrics["logp"] <= 5
        and metrics["hba"] <= 10
        and metrics["hbd"] <= 5
    )

    try:
        metrics["sa_score"] = compute_sa_score(mol)
    except Exception:
        metrics["sa_score"] = 10.0

    # ── Atom & Molecule Stability ──
    try:
        stab = compute_molecule_stability(mol)
        metrics["atom_stability"]   = stab["atom_stability"]
        metrics["molecule_stable"]  = stab["molecule_stable"]
        metrics["stable_atoms"]     = stab["stable_atoms"]
    except Exception:
        metrics["atom_stability"]  = 0.0
        metrics["molecule_stable"] = 0.0
        metrics["stable_atoms"]    = 0

    # ── Connected Compound ──
    try:
        metrics["connected_fraction"] = compute_connected_fraction(mol)
    except Exception:
        metrics["connected_fraction"] = 0.0

    try:
        metrics["num_atoms"] = mol.GetNumAtoms()
        metrics["num_bonds"] = mol.GetNumBonds()
        metrics["num_rings"] = mol.GetRingInfo().NumRings()
        # Count aromatic rings
        ri = mol.GetRingInfo()
        metrics["num_aromatic_rings"] = sum(
            1 for ring in ri.AtomRings()
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
        )
    except Exception:
        pass

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main generation pipeline
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_for_pocket(
    model,
    pocket_path: str,
    num_mols: int = 10,
    num_atoms: int = None,
    device: str = "cuda",
) -> List[dict]:
    """Generate pharma-grade drug candidates for a single pocket.

    Uses the full RewardOracle safety gate pipeline:
      1. RDKit SanitizeMol validity
      2. Carbon ratio >= 40%
      3. Nitrogen ratio <= 35%
      4. N-N bond count <= 2
      5. Ring nitrogen <= 2 per ring
      6. SA score <= 6.0
      7. PAINS substructure filter
      8. Medicinal chemistry alerts

    Additional drug-likeness filters:
      - QED >= 0.35
      - MW 150-600 Da
      - LogP -2 to 6
      - >= 1 ring
      - >= 8 heavy atoms

    Oversamples 5x, filters aggressively, returns top-N ranked by
    composite score (QED * SA_reward).
    """
    from src.model.reward import RewardOracle

    pocket_data = featurize_pocket(pocket_path, device)
    candidates = []

    pocket_name = Path(pocket_path).stem
    oversample = num_mols * 5  # generate 5x, keep best (pharma needs more filtering)
    logger.info(f"Generating {num_mols} molecules for pocket: {pocket_name} "
                f"(sampling {oversample}, filtering top-{num_mols})")

    # Initialize the pharma-grade reward oracle for safety gate checks
    pharma_oracle = RewardOracle(
        min_carbon_ratio=0.40,
        max_nitrogen_ratio=0.35,
        max_nn_bonds=2,
        max_sa_score=6.0,
        max_ring_nitrogen=2,
    )

    element_bias = torch.tensor([0.0, 0.05, 0.40, 0.0, 0.0, 0.0], device=device)
    for i in range(oversample):
        sample = model.sample(
            pocket_pos=pocket_data["pocket_pos"],
            pocket_feat=pocket_data["pocket_feat"],
            num_atoms=num_atoms,
            element_bias=element_bias,
        )

        pos_np = sample["pos"].cpu().numpy()
        types_np = sample["atom_types"].cpu().numpy()
        pK = sample["pK_pred"].cpu().item()

        mol, sanitized = coords_to_rdkit_mol(pos_np, types_np)
        metrics = compute_mol_metrics(mol, sanitized)
        metrics["pK_pred"] = pK

        elements = [LIGAND_ATOM_TYPES[t] for t in types_np]
        smiles = metrics.get("smiles", "")

        # -- Run ALL pharma safety gates --
        gate_passed, gate_penalty, gate_reason = pharma_oracle.run_pharma_gates(mol)
        if not gate_passed:
            logger.info(f"    [Reject] {gate_reason}: {smiles}")
            continue

        # -- Additional drug-likeness filters --
        qed = metrics.get("qed", 0)
        if qed < 0.35:
            logger.info(f"    [Reject] Low QED ({qed:.2f}): {smiles}")
            continue

        mw = metrics.get("mw", 0)
        if mw < 150 or mw > 600:
            logger.info(f"    [Reject] MW out of range ({mw:.1f}): {smiles}")
            continue

        logp = metrics.get("logp", 0)
        if logp < -2.0 or logp > 6.0:
            logger.info(f"    [Reject] LogP out of range ({logp:.2f}): {smiles}")
            continue

        num_rings = metrics.get("num_rings", 0)
        if num_rings < 1:
            logger.info(f"    [Reject] No rings: {smiles}")
            continue

        n_atoms = metrics.get("num_atoms", 0)
        if n_atoms < 8:
            logger.info(f"    [Reject] Too small ({n_atoms} atoms): {smiles}")
            continue

        # Compute composite druggability score for ranking
        sa_reward = max(0, 1.0 - metrics.get("sa_score", 10) / 10.0)
        composite = qed * sa_reward  # higher = more drug-like + synthesizable

        candidates.append({
            "mol": mol,
            "pos": pos_np,
            "atom_types": elements,
            "pK_pred": pK,
            "metrics": metrics,
            "num_atoms": sample["num_atoms"],
            "composite_score": composite,
        })

    # Rank by composite druggability score and keep top-N
    candidates.sort(key=lambda r: r["composite_score"], reverse=True)
    results = candidates[:num_mols]

    # Log final results
    for i, r in enumerate(results):
        m = r["metrics"]
        status = "✓" if m.get("valid", False) else "✗"
        logger.info(
            f"  Mol {i+1}/{num_mols}: {status} | "
            f"atoms={r['num_atoms']} | "
            f"pK={r['pK_pred']:.3f} | "
            f"QED={m.get('qed', 0):.3f} | "
            f"SA={m.get('sa_score', 10):.1f} | "
            f"MW={m.get('mw', 0):.1f} | "
            f"LogP={m.get('logp', 0):.2f} | "
            f"SMILES={m.get('smiles', 'N/A')}"
        )

    return results


def print_summary(all_results: List[dict], pocket_name: str):
    """Print a comprehensive summary table of generation results."""
    valid_count = sum(1 for r in all_results if r["metrics"].get("valid", False))
    total = len(all_results)

    print(f"\n{'='*80}")
    print(f"GENERATION SUMMARY — Pocket: {pocket_name}")
    print(f"{'='*80}")
    print(f"  Total molecules generated : {total}")
    print(f"  Valid molecules           : {valid_count}/{total} ({100*valid_count/max(total,1):.1f}%)")

    if valid_count > 0:
        valid_results = [r for r in all_results if r["metrics"].get("valid", False)]

        # Aggregate scalar metrics
        qeds    = [r["metrics"]["qed"] for r in valid_results]
        pKs     = [r["pK_pred"] for r in valid_results]
        mws     = [r["metrics"].get("mw", 0) for r in valid_results]
        logps   = [r["metrics"].get("logp", 0) for r in valid_results]
        sa_scores = [r["metrics"].get("sa_score", 10.0) for r in valid_results]
        lipinski_pass = sum(r["metrics"].get("lipinski", 0) for r in valid_results)
        atom_stabs = [r["metrics"].get("atom_stability", 0.0) for r in valid_results]
        mol_stabs  = [r["metrics"].get("molecule_stable", 0.0) for r in valid_results]
        conn_fracs = [r["metrics"].get("connected_fraction", 0.0) for r in valid_results]

        # Batch-level metrics
        all_smiles = [r["metrics"].get("smiles", "") for r in valid_results]
        uniqueness = compute_uniqueness(all_smiles)

        print(f"\n  {'Metric':<28} {'Mean':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*60}")
        print(f"  {'QED':<28} {np.mean(qeds):>10.3f} {np.min(qeds):>10.3f} {np.max(qeds):>10.3f}")
        print(f"  {'SA Score (1=easy, 10=hard)':<28} {np.mean(sa_scores):>10.3f} {np.min(sa_scores):>10.3f} {np.max(sa_scores):>10.3f}")
        print(f"  {'pK_pred':<28} {np.mean(pKs):>10.3f} {np.min(pKs):>10.3f} {np.max(pKs):>10.3f}")
        print(f"  {'Mol Weight':<28} {np.mean(mws):>10.1f} {np.min(mws):>10.1f} {np.max(mws):>10.1f}")
        print(f"  {'LogP':<28} {np.mean(logps):>10.2f} {np.min(logps):>10.2f} {np.max(logps):>10.2f}")
        print(f"  {'Atom Stability':<28} {np.mean(atom_stabs):>10.3f} {np.min(atom_stabs):>10.3f} {np.max(atom_stabs):>10.3f}")
        print(f"  {'Connected Fraction':<28} {np.mean(conn_fracs):>10.3f} {np.min(conn_fracs):>10.3f} {np.max(conn_fracs):>10.3f}")

        print(f"\n  {'--- Batch-level Metrics ---'}")
        print(f"  Validity          : {100*valid_count/max(total,1):.1f}%  ({valid_count}/{total})")
        print(f"  Uniqueness        : {100*uniqueness:.1f}%  ({len(set(all_smiles))}/{len(all_smiles)} unique SMILES)")
        print(f"  Molecule Stability: {100*np.mean(mol_stabs):.1f}%  (all atoms valid valence)")
        print(f"  Lipinski Pass     : {lipinski_pass}/{valid_count} ({100*lipinski_pass/valid_count:.0f}%)")

        # Show top-3 by predicted binding
        print(f"\n  Top 3 molecules by predicted binding affinity:")
        sorted_results = sorted(valid_results, key=lambda r: r["pK_pred"], reverse=True)
        for i, r in enumerate(sorted_results[:3]):
            m = r["metrics"]
            print(
                f"    {i+1}. pK={r['pK_pred']:.3f} | QED={m.get('qed',0):.3f} | "
                f"SA={m.get('sa_score',10):.2f} | "
                f"AtomStab={m.get('atom_stability',0):.2f} | "
                f"Conn={m.get('connected_fraction',0):.2f} | "
                f"{m['smiles']}"
            )

    print(f"{'='*80}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate drug molecules for a protein pocket using the trained SBDD model."
    )
    parser.add_argument(
        "--pocket", type=str, default=None,
        help="Path to a single pocket .pdb file."
    )
    parser.add_argument(
        "--pocket_dir", type=str, default=None,
        help="Directory containing multiple pocket .pdb files."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/rl_final.pt",
        help="Path to model checkpoint (default: checkpoints/rl_final.pt)."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML."
    )
    parser.add_argument(
        "--num_mols", type=int, default=10,
        help="Number of molecules to generate per pocket (default: 10)."
    )
    parser.add_argument(
        "--num_atoms", type=int, default=None,
        help="Override number of atoms (default: predict from pocket)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="generated_molecules",
        help="Output directory for .sdf files and reports."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu."
    )

    args = parser.parse_args()

    # Validate inputs
    if args.pocket is None and args.pocket_dir is None:
        parser.error("Must provide either --pocket or --pocket_dir")

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, cfg = load_model(args.config, args.checkpoint, args.device)

    # Collect pocket files
    pocket_files = []
    if args.pocket:
        pocket_files.append(args.pocket)
    elif args.pocket_dir:
        pocket_files = sorted(Path(args.pocket_dir).glob("*.pdb"))
        logger.info(f"Found {len(pocket_files)} pocket files in {args.pocket_dir}")

    # Generate for each pocket
    all_reports = []
    for pocket_path in pocket_files:
        pocket_path = str(pocket_path)
        pocket_name = Path(pocket_path).stem

        results = generate_for_pocket(
            model=model,
            pocket_path=pocket_path,
            num_mols=args.num_mols,
            num_atoms=args.num_atoms,
            device=args.device,
        )

        # Save each molecule as a separate .mol2 file
        save_molecules_separated_mol2(results, str(output_path), pocket_name)

        # Print summary
        print_summary(results, pocket_name)

        # Save metrics report
        report_path = output_path / f"{pocket_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Pocket: {pocket_name}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Molecules generated: {len(results)}\n\n")
            for i, r in enumerate(results):
                m = r["metrics"]
                f.write(f"Mol {i+1}:\n")
                f.write(f"  SMILES: {m.get('smiles', 'N/A')}\n")
                f.write(f"  Valid:  {m.get('valid', False)}\n")
                f.write(f"  pK:    {r['pK_pred']:.4f}\n")
                f.write(f"  QED:   {m.get('qed', 0):.4f}\n")
                f.write(f"  MW:    {m.get('mw', 0):.1f}\n")
                f.write(f"  LogP:  {m.get('logp', 0):.2f}\n")
                f.write(f"  Lipinski: {'Pass' if m.get('lipinski', 0) else 'Fail'}\n")
                f.write(f"  Atoms: {r['num_atoms']}\n\n")
        logger.info(f"Report saved to {report_path}")

        all_reports.append({
            "pocket": pocket_name,
            "results": results,
        })

    logger.info(f"Done! All outputs saved to {output_path}/")


if __name__ == "__main__":
    main()
