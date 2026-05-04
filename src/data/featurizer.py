"""
featurizer.py — Molecular featurization for pockets (.pdb / .mol2) and
ligands (.sdf).  Builds k-NN graphs on pocket atoms and extracts per-atom
feature vectors compatible with the SE(3)-EGNN architecture.

All features are aligned to a *unified atom vocabulary* so that .pdb and
.mol2 pockets produce the same feature dimensionality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Element vocabulary (15 elements + UNK → 16-dim one-hot) ──
ELEMENT_LIST = [
    "C", "N", "O", "S", "H", "F", "Cl", "Br", "P", "I",
    "B", "Se", "Si", "Fe", "Zn",
]
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENT_LIST)}
NUM_ELEMENTS = len(ELEMENT_LIST) + 1  # +1 for UNK

# ── Amino acid vocabulary (20 standard + UNK → 21-dim one-hot) ──
AA_LIST = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
NUM_AA = len(AA_LIST) + 1  # +1 for UNK

# ── Ligand atom type vocabulary (for categorical flow) ──
LIGAND_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "B"]
LIGAND_ATOM_TO_IDX = {a: i for i, a in enumerate(LIGAND_ATOM_TYPES)}
NUM_LIGAND_ATOM_TYPES = len(LIGAND_ATOM_TYPES)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: k-NN graph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_knn_graph(
    pos: torch.Tensor,   # (N, 3)
    k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a k-nearest-neighbour graph from 3-D coordinates.

    Returns
    -------
    edge_index : (2, E)  long tensor of directed edges
    edge_dist  : (E,)    Euclidean distances for each edge
    """
    # Pairwise distance matrix
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N, N, 3)
    dist = torch.norm(diff, dim=-1)                # (N, N)

    # For each node, pick the k nearest neighbours (excluding self)
    n = pos.size(0)
    k_actual = min(k, n - 1)

    # Set self-distance to inf so it is never selected
    dist_no_self = dist.clone()
    dist_no_self.fill_diagonal_(float("inf"))

    _, knn_idx = dist_no_self.topk(k_actual, dim=-1, largest=False)  # (N, k)

    # Build edge_index [src, dst]
    src = torch.arange(n).unsqueeze(1).expand_as(knn_idx).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)  # (2, N*k)

    # Edge distances
    edge_dist = dist[src, dst]

    return edge_index, edge_dist


# ──────────────────────────────────────────────────────────────────────────────
# Radial Basis Function (RBF) edge features
# ──────────────────────────────────────────────────────────────────────────────

def rbf_encode(
    distances: torch.Tensor,
    d_min: float = 0.0,
    d_max: float = 20.0,
    num_rbf: int = 16,
) -> torch.Tensor:
    """Expand scalar distances into RBF features.

    Parameters
    ----------
    distances : (E,) tensor of distances
    d_min, d_max : range of RBF centres
    num_rbf : number of Gaussian centres

    Returns
    -------
    rbf_feat : (E, num_rbf) tensor
    """
    mu = torch.linspace(d_min, d_max, num_rbf, device=distances.device)
    sigma = (d_max - d_min) / num_rbf
    diff = distances.unsqueeze(-1) - mu.unsqueeze(0)
    return torch.exp(-0.5 * (diff / sigma) ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Pocket Featurizer
# ──────────────────────────────────────────────────────────────────────────────

class PocketFeaturizer:
    """Featurize a pocket file (.pdb or .mol2) into a dict of tensors.

    Output dict keys:
        pos  : (N_P, 3)   — atom 3-D coordinates
        feat : (N_P, F)   — per-atom feature vector
    """

    def __init__(self, cutoff: float = 8.0, knn_k: int = 16):
        self.cutoff = cutoff
        self.knn_k = knn_k

    def featurize(self, path: str) -> Dict[str, torch.Tensor]:
        p = Path(path)
        if p.suffix == ".pdb":
            return self._featurize_pdb(path)
        elif p.suffix == ".mol2":
            return self._featurize_mol2(path)
        else:
            raise ValueError(f"Unsupported pocket format: {p.suffix}")

    # ── PDB pocket parsing (BioPython) ──
    def _featurize_pdb(self, path: str) -> Dict[str, torch.Tensor]:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pocket", path)

        coords = []
        features = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname().strip()
                    aa_idx = AA_TO_IDX.get(res_name, NUM_AA - 1)

                    for atom in residue:
                        element = atom.element.strip()
                        elem_idx = ELEMENT_TO_IDX.get(element, NUM_ELEMENTS - 1)

                        # One-hot element (16-dim)
                        elem_oh = np.zeros(NUM_ELEMENTS, dtype=np.float32)
                        elem_oh[elem_idx] = 1.0

                        # One-hot amino acid (21-dim)
                        aa_oh = np.zeros(NUM_AA, dtype=np.float32)
                        aa_oh[aa_idx] = 1.0

                        # Backbone / sidechain flag
                        is_backbone = 1.0 if atom.get_name() in (
                            "N", "CA", "C", "O"
                        ) else 0.0

                        # B-factor (normalised)
                        bfactor = atom.get_bfactor() / 100.0

                        feat = np.concatenate([
                            elem_oh,                     # 16
                            aa_oh,                       # 21
                            [is_backbone],               # 1
                            [bfactor],                   # 1
                            [0.0],                       # 1  placeholder for dist_to_centroid
                        ])  # total = 40

                        coords.append(atom.get_vector().get_array())
                        features.append(feat)

        pos = torch.tensor(np.array(coords), dtype=torch.float32)
        feat = torch.tensor(np.array(features), dtype=torch.float32)

        # Distance to pocket centroid (normalised by max distance)
        centroid = pos.mean(dim=0, keepdim=True)              # (1, 3)
        dist_to_centroid = torch.norm(pos - centroid, dim=-1)  # (N,)
        max_dist = dist_to_centroid.max().clamp(min=1e-8)
        feat[:, -1] = dist_to_centroid / max_dist              # fill last column

        return {"pos": pos, "feat": feat}

    # ── MOL2 pocket parsing (RDKit) ──
    def _featurize_mol2(self, path: str) -> Dict[str, torch.Tensor]:
        from rdkit import Chem

        mol = Chem.MolFromMol2File(path, sanitize=False, removeHs=True)
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse MOL2: {path}")

        conf = mol.GetConformer()
        coords = []
        features = []

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos3d = conf.GetAtomPosition(idx)

            element = atom.GetSymbol()
            elem_idx = ELEMENT_TO_IDX.get(element, NUM_ELEMENTS - 1)
            elem_oh = np.zeros(NUM_ELEMENTS, dtype=np.float32)
            elem_oh[elem_idx] = 1.0

            # For .mol2, we don't have residue info → use UNK
            aa_oh = np.zeros(NUM_AA, dtype=np.float32)
            aa_oh[NUM_AA - 1] = 1.0  # UNK

            # No backbone flag for mol2
            is_backbone = 0.0
            bfactor = 0.0

            feat = np.concatenate([
                elem_oh,            # 16
                aa_oh,              # 21
                [is_backbone],      # 1
                [bfactor],          # 1
                [0.0],              # 1  placeholder for dist_to_centroid
            ])  # total = 40

            coords.append([pos3d.x, pos3d.y, pos3d.z])
            features.append(feat)

        pos = torch.tensor(np.array(coords), dtype=torch.float32)
        feat = torch.tensor(np.array(features), dtype=torch.float32)

        # Distance to pocket centroid (normalised by max distance)
        centroid = pos.mean(dim=0, keepdim=True)
        dist_to_centroid = torch.norm(pos - centroid, dim=-1)
        max_dist = dist_to_centroid.max().clamp(min=1e-8)
        feat[:, -1] = dist_to_centroid / max_dist

        return {"pos": pos, "feat": feat}


# ──────────────────────────────────────────────────────────────────────────────
# Ligand Featurizer
# ──────────────────────────────────────────────────────────────────────────────

class LigandFeaturizer:
    """Featurize a ligand SDF file into tensors.

    Output dict keys:
        pos        : (N_L, 3)   — atom 3-D coordinates
        feat       : (N_L, F)   — per-atom features
        atom_types : (N_L,)     — integer atom type indices for categorical flow
    """

    def featurize(self, path: str) -> Dict[str, torch.Tensor]:
        from rdkit import Chem

        suppl = Chem.SDMolSupplier(path, sanitize=True, removeHs=True)
        mol = next(iter(suppl), None)
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse SDF: {path}")

        conf = mol.GetConformer()
        coords = []
        features = []
        atom_type_indices = []

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos3d = conf.GetAtomPosition(idx)

            element = atom.GetSymbol()
            at_idx = LIGAND_ATOM_TO_IDX.get(element, 0)  # default to C
            atom_type_indices.append(at_idx)

            # One-hot element
            elem_idx = ELEMENT_TO_IDX.get(element, NUM_ELEMENTS - 1)
            elem_oh = np.zeros(NUM_ELEMENTS, dtype=np.float32)
            elem_oh[elem_idx] = 1.0

            # Aromaticity
            is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0

            # Degree
            degree = atom.GetDegree() / 6.0  # normalised

            # Formal charge
            charge = float(atom.GetFormalCharge())

            # In ring
            in_ring = 1.0 if atom.IsInRing() else 0.0

            feat = np.concatenate([
                elem_oh,             # 16
                [is_aromatic],       # 1
                [degree],            # 1
                [charge],            # 1
                [in_ring],           # 1
            ])  # total = 20

            coords.append([pos3d.x, pos3d.y, pos3d.z])
            features.append(feat)

        pos = torch.tensor(np.array(coords), dtype=torch.float32)
        feat = torch.tensor(np.array(features), dtype=torch.float32)
        atom_types = torch.tensor(atom_type_indices, dtype=torch.long)

        return {"pos": pos, "feat": feat, "atom_types": atom_types}
