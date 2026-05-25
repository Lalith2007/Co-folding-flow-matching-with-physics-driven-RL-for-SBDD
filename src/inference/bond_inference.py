"""
bond_inference.py — Convert raw 3D coordinates + atom type indices into SMILES.

Two strategies:
  1. Distance-based single bond inference + iterative valence repair (primary)
     This is robust to dense/compressed geometries from the generative model.
  2. RDKit's rdDetermineBonds (used when strategy 1 produces a valid mol)

The distance-based approach mirrors the proven logic in generate.py.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ── Atom type index → element symbol mapping ──
# Must match LIGAND_ATOM_TYPES in featurizer.py
IDX_TO_ELEMENT = ['C', 'N', 'O', 'S', 'F', 'Cl']

# Typical covalent radii (Angstroms)
COVALENT_RADII = {
    "C": 0.77, "N": 0.75, "O": 0.73, "S": 1.05, "F": 0.71,
    "Cl": 0.99, "Br": 1.14, "I": 1.33, "P": 1.10, "B": 0.82,
}

# RDKit atomic numbers
ELEMENT_TO_ATOMIC_NUM = {
    "C": 6, "N": 7, "O": 8, "S": 16, "F": 9,
    "Cl": 17, "Br": 35, "I": 53, "P": 15, "B": 5,
}


def coords_to_smiles(
    coords: np.ndarray,       # (N, 3) float
    atom_types: np.ndarray,   # (N,) int indices
    method: str = "rdkit",
    charge: int = 0,
) -> Dict:
    """Convert 3D atomic coordinates and types into a SMILES string.

    Uses distance-based bond inference with iterative valence repair
    as the primary strategy (robust to compressed geometries).
    Falls back to rdDetermineBonds if distance-based produces invalid results.

    Returns dict with smiles, mol, success, error, num_atoms, num_bonds.
    """
    N = len(coords)
    if N == 0:
        return _fail("Empty coordinate array")

    elements = []
    for idx in atom_types:
        if 0 <= idx < len(IDX_TO_ELEMENT):
            elements.append(IDX_TO_ELEMENT[idx])
        else:
            elements.append("C")  # fallback to carbon

    # Strategy 1: Distance-based bonds + iterative valence repair
    # This is the same proven approach used in generate.py
    result = _distance_based_bond_inference(coords, elements)
    if result["success"]:
        return result
    logger.debug(f"Distance-based bond inference failed: {result['error']}")

    # Strategy 2: Try RDKit's rdDetermineBonds
    if method == "rdkit":
        result = _rdkit_bond_perception(coords, elements, charge)
        if result["success"]:
            return result
        logger.warning(f"RDKit bond perception failed: {result['error']}")

    # Strategy 3: Fallback to OpenBabel
    result = _openbabel_bond_perception(coords, elements, charge)
    if result["success"]:
        return result

    return _fail(f"All bond perception methods failed for {N} atoms")


def _distance_based_bond_inference(
    coords: np.ndarray,
    elements: List[str],
    bond_tolerance: float = 0.3,
) -> Dict:
    """Distance-based single bond inference with iterative valence repair.

    This mirrors the proven approach from generate.py that successfully
    converts even compressed geometries into valid SMILES.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import GetPeriodicTable
        from rdkit.Geometry import Point3D

        pt = GetPeriodicTable()
        N = len(elements)

        mol = Chem.RWMol()
        for elem in elements:
            atomic_num = ELEMENT_TO_ATOMIC_NUM.get(elem, 6)
            mol.AddAtom(Chem.Atom(atomic_num))

        # Add single bonds between atoms within covalent distance
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(coords[i] - coords[j])
                r_i = COVALENT_RADII.get(elements[i], 1.0)
                r_j = COVALENT_RADII.get(elements[j], 1.0)
                if dist < r_i + r_j + bond_tolerance:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        # Add conformer
        conf = Chem.Conformer(N)
        for i in range(N):
            conf.SetAtomPosition(i, Point3D(
                float(coords[i, 0]),
                float(coords[i, 1]),
                float(coords[i, 2]),
            ))
        mol.AddConformer(conf, assignId=True)

        # Iterative valence repair: remove longest bonds from over-bonded atoms
        max_iterations = 500  # safety limit
        for _ in range(max_iterations):
            try:
                mol_copy = Chem.Mol(mol)
                Chem.SanitizeMol(mol_copy)
                # Success! Get SMILES from the largest fragment
                frags = Chem.GetMolFrags(mol_copy, asMols=True)
                if frags:
                    largest = max(frags, key=lambda f: f.GetNumAtoms())
                    try:
                        Chem.SanitizeMol(largest)
                    except Exception:
                        pass
                    smiles = Chem.MolToSmiles(largest)
                    if smiles:
                        return {
                            "smiles": smiles,
                            "mol": largest,
                            "success": True,
                            "error": None,
                            "num_atoms": largest.GetNumAtoms(),
                            "num_bonds": largest.GetNumBonds(),
                        }
                return _fail("Distance-based produced empty SMILES")
            except Exception:
                pass

            # Find an over-bonded atom and remove its longest bond
            fixed = False
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                sym = atom.GetSymbol()
                max_v = pt.GetDefaultValence(atom.GetAtomicNum())
                # Override known max valences
                if sym == 'N': max_v = 3
                if sym == 'O': max_v = 2
                if sym == 'S': max_v = max(max_v, 6)
                if sym == 'P': max_v = max(max_v, 5)

                if atom.GetDegree() > max_v:
                    longest_bond, max_d = None, -1.0
                    for bond in atom.GetBonds():
                        n_idx = bond.GetOtherAtom(atom).GetIdx()
                        d = np.linalg.norm(coords[idx] - coords[n_idx])
                        if d > max_d:
                            max_d, longest_bond = d, bond
                    if longest_bond:
                        mol.RemoveBond(
                            longest_bond.GetBeginAtomIdx(),
                            longest_bond.GetEndAtomIdx(),
                        )
                        fixed = True
                        break
            if not fixed:
                # No more over-bonded atoms but sanitization still fails
                # Try to get SMILES anyway
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles:
                        return {
                            "smiles": smiles,
                            "mol": mol.GetMol(),
                            "success": True,
                            "error": None,
                            "num_atoms": mol.GetNumAtoms(),
                            "num_bonds": mol.GetNumBonds(),
                        }
                except Exception:
                    pass
                return _fail("Distance-based could not repair valence")

        return _fail("Distance-based valence repair exceeded iteration limit")

    except ImportError:
        return _fail("RDKit not installed")
    except Exception as e:
        return _fail(f"Distance-based error: {str(e)}")


def _rdkit_bond_perception(
    coords: np.ndarray,
    elements: List[str],
    charge: int = 0,
) -> Dict:
    """Use RDKit's DetermineBonds to infer connectivity and bond orders."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D

        mol = Chem.RWMol()
        conf = Chem.Conformer(len(elements))

        for i, elem in enumerate(elements):
            atom_idx = mol.AddAtom(Chem.Atom(elem))
            conf.SetAtomPosition(atom_idx, Point3D(
                float(coords[i, 0]),
                float(coords[i, 1]),
                float(coords[i, 2]),
            ))

        mol.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False)
        rdDetermineBonds.DetermineBondOrders(mol, charge=charge)

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )

        smiles = Chem.MolToSmiles(mol)
        if not smiles:
            return _fail("RDKit produced empty SMILES")

        return {
            "smiles": smiles,
            "mol": mol.GetMol(),
            "success": True,
            "error": None,
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
        }

    except ImportError:
        return _fail("RDKit not installed")
    except Exception as e:
        return _fail(f"RDKit error: {str(e)}")


def _openbabel_bond_perception(
    coords: np.ndarray,
    elements: List[str],
    charge: int = 0,
) -> Dict:
    """Fallback: write XYZ file and use OpenBabel to perceive bonds."""
    try:
        from openbabel import openbabel as ob

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write(f"{len(elements)}\n")
            f.write("Generated by SBDD pipeline\n")
            for i, elem in enumerate(elements):
                f.write(
                    f"{elem:2s}  {coords[i,0]:12.6f}  "
                    f"{coords[i,1]:12.6f}  {coords[i,2]:12.6f}\n"
                )
            xyz_path = f.name

        conv = ob.OBConversion()
        conv.SetInFormat("xyz")
        conv.SetOutFormat("smi")

        mol = ob.OBMol()
        conv.ReadFile(mol, xyz_path)

        mol.SetTotalCharge(charge)
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()

        smiles = conv.WriteString(mol).strip().split()[0]
        Path(xyz_path).unlink(missing_ok=True)

        if not smiles:
            return _fail("OpenBabel produced empty SMILES")

        try:
            from rdkit import Chem
            rdmol = Chem.MolFromSmiles(smiles)
            if rdmol is None:
                return _fail(f"OpenBabel SMILES '{smiles}' failed RDKit validation")
            smiles = Chem.MolToSmiles(rdmol)
        except ImportError:
            pass

        return {
            "smiles": smiles,
            "mol": None,
            "success": True,
            "error": None,
            "num_atoms": len(elements),
            "num_bonds": mol.NumBonds(),
        }

    except ImportError:
        return _fail("OpenBabel not installed")
    except Exception as e:
        return _fail(f"OpenBabel error: {str(e)}")


def _fail(error: str) -> Dict:
    """Return a standardized failure result."""
    return {
        "smiles": "",
        "mol": None,
        "success": False,
        "error": error,
        "num_atoms": 0,
        "num_bonds": 0,
    }


def validate_smiles(smiles: str) -> Dict:
    """Validate a SMILES string and compute basic molecular properties."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"valid": False, "error": "RDKit could not parse SMILES"}

        return {
            "valid": True,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "qed": round(QED.qed(mol), 4),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
        }

    except ImportError:
        return {"valid": False, "error": "RDKit not installed"}
    except Exception as e:
        return {"valid": False, "error": str(e)}
