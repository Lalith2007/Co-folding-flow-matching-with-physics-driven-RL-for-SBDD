"""
bond_inference.py — Convert raw 3D coordinates + atom types into valid SMILES.

The Flow Matching model outputs continuous 3D coordinates and categorical
atom types, but NO explicit bonds. This module reconstructs the molecular
graph (bonds, bond orders, charges) from the point cloud using
cheminformatics heuristics.

Strategy:
  1. Build an RDKit RWMol with the correct atoms at the predicted positions.
  2. Use rdDetermineBonds.DetermineConnectivity to infer single bonds
     based on van der Waals radii.
  3. Use rdDetermineBonds.DetermineBondOrders to assign bond orders
     (single, double, aromatic) based on valence rules.
  4. Sanitize and canonicalize the SMILES.
  5. If RDKit fails, fall back to OpenBabel's perception.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Atom type index → element symbol mapping ──
# Must match LIGAND_ATOM_TYPES in featurizer.py
IDX_TO_ELEMENT = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "B"]


def coords_to_smiles(
    coords: np.ndarray,       # (N, 3) float
    atom_types: np.ndarray,   # (N,) int indices
    method: str = "rdkit",
    charge: int = 0,
) -> Dict:
    """Convert 3D atomic coordinates and types into a SMILES string.

    Parameters
    ----------
    coords     : (N, 3) array of atom positions in Angstroms
    atom_types : (N,)   array of integer atom type indices
    method     : 'rdkit' (default) or 'openbabel' fallback
    charge     : net molecular charge (default 0)

    Returns
    -------
    dict with:
        smiles       : str — canonical SMILES (empty string if failed)
        mol          : rdkit.Chem.Mol or None
        success      : bool
        error        : str or None
        num_atoms    : int
        num_bonds    : int
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

    # Try RDKit first
    if method == "rdkit":
        result = _rdkit_bond_perception(coords, elements, charge)
        if result["success"]:
            return result
        logger.warning(f"RDKit bond perception failed: {result['error']}")

    # Fallback to OpenBabel
    result = _openbabel_bond_perception(coords, elements, charge)
    if result["success"]:
        return result

    return _fail(f"All bond perception methods failed for {N} atoms")


def _rdkit_bond_perception(
    coords: np.ndarray,
    elements: List[str],
    charge: int = 0,
) -> Dict:
    """Use RDKit's DetermineBonds to infer connectivity and bond orders."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds, AllChem
        from rdkit.Geometry import Point3D

        # Build an editable molecule with atoms at the right positions
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

        # Step 1: Determine connectivity (single bonds from distances)
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False)

        # Step 2: Determine bond orders (double, triple, aromatic)
        rdDetermineBonds.DetermineBondOrders(mol, charge=charge)

        # Sanitize
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.debug(f"Sanitization failed, trying partial: {e}")
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )

        # Generate canonical SMILES
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

        # Write temporary XYZ
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

        # Read with OpenBabel
        conv = ob.OBConversion()
        conv.SetInFormat("xyz")
        conv.SetOutFormat("smi")

        mol = ob.OBMol()
        conv.ReadFile(mol, xyz_path)

        mol.SetTotalCharge(charge)
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()

        smiles = conv.WriteString(mol).strip().split()[0]

        # Cleanup
        Path(xyz_path).unlink(missing_ok=True)

        if not smiles:
            return _fail("OpenBabel produced empty SMILES")

        # Validate with RDKit if available
        try:
            from rdkit import Chem
            rdmol = Chem.MolFromSmiles(smiles)
            if rdmol is None:
                return _fail(f"OpenBabel SMILES '{smiles}' failed RDKit validation")
            smiles = Chem.MolToSmiles(rdmol)  # canonicalize
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
    """Validate a SMILES string and compute basic molecular properties.

    Returns dict with validity, canonical SMILES, MW, QED, etc.
    """
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
            "num_rings": Chem.rdMolDescriptors.CalcNumRings(mol),
        }
    except ImportError:
        return {"valid": False, "error": "RDKit not installed"}
    except Exception as e:
        return {"valid": False, "error": str(e)}
