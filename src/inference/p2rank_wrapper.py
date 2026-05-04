"""
p2rank_wrapper.py — Automatic pocket detection from a raw PDB file.

Wraps the P2Rank tool (https://github.com/rdk/p2rank) via subprocess.
P2Rank is a machine learning method for prediction of ligand binding
sites from protein structure. It outputs ranked pocket predictions.

This module:
  1. Downloads p2rank automatically if not found on the system.
  2. Runs `prank predict -f <pdb_file>` via subprocess.
  3. Parses the output CSV to extract the top-ranked pocket residues.
  4. Crops the original PDB to just those residues (the pocket).
"""

from __future__ import annotations

import csv
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Default P2Rank installation directory ──
_DEFAULT_P2RANK_DIR = Path.home() / ".sbdd_tools" / "p2rank"
_P2RANK_VERSION = "2.4.2"
_P2RANK_URL = (
    f"https://github.com/rdk/p2rank/releases/download/"
    f"v{_P2RANK_VERSION}/p2rank_{_P2RANK_VERSION}.tar.gz"
)


def _find_prank_binary(p2rank_home: Optional[str] = None) -> Path:
    """Locate the prank executable, downloading P2Rank if necessary.

    Search order:
      1. User-supplied p2rank_home
      2. PRANK_HOME environment variable
      3. System PATH (`which prank`)
      4. Default install at ~/.sbdd_tools/p2rank/

    If none found, automatically downloads and extracts P2Rank.
    """
    # 1. Explicit path
    if p2rank_home:
        candidate = Path(p2rank_home) / "prank"
        if candidate.exists():
            return candidate

    # 2. Environment variable
    env_home = os.environ.get("PRANK_HOME")
    if env_home:
        candidate = Path(env_home) / "prank"
        if candidate.exists():
            return candidate

    # 3. System PATH
    which = shutil.which("prank")
    if which:
        return Path(which)

    # 4. Default installation
    default_bin = _DEFAULT_P2RANK_DIR / f"p2rank_{_P2RANK_VERSION}" / "prank"
    if default_bin.exists():
        return default_bin

    # ── Auto-install ──
    logger.info(f"P2Rank not found. Downloading v{_P2RANK_VERSION}...")
    _download_and_install_p2rank()
    if default_bin.exists():
        return default_bin

    raise FileNotFoundError(
        "P2Rank could not be found or installed. "
        "Please install manually from https://github.com/rdk/p2rank "
        "and set the PRANK_HOME environment variable."
    )


def _download_and_install_p2rank():
    """Download and extract P2Rank to the default directory."""
    import urllib.request

    _DEFAULT_P2RANK_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = _DEFAULT_P2RANK_DIR / f"p2rank_{_P2RANK_VERSION}.tar.gz"

    logger.info(f"Downloading P2Rank from {_P2RANK_URL}")
    urllib.request.urlretrieve(_P2RANK_URL, str(tar_path))

    logger.info(f"Extracting to {_DEFAULT_P2RANK_DIR}")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(path=str(_DEFAULT_P2RANK_DIR))

    # Make executable
    prank_bin = _DEFAULT_P2RANK_DIR / f"p2rank_{_P2RANK_VERSION}" / "prank"
    if prank_bin.exists():
        prank_bin.chmod(0o755)
        logger.info(f"P2Rank installed successfully at {prank_bin}")

    # Cleanup tarball
    tar_path.unlink(missing_ok=True)


def run_p2rank(
    pdb_path: str,
    p2rank_home: Optional[str] = None,
    output_dir: Optional[str] = None,
    timeout: int = 300,
) -> Dict:
    """Run P2Rank on a PDB file and return pocket predictions.

    Parameters
    ----------
    pdb_path    : absolute path to the input .pdb file
    p2rank_home : path to the p2rank installation directory (optional)
    output_dir  : where to write p2rank output (default: temp dir)
    timeout     : max seconds to wait for p2rank (default: 300)

    Returns
    -------
    dict with:
        pockets : list of dicts, each with 'rank', 'score', 'residues', 'center'
        output_dir : path to the p2rank output directory
    """
    prank = _find_prank_binary(p2rank_home)
    pdb_path = Path(pdb_path).resolve()

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="p2rank_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run p2rank
    cmd = [
        str(prank), "predict",
        "-f", str(pdb_path),
        "-o", str(output_dir),
    ]

    logger.info(f"Running P2Rank: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            logger.error(f"P2Rank stderr: {result.stderr}")
            raise RuntimeError(f"P2Rank failed with return code {result.returncode}")
    except FileNotFoundError:
        raise RuntimeError(
            "P2Rank execution failed. Ensure Java is installed (java -version)."
        )

    # Parse predictions
    pdb_name = pdb_path.stem
    predictions_csv = output_dir / f"{pdb_name}.pdb_predictions.csv"
    residues_csv = output_dir / f"{pdb_name}.pdb_residues.csv"

    pockets = _parse_predictions(predictions_csv, residues_csv)

    logger.info(f"P2Rank found {len(pockets)} pockets")
    for p in pockets[:3]:
        logger.info(
            f"  Pocket {p['rank']}: score={p['score']:.2f}, "
            f"residues={len(p['residues'])}, center={p['center']}"
        )

    return {"pockets": pockets, "output_dir": str(output_dir)}


def _parse_predictions(
    predictions_csv: Path,
    residues_csv: Path,
) -> List[Dict]:
    """Parse P2Rank output CSVs into structured pocket data."""
    pockets = []

    if not predictions_csv.exists():
        logger.warning(f"Predictions CSV not found: {predictions_csv}")
        return pockets

    # Parse predictions.csv for pocket centers and scores
    with open(predictions_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pocket = {
                "rank": int(row.get("rank", row.get(" rank", "0")).strip()),
                "name": row.get("name", row.get(" name", "")).strip(),
                "score": float(row.get("score", row.get(" score", "0")).strip()),
                "center": [
                    float(row.get("center_x", row.get("   center_x", "0")).strip()),
                    float(row.get("center_y", row.get("   center_y", "0")).strip()),
                    float(row.get("center_z", row.get("   center_z", "0")).strip()),
                ],
                "residues": [],
            }
            pockets.append(pocket)

    # Parse residues.csv to assign residues to pockets
    if residues_csv.exists():
        with open(residues_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # P2Rank uses space-padded headers
                pocket_str = row.get("pocket", row.get("   pocket", "")).strip()
                if not pocket_str:
                    continue
                try:
                    pocket_idx = int(pocket_str) - 1  # 1-indexed → 0-indexed
                except (ValueError, IndexError):
                    continue
                if 0 <= pocket_idx < len(pockets):
                    chain = row.get("chain", row.get("   chain", "")).strip()
                    residue_label = row.get("residue_label",
                                           row.get("   residue_label", "")).strip()
                    residue_name = row.get("residue_name",
                                          row.get("   residue_name", "")).strip()
                    pockets[pocket_idx]["residues"].append({
                        "chain": chain,
                        "label": residue_label,
                        "name": residue_name,
                    })

    return pockets


def extract_pocket_pdb(
    pdb_path: str,
    pocket: Dict,
    cutoff: float = 8.0,
    output_path: Optional[str] = None,
) -> str:
    """Extract pocket residues from a PDB file and write a cropped PDB.

    Parameters
    ----------
    pdb_path   : path to the original PDB
    pocket     : pocket dict from run_p2rank (must have 'residues' or 'center')
    cutoff     : Å cutoff around pocket center for fallback selection
    output_path: where to write the pocket PDB (default: temp file)

    Returns
    -------
    path to the cropped pocket PDB file
    """
    from Bio.PDB import PDBParser, PDBIO, Select

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    # Build set of pocket residue identifiers
    pocket_residue_keys = set()
    for res in pocket.get("residues", []):
        # Key: (chain_id, residue_label)
        pocket_residue_keys.add((res["chain"], res["label"]))

    class PocketSelect(Select):
        def accept_residue(self, residue):
            chain_id = residue.get_parent().get_id()
            res_id = str(residue.get_id()[1])
            if pocket_residue_keys:
                return (chain_id, res_id) in pocket_residue_keys
            else:
                # Fallback: select residues within cutoff of pocket center
                import numpy as np
                center = np.array(pocket["center"])
                for atom in residue:
                    dist = np.linalg.norm(atom.get_vector().get_array() - center)
                    if dist <= cutoff:
                        return True
                return False

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_pocket.pdb")
        os.close(fd)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, PocketSelect())

    logger.info(f"Extracted pocket PDB ({len(pocket_residue_keys)} residues) → {output_path}")
    return output_path
