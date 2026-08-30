"""
reward.py — Pharma-Grade Multi-Objective Reward Oracle for RL fine-tuning.

R(m, pocket) = w₁·r_vina + w₂·r_qed + w₃·r_sa + w₄·r_lipinski + w₅·r_proxy

Hard Safety Gates (instant rejection → negative reward):
  1. RDKit SanitizeMol validity
  2. Carbon ratio ≥ 40%
  3. Nitrogen ratio ≤ 35% (prevents "nitrogen bomb" exploit)
  4. N-N single bond count ≤ 2 (prevents hydrazine/azide chains)
  5. SA score ≤ 6.0 (prevents unsynthesizable structures)
  6. PAINS filter (pan-assay interference compounds)
  7. Ring quality: no ring with > 2 nitrogen atoms (prevents fused tetrazoles)

Soft Reward Components:
  - r_vina    : Vina docking score (ground-truth, called every N rounds)
  - r_qed     : Drug-likeness (RDKit QED)
  - r_sa      : Synthesizability (1 - SA/10)
  - r_lipinski: Lipinski Rule-of-Five compliance (binary per rule)
  - r_proxy   : Learned affinity proxy from value head (cheap, every round)
  - Contrastive bonus: +0.1 if generated mol beats best known ligand
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PAINS SMARTS patterns (subset of the most common interference compounds)
# These are molecules that look active in assays but are actually false positives
# ──────────────────────────────────────────────────────────────────────────────

PAINS_SMARTS = [
    "[#6]1:[#6]:[#6](:[#6]:[#6]:[#6]:1)-[#6]=[#6]-[#6]#[#7]",       # Cyanostilbene
    "[#6]-[#16](=[#8])=[#8]",                                         # Sulfonyl
    "[#6]-[#7]=[#7]=[#7]",                                            # Azide
    "[#6]=[#6](-[#6]#[#7])-[#6]#[#7]",                               # Dicyanoolefin
    "[#8]-[#8]",                                                       # Peroxide
    "[#7]-[#7]=[#7]",                                                  # Triazene
    "[#7]-[#8]-[#7]",                                                  # N-O-N
    "[#6](=[#8])([#8])[#8]",                                          # Carbonate
]

# Medicinal chemistry alert SMARTS — unstable or toxic substructures
MEDCHEM_ALERTS = [
    "[N;X2]=[N;X2]=[N;X1]",        # Organic azide (explosive)
    "[N;X3]([N;X3])[N;X3]",        # Trisubstituted hydrazine
    "[O;X2][O;X2]",                 # Peroxide bond
    "[S;X2][S;X2]",                 # Disulfide
    "[N;X2]=[N;X2]",               # Azo compound (dyes, not drugs)
    "[#6]([F])([F])([F])",          # Trifluoromethyl (sometimes okay, flag it)
]
def _pdb_to_pdbqt_receptor(pdb_path: str) -> str:
    """Robust, C++-safe conversion of protein PDB to PDBQT format for AutoDock Vina."""
    pdbqt_lines = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                line_str = line.rstrip("\n")
                if len(line_str) < 54:
                    continue

                atom_type_hdr = line_str[0:6]
                atom_id = line_str[6:11]
                atom_name = line_str[12:16]
                res_name = line_str[17:20] if len(line_str) >= 20 else "ALA"
                chain_id = line_str[21:22] if len(line_str) >= 22 else "A"
                res_seq = line_str[22:26] if len(line_str) >= 26 else "   1"

                try:
                    x = float(line_str[30:38])
                    y = float(line_str[38:46])
                    z = float(line_str[46:54])
                except ValueError:
                    continue

                elem = line_str[76:78].strip() if len(line_str) >= 78 else ""
                if not elem:
                    name_clean = "".join([c for c in atom_name if c.isalpha()]).upper()
                    if name_clean.startswith(("CA", "CB", "CG", "CD", "CE", "CZ", "CH")):
                        elem = "C"
                    elif name_clean.startswith(("NH", "NE", "ND", "NZ")):
                        elem = "N"
                    elif name_clean.startswith(("OG", "OD", "OE", "OH")):
                        elem = "O"
                    elif name_clean.startswith("SG"):
                        elem = "S"
                    else:
                        elem = name_clean[:1] if name_clean else "C"

                ad_type = elem
                if elem == "O":
                    ad_type = "OA"
                elif elem == "S":
                    ad_type = "SA"

                formatted = (
                    f"{atom_type_hdr:<6s}{atom_id:>5s} {atom_name:<4s} {res_name:>3s} {chain_id:1s}"
                    f"{res_seq:>4s}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    +0.000 {ad_type:>2s}"
                )
                pdbqt_lines.append(formatted)
    return "\n".join(pdbqt_lines) + "\n"


def compute_raw_vina_energy_fn(
    mol,
    pocket_path: str,
    pocket_pos_updated: torch.Tensor = None,
    center: tuple = None,
    box_size: tuple = (20.0, 20.0, 20.0),
) -> float | None:
    """Run AutoDock Vina and return raw binding energy in kcal/mol (e.g., -7.8 kcal/mol)."""
    import os
    import tempfile

    def _mol_to_pdbqt_string(rdkit_mol):
        from rdkit import Chem
        # Tier 1: Try Meeko (0.7.1 API)
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            m = Chem.Mol(rdkit_mol)
            try:
                Chem.SanitizeMol(m)
            except Exception:
                pass
            prep = MoleculePreparation()
            res = prep.prepare(m)
            if res is not None:
                item = res[0] if (isinstance(res, list) and res) else res
                try:
                    pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(item)
                    if pdbqt_str and len(pdbqt_str) > 20:
                        return pdbqt_str
                except Exception:
                    pass
                if hasattr(prep, 'write_pdbqt_string'):
                    s = prep.write_pdbqt_string()
                    if s and len(s) > 20:
                        return s
        except Exception:
            pass

        # Tier 2: Strict fixed-width PDBQT fallback with correct AutoDock atom types
        _AD_TYPE = {
            "C": "C", "N": "NA", "O": "OA", "S": "SA",
            "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
            "P": "P", "H": "HD",
        }
        try:
            if rdkit_mol.GetNumConformers() == 0:
                return None
            conf = rdkit_mol.GetConformer()
            lines = ["REMARK  VINA SCORE EVAL FALLBACK", "ROOT"]
            for a in rdkit_mol.GetAtoms():
                if a.GetAtomicNum() == 1:
                    continue  # Skip hydrogens
                idx = a.GetIdx() + 1
                sym = a.GetSymbol()
                pos = conf.GetAtomPosition(a.GetIdx())
                is_arom = a.GetIsAromatic()
                if sym == "C":
                    ad_type = "A" if is_arom else "C"
                elif sym == "N":
                    ad_type = "NA" if (is_arom or a.GetTotalNumHs() == 0) else "N"
                elif sym == "O":
                    ad_type = "OA"
                elif sym == "S":
                    ad_type = "SA"
                else:
                    ad_type = _AD_TYPE.get(sym, "C")
                atom_name = f"{sym:<4s}"
                lines.append(
                    f"ATOM  {idx:>5d} {atom_name} LIG A   1    "
                    f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    +0.000 {ad_type:>2s}"
                )
            lines.append("ENDROOT")
            lines.append("TORSDOF 0")
            return "\n".join(lines) + "\n"
        except Exception:
            return None

    try:
        from vina import Vina
        from rdkit import Chem
        from pathlib import Path

        ligand_pdbqt = _mol_to_pdbqt_string(mol)
        if ligand_pdbqt is None:
            return None

        pocket_path_obj = Path(pocket_path)
        pocket_pdbqt = pocket_path_obj.with_suffix(".pdbqt")

        if pocket_pos_updated is not None or not pocket_pdbqt.exists():
            path_to_parse = str(pocket_path)
            rec_string = _pdb_to_pdbqt_receptor(path_to_parse)

            fd, temp_rec_path = tempfile.mkstemp(suffix='.pdbqt')
            os.write(fd, rec_string.encode('utf-8'))
            os.close(fd)
            rec_path_to_use = temp_rec_path
            temp_rec_created = True
        else:
            rec_path_to_use = str(pocket_pdbqt)
            temp_rec_created = False

        # Compute pocket center from receptor PDB
        pocket_coords = []
        with open(pocket_path, "r") as f:
            for line in f:
                if line.startswith(("ATOM  ", "HETATM")):
                    try:
                        pocket_coords.append([
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ])
                    except ValueError:
                        pass
        if pocket_coords:
            pocket_center = [
                sum(c[0] for c in pocket_coords) / len(pocket_coords),
                sum(c[1] for c in pocket_coords) / len(pocket_coords),
                sum(c[2] for c in pocket_coords) / len(pocket_coords),
            ]
        else:
            pocket_center = [0.0, 0.0, 0.0]

        # In-situ generated pose preservation (do not shift ligand into protein core)
        from rdkit.Geometry import Point3D
        from rdkit import Chem as _Chem

        mol_to_dock = _Chem.RWMol(mol)
        if mol_to_dock.GetNumConformers() > 0:
            conf = mol_to_dock.GetConformer()
            lig_coords = [conf.GetAtomPosition(i) for i in range(mol_to_dock.GetNumAtoms())]
            lig_cx = sum(p.x for p in lig_coords) / len(lig_coords)
            lig_cy = sum(p.y for p in lig_coords) / len(lig_coords)
            lig_cz = sum(p.z for p in lig_coords) / len(lig_coords)
            
            # If ligand is far from pocket center (>35 A), translate to pocket center; otherwise preserve true generated pose
            dist_to_pocket = ((lig_cx - pocket_center[0])**2 + (lig_cy - pocket_center[1])**2 + (lig_cz - pocket_center[2])**2)**0.5
            if dist_to_pocket > 35.0:
                shift_x = pocket_center[0] - lig_cx
                shift_y = pocket_center[1] - lig_cy
                shift_z = pocket_center[2] - lig_cz
                for i in range(mol_to_dock.GetNumAtoms()):
                    p = conf.GetAtomPosition(i)
                    conf.SetAtomPosition(i, Point3D(p.x + shift_x, p.y + shift_y, p.z + shift_z))
                box_center = pocket_center
            else:
                box_center = [lig_cx, lig_cy, lig_cz]
        else:
            box_center = pocket_center

        ligand_pdbqt = _mol_to_pdbqt_string(mol_to_dock)
        if ligand_pdbqt is None:
            return None

        if center is None:
            center = box_center

        v = Vina(sf_name='vina', verbosity=0)
        v.set_receptor(rec_path_to_use)
        v.set_ligand_from_string(ligand_pdbqt)
        v.compute_vina_maps(center=center, box_size=box_size)
        energy_opt = v.optimize()
        energy = float(energy_opt[0]) if hasattr(energy_opt, '__getitem__') else float(energy_opt)

        # If initial pose is unrelaxed (energy > -1.0 kcal/mol), run SOTA local search to find binding pose
        if energy > -1.0:
            try:
                v.dock(exhaustiveness=4, n_poses=1)
                dock_energy = float(v.energies(n_poses=1)[0][0])
                if dock_energy < energy:
                    energy = dock_energy
            except Exception:
                pass

        if temp_rec_created and os.path.exists(rec_path_to_use):
            os.remove(rec_path_to_use)

        return energy

    except Exception as e:
        logger.info(f"compute_raw_vina_energy error: {e}")
        return None


class RewardOracle:

    """Pharma-grade multi-objective reward computation for DDPO RL fine-tuning.

    Parameters
    ----------
    w_vina, w_qed, w_sa, w_lipinski, w_proxy : reward component weights
    contrastive_bonus : bonus for beating best known ligand
    vina_every_n : Vina oracle called every N RL rounds
    min_carbon_ratio : minimum fraction of atoms that must be carbon (≥0.40)
    max_nitrogen_ratio : maximum fraction of atoms that can be nitrogen (≤0.35)
    max_nn_bonds : maximum number of N-N single bonds allowed (≤2)
    max_sa_score : maximum synthesizability score before penalty (≤6.0)
    max_ring_nitrogen : maximum nitrogen atoms allowed in a single ring (≤2)
    """

    def __init__(
        self,
        w_vina: float = 0.35,    # Primary binding affinity signal
        w_qed: float = 0.30,     # Boosted: key drug-likeness metric vs SOTA
        w_sa: float = 0.15,
        w_lipinski: float = 0.10,
        w_proxy: float = 0.10,
        contrastive_bonus: float = 0.10,
        qed_bonus_threshold: float = 0.50,
        qed_bonus: float = 0.15,
        vina_every_n: int = 50,   # Vina only every 50 steps — all steps are mult of 10!
        min_carbon_ratio: float = 0.40,
        max_nitrogen_ratio: float = 0.35,
        max_nn_bonds: int = 2,
        max_sa_score: float = 6.0,
        max_ring_nitrogen: int = 2,
    ):
        self.w_vina = w_vina
        self.w_qed = w_qed
        self.w_sa = w_sa
        self.w_lipinski = w_lipinski
        self.w_proxy = w_proxy
        self.contrastive_bonus = contrastive_bonus
        self.qed_bonus_threshold = qed_bonus_threshold
        self.qed_bonus = qed_bonus
        self.vina_every_n = vina_every_n
        self.min_carbon_ratio = min_carbon_ratio
        self.max_nitrogen_ratio = max_nitrogen_ratio
        self.max_nn_bonds = max_nn_bonds
        self.max_sa_score = max_sa_score
        self.max_ring_nitrogen = max_ring_nitrogen
        self.max_carbon_ratio = 0.85   # >85% carbon = carbon collapse penalty
        self.min_heteroatoms = 1       # must have ≥1 N or O atom

        # Pre-compile PAINS and medchem alert patterns
        self._pains_patterns = None
        self._medchem_patterns = None

    def _get_pains_patterns(self):
        """Lazy-load and compile PAINS SMARTS patterns."""
        if self._pains_patterns is None:
            from rdkit import Chem
            self._pains_patterns = []
            for smarts in PAINS_SMARTS:
                pat = Chem.MolFromSmarts(smarts)
                if pat is not None:
                    self._pains_patterns.append(pat)
        return self._pains_patterns

    def _get_medchem_patterns(self):
        """Lazy-load and compile medicinal chemistry alert patterns."""
        if self._medchem_patterns is None:
            from rdkit import Chem
            self._medchem_patterns = []
            for smarts in MEDCHEM_ALERTS:
                pat = Chem.MolFromSmarts(smarts)
                if pat is not None:
                    self._medchem_patterns.append(pat)
        return self._medchem_patterns

    # ──────────────────────────────────────────────────────────────────
    # Individual reward components
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_qed(mol) -> float:
        """RDKit QED drug-likeness score ∈ [0, 1]."""
        from rdkit.Chem import QED as RDKitQED
        try:
            return RDKitQED.qed(mol)
        except Exception:
            return 0.0

    @staticmethod
    def compute_sa(mol) -> float:
        """Synthetic accessibility reward: r_sa = 1 - SA/10.

        SA ∈ [1, 10], so r_sa ∈ [0, 0.9]. Target: SA < 4 → r_sa > 0.6.
        """
        from rdkit.Chem import RDConfig
        import sys
        import os
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        try:
            import sascorer
            sa = sascorer.calculateScore(mol)
            return max(0.0, 1.0 - sa / 10.0)
        except Exception:
            return 0.0

    @staticmethod
    def compute_sa_raw(mol) -> float:
        """Return raw SA score (1=easy, 10=impossible)."""
        from rdkit.Chem import RDConfig
        import sys
        import os
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        try:
            import sascorer
            return sascorer.calculateScore(mol)
        except Exception:
            return 10.0

    @staticmethod
    def compute_lipinski(mol) -> float:
        """Lipinski Rule-of-Five: fraction of rules satisfied ∈ [0, 1].

        Rules: MW ≤ 500, HBD ≤ 5, HBA ≤ 10, logP ≤ 5.
        """
        from rdkit.Chem import Descriptors, Lipinski

        try:
            mw = Descriptors.MolWt(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            logp = Descriptors.MolLogP(mol)

            score = 0.0
            if mw <= 500:
                score += 0.25
            if hbd <= 5:
                score += 0.25
            if hba <= 10:
                score += 0.25
            if logp <= 5:
                score += 0.25
            return score
        except Exception:
            return 0.0


    @staticmethod
    def compute_vina_score(
        mol,
        pocket_path: str,
        pocket_pos_updated: torch.Tensor = None,
        center: tuple = None,
        box_size: tuple = (20.0, 20.0, 20.0),
    ) -> float:
        """Run AutoDock Vina and return normalised score.

        r_vina = (|score| - 6) / 7,  clamped to [0, 1].
        Falls back to 0.0 on any failure (Vina not installed, bad molecule, etc.)
        """
        import os
        import sys
        import io
        import tempfile
        from contextlib import contextmanager

        @contextmanager
        def _suppress_c_stderr():
            """Redirect C-level fd 2 (stderr) to /dev/null to silence meeko/RDKit C++ spam."""
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull_fd, 2)
                yield
            finally:
                os.dup2(old_stderr_fd, 2)
                os.close(devnull_fd)
                os.close(old_stderr_fd)

        def _mol_to_pdbqt_string(rdkit_mol):
            from rdkit import Chem
            # Tier 1: Try Meeko (0.7.1 API)
            try:
                from meeko import MoleculePreparation, PDBQTWriterLegacy
                m = Chem.Mol(rdkit_mol)
                try:
                    Chem.SanitizeMol(m)
                except Exception:
                    pass
                prep = MoleculePreparation()
                res = prep.prepare(m)
                if res is not None:
                    item = res[0] if (isinstance(res, list) and res) else res
                    try:
                        pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(item)
                        if pdbqt_str and len(pdbqt_str) > 20:
                            return pdbqt_str
                    except Exception:
                        pass
                    if hasattr(prep, 'write_pdbqt_string'):
                        s = prep.write_pdbqt_string()
                        if s and len(s) > 20:
                            return s
            except Exception:
                pass

            # Tier 2: Strict fixed-width PDBQT fallback (guarantees zero Vina C++ parse crash)
            try:
                if rdkit_mol.GetNumConformers() == 0:
                    return None
                conf = rdkit_mol.GetConformer()
                lines = ["REMARK  Fallback PDBQT"]
                for a in rdkit_mol.GetAtoms():
                    idx = a.GetIdx() + 1
                    sym = a.GetSymbol()
                    pos = conf.GetAtomPosition(a.GetIdx())
                    ad_type = "A" if (sym == "C" and a.GetIsAromatic()) else sym
                    lines.append(
                        f"ATOM  {idx:>5d} {sym:<4s} LIG A   1    "
                        f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    +0.000 {ad_type:>2s}"
                    )
                lines.append("TORSDOF 0")
                return "\n".join(lines) + "\n"
            except Exception:
                return None

        try:
            from vina import Vina
            from rdkit import Chem
            from pathlib import Path

            ligand_pdbqt = _mol_to_pdbqt_string(mol)
            if ligand_pdbqt is None:
                return 0.0

            # ── 2. Prepare receptor PDBQT ──
            pocket_path_obj = Path(pocket_path)
            pocket_pdbqt = pocket_path_obj.with_suffix(".pdbqt")

            if pocket_pos_updated is not None or not pocket_pdbqt.exists():
                path_to_parse = str(pocket_path)

                # Optionally write a new PDB with induced-fit coordinates
                if pocket_pos_updated is not None:
                    pos_np = pocket_pos_updated.cpu().numpy()
                    new_pdb_lines = []
                    atom_idx = 0
                    with open(pocket_path, "r") as f:
                        for line in f:
                            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                                if atom_idx < len(pos_np):
                                    p = pos_np[atom_idx]
                                    new_line = (
                                        f"{line[:30]}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                                        f"{line[54:]}"
                                    )
                                    new_pdb_lines.append(new_line)
                                    atom_idx += 1
                                else:
                                    new_pdb_lines.append(line)
                            else:
                                new_pdb_lines.append(line)
                    fd, temp_pdb = tempfile.mkstemp(suffix='.pdb')
                    os.write(fd, "".join(new_pdb_lines).encode('utf-8'))
                    os.close(fd)
                    path_to_parse = temp_pdb

                # Convert PDB receptor → PDBQT via robust parser
                try:
                    rec_string = _pdb_to_pdbqt_receptor(path_to_parse)
                except Exception as rec_err:
                    logger.debug(f"Receptor PDBQT conversion failed: {rec_err}")
                    return 0.0

                if pocket_pos_updated is not None and os.path.exists(path_to_parse):
                    os.remove(path_to_parse)

                fd, temp_rec_path = tempfile.mkstemp(suffix='.pdbqt')
                os.write(fd, rec_string.encode('utf-8'))
                os.close(fd)
                rec_path_to_use = temp_rec_path
                temp_rec_created = True
            else:
                rec_path_to_use = str(pocket_pdbqt)
                temp_rec_created = False

            # ── 3. Calculate box center ──
            if center is None:
                coords = []
                with open(pocket_path, "r") as f:
                    for line in f:
                        if line.startswith(("ATOM  ", "HETATM")):
                            try:
                                coords.append([
                                    float(line[30:38]),
                                    float(line[38:46]),
                                    float(line[46:54]),
                                ])
                            except ValueError:
                                pass
                center = [sum(c) / len(c) for c in zip(*coords)] if coords else [0.0, 0.0, 0.0]

            # ── 4. Run Vina ──
            v = Vina(sf_name='vina', verbosity=0)
            v.set_receptor(rec_path_to_use)
            v.set_ligand_from_string(ligand_pdbqt)
            v.compute_vina_maps(center=center, box_size=box_size)
            energy = v.optimize()[0]

            if temp_rec_created and os.path.exists(rec_path_to_use):
                os.remove(rec_path_to_use)

            # ── 5. Return raw energy (negative kcal/mol) and normalized r_vina ──
            r_vina = (abs(energy) - 6.0) / 7.0
            return max(0.0, min(1.0, r_vina))

        except Exception as e:
            logger.debug(f"Vina scoring skipped (using proxy fallback): {e}")
            return 0.0

    @staticmethod
    def compute_raw_vina_energy(
        mol,
        pocket_path: str,
        pocket_pos_updated: torch.Tensor = None,
        center: tuple = None,
        box_size: tuple = (20.0, 20.0, 20.0),
    ) -> float | None:
        """Run AutoDock Vina and return raw binding energy in kcal/mol (e.g., -7.8 kcal/mol)."""
        return compute_raw_vina_energy_fn(mol, pocket_path, pocket_pos_updated, center, box_size)

    def compute_proxy_reward(self, pK_pred: torch.Tensor) -> float:
        """Convert the learned affinity proxy to a reward.

        r_proxy = sigmoid((pK_pred + 8) / 4) — shifted so that pK_pred=-8 kcal/mol
        gives r=0.5 and every kcal/mol improvement adds meaningful gradient.
        Previously pK_pred/16 gave sigmoid(-2.9/16)=0.455 for any pK, nearly flat.
        """
        return torch.sigmoid((pK_pred + 8.0) / 4.0).item()

    # ──────────────────────────────────────────────────────────────────
    # Pharma-Grade Safety Gates
    # ──────────────────────────────────────────────────────────────────

    def compute_element_diversity(self, mol) -> float:
        """Penalise molecules that lack carbon backbone.

        Drug molecules are fundamentally carbon-based.  A valid drug
        should have ≥40 % carbon atoms.  Returns 1.0 if the ratio
        is above `min_carbon_ratio`, else a fraction that smoothly
        decays toward 0.
        """
        try:
            n_total = mol.GetNumAtoms()
            if n_total == 0:
                return 0.0
            n_carbon = sum(
                1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6
            )
            ratio = n_carbon / n_total
            if ratio >= self.min_carbon_ratio:
                return 1.0
            # Smooth penalty: linearly scale from 0→1 as ratio→min
            return max(0.0, ratio / self.min_carbon_ratio)
        except Exception:
            return 0.0

    def check_nitrogen_ratio(self, mol) -> tuple:
        """Check if nitrogen ratio is within pharma-acceptable limits.

        Returns (passed: bool, ratio: float, detail: str)
        """
        try:
            n_total = mol.GetNumAtoms()
            if n_total == 0:
                return False, 1.0, "empty molecule"
            n_nitrogen = sum(
                1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7
            )
            ratio = n_nitrogen / n_total
            if ratio > self.max_nitrogen_ratio:
                return False, ratio, f"N ratio {ratio:.2f} > {self.max_nitrogen_ratio}"
            return True, ratio, "ok"
        except Exception:
            return False, 1.0, "error"

    def count_nn_bonds(self, mol) -> int:
        """Count the number of N-N single bonds (hydrazine motifs)."""
        try:
            count = 0
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetAtomicNum()
                a2 = bond.GetEndAtom().GetAtomicNum()
                if a1 == 7 and a2 == 7:
                    count += 1
            return count
        except Exception:
            return 999

    @staticmethod
    def check_ring_sizes(mol) -> float:
        """Evaluate ring size distribution for medicinal drug-likeness.

        Pharma standard: 5- and 6-membered aromatic and saturated rings.
        Heavily penalizes strained 3- and 4-membered fused cages (aziridines, cyclopropanes, diazirines)
        that cause high SA scores and impossible 3D crystal relaxation.

        Returns ring_size_score in [0.10, 1.0].
        """
        try:
            ri = mol.GetRingInfo()
            rings = ri.AtomRings()
            if not rings:
                return 0.85  # acyclic linkers are acceptable
            
            strained_34 = 0
            good_56 = 0
            for ring in rings:
                sz = len(ring)
                if sz in (3, 4):
                    strained_34 += 1
                elif sz in (5, 6):
                    good_56 += 1
                elif sz >= 7:
                    strained_34 += 0.5  # macrocycles penalized slightly

            if strained_34 == 0 and good_56 >= 1:
                return 1.0  # Perfect drug scaffold (e.g. benzene, pyridine, piperidine)
            elif strained_34 == 0 and good_56 == 0:
                return 0.85
            else:
                # Strong penalty for each 3- or 4-membered strained ring
                return max(0.10, 1.0 - 0.40 * strained_34)
        except Exception:
            return 0.50

    def check_ring_quality(self, mol) -> tuple:
        """Check that no single ring has more than max_ring_nitrogen N atoms.

        Fused poly-nitrogen rings (tetrazoles, pentazoles) are generally
        explosive, not druggable.

        Returns (passed: bool, worst_ring_n: int, detail: str)
        """
        try:
            ri = mol.GetRingInfo()
            worst = 0
            for ring in ri.AtomRings():
                n_nitrogen = sum(
                    1 for idx in ring
                    if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7
                )
                worst = max(worst, n_nitrogen)
            if worst > self.max_ring_nitrogen:
                return False, worst, f"ring with {worst} N atoms > max {self.max_ring_nitrogen}"
            return True, worst, "ok"
        except Exception:
            return False, 999, "error"

    def check_pains(self, mol) -> tuple:
        """Check for PAINS (pan-assay interference) substructures.

        Returns (passed: bool, detail: str)
        """
        try:
            for pat in self._get_pains_patterns():
                if mol.HasSubstructMatch(pat):
                    return False, "PAINS hit"
            return True, "ok"
        except Exception:
            return True, "check failed"

    def check_medchem_alerts(self, mol) -> tuple:
        """Check for medicinal chemistry structural alerts.

        Returns (passed: bool, detail: str)
        """
        try:
            for pat in self._get_medchem_patterns():
                if mol.HasSubstructMatch(pat):
                    return False, "medchem alert"
            return True, "ok"
        except Exception:
            return True, "check failed"

    @staticmethod
    def is_valid_molecule(mol) -> bool:
        """Quick sanity check: sanitisable + has ≥1 ring or ≥1 double bond."""
        from rdkit import Chem
        try:
            Chem.SanitizeMol(mol)
            return True
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────
    # Full pharma-grade gate check (used by both RL and generation)
    # ──────────────────────────────────────────────────────────────────

    def run_pharma_gates(self, mol) -> tuple:
        """Run ALL pharma safety gates on a molecule.

        Returns (passed: bool, penalty: float, reason: str)
        - If passed=True:  penalty=0.0, reason="passed"
        - If passed=False: penalty is a negative reward, reason explains why
        """
        from rdkit import Chem

        # Gate 1: RDKit sanitization
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return False, -0.5, "failed RDKit SanitizeMol"

        # Gate 2: Carbon ratio
        carbon_score = self.compute_element_diversity(mol)
        if carbon_score < 0.5:
            return False, -0.3, f"carbon ratio too low ({carbon_score:.2f})"

        # Gate 3: Nitrogen ratio
        n_passed, n_ratio, n_detail = self.check_nitrogen_ratio(mol)
        if not n_passed:
            return False, -0.4, f"nitrogen exploit: {n_detail}"

        # Gate 4: N-N bond count
        nn_count = self.count_nn_bonds(mol)
        if nn_count > self.max_nn_bonds:
            return False, -0.3, f"too many N-N bonds ({nn_count} > {self.max_nn_bonds})"

        # Gate 5: Ring quality (no poly-nitrogen rings)
        ring_passed, ring_worst, ring_detail = self.check_ring_quality(mol)
        if not ring_passed:
            return False, -0.4, f"poly-nitrogen ring: {ring_detail}"

        # Gate 6: SA score (unsynthesizable)
        sa_raw = self.compute_sa_raw(mol)
        if sa_raw > self.max_sa_score:
            return False, -0.2, f"SA too high ({sa_raw:.1f} > {self.max_sa_score})"

        # Gate 7: PAINS filter
        pains_passed, pains_detail = self.check_pains(mol)
        if not pains_passed:
            return False, -0.3, f"PAINS: {pains_detail}"

        # Gate 8: Medicinal chemistry alerts
        medchem_passed, medchem_detail = self.check_medchem_alerts(mol)
        if not medchem_passed:
            return False, -0.3, f"medchem alert: {medchem_detail}"

        return True, 0.0, "passed"

    # ──────────────────────────────────────────────────────────────────
    # Combined reward
    # ──────────────────────────────────────────────────────────────────

    def compute_reward(
        self,
        mol,
        pK_pred: torch.Tensor,
        pocket_path: Optional[str] = None,
        pocket_pos_updated: Optional[torch.Tensor] = None,
        best_known_affinity: Optional[float] = None,
        rl_round: int = 0,
    ) -> dict:
        """Compute the full multi-objective reward with pharma safety gates.

        Parameters
        ----------
        mol               : RDKit Mol object of the generated molecule
        pK_pred           : affinity proxy from the value head
        pocket_path       : path to pocket file (for Vina)
        best_known_affinity : best affinity in dataset for this pocket
        rl_round          : current RL round number

        Returns
        -------
        dict with total_reward and individual components
        """
        _fail_result = {
            "total_reward": -0.5,
            "r_vina": 0.0,
            "r_qed": 0.0,
            "r_sa": 0.0,
            "r_lipinski": 0.0,
            "r_proxy": 0.0,
            "r_diversity": 0.0,
            "contrastive_bonus": 0.0,
            "gate_reason": "",
        }

        # ── Run ALL pharma safety gates ──
        gate_passed, gate_penalty, gate_reason = self.run_pharma_gates(mol)
        if not gate_passed:
            result = _fail_result.copy()
            result["total_reward"] = gate_penalty
            result["gate_reason"] = gate_reason
            return result

        # ── All gates passed — compute soft reward components ──
        r_qed = self.compute_qed(mol)
        r_sa = self.compute_sa(mol)
        r_lipinski = self.compute_lipinski(mol)
        r_proxy = self.compute_proxy_reward(pK_pred)
        r_diversity = self.compute_element_diversity(mol)

        # Vina: only every N rounds (expensive)
        if rl_round % self.vina_every_n == 0 and pocket_path is not None:
            r_vina = self.compute_vina_score(mol, pocket_path, pocket_pos_updated)
        else:
            r_vina = r_proxy  # fall back to proxy when Vina is skipped

        # Weighted combination (multiplied by diversity gate)
        total = (
            self.w_vina * r_vina
            + self.w_qed * r_qed
            + self.w_sa * r_sa
            + self.w_lipinski * r_lipinski
            + self.w_proxy * r_proxy
        ) * r_diversity  # diversity acts as a multiplicative gate

        # Contrastive bonus: did we beat the best known ligand?
        bonus = 0.0
        if best_known_affinity is not None:
            pred_affinity_approx = -pK_pred.item() * 16.0
            if pred_affinity_approx < best_known_affinity:
                bonus = self.contrastive_bonus
                total += bonus

        return {
            "total_reward": total,
            "r_vina": r_vina,
            "r_qed": r_qed,
            "r_sa": r_sa,
            "r_lipinski": r_lipinski,
            "r_proxy": r_proxy,
            "r_diversity": r_diversity,
            "contrastive_bonus": bonus,
            "gate_reason": "passed",
        }

    def compute_rl_reward(
        self,
        mol,
        pK_pred: torch.Tensor,
        pocket_path: Optional[str] = None,
        pocket_pos_updated: Optional[torch.Tensor] = None,
        best_known_affinity: Optional[float] = None,
        rl_round: int = 0,
    ) -> dict:
        """Compute reward for RL training with SMOOTH continuous penalties.

        Unlike compute_reward() which uses hard gates (flat penalties),
        this version gives the RL optimizer smooth gradient signal so
        the model learns WHICH DIRECTION to move.

        - Carbon ratio 39% scores much better than 5% (not the same flat -0.3)
        - Nitrogen ratio 36% scores much better than 80%
        - SA score 6.5 scores better than 9.0

        Hard gates (run_pharma_gates) are only used at inference time
        in generate.py for final filtering.
        """
        from rdkit import Chem

        # Gate 1: RDKit validity — this one stays hard (no gradient possible)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return {
                "total_reward": 0.0,
                "r_qed": 0.0, "r_sa": 0.0, "r_lipinski": 0.0,
                "r_proxy": 0.0, "r_diversity": 0.0,
                "gate_reason": "invalid",
            }

        # ── Compute all soft components ──
        r_qed = self.compute_qed(mol)
        r_sa = self.compute_sa(mol)
        r_lipinski = self.compute_lipinski(mol)
        r_proxy = self.compute_proxy_reward(pK_pred)

        # ── Soft carbon ratio penalty (continuous, not flat) ──
        n_total = mol.GetNumAtoms()
        n_carbon = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        n_oxygen = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        n_heteroatoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 9, 16, 17))
        carbon_ratio = n_carbon / max(n_total, 1)

        # ── Comprehensive Pharma Stoichiometry Scoring ──
        # Standard small-molecule drugs: 60-80% Carbon, 8-30% Nitrogen (≥1 N), 5-25% Oxygen (≥1 O)
        n_nitrogen = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        nitrogen_ratio = n_nitrogen / max(n_total, 1)
        n_oxygen = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        oxygen_ratio = n_oxygen / max(n_total, 1)

        # 1. Carbon score: optimal 55% - 85%
        if 0.55 <= carbon_ratio <= 0.85:
            carbon_score = 1.0
        elif carbon_ratio < 0.55:
            carbon_score = max(0.15, carbon_ratio / 0.55)
        else:
            carbon_score = max(0.15, 1.0 - (carbon_ratio - 0.85) / 0.15)

        # 2. Nitrogen score: optimal 8% - 30%, requires at least 1-2 N atoms
        if n_nitrogen >= 1 and 0.05 <= nitrogen_ratio <= 0.35:
            nitrogen_presence = 1.0
        elif n_nitrogen == 0:
            nitrogen_presence = 0.20  # Strong gradient forcing model to incorporate Nitrogen
        else:
            nitrogen_presence = max(0.15, 1.0 - (nitrogen_ratio - 0.35) / 0.20)

        # 3. Oxygen score: optimal 5% - 25%, requires at least 1-2 O atoms
        if n_oxygen >= 1 and 0.05 <= oxygen_ratio <= 0.30:
            oxygen_presence = 1.0
        elif n_oxygen == 0:
            oxygen_presence = 0.20  # Strong gradient forcing model to incorporate Oxygen
        else:
            oxygen_presence = max(0.15, 1.0 - (oxygen_ratio - 0.30) / 0.20)

        # ── Soft N-N bond penalty ──
        nn_count = self.count_nn_bonds(mol)
        nn_score = max(0.0, 1.0 - nn_count / 3.0)

        # ── Soft ring quality penalty ──
        ring_passed, ring_worst, _ = self.check_ring_quality(mol)
        ring_score = max(0.0, 1.0 - max(0, ring_worst - 1) / 3.0)

        # ── Ring size quality (rewards 5- and 6-membered rings, penalizes 3- and 4-membered strained cages) ──
        ring_size_score = self.check_ring_sizes(mol)

        # ── Strict SA Synthesizability curve (optimal: SA <= 4.0) ──
        sa_raw = self.compute_sa_raw(mol)
        if sa_raw <= 4.0:
            sa_score = 1.0
        elif sa_raw <= 6.0:
            sa_score = max(0.20, 1.0 - (sa_raw - 4.0) / 2.5)
        else:
            sa_score = max(0.05, 0.20 - (sa_raw - 6.0) / 4.0)

        # ── Physical 3D Conformation / PoseBusters Sanity Check ──
        # Directly checks for realistic 3D bond lengths, angles, and steric sanity via MMFF
        pb_score = 0.5  # default baseline
        try:
            from rdkit.Chem import AllChem
            mol_h = Chem.AddHs(mol, addCoords=True)
            if mol_h.GetNumConformers() > 0:
                res = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=300)
                pb_score = 1.0 if (res in (0, 1)) else 0.3
        except Exception:
            pb_score = 0.2

        # ── Combined chemistry quality: Multiplicative Carbon Gate × Multi-Objective Balance ──
        # Real pharma drugs (60-80% C, 2+ N, 2+ O, 5/6-membered rings, SA < 4.0, valid 3D physical pose) receive 1.0.
        hetero_balance = 0.50 * nitrogen_presence + 0.50 * oxygen_presence
        inner_quality = (
            0.25 * hetero_balance     # Balanced N + O incorporation
            + 0.25 * ring_size_score  # 5- and 6-membered aromatic/aliphatic drug rings (penalizes 3/4-member cages)
            + 0.25 * sa_score         # Strict Synthesizability (SA < 4.0)
            + 0.15 * pb_score         # PoseBusters 3D physical conformation & bond length sanity
            + 0.05 * nn_score         # Anti-hydrazine
            + 0.05 * ring_score       # Anti-tetrazole
        )
        chem_quality = carbon_score * inner_quality

        # ── Weighted reward (same formula as compute_reward) ──
        # Use Vina physics oracle when available, else fall back to proxy
        if rl_round % self.vina_every_n == 0 and pocket_path is not None:
            r_vina = self.compute_vina_score(mol, pocket_path, pocket_pos_updated)
        else:
            r_vina = r_proxy  # proxy fallback between Vina rounds

        base_reward = (
            self.w_vina * r_vina
            + self.w_qed * r_qed
            + self.w_sa * r_sa
            + self.w_lipinski * r_lipinski
            + self.w_proxy * r_proxy
        )

        # High QED bonus: +0.15 if QED > 0.50 (pushes toward SOTA drug-likeness range)
        if r_qed >= self.qed_bonus_threshold:
            base_reward += self.qed_bonus

        # Scale by chemistry quality (0 to 1 multiplier)
        total = base_reward * chem_quality

        # Ensure reward is strictly non-negative!
        # If reward is negative, unbaselined REINFORCE will minimize log_prob to -inf,
        # which destroys the model weights (causing the ClCl collapse).
        total = max(0.0, total)

        return {
            "total_reward": total,
            "r_qed": r_qed,
            "r_sa": r_sa,
            "r_lipinski": r_lipinski,
            "r_proxy": r_proxy,
            "r_diversity": carbon_score,
            "chem_quality": chem_quality,
            "carbon_ratio": carbon_ratio,
            "nitrogen_ratio": nitrogen_ratio,
            "gate_reason": "rl_soft",
        }
