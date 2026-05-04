"""
reward.py — Multi-Objective Reward Oracle for RL fine-tuning.

R(m, pocket) = w₁·r_vina + w₂·r_qed + w₃·r_sa + w₄·r_lipinski + w₅·r_proxy

Components:
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


class RewardOracle:
    """Multi-objective reward computation for DDPO RL fine-tuning.

    Parameters
    ----------
    w_vina, w_qed, w_sa, w_lipinski, w_proxy : reward component weights
    contrastive_bonus : bonus for beating best known ligand
    vina_every_n : Vina oracle called every N RL rounds
    """

    def __init__(
        self,
        w_vina: float = 0.40,
        w_qed: float = 0.20,
        w_sa: float = 0.15,
        w_lipinski: float = 0.10,
        w_proxy: float = 0.15,
        contrastive_bonus: float = 0.10,
        vina_every_n: int = 10,
    ):
        self.w_vina = w_vina
        self.w_qed = w_qed
        self.w_sa = w_sa
        self.w_lipinski = w_lipinski
        self.w_proxy = w_proxy
        self.contrastive_bonus = contrastive_bonus
        self.vina_every_n = vina_every_n

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
        center: tuple = None,
        box_size: tuple = (20.0, 20.0, 20.0),
    ) -> float:
        """Run AutoDock Vina and return normalised score.

        r_vina = (|score| - 6) / 7,  clamped to [0, 1].

        NOTE: This requires Vina to be installed on the server.
        For now, this is a placeholder that should be connected to your
        local Vina installation.
        """
        logger.warning(
            "Vina scoring not yet connected — returning 0.0. "
            "Connect your server's Vina wrapper here."
        )
        return 0.0

    def compute_proxy_reward(self, pK_pred: torch.Tensor) -> float:
        """Convert the learned affinity proxy to a reward.

        r_proxy = sigmoid(pK_pred / 16) — aligned to Vina during pretraining.
        """
        return torch.sigmoid(pK_pred / 16.0).item()

    # ──────────────────────────────────────────────────────────────────
    # Combined reward
    # ──────────────────────────────────────────────────────────────────

    def compute_reward(
        self,
        mol,
        pK_pred: torch.Tensor,
        pocket_path: Optional[str] = None,
        best_known_affinity: Optional[float] = None,
        rl_round: int = 0,
    ) -> dict:
        """Compute the full multi-objective reward.

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
        r_qed = self.compute_qed(mol)
        r_sa = self.compute_sa(mol)
        r_lipinski = self.compute_lipinski(mol)
        r_proxy = self.compute_proxy_reward(pK_pred)

        # Vina: only every N rounds (expensive)
        if rl_round % self.vina_every_n == 0 and pocket_path is not None:
            r_vina = self.compute_vina_score(mol, pocket_path)
        else:
            r_vina = r_proxy  # fall back to proxy when Vina is skipped

        # Weighted combination
        total = (
            self.w_vina * r_vina
            + self.w_qed * r_qed
            + self.w_sa * r_sa
            + self.w_lipinski * r_lipinski
            + self.w_proxy * r_proxy
        )

        # Contrastive bonus: did we beat the best known ligand?
        bonus = 0.0
        if best_known_affinity is not None:
            # If proxy predicts stronger binding than best known
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
            "contrastive_bonus": bonus,
        }
