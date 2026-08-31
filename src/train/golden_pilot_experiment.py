"""
src/train/golden_pilot_experiment.py — Golden-Baseline-Anchored SDE Flow-GRPO Pilot.

Initializes from the verified production model (checkpoints/rl_final.pt) and executes:
  - G0: Golden Baseline (frozen rl_final.pt, official 50-step deterministic ODE protocol)
  - G1: SDE Flow-GRPO (initialized from rl_final.pt, G=4, K=20, lr=5e-6, beta=0.01)
  - G2: Historical Heuristic (matched control initialized from rl_final.pt)

Evaluates on fixed benchmark targets at Step 0, 10, 25, 50 using the official benchmark protocol.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.featurizer import PocketFeaturizer
from src.model.flow_matching import FlowMatching
from src.model.pocket_encoder import PocketEncoder
from src.model.egnn import SE3EGNN
from src.model.reward import RewardOracle
from src.train.sde_likelihood import (
    compute_group_advantages,
    compute_stabilized_ratio,
    evaluate_trajectory_probability,
)
from src.train.rl_finetune import train_grpo_step, compute_velocity_equivariance_diagnostic
from generate import coords_to_rdkit_mol
from evaluate import compute_pb_validity, compute_tanimoto_diversity

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger("golden_pilot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_file_sha256(filepath: Path | str) -> str:
    """Computes the SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Official 50-Step Evaluation Protocol
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_golden_protocol(
    model: FlowMatching,
    pockets: List[Dict[str, Any]],
    reward_oracle: RewardOracle,
    device: str = "cpu",
    mols_per_pocket: int = 10,
    num_steps: int = 50,
) -> Dict[str, Any]:
    """
    Evaluates a model under the official production 50-step deterministic ODE protocol:
      - 50-step Euler integration
      - temperature = 0.8
      - element_bias = [0.0, 0.05, 0.40, 0.0, 0.0, 0.0]
      - official PoseBusters (compute_pb_validity with forcefield relaxation)
      - official RDKFingerprint Tanimoto diversity
    """
    model.eval()
    element_bias = torch.tensor([0.0, 0.05, 0.40, 0.0, 0.0, 0.0], device=device)

    all_mols = []
    valid_mols = []
    smiles_list = []
    qeds = []
    sas = []
    lipinskis = []
    pb_valids = []
    rewards = []
    gen_times = []

    for p in pockets:
        pos = p["pos"].to(device)
        feat = p["feat"].to(device)
        p_com = pos.mean(dim=0).cpu().numpy()
        p_path = p.get("path", "dummy.pdb")

        for _ in range(mols_per_pocket):
            t0 = time.perf_counter()
            res = model.sample(
                pocket_pos=pos,
                pocket_feat=feat,
                num_atoms=22,
                temperature=0.8,
                num_steps=num_steps,
                element_bias=element_bias,
                stochastic=False,
            )
            elapsed = time.perf_counter() - t0
            gen_times.append(elapsed)

            pos_np = res["pos"].cpu().numpy()
            types_np = res["atom_types"].cpu().numpy()
            pos_pocket = pos_np + p_com
            mol, sanitized = coords_to_rdkit_mol(pos_pocket, types_np)

            all_mols.append(mol)
            if sanitized and mol is not None:
                valid_mols.append(mol)
                smi = Chem.MolToSmiles(mol)
                smiles_list.append(smi)

                # QED & SA
                qeds.append(QED.qed(mol))
                sas.append(RewardOracle.compute_sa_raw(mol))

                # Lipinski Rule-of-5
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                lip = (mw <= 500) and (logp <= 5.0) and (hbd <= 5) and (hba <= 10)
                lipinskis.append(1.0 if lip else 0.0)

                # Official PoseBusters PB-Valid
                pb = compute_pb_validity(mol)
                pb_valids.append(1.0 if pb else 0.0)

                # Reward Oracle score
                rd = reward_oracle.compute_rl_reward(
                    mol=mol,
                    pK_pred=res["pK_pred"],
                    pocket_path=p_path,
                    pocket_pos_updated=res.get("pocket_pos_updated"),
                    rl_round=0,
                )
                rewards.append(float(rd["total_reward"]))
            else:
                rewards.append(0.0)

    n_total = len(all_mols)
    n_valid = len(valid_mols)
    validity = n_valid / max(n_total, 1)
    diversity = compute_tanimoto_diversity(smiles_list)
    unique_rate = len(set(smiles_list)) / max(n_valid, 1)

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "validity_rate": validity,
        "pb_validity_rate": float(np.mean(pb_valids)) if pb_valids else 0.0,
        "qed_mean": float(np.mean(qeds)) if qeds else 0.0,
        "qed_median": float(np.median(qeds)) if qeds else 0.0,
        "qed_std": float(np.std(qeds)) if qeds else 0.0,
        "sa_mean": float(np.mean(sas)) if sas else 0.0,
        "sa_std": float(np.std(sas)) if sas else 0.0,
        "lipinski_rate": float(np.mean(lipinskis)) if lipinskis else 0.0,
        "internal_diversity": diversity,
        "unique_smiles_rate": unique_rate,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_median": float(np.median(rewards)) if rewards else 0.0,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        "avg_gen_time_s": float(np.mean(gen_times)) if gen_times else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Historical Heuristic Single Step for G2
# ──────────────────────────────────────────────────────────────────────────────

def train_heuristic_step_g2(
    model: FlowMatching,
    model_ref: FlowMatching,
    optimizer: torch.optim.Optimizer,
    pocket_pos: torch.Tensor,
    pocket_feat: torch.Tensor,
    reward_oracle: RewardOracle,
    pocket_path: str,
    G: int = 4,
    K: int = 20,
    beta: float = 0.01,
    device: str = "cpu",
    max_grad_norm: float = 1.0,
) -> Dict[str, Any]:
    """Single step of the historical B1/G2 heuristic objective."""
    device_obj = torch.device(device)
    pocket_pos = pocket_pos.to(device_obj)
    pocket_feat = pocket_feat.to(device_obj)

    model.eval()
    with torch.no_grad():
        pocket_pos_centered = pocket_pos - pocket_pos.mean(dim=0, keepdim=True)
        h_P = model.pocket_encoder(pocket_pos_centered, pocket_feat)["h_P"]

        candidates = []
        rewards = []
        for _ in range(G):
            gen = model.sample(pocket_pos, pocket_feat, num_steps=K, stochastic=False)
            candidates.append(gen)

            pos_np = gen["pos"].cpu().numpy()
            types_np = gen["atom_types"].cpu().numpy()
            mol, sanitized = coords_to_rdkit_mol(pos_np, types_np)

            if not sanitized or mol is None:
                r = 0.0
            else:
                rd = reward_oracle.compute_rl_reward(
                    mol=mol,
                    pK_pred=gen["pK_pred"],
                    pocket_path=pocket_path,
                    pocket_pos_updated=gen.get("pocket_pos_updated"),
                    rl_round=0,
                )
                r = float(rd["total_reward"])
            rewards.append(r)

    model.train()
    optimizer.zero_grad()
    h_P_train = model.pocket_encoder(pocket_pos_centered, pocket_feat)["h_P"]

    losses = []
    dt = 1.0 / K
    for g in range(G):
        gen = candidates[g]
        r_g = rewards[g]
        N_L = gen["num_atoms"]

        z_coord = torch.randn(N_L, 3, device=device_obj)
        z_coord = z_coord - z_coord.mean(dim=0, keepdim=True)
        z_type = torch.ones(N_L, 6, device=device_obj) / 6.0
        h_L_raw = torch.zeros(N_L, 4, device=device_obj)

        logp_proxy = torch.tensor(0.0, device=device_obj)
        for s in range(K):
            t_val = s * dt
            t = torch.tensor([t_val], device=device_obj)
            out = model.egnn(x_L=z_coord, h_L_raw=h_L_raw, atom_types_onehot=z_type, t=t, h_P=h_P_train)
            vel = out["vel_coord"]
            logp_proxy = logp_proxy - 0.5 * (vel ** 2).sum() * dt
            z_coord = z_coord + vel * dt
            z_type = z_type + out["vel_type"] * dt
            z_coord = z_coord - z_coord.mean(dim=0, keepdim=True)

        losses.append(-logp_proxy * r_g)

    # Midpoint KL penalty
    t_mid = torch.tensor([0.5], device=device_obj)
    with torch.no_grad():
        out_ref = model_ref.egnn(x_L=z_coord.detach(), h_L_raw=h_L_raw, atom_types_onehot=z_type.detach(), t=t_mid, h_P=h_P)
        v_ref = out_ref["vel_coord"]

    out_cur = model.egnn(x_L=z_coord.detach(), h_L_raw=h_L_raw, atom_types_onehot=z_type.detach(), t=t_mid, h_P=h_P_train)
    v_cur = out_cur["vel_coord"]
    kl_penalty = F.mse_loss(v_cur, v_ref)

    total_loss = torch.stack(losses).mean() + beta * kl_penalty
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    return {
        "policy_loss": torch.stack(losses).mean().item(),
        "kl_loss": kl_penalty.item(),
        "total_loss": total_loss.item(),
        "mean_reward": float(np.mean(rewards)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Main Pilot Experiment Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_golden_pilot_experiment(
    max_steps: int = 50,
    checkpoints_eval_steps: List[int] = [0, 10, 25, 50],
    G: int = 4,
    K: int = 20,
    lr: float = 5.0e-6,
    beta: float = 0.01,
    output_dir: str = "checkpoints/rl_pilot_500",
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Executes the golden-baseline-anchored pilot comparing G0 (Golden Baseline),
    G1 (SDE Flow-GRPO), and G2 (Historical Heuristic Control).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Verify and Load Golden Production Checkpoint
    golden_ckpt_path = PROJECT_ROOT / "checkpoints" / "rl_final.pt"
    assert golden_ckpt_path.exists(), f"Golden checkpoint not found at {golden_ckpt_path}"
    sha256_golden = get_file_sha256(golden_ckpt_path)
    logger.info(f"Golden checkpoint SHA256 verified: {sha256_golden}")

    pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=128, num_layers=4, knn_k=16)
    egnn = SE3EGNN(ligand_in_dim=4, pocket_dim=128, hidden_dim=128, num_layers=9, knn_k=16, num_atom_types=6)
    model_golden = FlowMatching(pocket_encoder=pocket_encoder, egnn=egnn, num_steps=50)

    golden_ckpt = torch.load(golden_ckpt_path, map_location="cpu", weights_only=False)
    model_golden.load_state_dict(golden_ckpt["model_state_dict"])
    model_golden.eval()

    # 2. Frozen Reference Model
    model_ref = copy.deepcopy(model_golden)
    for p in model_ref.parameters():
        p.requires_grad_(False)
    model_ref.eval()

    # 3. Load Benchmark Targets
    pf = PocketFeaturizer()
    candidate_pdb_paths = sorted(list(set(
        list(PROJECT_ROOT.glob("figures/*.pdb")) + list(PROJECT_ROOT.glob("uploads/**/*.pdb"))
    )))

    benchmark_pockets = []
    for p_path in candidate_pdb_paths:
        try:
            feat_dict = pf.featurize(str(p_path))
            if feat_dict["pos"] is not None and feat_dict["pos"].shape[0] >= 30:
                benchmark_pockets.append({
                    "path": str(p_path),
                    "name": p_path.stem,
                    "pos": feat_dict["pos"],
                    "feat": feat_dict["feat"],
                })
        except Exception:
            pass

    benchmark_pockets = benchmark_pockets[:5]
    logger.info(f"Loaded {len(benchmark_pockets)} benchmark pockets: {[p['name'] for p in benchmark_pockets]}")

    reward_oracle = RewardOracle(
        vina_every_n=1,
        min_carbon_ratio=0.40,
        max_nitrogen_ratio=0.35,
        max_nn_bonds=2,
        max_sa_score=6.0,
        max_ring_nitrogen=2,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. G0 — Evaluate Golden Baseline
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("Evaluating G0 — Golden PROTEUS Baseline (rl_final.pt)...")
    g0_metrics = evaluate_golden_protocol(model_golden, benchmark_pockets, reward_oracle, device=device)
    logger.info(
        f"[G0 Golden Baseline] PB-Valid: {g0_metrics['pb_validity_rate']*100:.1f}% | "
        f"QED: {g0_metrics['qed_mean']:.4f} (median: {g0_metrics['qed_median']:.4f}) | "
        f"SA: {g0_metrics['sa_mean']:.4f} | "
        f"Lipinski: {g0_metrics['lipinski_rate']*100:.1f}% | "
        f"Diversity: {g0_metrics['internal_diversity']:.4f} | "
        f"Reward: {g0_metrics['reward_mean']:.4f}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 5. G1 — SDE Flow-GRPO Pilot
    # ──────────────────────────────────────────────────────────────────────────
    logger.info(f"Starting G1 — SDE Flow-GRPO Pilot ({max_steps} steps, G={G}, K={K}, lr={lr})...")
    model_g1 = copy.deepcopy(model_golden).to(device)
    optimizer_g1 = torch.optim.Adam(model_g1.parameters(), lr=lr)

    g1_logs = {}
    # Initial Step 0 Evaluation
    g1_step0 = evaluate_golden_protocol(model_g1, benchmark_pockets, reward_oracle, device=device)
    g1_step0["ref_kl"] = 0.0
    g1_logs["step_000"] = g1_step0

    # Directory for step 0
    (out_path / "step_000").mkdir(parents=True, exist_ok=True)
    torch.save({"step": 0, "model_state_dict": model_g1.state_dict(), "metrics": g1_step0}, out_path / "step_000" / "g1_model.pt")

    for step in range(1, max_steps + 1):
        p_curr = benchmark_pockets[(step - 1) % len(benchmark_pockets)]

        step_diag = train_grpo_step(
            model=model_g1,
            model_ref=model_ref,
            optimizer=optimizer_g1,
            pocket_pos=p_curr["pos"],
            pocket_feat=p_curr["feat"],
            reward_oracle=reward_oracle,
            pocket_path=p_curr["path"],
            G=G,
            K=K,
            eps_clip=0.20,
            beta=beta,
            device=device,
        )

        if step in checkpoints_eval_steps:
            m = evaluate_golden_protocol(model_g1, benchmark_pockets, reward_oracle, device=device)
            m["ref_kl"] = step_diag["reference_kl"]
            m["mean_ratio"] = step_diag["mean_ratio"]
            m["max_param_delta"] = step_diag["max_param_delta"]
            step_tag = f"step_{step:03d}"
            g1_logs[step_tag] = m

            step_dir = out_path / step_tag
            step_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "model_state_dict": model_g1.state_dict(), "metrics": m}, step_dir / "g1_model.pt")

            logger.info(
                f"[G1 SDE-GRPO {step_tag}] Reward: {m['reward_mean']:.4f} (max: {m['reward_max']:.4f}) | "
                f"QED: {m['qed_mean']:.4f} | "
                f"PB-Valid: {m['pb_validity_rate']*100:.1f}% | "
                f"Diversity: {m['internal_diversity']:.4f} | "
                f"Ref-KL: {m['ref_kl']:.6f}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 6. G2 — Historical Heuristic Control
    # ──────────────────────────────────────────────────────────────────────────
    logger.info(f"Starting G2 — Historical Heuristic Control ({max_steps} steps, matched budget)...")
    model_g2 = copy.deepcopy(model_golden).to(device)
    optimizer_g2 = torch.optim.Adam(model_g2.parameters(), lr=lr)

    g2_logs = {}
    g2_logs["step_000"] = g1_step0

    for step in range(1, max_steps + 1):
        p_curr = benchmark_pockets[(step - 1) % len(benchmark_pockets)]

        step_diag = train_heuristic_step_g2(
            model=model_g2,
            model_ref=model_ref,
            optimizer=optimizer_g2,
            pocket_pos=p_curr["pos"],
            pocket_feat=p_curr["feat"],
            reward_oracle=reward_oracle,
            pocket_path=p_curr["path"],
            G=G,
            K=K,
            beta=beta,
            device=device,
        )

        if step in checkpoints_eval_steps:
            m = evaluate_golden_protocol(model_g2, benchmark_pockets, reward_oracle, device=device)
            m["kl_loss"] = step_diag["kl_loss"]
            step_tag = f"step_{step:03d}"
            g2_logs[step_tag] = m

            step_dir = out_path / step_tag
            step_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "model_state_dict": model_g2.state_dict(), "metrics": m}, step_dir / "g2_model.pt")

            logger.info(
                f"[G2 Heuristic {step_tag}] Reward: {m['reward_mean']:.4f} (max: {m['reward_max']:.4f}) | "
                f"QED: {m['qed_mean']:.4f} | "
                f"PB-Valid: {m['pb_validity_rate']*100:.1f}% | "
                f"Diversity: {m['internal_diversity']:.4f}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Comprehensive Summary Export
    # ──────────────────────────────────────────────────────────────────────────
    summary = {
        "metadata": {
            "golden_checkpoint": str(golden_ckpt_path),
            "golden_sha256": sha256_golden,
            "max_steps": max_steps,
            "G": G,
            "K": K,
            "lr": lr,
            "beta": beta,
            "seed": seed,
            "benchmark_pockets": [p["name"] for p in benchmark_pockets],
        },
        "G0_Golden_PROTEUS": g0_metrics,
        "G1_SDE_Flow_GRPO": g1_logs,
        "G2_Historical_Heuristic": g2_logs,
    }

    summary_file = out_path / "pilot_500_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Pilot experiment complete! Full record saved to {summary_file}")
    return summary


if __name__ == "__main__":
    run_golden_pilot_experiment(
        max_steps=50,
        checkpoints_eval_steps=[0, 10, 25, 50],
        G=4,
        K=20,
        lr=5.0e-6,
        beta=0.01,
    )
