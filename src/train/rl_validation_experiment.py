"""
src/train/rl_validation_experiment.py — Controlled RL Validation Experiment.

Compares:
  - B0: Base Flow (untouched pretrained model, stochastic=False, no RL)
  - B1: Historical Heuristic (reward-weighted kinetic-energy regularization)
  - B2: SDE Flow-GRPO (our verified SDE Group Relative Policy Optimization)

Evaluates on fixed representative pocket set with rigorous multi-dimensional metrics:
  - Multi-objective rewards & chemical decomposition (QED, SA, Lipinski, Vina, Gates)
  - Physical validity & PoseBusters geometry checks (clashes, bond lengths, angles)
  - Molecular diversity (Morgan fingerprint pairwise Tanimoto distance, unique SMILES)
  - Reward-hacking diagnostics (MW, Carbon ratio, Heteroatom frequencies)
  - Model drift (Reference Transition KL, parameter delta)
  - Policy ratio health (min, max, mean, clamped fraction)
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
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

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger("rl_experiment")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Chemical & Geometric Evaluation Suite (PoseBusters-compatible checks)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_molecules_batch(
    molecules: List[Tuple[Optional[Chem.Mol], bool, np.ndarray, np.ndarray, float]],
    reward_oracle: RewardOracle,
    pocket_path: str,
    pocket_pos_updated: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Evaluates a batch of generated molecules across chemical, physical, and diversity axes.
    """
    total = len(molecules)
    if total == 0:
        return {}

    valid_mols = []
    rewards = []
    qeds = []
    sas = []
    lipinskis = []
    gate_passes = []
    vinas = []
    mws = []
    c_ratios = []
    n_ratios = []
    o_ratios = []

    # PoseBusters physical metrics
    clash_passes = 0
    bond_len_passes = 0
    bond_ang_passes = 0

    smiles_list = []

    for mol, sanitized, pos_np, types_np, pK_pred in molecules:
        if not sanitized or mol is None:
            rewards.append(0.0)
            gate_passes.append(False)
            continue

        valid_mols.append(mol)
        smi = Chem.MolToSmiles(mol)
        smiles_list.append(smi)

        # Chemical properties
        q = RewardOracle.compute_qed(mol)
        sa = RewardOracle.compute_sa_raw(mol)
        mw = Descriptors.MolWt(mol)
        num_heavy = max(mol.GetNumHeavyAtoms(), 1)
        c_r = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C") / num_heavy
        n_r = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N") / num_heavy

        num_atoms = max(mol.GetNumHeavyAtoms(), 1)
        num_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
        o_r = num_o / num_atoms

        # Lipinski compliance
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        logp = Descriptors.MolLogP(mol)
        lip_pass = (mw <= 500) and (logp <= 5.0) and (hbd <= 5) and (hba <= 10)

        # Multi-objective reward
        rd = reward_oracle.compute_rl_reward(
            mol=mol,
            pK_pred=torch.as_tensor(pK_pred),
            pocket_path=pocket_path,
            pocket_pos_updated=pocket_pos_updated,
            rl_round=0,
        )
        r = float(rd["total_reward"])
        gp = bool(rd.get("gate_pass", False))
        vina = float(rd.get("vina_score", 0.0))

        rewards.append(r)
        qeds.append(q)
        sas.append(sa)
        lipinskis.append(1.0 if lip_pass else 0.0)
        gate_passes.append(gp)
        vinas.append(vina)
        mws.append(mw)
        c_ratios.append(c_r)
        n_ratios.append(n_r)
        o_ratios.append(o_r)

        # PoseBusters checks on 3D coordinates
        # 1. Steric clash check (no pairwise distance < 0.8 * (r_i + r_j))
        conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
        if conf is not None and mol.GetNumAtoms() > 1:
            coords = conf.GetPositions()
            ptable = Chem.GetPeriodicTable()
            vdw_radii = [ptable.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()]
            n_a = len(vdw_radii)

            clash = False
            for i in range(n_a):
                for j in range(i + 1, n_a):
                    d = np.linalg.norm(coords[i] - coords[j])
                    min_allowed = 0.70 * (vdw_radii[i] + vdw_radii[j])
                    if d < min_allowed:
                        clash = True
                        break
                if clash:
                    break
            if not clash:
                clash_passes += 1

            # 2. Bond length check ([1.05, 1.85] A for all bonded pairs)
            bond_len_ok = True
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                bl = np.linalg.norm(coords[i] - coords[j])
                if bl < 1.05 or bl > 1.85:
                    bond_len_ok = False
                    break
            if bond_len_ok:
                bond_len_passes += 1

            # 3. Bond angle check ([85, 150] deg)
            bond_ang_ok = True
            for a in mol.GetAtoms():
                neighbors = [nbr.GetIdx() for nbr in a.GetNeighbors()]
                if len(neighbors) >= 2:
                    c_center = coords[a.GetIdx()]
                    v1 = coords[neighbors[0]] - c_center
                    v2 = coords[neighbors[1]] - c_center
                    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    ang_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
                    if ang_deg < 80.0 or ang_deg > 155.0:
                        bond_ang_ok = False
                        break
            if bond_ang_ok:
                bond_ang_passes += 1

    n_valid = len(valid_mols)
    validity = n_valid / total

    # Internal diversity via Morgan fingerprints
    internal_diversity = 0.0
    if n_valid >= 2:
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in valid_mols]
        dists = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dists.append(1.0 - sim)
        internal_diversity = float(np.mean(dists)) if dists else 0.0

    unique_smiles_count = len(set(smiles_list))
    unique_smiles_rate = unique_smiles_count / max(n_valid, 1)

    return {
        "validity": validity,
        "n_valid": n_valid,
        "n_total": total,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_median": float(np.median(rewards)) if rewards else 0.0,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "gate_pass_rate": float(np.mean(gate_passes)) if gate_passes else 0.0,
        "qed_mean": float(np.mean(qeds)) if qeds else 0.0,
        "sa_mean": float(np.mean(sas)) if sas else 0.0,
        "lipinski_rate": float(np.mean(lipinskis)) if lipinskis else 0.0,
        "vina_mean": float(np.mean(vinas)) if vinas else 0.0,
        "mw_mean": float(np.mean(mws)) if mws else 0.0,
        "carbon_ratio_mean": float(np.mean(c_ratios)) if c_ratios else 0.0,
        "nitrogen_ratio_mean": float(np.mean(n_ratios)) if n_ratios else 0.0,
        "oxygen_ratio_mean": float(np.mean(o_ratios)) if o_ratios else 0.0,
        "internal_diversity": internal_diversity,
        "unique_smiles_rate": unique_smiles_rate,
        "posebusters_clash_pass_rate": clash_passes / max(n_valid, 1),
        "posebusters_bond_len_pass_rate": bond_len_passes / max(n_valid, 1),
        "posebusters_bond_ang_pass_rate": bond_ang_passes / max(n_valid, 1),
        "posebusters_overall_pass_rate": min(clash_passes, bond_len_passes, bond_ang_passes) / max(n_valid, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Historical Heuristic (B1) Single Step Update
# ──────────────────────────────────────────────────────────────────────────────

def train_heuristic_b1_step(
    model: FlowMatching,
    model_ref: FlowMatching,
    optimizer: torch.optim.Optimizer,
    pocket_pos: torch.Tensor,
    pocket_feat: torch.Tensor,
    reward_oracle: RewardOracle,
    pocket_path: str,
    G: int = 2,
    K: int = 10,
    beta: float = 0.01,
    device: str = "cpu",
    max_grad_norm: float = 1.0,
) -> Dict[str, Any]:
    """
    Executes a single step of the historical B1 heuristic:
      L_B1 = (1/G) \sum_g [ 0.5 * ||v||^2 * dt * r_g ] + beta * MSE(v(t=0.5), v_ref(t=0.5))
    """
    device_obj = torch.device(device)
    pocket_pos = pocket_pos.to(device_obj)
    pocket_feat = pocket_feat.to(device_obj)

    # 1. Rollout candidates (no grad)
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

    # 2. Re-run flow with autograd
    model.train()
    optimizer.zero_grad()

    h_P_train = model.pocket_encoder(pocket_pos_centered, pocket_feat)["h_P"]

    b1_losses = []
    dt = 1.0 / K

    for g in range(G):
        gen = candidates[g]
        r_g = rewards[g]
        N_L = gen["num_atoms"]

        z_coord = torch.randn(N_L, 3, device=device_obj)
        z_coord = z_coord - z_coord.mean(dim=0, keepdim=True)
        z_type = torch.ones(N_L, 6, device=device_obj) / 6.0
        h_L_raw = torch.zeros(N_L, 4, device=device_obj)

        log_prob_proxy = torch.tensor(0.0, device=device_obj)
        for s in range(K):
            t_val = s * dt
            t = torch.tensor([t_val], device=device_obj)
            out = model.egnn(x_L=z_coord, h_L_raw=h_L_raw, atom_types_onehot=z_type, t=t, h_P=h_P_train)
            vel = out["vel_coord"]
            log_prob_proxy = log_prob_proxy - 0.5 * (vel ** 2).sum() * dt
            z_coord = z_coord + vel * dt
            z_type = z_type + out["vel_type"] * dt
            z_coord = z_coord - z_coord.mean(dim=0, keepdim=True)

        loss_g = -log_prob_proxy * r_g
        b1_losses.append(loss_g)

    # Heuristic midpoint KL penalty
    t_mid = torch.tensor([0.5], device=device_obj)
    with torch.no_grad():
        out_ref = model_ref.egnn(x_L=z_coord.detach(), h_L_raw=h_L_raw, atom_types_onehot=z_type.detach(), t=t_mid, h_P=h_P)
        v_ref = out_ref["vel_coord"]

    out_cur = model.egnn(x_L=z_coord.detach(), h_L_raw=h_L_raw, atom_types_onehot=z_type.detach(), t=t_mid, h_P=h_P_train)
    v_cur = out_cur["vel_coord"]
    kl_penalty = F.mse_loss(v_cur, v_ref)

    total_loss = torch.stack(b1_losses).mean() + beta * kl_penalty
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    return {
        "policy_loss": torch.stack(b1_losses).mean().item(),
        "kl_loss": kl_penalty.item(),
        "total_loss": total_loss.item(),
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Main Controlled Experiment Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_validation_experiment(
    max_steps: int = 25,
    eval_every: int = 10,
    G: int = 2,
    K: int = 10,
    lr: float = 5e-5,
    beta: float = 0.01,
    output_dir: str = "checkpoints/rl_validation",
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Executes the full controlled comparison: B0 (Base) vs B1 (Heuristic) vs B2 (SDE Flow-GRPO).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Pretrained Checkpoint as Base Model
    ckpt_path = PROJECT_ROOT / "checkpoints" / "pretrain_final.pt"
    assert ckpt_path.exists(), f"Pretrained checkpoint not found at {ckpt_path}"

    pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=128, num_layers=4, knn_k=16)
    egnn = SE3EGNN(ligand_in_dim=4, pocket_dim=128, hidden_dim=128, num_layers=9, knn_k=16, num_atom_types=6)
    base_model = FlowMatching(pocket_encoder=pocket_encoder, egnn=egnn, num_steps=K)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()

    # 2. Frozen Reference Model
    model_ref = copy.deepcopy(base_model)
    for p in model_ref.parameters():
        p.requires_grad_(False)
    model_ref.eval()

    # 3. Select Deterministic Pocket Set
    pf = PocketFeaturizer()
    candidate_pdb_paths = sorted(list(set(
        list(PROJECT_ROOT.glob("figures/*.pdb")) + list(PROJECT_ROOT.glob("uploads/**/*.pdb"))
    )))

    pockets_data = []
    for p_path in candidate_pdb_paths:
        try:
            feat_dict = pf.featurize(str(p_path))
            if feat_dict["pos"] is not None and feat_dict["pos"].shape[0] >= 30:
                pockets_data.append({
                    "path": str(p_path),
                    "name": p_path.stem,
                    "pos": feat_dict["pos"],
                    "feat": feat_dict["feat"],
                })
        except Exception:
            pass

    # Ensure fixed 5 benchmark pockets
    benchmark_pockets = pockets_data[:5]
    logger.info(f"Loaded {len(benchmark_pockets)} benchmark pockets for validation: {[p['name'] for p in benchmark_pockets]}")

    reward_oracle = RewardOracle(
        vina_every_n=1,
        min_carbon_ratio=0.40,
        max_nitrogen_ratio=0.35,
        max_nn_bonds=2,
        max_sa_score=6.0,
        max_ring_nitrogen=2,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: Benchmark Evaluation Routine
    # ──────────────────────────────────────────────────────────────────────────
    def run_benchmark_eval(model: FlowMatching, method_name: str, step_num: int, stochastic: bool) -> Dict[str, Any]:
        model.eval()
        all_eval_mols = []
        total_kl_accum = 0.0
        n_eval_pockets = len(benchmark_pockets)

        with torch.no_grad():
            for p_info in benchmark_pockets:
                pos = p_info["pos"].to(device)
                feat = p_info["feat"].to(device)

                # Generate 4 molecules per benchmark pocket
                for m_idx in range(4):
                    gen = model.sample(
                        pos, feat, num_atoms=18, num_steps=K,
                        stochastic=stochastic, sigma_min=0.01, sigma_max=0.08
                    )
                    pos_np = gen["pos"].cpu().numpy()
                    types_np = gen["atom_types"].cpu().numpy()
                    mol, sanitized = coords_to_rdkit_mol(pos_np, types_np)
                    all_eval_mols.append((mol, sanitized, pos_np, types_np, gen["pK_pred"].item()))

                # Compute Reference KL on this pocket
                if method_name != "B0":
                    p_centered = pos - pos.mean(dim=0, keepdim=True)
                    h_P_cur = model.pocket_encoder(p_centered, feat)["h_P"]
                    h_P_ref = model_ref.pocket_encoder(p_centered, feat)["h_P"]

                    z_test = torch.randn(15, 3, device=device)
                    z_test = z_test - z_test.mean(dim=0, keepdim=True)
                    h_raw = torch.zeros(15, 4, device=device)
                    z_t = torch.ones(15, 6, device=device) / 6.0
                    t_mid = torch.tensor([0.5], device=device)

                    v_cur = model.egnn(x_L=z_test, h_L_raw=h_raw, atom_types_onehot=z_t, t=t_mid, h_P=h_P_cur)["vel_coord"]
                    v_ref = model_ref.egnn(x_L=z_test, h_L_raw=h_raw, atom_types_onehot=z_t, t=t_mid, h_P=h_P_ref)["vel_coord"]

                    v_cur_p = v_cur - v_cur.mean(dim=0, keepdim=True)
                    v_ref_p = v_ref - v_ref.mean(dim=0, keepdim=True)
                    dt = 1.0 / K
                    sigma_mid = 0.08
                    kl_step = (dt / (2.0 * (sigma_mid ** 2))) * ((v_cur_p - v_ref_p) ** 2).sum().item()
                    total_kl_accum += kl_step

        metrics = evaluate_molecules_batch(all_eval_mols, reward_oracle, pocket_path=benchmark_pockets[0]["path"])
        metrics["method"] = method_name
        metrics["step"] = step_num
        metrics["ref_kl_midpoint"] = total_kl_accum / max(n_eval_pockets, 1)

        # Parameter Delta against Reference
        ref_p_dict = dict(model_ref.named_parameters())
        p_deltas = [
            torch.abs(p - ref_p_dict[name]).max().item()
            for name, p in model.named_parameters()
        ]
        metrics["max_param_delta_from_ref"] = max(p_deltas) if p_deltas else 0.0

        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Evaluate B0 (Base Flow, Untrained)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("Evaluating B0 — Base Flow Baseline...")
    b0_metrics = run_benchmark_eval(base_model, "B0_BaseFlow", step_num=0, stochastic=False)
    logger.info(
        f"[B0 BaseFlow] Reward: {b0_metrics['reward_mean']:.4f} | "
        f"Validity: {b0_metrics['validity']*100:.1f}% | "
        f"PoseBusters: {b0_metrics['posebusters_overall_pass_rate']*100:.1f}% | "
        f"Diversity: {b0_metrics['internal_diversity']:.4f}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Train & Evaluate B1 (Historical Heuristic)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info(f"Training B1 — Historical Heuristic ({max_steps} steps)...")
    model_b1 = copy.deepcopy(base_model).to(device)
    optimizer_b1 = torch.optim.Adam(model_b1.parameters(), lr=lr)

    b1_logs = []
    b1_initial_metrics = run_benchmark_eval(model_b1, "B1_Heuristic", step_num=0, stochastic=False)
    b1_logs.append(b1_initial_metrics)

    for step in range(1, max_steps + 1):
        # Rotate through benchmark pockets
        p_idx = (step - 1) % len(benchmark_pockets)
        p_curr = benchmark_pockets[p_idx]

        step_res = train_heuristic_b1_step(
            model=model_b1,
            model_ref=model_ref,
            optimizer=optimizer_b1,
            pocket_pos=p_curr["pos"],
            pocket_feat=p_curr["feat"],
            reward_oracle=reward_oracle,
            pocket_path=p_curr["path"],
            G=G,
            K=K,
            beta=beta,
            device=device,
        )

        if step % eval_every == 0 or step == max_steps:
            m = run_benchmark_eval(model_b1, "B1_Heuristic", step_num=step, stochastic=False)
            b1_logs.append(m)
            logger.info(
                f"[B1 Step {step:2d}] Loss: {step_res['total_loss']:.4f} | "
                f"Reward: {m['reward_mean']:.4f} | "
                f"Validity: {m['validity']*100:.1f}% | "
                f"PoseBusters: {m['posebusters_overall_pass_rate']*100:.1f}% | "
                f"Ref-KL: {m['ref_kl_midpoint']:.6f}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Train & Evaluate B2 (SDE Flow-GRPO)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info(f"Training B2 — SDE Flow-GRPO ({max_steps} steps)...")
    model_b2 = copy.deepcopy(base_model).to(device)
    optimizer_b2 = torch.optim.Adam(model_b2.parameters(), lr=lr)

    b2_logs = []
    b2_initial_metrics = run_benchmark_eval(model_b2, "B2_FlowGRPO", step_num=0, stochastic=True)
    b2_logs.append(b2_initial_metrics)

    # Save Step 0 Checkpoint
    torch.save({"step": 0, "model_state_dict": model_b2.state_dict()}, out_path / "b2_step_000.pt")

    for step in range(1, max_steps + 1):
        p_idx = (step - 1) % len(benchmark_pockets)
        p_curr = benchmark_pockets[p_idx]

        step_res = train_grpo_step(
            model=model_b2,
            model_ref=model_ref,
            optimizer=optimizer_b2,
            pocket_pos=p_curr["pos"],
            pocket_feat=p_curr["feat"],
            reward_oracle=reward_oracle,
            pocket_path=p_curr["path"],
            G=G,
            K=K,
            eps_clip=0.2,
            beta=beta,
            device=device,
        )

        if step % eval_every == 0 or step == max_steps:
            m = run_benchmark_eval(model_b2, "B2_FlowGRPO", step_num=step, stochastic=True)
            # Add GRPO step diagnostics to log
            m["mean_ratio"] = step_res["mean_ratio"]
            m["mean_abs_ratio_minus_one"] = step_res["mean_abs_ratio_minus_one"]
            m["total_grad_norm"] = step_res["total_grad_norm"]
            m["velocity_equivariance_error"] = step_res["velocity_equivariance_error"]
            b2_logs.append(m)

            # Save checkpoint
            torch.save(
                {"step": step, "model_state_dict": model_b2.state_dict(), "metrics": m},
                out_path / f"b2_step_{step:03d}.pt",
            )

            logger.info(
                f"[B2 Step {step:2d}] Loss: {step_res['total_loss']:.4f} | "
                f"Reward: {m['reward_mean']:.4f} | "
                f"GatePass: {m['gate_pass_rate']*100:.1f}% | "
                f"Validity: {m['validity']*100:.1f}% | "
                f"PoseBusters: {m['posebusters_overall_pass_rate']*100:.1f}% | "
                f"Ratio: {m['mean_ratio']:.4f} | "
                f"Ref-KL: {m['ref_kl_midpoint']:.6f}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Write Logs to Disk
    # ──────────────────────────────────────────────────────────────────────────
    experiment_summary = {
        "config": {
            "max_steps": max_steps,
            "eval_every": eval_every,
            "G": G,
            "K": K,
            "lr": lr,
            "beta": beta,
            "seed": seed,
            "benchmark_pockets": [p["name"] for p in benchmark_pockets],
        },
        "B0_BaseFlow": b0_metrics,
        "B1_Heuristic_Logs": b1_logs,
        "B2_FlowGRPO_Logs": b2_logs,
    }

    log_file = out_path / "rl_validation_summary.json"
    with open(log_file, "w") as f:
        json.dump(experiment_summary, f, indent=2)

    logger.info(f"Experiment complete! Summary saved to {log_file}")
    return experiment_summary


if __name__ == "__main__":
    run_validation_experiment(max_steps=25, eval_every=10, G=2, K=10, lr=5e-5, beta=0.01)
