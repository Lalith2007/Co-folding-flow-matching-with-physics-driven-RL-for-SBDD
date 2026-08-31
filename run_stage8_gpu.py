#!/usr/bin/env python3
"""
run_stage8_gpu.py — PROTEUS Stage 8 Clean Final-Scale Multi-Seed Validation.

Strict Separation Architecture:
- Training Pool: 43,127 training complexes across 8,907 unique PDBs (data/strict_server_train_pairs.json)
- Evaluation Pool: 20 strictly held-out benchmark PDBs (BENCHMARK_TEST_PDBS)
- Runtime Assertions: TRAIN_PDB_IDS ∩ BENCHMARK_TEST_PDBS == ∅
- Golden Baseline: checkpoints/rl_final.pt (Immutable, SHA256 verified)
- Optimization: SDE Flow-GRPO across Seeds [42, 123, 2026] (500 steps each)
- Evaluation: Deterministic 50 ODE steps, 200 molecules per checkpoint pass
- Memory Management: Memory-efficient k-NN via torch.cdist, pocket node truncation (N<=800), and explicit CUDA cache clearing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Auto-locate repository root
possible_roots = [
    Path.cwd(),
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
    Path("/Users/lalith/Desktop/SM_Generation"),
    Path("/Users/lalith/Desktop/StudyNew/K-HUB/dd_pipeline/SM_Generation"),
]
for r in possible_roots:
    if r.exists() and (r / "src").exists() and str(r) not in sys.path:
        sys.path.insert(0, str(r))

import argparse
import copy
import hashlib
import json
import logging
import math
import ssl
import time
import urllib.request
from typing import Dict, List, Any, Tuple, Optional, Set

import numpy as np
import torch
from rdkit import Chem, RDLogger

# Suppress RDKit C++ warnings from cluttering logs
RDLogger.DisableLog("rdApp.*")

# ──────────────────────────────────────────────────────────────────────────────
# Memory Optimization: Fast Memory-Efficient k-NN Graph Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_knn_graph_fast(
    pos: torch.Tensor,   # (N, 3)
    k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient k-NN using torch.cdist without allocating (N, N, 3) diff tensor."""
    n = pos.size(0)
    k_actual = min(k, max(n - 1, 1))
    with torch.no_grad():
        dist_det = torch.cdist(pos, pos)  # (N, N)
        dist_det.fill_diagonal_(float("inf"))
        _, knn_idx = dist_det.topk(k_actual, dim=-1, largest=False)  # (N, k)

    src = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(knn_idx).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)  # (2, N*k)

    edge_diff = pos[src] - pos[dst]
    edge_dist = torch.sqrt((edge_diff ** 2).sum(dim=-1) + 1e-8)
    return edge_index, edge_dist


import src.data.featurizer
import src.model.pocket_encoder
src.data.featurizer.build_knn_graph = build_knn_graph_fast
src.model.pocket_encoder.build_knn_graph = build_knn_graph_fast

from src.data.featurizer import PocketFeaturizer
from src.model.flow_matching import FlowMatching
from src.model.pocket_encoder import PocketEncoder
from src.model.egnn import SE3EGNN
from src.model.reward import RewardOracle
from src.train.sde_likelihood import (
    evaluate_trajectory_probability,
    compute_stabilized_ratio,
    compute_group_advantages,
    grpo_clipped_surrogate,
    compute_transition_kl,
)
from evaluate import compute_pb_validity, compute_tanimoto_diversity, coords_to_rdkit_mol, compute_mol_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stage8_clean_validation")

# ──────────────────────────────────────────────────────────────────────────────
# 0. Immutable Benchmark Targets & Checksum Constants
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_TEST_PDBS: List[str] = [
    "19gs", "1a27", "1a52", "1byg", "1c4u",
    "1cbq", "1d3p", "1d4p", "1dmt", "1e5t",
    "1e7a", "1e8z", "1eou", "1fax", "1fdt",
    "1g3m", "1g45", "1g50", "1g7f", "1gi7"
]

EXPECTED_GOLDEN_SHA256 = "b99ef527f009f50e99b3c376c8ea11323c2b1b5fb654ddb46454f51954d90d9e"


def get_file_sha256(filepath: str | Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Pocket Manager: Dynamic Loader & Cache for 8,907 Training PDBs
# ──────────────────────────────────────────────────────────────────────────────

class TrainingPocketManager:
    """Manages downloading, featurizing, node-capping, and in-memory caching of training pockets."""

    def __init__(self, cache_dir: str | Path = "data/train_pockets_cache", max_nodes: int = 800):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pf = PocketFeaturizer()
        self.max_nodes = max_nodes
        self.featurized_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.ssl_ctx = ssl._create_unverified_context()

    def get_pocket(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pdb_id = record["pdb_id"].lower()
        if pdb_id in self.featurized_cache:
            return self.featurized_cache[pdb_id]

        candidates = [
            Path(record.get("pocket_path", "")),
            Path("data") / record.get("pocket_path", ""),
            Path(record.get("protein_path", "")),
            Path("data") / record.get("protein_path", ""),
            Path(f"pdb_files/{pdb_id}.pdb"),
            Path(f"data/pdb_files/{pdb_id}.pdb"),
            self.cache_dir / f"{pdb_id}.pdb",
        ]

        pocket_file = None
        for cand in candidates:
            if cand.exists() and cand.is_file():
                pocket_file = cand
                break

        if pocket_file is None:
            target_path = self.cache_dir / f"{pdb_id}.pdb"
            try:
                url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
                with urllib.request.urlopen(url, context=self.ssl_ctx, timeout=10) as resp, open(target_path, "wb") as out_f:
                    out_f.write(resp.read())
                pocket_file = target_path
            except Exception:
                try:
                    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
                    with urllib.request.urlopen(url, context=self.ssl_ctx, timeout=10) as resp, open(target_path, "wb") as out_f:
                        out_f.write(resp.read())
                    pocket_file = target_path
                except Exception:
                    pocket_file = None

        if pocket_file is None or not pocket_file.exists():
            return None

        try:
            fd = self.pf.featurize(str(pocket_file))
            if fd["pos"].shape[0] < 5:
                return None

            pos = fd["pos"]
            feat = fd["feat"]

            # Cap large proteins to max_nodes closest to centroid to prevent CUDA OOM
            if pos.shape[0] > self.max_nodes:
                centroid = pos.mean(dim=0, keepdim=True)
                dists = torch.norm(pos - centroid, dim=-1)
                top_indices = torch.topk(dists, k=self.max_nodes, largest=False).indices
                top_indices = torch.sort(top_indices).values
                pos = pos[top_indices]
                feat = feat[top_indices]

            res = {
                "pdb_id": pdb_id,
                "path": str(pocket_file),
                "pos": pos,
                "feat": feat,
            }
            self.featurized_cache[pdb_id] = res
            return res
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# 2. Isolated Benchmark Evaluation Engine (Zero Optimizer Access)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_benchmark(
    model: FlowMatching,
    benchmark_pockets: List[Dict[str, Any]],
    reward_oracle: RewardOracle,
    mols_per_pocket: int = 10,
    device: str = "cuda",
    temperature: float = 0.8,
    element_bias: Optional[torch.Tensor] = None,
    eval_vina_mols: int = 2,
) -> Dict[str, Any]:
    """Strictly deterministic evaluation across benchmark pockets. Zero optimizer access."""
    model.eval()
    device_obj = torch.device(device)
    if element_bias is None:
        element_bias = torch.tensor([0.0, 0.05, 0.40, 0.0, 0.0, 0.0], device=device_obj)
    else:
        element_bias = element_bias.to(device_obj)

    all_molecules = []
    all_metrics = []
    all_smiles = []
    pb_valid_flags = []
    vina_scores = []
    rewards = []
    pocket_metrics = {p["pdb_id"]: [] for p in benchmark_pockets}
    per_pocket_smiles = {p["pdb_id"]: [] for p in benchmark_pockets}

    t_start_total = time.time()

    for p_idx, pocket_info in enumerate(benchmark_pockets):
        p_name = pocket_info["pdb_id"].upper()
        p_pos = pocket_info["pos"].to(device_obj)
        p_feat = pocket_info["feat"].to(device_obj)
        p_path = pocket_info["path"]
        pocket_com = p_pos.mean(dim=0).cpu().numpy()

        for mi in range(mols_per_pocket):
            t0 = time.time()
            res = model.sample(
                pocket_pos=p_pos,
                pocket_feat=p_feat,
                stochastic=False,
                temperature=temperature,
                element_bias=element_bias,
            )
            gen_time = time.time() - t0

            pos_np = res["pos"].cpu().numpy()
            types_np = res["atom_types"].cpu().numpy()
            pos_pocket = pos_np + pocket_com

            mol, sanitized = coords_to_rdkit_mol(pos_pocket, types_np)
            m_dict = compute_mol_metrics(mol, sanitized)
            m_dict["pocket"] = p_name
            m_dict["gen_time_s"] = gen_time

            pb_val = compute_pb_validity(mol) if sanitized else False
            m_dict["pb_valid"] = pb_val
            pb_valid_flags.append(pb_val)

            r_val = 0.0
            vina_val = None
            if sanitized and mol is not None:
                try:
                    pk_val = res.get("pK_pred", torch.tensor(0.0, device=device_obj))
                    run_vina = (mi < eval_vina_mols)
                    r_out = reward_oracle.compute_reward(
                        mol=mol,
                        pK_pred=pk_val,
                        pocket_path=p_path if run_vina else None,
                        pocket_pos_updated=p_pos.cpu(),
                        rl_round=0 if run_vina else 1,
                    )
                    r_val = float(r_out.get("total_reward", 0.0))
                    vina_val = float(r_out.get("r_vina", 0.0)) if (run_vina and r_out.get("r_vina") is not None) else None
                except Exception:
                    r_val = 0.0
            rewards.append(r_val)
            m_dict["reward"] = r_val
            m_dict["vina_score"] = vina_val
            if vina_val is not None and vina_val != 0.0:
                vina_scores.append(vina_val)

            if m_dict.get("valid", False) and m_dict.get("smiles"):
                smi = m_dict["smiles"]
                all_smiles.append(smi)
                per_pocket_smiles[pocket_info["pdb_id"]].append(smi)
                all_molecules.append(mol)

            all_metrics.append(m_dict)
            pocket_metrics[pocket_info["pdb_id"]].append(m_dict)

    tot_time = time.time() - t_start_total
    valid_metrics = [m for m in all_metrics if m.get("valid", False)]
    n_total = len(all_metrics)
    n_valid = len(valid_metrics)

    qeds = [m["qed"] for m in valid_metrics] if valid_metrics else [0.0]
    sas = [m.get("sa_score", 10.0) for m in valid_metrics] if valid_metrics else [10.0]
    lipinskis = [m.get("lipinski", 0) for m in valid_metrics] if valid_metrics else [0]

    pocket_divs = [
        compute_tanimoto_diversity(s_list)
        for s_list in per_pocket_smiles.values()
        if len(s_list) >= 2
    ]
    int_div = float(np.mean(pocket_divs)) if pocket_divs else 0.0

    per_pocket_summary = {}
    for pid, p_list in pocket_metrics.items():
        p_val = [m for m in p_list if m.get("valid", False)]
        per_pocket_summary[pid] = {
            "n_valid": len(p_val),
            "pb_valid_rate": float(np.mean([m["pb_valid"] for m in p_list])),
            "reward_mean": float(np.mean([m["reward"] for m in p_list])),
            "qed_mean": float(np.mean([m["qed"] for m in p_val])) if p_val else 0.0,
            "sa_mean": float(np.mean([m.get("sa_score", 10.0) for m in p_val])) if p_val else 10.0,
        }

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "validity_rate": float(n_valid / max(n_total, 1)),
        "pb_validity_rate": float(np.mean(pb_valid_flags)) if pb_valid_flags else 0.0,
        "qed_mean": float(np.mean(qeds)),
        "qed_median": float(np.median(qeds)),
        "qed_std": float(np.std(qeds)),
        "sa_mean": float(np.mean(sas)),
        "sa_std": float(np.std(sas)),
        "lipinski_rate": float(np.mean(lipinskis)) if lipinskis else 0.0,
        "internal_diversity": int_div,
        "unique_smiles_rate": float(len(set(all_smiles)) / max(len(all_smiles), 1)),
        "reward_mean": float(np.mean(rewards)),
        "reward_median": float(np.median(rewards)),
        "reward_max": float(np.max(rewards)),
        "vina_mean": float(np.mean(vina_scores)) if vina_scores else None,
        "avg_gen_time_s": float(tot_time / max(n_total, 1)),
        "per_pocket_summary": per_pocket_summary,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Isolated Training Step (SDE Flow-GRPO)
# ──────────────────────────────────────────────────────────────────────────────

def train_rl_step(
    model: FlowMatching,
    model_ref: FlowMatching,
    optimizer: torch.optim.Optimizer,
    pocket_pos: torch.Tensor,
    pocket_feat: torch.Tensor,
    reward_oracle: RewardOracle,
    pocket_path: Optional[str] = None,
    G: int = 4,
    K: int = 20,
    eps_clip: float = 0.20,
    beta: float = 0.01,
    temperature: float = 0.8,
    sigma_min: float = 0.01,
    sigma_max: float = 0.08,
    device: str = "cuda",
    max_grad_norm: float = 1.0,
) -> Dict[str, Any]:
    """Executes one single GRPO policy optimization step."""
    device_obj = torch.device(device)
    pocket_pos = pocket_pos.to(device_obj)
    pocket_feat = pocket_feat.to(device_obj)
    pocket_pos_centered = pocket_pos - pocket_pos.mean(dim=0, keepdim=True)
    pocket_com = pocket_pos.mean(dim=0).cpu().numpy()

    # 1. Rollouts under current policy
    model.eval()
    with torch.no_grad():
        h_P_rollout = model.pocket_encoder(pocket_pos_centered, pocket_feat)["h_P"]
        candidates = []
        old_log_probs = []
        rewards = []

        for g in range(G):
            gen = model.sample(
                pocket_pos=pocket_pos,
                pocket_feat=pocket_feat,
                stochastic=True,
                temperature=temperature,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                num_steps=K,
            )
            pos_np = gen["pos"].cpu().numpy()
            types_np = gen["atom_types"].cpu().numpy()
            pos_pocket = pos_np + pocket_com

            mol, sanitized = coords_to_rdkit_mol(pos_pocket, types_np)
            r_val = 0.0
            if sanitized and mol is not None:
                try:
                    pk_val = gen.get("pK_pred", torch.tensor(0.0, device=device_obj))
                    r_out = reward_oracle.compute_reward(
                        mol=mol,
                        pK_pred=pk_val,
                        pocket_path=pocket_path,
                        pocket_pos_updated=pocket_pos.cpu(),
                        rl_round=1,
                    )
                    r_val = float(r_out.get("total_reward", 0.0))
                except Exception:
                    r_val = 0.0

            rewards.append(r_val)
            candidates.append(gen)

            # Evaluate old trajectory log probability
            if "trajectory_states" in gen and len(gen["trajectory_states"]) > 0:
                old_prob_obj = evaluate_trajectory_probability(
                    model=model,
                    trajectory_states=gen["trajectory_states"],
                    z_type_rollout=gen.get("z_type_final"),
                    atom_types=gen["atom_types"],
                    step_sigmas=gen["step_sigmas"],
                    timesteps=gen["timesteps"],
                    h_P=h_P_rollout,
                    trajectory_types=gen.get("trajectory_types"),
                    temperature=temperature,
                )
                old_log_probs.append(old_prob_obj.total_log_prob.detach())
            else:
                old_log_probs.append(torch.tensor(0.0, device=device_obj))

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device_obj)
    advantages = compute_group_advantages(rewards_t)

    # 2. Policy Gradient update
    model.train()
    optimizer.zero_grad()

    h_P = model.pocket_encoder(pocket_pos_centered, pocket_feat)["h_P"]

    total_loss = torch.tensor(0.0, device=device_obj)
    ratios = []
    kls = []

    for g in range(G):
        gen = candidates[g]
        if "trajectory_states" not in gen or len(gen["trajectory_states"]) == 0:
            continue

        new_prob_obj = evaluate_trajectory_probability(
            model=model,
            trajectory_states=gen["trajectory_states"],
            z_type_rollout=gen.get("z_type_final"),
            atom_types=gen["atom_types"],
            step_sigmas=gen["step_sigmas"],
            timesteps=gen["timesteps"],
            h_P=h_P,
            trajectory_types=gen.get("trajectory_types"),
            model_ref=model_ref,
            temperature=temperature,
        )

        new_log_prob = new_prob_obj.total_log_prob
        kl_val = new_prob_obj.total_kl if new_prob_obj.total_kl is not None else torch.tensor(0.0, device=device_obj)
        kls.append(kl_val.item())

        ratio, ratio_diag = compute_stabilized_ratio(new_log_prob, old_log_probs[g])
        ratios.append(ratio_diag.mean_log_ratio)

        # GRPO maximizes surrogate => minimize negative surrogate
        surr_loss = -grpo_clipped_surrogate(ratio, advantages[g], eps_clip=eps_clip)

        cand_loss = surr_loss + beta * kl_val
        total_loss = total_loss + cand_loss

    total_loss = total_loss / max(len(candidates), 1)

    if total_loss.requires_grad:
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0)

    return {
        "loss": total_loss.item(),
        "mean_reward": float(rewards_t.mean().item()),
        "max_reward": float(rewards_t.max().item()),
        "mean_ratio": float(np.exp(np.mean(ratios))) if ratios else 1.0,
        "reference_kl": float(np.mean(kls)) if kls else 0.0,
        "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
    }


def build_proteus_model(device: str = "cuda") -> FlowMatching:
    pe = PocketEncoder(in_dim=40, hidden_dim=128, num_layers=4, knn_k=16)
    egnn = SE3EGNN(ligand_in_dim=4, pocket_dim=128, hidden_dim=128, num_layers=9, knn_k=16, num_atom_types=6)
    return FlowMatching(pocket_encoder=pe, egnn=egnn, num_steps=50).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Main Execution Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PROTEUS Stage 8 Clean Final-Scale Validation")
    parser.add_argument("--golden_ckpt", type=str, default="checkpoints/rl_final.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/rl_final_scale_clean")
    parser.add_argument("--train_pairs_json", type=str, default="data/strict_server_train_pairs.json")
    parser.add_argument("--benchmark_json", type=str, default="data/benchmark_20_pockets/benchmark_pockets.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--eps_clip", type=float, default=0.20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    logger.info("=" * 80)
    logger.info("PROTEUS STAGE 8 — CLEAN FINAL-SCALE MULTI-SEED VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Compute Device: {device.upper()}")
    if device == "cuda":
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Pre-Flight Verification
    golden_path = Path(args.golden_ckpt)
    assert golden_path.exists(), f"Golden checkpoint not found at {golden_path}"
    sha256_initial = get_file_sha256(golden_path)
    logger.info(f"Golden Checkpoint: {golden_path} | SHA256: {sha256_initial}")
    assert sha256_initial == EXPECTED_GOLDEN_SHA256, (
        f"CRITICAL: Golden checkpoint SHA256 mismatch! Expected {EXPECTED_GOLDEN_SHA256}, got {sha256_initial}"
    )

    train_pairs_file = Path(args.train_pairs_json)
    if not train_pairs_file.exists():
        train_pairs_file = Path("strict_server_train_pairs.json")
    assert train_pairs_file.exists(), f"Strict training pairs JSON not found at {train_pairs_file}"

    with open(train_pairs_file, "r") as f:
        train_pairs: List[Dict[str, Any]] = json.load(f)

    train_pdb_ids: Set[str] = set(p["pdb_id"].lower() for p in train_pairs)
    benchmark_test_set: Set[str] = set(p.lower() for p in BENCHMARK_TEST_PDBS)

    assert train_pdb_ids.isdisjoint(benchmark_test_set), "FATAL: Train and benchmark sets overlap!"
    logger.info("  --> Hard Assertion Passed: TRAIN_PDB_IDS ∩ BENCHMARK_TEST_PDBS == ∅ (Zero Overlap).")

    pf = PocketFeaturizer()
    with open(args.benchmark_json, "r") as f:
        bench_meta = json.load(f)

    benchmark_pockets: List[Dict[str, Any]] = []
    for entry in bench_meta:
        p_path = entry["path"]
        fd = pf.featurize(p_path)
        benchmark_pockets.append({
            "pdb_id": entry["pdb_id"].lower(),
            "path": p_path,
            "pos": fd["pos"],
            "feat": fd["feat"],
        })
    logger.info(f"Loaded and featurized {len(benchmark_pockets)} held-out benchmark test pockets.")

    reward_oracle = RewardOracle(
        vina_every_n=2,
        min_carbon_ratio=0.40,
        max_nitrogen_ratio=0.35,
        max_nn_bonds=2,
        max_sa_score=6.0,
        max_ring_nitrogen=2,
    )
    pocket_manager = TrainingPocketManager(max_nodes=800)

    # Dry-Run Pre-Flight
    model_dry = build_proteus_model(device=device)
    golden_ckpt = torch.load(golden_path, map_location=device, weights_only=False)
    model_dry.load_state_dict(golden_ckpt["model_state_dict"])
    model_dry_ref = copy.deepcopy(model_dry)
    model_dry_ref.requires_grad_(False)
    opt_dry = torch.optim.Adam(model_dry.parameters(), lr=args.lr)

    dry_pocket = pocket_manager.get_pocket(train_pairs[0])
    assert dry_pocket is not None
    assert dry_pocket["pdb_id"].lower() not in benchmark_test_set

    dry_diag = train_rl_step(
        model=model_dry,
        model_ref=model_dry_ref,
        optimizer=opt_dry,
        pocket_pos=dry_pocket["pos"],
        pocket_feat=dry_pocket["feat"],
        reward_oracle=reward_oracle,
        G=2,
        K=5,
        eps_clip=args.eps_clip,
        beta=args.beta,
        device=device,
    )
    del model_dry, model_dry_ref, opt_dry
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"Dry-run passed: Loss={dry_diag['loss']:.4f} | Reward={dry_diag['mean_reward']:.4f}")

    # Phase 1: G0 Evaluation
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: G0 — EVALUATING IMMUTABLE GOLDEN BASELINE (200 Molecules)...")
    logger.info("=" * 80)
    model_golden = build_proteus_model(device=device)
    model_golden.load_state_dict(golden_ckpt["model_state_dict"])
    model_golden.eval()

    g0_metrics = evaluate_benchmark(model_golden, benchmark_pockets, reward_oracle, mols_per_pocket=10, device=device)
    logger.info(f"[G0 Golden Baseline] Validity: {g0_metrics['validity_rate']*100:.1f}% | "
                f"PB-Valid: {g0_metrics['pb_validity_rate']*100:.1f}% | "
                f"QED: {g0_metrics['qed_mean']:.4f} | SA: {g0_metrics['sa_mean']:.4f} | "
                f"Reward Mean: {g0_metrics['reward_mean']:.4f} (Max: {g0_metrics['reward_max']:.4f})")

    # Explicitly free G0 model and VRAM cache before G1 training
    del model_golden
    if device == "cuda":
        torch.cuda.empty_cache()

    # Phase 2: G1 Training across Seeds
    eval_checkpoints = [0, 25, 50, 100, 200, 300, 400, 500]
    all_seed_results: Dict[str, Any] = {}
    sampled_training_pdbs_global: Set[str] = set()

    for seed in args.seeds:
        logger.info("\n" + "=" * 80)
        logger.info(f"PHASE 2: G1 — SDE FLOW-GRPO TRAINING [SEED {seed}] (500 Steps)...")
        logger.info("=" * 80)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        seed_dir = out_path / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        model_g1 = build_proteus_model(device=device)
        model_g1.load_state_dict(golden_ckpt["model_state_dict"])

        model_ref = copy.deepcopy(model_g1)
        for p in model_ref.parameters():
            p.requires_grad_(False)
        model_ref.eval()

        optimizer = torch.optim.Adam(model_g1.parameters(), lr=args.lr)
        checkpoint_records: Dict[str, Any] = {}
        seed_sampled_pdbs: List[str] = []

        if 0 in eval_checkpoints:
            logger.info(f"[Seed {seed} step_000] Evaluating baseline anchor on 20 held-out benchmark pockets...")
            m0 = evaluate_benchmark(model_g1, benchmark_pockets, reward_oracle, mols_per_pocket=10, device=device)
            m0["ref_kl"] = 0.0
            m0["mean_ratio"] = 1.0
            checkpoint_records["step_000"] = m0
            step_dir = seed_dir / "step_000"
            step_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"step": 0, "model_state_dict": model_g1.state_dict(), "metrics": m0}, step_dir / "g1_model.pt")

        rng_seed = np.random.RandomState(seed)
        t_start_seed = time.time()

        for step in range(1, args.steps + 1):
            idx = rng_seed.randint(0, len(train_pairs))
            rec = train_pairs[idx]
            train_pdb = rec["pdb_id"].lower()
            assert train_pdb not in benchmark_test_set, f"FATAL LEAKAGE: Step {step} sampled test PDB {train_pdb}"

            p_data = pocket_manager.get_pocket(rec)
            attempts = 0
            while p_data is None and attempts < 10:
                idx = rng_seed.randint(0, len(train_pairs))
                rec = train_pairs[idx]
                train_pdb = rec["pdb_id"].lower()
                if train_pdb not in benchmark_test_set:
                    p_data = pocket_manager.get_pocket(rec)
                attempts += 1

            if p_data is None:
                p_data = dry_pocket
                train_pdb = dry_pocket["pdb_id"].lower()

            seed_sampled_pdbs.append(train_pdb)
            sampled_training_pdbs_global.add(train_pdb)

            step_diag = train_rl_step(
                model=model_g1,
                model_ref=model_ref,
                optimizer=optimizer,
                pocket_pos=p_data["pos"],
                pocket_feat=p_data["feat"],
                reward_oracle=reward_oracle,
                pocket_path=p_data.get("path"),
                G=4,
                K=20,
                eps_clip=args.eps_clip,
                beta=args.beta,
                device=device,
            )

            if step % 25 == 0 or step in eval_checkpoints:
                logger.info(f"  [Seed {seed} step {step:03d}/500 | PDB {train_pdb.upper()}] Reward: {step_diag['mean_reward']:.4f} "
                            f"(max: {step_diag['max_reward']:.4f}) | Ref-KL: {step_diag['reference_kl']:.6f} | "
                            f"Ratio: {step_diag['mean_ratio']:.4f} | Grad: {step_diag['grad_norm']:.4f}")

            if step in eval_checkpoints and step > 0:
                logger.info(f"--> [Seed {seed} Checkpoint step_{step:03d}] Running official 200-molecule benchmark...")
                m_step = evaluate_benchmark(model_g1, benchmark_pockets, reward_oracle, mols_per_pocket=10, device=device)
                m_step["ref_kl"] = step_diag["reference_kl"]
                m_step["mean_ratio"] = step_diag["mean_ratio"]
                checkpoint_records[f"step_{step:03d}"] = m_step

                step_dir = seed_dir / f"step_{step:03d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"step": step, "model_state_dict": model_g1.state_dict(), "metrics": m_step}, step_dir / "g1_model.pt")

                logger.info(f"    [step_{step:03d} Benchmark] Reward: {m_step['reward_mean']:.4f} (max: {m_step['reward_max']:.4f}) | "
                            f"QED: {m_step['qed_mean']:.4f} | PB-Valid: {m_step['pb_validity_rate']*100:.1f}% | "
                            f"Div: {m_step['internal_diversity']:.4f} | SA: {m_step['sa_mean']:.4f}")

        tot_time_seed = time.time() - t_start_seed
        logger.info(f"[Seed {seed}] Completed 500 steps in {tot_time_seed:.1f}s ({tot_time_seed/60:.2f}m).")
        
        seed_sampled_set = set(seed_sampled_pdbs)
        assert seed_sampled_set.isdisjoint(benchmark_test_set), "Leakage detected post-seed!"
        
        all_seed_results[f"seed_{seed}"] = {
            "checkpoints": checkpoint_records,
            "sampled_training_pdbs_count": len(seed_sampled_set),
            "runtime_seconds": tot_time_seed,
        }

        # Clear VRAM after each seed
        del model_g1, model_ref, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

        sha256_post_seed = get_file_sha256(golden_path)
        assert sha256_post_seed == EXPECTED_GOLDEN_SHA256

    # Summary
    summary = {
        "metadata": {
            "experiment": "Stage 8 Clean Final-Scale Multi-Seed Validation",
            "golden_checkpoint": str(golden_path),
            "sha256": sha256_initial,
            "seeds": args.seeds,
            "steps": args.steps,
            "sampled_unique_training_pdbs": len(sampled_training_pdbs_global),
            "benchmark_targets": BENCHMARK_TEST_PDBS,
            "leakage_count": len(sampled_training_pdbs_global & benchmark_test_set),
        },
        "G0_Golden_PROTEUS": g0_metrics,
        "G1_SDE_Flow_GRPO": all_seed_results,
    }

    summary_file = out_path / "stage8_clean_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nConsolidated Stage 8 Clean Summary written to: {summary_file}")


if __name__ == "__main__":
    main()
