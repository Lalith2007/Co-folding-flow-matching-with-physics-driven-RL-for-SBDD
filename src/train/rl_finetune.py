"""
rl_finetune.py — Phase B: SDE Flow-GRPO Policy Optimization.

Reward-guided optimization of equivariant Flow Matching via Group Relative
Policy Optimization (GRPO) on continuous stochastic coordinate-simplex trajectories.

Training workflow per round:
  1. Sample pockets from dataset.
  2. For each pocket, generate a group of G stochastic trajectories (Euler-Maruyama SDE).
  3. Score all G molecules with RewardOracle (binding affinity, safety gates, QED, SA).
  4. Compute group-normalized advantages: A_g = (r_g - mean(r)) / (std(r) + eps).
  5. Evaluate exact discrete transition likelihoods under current policy theta and frozen theta_0.
  6. Compute PPO/GRPO clipped surrogate objective: L_policy = -min(r·A, clip(r, 1-eps, 1+eps)·A).
  7. Regularize with timestep-weighted transition KL: L_KL = dt / (2*sigma_s^2) * ||v_theta - v_ref||^2.
  8. Update theta with Adam optimizer and gradient clipping.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import SBDDDataset, get_rl_subset
from ..model.flow_matching import FlowMatching
from ..model.reward import RewardOracle
from ..model.utils import CosineBetaSchedule
from .sde_likelihood import (
    TrajectoryProbability,
    PolicyRatioDiagnostics,
    compute_group_advantages,
    compute_stabilized_ratio,
    evaluate_trajectory_probability,
    grpo_clipped_surrogate,
)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


def compute_velocity_equivariance_diagnostic(
    model: FlowMatching,
    pocket_pos: torch.Tensor,
    pocket_feat: torch.Tensor,
    N_L: int = 10,
    device: torch.device = None,
) -> float:
    """
    Diagnostic measurement of rotation equivariance error for the current vel_coord head.
    Reports ||v(Rx, RP) - R v(x, P)||_max without altering model architecture.
    """
    model.eval()
    with torch.no_grad():
        x_L = torch.randn(N_L, 3, device=device)
        x_L = x_L - x_L.mean(dim=0, keepdim=True)
        h_L_raw = torch.zeros(N_L, 4, device=device)
        z_type = torch.ones(N_L, 6, device=device) / 6.0
        t = torch.tensor([0.5], device=device)

        pocket_pos_centered = pocket_pos - pocket_pos.mean(dim=0, keepdim=True)
        h_P = model.pocket_encoder(pocket_pos_centered, pocket_feat)["h_P"]

        out1 = model.egnn(x_L=x_L, h_L_raw=h_L_raw, atom_types_onehot=z_type, t=t, h_P=h_P)
        v1 = out1["vel_coord"]

        # Random SO(3) rotation
        A = torch.randn(3, 3, device=device)
        Q, R = torch.linalg.qr(A)
        if torch.linalg.det(Q) < 0:
            Q[:, 2] = -Q[:, 2]
        R_mat = Q

        x_L_rot = x_L @ R_mat.T
        pocket_pos_rot = pocket_pos_centered @ R_mat.T
        h_P_rot = model.pocket_encoder(pocket_pos_rot, pocket_feat)["h_P"]

        out2 = model.egnn(x_L=x_L_rot, h_L_raw=h_L_raw, atom_types_onehot=z_type, t=t, h_P=h_P_rot)
        v2 = out2["vel_coord"]

        v_expected = v1 @ R_mat.T
        error = torch.abs(v2 - v_expected).max().item()

    return error


def train_grpo_step(
    model: FlowMatching,
    model_ref: FlowMatching,
    optimizer: torch.optim.Optimizer,
    pocket_pos: torch.Tensor,
    pocket_feat: torch.Tensor,
    reward_oracle: RewardOracle,
    pocket_path: Optional[str] = None,
    G: int = 2,
    K: int = 5,
    eps_clip: float = 0.2,
    beta: float = 0.01,
    temperature: float = 0.8,
    sigma_min: float = 0.01,
    sigma_max: float = 0.08,
    device: str = "cpu",
    max_grad_norm: float = 1.0,
) -> Dict[str, Any]:
    """
    Executes a single Group Relative Policy Optimization (GRPO) training step on one pocket.

    Parameters
    ----------
    model         : Trainable FlowMatching policy
    model_ref     : Frozen FlowMatching reference model (theta_0)
    optimizer     : PyTorch optimizer
    pocket_pos    : (N_P, 3) pocket coordinates
    pocket_feat   : (N_P, F_P) pocket features
    reward_oracle : Multi-objective chemical reward evaluator
    G             : Group size (number of stochastic rollouts per pocket)
    K             : Number of SDE integration steps
    eps_clip      : PPO/GRPO clipping threshold (default 0.2)
    beta          : Timestep-weighted transition KL penalty coefficient
    """
    from generate import coords_to_rdkit_mol

    device_obj = torch.device(device)
    pocket_pos = pocket_pos.to(device_obj)
    pocket_feat = pocket_feat.to(device_obj)

    # 1. Evaluate pocket context for rollout
    model.eval()
    with torch.no_grad():
        pocket_pos_centered = pocket_pos - pocket_pos.mean(dim=0, keepdim=True)
        pocket_out = model.pocket_encoder(pocket_pos_centered, pocket_feat)
        h_P_rollout = pocket_out["h_P"]

        # 2. Rollout G stochastic trajectories under current policy theta_old
        candidates = []
        old_probs: List[TrajectoryProbability] = []
        rewards: List[float] = []
        reward_details: List[Dict[str, Any]] = []

        for g in range(G):
            gen = model.sample(
                pocket_pos=pocket_pos,
                pocket_feat=pocket_feat,
                temperature=temperature,
                num_steps=K,
                stochastic=True,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            candidates.append(gen)

            # Evaluate exact old-policy trajectory log probability (detached)
            old_p = evaluate_trajectory_probability(
                model=model,
                trajectory_states=gen["trajectory_states"],
                trajectory_types=gen.get("trajectory_types"),
                z_type_rollout=gen["z_type_final"],
                atom_types=gen["atom_types"],
                step_sigmas=gen["step_sigmas"],
                timesteps=gen["timesteps"],
                h_P=h_P_rollout,
                temperature=temperature,
            )
            old_probs.append(old_p)

            # Score molecule with multi-objective reward oracle
            try:
                pos_np = gen["pos"].cpu().numpy()
                types_np = gen["atom_types"].cpu().numpy()
                mol, sanitized = coords_to_rdkit_mol(pos_np, types_np)

                if not sanitized or mol is None:
                    r = 0.0
                    rd = {"total_reward": 0.0, "gate_pass": False, "gate_reason": "sanitization_failed"}
                else:
                    rd = reward_oracle.compute_rl_reward(
                        mol=mol,
                        pK_pred=gen["pK_pred"],
                        pocket_path=str(pocket_path) if pocket_path else "synthetic_pocket.pdb",
                        pocket_pos_updated=gen.get("pocket_pos_updated"),
                        rl_round=0,
                    )
                    r = float(rd["total_reward"])
            except Exception as e:
                r = 0.0
                rd = {"total_reward": 0.0, "gate_pass": False, "gate_reason": f"exception: {str(e)}"}

            rewards.append(r)
            reward_details.append(rd)

    # 3. Compute group-normalized advantages
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device_obj)
    advantages = compute_group_advantages(rewards_tensor)

    # 4. Differentiable forward pass for current policy theta
    model.train()
    optimizer.zero_grad()

    h_P_train = model.pocket_encoder(pocket_pos_centered, pocket_feat)["h_P"]

    policy_losses: List[torch.Tensor] = []
    kl_losses: List[torch.Tensor] = []
    ratios: List[torch.Tensor] = []
    raw_log_ratios: List[torch.Tensor] = []

    for g in range(G):
        gen = candidates[g]
        adv_g = advantages[g]
        old_lp = old_probs[g].total_log_prob.detach()

        # Re-evaluate under trainable model and frozen reference model
        new_p = evaluate_trajectory_probability(
            model=model,
            trajectory_states=gen["trajectory_states"],
            trajectory_types=gen.get("trajectory_types"),
            z_type_rollout=gen["z_type_final"],
            atom_types=gen["atom_types"],
            step_sigmas=gen["step_sigmas"],
            timesteps=gen["timesteps"],
            h_P=h_P_train,
            model_ref=model_ref,
            temperature=temperature,
        )

        # Policy ratio r = exp(clamp(logp_new - logp_old))
        ratio_g, diag_g = compute_stabilized_ratio(new_p.total_log_prob, old_lp)
        ratios.append(ratio_g)
        raw_log_ratios.append(diag_g.raw_log_ratio)

        # PPO/GRPO clipped surrogate
        surr1 = ratio_g * adv_g
        surr2 = torch.clamp(ratio_g, 1.0 - eps_clip, 1.0 + eps_clip) * adv_g
        p_loss_g = -torch.min(surr1, surr2)
        policy_losses.append(p_loss_g)

        # Timestep-weighted transition KL
        kl_g = new_p.total_kl
        kl_losses.append(kl_g)

    mean_policy_loss = torch.stack(policy_losses).mean()
    mean_kl_loss = torch.stack(kl_losses).mean()
    total_loss = mean_policy_loss + beta * mean_kl_loss

    # 5. Backward pass and gradient instrumentation
    total_loss.backward()

    # Pre-step gradient diagnostics
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item()
    nonzero_grad_params = sum(1 for g in grads if g.abs().sum().item() > 0)
    max_grad_val = max(g.abs().max().item() for g in grads) if grads else 0.0

    # Record parameter snapshots before step
    params_before = {name: p.clone().detach() for name, p in model.named_parameters()}

    # Execute single optimizer step
    optimizer.step()

    # Post-step parameter delta diagnostics
    param_deltas = [
        torch.abs(p.detach() - params_before[name]).max().item()
        for name, p in model.named_parameters()
    ]
    max_param_delta = max(param_deltas) if param_deltas else 0.0
    mean_param_delta = sum(param_deltas) / max(len(param_deltas), 1)

    # Ratio sanity checks
    stacked_ratios = torch.stack(ratios).detach()
    stacked_raw_log_ratios = torch.stack(raw_log_ratios).detach()

    # Equivariance diagnostic
    equiv_err = compute_velocity_equivariance_diagnostic(model, pocket_pos, pocket_feat, device=device_obj)

    return {
        "step_executed": True,
        "policy_loss": mean_policy_loss.item(),
        "reference_kl": mean_kl_loss.item(),
        "total_loss": total_loss.item(),
        "rewards": rewards,
        "mean_reward": rewards_tensor.mean().item(),
        "std_reward": rewards_tensor.std(unbiased=False).item(),
        "advantages": advantages.tolist(),
        "ratios": stacked_ratios.tolist(),
        "mean_ratio": stacked_ratios.mean().item(),
        "mean_abs_ratio_minus_one": torch.abs(stacked_ratios - 1.0).mean().item(),
        "raw_log_ratios": stacked_raw_log_ratios.tolist(),
        "mean_log_ratio": stacked_raw_log_ratios.mean().item(),
        "total_grad_norm": total_grad_norm,
        "nonzero_grad_params": nonzero_grad_params,
        "max_grad_val": max_grad_val,
        "max_param_delta": max_param_delta,
        "mean_param_delta": mean_param_delta,
        "reward_details": reward_details,
        "velocity_equivariance_error": equiv_err,
    }


def rl_finetune(
    model: FlowMatching,
    pretrained_checkpoint: str,
    train_pairs: list,
    base_dir: str,
    val_pairs: list = None,
    resume_checkpoint: str = None,
    max_steps: int = 50_000,
    lr: float = 1e-5,
    batch_pockets: int = 4,
    group_size: int = 4,
    num_sde_steps: int = 20,
    kl_beta_start: float = 0.01,
    kl_beta_end: float = 0.001,
    save_dir: str = "checkpoints",
    device: str = "cpu",
):
    """
    SDE Flow-GRPO Multi-Round Fine-Tuning Loop.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize frozen reference model theta_0
    model_ref = copy.deepcopy(model)
    ref_ckpt = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    model_ref.load_state_dict(ref_ckpt["model_state_dict"], strict=False)
    model_ref = model_ref.to(device)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 2. Reward Oracle
    reward_oracle = RewardOracle(
        vina_every_n=10,
        min_carbon_ratio=0.40,
        max_nitrogen_ratio=0.35,
        max_nn_bonds=2,
        max_sa_score=6.0,
        max_ring_nitrogen=2,
    )

    beta_schedule = CosineBetaSchedule(kl_beta_start, kl_beta_end, max_steps)
    logger.info(f"Initialized SDE Flow-GRPO fine-tuning loop on device {device}.")
    return {"status": "initialized", "max_steps": max_steps}
