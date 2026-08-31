"""
src/train/sde_likelihood.py — Exact SDE Trajectory Likelihood & Policy-Ratio Engine.

Stage 6C/6D Component:
Implements exact discrete-time Gaussian transition log densities on the zero-Center-of-Mass
manifold V_CoM (dimension d = 3(N_L - 1)), reference policy evaluation, timestep-weighted
transition KL divergence, and numerically stabilized policy log-ratios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrajectoryProbability:
    """Structured container for trajectory log-probabilities and diagnostics."""
    coord_log_prob: torch.Tensor          # Sum of transition log-probabilities over K steps
    type_log_prob: torch.Tensor           # Terminal categorical log-probability over N_L atoms
    total_log_prob: torch.Tensor          # coord_log_prob + lambda_type * type_log_prob
    step_log_probs: List[torch.Tensor]    # Step-by-step transition log-probabilities (length K)
    step_kl_divs: Optional[List[torch.Tensor]] = None # Step-by-step transition KL to ref (length K)
    total_kl: Optional[torch.Tensor] = None           # Total trajectory KL divergence


@dataclass
class PolicyRatioDiagnostics:
    """Diagnostic metrics for policy log-ratio monitoring."""
    ratio: torch.Tensor                   # Stabilized importance ratio exp(clamp(log_r, -20, 20))
    raw_log_ratio: torch.Tensor           # Exact mathematical log_ratio (log_p_new - log_p_old)
    min_log_ratio: float
    max_log_ratio: float
    mean_log_ratio: float
    clamped_fraction: float               # Fraction of elements clamped by the safeguard


# ──────────────────────────────────────────────────────────────────────────────
# 1. Deterministic Orthonormal Basis for the Zero-CoM Subspace
# ──────────────────────────────────────────────────────────────────────────────

_BASIS_CACHE: Dict[int, torch.Tensor] = {}


def get_helmert_basis(N: int, device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Constructs an explicit, deterministic orthonormal basis Q_atom in R^{N x (N-1)}
    for the orthogonal complement of the all-ones vector 1_N (zero-CoM subspace).

    Properties:
        Q_atom^T Q_atom = I_{N-1}
        Q_atom^T 1_N = 0
    """
    if N < 2:
        raise ValueError(f"Helmert basis requires N >= 2, got {N}")

    cache_key = N
    if cache_key in _BASIS_CACHE:
        Q = _BASIS_CACHE[cache_key]
        if Q.device != device or Q.dtype != dtype:
            return Q.to(device=device, dtype=dtype)
        return Q

    # Analytical Helmert sub-matrix construction
    Q = torch.zeros(N, N - 1, dtype=torch.float64)
    for k in range(1, N):
        col = k - 1
        denom = math.sqrt(k * (k + 1.0))
        Q[:k, col] = 1.0 / denom
        Q[k, col] = -float(k) / denom

    Q_tensor = Q.to(dtype=dtype)
    _BASIS_CACHE[cache_key] = Q_tensor
    if device is not None:
        return Q_tensor.to(device=device)
    return Q_tensor


def project_centered(x: torch.Tensor) -> torch.Tensor:
    """Projects coordinates or velocities onto the zero-CoM subspace."""
    return x - x.mean(dim=-2, keepdim=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Exact Transition Log-Probability on Manifold V_CoM
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_transition_log_prob(
    z_next: torch.Tensor,       # (N_L, 3) next state coordinates (zero CoM)
    z_curr: torch.Tensor,       # (N_L, 3) current state coordinates (zero CoM)
    v_pred: torch.Tensor,       # (N_L, 3) predicted coordinate drift velocity
    dt: float,                  # Timestep size
    sigma: float,               # Noise scale at this step (must be > 0)
    check_com: bool = True,     # Verify zero-CoM manifold invariant
) -> torch.Tensor:
    """
    Computes the exact log probability density of the discrete SDE transition
    restricted to the zero-Center-of-Mass linear subspace V_CoM.

    The manifold dimension is d = 3 * (N_L - 1).
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")
    if dt <= 0.0:
        raise ValueError(f"dt must be strictly positive, got {dt}")

    N_L = z_curr.size(0)
    if N_L < 2:
        raise ValueError("Projected zero-CoM manifold requires at least 2 atoms.")

    if check_com:
        com_curr = torch.abs(z_curr.mean(dim=0)).max().item()
        com_next = torch.abs(z_next.mean(dim=0)).max().item()
        if com_curr > 1e-4 or com_next > 1e-4:
            raise ValueError(
                f"State coordinates not on zero-CoM manifold: curr={com_curr:.6f}, next={com_next:.6f}"
            )

    # Ensure velocity operates in the zero-CoM tangent space
    v_proj = project_centered(v_pred)

    # Residual vector on the zero-CoM manifold
    residual = z_next - z_curr - v_proj * dt

    # Exact log density on the zero-CoM subspace
    d_manifold = 3 * (N_L - 1)
    sse = (residual ** 2).sum()
    variance = (sigma ** 2) * dt
    log_p = -0.5 * (sse / variance) - 0.5 * d_manifold * math.log(2.0 * math.pi * variance)
    return log_p


# ──────────────────────────────────────────────────────────────────────────────
# 3. Timestep-Weighted Reference Transition KL Divergence
# ──────────────────────────────────────────────────────────────────────────────

def compute_transition_kl(
    v_theta: torch.Tensor,      # (N_L, 3) current policy velocity
    v_ref: torch.Tensor,        # (N_L, 3) reference policy velocity
    dt: float,                  # Timestep size
    sigma: float,               # Noise scale at this step
) -> torch.Tensor:
    """
    Computes the exact transition-level KL divergence:
        KL( p_theta(· | z_s) || p_ref(· | z_s) ) = (dt / (2 * sigma_s^2)) * ||v_theta - v_ref||_F^2
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")

    v_theta_proj = project_centered(v_theta)
    v_ref_proj = project_centered(v_ref)

    vel_diff_sq = ((v_theta_proj - v_ref_proj) ** 2).sum()
    kl = (dt / (2.0 * (sigma ** 2))) * vel_diff_sq
    return kl


# ──────────────────────────────────────────────────────────────────────────────
# 4. Trajectory Re-Evaluation Engine (Old vs New Policy)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_trajectory_probability(
    model: nn.Module,
    trajectory_states: List[torch.Tensor],       # [z_0, z_1, ..., z_K], each (N_L, 3)
    z_type_rollout: torch.Tensor,                # (N_L, 6) continuous simplex states or None
    atom_types: torch.Tensor,                    # (N_L,) sampled discrete atom types
    step_sigmas: List[float],                    # length K
    timesteps: List[float],                      # length K
    h_P: torch.Tensor,                           # (N_P, hidden_dim) pocket embedding
    trajectory_types: Optional[List[torch.Tensor]] = None, # [z_type_0, ..., z_type_K]
    model_ref: Optional[nn.Module] = None,       # Optional frozen reference model
    temperature: float = 0.8,
    lambda_type: float = 1.0,
) -> TrajectoryProbability:
    """
    Re-evaluates the exact trajectory probability under `model` given the states
    and transitions recorded during rollout.
    """
    K = len(step_sigmas)
    if len(trajectory_states) != K + 1:
        raise ValueError(f"Expected {K+1} states for {K} steps, got {len(trajectory_states)}")

    dt = 1.0 / K
    step_log_probs: List[torch.Tensor] = []
    step_kl_divs: List[torch.Tensor] = []

    coord_log_prob = torch.tensor(0.0, device=trajectory_states[0].device)
    total_kl = torch.tensor(0.0, device=trajectory_states[0].device)

    N_L = trajectory_states[0].size(0)
    h_L_raw = torch.zeros(N_L, 4, device=trajectory_states[0].device)

    # Initial simplex state
    if trajectory_types is not None:
        z_type = trajectory_types[0]
    else:
        z_type = torch.ones(N_L, 6, device=trajectory_states[0].device) / 6.0

    for s in range(K):
        t_val = timesteps[s]
        t = torch.tensor([t_val], device=trajectory_states[0].device)
        sigma_s = step_sigmas[s]

        z_s = trajectory_states[s]
        z_next = trajectory_states[s + 1]

        if trajectory_types is not None and s < len(trajectory_types):
            current_z_type = trajectory_types[s]
        else:
            current_z_type = z_type

        # Forward pass on stored state z_s
        out = model.egnn(
            x_L=z_s,
            h_L_raw=h_L_raw,
            atom_types_onehot=current_z_type,
            t=t,
            h_P=h_P,
            ligand_bonds=None,
        )
        vel_coord = out["vel_coord"]

        # Exact transition density
        step_lp = gaussian_transition_log_prob(
            z_next=z_next,
            z_curr=z_s,
            v_pred=vel_coord,
            dt=dt,
            sigma=sigma_s,
        )
        step_log_probs.append(step_lp)
        coord_log_prob = coord_log_prob + step_lp

        # Reference model evaluation if provided
        if model_ref is not None:
            with torch.no_grad():
                out_ref = model_ref.egnn(
                    x_L=z_s,
                    h_L_raw=h_L_raw,
                    atom_types_onehot=current_z_type,
                    t=t,
                    h_P=h_P,
                    ligand_bonds=None,
                )
                vel_ref = out_ref["vel_coord"]

            step_kl = compute_transition_kl(vel_coord, vel_ref, dt=dt, sigma=sigma_s)
            step_kl_divs.append(step_kl)
            total_kl = total_kl + step_kl

        # Evolve continuous atom-type simplex state if not using pre-stored types
        if trajectory_types is None:
            z_type = z_type + out["vel_type"] * dt

    # Terminal discrete atom-type log probability
    effective_temp = max(temperature, 0.1)
    if trajectory_types is not None:
        final_z_type = trajectory_types[-1]
    elif z_type_rollout is not None:
        final_z_type = z_type_rollout
    else:
        final_z_type = z_type

    type_logits = final_z_type / effective_temp
    type_log_probs_all = F.log_softmax(type_logits, dim=-1)
    atom_indices = torch.arange(N_L, device=trajectory_states[0].device)
    type_log_prob = type_log_probs_all[atom_indices, atom_types].sum()

    total_log_prob = coord_log_prob + lambda_type * type_log_prob

    return TrajectoryProbability(
        coord_log_prob=coord_log_prob,
        type_log_prob=type_log_prob,
        total_log_prob=total_log_prob,
        step_log_probs=step_log_probs,
        step_kl_divs=step_kl_divs if model_ref is not None else None,
        total_kl=total_kl if model_ref is not None else None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Numerically Stabilized Policy Ratio with Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stabilized_ratio(
    log_p_new: torch.Tensor,
    log_p_old: torch.Tensor,
    clamp_min: float = -20.0,
    clamp_max: float = 20.0,
) -> Tuple[torch.Tensor, PolicyRatioDiagnostics]:
    """
    Computes importance ratio r = exp(log_p_new - log_p_old) with numerical clamping
    and full diagnostic tracking.
    """
    raw_log_r = log_p_new - log_p_old
    clamped_log_r = torch.clamp(raw_log_r, min=clamp_min, max=clamp_max)
    ratio = torch.exp(clamped_log_r)

    is_clamped = (raw_log_r < clamp_min) | (raw_log_r > clamp_max)
    clamped_fraction = is_clamped.float().mean().item() if is_clamped.numel() > 0 else 0.0

    diag = PolicyRatioDiagnostics(
        ratio=ratio.detach(),
        raw_log_ratio=raw_log_r.detach(),
        min_log_ratio=raw_log_r.min().item(),
        max_log_ratio=raw_log_r.max().item(),
        mean_log_ratio=raw_log_r.mean().item(),
        clamped_fraction=clamped_fraction,
    )
    return ratio, diag


# ──────────────────────────────────────────────────────────────────────────────
# 6. Group Advantage Normalization & Clipped Surrogate
# ──────────────────────────────────────────────────────────────────────────────

def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes group-normalized advantages for GRPO:
        A_g = (r_g - mean(r)) / (std(r) + eps)
    If all rewards are identical (zero variance), advantages are strictly zero.
    """
    if rewards.dim() == 0 or rewards.numel() <= 1:
        return torch.zeros_like(rewards)

    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)

    if std_r < 1e-7:
        return torch.zeros_like(rewards)

    return (rewards - mean_r) / (std_r + eps)


def grpo_clipped_surrogate(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    eps_clip: float = 0.2,
) -> torch.Tensor:
    """Computes the standard PPO/GRPO clipped surrogate objective."""
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantage
    return torch.min(surr1, surr2)
