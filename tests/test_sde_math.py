"""
tests/test_sde_math.py — Mathematical Verification and Unit Test Suite for PROTEUS.

Stage 6A Checkpoint:
Validates all mathematical foundations, symmetry invariants, manifold probability densities,
and stability safeguards BEFORE any production code is modified.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.pocket_encoder import PocketEncoder
from src.model.egnn import SE3EGNN
from src.model.flow_matching import FlowMatching


# ──────────────────────────────────────────────────────────────────────────────
# Mathematical Helper Functions (Standalone implementations for verification)
# ──────────────────────────────────────────────────────────────────────────────

def project_centered(eps: torch.Tensor) -> torch.Tensor:
    """Project coordinate noise onto the zero-Center-of-Mass linear subspace."""
    return eps - eps.mean(dim=0, keepdim=True)


def projected_gaussian_log_density(
    z_next: torch.Tensor,
    z_curr: torch.Tensor,
    v_pred: torch.Tensor,
    dt: float,
    sigma: float,
) -> torch.Tensor:
    """
    Computes the exact log probability density of a Gaussian transition on the
    zero-Center-of-Mass manifold V_CoM of dimension d = 3(N - 1).
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")
    if dt <= 0.0:
        raise ValueError(f"dt must be strictly positive, got {dt}")

    N = z_curr.size(0)
    if N < 2:
        raise ValueError("Projected zero-CoM manifold requires at least 2 atoms.")

    d_manifold = 3 * (N - 1)
    residual = z_next - z_curr - v_pred * dt

    sse = (residual ** 2).sum()
    variance = (sigma ** 2) * dt
    log_p = -0.5 * (sse / variance) - 0.5 * d_manifold * math.log(2.0 * math.pi * variance)
    return log_p


def compute_stabilized_ratio(
    log_p_new: torch.Tensor,
    log_p_old: torch.Tensor,
    clamp_min: float = -20.0,
    clamp_max: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes importance sampling ratio with clamping."""
    log_ratio_raw = log_p_new - log_p_old
    log_ratio_clamped = torch.clamp(log_ratio_raw, min=clamp_min, max=clamp_max)
    ratio = torch.exp(log_ratio_clamped)
    return ratio, log_ratio_raw


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes group-normalized advantages for GRPO."""
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
    """Computes the PPO/GRPO clipped surrogate objective."""
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantage
    return torch.min(surr1, surr2)


def get_random_so3_rotation(seed: int = None) -> torch.Tensor:
    """Generates a random proper rotation matrix R in SO(3) with det(R) = +1."""
    if seed is not None:
        torch.manual_seed(seed)
    A = torch.randn(3, 3)
    Q, R = torch.linalg.qr(A)
    det = torch.linalg.det(Q)
    if det < 0:
        Q[:, 2] = -Q[:, 2]
    return Q


# ──────────────────────────────────────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────────────────────────────────────

class TestSDEMathematics(unittest.TestCase):
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "pretrain_final.pt"

    def test_01_projected_centered_noise_sum(self):
        """PART 2: Verify that projected noise sums to exactly zero across atoms."""
        for N in [2, 5, 10, 32]:
            torch.manual_seed(100 + N)
            eps = torch.randn(N, 3)
            eps_proj = project_centered(eps)
            max_drift = torch.abs(eps_proj.sum(dim=0)).max().item()
            # Standard single-precision float32 sum tolerance across N particles
            self.assertLess(max_drift, 1e-5, f"CoM drift {max_drift} exceeds 1e-5 for N={N}")

    def test_02_projected_noise_subspace_dimension(self):
        """PART 3: Verify that the zero-CoM projection matrix has rank exactly 3*(N - 1)."""
        for N in [2, 5, 10, 20]:
            P_com = torch.eye(N) - (1.0 / N) * torch.ones(N, N)
            I_3 = torch.eye(3)
            M = torch.kron(P_com, I_3)

            singular_values = torch.linalg.svdvals(M)
            rank = (singular_values > 1e-5).sum().item()
            expected_rank = 3 * (N - 1)
            self.assertEqual(rank, expected_rank, f"Rank {rank} != {expected_rank} for N={N}")

    def test_03_rotation_equivariance_egnn_coordinate_layers(self):
        """PART 4: Verify that x_L_updated(R x, R P) = (x_L_updated(x, P)) R^T for R in SO(3)."""
        torch.manual_seed(42)
        pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=64, num_layers=2, knn_k=8)
        egnn = SE3EGNN(ligand_in_dim=4, pocket_dim=64, hidden_dim=64, num_layers=3, knn_k=8, num_atom_types=6)
        pocket_encoder.eval()
        egnn.eval()

        torch.manual_seed(123)
        N_L, N_P = 12, 24
        x_L = torch.randn(N_L, 3)
        x_L = x_L - x_L.mean(dim=0, keepdim=True)
        h_L_raw = torch.randn(N_L, 4)
        z_type = F.softmax(torch.randn(N_L, 6), dim=-1)
        t = torch.tensor([0.4])

        x_P = torch.randn(N_P, 3)
        x_P = x_P - x_P.mean(dim=0, keepdim=True)
        h_P_raw = torch.randn(N_P, 40)

        # 1. Unrotated forward pass
        with torch.no_grad():
            h_P = pocket_encoder(x_P, h_P_raw)["h_P"]
            out1 = egnn(x_L=x_L, h_L_raw=h_L_raw, atom_types_onehot=z_type, t=t, h_P=h_P)
            x_L_updated1 = out1["x_L_updated"]
            vel_type1 = out1["vel_type"]

        # 2. Sample proper rotation R in SO(3)
        R = get_random_so3_rotation(seed=999)
        self.assertAlmostEqual(torch.linalg.det(R).item(), 1.0, places=5)

        x_L_rot = x_L @ R.T
        x_P_rot = x_P @ R.T

        # 3. Rotated forward pass
        with torch.no_grad():
            h_P_rot = pocket_encoder(x_P_rot, h_P_raw)["h_P"]
            out2 = egnn(x_L=x_L_rot, h_L_raw=h_L_raw, atom_types_onehot=z_type, t=t, h_P=h_P_rot)
            x_L_updated2 = out2["x_L_updated"]
            vel_type2 = out2["vel_type"]

        x_L_expected = x_L_updated1 @ R.T

        # Verify 9-layer EGNN coordinate update is exactly rotation-equivariant
        coord_diff = torch.abs(x_L_updated2 - x_L_expected).max().item()
        self.assertLess(coord_diff, 1e-4, f"Layer coordinate rotation equivariance error: {coord_diff}")

        # Verify scalar/categorical outputs are rotation-invariant
        type_diff = torch.abs(vel_type2 - vel_type1).max().item()
        self.assertLess(type_diff, 1e-4, f"Type invariant error: {type_diff}")

    def test_04_translation_equivariance_deterministic_velocity(self):
        """PART 5: Verify that x_L_updated(x + t, P + t) = x_L_updated(x, P) + t."""
        torch.manual_seed(42)
        pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=64, num_layers=2, knn_k=8)
        egnn = SE3EGNN(ligand_in_dim=4, pocket_dim=64, hidden_dim=64, num_layers=3, knn_k=8, num_atom_types=6)
        pocket_encoder.eval()
        egnn.eval()

        torch.manual_seed(456)
        N_L, N_P = 10, 20
        x_L = torch.randn(N_L, 3)
        x_L = x_L - x_L.mean(dim=0, keepdim=True)
        h_L_raw = torch.randn(N_L, 4)
        z_type = F.softmax(torch.randn(N_L, 6), dim=-1)
        t = torch.tensor([0.5])

        x_P = torch.randn(N_P, 3)
        x_P = x_P - x_P.mean(dim=0, keepdim=True)
        h_P_raw = torch.randn(N_P, 40)

        with torch.no_grad():
            h_P1 = pocket_encoder(x_P, h_P_raw)["h_P"]
            out1 = egnn(x_L=x_L, h_L_raw=h_L_raw, atom_types_onehot=z_type, t=t, h_P=h_P1)
            x_L_updated1 = out1["x_L_updated"]

        t_offset = torch.tensor([[150.0, -80.0, 42.0]])
        x_L_trans = x_L + t_offset
        x_P_trans = x_P + t_offset

        with torch.no_grad():
            x_P_trans_centered = x_P_trans - x_P_trans.mean(dim=0, keepdim=True)
            h_P2 = pocket_encoder(x_P_trans_centered, h_P_raw)["h_P"]
            out2 = egnn(x_L=x_L_trans, h_L_raw=h_L_raw, atom_types_onehot=z_type, t=t, h_P=h_P2)
            x_L_updated2 = out2["x_L_updated"]

        # Translated coordinates should translate identically
        trans_diff = torch.abs(x_L_updated2 - (x_L_updated1 + t_offset)).max().item()
        self.assertLess(trans_diff, 1e-4, f"Translation equivariance error: {trans_diff}")

    def test_05_cross_product_parity_and_rotation(self):
        """PART 6: Verify reflection parity: cross(P v_i, P v_j) = - P cross(v_i, v_j)."""
        torch.manual_seed(789)
        N = 8
        x = torch.randn(N, 3)
        v = x - x.mean(dim=0, keepdim=True)

        i, j = 1, 4
        v_i, v_j = v[i], v[j]
        c_orig = torch.cross(v_i, v_j, dim=-1)

        # 1. Proper Rotation R in SO(3)
        R = get_random_so3_rotation(seed=111)
        v_i_R = v_i @ R.T
        v_j_R = v_j @ R.T
        c_R = torch.cross(v_i_R, v_j_R, dim=-1)
        c_R_expected = c_orig @ R.T
        self.assertTrue(torch.allclose(c_R, c_R_expected, atol=1e-5))

        # 2. Reflection Matrix P with det(P) = -1
        P = R @ torch.diag(torch.tensor([1.0, 1.0, -1.0]))
        self.assertAlmostEqual(torch.linalg.det(P).item(), -1.0, places=5)

        v_i_P = v_i @ P.T
        v_j_P = v_j @ P.T
        c_P = torch.cross(v_i_P, v_j_P, dim=-1)
        c_P_expected = -(c_orig @ P.T)
        self.assertTrue(torch.allclose(c_P, c_P_expected, atol=1e-5))

    def test_06_manifold_density_vs_orthonormal_basis(self):
        """PART 7: Verify manifold Gaussian log density equals orthonormal basis projection."""
        for N in [3, 8, 15]:
            torch.manual_seed(333 + N)
            dt = 0.05
            sigma = 0.08

            z_curr = project_centered(torch.randn(N, 3))
            v_pred = project_centered(torch.randn(N, 3))
            eps_proj = project_centered(torch.randn(N, 3))
            z_next = z_curr + v_pred * dt + sigma * math.sqrt(dt) * eps_proj

            # 1. Closed-form formula on manifold
            log_p_manifold = projected_gaussian_log_density(z_next, z_curr, v_pred, dt, sigma)

            # 2. Explicit Orthonormal Basis Q in R^{N x (N-1)}
            P_com = torch.eye(N) - (1.0 / N) * torch.ones(N, N)
            U, S, Vh = torch.linalg.svd(P_com)
            Q = U[:, :N-1]  # (N, N-1)

            residual = z_next - z_curr - v_pred * dt
            residual_sub = Q.T @ residual  # ((N-1), 3)

            d_sub = 3 * (N - 1)
            sse_sub = (residual_sub ** 2).sum()
            var = (sigma ** 2) * dt
            log_p_basis = -0.5 * (sse_sub / var) - 0.5 * d_sub * math.log(2.0 * math.pi * var)

            self.assertTrue(
                torch.allclose(log_p_manifold, log_p_basis, atol=1e-5),
                f"Manifold {log_p_manifold.item()} != Basis {log_p_basis.item()}"
            )

    def test_07_kl_divergence_monte_carlo_agreement(self):
        """PART 8: Verify analytical transition KL equals Monte Carlo expectation."""
        torch.manual_seed(444)
        N = 5
        dt = 0.05
        sigma = 0.08

        z_curr = project_centered(torch.randn(N, 3))
        v_theta = project_centered(torch.randn(N, 3))
        v_ref = project_centered(torch.randn(N, 3))

        vel_diff_sq = ((v_theta - v_ref) ** 2).sum().item()
        kl_analytical = (dt / (2.0 * (sigma ** 2))) * vel_diff_sq

        M = 50000
        eps = project_centered(torch.randn(M, N, 3))
        z_next_samples = z_curr.unsqueeze(0) + v_theta.unsqueeze(0) * dt + sigma * math.sqrt(dt) * eps

        var = (sigma ** 2) * dt
        res_theta = z_next_samples - (z_curr + v_theta * dt)
        res_ref = z_next_samples - (z_curr + v_ref * dt)

        sse_theta = (res_theta ** 2).sum(dim=(1, 2))
        sse_ref = (res_ref ** 2).sum(dim=(1, 2))

        log_diff = 0.5 * (sse_ref - sse_theta) / var
        kl_monte_carlo = log_diff.mean().item()

        rel_error = abs(kl_monte_carlo - kl_analytical) / kl_analytical
        self.assertLess(rel_error, 0.02, f"Rel error {rel_error*100:.2f}% exceeds 2%")

    def test_08_zero_sigma_raises_error(self):
        """PART 9: Verify sigma <= 0 raises ValueError."""
        z = torch.zeros(4, 3)
        v = torch.zeros(4, 3)
        with self.assertRaises(ValueError):
            projected_gaussian_log_density(z, z, v, dt=0.02, sigma=0.0)

        with self.assertRaises(ValueError):
            projected_gaussian_log_density(z, z, v, dt=0.02, sigma=-0.05)

    def test_09_log_ratio_clamping_prevents_overflow(self):
        """PART 10: Verify extreme log probabilities are safely clamped."""
        log_p_new = torch.tensor([1000.0, -1000.0, 0.0])
        log_p_old = torch.tensor([0.0, 0.0, 0.0])

        ratio, raw_log_r = compute_stabilized_ratio(log_p_new, log_p_old, clamp_min=-20.0, clamp_max=20.0)
        self.assertTrue(torch.isfinite(ratio).all())
        # Float32 precision relative check for exp(20) ~ 4.85e8
        rel_diff0 = abs(ratio[0].item() - math.exp(20.0)) / math.exp(20.0)
        self.assertLess(rel_diff0, 1e-4)
        rel_diff1 = abs(ratio[1].item() - math.exp(-20.0)) / math.exp(-20.0)
        self.assertLess(rel_diff1, 1e-4)
        self.assertAlmostEqual(ratio[2].item(), 1.0, delta=1e-6)

    def test_10_group_advantages_scenarios(self):
        """PART 11: Verify group advantages across all edge cases."""
        # 1. Non-zero variance
        r1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        a1 = compute_group_advantages(r1)
        self.assertAlmostEqual(a1.mean().item(), 0.0, places=5)
        self.assertTrue(a1[0] < a1[1] < a1[2] < a1[3])

        # 2. Equal rewards
        r2 = torch.tensor([3.5, 3.5, 3.5, 3.5])
        a2 = compute_group_advantages(r2)
        self.assertTrue(torch.allclose(a2, torch.zeros(4)))

        # 3. All hard-gate failures
        r3 = torch.tensor([0.0, 0.0, 0.0, 0.0])
        a3 = compute_group_advantages(r3)
        self.assertTrue(torch.allclose(a3, torch.zeros(4)))

        # 4. Mixed valid and failed
        r4 = torch.tensor([0.0, 0.0, 2.0, 4.0])
        a4 = compute_group_advantages(r4)
        self.assertEqual(a4[0].item(), a4[1].item())
        self.assertTrue(a4[0] < 0.0 < a4[2] < a4[3])

    def test_11_grpo_clipping_logic(self):
        """PART 12: Verify GRPO clipped surrogate logic."""
        eps = 0.2
        A_pos = torch.tensor(2.0)
        self.assertTrue(torch.allclose(grpo_clipped_surrogate(torch.tensor(1.1), A_pos, eps), torch.tensor(2.2)))
        self.assertTrue(torch.allclose(grpo_clipped_surrogate(torch.tensor(1.5), A_pos, eps), torch.tensor(2.4)))

        A_neg = torch.tensor(-2.0)
        self.assertTrue(torch.allclose(grpo_clipped_surrogate(torch.tensor(0.9), A_neg, eps), torch.tensor(-1.8)))
        self.assertTrue(torch.allclose(grpo_clipped_surrogate(torch.tensor(0.5), A_neg, eps), torch.tensor(-1.6)))

    def test_12_identical_policy_gives_ratio_one(self):
        """PART 13: Verify that on-policy re-evaluation gives log_ratio = 0 -> r = 1."""
        torch.manual_seed(555)
        N = 6
        dt = 0.05
        sigma = 0.08

        z_curr = project_centered(torch.randn(N, 3))
        v_pred = project_centered(torch.randn(N, 3))
        eps = project_centered(torch.randn(N, 3))
        z_next = z_curr + v_pred * dt + sigma * math.sqrt(dt) * eps

        log_p_old = projected_gaussian_log_density(z_next, z_curr, v_pred, dt, sigma)
        log_p_new = projected_gaussian_log_density(z_next, z_curr, v_pred, dt, sigma)

        ratio, raw_diff = compute_stabilized_ratio(log_p_new, log_p_old)
        self.assertLess(abs(raw_diff.item()), 1e-6)
        self.assertLess(abs(ratio.item() - 1.0), 1e-6)

    def test_13_checkpoint_integrity(self):
        """PART 15: Verify pretrain_final.pt exists, loads cleanly, and is intact."""
        self.assertTrue(self.CHECKPOINT_PATH.exists(), f"Checkpoint not found at {self.CHECKPOINT_PATH}")

        ckpt = torch.load(self.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt)

        state_dict = ckpt["model_state_dict"]
        self.assertIn("egnn.layers.0.cross_attn.q_proj.weight", state_dict)
        self.assertTrue(any("coord_cross_mlp" in k for k in state_dict))
        self.assertIn("pocket_encoder.layers.0.edge_mlp.0.weight", state_dict)

        total_params = sum(p.numel() for p in state_dict.values())
        self.assertGreater(total_params, 1_000_000)

    def test_14_deterministic_sampler_repeatability(self):
        """PART 14: Verify deterministic sampling is 100% reproducible given random seed."""
        if not self.CHECKPOINT_PATH.exists():
            self.skipTest("Checkpoint not available for deterministic baseline test.")

        torch.manual_seed(42)
        pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=128, num_layers=4, knn_k=16)
        egnn = SE3EGNN(ligand_in_dim=4, pocket_dim=128, hidden_dim=128, num_layers=9, knn_k=16, num_atom_types=6)
        model = FlowMatching(pocket_encoder=pocket_encoder, egnn=egnn, num_steps=10)

        ckpt = torch.load(self.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        N_P = 30
        pocket_pos = torch.randn(N_P, 3)
        pocket_feat = torch.randn(N_P, 40)

        torch.manual_seed(100)
        out1 = model.sample(pocket_pos, pocket_feat, num_atoms=15, num_steps=10)

        torch.manual_seed(100)
        out2 = model.sample(pocket_pos, pocket_feat, num_atoms=15, num_steps=10)

        self.assertTrue(torch.allclose(out1["pos"], out2["pos"], atol=1e-6))
        self.assertTrue(torch.equal(out1["atom_types"], out2["atom_types"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
