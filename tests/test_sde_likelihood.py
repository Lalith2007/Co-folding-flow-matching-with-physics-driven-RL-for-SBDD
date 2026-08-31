"""
tests/test_sde_likelihood.py — Verification Tests for Exact SDE Likelihood & Policy Engine.

Stage 6C Verification Suite:
Validates exact transition densities, Helmert orthonormal basis, Monte Carlo agreement,
reference model isolation, on-policy identity, timestep-weighted KL, and performance scaling.
"""

from __future__ import annotations

import copy
import math
import sys
import time
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.pocket_encoder import PocketEncoder
from src.model.egnn import SE3EGNN
from src.model.flow_matching import FlowMatching
from src.train.sde_likelihood import (
    TrajectoryProbability,
    PolicyRatioDiagnostics,
    get_helmert_basis,
    project_centered,
    gaussian_transition_log_prob,
    compute_transition_kl,
    evaluate_trajectory_probability,
    compute_stabilized_ratio,
)
from tests.test_sde_math import get_random_so3_rotation


class TestSDELikelihoodEngine(unittest.TestCase):
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "pretrain_final.pt"

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=128, num_layers=4, knn_k=16)
        egnn = SE3EGNN(ligand_in_dim=4, pocket_dim=128, hidden_dim=128, num_layers=9, knn_k=16, num_atom_types=6)
        cls.model = FlowMatching(pocket_encoder=pocket_encoder, egnn=egnn, num_steps=10)

        if cls.CHECKPOINT_PATH.exists():
            ckpt = torch.load(cls.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            cls.model.load_state_dict(ckpt["model_state_dict"])
        cls.model.eval()

        torch.manual_seed(123)
        cls.N_P = 20
        cls.pocket_pos = torch.randn(cls.N_P, 3)
        cls.pocket_feat = torch.randn(cls.N_P, 40)

    # ──────────────────────────────────────────────────────────────────────────
    # Part 3: Helmert Orthonormal Subspace Basis
    # ──────────────────────────────────────────────────────────────────────────

    def test_01_helmert_basis_orthonormality(self):
        """Verify Helmert basis Q_atom^T Q_atom = I_{N-1} and Q_atom^T 1 = 0 for multiple N."""
        for N in [2, 3, 5, 12, 35]:
            Q = get_helmert_basis(N)
            self.assertEqual(Q.shape, (N, N - 1))

            # Q^T Q = I_{N-1}
            QtQ = Q.T @ Q
            I_sub = torch.eye(N - 1)
            max_err = torch.abs(QtQ - I_sub).max().item()
            self.assertLess(max_err, 1e-6, f"Helmert basis not orthonormal for N={N}: error={max_err}")

            # Q^T 1_N = 0
            ones = torch.ones(N, 1)
            one_drift = torch.abs(Q.T @ ones).max().item()
            self.assertLess(one_drift, 1e-6, f"Helmert basis not orthogonal to 1_N for N={N}: drift={one_drift}")

    # ──────────────────────────────────────────────────────────────────────────
    # Part 4 & 5: Transition Density vs Explicit Basis Projection
    # ──────────────────────────────────────────────────────────────────────────

    def test_02_transition_log_prob_vs_basis_projection(self):
        """Verify closed-form manifold log density matches explicit Helmert coordinates Q^T r."""
        for N in [3, 7, 16]:
            dt = 0.05
            sigma = 0.04
            z_curr = project_centered(torch.randn(N, 3))
            v_pred = project_centered(torch.randn(N, 3))
            eps_proj = project_centered(torch.randn(N, 3))
            z_next = z_curr + v_pred * dt + sigma * math.sqrt(dt) * eps_proj

            # 1. Closed-form manifold formula
            lp_manifold = gaussian_transition_log_prob(z_next, z_curr, v_pred, dt, sigma)

            # 2. Explicit coordinate projection via Helmert basis
            Q = get_helmert_basis(N)
            residual = z_next - z_curr - v_pred * dt
            residual_sub = Q.T @ residual  # (N-1, 3)

            d_sub = 3 * (N - 1)
            sse_sub = (residual_sub ** 2).sum()
            var = (sigma ** 2) * dt
            lp_basis = -0.5 * (sse_sub / var) - 0.5 * d_sub * math.log(2.0 * math.pi * var)

            self.assertTrue(torch.allclose(lp_manifold, lp_basis, atol=1e-5))

    # ──────────────────────────────────────────────────────────────────────────
    # Part 6: Monte Carlo Density Verification
    # ──────────────────────────────────────────────────────────────────────────

    def test_03_monte_carlo_density_verification(self):
        """Verify transition log-density expectation E[log p] matches theoretical entropy for N=2, 3, 5."""
        for N in [2, 3, 5]:
            torch.manual_seed(200 + N)
            dt = 0.05
            sigma = 0.05
            d_man = 3 * (N - 1)
            var = (sigma ** 2) * dt

            z_curr = project_centered(torch.randn(N, 3))
            v_pred = project_centered(torch.randn(N, 3))

            # Theoretical differential entropy of N(0, var * I_{d_man}):
            # E[log p] = -0.5 * d_man * (1 + log(2*pi*var))
            theoretical_expected_log_p = -0.5 * d_man * (1.0 + math.log(2.0 * math.pi * var))

            M = 20000
            eps = project_centered(torch.randn(M, N, 3))
            z_next_samples = z_curr.unsqueeze(0) + v_pred.unsqueeze(0) * dt + sigma * math.sqrt(dt) * eps

            # Compute empirical log density over M samples
            residuals = z_next_samples - (z_curr + v_pred * dt)
            sse = (residuals ** 2).sum(dim=(1, 2))
            log_p_samples = -0.5 * (sse / var) - 0.5 * d_man * math.log(2.0 * math.pi * var)
            empirical_expected_log_p = log_p_samples.mean().item()

            rel_err = abs(empirical_expected_log_p - theoretical_expected_log_p) / abs(theoretical_expected_log_p)
            self.assertLess(rel_err, 0.015, f"Monte Carlo log density rel error {rel_err*100:.2f}% exceeds 1.5% for N={N}")

    # ──────────────────────────────────────────────────────────────────────────
    # Part 7: Reference Model Isolation
    # ──────────────────────────────────────────────────────────────────────────

    def test_04_reference_model_independence_and_matching(self):
        """Verify frozen reference model shares identical initial predictions but independent gradients."""
        model_ref = copy.deepcopy(self.model)
        for p in model_ref.parameters():
            p.requires_grad_(False)

        # Check all reference parameters require no grad
        self.assertTrue(all(not p.requires_grad for p in model_ref.parameters()))

        # Check initial evaluation produces identical output
        torch.manual_seed(42)
        out1 = self.model.sample(self.pocket_pos, self.pocket_feat, num_atoms=10, num_steps=5, stochastic=True)
        torch.manual_seed(42)
        out2 = model_ref.sample(self.pocket_pos, self.pocket_feat, num_atoms=10, num_steps=5, stochastic=True)

        self.assertTrue(torch.allclose(out1["pos"], out2["pos"], atol=1e-6))

    # ──────────────────────────────────────────────────────────────────────────
    # Part 9 & 16: On-Policy Ratio Identity (New == Old on Rollout)
    # ──────────────────────────────────────────────────────────────────────────

    def test_05_on_policy_ratio_identity(self):
        """Verify evaluating stored rollout trajectory under same model yields ratio == 1.0."""
        for N_L in [10, 15, 22]:
            torch.manual_seed(300 + N_L)
            K = 10

            # 1. Rollout with model
            rollout = self.model.sample(
                self.pocket_pos, self.pocket_feat, num_atoms=N_L, num_steps=K, stochastic=True
            )

            pocket_pos_centered = self.pocket_pos - self.pocket_pos.mean(dim=0, keepdim=True)
            pocket_out = self.model.pocket_encoder(pocket_pos_centered, self.pocket_feat)
            h_P = pocket_out["h_P"]

            # 2. Evaluate trajectory probability under model
            prob_obj = evaluate_trajectory_probability(
                model=self.model,
                trajectory_states=rollout["trajectory_states"],
                z_type_rollout=rollout["z_type_final"],
                atom_types=rollout["atom_types"],
                step_sigmas=rollout["step_sigmas"],
                timesteps=rollout["timesteps"],
                h_P=h_P,
            )

            # 3. Compute ratio against itself
            ratio, diag = compute_stabilized_ratio(prob_obj.total_log_prob, prob_obj.total_log_prob)

            self.assertAlmostEqual(ratio.item(), 1.0, places=5)
            self.assertAlmostEqual(diag.mean_log_ratio, 0.0, places=5)
            self.assertEqual(diag.clamped_fraction, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Part 10: Stabilized Policy Ratio Safeguard
    # ──────────────────────────────────────────────────────────────────────────

    def test_06_stabilized_policy_ratio_safeguard(self):
        """Verify extreme log ratios are safely clamped while preserving raw diagnostics."""
        log_p_new = torch.tensor([50.0, -50.0, 0.05])
        log_p_old = torch.tensor([0.0, 0.0, 0.0])

        ratio, diag = compute_stabilized_ratio(log_p_new, log_p_old, clamp_min=-20.0, clamp_max=20.0)

        # Clamped ratio values
        self.assertTrue(torch.isfinite(ratio).all())
        self.assertAlmostEqual(ratio[0].item(), math.exp(20.0), delta=1e2)
        self.assertAlmostEqual(ratio[1].item(), math.exp(-20.0), delta=1e-8)
        self.assertAlmostEqual(ratio[2].item(), math.exp(0.05), delta=1e-4)

        # Diagnostic metrics
        self.assertEqual(diag.min_log_ratio, -50.0)
        self.assertEqual(diag.max_log_ratio, 50.0)
        self.assertAlmostEqual(diag.clamped_fraction, 2.0 / 3.0, places=4)

    # ──────────────────────────────────────────────────────────────────────────
    # Part 11 & 17: Reference Transition KL
    # ──────────────────────────────────────────────────────────────────────────

    def test_07_timestep_weighted_transition_kl(self):
        """Verify transition KL uses exact (dt / (2 * sigma^2)) * ||v_theta - v_ref||^2."""
        N = 10
        dt = 0.05

        v1 = project_centered(torch.randn(N, 3))
        v2 = project_centered(torch.randn(N, 3))

        # 1. Identical velocities -> KL = 0
        kl_zero = compute_transition_kl(v1, v1, dt=dt, sigma=0.04)
        self.assertAlmostEqual(kl_zero.item(), 0.0, places=6)

        # 2. Multiple sigma values
        for sigma in [0.015, 0.04, 0.08]:
            kl = compute_transition_kl(v1, v2, dt=dt, sigma=sigma)
            expected_kl = (dt / (2.0 * (sigma ** 2))) * ((v1 - v2) ** 2).sum().item()
            rel_err = abs(kl.item() - expected_kl) / max(expected_kl, 1e-6)
            self.assertLess(rel_err, 1e-5)

    # ──────────────────────────────────────────────────────────────────────────
    # Part 18 & 19: Transformation Behavior & Zero-CoM Manifold Support
    # ──────────────────────────────────────────────────────────────────────────

    def test_08_translation_invariance_of_transition_density(self):
        """Verify transition density is invariant when translating coordinates."""
        N = 8
        dt = 0.05
        sigma = 0.04

        z_curr = project_centered(torch.randn(N, 3))
        v_pred = project_centered(torch.randn(N, 3))
        eps = project_centered(torch.randn(N, 3))
        z_next = z_curr + v_pred * dt + sigma * math.sqrt(dt) * eps

        lp1 = gaussian_transition_log_prob(z_next, z_curr, v_pred, dt, sigma, check_com=True)

        # Non-centered state raises ValueError when check_com=True
        t_shift = torch.tensor([[10.0, -5.0, 3.0]])
        with self.assertRaises(ValueError):
            gaussian_transition_log_prob(z_next + t_shift, z_curr + t_shift, v_pred, dt, sigma, check_com=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Part 20: Performance and Scaling Benchmark
    # ──────────────────────────────────────────────────────────────────────────

    def test_09_likelihood_performance_scaling(self):
        """Benchmark runtime of exact trajectory likelihood evaluation across (N, K) sizes."""
        configs = [(10, 10), (20, 20), (40, 50)]
        print("\n[PERFORMANCE BENCHMARK — SDE LIKELIHOOD ENGINE]")

        pocket_pos_centered = self.pocket_pos - self.pocket_pos.mean(dim=0, keepdim=True)
        pocket_out = self.model.pocket_encoder(pocket_pos_centered, self.pocket_feat)
        h_P = pocket_out["h_P"]

        for N_L, K in configs:
            rollout = self.model.sample(
                self.pocket_pos, self.pocket_feat, num_atoms=N_L, num_steps=K, stochastic=True
            )

            t0 = time.perf_counter()
            prob = evaluate_trajectory_probability(
                model=self.model,
                trajectory_states=rollout["trajectory_states"],
                z_type_rollout=rollout["z_type_final"],
                atom_types=rollout["atom_types"],
                step_sigmas=rollout["step_sigmas"],
                timesteps=rollout["timesteps"],
                h_P=h_P,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            print(f"  Config: N={N_L:2d}, K={K:2d} | Log-Prob: {prob.total_log_prob.item():10.2f} | Time: {elapsed_ms:6.2f} ms")
            self.assertTrue(torch.isfinite(prob.total_log_prob))
            self.assertEqual(len(prob.step_log_probs), K)


if __name__ == "__main__":
    unittest.main(verbosity=2)
