"""
tests/test_grpo_smoke.py — Verification Suite for SDE Flow-GRPO Smoke Test.

Stage 6D Component:
Validates end-to-end execution of a single SDE Flow-GRPO training step,
verifying trajectory rollouts, reward evaluation, advantage normalization,
on-policy ratio identity, reference KL regularization, gradient flow,
and parameter update isolation.
"""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.pocket_encoder import PocketEncoder
from src.model.egnn import SE3EGNN
from src.model.flow_matching import FlowMatching
from src.model.reward import RewardOracle
from src.train.rl_finetune import train_grpo_step, compute_velocity_equivariance_diagnostic
from src.train.sde_likelihood import compute_group_advantages


class TestGRPOSmoke(unittest.TestCase):
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "pretrain_final.pt"

    @classmethod
    def setUpClass(cls):
        """Instantiate trainable model and frozen reference model."""
        torch.manual_seed(42)
        pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=128, num_layers=4, knn_k=16)
        egnn = SE3EGNN(ligand_in_dim=4, pocket_dim=128, hidden_dim=128, num_layers=9, knn_k=16, num_atom_types=6)
        cls.model = FlowMatching(pocket_encoder=pocket_encoder, egnn=egnn, num_steps=10)

        if cls.CHECKPOINT_PATH.exists():
            ckpt = torch.load(cls.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            cls.model.load_state_dict(ckpt["model_state_dict"])

        cls.model_ref = copy.deepcopy(cls.model)
        for p in cls.model_ref.parameters():
            p.requires_grad_(False)
        cls.model_ref.eval()

        cls.reward_oracle = RewardOracle(
            vina_every_n=1,
            min_carbon_ratio=0.40,
            max_nitrogen_ratio=0.35,
            max_nn_bonds=2,
            max_sa_score=6.0,
            max_ring_nitrogen=2,
        )

        torch.manual_seed(777)
        cls.N_P = 20
        cls.pocket_pos = torch.randn(cls.N_P, 3)
        cls.pocket_feat = torch.randn(cls.N_P, 40)

    def test_01_trajectory_generation_and_shapes(self):
        """Test 1 & 2: Verify G=2 trajectory generation and tensor shapes."""
        G = 2
        K = 5
        torch.manual_seed(101)
        trajectories = []
        for _ in range(G):
            gen = self.model.sample(
                self.pocket_pos, self.pocket_feat, num_atoms=12, num_steps=K, stochastic=True
            )
            trajectories.append(gen)

        self.assertEqual(len(trajectories), G)
        for gen in trajectories:
            self.assertEqual(len(gen["trajectory_states"]), K + 1)
            self.assertEqual(len(gen["trajectory_types"]), K + 1)
            self.assertEqual(len(gen["step_sigmas"]), K)
            self.assertEqual(len(gen["timesteps"]), K)
            for state in gen["trajectory_states"]:
                self.assertEqual(state.shape, (12, 3))
                self.assertFalse(state.requires_grad)

    def test_02_group_advantages_zero_variance(self):
        """Test 6: Verify equal or all-zero rewards produce strictly zero advantages."""
        r_equal = torch.tensor([0.0, 0.0])
        adv = compute_group_advantages(r_equal)
        self.assertTrue(torch.allclose(adv, torch.zeros(2)))

        r_diff = torch.tensor([1.0, 3.0])
        adv_diff = compute_group_advantages(r_diff)
        self.assertAlmostEqual(adv_diff[0].item(), -1.0, places=4)
        self.assertAlmostEqual(adv_diff[1].item(), 1.0, places=4)

    def test_03_end_to_end_single_grpo_step(self):
        """
        Test 3-10: Execute a complete single-step GRPO update and verify all diagnostics:
        - finite log probs
        - on-policy ratio ≈ 1.0
        - initial reference KL ≈ 0.0
        - finite loss and gradients
        - model parameter modification
        - reference parameter immutability
        """
        torch.manual_seed(999)
        model_copy = copy.deepcopy(self.model)
        model_ref_copy = copy.deepcopy(self.model_ref)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=1e-4)

        # Snapshot reference parameters
        ref_params_before = {name: p.clone().detach() for name, p in model_ref_copy.named_parameters()}

        # Run single GRPO training step
        step_diag = train_grpo_step(
            model=model_copy,
            model_ref=model_ref_copy,
            optimizer=optimizer,
            pocket_pos=self.pocket_pos,
            pocket_feat=self.pocket_feat,
            reward_oracle=self.reward_oracle,
            G=2,
            K=5,
            eps_clip=0.2,
            beta=0.01,
            device="cpu",
        )

        print("\n[GRPO SMOKE TEST DIAGNOSTICS]")
        for k, v in step_diag.items():
            if k != "reward_details":
                print(f"  {k:30s}: {v}")

        # Test 3: Losses are finite
        self.assertTrue(math.isfinite(step_diag["policy_loss"]))
        self.assertTrue(math.isfinite(step_diag["reference_kl"]))
        self.assertTrue(math.isfinite(step_diag["total_loss"]))

        # Test 4: On-policy ratio sanity (|ratio - 1| < 1e-5)
        self.assertLess(
            step_diag["mean_abs_ratio_minus_one"],
            1e-5,
            f"On-policy ratio mismatch: {step_diag['mean_abs_ratio_minus_one']}",
        )

        # Test 5: Reference KL sanity (KL ≈ 0 before update)
        self.assertLess(
            step_diag["reference_kl"],
            1e-4,
            f"Initial reference KL is not approximately zero: {step_diag['reference_kl']}",
        )

        # Test 8: Finite gradients
        self.assertGreater(step_diag["nonzero_grad_params"], 0)
        self.assertTrue(math.isfinite(step_diag["total_grad_norm"]))

        # Test 9: Model parameters modified if gradient flowed
        if step_diag["total_grad_norm"] > 0:
            self.assertGreater(step_diag["max_param_delta"], 0.0)

        # Test 10: Reference model parameters remained strictly identical
        for name, p in model_ref_copy.named_parameters():
            diff = torch.abs(p - ref_params_before[name]).max().item()
            self.assertEqual(diff, 0.0, f"Reference parameter {name} changed during training step!")

    def test_04_deterministic_sampler_unaffected(self):
        """Test 11: Verify deterministic sampler remains 100% reproducible."""
        torch.manual_seed(555)
        out1 = self.model.sample(self.pocket_pos, self.pocket_feat, num_atoms=10, num_steps=5, stochastic=False)

        torch.manual_seed(555)
        out2 = self.model.sample(self.pocket_pos, self.pocket_feat, num_atoms=10, num_steps=5, stochastic=False)

        self.assertTrue(torch.allclose(out1["pos"], out2["pos"], atol=1e-6))
        self.assertTrue(torch.equal(out1["atom_types"], out2["atom_types"]))

    def test_05_velocity_equivariance_diagnostic_reporting(self):
        """Test 12: Ensure rotation equivariance error is explicitly diagnosed and reported."""
        err = compute_velocity_equivariance_diagnostic(self.model, self.pocket_pos, self.pocket_feat)
        self.assertTrue(math.isfinite(err))
        print(f"\n[EQUIVARIANCE DIAGNOSTIC] vel_coord rotation error: {err:.6f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
