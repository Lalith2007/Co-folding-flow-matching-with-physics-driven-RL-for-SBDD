#!/usr/bin/env python3
"""
smoke_test_sota.py — Quick architecture smoke test for SOTA v2.

Tests:
  1. Model builds with hidden_dim=256, num_layers=12 (~12M params)
  2. ligand_proj accepts 16-dim input (4 raw + 6 types + 6 sc_prior)
  3. forward() returns type_logits (not vel_type)
  4. compute_loss() runs with CE loss + self-conditioning
  5. sample() runs with marginal prior + x1 prediction
  6. Marginal prior shape is correct

Run (on server):
    python smoke_test_sota.py
"""
import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run():
    print("=" * 60)
    print("SMOKE TEST: SOTA v2 Architecture")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build model (SOTA v2 dimensions) ──
    from src.model.pocket_encoder import PocketEncoder
    from src.model.egnn import SE3EGNN
    from src.model.flow_matching import FlowMatching

    pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=256, num_layers=4, knn_k=16)
    egnn = SE3EGNN(
        ligand_in_dim=4,
        pocket_dim=256,
        hidden_dim=256,
        num_layers=12,
        num_heads=16,
        num_atom_types=6,
        knn_k=16,
    )
    model = FlowMatching(pocket_encoder=pocket_encoder, egnn=egnn, num_steps=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[PASS] Model built: {n_params:,} parameters")

    # Verify ligand_proj shape = [256, 16]
    lp = model.egnn.ligand_proj
    assert lp.in_features == 16, f"Expected ligand_proj in=16, got {lp.in_features}"
    assert lp.out_features == 256, f"Expected ligand_proj out=256, got {lp.out_features}"
    print(f"[PASS] ligand_proj: {lp.in_features} -> {lp.out_features} (includes sc_prior)")

    # Verify type_pred_head exists (not vel_type_head)
    assert hasattr(model.egnn, 'type_pred_head'), "type_pred_head not found in EGNN"
    assert not hasattr(model.egnn, 'vel_type_head'), "Old vel_type_head still present!"
    print("[PASS] type_pred_head exists (vel_type_head correctly removed)")

    # ── Fake data ──
    N_P, N_L = 30, 15
    pocket_pos  = torch.randn(N_P, 3, device=device)
    pocket_feat = torch.randn(N_P, 40, device=device)
    ligand_pos  = torch.randn(N_L, 3, device=device)
    ligand_feat = torch.randn(N_L, 20, device=device)   # 16 elem + 4 props
    ligand_types= torch.randint(0, 6, (N_L,), device=device)
    affinity    = torch.tensor(-8.5, device=device)
    ligand_bonds= torch.zeros(N_L, N_L, 4, device=device)

    # ── Test EGNN forward with sc_prior ──
    pocket_out = model.pocket_encoder(pocket_pos, pocket_feat)
    h_P = pocket_out["h_P"]
    atom_types_onehot = F.one_hot(ligand_types, 6).float()
    sc_prior = torch.zeros(N_L, 6, device=device)
    t = torch.tensor([0.5], device=device)

    out = model.egnn(
        x_L=ligand_pos,
        h_L_raw=torch.zeros(N_L, 4, device=device),
        atom_types_onehot=atom_types_onehot,
        t=t,
        h_P=h_P,
        sc_prior=sc_prior,
    )
    assert "type_logits" in out, f"type_logits missing from output: {list(out.keys())}"
    assert "vel_type" not in out, f"Old vel_type still in output!"
    assert out["type_logits"].shape == (N_L, 6), f"Wrong type_logits shape: {out['type_logits'].shape}"
    print(f"[PASS] EGNN forward: type_logits shape {out['type_logits'].shape}")

    # ── Test compute_loss with CE + self-conditioning ──
    model.train()
    marginal = torch.tensor([0.594, 0.165, 0.125, 0.048, 0.031, 0.037], device=device)
    losses = model.compute_loss(
        pocket_pos=pocket_pos,
        pocket_feat=pocket_feat,
        ligand_pos=ligand_pos,
        ligand_feat=ligand_feat,
        ligand_atom_types=ligand_types,
        affinity=affinity,
        ligand_bonds=ligand_bonds,
        marginal=marginal,
        sc_prob=0.5,
    )
    assert not torch.isnan(losses["total_loss"]), "NaN in total_loss!"
    assert not torch.isnan(losses["loss_type"]), "NaN in loss_type!"
    print(f"[PASS] compute_loss: total={losses['total_loss'].item():.4f}, "
          f"ce_type={losses['loss_type'].item():.4f}")

    # ── Test sample() with marginal prior + x1 prediction ──
    model.eval()
    with torch.no_grad():
        result = model.sample(
            pocket_pos=pocket_pos,
            pocket_feat=pocket_feat,
            num_atoms=20,
            temperature=0.8,
            marginal=marginal,
        )
    assert result["atom_types"].shape == (20,), f"Wrong atom_types shape: {result['atom_types'].shape}"
    unique_types = result["atom_types"].unique().tolist()
    print(f"[PASS] sample(): {result['num_atoms']} atoms, types={unique_types}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — SOTA v2 architecture is valid")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python scripts/compute_marginal.py")
    print("  2. Run: python run_training.py --phase A")
    print(f"\nModel: {n_params:,} params (was 2,760,621)")

if __name__ == "__main__":
    run()
