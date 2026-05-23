"""Diagnose atom type prediction in the pretrained model.

This script generates a few molecules and inspects the RAW z_type
distributions to understand why the model favors nitrogen over carbon.
"""

import torch
import yaml
import numpy as np
from pathlib import Path

from src.model.flow_matching import FlowMatching
from src.data.featurizer import PocketFeaturizer, LIGAND_ATOM_TYPES

def load_model(config_path, checkpoint_path, device):
    from src.model.pocket_encoder import PocketEncoder
    from src.model.egnn import SBDDEGNN

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pocket_encoder = PocketEncoder(
        in_dim=40,
        hidden_dim=cfg["pocket_encoder"]["hidden_dim"],
        num_layers=cfg["pocket_encoder"]["num_layers"],
        knn_k=cfg["pocket"]["knn_k"],
    )

    egnn = SBDDEGNN(
        ligand_in_dim=20,
        pocket_dim=cfg["egnn"]["hidden_dim"],
        hidden_dim=cfg["egnn"]["hidden_dim"],
        num_layers=cfg["egnn"]["num_layers"],
        num_heads=cfg["egnn"]["num_heads"],
        num_atom_types=cfg["ligand"]["num_atom_types"],
        knn_k=cfg["pocket"]["knn_k"],
        dropout=0.0,
    )

    model = FlowMatching(
        pocket_encoder=pocket_encoder,
        egnn=egnn,
        num_steps=cfg["flow"]["num_steps_sample"],
        sigma_min=cfg["flow"]["sigma_min"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model("configs/default.yaml", "checkpoints/pretrain_final.pt", device)

    # Load pocket
    pocket_path = "/home/jovyan/dd_pipeline/datasets/crossdocked/12gs_A_rec_4pgt_gbx_lig_tt_min_0/pocket.pdb"
    featurizer = PocketFeaturizer()
    pocket_data = featurizer.featurize(pocket_path)
    pocket_pos = pocket_data["pos"].to(device)
    pocket_feat = pocket_data["feat"].to(device)

    print(f"Atom types: {LIGAND_ATOM_TYPES}")
    print(f"Indices:     {list(range(len(LIGAND_ATOM_TYPES)))}")
    print(f"Expected:    C=0, N=1, O=2, S=3, F=4, Cl=5")
    print()

    # Generate 5 molecules and inspect raw z_type
    for mol_idx in range(5):
        print(f"{'='*70}")
        print(f"MOLECULE {mol_idx + 1}")
        print(f"{'='*70}")

        with torch.no_grad():
            # Replicate model.sample() but capture intermediate z_type
            from src.model.flow_matching import subtract_com
            pocket_pos_centered = subtract_com(pocket_pos)
            pocket_out = model.pocket_encoder(pocket_pos_centered, pocket_feat)
            h_P = pocket_out["h_P"]

            N_L = int(torch.randint(20, 35, (1,)).item())
            z_coord = torch.randn(N_L, 3, device=device)
            z_coord = z_coord - z_coord.mean(0, keepdim=True)
            z_type = torch.ones(N_L, model.egnn.num_atom_types, device=device) / model.egnn.num_atom_types
            h_L_raw = torch.zeros(N_L, 20, device=device)

            dt = 1.0 / model.num_steps

            print(f"  Num atoms: {N_L}")
            print(f"  ODE steps: {model.num_steps}")
            print(f"  Initial z_type (uniform): {z_type[0].cpu().numpy()}")
            print()

            # Track type evolution for first atom
            type_history = [z_type[0].cpu().numpy().copy()]

            for s in range(model.num_steps):
                t_val = s * dt
                t = torch.tensor([t_val], device=device)

                out = model.egnn(
                    x_L=z_coord, h_L_raw=h_L_raw,
                    atom_types_onehot=z_type, t=t, h_P=h_P,
                )

                vel_type = out["vel_type"]
                z_coord = z_coord + out["vel_coord"] * dt
                z_type = z_type + vel_type * dt
                z_coord = z_coord - z_coord.mean(0, keepdim=True)

                # Log at key timesteps
                if s in [0, 9, 24, 39, 49]:
                    type_history.append(z_type[0].cpu().numpy().copy())

            # Final analysis
            final_types = z_type.argmax(dim=-1).cpu().numpy()
            type_counts = {LIGAND_ATOM_TYPES[i]: int((final_types == i).sum()) for i in range(len(LIGAND_ATOM_TYPES))}

            print(f"  Final atom type counts: {type_counts}")
            print(f"  Carbon ratio: {type_counts.get('C', 0) / N_L:.2%}")
            print(f"  Nitrogen ratio: {type_counts.get('N', 0) / N_L:.2%}")
            print()

            # Show z_type evolution for atom 0
            print(f"  z_type evolution for atom 0:")
            print(f"  {'Step':>6}  {'C':>8}  {'N':>8}  {'O':>8}  {'S':>8}  {'F':>8}  {'Cl':>8}  argmax")
            labels = ["init", "s=0", "s=9", "s=24", "s=39", "s=49"]
            for label, vals in zip(labels, type_history):
                am = LIGAND_ATOM_TYPES[np.argmax(vals)]
                print(f"  {label:>6}  {vals[0]:>8.4f}  {vals[1]:>8.4f}  {vals[2]:>8.4f}  "
                      f"{vals[3]:>8.4f}  {vals[4]:>8.4f}  {vals[5]:>8.4f}  → {am}")
            print()

            # Show final z_type for ALL atoms (raw scores before argmax)
            print(f"  Final z_type for all {N_L} atoms (top-2 per atom):")
            for atom_idx in range(min(N_L, 15)):  # Show first 15
                vals = z_type[atom_idx].cpu().numpy()
                sorted_idx = np.argsort(vals)[::-1]
                top1 = LIGAND_ATOM_TYPES[sorted_idx[0]]
                top2 = LIGAND_ATOM_TYPES[sorted_idx[1]]
                gap = vals[sorted_idx[0]] - vals[sorted_idx[1]]
                print(f"    atom {atom_idx:2d}: {top1}({vals[sorted_idx[0]]:.4f}) > "
                      f"{top2}({vals[sorted_idx[1]]:.4f})  gap={gap:.4f}")
            if N_L > 15:
                print(f"    ... ({N_L - 15} more atoms)")
        print()

if __name__ == "__main__":
    main()
