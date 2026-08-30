#!/usr/bin/env python
"""Debug script: inspect what the model actually generates."""

import torch
import numpy as np
from src.data.featurizer import PocketFeaturizer
from src.model.pocket_encoder import PocketEncoder
from src.model.egnn import SE3EGNN
from src.model.flow_matching import FlowMatching

IDX_TO_ELEMENT = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B']

# Build model (same as pipeline.py)
pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=128, num_layers=4, knn_k=16)
egnn = SE3EGNN(ligand_in_dim=20, hidden_dim=128, num_layers=9, num_heads=16, num_atom_types=10, knn_k=16)
model = FlowMatching(pocket_encoder=pocket_encoder, egnn=egnn, num_steps=50)

ckpt = torch.load("checkpoints/rl_final.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
print("Model loaded OK")

# Featurize pocket (use extracted pocket from last run, or the full PDB)
featurizer = PocketFeaturizer()
# Use the uploaded PDB directly for testing
import glob
pdb_files = glob.glob("uploads/*/7IN2.pdb")
if pdb_files:
    pdb_path = pdb_files[0]
else:
    pdb_path = "uploads/a56ae4e6/7IN2.pdb"

pocket_data = featurizer.featurize(pdb_path)
pocket_pos = pocket_data["pos"]
pocket_feat = pocket_data["feat"]
print(f"Pocket: {pocket_pos.shape[0]} atoms, feat_dim={pocket_feat.shape[1]}")

# Generate ONE molecule
with torch.no_grad():
    gen = model.sample(pocket_pos, pocket_feat)

coords = gen["pos"].numpy()
atom_type_indices = gen["atom_types"].numpy()
type_probs = gen["type_probs"].numpy()

print(f"\n{'='*60}")
print(f"Generated molecule: {len(coords)} atoms")
print(f"{'='*60}")

# 1. Show atom type distribution
print("\n--- Atom Type Distribution ---")
unique, counts = np.unique(atom_type_indices, return_counts=True)
for u, c in zip(unique, counts):
    elem = IDX_TO_ELEMENT[u] if u < len(IDX_TO_ELEMENT) else f"UNK({u})"
    print(f"  {elem}: {c} atoms ({100*c/len(atom_type_indices):.0f}%)")

# 2. Show coordinate statistics
print("\n--- Coordinate Statistics ---")
print(f"  x: min={coords[:,0].min():.3f}, max={coords[:,0].max():.3f}, range={coords[:,0].max()-coords[:,0].min():.3f}")
print(f"  y: min={coords[:,1].min():.3f}, max={coords[:,1].max():.3f}, range={coords[:,1].max()-coords[:,1].min():.3f}")
print(f"  z: min={coords[:,2].min():.3f}, max={coords[:,2].max():.3f}, range={coords[:,2].max()-coords[:,2].min():.3f}")

# 3. Pairwise distances
from scipy.spatial.distance import pdist
dists = pdist(coords)
print(f"\n--- Pairwise Distances ---")
print(f"  min: {dists.min():.4f} Å")
print(f"  max: {dists.max():.4f} Å")
print(f"  mean: {dists.mean():.4f} Å")
print(f"  median: {np.median(dists):.4f} Å")
print(f"  distances < 0.5 Å: {(dists < 0.5).sum()}/{len(dists)} ({100*(dists < 0.5).sum()/len(dists):.1f}%)")
print(f"  distances < 1.0 Å: {(dists < 1.0).sum()}/{len(dists)} ({100*(dists < 1.0).sum()/len(dists):.1f}%)")
print(f"  distances 1.0-2.0 Å (bond range): {((dists >= 1.0) & (dists <= 2.0)).sum()}")
print(f"  distances > 2.0 Å: {(dists > 2.0).sum()}")

# 4. Show the type probabilities - are they uniform/collapsed?
print(f"\n--- Type Probability Stats (softmax over z_type) ---")
print(f"  Mean max prob:   {type_probs.max(axis=1).mean():.4f}")
print(f"  Mean entropy:    {-(type_probs * np.log(type_probs + 1e-10)).sum(axis=1).mean():.4f}")
print(f"  Max possible entropy (uniform over 10): {np.log(10):.4f}")
print(f"  First 5 atoms type probs:")
for i in range(min(5, len(type_probs))):
    probs_str = " ".join(f"{p:.3f}" for p in type_probs[i])
    print(f"    Atom {i} (assigned={IDX_TO_ELEMENT[atom_type_indices[i]]}): [{probs_str}]")

# 5. Raw coordinates
print(f"\n--- Raw Coordinates (first 10) ---")
for i in range(min(10, len(coords))):
    elem = IDX_TO_ELEMENT[atom_type_indices[i]] if atom_type_indices[i] < len(IDX_TO_ELEMENT) else "?"
    print(f"  Atom {i:2d} ({elem:2s}): ({coords[i,0]:8.4f}, {coords[i,1]:8.4f}, {coords[i,2]:8.4f})")

# 6. Try bond inference
print(f"\n--- Bond Inference Test ---")
from src.inference.bond_inference import coords_to_smiles
result = coords_to_smiles(coords, atom_type_indices)
print(f"  Success: {result['success']}")
print(f"  SMILES:  {result.get('smiles', 'N/A')}")
print(f"  Error:   {result.get('error', 'None')}")

# Also try generate.py's method
from generate import coords_to_rdkit_mol, compute_mol_metrics, LIGAND_ATOM_TYPES
mol, sanitized = coords_to_rdkit_mol(coords, atom_type_indices)
metrics = compute_mol_metrics(mol, sanitized)
print(f"\n--- generate.py Method ---")
print(f"  Sanitized: {sanitized}")
print(f"  SMILES:    {metrics.get('smiles', 'N/A')}")
print(f"  Valid:     {metrics.get('valid', False)}")
print(f"  QED:       {metrics.get('qed', 'N/A')}")
print(f"  MW:        {metrics.get('mw', 'N/A')}")
