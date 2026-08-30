import json
import yaml
from pathlib import Path
import torch
from src.data.dataset import SBDDDataset
from run_training import build_model

def main():
    print("=== DATASET & MODEL CUDA DIAGNOSTIC ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # 1. Load config
    try:
        with open("configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        base_dir = Path(cfg["data"]["base_dir"])
        print(f"Configured base_dir: {base_dir}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. Load dataset JSON
    dataset_json_path = cfg["data"].get("dataset_json", "server_final_dataset.json")
    try:
        with open(dataset_json_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {dataset_json_path}, found {len(data)} proteins.")
    except Exception as e:
        print(f"Error loading {dataset_json_path}: {e}")
        return

    # 3. Instantiate Dataset and load 1 valid sample
    flat_pairs = []
    for prot in data.values():
        for src in prot["sources"]:
            flat_pairs.append({
                "pdb_id": prot["pdb_id"],
                "pocket_path": src["pocket_path"],
                "ligand_path": src["ligand_path"],
                "affinity": src["affinity"],
                "dataset": src["dataset"],
            })
    
    dataset = SBDDDataset(flat_pairs, base_dir)
    
    print("\nLoading first sample to CPU...")
    sample = None
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is not None:
            print(f"Successfully loaded index {i} (PDB: {sample['pdb_id']}) to CPU.")
            break
            
    if sample is None:
        print("Error: Could not load any valid sample.")
        return

    # 4. Instantiate Model and push to CUDA
    print("\nBuilding model and moving to CUDA...")
    try:
        model = build_model(cfg, device)
        model.train()
        print("Model successfully built and pushed to device.")
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Push tensors to GPU
    print("\nMoving tensors to CUDA...")
    try:
        pocket_pos = sample["pocket_pos"].to(device)
        pocket_feat = sample["pocket_feat"].to(device)
        ligand_pos = sample["ligand_pos"].to(device)
        ligand_feat = sample["ligand_feat"].to(device)
        ligand_types = sample["ligand_atom_types"].to(device)
        affinity = sample["affinity"].to(device)
        weight = sample["weight"].to(device)
        ligand_bonds = sample["ligand_bonds"].to(device)
        
        # Batch index vectors (single sample batch)
        batch_P = torch.zeros(pocket_pos.size(0), dtype=torch.long, device=device)
        batch_L = torch.zeros(ligand_pos.size(0), dtype=torch.long, device=device)
        
        print("Tensors successfully moved.")
    except Exception as e:
        print(f"Error moving tensors to device: {e}")
        return

    # 6. Simulate Model Forward & Backward
    print("\nRunning simulated forward pass on CUDA...")
    try:
        losses = model.compute_loss(
            pocket_pos=pocket_pos,
            pocket_feat=pocket_feat,
            ligand_pos=ligand_pos,
            ligand_feat=ligand_feat,
            ligand_atom_types=ligand_types,
            affinity=affinity,
            weight=weight,
            ligand_bonds=ligand_bonds,
            batch_P=batch_P,
            batch_L=batch_L,
        )
        print(f"Forward pass success! Losses: {losses}")
    except Exception as e:
        print("Forward pass FAILED:")
        import traceback
        traceback.print_exc()
        return

    print("\nRunning simulated backward pass on CUDA...")
    try:
        losses["total_loss"].backward()
        print("Backward pass success!")
    except Exception as e:
        print("Backward pass FAILED:")
        import traceback
        traceback.print_exc()
        return

    print("\n=== ALL CUDA TEST PATHS PASSED SUCCESSFULLY ✅ ===")

if __name__ == "__main__":
    main()
