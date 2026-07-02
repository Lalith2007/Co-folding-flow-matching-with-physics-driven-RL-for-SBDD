import json
import yaml
from pathlib import Path
from src.data.dataset import SBDDDataset

def main():
    print("=== DATASET DIAGNOSTIC SCRIPT ===")
    
    # 1. Load config
    try:
        with open("configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        base_dir = Path(cfg["data"]["base_dir"])
        print(f"Configured base_dir: {base_dir}")
        print(f"base_dir exists: {base_dir.exists()}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. Load dataset JSON
    try:
        with open("final_dataset.json", "r") as f:
            data = json.load(f)
        print(f"Loaded final_dataset.json, found {len(data)} proteins.")
    except Exception as e:
        print(f"Error loading final_dataset.json: {e}")
        return

    # 3. Inspect first few paths
    first_protein = list(data.values())[0]
    sources = first_protein["sources"]
    print(f"\nChecking first protein ({first_protein['pdb_id']}) with {len(sources)} sources:")
    for idx, src in enumerate(sources):
        pocket_path = base_dir / src["pocket_path"]
        ligand_path = base_dir / src["ligand_path"]
        print(f"  [{idx}] Pocket path: {pocket_path} (Exists: {pocket_path.exists()})")
        print(f"      Ligand path: {ligand_path} (Exists: {ligand_path.exists()})")

    # 4. Instantiate Dataset and try to load
    print("\nAttempting to load via SBDDDataset class...")
    # Gather flat list of pairs
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
    print(f"Total pairs in dataset object: {len(dataset)}")
    
    # Let's try to get the first 5 samples and catch the exact errors
    success_count = 0
    for i in range(min(5, len(dataset))):
        pair = dataset.pairs[i]
        pocket_path = base_dir / pair["pocket_path"]
        ligand_path = base_dir / pair["ligand_path"]
        print(f"\n--- Loading sample {i} (PDB: {pair['pdb_id']}) ---")
        try:
            # Replicate the dataset load logic manually to print the traceback
            pocket_data = dataset.pocket_feat.featurize(str(pocket_path))
            ligand_data = dataset.ligand_feat.featurize(str(ligand_path))
            
            if pocket_data["pos"] is None or ligand_data["pos"] is None:
                print("  FAIL: featurizer returned None positions.")
                continue
                
            from rdkit import Chem
            from src.data.featurizer import LIGAND_ATOM_TYPES
            suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=True, removeHs=True)
            mol = next(iter(suppl), None)
            if mol is None:
                print("  FAIL: RDKit failed to read molecule.")
                continue
                
            exotic = False
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in LIGAND_ATOM_TYPES:
                    print(f"  FAIL: Found exotic atom type '{atom.GetSymbol()}'")
                    exotic = True
                    break
            if exotic:
                continue
                
            print("  SUCCESS: Loaded successfully!")
            success_count += 1
        except Exception as e:
            import traceback
            print("  FAIL: Exception raised during loading:")
            traceback.print_exc()

    print(f"\nDiagnostic finished. Loaded {success_count}/{min(5, len(dataset))} successfully.")

if __name__ == "__main__":
    main()
