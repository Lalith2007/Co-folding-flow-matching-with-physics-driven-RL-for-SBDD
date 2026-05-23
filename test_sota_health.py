import sys
import torch
import yaml
from pathlib import Path

def print_result(name, passed, detail=""):
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{status:<10} | {name:<30} | {detail}")

print("============================================================")
print("  SOTA PRE-FLIGHT HEALTH CHECK")
print("============================================================\n")

# 1. Check Vina & Meeko
try:
    import vina
    import meeko
    print_result("Vina & Meeko Dependencies", True, "Successfully imported vina and meeko.")
except ImportError as e:
    print_result("Vina & Meeko Dependencies", False, f"Missing dependency: {e}. Run: pip install vina meeko")

# 2. Check type_loss_weight in config
try:
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    weight = cfg["pretrain"]["type_loss_weight"]
    if weight >= 5.0:
        print_result("Atom Type Loss Weight", True, f"type_loss_weight = {weight} (≥ 5.0)")
    else:
        print_result("Atom Type Loss Weight", False, f"type_loss_weight = {weight} (Should be ≥ 5.0)")
except Exception as e:
    print_result("Atom Type Loss Weight", False, f"Could not read config: {e}")

# 3. Check Pocket Flexibility Zero-Init
try:
    from src.model.egnn import SBDDEGNN
    egnn = SBDDEGNN(
        ligand_in_dim=20,
        pocket_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_atom_types=6,
        knn_k=16
    )
    
    weight_sum = egnn.vel_pocket_head[-1].weight.abs().sum().item()
    bias_sum = egnn.vel_pocket_head[-1].bias.abs().sum().item()
    
    if weight_sum == 0.0 and bias_sum == 0.0:
        print_result("Pocket Head Zero-Init", True, "vel_pocket_head weights are perfectly 0.0")
    else:
        print_result("Pocket Head Zero-Init", False, f"Weights not zero! sum={weight_sum}")
except Exception as e:
    print_result("Pocket Head Zero-Init", False, f"Model failed to initialize: {e}")

print("\n============================================================")
