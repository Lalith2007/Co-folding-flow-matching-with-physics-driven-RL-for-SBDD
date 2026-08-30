"""
Environment & Dependency Verification Script
Checks all required Python packages for DrugParadigm SBDD Flow Matching.
"""
import sys

PACKAGES = [
    ("PyTorch", "torch"),
    ("RDKit", "rdkit"),
    ("BioPython", "Bio"),
    ("AutoDock Vina", "vina"),
    ("Meeko (PDBQT prep)", "meeko"),
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("Pandas", "pandas"),
    ("PyYAML", "yaml"),
    ("tqdm", "tqdm"),
    ("FastAPI", "fastapi"),
    ("Uvicorn", "uvicorn"),
]

OPTIONAL_PACKAGES = [
    ("OpenMM (MD Simulation)", "openmm"),
    ("OpenBabel (Fallback)", "openbabel"),
    ("PoseBusters CLI", "posebusters"),
    ("Matplotlib", "matplotlib"),
    ("Seaborn", "seaborn"),
]

def check_modules():
    print("=" * 60)
    print(" DrugParadigm Environment Verification")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")
    print("-" * 60)
    
    missing_core = []
    
    print("--- Core SBDD & Physics Dependencies ---")
    for name, mod in PACKAGES:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"  [OK] {name:<22}: {ver}")
        except Exception as e:
            print(f"  [MISSING] {name:<22} (Error: {e})")
            missing_core.append(mod)
            
    print("\n--- Optional & MD Simulation Packages ---")
    for name, mod in OPTIONAL_PACKAGES:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"  [OK] {name:<22}: {ver}")
        except Exception as e:
            print(f"  [OPTIONAL] {name:<22} (not installed: {e})")

    # CUDA Check
    print("\n--- GPU & CUDA Acceleration ---")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA Available: True (Device: {torch.cuda.get_device_name(0)})")
            print(f"  [OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("  [WARN] CUDA Available: False (Running on CPU)")
    except Exception as e:
        print(f"  [ERROR] PyTorch CUDA Check failed: {e}")

    print("=" * 60)
    if missing_core:
        print(f"To install missing packages, run:")
        print(f"  pip install {' '.join(missing_core)}")
    else:
        print(" All core dependencies are installed and ready to go!")
    print("=" * 60)

if __name__ == "__main__":
    check_modules()
