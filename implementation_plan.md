# Upgrading SBDD Pipeline to Industry SOTA

This document details the exact architectural and pipeline changes required to push our current SBDD flow matching model from an "academic open-source" level to a "Big Pharma" industry SOTA level (similar to Isomorphic Labs or advanced AlphaFold 3 variations).

## Goal Description
To achieve true SOTA, we must bridge the gap between static, proxy-scored generation and physics-driven, dynamic co-folding. We will achieve this via two major upgrades:
1. **Physics-Driven RL (AutoDock Vina Oracle):** Replacing the neural network proxy reward with true physics-based docking energies in the RL loop.
2. **Dynamic Protein Flexibility (Induced Fit):** Allowing the protein pocket to dynamically shift and "breathe" around the generated ligand during the ODE integration.

## User Review Required

> [!IMPORTANT]
> **Vina Dependencies:** To implement Phase 1, you will need to install `vina` and `meeko` (for PDBQT conversion) on your Jupyter server: `pip install vina meeko rdkit`. Please confirm if you can install these in your Docker environment.

> [!WARNING]
> **Compute Cost for Phase 2:** Adding Protein Flexibility means the EGNN must update coordinates for the pocket atoms (~500 atoms) alongside the ligand (~30 atoms). This will roughly **double your GPU memory usage** and increase training time. Please confirm if your A100 has sufficient memory overhead to support this.

## Proposed Changes

### Phase 1: AutoDock Vina Physics Oracle
Currently, `reward.py` has a `compute_vina_score` function that is just a placeholder returning `0.0`. We will fully implement this.

#### [MODIFY] [reward.py](file:///Users/lalithpraveen/Desktop/StudyNew/K-HUB/dd_pipeline/SM_Generation/src/model/reward.py)
- Import `vina` and `meeko`
- Add a function to convert the RDKit `mol` and the pocket `pdb` into `.pdbqt` strings in memory.
- Compute the bounding box around the pocket coordinates automatically.
- Initialize the `vina.Vina` object, dock the generated ligand in memory (no disk I/O to stay fast), and return the actual kcal/mol binding energy as the RL reward `r_vina`.

### Phase 2: Protein Flexibility (Induced Fit Co-Folding)
We will modify the model so the pocket isn't a frozen point-cloud. It will "move" to accommodate the generated ligand, mimicking real-world protein induced fit.

#### [MODIFY] [egnn.py](file:///Users/lalithpraveen/Desktop/StudyNew/K-HUB/dd_pipeline/SM_Generation/src/model/egnn.py)
- Update `EGNNLayerWithCrossAttn` to calculate messages from Ligand → Pocket as well, not just Pocket → Ligand.
- Add a `coord_pocket_mlp` to predict `delta_x_P` (the shift in pocket atom coordinates).
- Add a `vel_pocket_head` to output the final velocity for the pocket atoms.

#### [MODIFY] [flow_matching.py](file:///Users/lalithpraveen/Desktop/StudyNew/K-HUB/dd_pipeline/SM_Generation/src/model/flow_matching.py)
- Modify `forward_interpolation` to also apply a tiny amount of noise to the ground-truth pocket coordinates.
- Modify `compute_loss` to compute `loss_pocket_coord` alongside `loss_coord` and `loss_type`.
- Modify `sample()` so that `pocket_pos` is updated via Euler integration at each step (`pocket_pos += out["vel_pocket"] * dt`).

#### [MODIFY] [rl_finetune.py](file:///Users/lalithpraveen/Desktop/StudyNew/K-HUB/dd_pipeline/SM_Generation/src/train/rl_finetune.py)
- Ensure the modified flexible pocket coordinates are passed into the Vina Oracle so it scores the final "induced fit" structure, not just the starting static pocket.

## Verification Plan

### Automated Tests
- Run `python test_rdkit_sanitization.py` to ensure `meeko` PDBQT string conversions don't break our existing sanitization pipeline.
- Run a dummy `vina` pass in a scratch script to verify Python bindings work on the server.

### Manual Verification
- Kick off RL Phase B and verify that the Vina scores are actually driving the DDPO gradients.
- Export the `pocket_pos` at $t=0$ and $t=1$ during generation and open them in PyMOL to visually verify the sidechains are dynamically shifting around the ligand!
