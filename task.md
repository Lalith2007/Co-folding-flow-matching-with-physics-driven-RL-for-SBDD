# Big Pharma SOTA SBDD Upgrade Tasks

- `[/]` Phase 1: AutoDock Vina Physics Oracle
  - `[x]` Implement `compute_vina_score` in `reward.py` using `vina` and `meeko`.
  - `[x]` Handle in-memory PDBQT string conversions for both ligand and pocket.
  - `[x]` Add automated bounding box calculation based on pocket coordinates.
  - `[x]` Connect Vina reward to the DDPO RL loop in `rl_finetune.py`.

- `[x]` Phase 2: Protein Flexibility (Induced Fit Co-Folding)
  - `[x]` Update `EGNNLayerWithCrossAttn` in `egnn.py` to allow Ligand → Pocket messages.
  - `[x]` Add `coord_pocket_mlp` and `vel_pocket_head` to predict `vel_pocket`.
  - `[x]` Update `flow_matching.py` to noise pocket coordinates and compute `loss_pocket_coord`.
  - `[x]` Update `sample()` in `flow_matching.py` to apply Euler steps to `pocket_pos`.
  - `[x]` Ensure Vina uses the *flexible* generated pocket coordinates for final scoring in RL.
