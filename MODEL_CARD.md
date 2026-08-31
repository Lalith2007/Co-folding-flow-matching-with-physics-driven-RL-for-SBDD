# PROTEUS Model Card & Release Specifications

## Model Overview
- **Model Name**: PROTEUS (Protein-Conditioned Equivariant Flow Matching for Structure-Based Drug Design)
- **Model Architecture**: SE(3)-Equivariant Continuous Flow Matching with E(n)-Equivariant Graph Neural Network (EGNN) vector fields and cross-attention pocket conditioning.
- **Production Baseline Checkpoint**: `checkpoints/rl_final.pt` (SHA256: `b99ef527f009f50e99b3c376c8ea11323c2b1b5fb654ddb46454f51954d90d9e`)
- **Experimental SDE Flow-GRPO Checkpoints**: `checkpoints/rl_final_scale_clean/seed_{42,123,2026}/step_{400,500}.pt`
- **Primary Task**: De novo 3D small-molecule ligand generation directly within target receptor pockets.

## Intended Use
- **De Novo Structure-Based Drug Design (SBDD)**: Generating 3D atomic coordinates and discrete atom types complementary to target protein binding cavities.
- **Target-Adaptive Lead Optimization**: Generating physically plausible (98.9% PoseBusters) and chemically valid (100.0% RDKit) small molecules for difficult/low-baseline enzymatic clefts.
- **Molecular Dynamics Lead Generation**: Generating starting 3D conformations capable of sub-2.0 Å stability in explicit-solvent all-atom MD simulations.

## Model Inputs & Outputs
- **Input**: Target protein binding pocket point cloud (heavy atoms and C-alpha centers within a 10.0 Å radius, capped at 800 nodes).
- **Output**: 3D atomic coordinates $X \in \mathbb{R}^{N_L \times 3}$ on the zero-Center-of-Mass manifold and discrete atom types $H \in \{0, 1\}^{N_L \times C}$ ($C \in \{\text{C, N, O, F, P, S, Cl}\}$).

## Benchmark Performance
- **Chemical Validity (RDKit)**: 100.0% across 7,000 generated molecules.
- **Unrelaxed 3D Plausibility (PoseBusters)**: 98.9% ± 0.1% without post-hoc force-field energy minimization.
- **20-Target Benchmark Suite (Multi-Seed Mean)**: Baseline Reward $= 0.6239 \rightarrow$ SDE Flow-GRPO Reward $= 0.6691 \pm 0.0185$ ($\Delta = +0.0452$, Cohen's $d = 0.852$, $p = 0.0018$, 80.0% win rate).
- **Expanded 100-Target Generalization Suite**: Baseline Reward $= 0.6428 \rightarrow$ SDE Flow-GRPO Reward $= 0.6487 \pm 0.0271$ ($\Delta = +0.0059$, $95\%$ bootstrap CI $[-0.0019, +0.0139]$, Wilcoxon $p = 0.8205$, Cohen's $d = 0.1435$).
- **Peak Single Seed (Seed 2026)**: Step 400 Reward $= 0.6722$ ($\Delta = +0.0294$, $p = 0.0042$).
- **Explicit-Solvent Molecular Dynamics (600 ns total)**: Mean ligand RMSD $= 1.42$ Å, Protein C$\alpha$ RMSD $= 1.18$ Å, 3 persistent hydrogen bonds per target across DHFR, CK2, and CPA.

## Known Limitations & Caveats
1. **Generalization Scope**: On broad multi-target suites (100 targets), the aggregate reward improvement ($\Delta = +0.0059$) is modest and spans zero in statistical uncertainty; the primary benefit is observed on challenging, low-baseline pockets ($r = -0.584$).
2. **Equivariance Framing**: Internal SE(3)-EGNN message-passing and coordinate updates are strictly equivariant on the zero-Center-of-Mass subspace, while the final velocity projection head exhibits small coordinate frame orientation variance.
3. **Biological Validation**: Docking scores and 600 ns explicit-solvent simulations provide computational evidence; experimental biochemical assay validation is required for clinical lead progression.

## License & Distribution
Released under the MIT License.
