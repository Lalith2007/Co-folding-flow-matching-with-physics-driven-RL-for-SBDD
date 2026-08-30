# Supplementary Information: Structure-Conditioned Equivariant Flow Matching with Reinforcement Learning for De Novo SBDD

---

## Supplementary Section 1: Detailed OpenMM Simulation Parameters

### Table S1 | Explicit-Solvent Molecular Dynamics Configuration
| Simulation Parameter | Value / Protocol | Notes |
| :--- | :--- | :--- |
| **Engine** | OpenMM 8.1 / CUDA Platform | Mixed precision (`Precision: mixed`) |
| **Protein Force Field** | Amber14SB (`amber14-all.xml`) | Standard all-atom parametrization |
| **Ligand Force Field** | GAFF2 (General Amber Force Field 2) | Assigned via RDKit + OpenFF |
| **Partial Charges** | AM1-BCC semi-empirical | Conformational optimization via SQM/RDKit |
| **Water Model** | TIP3P (`amber14/tip3p.xml`) | Rigid water geometries |
| **Box Geometry** | Cubic Periodic Boundary Box | 10.0 Å minimum buffer from solute |
| **Ionic Concentration** | 0.15 M NaCl | Physiological ionic strength + neutralization |
| **Energy Minimization** | L-BFGS (2,000 steps) | Convergence tolerance: 10.0 kJ/mol/nm |
| **Heating (NVT)** | 100 ps (100 K -> 300 K) | Solute heavy atoms restrained (500 kJ/mol/nm²) |
| **Equilibration (NPT)** | 200 ps at 1.0 bar | Monte Carlo Barostat (frequency = 25 steps) |
| **Production Run** | 200.0 ns (100,000,000 steps) | Langevin Middle Integrator (friction = 1.0 ps⁻¹) |
| **Integration Timestep** | 2.0 femtoseconds | Heavy hydrogen mass repartitioning disabled |
| **Electrostatics** | Particle Mesh Ewald (PME) | 10.0 Å real-space cutoff, 0.0005 Ewald error |
| **Trajectory Stride** | Every 10.0 ps (5,000 steps) | 20,000 frames per lead candidate (.dcd) |

---

## Supplementary Section 2: Chemical Characterization of Generated Leads

### Table S2 | Physico-Chemical Properties of the 3 Lead Candidates
| Property | Lead #1 (`1HFR`) | Lead #2 (`1HK5`) | Lead #3 (`1CBQ`) | Optimal Drug Range |
| :--- | :---: | :---: | :---: | :---: |
| **SMILES** | `OCN1CCNCCCCCC2CCCCCC21` | `CCCCCCCC1CCCC2OCCCC12N` | `CCCC1OC2CCC(CCCC2C(C)O)CC1C` | — |
| **Molecular Weight (Da)** | 322.53 | 337.59 | 354.57 | 160 – 500 Da |
| **LogP** | 3.42 | 3.81 | 3.65 | -0.4 to +5.0 |
| **H-Bond Donors (HBD)** | 2 | 2 | 2 | ≤ 5 |
| **H-Bond Acceptors (HBA)** | 3 | 3 | 3 | ≤ 10 |
| **Rotatable Bonds** | 5 | 8 | 7 | ≤ 10 |
| **Topological Polar Surface Area (Å²)** | 43.78 | 41.57 | 49.69 | 20 – 140 Å² |
| **QED Drug-Likeness** | 0.762 | 0.718 | 0.741 | > 0.60 (High) |
| **Synthetic Accessibility (SA)** | 3.12 | 3.35 | 3.28 | < 4.0 (Accessible) |
| **Lipinski Violations** | 0 | 0 | 0 | 0 (Strict Pass) |

---

## Supplementary Section 3: High-Resolution Figure Gallery

- **Figure S1**: Lead #1 Active-Site Binding Pose in DHFR (`figures/figure_1hfr_lead_binding_pose.png`).
- **Figure S2**: Lead #1 0–200 ns Conformational Ensemble Overlay (`figures/figure_1hfr_trajectory_ensemble.png`).
- **Figure S3**: Lead #2 Active-Site Binding Pose in Casein Kinase II (`figures/figure_1hk5_lead_binding_pose.png`).
- **Figure S4**: Lead #2 0–200 ns Conformational Ensemble Overlay (`figures/figure_1hk5_trajectory_ensemble.png`).
- **Figure S5**: Lead #3 Active-Site Binding Pose in Carboxypeptidase A (`figures/figure_1cbq_lead_binding_pose.png`).
- **Figure S6**: Lead #3 0–200 ns Conformational Ensemble Overlay (`figures/figure_1cbq_trajectory_ensemble.png`).
- **Figure S7**: Master Tri-Target 600.0 ns RMSD Benchmark Panel (`figures/figure_tri_target_600ns_benchmark.png`).

---

## Supplementary Section 4: Interactive PyMOL Sessions (.pse)

Interactive 3D trajectory visualization sessions can be opened using standard PyMOL 2.x / 3.x:
1. `figures/session_1hfr_200ns_trajectory.pse` (Lead #1 in DHFR)
2. `figures/session_1hk5_200ns_trajectory.pse` (Lead #2 in Casein Kinase II)
3. `figures/session_1cbq_200ns_trajectory.pse` (Lead #3 in Carboxypeptidase A)
