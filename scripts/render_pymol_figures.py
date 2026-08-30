# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
#     "rdkit",
#     "numpy",
#     "requests",
# ]
# ///

import os
import sys
import urllib.request
from pathlib import Path
import numpy as np

# Set environment variable for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pymol
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()

from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem

def main():
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pdb_id = "1hfr"
    pdb_path = out_dir / f"{pdb_id}.pdb"
    
    # 1. Download 1hfr.pdb if not present
    if not pdb_path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Downloading {pdb_id} from RCSB PDB...")
        urllib.request.urlretrieve(url, pdb_path)
    
    # Extract pocket center from native MTX in 1hfr.pdb
    pocket_coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resn = line[17:20].strip()
                if resn == "MTX":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    pocket_coords.append([x, y, z])
    
    if pocket_coords:
        pocket_center = np.mean(pocket_coords, axis=0)
    else:
        pocket_center = np.array([26.34, 8.51, 8.52])
    print(f"Target Pocket Center: {pocket_center}")
    
    # 2. Build 3D generated ligand (OCN1CCNCCCCCC2CCCCCC21)
    smiles = "OCN1CCNCCCCCC2CCCCCC21"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    lig_center = np.mean(pos, axis=0)
    shift = pocket_center - lig_center
    
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (float(p.x + shift[0]), float(p.y + shift[1]), float(p.z + shift[2])))
        
    lig_pdb = out_dir / "1hfr_generated_lead.pdb"
    Chem.MolToPDBFile(mol, str(lig_pdb))
    print(f"Saved positioned lead PDB: {lig_pdb}")
    
    # 3. Load into PyMOL
    cmd.reinitialize()
    cmd.set("ray_opaque_background", 1)
    cmd.set("bg_rgb", [1, 1, 1])  # Crisp publication white background
    
    cmd.load(str(pdb_path), "prot")
    if cmd.count_atoms("prot") == 0:
        print("Error: Failed to load protein structure.")
        cmd.quit()
        return

    # Clean protein: remove crystallographic waters and native MTX
    cmd.remove("resn HOH")
    cmd.remove("resn MTX")
    cmd.remove("resn NDP")
    
    # Load generated lead ligand
    cmd.load(str(lig_pdb), "lead_ligand")
    if cmd.count_atoms("lead_ligand") == 0:
        print("Error: Failed to load generated lead.")
        cmd.quit()
        return

    # 4. Professional Publication Styling
    cmd.hide("everything", "all")
    
    # Protein cartoon ribbon styling
    cmd.show("cartoon", "prot")
    cmd.color("slate", "prot")
    cmd.set("cartoon_transparency", 0.25, "prot")
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    
    # Ligand styling: Vibrant Cyan Carbon Sticks
    cmd.show("sticks", "lead_ligand")
    cmd.set("stick_radius", 0.28, "lead_ligand")
    cmd.color("cyan", "lead_ligand and elem C")
    cmd.color("firebrick", "lead_ligand and elem O")
    cmd.color("marine", "lead_ligand and elem N")
    cmd.color("white", "lead_ligand and elem H")
    
    # Active site binding residues within 4.8 Angstroms
    cmd.select("pocket_res", "byres (prot within 4.8 of lead_ligand)")
    cmd.show("sticks", "pocket_res and not (name c,n,o and not resn pro)")
    cmd.set("stick_radius", 0.18, "pocket_res")
    cmd.color("salmon", "pocket_res and elem C")
    cmd.color("firebrick", "pocket_res and elem O")
    cmd.color("marine", "pocket_res and elem N")
    
    # Label key pocket residues
    cmd.set("label_font_id", 7)
    cmd.set("label_size", 14)
    cmd.set("label_color", "black")
    cmd.label("pocket_res and name CA", '"%s %s" % (resn, resi)')
    
    # Compute and display Hydrogen Bonds
    cmd.distance("hbonds", "lead_ligand", "pocket_res", 3.4, mode=2)
    cmd.set("dash_color", "yellow")
    cmd.set("dash_width", 3.5)
    cmd.set("dash_gap", 0.25)
    cmd.set("dash_radius", 0.06)
    cmd.set("dash_length", 0.15)
    cmd.hide("labels", "hbonds")
    
    # View 1: Focused Active-Site Binding Pose
    cmd.zoom("lead_ligand", 4.5)
    cmd.orient("pocket_res")
    pose_png = out_dir / "figure_1hfr_lead_binding_pose.png"
    cmd.png(str(pose_png), width=2400, height=1800, dpi=300)
    print(f"Generated: {pose_png}")
    
    # View 2: Surface Pocket / Cavity View
    cmd.show("surface", "prot")
    cmd.set("surface_color", "gray90", "prot")
    cmd.set("transparency", 0.45, "prot")
    surface_png = out_dir / "figure_1hfr_lead_surface_pocket.png"
    cmd.png(str(surface_png), width=2400, height=1800, dpi=300)
    print(f"Generated: {surface_png}")
    
    # View 3: Full Complex Overview
    cmd.hide("surface", "prot")
    cmd.zoom("prot", 2.0)
    overview_png = out_dir / "figure_1hfr_complex_overview.png"
    cmd.png(str(overview_png), width=2400, height=1800, dpi=300)
    print(f"Generated: {overview_png}")
    
    # Save PyMOL Session (.pse) for user
    pse_file = out_dir / "session_1hfr_lead.pse"
    cmd.save(str(pse_file))
    print(f"Generated PyMOL Session: {pse_file}")
    
    cmd.quit()
    print("\nPyMOL rendering pipeline successfully finished!")

if __name__ == "__main__":
    main()
