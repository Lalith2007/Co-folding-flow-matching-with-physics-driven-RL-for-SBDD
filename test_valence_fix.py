import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
pt = GetPeriodicTable()

LIGAND_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "B"]
COVALENT_RADII = {
    "C": 0.77, "N": 0.75, "O": 0.73, "S": 1.05, "F": 0.71,
    "Cl": 0.99, "Br": 1.14, "P": 1.10, "I": 1.33, "B": 0.82,
}
ELEMENT_TO_ATOMIC_NUM = {
    "C": 6, "N": 7, "O": 8, "S": 16, "F": 9,
    "Cl": 17, "Br": 35, "P": 15, "I": 53, "B": 5,
}

def fix_valences(mol, pos):
    """Iteratively remove longest bonds from over-bonded atoms."""
    while True:
        try:
            # We must catch the exception to know which atom is failing
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
            return mol_copy, True # Success!
        except ValueError as e:
            # RDKit doesn't expose the atom index easily in Python exceptions sometimes, 
            # so let's manually check valences.
            pass
        except Exception as e:
            pass
            
        # Manually count bonds and compare to max valence
        fixed_something = False
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            # Default max valences
            max_v = pt.GetDefaultValence(atom.GetAtomicNum())
            if symbol == 'N': max_v = 3
            if symbol == 'O': max_v = 2
            if symbol == 'S': max_v = 6
            if symbol == 'P': max_v = 5
            
            degree = atom.GetDegree()
            if degree > max_v:
                # Find the longest bond connected to this atom
                longest_bond = None
                max_dist = -1.0
                for bond in atom.GetBonds():
                    neighbor = bond.GetOtherAtom(atom)
                    n_idx = neighbor.GetIdx()
                    dist = np.linalg.norm(pos[idx] - pos[n_idx])
                    if dist > max_dist:
                        max_dist = dist
                        longest_bond = bond
                        
                if longest_bond is not None:
                    mol.RemoveBond(longest_bond.GetBeginAtomIdx(), longest_bond.GetEndAtomIdx())
                    fixed_something = True
                    break # Break and retry
        
        if not fixed_something:
            # If we couldn't fix it, break out
            return mol, False

def coords_to_rdkit_mol(pos, atom_type_indices, bond_tolerance=0.15):
    from rdkit.Geometry import Point3D
    N = len(pos)
    elements = [LIGAND_ATOM_TYPES[i] for i in atom_type_indices]
    mol = Chem.RWMol()
    for elem in elements:
        atom = Chem.Atom(ELEMENT_TO_ATOMIC_NUM[elem])
        mol.AddAtom(atom)
        
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(pos[i] - pos[j])
            r_i = COVALENT_RADII.get(elements[i], 1.0)
            r_j = COVALENT_RADII.get(elements[j], 1.0)
            if dist < r_i + r_j + bond_tolerance:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
                
    conf = Chem.Conformer(N)
    for i in range(N):
        conf.SetAtomPosition(i, Point3D(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
    mol.AddConformer(conf, assignId=True)
    
    # Run the fixer
    fixed_mol, sanitized = fix_valences(mol, pos)
    
    # Get largest connected fragment
    frags = Chem.GetMolFrags(fixed_mol, asMols=True)
    if not frags:
        return fixed_mol, False
        
    largest_frag = max(frags, key=lambda f: f.GetNumAtoms())
    
    return largest_frag, sanitized

# Test over-bonded carbon (5 bonds)
pos_bad = np.array([
    [0.0, 0.0, 0.0],  # C
    [1.0, 0.0, 0.0],  # F (1.0A)
    [-1.0, 0.0, 0.0], # F (1.0A)
    [0.0, 1.0, 0.0],  # F (1.0A)
    [0.0, -1.0, 0.0], # F (1.0A)
    [0.0, 0.0, 1.2],  # F (1.2A - should be removed!)
])
types_bad = np.array([0, 4, 4, 4, 4, 4])
mol_bad, valid_bad = coords_to_rdkit_mol(pos_bad, types_bad)
print(f"Bad Valid: {valid_bad}")
if valid_bad:
    print(Chem.MolToSmiles(mol_bad))

