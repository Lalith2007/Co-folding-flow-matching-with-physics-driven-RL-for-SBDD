import numpy as np
from rdkit import Chem

LIGAND_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "B"]
COVALENT_RADII = {
    "C": 0.77, "N": 0.75, "O": 0.73, "S": 1.05, "F": 0.71,
    "Cl": 0.99, "Br": 1.14, "P": 1.10, "I": 1.33, "B": 0.82,
}
ELEMENT_TO_ATOMIC_NUM = {
    "C": 6, "N": 7, "O": 8, "S": 16, "F": 9,
    "Cl": 17, "Br": 35, "P": 15, "I": 53, "B": 5,
}

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
    
    sanitized = False
    try:
        Chem.SanitizeMol(mol)
        sanitized = True
    except Exception as e:
        print(f"Sanitization error: {e}")
    return mol.GetMol(), sanitized

# Create a fake C-C-O molecule
pos = np.array([
    [0.0, 0.0, 0.0],  # C
    [1.5, 0.0, 0.0],  # C (approx 1.5A bond)
    [2.5, 1.0, 0.0],  # O (approx 1.4A bond from C2)
])
# indices: C=0, C=0, O=2
types = np.array([0, 0, 2])

mol, valid = coords_to_rdkit_mol(pos, types)
print(f"Valid: {valid}")
if valid:
    print(Chem.MolToSmiles(mol))

# Test over-bonded carbon (5 bonds)
pos_bad = np.array([
    [0.0, 0.0, 0.0],  # C
    [1.0, 0.0, 0.0],  # F
    [-1.0, 0.0, 0.0], # F
    [0.0, 1.0, 0.0],  # F
    [0.0, -1.0, 0.0], # F
    [0.0, 0.0, 1.0],  # F (5th bond to carbon!)
])
types_bad = np.array([0, 4, 4, 4, 4, 4])
mol_bad, valid_bad = coords_to_rdkit_mol(pos_bad, types_bad)
print(f"Bad Valid: {valid_bad}")

