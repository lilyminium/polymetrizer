from rdkit import Chem
from typing import Iterable
from openff.toolkit.topology import Molecule as OFFMolecule


from . import utils

def fragment_into_dummy_smiles(offmol, cleave_bonds=[]):
    rdmol = Chem.RWMol(offmol.to_rdkit())
    utils.clear_atom_map_numbers(rdmol)
    Chem.rdmolops.AssignStereochemistryFrom3D(rdmol)
    dummy = Chem.Atom("*")
    r_linkages = {}
    counter = 1
    for bond in cleave_bonds:
        bond_type = rdmol.GetBondBetweenAtoms(*bond).GetBondType()
        rdmol.RemoveBond(*bond)
        r_linkages[counter] = [counter + 1]
        for atom_index in bond:
            dummy_copy = Chem.Atom(dummy)
            dummy_copy.SetAtomMapNum(counter)
            new_atom_index = rdmol.AddAtom(dummy_copy)
            rdmol.AddBond(atom_index, new_atom_index, bond_type)
            counter += 1
    mols = Chem.GetMolFrags(rdmol, asMols=True)
    for mol in mols:
        counter = 1
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "*":
                atom.SetAtomMapNum(counter)
                counter += 1
    smiles = [utils.mol_to_smiles(m) for m in mols]
    return smiles, r_linkages


def subset_mol(
        offmol: OFFMolecule,
        atom_indices: Iterable[int],
    ) -> OFFMolecule:

    rdmol = Chem.RWMol(offmol.to_rdkit())
    for index, num in offmol.properties.get("atom_map", {}).items():
        rdmol.GetAtomWithIdx(index).SetAtomMapNum(num)

    to_remove = [i for i in range(offmol.n_atoms) if i not in atom_indices]
    for i in to_remove[::-1]:
        rdmol.RemoveAtom(i)
    rdmol = Chem.AddHs(rdmol)
    Chem.SanitizeMol(rdmol)
    return OFFMolecule.from_rdkit(rdmol, allow_undefined_stereo=True)