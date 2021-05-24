from rdkit import Chem
from typing import Iterable, List

import numpy as np
from openff.toolkit.topology import Molecule as OFFMolecule


from . import utils

def fragment_into_dummy_smiles(offmol, cleave_bonds=[]):
    rdmol = Chem.RWMol(offmol.to_rdkit())
    utils.clear_atom_map_numbers(rdmol)
    utils.assign_stereochemistry(rdmol)
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
        Chem.AssignStereochemistry(mol)
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
    rdmol = offmol.to_rdkit()
    for index, num in offmol.properties.get("atom_map", {}).items():
        rdmol.GetAtomWithIdx(index).SetAtomMapNum(num)
    rdmol = utils.subset_rdmol(rdmol, atom_indices)
    rdmol = Chem.AddHs(rdmol)
    Chem.SanitizeMol(rdmol)
    return utils.offmol_from_mol(rdmol)


def get_sub_smarts(offmol, atom_indices: List[int] = [],
                   label_indices: List[int] = []):
    rdmol = offmol.to_rdkit()
    utils.assign_stereochemistry(rdmol)
    for i, lix in enumerate(label_indices, 1):
        at = rdmol.GetAtomWithIdx(int(lix))
        at.SetAtomMapNum(i)
    indices = list(np.concatenate([atom_indices, label_indices]))
    atom_indices = sorted(map(int, set(indices)))
    # rdmol = utils.subset_rdmol(rdmol, indices, check_bonds=False)

    rdmol = Chem.RWMol(rdmol)
    to_remove = [i for i in range(rdmol.GetNumAtoms()) if i not in atom_indices]
    for i in to_remove[::-1]:
        rdmol.RemoveAtom(i)
    smarts = Chem.MolToSmarts(rdmol)
    smarts = smarts.replace("#0", "*")
    return smarts