from rdkit import Chem
from typing import Iterable, List

import numpy as np
from openff.toolkit.topology import Molecule as OFFMolecule


from . import utils


def fragment_into_dummy_smiles(offmol, cleave_bonds=[], unique_r_groups=True):
    rdmol = Chem.RWMol(offmol.to_rdkit())
    for atom in rdmol.GetAtoms():
        atom.SetAtomMapNum(0)
    utils.assign_stereochemistry(rdmol)
    dummy = Chem.Atom("*")
    r_linkages = {}

    if unique_r_groups:
        r_groups = [(i, i + 1) for i in range(1, (len(cleave_bonds) + 1) * 2, 2)]
    else:
        r_groups = [(1, 2)] * len(cleave_bonds)
    for bond, rs in zip(cleave_bonds, r_groups):
        bond_type = rdmol.GetBondBetweenAtoms(*bond).GetBondType()
        rdmol.RemoveBond(*bond)
        r_linkages[rs[0]] = [rs[1]]
        for atom_index, r in zip(bond, rs):
            dummy_copy = Chem.Atom(dummy)
            dummy_copy.SetAtomMapNum(r)
            new_atom_index = rdmol.AddAtom(dummy_copy)
            rdmol.AddBond(atom_index, new_atom_index, bond_type)
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


def subset_offmol(offmol: OFFMolecule,
                  atom_indices: Iterable[int],
                  check_bonds: bool = True,
                  return_atom_indices: bool = False,
                  sanitize: bool = True,
                  add_hs: bool = True) -> OFFMolecule:
    rdmol = offmol.to_rdkit()
    for index, num in offmol.properties.get("atom_map", {}).items():
        rdmol.GetAtomWithIdx(index).SetAtomMapNum(num)
    rdmol, used_indices = utils.subset_rdmol(rdmol, atom_indices, check_bonds=check_bonds,
                                             return_atom_indices=True)
    rdmol.UpdatePropertyCache()
    if add_hs:
        rdmol = Chem.AddHs(rdmol)
    if sanitize:
        Chem.SanitizeMol(rdmol)
    mol = utils.mol_to_offmol(rdmol)
    if return_atom_indices:
        return mol, used_indices
    return mol


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
