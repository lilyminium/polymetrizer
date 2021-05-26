from typing import Iterable, Set
from collections import defaultdict

from rdkit import Chem
from openff.toolkit.topology import Molecule as OFFMolecule
import numpy as np

def mol_from_smiles(smiles):
    smiles_parser = Chem.rdmolfiles.SmilesParserParams()
    smiles_parser.removeHs = False
    return Chem.MolFromSmiles(smiles, smiles_parser)

def mol_to_smarts(rdmol):
    smarts = Chem.MolToSmarts(rdmol, isomericSmiles=True)
    smarts = smarts.replace("#0", "*")
    return smarts

def mol_to_smiles(rdmol):
    return Chem.MolToSmiles(rdmol, isomericSmiles=True, allHsExplicit=True)

def assign_stereochemistry(rdmol):
    if rdmol.GetNumConformers():
        Chem.AssignStereochemistryFrom3D(rdmol)
    else:
        Chem.AssignStereochemistry(rdmol, cleanIt=True)

def offmol_from_mol(rdmol) -> OFFMolecule:
    Chem.SanitizeMol(rdmol)
    assign_stereochemistry(rdmol)
    return OFFMolecule.from_rdkit(rdmol, allow_undefined_stereo=True)

def clear_atom_map_numbers(rdmol):
    for atom in rdmol.GetAtoms():
        atom.SetAtomMapNum(0)

def subset_rdmol(
        rdmol: Chem.Mol,
        atom_indices: Iterable[int],
        check_bonds: bool = True,
    ) -> Chem.Mol:
    rdmol = Chem.RWMol(rdmol)
    to_remove = [i for i in range(rdmol.GetNumAtoms()) if i not in atom_indices]
    if check_bonds:
        multiple_bonds = []
        # check bonds
        for i in to_remove:
            atom = rdmol.GetAtomWithIdx(i)
            n_bonds = 0
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(i)
                if other in atom_indices:
                    n_bonds += 1
            if n_bonds > 1:
                multiple_bonds.append(i)
        to_remove = [i for i in to_remove if i not in multiple_bonds]
    for i in to_remove[::-1]:
        rdmol.RemoveAtom(i)
    rdmol.UpdatePropertyCache()
    return rdmol


def get_atom_indices_bonded_to_indices(
        rdmol: Chem.Mol,
        atom_indices: Iterable[int],
    ) -> Set[int]:
    if not atom_indices:
        return set()
    partners = defaultdict(set)
    for index in atom_indices:
        index = int(index)
        atom = rdmol.GetAtomWithIdx(index)
        for bond in atom.GetBonds():
            partners[index].add(bond.GetOtherAtomIdx(index))
    
    bonded = partners.get(atom_indices[0], set())
    for index in atom_indices[1:]:
        bonded &= partners.get(index, set())
    return bonded


def substructure_search(target, pattern):
    # smarts gives us query atoms
    copy = Chem.RWMol(pattern)
    for atom in copy.GetAtoms():
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    
    tmp = mol_from_smiles(mol_to_smiles(copy))
    query = Chem.MolFromSmarts(mol_to_smarts(tmp))


    # rearrange smarts into original order
    match = pattern.GetSubstructMatch(query)
    ordering = list(map(int, np.argsort(match)))
    query = Chem.RenumberAtoms(query, ordering)
    query = Chem.RWMol(query)


    return target.GetSubstructMatches(query)
