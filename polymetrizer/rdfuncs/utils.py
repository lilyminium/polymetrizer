from typing import List

from rdkit import Chem
import numpy as np

def mol_from_smiles(smiles):
    smiles_parser = Chem.rdmolfiles.SmilesParserParams()
    smiles_parser.removeHs = False
    return Chem.MolFromSmiles(smiles, smiles_parser)

def mol_to_smarts(rdmol):
    return Chem.MolToSmarts(rdmol, isomericSmiles=True)

def mol_to_smiles(rdmol):
    return Chem.MolToSmiles(rdmol, isomericSmiles=True, allHsExplicit=True)

def get_sub_smarts(offmol, atom_indices: List[int] = [],
                   label_indices: List[int] = []):
    rdmol = offmol.to_rdkit()
    for i, lix in enumerate(label_indices, 1):
        at = rdmol.GetAtomWithIdx(int(lix))
        at.SetAtomMapNum(i)
    indices = list(np.concatenate([atom_indices, label_indices]))
    indices = sorted(map(int, set(indices)))
    smarts = Chem.MolFragmentToSmarts(rdmol, atomsToUse=indices)
    smarts = smarts.replace("#0", "*")
    return smarts

def clear_atom_map_numbers(rdmol):
    for atom in rdmol.GetAtoms():
        atom.SetAtomMapNum(0)