from typing import Tuple

import rdkit
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from openff.toolkit.topology import Molecule as OFFMolecule


from ..utils import tuple_from_string
from . import utils

SUBSTITUENT_ISOTOPE = 2
SUBSTITUENT_ATOMNUM = 2
OLIGOMER_ATOM_INDEX_PROP = 'oligomerAtomIndex'
RGROUP_PROP = 'RGroupNum'

def oligomer_to_reactant(oligomer, r_group_number=None):
    rdreactant = oligomer.offmol.to_rdkit()
    for index, num in oligomer.central_atom_map.items():
        rdreactant.GetAtomWithIdx(index).SetAtomMapNum(int(num))
    for index, oligindex in oligomer.atom_oligomer_map.items():
        atom = rdreactant.GetAtomWithIdx(int(index))
        atom.SetProp(OLIGOMER_ATOM_INDEX_PROP, str(oligindex))
    if r_group_number is None:
        return rdreactant
    r_group_index = oligomer.r_group_indices[r_group_number]
    for num, index in oligomer.r_group_indices.items():
        atom = rdreactant.GetAtomWithIdx(index)
        atom.SetIsotope(num)
        atom.SetProp(RGROUP_PROP, str(num))
    return rdreactant

def oligomer_to_substituent(oligomer, r_group_number=None):
    rdsub = oligomer.offmol.to_rdkit()
    for index, oligindex in oligomer.atom_oligomer_map.items():
        atom = rdsub.GetAtomWithIdx(int(index))
        atom.SetProp(OLIGOMER_ATOM_INDEX_PROP, str(oligindex))
    r_group_index = oligomer.r_group_indices[r_group_number]
    sub_atom = rdsub.GetAtomWithIdx(int(r_group_index))
    sub_atom.SetAtomMapNum(SUBSTITUENT_ATOMNUM)
    for num, index in oligomer.r_group_indices.items():
        atom = rdsub.GetAtomWithIdx(index)
        atom.SetIsotope(num)
        atom.SetProp(RGROUP_PROP, str(num))
    # sub_atom.SetIsotope(SUBSTITUENT_ISOTOPE)
    return rdsub

def attach_substituent(
        reactant_r_group: int,
        substituent_r_group: int,
        reactant: "Oligomer",
        substituent: "Oligomer",
    ) -> rdkit.Chem.Mol:
    rdreactant = oligomer_to_reactant(reactant, reactant_r_group)
    rdsub = oligomer_to_substituent(substituent, substituent_r_group)
    # run reaction
    reaction = f"[*:{SUBSTITUENT_ATOMNUM}]-[{reactant_r_group}*]>>"
    reaction += utils.mol_to_smiles(rdsub)
    rxn = rdChemReactions.ReactionFromSmarts(reaction)
    products = rxn.RunReactants((rdreactant,))
    assert len(products) == 1 and len(products[0]) == 1
    product = products[0][0]
    Chem.SanitizeMol(product)
    return product

def attach_substituent_and_label(
        reactant_r_group: int,
        substituent_r_group: int,
        reactant: "Oligomer",
        substituent: "Oligomer",
    ) -> "Oligomer":
    product = attach_substituent(reactant_r_group, substituent_r_group, reactant, substituent)
    central_atom_map = {}
    r_groups = {}
    atom_oligomer_map = {}
    for atom in product.GetAtoms():
        num = atom.GetAtomMapNum()
        idx = atom.GetIdx()
        if num != 0:
            central_atom_map[idx] = num
            atom.SetAtomMapNum(0)
        if atom.HasProp(RGROUP_PROP):
            r_group = int(atom.GetProp(RGROUP_PROP))
            central_atom_map[idx] = r_group
            r_groups[idx] = r_group
        # TODO: tidy this up.....
        elif atom.GetSymbol() == "*":
            r_groups[idx] = atom.GetIsotope()
        if atom.HasProp(OLIGOMER_ATOM_INDEX_PROP):
            string = atom.GetProp(OLIGOMER_ATOM_INDEX_PROP)
            atom_oligomer_map[idx] = tuple_from_string(string)

    offmol = OFFMolecule.from_rdkit(product, allow_undefined_stereo=True)
    offmol.properties["atom_map"] = central_atom_map
    offmol.properties["r_groups"] = r_groups

    return type(substituent).from_offmolecule(offmol,
                                              central_atom_map=central_atom_map,
                                              atom_oligomer_map=atom_oligomer_map)
