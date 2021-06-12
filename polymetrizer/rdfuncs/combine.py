from typing import Tuple
import itertools

import rdkit
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from openff.toolkit.topology import Molecule as OFFMolecule


from ..utils import tuple_from_string
from . import utils

SUBSTITUENT_ATOMNUM = 2

def get_wildcard_indices_and_nums(rdmol):
    indices = []
    nums = []
    for i, atom in enumerate(rdmol.GetAtoms()):
        if atom.GetSymbol() == "*":
            indices.append(i)
            nums.append(atom.GetAtomMapNum())
    return indices, nums

def match_with_atom_map_wildcard(target, reference):
    matches = []
    dummy_indices, dummy_nums = get_wildcard_indices_and_nums(target)
    clean_target = utils.mol_from_smiles(utils.mol_to_smiles(target))
    mapping = clean_target.GetSubstructMatch(target)
    assert len(mapping) == target.GetNumAtoms()
    for clean_match in reference.GetSubstructMatches(clean_target, uniquify=True):
        # convert to original mol
        match = [clean_match[i] for i in mapping]
        for target_index, target_num in zip(dummy_indices, dummy_nums):
            ref_index = match[target_index]
            ref_atom = reference.GetAtomWithIdx(ref_index)
            if not ref_atom.GetSymbol() == "*" and ref_atom.GetAtomMapNum() == target_num:
                break
        else:
            matches.append(match)
    return matches

def oligomer_to_reactant(oligomer, r_group_number=None):
    rdreactant = oligomer.offmol.to_rdkit()
    for atom in rdreactant.GetAtoms():
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    if r_group_number is None:
        return rdreactant
    r_group_index = oligomer.r_group_indices[r_group_number]
    rdreactant.GetAtomWithIdx(r_group_index).SetIsotope(r_group_number)
    return rdreactant, r_group_index

def oligomer_to_substituent(oligomer, r_group_number=None):
    rdsub = oligomer.offmol.to_rdkit()
    for atom in rdsub.GetAtoms():
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)

    r_group_index = oligomer.r_group_indices[r_group_number]
    sub_atom = rdsub.GetAtomWithIdx(int(r_group_index))
    sub_atom.SetAtomMapNum(SUBSTITUENT_ATOMNUM)
    return rdsub, r_group_index

def attach_substituent_and_label(
        reactant_r_group: int,
        substituent_r_group: int,
        reactant: "Oligomer",
        substituent: "Oligomer",
    ) -> "Oligomer":

    rdreactant, remove_reactant = oligomer_to_reactant(reactant, reactant_r_group)
    rdsub, remove_sub = oligomer_to_substituent(substituent, substituent_r_group)

    reaction = f"[*:{SUBSTITUENT_ATOMNUM}]-[{reactant_r_group}*]>>"
    reaction += utils.mol_to_smiles(rdsub)
    
    rxn = rdChemReactions.ReactionFromSmarts(reaction)
    products = rxn.RunReactants((rdreactant,))
    assert len(products) == 1 and len(products[0]) == 1
    product = products[0][0]
    Chem.SanitizeMol(product)
    reactant_indices = [i for i in range(rdreactant.GetNumAtoms()) if i != remove_reactant]
    sub_indices = [i for i in range(rdsub.GetNumAtoms()) if i != remove_sub]

    
    rdreactant = Chem.RWMol(rdreactant)
    rdsub = Chem.RWMol(rdsub)
    rdreactant.RemoveAtom(remove_reactant)
    rdsub.RemoveAtom(remove_sub)
    Chem.SanitizeMol(rdsub)
    Chem.SanitizeMol(rdreactant)

    rc_matches = match_with_atom_map_wildcard(rdreactant, product)
    sub_matches = match_with_atom_map_wildcard(rdsub, product)
    assert rc_matches and sub_matches

    # central_atom_map = {}
    atom_oligomer_map = {}
    r_groups = {}
    monomer_bonds = set()
    central_atom_indices = []

    for rc, sb in itertools.product(rc_matches, sub_matches):
        if not set(rc) & set(sb):
            for match_index, product_index in enumerate(rc):
                reactant_index = reactant_indices[match_index]
                if reactant_index in reactant.central_atom_indices:
                    central_atom_indices.append(product_index)
            
            groups = [(rc, reactant, reactant_indices),
                      (sb, substituent, sub_indices)]
            for match, oligomer, indices in groups:
                properties = [(atom_oligomer_map, oligomer.atom_oligomer_map),
                              (r_groups, oligomer.reverse_r_group_indices)]
                origin_to_product = {}
                for match_index, product_index in enumerate(match):
                    index = indices[match_index]
                    origin_to_product[index] = product_index
                    for destination, origin in properties:
                        try:
                            destination[product_index] = origin[index]
                        except KeyError:
                            pass
                for atom1, atom2 in oligomer.monomer_bonds:
                    new_atom1 = origin_to_product[atom1]
                    new_atom2 = origin_to_product[atom2]
                    bond = (new_atom1, new_atom2)
                    monomer_bonds.add((new_atom1, new_atom2))
            for atom1, atom2 in itertools.product(rc, sb):
                if product.GetBondBetweenAtoms(atom1, atom2):
                    monomer_bonds.add((atom1, atom2))
            break
    else:
        raise ValueError("could not match reactant and substituent")
            
    offmol = utils.mol_to_offmol(product)

    offmol.properties["atom_map"] = r_groups

    return type(reactant)(offmol,
                          central_atom_indices=central_atom_indices,
                          atom_oligomer_map=atom_oligomer_map,
                          monomer_bonds=monomer_bonds)
