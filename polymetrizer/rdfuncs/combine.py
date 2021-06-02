from typing import Tuple
import itertools

import rdkit
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from openff.toolkit.topology import Molecule as OFFMolecule


from ..utils import tuple_from_string
from . import utils

SUBSTITUENT_ISOTOPE = 2
SUBSTITUENT_ATOMNUM = 2
OLIGOMER_ATOM_INDEX_PROP = 'oligomerAtomIndex'
ATOM_MAP_PROP = 'centralAtomMap'
RGROUP_PROP = 'RGroupNum'
BYSTANDER_PROP = "isBystander"

CUSTOM_KWARGS = [OLIGOMER_ATOM_INDEX_PROP, RGROUP_PROP, BYSTANDER_PROP]

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



# def replace_props(target, pattern, reverse=False):
#     matches = utils.substructure_search(target, pattern)
#     if not matches:
#         return target
    
#     n_oligomer_props = []
#     for match in matches:
#         n_props = 0
#         n_matching_props = 0
#         for i, index in enumerate(match):
#             atom = target.GetAtomWithIdx(index)
#             if atom.HasProp(OLIGOMER_ATOM_INDEX_PROP):
#                 n_props += 1
#                 atom2 = pattern.GetAtomWithIdx(i)
#                 if not atom2.HasProp(OLIGOMER_ATOM_INDEX_PROP):
#                     continue
#                 if atom2.GetProp(OLIGOMER_ATOM_INDEX_PROP) == atom.GetProp(OLIGOMER_ATOM_INDEX_PROP):
#                     n_matching_props += 1
#         n_oligomer_props.append((n_matching_props, match))
    
#     for *_, match in sorted(n_oligomer_props, reverse=reverse):
#         empty = True
#         for i, pidx in enumerate(match):
#             atom = target.GetAtomWithIdx(pidx)
#             if atom.HasProp(OLIGOMER_ATOM_INDEX_PROP) or atom.HasProp(ATOM_MAP_PROP):
#                 empty = False
#         if empty:
#             for i, pidx in enumerate(match):
#                 props = pattern.GetAtomWithIdx(i).GetPropsAsDict()
#                 atom = target.GetAtomWithIdx(pidx)
#                 for k, v in props.items():
#                     if k in CUSTOM_KWARGS:
#                         atom.SetProp(k, str(v))
#             break
#     return target

def oligomer_to_reactant(oligomer, r_group_number=None):
    rdreactant = oligomer.offmol.to_rdkit()
    for atom in rdreactant.GetAtoms():
        # if atom.GetSymbol() != "*":
        #     atom.SetProp(BYSTANDER_PROP, "Y")
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    # for index, num in oligomer.central_atom_map.items():
    #     atom = rdreactant.GetAtomWithIdx(int(index))
    #     # atom.SetAtomMapNum(int(num))
    #     atom.SetProp(ATOM_MAP_PROP, str(num))
    # for index, oligindex in oligomer.atom_oligomer_map.items():
    #     atom = rdreactant.GetAtomWithIdx(int(index))
    #     atom.SetProp(OLIGOMER_ATOM_INDEX_PROP, str(oligindex))
    if r_group_number is None:
        return rdreactant
    r_group_index = oligomer.r_group_indices[r_group_number]
    rdreactant.GetAtomWithIdx(r_group_index).SetIsotope(r_group_number)
    for num, index in oligomer.r_group_indices.items():
        atom = rdreactant.GetAtomWithIdx(index)
        atom.SetIsotope(num)
        atom.SetAtomMapNum(num)
        atom.SetProp(RGROUP_PROP, str(num))
    return rdreactant, r_group_index

def oligomer_to_substituent(oligomer, r_group_number=None):
    rdsub = oligomer.offmol.to_rdkit()
    for atom in rdsub.GetAtoms():
        # if atom.GetSymbol() != "*":
        #     atom.SetProp(BYSTANDER_PROP, "Y")
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    # for index, oligindex in oligomer.atom_oligomer_map.items():
    #     atom = rdsub.GetAtomWithIdx(int(index))
    #     atom.SetProp(OLIGOMER_ATOM_INDEX_PROP, str(oligindex))

    r_group_index = oligomer.r_group_indices[r_group_number]
    sub_atom = rdsub.GetAtomWithIdx(int(r_group_index))
    sub_atom.SetAtomMapNum(SUBSTITUENT_ATOMNUM)
    for num, index in oligomer.r_group_indices.items():
        atom = rdsub.GetAtomWithIdx(index)
        atom.SetIsotope(num)
        atom.SetProp(RGROUP_PROP, str(num))
    # sub_atom.SetIsotope(SUBSTITUENT_ISOTOPE)
    
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
    # print(reaction)
    
    rxn = rdChemReactions.ReactionFromSmarts(reaction)
    products = rxn.RunReactants((rdreactant,))
    assert len(products) == 1 and len(products[0]) == 1
    product = products[0][0]
    Chem.SanitizeMol(product)
    # print(utils.mol_to_smiles(product))
    # print("----")

    # match reactant and sub to product
    # remove_reactant = 0
    reactant_indices = [i for i in range(rdreactant.GetNumAtoms()) if i != remove_reactant]
    # remove_sub = 0
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

    central_atom_map = {}
    atom_oligomer_map = {}
    r_groups = {}

    for rc, sb in itertools.product(rc_matches, sub_matches):
        if not set(rc) & set(sb):
            groups = [(rc, reactant, reactant_indices),
                      (sb, substituent, sub_indices)]
            for i, (match, oligomer, indices) in enumerate(groups):
                for m_idx, prod_idx in enumerate(match):
                    idx = indices[m_idx]
                    rev_r_groups = {v: k for k, v in oligomer.r_group_indices.items()} #dict(oligomer.r_group_indices.items())
                    pairs = [(central_atom_map, oligomer.central_atom_map),
                             (atom_oligomer_map, oligomer.atom_oligomer_map),
                             (r_groups, rev_r_groups)][i:]  # shortcut to not copy central atoms from sub
                    for new_map, old_map in pairs:
                        try:
                            new_map[prod_idx] = old_map[idx]
                        except KeyError:
                            pass
                    
            break
    else:
        # print(Chem.MolToSmiles(product))
        # print(Chem.MolToSmiles(rdreactant))
        # print(Chem.MolToSmiles(rdsub))
        # print(rc_matches)
        # print(sub_matches)
        # print("--")
        # print("1.", rdreactant.GetSubstructMatches(rdreactant))
        # print("2.", product.GetSubstructMatches(rdreactant))
        # print("3.", rdreactant.GetSubstructMatches(utils.mol_from_smiles("[H]C([H])N([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])N([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])N([H])C12C([H])([H])C3([H])C([H])([H])C([H])(C([H])([H])C([H])(C3([H])[H])C1([H])[H])C2([H])[H]")))
        # print("4.", product.GetSubstructMatches(utils.mol_from_smiles("[H]C([H])N([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])N([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])N([H])C12C([H])([H])C3([H])C([H])([H])C([H])(C([H])([H])C([H])(C3([H])[H])C1([H])[H])C2([H])[H]")))
        # print("5.", product.GetSubstructMatches(utils.mol_from_smiles("[H]C([H])C([H])([H])C([H])([H])N([H])C12C([H])([H])C3([H])C([H])([H])C([H])(C([H])([H])C([H])(C3([H])[H])C1([H])[H])C2([H])[H]")))
        # # print(product.GetSubstructMatches(Chem.MolFromSmiles("[H]C([H])N([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])N([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])N([H])C12C([H])([H])C3([H])C([H])([H])C([H])(C([H])([H])C([H])(C3([H])[H])C1([H])[H])C2([H])[H]")))
        # # print(product.GetSubstructMatches(Chem.MolFromSmiles("[H]C([H])C([H])([H])C([H])([H])N([H])C12C([H])([H])C3([H])C([H])([H])C([H])(C([H])([H])C([H])(C3([H])[H])C1([H])[H])C2([H])[H]")))
        raise ValueError("could not match reactant and substituent")
            
    # replace_props(product, rdreactant, reverse=True)

    
    offmol = utils.offmol_from_mol(product)

    offmol.properties["atom_map"] = central_atom_map
    offmol.properties["r_groups"] = r_groups

    return type(substituent).from_offmolecule(offmol,
                                              central_atom_map=central_atom_map,
                                              atom_oligomer_map=atom_oligomer_map)
