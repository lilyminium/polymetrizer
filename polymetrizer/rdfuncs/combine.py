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
ATOM_MAP_PROP = 'centralAtomMap'
RGROUP_PROP = 'RGroupNum'
BYSTANDER_PROP = "isBystander"

CUSTOM_KWARGS = [OLIGOMER_ATOM_INDEX_PROP, ATOM_MAP_PROP, RGROUP_PROP, BYSTANDER_PROP]

def replace_props(target, pattern, reverse=False):
    matches = utils.substructure_search(target, pattern)
    if not matches:
        return target
    
    n_oligomer_props = []
    for match in matches:
        n_props = 0
        n_matching_props = 0
        for i, index in enumerate(match):
            atom = target.GetAtomWithIdx(index)
            if atom.HasProp(OLIGOMER_ATOM_INDEX_PROP):
                n_props += 1
                atom2 = pattern.GetAtomWithIdx(i)
                if not atom2.HasProp(OLIGOMER_ATOM_INDEX_PROP):
                    continue
                if atom2.GetProp(OLIGOMER_ATOM_INDEX_PROP) == atom.GetProp(OLIGOMER_ATOM_INDEX_PROP):
                    n_matching_props += 1
        n_oligomer_props.append((n_props, n_matching_props, match))
    
    
    for i, pidx in enumerate(sorted(n_oligomer_props, reverse=reverse)[0][-1]):
        props = pattern.GetAtomWithIdx(i).GetPropsAsDict()
        atom = target.GetAtomWithIdx(pidx)
        for k, v in props.items():
            if k in CUSTOM_KWARGS:
                atom.SetProp(k, str(v))
    return target

def oligomer_to_reactant(oligomer, r_group_number=None):
    rdreactant = oligomer.offmol.to_rdkit()
    for atom in rdreactant.GetAtoms():
        if atom.GetSymbol() != "*":
            atom.SetProp(BYSTANDER_PROP, "Y")
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    for index, num in oligomer.central_atom_map.items():
        atom = rdreactant.GetAtomWithIdx(int(index))
        # atom.SetAtomMapNum(int(num))
        atom.SetProp(ATOM_MAP_PROP, str(num))
    for index, oligindex in oligomer.atom_oligomer_map.items():
        atom = rdreactant.GetAtomWithIdx(int(index))
        atom.SetProp(OLIGOMER_ATOM_INDEX_PROP, str(oligindex))
    if r_group_number is None:
        return rdreactant
    r_group_index = oligomer.r_group_indices[r_group_number]
    rdreactant.GetAtomWithIdx(r_group_index).SetIsotope(r_group_number)
    for num, index in oligomer.r_group_indices.items():
        atom = rdreactant.GetAtomWithIdx(index)
        atom.SetIsotope(num)
        atom.SetAtomMapNum(num)
        atom.SetProp(RGROUP_PROP, str(num))
    return rdreactant

def oligomer_to_substituent(oligomer, r_group_number=None):
    rdsub = oligomer.offmol.to_rdkit()
    for atom in rdsub.GetAtoms():
        if atom.GetSymbol() != "*":
            atom.SetProp(BYSTANDER_PROP, "Y")
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
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

    # print(utils.mol_to_smiles(rdreactant))
    # print(utils.mol_to_smiles(rdsub))
    # 
    # run reaction
    reaction = f"[*:{SUBSTITUENT_ATOMNUM}]-[{reactant_r_group}*]>>"
    reaction += utils.mol_to_smiles(rdsub)
    # print(reaction)
    
    rxn = rdChemReactions.ReactionFromSmarts(reaction)
    products = rxn.RunReactants((rdreactant,))
    assert len(products) == 1 and len(products[0]) == 1
    product = products[0][0]

    # replace_props(product, rdreactant, reverse=True)
    replace_props(product, rdsub, reverse=False)

    Chem.SanitizeMol(product)
    return product

def attach_substituent_and_label(
        reactant_r_group: int,
        substituent_r_group: int,
        reactant: "Oligomer",
        substituent: "Oligomer",
    ) -> "Oligomer":
    product = attach_substituent(reactant_r_group, substituent_r_group, reactant, substituent)

    # utils.assign_stereochemistry(product)
    central_atom_map = {}
    r_groups = {}
    atom_oligomer_map = {}
    for atom in product.GetAtoms():
        idx = atom.GetIdx()
        atom.SetAtomMapNum(0)
        if atom.HasProp(ATOM_MAP_PROP):
            central_atom_map[idx] = int(atom.GetProp(ATOM_MAP_PROP))
        
        if atom.HasProp(RGROUP_PROP) and atom.GetSymbol() == "*":
            r_group = int(atom.GetProp(RGROUP_PROP))
            # central_atom_map[idx] = r_group
            r_groups[idx] = r_group
        # TODO: tidy this up.....
        elif atom.GetSymbol() == "*":
            r_groups[idx] = atom.GetIsotope()
        if atom.HasProp(OLIGOMER_ATOM_INDEX_PROP):
            string = atom.GetProp(OLIGOMER_ATOM_INDEX_PROP)
            atom_oligomer_map[idx] = tuple_from_string(string)
        atom.SetIsotope(0)

    reverse = reactant.reverse_central_atom_map
    for i, num in central_atom_map.items():
        if i not in atom_oligomer_map:
            original_i = reverse[num]
            try:
                atom_oligomer_map[i] = reactant.atom_oligomer_map[original_i]
            except KeyError:
                pass
    
    matches = utils.substructure_search(product, reactant.offmol.to_rdkit())
    n_matches = []
    for match in matches:
        n = 0
        for i, index in enumerate(match):
            if index in atom_oligomer_map and i in reactant.atom_oligomer_map:
                if atom_oligomer_map[index] == reactant.atom_oligomer_map[i]:
                    n += 1
                else:
                    n -= 2
        n_matches.append((n, match))
    n_matches = sorted(n_matches, reverse=True)
    for i, index in enumerate(n_matches[0][1]):
        if i in reactant.atom_oligomer_map:
            # if index not in atom_oligomer_map:
            atom_oligomer_map[index] = reactant.atom_oligomer_map[i]
        # else:
            
    
    matches = utils.substructure_search(product, substituent.offmol.to_rdkit())
    n_matches = []
    for match in matches:
        n = 0
        for i, index in enumerate(match):
            if index in atom_oligomer_map and i in substituent.atom_oligomer_map:
                if atom_oligomer_map[index] == substituent.atom_oligomer_map[i]:
                    n += 1
                else:
                    n -= 2
        n_matches.append((n, match))
    n_matches = sorted(n_matches, reverse=True)
    for i, index in enumerate(n_matches[0][1]):
        
        if i in substituent.atom_oligomer_map:
            if index not in atom_oligomer_map:
                atom_oligomer_map[index] = substituent.atom_oligomer_map[i]
    
    
    offmol = utils.offmol_from_mol(product)

    # unfound = [x for x in reactant.oligomer_bond_map if x not in atom_oligomer_map.values()]
    # ignore = []
    # print("unfound", unfound)
    # # while unfound:
    # oligomer_atom_map = {v: k for k, v in atom_oligomer_map.items()}
    # for qual in unfound:
    #     partners = reactant.oligomer_bond_map[qual]
    #     available = [p for p in partners if p in oligomer_atom_map]
    #     # print(available, partners, oligomer_atom_map)
    #     if not available:
    #         continue
    #     available_indices = [oligomer_atom_map[i] for i in available]
    #     bonded = []
    #     for atom in offmol.atoms[available_indices[0]].bonded_atoms:
    #         if atom not in atom_oligomer_map:
    #             bonded.append(atom.molecule_atom_index)
    #     bonded = set(bonded)
    #     for index in available_indices[1:]:
    #         bonded_ = set([a.molecule_atom_index for a in offmol.atoms[index].bonded_atoms])
    #         bonded &= bonded_
    #     if len(bonded):
    #         for index in bonded:
    #             if index not in atom_oligomer_map:
    #                 atom_oligomer_map[index] = qual
    #                 print(f"Added {qual} to {index}")
    #                 break
    #         # index = bonded.pop()
    #         # atom_oligomer_map[index] = qual
    #             if index not in central_atom_map:
    #                 original_index = reactant.reverse_atom_oligomer_map[qual]
    #                 central_atom_map[index] = reactant.central_atom_map[original_index]
    #     else:
    #         ignore.append(qual)
    # unfound = [x for x in reactant.oligomer_bond_map if x not in atom_oligomer_map.values()]
    # # unfound = [x for x in unfound if x not in ignore]
    

    # print("unfound", unfound)
    
    # unfound = [x for x in substituent.oligomer_bond_map if x not in atom_oligomer_map.values()]
    # ignore = []
    # print("---")
    # print("unfound", unfound)
    
    # # while unfound:
    # oligomer_atom_map = {v: k for k, v in atom_oligomer_map.items()}
    # for qual in unfound:
    #     partners = substituent.oligomer_bond_map[qual]
    #     available = [p for p in partners if p in oligomer_atom_map]
    #     print(qual)
    #     print(partners)
    #     print(available)
    #     # print(available, partners, oligomer_atom_map)
    #     if not available:
    #         continue
    #     available_indices = [oligomer_atom_map[i] for i in available]
    #     bonded = []
    #     for atom in offmol.atoms[available_indices[0]].bonded_atoms:
    #         if atom.molecule_atom_index not in atom_oligomer_map:
    #             bonded.append(atom.molecule_atom_index)
    #     bonded = set(bonded)
    #     print("bonded to ", qual, ":", bonded)
    #     for index in available_indices[1:]:
    #         bonded_ = set([a.molecule_atom_index for a in offmol.atoms[index].bonded_atoms])
    #         print(">> ", bonded_)
    #         bonded &= bonded_
    #     if len(bonded):
    #         for index in bonded:
    #             if index not in atom_oligomer_map:
    #                 atom_oligomer_map[index] = qual
    #                 print(f"Added {qual} to {index}")
    #                 break
    #         # if index not in central_atom_map:
    #         #     original_index = substituent.reverse_atom_oligomer_map[qual]
    #         #     central_atom_map[index] = substituent.central_atom_map[original_index]
    #     else:
    #         ignore.append(qual)
    # unfound = [x for x in substituent.oligomer_bond_map if x not in atom_oligomer_map.values()]
    # # unfound = [x for x in unfound if x not in ignore]
    # print("unfound", unfound)

    # print(atom_oligomer_map)
    # print("")
    

    
    
    offmol.properties["atom_map"] = central_atom_map
    offmol.properties["r_groups"] = r_groups

    # print("oligomer map in combine", atom_oligomer_map)

    return type(substituent).from_offmolecule(offmol,
                                              central_atom_map=central_atom_map,
                                              atom_oligomer_map=atom_oligomer_map)
