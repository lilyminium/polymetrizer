import warnings
from typing import Union, Dict, List, Iterable, Tuple
import itertools
import functools
from collections import defaultdict

import numpy as np

from .monomer import Monomer
from . import utils, offutils

try:
    from .oefuncs import attach_substituent_and_label, get_sub_smarts, subset_mol
except ImportError:
    from .rdfuncs import attach_substituent_and_label, get_sub_smarts, subset_mol


class Oligomer(Monomer):

    @classmethod
    def with_oligomer_index(cls, *args, index: int=0, **kwargs):
        new = cls(*args, **kwargs)
        new.atom_oligomer_map = {i: (index, i) for i in new.indices}
        return new

    def __init__(
            self,
            r_group_smiles: Union[str, "Oligomer"],
            central_atom_map: Dict[int, int] = {},
            atom_oligomer_map: Dict[int, Tuple[int, int]] = {},
        ):
        if isinstance(r_group_smiles, Monomer):
            central_atom_map = r_group_smiles.central_atom_map
            if isinstance(r_group_smiles, type(self)):
                atom_oligomer_map = r_group_smiles.atom_oligomer_map
            r_group_smiles = r_group_smiles.r_group_smiles
        
        super().__init__(r_group_smiles, central_atom_map)
        self.atom_oligomer_map = atom_oligomer_map

    def attach_substituents(
            self,
            r_group_substituents: Dict[int, "Oligomer"] = {},
        ) -> "Oligomer":
        rct = self
        for rct_r, (sub_r, sub) in r_group_substituents.items():
            print(rct_r, sub_r)
            rct = attach_substituent_and_label(rct_r, sub_r, rct, sub)
        return rct

    def generate_substituted_caps(
            self,
            r_group_substituents: Dict[int, List[Tuple[int, "Oligomer"]]] = {},
        ) -> Dict[int, "Oligomer"]:
        combinations = self.get_cap_combinations(r_group_substituents)
        substituted = defaultdict(list)
        for num, groups in combinations.items():
            for group in groups:
                substituted[num].append(self.attach_substituents(group))
        return substituted

    def generate_substituted(
            self,
            r_group_substituents: Dict[int, List[Tuple[int, "Oligomer"]]] = {},
        ) -> List["Oligomer"]:
        combinations = self.get_applicable_caps(r_group_substituents)
        substituted = []
        for group in combinations:
            substituted.append(self.attach_substituents(group))
        return substituted


    def _build_next_combination(
            self,
            r_group_substituents: Dict[int, List["Oligomer"]] = {},
        ):
        groups = {}
        for r in self.r_group_numbers:
            try:
                groups[r] = r_group_substituents[r].pop(0)
            except IndexError:
                return self
        print("building next")
        substituted = self.attach_substituents(groups)
        return self._build_next_combination(r_group_substituents)

    
    def build_all_combinations(
            self,
            r_group_substituents: Dict[int, List["Oligomer"]] = {},
            repeat: int = 1,
        ):
        # TODO: pretty sure this logic is wrong for residues
        # with multiple linkages
        caps = {}
        for k, v in r_group_substituents.items():
            caps[k] = v * repeat
        return self._build_next_combination(caps)

    

    def fragment_around_central_atoms(self, fragmenter) -> "Oligomer":
        if isinstance(fragmenter, type):
            fragmenter = fragmenter()
        results = fragmenter.fragment(self.offmol)
        fragments = []
        central_atom_indices = set(self.central_atom_map)

        # find fragments that overlap with central region
        fragment_indices = set()
        for fragment in results.fragments:
            mol = fragment.molecule
            indices = set(i - 1 for i in mol.properties["atom_map"].values())
            if indices.intersection(central_atom_indices):
                fragment_indices |= indices
        
        atom_indices = sorted(fragment_indices)
        newmol = subset_mol(self.offmol, atom_indices)

        new_atom_map = {}
        new_oligomer_map = {}

        for new_index, old_index in enumerate(atom_indices):
            try:
                new_atom_map[new_index] = self.central_atom_map[old_index]
            except KeyError:
                pass

            try:
                new_oligomer_map[new_index] = self.atom_oligomer_map[old_index]
            except KeyError:
                pass

        return type(self).from_offmolecule(newmol,
                                           central_atom_map=new_atom_map,
                                           atom_oligomer_map=new_oligomer_map)

    

    def get_central_forcefield_parameters(self, forcefield, n_neighbors: int=3):
        handler_kwargs = self.get_forcefield_parameters(forcefield)
        relevant_indices = self.get_central_and_neighbor_indices(n_neighbors)
        central_kwargs = {}
        for handler_name, atomgroup_kwargs in handler_kwargs.items():
            central_handler = {}
            for indices, hkwargs in atomgroup_kwargs.items():
                if not all(i in relevant_indices for i in indices):
                    continue
                if not all(i in self.atom_oligomer_map for i in indices):
                    continue
                qualified = tuple(self.atom_oligomer_map[i] for i in indices)
                central_handler[qualified] = hkwargs
            central_kwargs[handler_name] = central_handler
        return central_kwargs

    def contains_qualified_atoms(
            self,
            qualified_atoms: List[Tuple[int, int]],
            handler_name: str = None,
        ) -> bool:
        qualified_atom_set = set(qualified_atoms)
        qualified = set(self.atom_oligomer_map.values())
        if not qualified_atom_set.issubset(qualified):
            return False
        qualified_to_index = {v: k for k, v in self.atom_oligomer_map.items()}

        # for stuff that stretches between residues,
        # first three atoms are always bonded A-B-C.
        # in proper torsions, C-D are bonded; in improper, B-D.

        for i, atom_index in enumerate(qualified_atoms[1:3]):
            previous_index = qualified_atom_indices[i]
            previous_atom = self.offmol.atoms[previous_index]
            for bond in previous_atom.bonds:
                other = offutils.get_other_bond_index(bond, previous_index)
                if other == atom_index:
                    break
            else:
                return False
        if len(qualified_atoms) == 4:
            last_index = qualified_to_index[qualified_atoms[-1]]
            last = self.offmols.atoms[last_index]
            if handler_name == "ImproperTorsions":
                bonded_index = qualified_to_index[qualified_atoms[1]]
            elif handler_name == "ProperTorsions":
                bonded_index = qualified_to_index[qualified_atoms[2]]
            for bond in last.bonds:
                other = offutils.get_other_bond_index(bond, last_index)
                if other == bonded_index:
                    break
            else:
                return False
        return True


    def get_residue_smirks(
            self,
            qualified_atoms: List[Tuple[int, int]],
            handler_name: str = None,
            include_atom_indices: list = [],
        ) -> str:
        if not self.contains_qualified_atoms(qualified_atoms, handler_name):
            return
        qualified_to_index = {v: k for k, v in self.atom_oligomer_map.items()}
        label_indices = [qualified_to_index[i] for i in qualified_atoms]
        atom_indices = list(include_atom_indices)
        monomers, _ = zip(*qualified_atoms)
        for (monomer, atom), index in qualified_to_index.items():
            if monomer in monomers:
                atom_indices.append(index)
        return get_sub_smarts(self.offmol, atom_indices, label_indices)
