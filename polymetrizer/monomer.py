import warnings
from typing import Union, Dict, List, Iterable, Optional, Tuple
import itertools
import functools
from collections import defaultdict

import numpy as np

from . import utils, offutils, ommutils


class Monomer:
    @classmethod
    def from_dummy_smiles(cls, dummy_smiles: str, **kwargs) -> "Monomer":
        r_group_smiles = utils.replace_dummy_with_R(dummy_smiles)
        return cls(r_group_smiles, **kwargs)

    @classmethod
    def from_offmolecule(cls, offmol, **kwargs) -> "Monomer":
        r_smiles = offmol.to_smiles(mapped=True)
        new = cls(r_smiles, **kwargs)
        new._setup_atom_accounting(offmol)
        return new
    

    def __init__(
            self,
            r_group_smiles: Union[str, "Monomer"],
            central_atom_map: Dict[int, int] = {},
        ):
        if isinstance(r_group_smiles, type(self)):
            central_atom_map = r_group_smiles.central_atom_map
            r_group_smiles = r_group_smiles.r_group_smiles
            
        self.r_group_smiles = r_group_smiles
        self.r_group_numbers = utils.get_r_group_numbers_from_smiles(r_group_smiles)
        self.dummy_smiles = utils.replace_R_with_dummy(r_group_smiles)
        self.central_atom_map = dict(central_atom_map)
        self._setup_atom_accounting()

    def _setup_atom_accounting(self, offmol=None):
        if offmol is not None:
            dummy_offmol = offutils.mol_from_smiles(self.dummy_smiles)
            assert offmol.n_atoms == dummy_offmol.n_atoms
            # print(self.dummy_smiles)
            # print(self.offmol.to_smiles())
            # assert len(offmol.chemical_environment_matches(self.dummy_smiles))
        else:
            offmol = offutils.mol_from_smiles(self.dummy_smiles)
            atom_map = offmol.properties.get("atom_map", {})
            r_groups = {}
            for i, atom in enumerate(offmol.atoms):
                if atom.atomic_number == 0:
                    r_groups[i] = atom_map.get(i, i + 1)
            offmol.properties["r_groups"] = r_groups

        self.offmol = offmol
        self.indices = np.arange(offmol.n_atoms)

        # atom accounting
        self.r_group_indices = {}
        self.r_bond_atom_indices = {}
        
        given_empty_atom_map = not self.central_atom_map
        off_map = self.offmol.properties.get("r_groups", {})
        for i, atom in enumerate(self.offmol.atoms):
            if atom.atomic_number == 0:  # dummy
                try:
                    num = off_map[i]
                except KeyError:
                    raise ValueError("`r_group_smiles` must have "
                                    "*numbered* R groups. Given: "
                                    + self.r_group_smiles)
                self.r_group_indices[num] = i
                for bond in atom.bonds:
                    index = offutils.get_other_bond_index(bond, i)
                    self.r_bond_atom_indices[num] = index
            elif given_empty_atom_map:
                self.central_atom_map[i] = len(self.central_atom_map) + 1
        self.offmol.properties["atom_map"] = self.central_atom_map

    def get_applicable_caps(
            self,
            r_group_substituents: Dict[int, List[Tuple[int, "Monomer"]]] = {},
            ignore_r: int = 0,
        ) -> List[Dict[int, "Monomer"]]:
        """
        Get the applicable substituents for all R-groups, except
        the ignored number.
        """
        r_nums = [r for r in self.r_group_numbers if r != ignore_r]
        # TODO: should I let this go through without fulfilling all Rs?
        keys, r_subs = [], []
        r_subs = [r_group_substituents.get(r, []) for r in r_nums]
        return [dict(zip(r_nums, x)) for x in itertools.product(*r_subs)]

    def get_cap_combinations(
            self,
            r_group_substituents: Dict[int, List[Tuple[int, "Monomer"]]] = {},
        ) -> Dict[int, List[Dict[int, Tuple[int, "Monomer"]]]]:
        combinations = {}
        for r in self.r_group_numbers:
            substituents = self.get_applicable_caps(r_group_substituents,
                                                    ignore_r=r)
            combinations[r] = substituents
        return combinations

    def get_forcefield_parameters(self, forcefield):
        system = forcefield.create_openmm_system(self.offmol.to_topology())
        handler_kwargs = {}
        for force in system.getForces():
            try:
                parser = ommutils.OPENMM_FORCE_PARSERS[type(force)]
            except KeyError:
                continue
            handler_kwargs.update(parser(force))
        return handler_kwargs

    def get_central_and_neighbor_indices(self, n_neighbors: int = 3) -> List[int]:
        seen = set(self.central_atom_map)
        layer = seen

        while n_neighbors:
            new_layer = set()
            for i in layer:
                for bond in self.offmol.atoms[i].bonds:
                    j = offutils.get_other_bond_index(bond, i)
                    if j not in seen:
                        new_layer.add(j)
            layer = new_layer
            seen |= layer
            n_neighbors -= 1

        return sorted(seen)

    



