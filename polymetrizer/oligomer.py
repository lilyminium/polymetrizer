import warnings
from typing import Union, Dict, List, Iterable, Tuple, Optional, Set
import itertools
import functools
from collections import defaultdict

import numpy as np
from openff.toolkit.topology import Molecule as OFFMolecule

from . import utils, offutils, ommutils
from .smirker import SingleOligomerAtomGroupParameter

try:
    from .oefuncs import attach_substituent_and_label, get_sub_smarts, subset_mol
except ImportError:
    from .rdfuncs import attach_substituent_and_label, get_sub_smarts, subset_mol


@functools.total_ordering
class AtomWrapper:
    def __init__(self, monomer, index):
        self.monomer = monomer
        self.index = index

    @property
    def offatom(self):
        return self.monomer.offmol.atoms[self.index]

    def __eq__(self, other):
        return self.monomer is other.monomer and self.index == other.index

    def __hash__(self):
        return hash((self.monomer.offmol, self.index))

    def __lt__(self, other):
        self_values = (self.monomer, self.index)
        other_values = (other.monomer, other.index)
        return self_values < other_values


@functools.total_ordering
class Oligomer:

    """
    Class to hold Oligomer information.
    """

    @classmethod
    def with_oligomer_index(cls, *args, index: int=0, **kwargs):
        new = cls(*args, **kwargs)
        indices = [i for i in new.indices if new.offmol.atoms[i].atomic_number != 0]
        new.atom_oligomer_map = {i: (index, j) for j, i in enumerate(indices)}
        return new

    def __init__(
            self,
            offmol: OFFMolecule,
            central_atom_indices: List[int] = [],
            atom_oligomer_map: Dict[int, AtomWrapper] = {},
            monomer_bonds: Set[Tuple[int, int]] = set(),
        ):

        if isinstance(offmol, Oligomer):
            central_atom_indices = offmol.central_atom_indices
            atom_oligomer_map = offmol.atom_oligomer_map
            offmol = offmol.offmol

        self.offmol = offmol
        self.central_atom_indices = sorted(central_atom_indices)
        self.atom_oligomer_map = dict(atom_oligomer_map)
        self.r_group_indices = {}
        self.monomer_bonds = set()

        for bond in monomer_bonds:
            if bond[0] > bond[1]:
                bond = bond[::-1]
            self.monomer_bonds.add(bond)

        off_map = self.offmol.properties.get("atom_map", {})

        for i, atom in enumerate(self.offmol.atoms):
            if atom.atomic_number == 0:  # dummy
                try:
                    num = off_map[i]
                except KeyError:
                    msg = ("R groups must be labelled. "
                           f"The atom at index {i} does not have a number, "
                           "so it is ignored")
                    raise ValueError(msg)  # TODO: convert to warning to support dummy atoms
                else:
                    self.r_group_indices[num] = i

    def _get_immutable_attrs(self):
        oligomer_map = tuple(sorted(self.atom_oligomer_map.items()))
        indices = tuple(sorted(self.central_atom_indices))
        return (indices, self.offmol.to_smiles(), oligomer_map)

    def __hash__(self):
        return hash(self._get_immutable_attrs())
    
    def __eq__(self, other):
        return self._get_immutable_attrs() == other._get_immutable_attrs()
    
    def __lt__(self, other):
        self_values = (self.offmol.n_atoms, len(self.central_atom_indices), self.offmol.to_smiles())
        other_values = (other.offmol.n_atoms, len(other.central_atom_indices), other.offmol.to_smiles())
        return self_values < other_values

    @property
    def reverse_atom_oligomer_map(self):
        return {v: k for k, v in self.atom_oligomer_map.items()}

    @property
    def reverse_r_group_indices(self):
        return dict((v, k) for k, v in self.r_group_indices.items())


    def get_applicable_caps(
            self,
            r_group_substituents: Dict[int, List[Tuple[int, "Oligomer"]]] = {},
            ignore_r: int = 0,
        ) -> List[Dict[int, Tuple[int, "Oligomer"]]]:
        """
        Get the applicable substituents for all R-groups, except
        the ignored number.
        """
        r_nums = [r for r in self.r_group_indices if r != ignore_r]
        # TODO: should I let this go through without fulfilling all Rs?
        keys, r_subs = [], []
        r_subs = [r_group_substituents.get(r, []) for r in r_nums]
        return [dict(zip(r_nums, x)) for x in itertools.product(*r_subs)]
    

    def get_cap_combinations(
            self,
            r_group_substituents: Dict[int, List[Tuple[int, "Oligomer"]]] = {},
        ) -> Dict[int, List[Dict[int, Tuple[int, "Oligomer"]]]]:
        combinations = {}
        for r in self.r_group_indices:
            substituents = self.get_applicable_caps(r_group_substituents,
                                                    ignore_r=r)
            combinations[r] = substituents
        return combinations

    
    def get_central_and_neighbor_indices(self, n_neighbors: int = 3) -> List[int]:
        seen = set(self.central_atom_indices)
        layer = seen

        while n_neighbors:
            new_layer = set()
            for i in layer:
                for bond in self.offmol.atoms[i].bonds:
                    j = offutils.get_other_bond_index(bond, i)
                    if j not in seen and self.offmol.atoms[j].atomic_number != 0:
                        new_layer.add(j)
            layer = new_layer
            seen |= layer
            n_neighbors -= 1

        return sorted(seen)


    def attach_substituents(
            self,
            r_group_substituents: Dict[int, "Oligomer"] = {},
        ) -> "Oligomer":
        rct = Oligomer(self)
        for rct_r, (sub_r, sub) in r_group_substituents.items():
            expected_r_groups = {r for r in rct.r_group_indices if r != rct_r}
            expected_r_groups |= {r for r in sub.r_group_indices if r != sub_r}
            rct = attach_substituent_and_label(rct_r, sub_r, rct, sub)
            # symmetry can cause issues
            r_groups = set(rct.r_group_indices)
            if r_groups != expected_r_groups :
                try:
                    old = (r_groups - expected_r_groups).pop()
                    new = (expected_r_groups - r_groups).pop()
                    rct.r_group_indices[new] = rct.r_group_indices.pop(old)
                except KeyError:
                    pass
        return rct

    def generate_substituted_caps(
            self,
            r_group_substituents: Dict[int, List[Tuple[int, "Oligomer"]]] = {},
        ) -> Dict[int, "Oligomer"]:
        combinations = self.get_cap_combinations(r_group_substituents)
        substituted = defaultdict(list)
        for num, groups in combinations.items():
            for group in groups:
                attached = self.attach_substituents(group)
                substituted[num].append(attached)
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

    def build_all_combinations(
            self,
            r_group_substituents: Dict[int, List["Oligomer"]] = {},
            repeat: int = 1,
        ):
        # TODO: pretty sure this logic is wrong for residues
        # with multiple linkages
        substituted = self.generate_substituted(r_group_substituents)
        substituted = substituted[0]
        
        while substituted.r_group_indices:
            substituted = substituted.generate_substituted(r_group_substituents)
            substituted = substituted[0]
        return substituted

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
        newmol, atom_indices = subset_mol(self.offmol, atom_indices, check_bonds=True,
                                          return_atom_indices=True)

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

    def get_monomer_atoms(self, indices, ordered=True):
        try:
            monomer_atoms = tuple(self.atom_oligomer_map[i] for i in indices)
        except KeyError:
            return
        if ordered and monomer_atoms[0] > monomer_atoms[-1]:
            monomer_atoms = monomer_atoms[::-1]
        return monomer_atoms

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

    def select_relevant_parameters(self, handler_kwargs, relevant_indices):
        central_kwargs = {}
        for handler_name, atomgroup_kwargs in handler_kwargs.items():
            central_handler = {}
            for indices, hkwargs in atomgroup_kwargs.items():
                if not all(i in relevant_indices for i in indices):
                    continue
                if not all(i in self.atom_oligomer_map for i in indices):
                    continue
                param = SingleOligomerAtomGroupParameter(indices, self, hkwargs)
                central_handler[param.monomer_atoms] = param
            central_kwargs[handler_name] = central_handler
        return central_kwargs

    def get_central_forcefield_parameters(self, forcefield, n_neighbors: int=3):
        handler_kwargs = self.get_forcefield_parameters(forcefield)
        relevant_indices = self.get_central_and_neighbor_indices(n_neighbors)
        return self.select_relevant_parameters(handler_kwargs, relevant_indices)

    def get_monomer_atomgroup_indices(self, monomer_atoms: List[AtomWrapper]):
        atom_indices = [self.get_monomer_atom_indices(atom) for atom in monomer_atoms]
        return itertools.product(*atom_indices)

    def contains_monomer_atoms(
            self,
            monomer_atoms: List[AtomWrapper],
            handler_name: str = None,
            return_indices: bool = False,
        ):
        contained = set(self.atom_oligomer_map.values())

        if not all(atom in contained for atom in monomer_atoms):
            if return_indices:
                return False, []
            return False
        
        if len(monomer_atoms) == 1:
            if return_indices:
                return True, [(i,) for i, v in self.atom_oligomer_map.items() if v == monomer_atoms[0]]
            return True

        atom_indices = sorted(map(tuple, self.get_monomer_atomgroup_indices(monomer_atoms)))
        combinations = [[self.offmol.atoms[i] for i in ix] for ix in atom_indices]

        bond_check = lambda atoms: offutils.atoms_are_bonded(atoms[0], atoms[1])
        angle_check = lambda atoms: offutils.atoms_are_bonded(atoms[1], atoms[2])
        proper_check = lambda atoms: offutils.atoms_are_bonded(atoms[2], atoms[3])
        improper_check = lambda atoms: offutils.atoms_are_bonded(atoms[1], atoms[3])

        n_monomer_atoms = len(monomer_atoms)

        check_functions = [bond_check]
        if n_monomer_atoms > 2:
            check_functions.append(angle_check)
        if n_monomer_atoms == 4:
            if handler_name == "ImproperTorsions":
                check_functions.append(improper_check)
            else:
                check_functions.append(proper_check)

        valid_indices = []
        for i, atoms in enumerate(combinations):
            if all(check(atoms) for check in check_functions):
                valid_indices.append(atom_indices[i])
        if valid_indices:
            if return_indices:
                return True, valid_indices
            return True

        if return_indices:
            return False, valid_indices
        return False
    
    def get_monomer_atom_indices(self, monomer_atom: AtomWrapper):
        indices = []
        for index, atom in self.atom_oligomer_map.items():
            if atom == monomer_atom:
                indices.append(index)
        return sorted(indices)



class Monomer(Oligomer):

    def __init__(
            self,
            smiles_or_offmol: Union[str, OFFMolecule, "Oligomer"]
        ):
        if isinstance(smiles_or_offmol, str):
            smiles = utils.replace_R_with_dummy(smiles_or_offmol)
            offmol = OFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)
        elif isinstance(smiles_or_offmol, Oligomer):
            offmol = smiles_or_offmol.offmol
        indices = [i for i, atom in enumerate(offmol.atoms)
                   if atom.atomic_number != 0]
        atom_oligomer_map = {j: AtomWrapper(self, j) for j in indices}
        super().__init__(offmol, indices, atom_oligomer_map)



        


HYDROGEN = Monomer("([R1])[H]")

def create_hydrogen_caps(r_group_numbers):
    return {r: [(1, HYDROGEN)] for r in r_group_numbers}

