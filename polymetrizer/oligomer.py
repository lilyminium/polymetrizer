import warnings
from typing import Union, Dict, List, Iterable, Tuple, Optional
import itertools
import functools
from collections import defaultdict

import numpy as np
from openff.toolkit.topology import Molecule as OFFMolecule

from . import utils, offutils, ommutils

try:
    from .oefuncs import attach_substituent_and_label, get_sub_smarts, subset_mol
except ImportError:
    from .rdfuncs import attach_substituent_and_label, get_sub_smarts, subset_mol


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
        return hash((self.monomer, self.index))


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
        ):

        if isinstance(offmol, Oligomer):
            central_atom_indices = offmol.central_atom_indices
            atom_oligomer_map = offmol.atom_oligomer_map
            offmol = offmol.offmol

        self.offmol = offmol
        self.central_atom_indices = sorted(central_atom_indices)
        self.atom_oligomer_map = dict(atom_oligomer_map)
        self.r_group_indices = {}

        off_map = self.offmol.properties.get("atom_map", {})

        for i, atom in enumerate(self.offmol.atoms):
            if atom.atomic_number == 0:  # dummy
                try:
                    num = off_map[i]
                except KeyError:
                    msg = ("R groups must be labelled. "
                           f"The atom at index {i} does not have a number, "
                           "so it is ignored")
                    raise ValueError(msg)  # TODO: convert to warning
                else:
                    self.r_group_indices[num] = i

    def _get_immutable_attrs(self):
        oligomer_map = tuple(sorted(self.atom_oligomer_map.items()))
        indices = tuple(sorted(self.central_atom_indices))
        return (self.offmol, oligomer_map, indices)

    def __hash__(self):
        return hash(self._get_immutable_attrs())
    
    def __eq__(self, other):
        return self._get_immutable_attrs() == other._get_immutable_attrs()

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

    def _build_next_combination(
            self,
            r_group_substituents: Dict[int, List["Oligomer"]] = {},
        ):
        groups = {}
        if not self.r_group_numbers:
            return self
        for r in self.r_group_numbers:
            groups[r] = r_group_substituents[r].pop(0)
        substituted = self.attach_substituents(groups)
        return substituted._build_next_combination(r_group_substituents)
    
    def build_all_combinations(
            self,
            r_group_substituents: Dict[int, List["Oligomer"]] = {},
            repeat: int = 1,
        ):
        # TODO: pretty sure this logic is wrong for residues
        # with multiple linkages
        # caps = {}
        # for k, v in r_group_substituents.items():
        #     caps[k] = v * repeat
        # return self._build_next_combination(caps)
        substituted = self.generate_substituted(r_group_substituents)
        substituted = substituted[0]
        
        while substituted.r_group_indices:
            substituted = substituted.generate_substituted(r_group_substituents)
            substituted = substituted[0]
            # raise ValueError
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

    def select_relevant_parameters(self, handler_kwargs, relevant_indices):
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

    def get_qualified_atoms(self, indices, ordered=True):
        try:
            qualified = tuple(self.atom_oligomer_map[i] for i in indices)
        except KeyError:
            return
        if ordered and qualified[0] > qualified[-1]:
            qualified = qualified[::-1]
        return qualified


    def get_central_forcefield_parameters(self, forcefield, n_neighbors: int=3):
        handler_kwargs = self.get_forcefield_parameters(forcefield)
        relevant_indices = self.get_central_and_neighbor_indices(n_neighbors)
        return self.select_relevant_parameters(handler_kwargs, relevant_indices)

    def contains_qualified_atoms(
            self,
            qualified_atoms: List[Tuple[int, int]],
            handler_name: str = None,
        ) -> bool:
        """
        Whether this Oligomer contains this atom topology.

        Parameters
        ----------
        qualified_atoms: list of tuple of ints
            List of atom identifiers in a Polymetrizer where
            the first value in the tuple is the index of the monomer
            and teh second value is the atom index
        handler_name: str (optional)
            The handler name is only necessary for distinguishing
            ImproperTorsions, as these have different bonding patterns
            to the other parameters.
        
        Returns
        -------
        bool
        """
        qualified_atom_set = set(qualified_atoms)
        qualified = set(self.atom_oligomer_map.values())
        if not qualified_atom_set.issubset(qualified):
            return False
        qualified_to_index = {v: k for k, v in self.atom_oligomer_map.items()}

        # for stuff that stretches between residues,
        # first three atoms are always bonded A-B-C.
        # in proper torsions, C-D are bonded; in improper, B-D.
        if handler_name in ["LibraryCharges", "vdW"]:
            return True

        for i, atom_index in enumerate(qualified_atoms[1:3]):
            previous_index = qualified_to_index[qualified_atoms[i]]
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
        ) -> Optional[str]:
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



class Monomer(Oligomer):

    def __init__(
            self,
            smiles_or_offmol: Union[str, OFFMolecule]
        ):
        if isinstance(smiles_or_offmol, str):
            smiles = utils.replace_R_with_dummy(smiles_or_offmol)
            offmol = OFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)
        indices = [i for i, atom in enumerate(offmol.atoms)
                   if atom.atomic_number != 0]
        atom_oligomer_map = {j: AtomWrapper(self, j) for j in indices}
        super().__init__(offmol, indices, atom_oligomer_map)


HYDROGEN = Monomer("([R1])[H]")

def create_hydrogen_caps(r_group_numbers):
    return {r: [(1, HYDROGEN)] for r in r_group_numbers}


