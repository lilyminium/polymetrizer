from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import warnings

from openff.fragmenter.fragment import Fragmenter
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField

from .monomer import Monomer
from .oligomer import Oligomer
from . import utils, offutils

try:
    from .oefuncs import fragment_into_dummy_smiles
except ImportError:
    from .rdfuncs import fragment_into_dummy_smiles

class Polymetrizer:

    @classmethod
    def from_molecule_smiles(
            cls,
            smiles: str,
            bond_smirks_pattern: Optional[str] = None,
            bond_atom_numbers: Tuple[int, int] = (1, 2),
            return_cleaved_bonds: bool = False,
        ):
        offmol = offutils.mol_from_smiles(smiles)
        return cls.from_offmolecule(offmol,
                                    bond_smirks_pattern,
                                    bond_atom_numbers,
                                    return_cleaved_bonds)

    @classmethod
    def from_offmolecule(
            cls,
            offmol: OFFMolecule,
            bond_smirks_pattern: Optional[str] = None,
            bond_atom_numbers: Tuple[int, int] = (1, 2),
            return_cleaved_bonds: bool = False,
        ):
        bonds = offutils.get_bonds(offmol,
                                   pattern=bond_smirks_pattern,
                                   ignore_neighbors=True,
                                   bond_atom_numbers=bond_atom_numbers,
                                   get_bonds_only=True)
        smiles, r_linkages = fragment_into_dummy_smiles(offmol, bonds)
        monomers = [Monomer.from_dummy_smiles(smi) for smi in smiles]
        new = cls(monomers, r_linkages)
        if return_cleaved_bonds:
            return new, bonds
        return new
        

    def __init__(
            self,
            monomers: List[Monomer] = [],
            r_linkages: Dict[int, Set[int]] = {},
        ):
        self.monomers = [Oligomer.with_oligomer_index(m, index=i) for i, m in enumerate(monomers)]

        # if no r_linkages, just set self-to-self
        if not r_linkages:
            r_linkages = {}
            for monomer in self.monomers:
                for num in monomer.r_group_numbers:
                    r_linkages[num] = {num}
    
        # symmetrize links
        self.r_linkages = defaultdict(set)
        for num, partners in r_linkages.items():
            self.r_linkages[num] |= set(partners)
            for partner in partners:
                self.r_linkages[partner].add(num)

        self.r_group_caps = defaultdict(list)
        for monomer in self.monomers:
            for num in monomer.r_group_numbers:
                for partner in self.r_linkages[num]:
                    self.r_group_caps[partner].append((num, monomer))
        
        self.monomer_oligomers = []
        self.oligomers = []
    
    def create_oligomers(
            self,
            n_neighbor_monomers: int = 1,
            fragmenter: Optional[Fragmenter] = None,
        ):
        # TODO: is Hs the best way to go?
        hydrogen = Oligomer("([R1])[H]")
        caps = {num: [(1, hydrogen)] for num in self.r_group_caps}

        while n_neighbor_monomers:
            new_caps = defaultdict(list)
            for monomer in self.monomers:
                substituted = monomer.generate_substituted_caps(caps)
                for r, monomer_list in substituted.items():
                    for partner in self.r_linkages[r]:
                        new_caps[partner].extend([(r, x) for x in monomer_list])
            caps = new_caps
            n_neighbor_monomers -= 1

        self.monomer_oligomers = []
        self.oligomers = []
        for i, monomer in enumerate(self.monomers):
            capped_monomers = monomer.generate_substituted(caps)
            self.monomer_oligomers.append(capped_monomers)
            if fragmenter is not None:
                capped_monomers = [x.fragment_around_central_atoms(fragmenter)
                                   for x in capped_monomers]
            self.oligomers.extend(capped_monomers)


    def get_forcefield_parameters(
            self,
            forcefield: ForceField,
            n_overlapping_atoms: int = 3,
        ):
        all_handler_kwargs = defaultdict(lambda: defaultdict(list))
        for oligomer in self.oligomers:
            central = oligomer.get_central_forcefield_parameters(forcefield, n_overlapping_atoms)
            for handler_name, handler_kwargs in central.items():
                for atoms, akwargs in handler_kwargs.items():
                    all_handler_kwargs[handler_name][atoms].append(akwargs)
        
        averaged_handler_kwargs = defaultdict(dict)
        for handler_name, handler_kwargs in all_handler_kwargs.items():
            for atoms, akwargs in handler_kwargs.items():
                mean_kwargs = utils.average_dicts(akwargs)
                averaged_handler_kwargs[handler_name][atoms] = mean_kwargs
        return averaged_handler_kwargs



    def build_residue_forcefield(
            self,
            averaged_handler_kwargs: Dict[str, dict]
        ) -> ForceField:

        new = ForceField()
        for handler_name, handler_kwargs in averaged_handler_kwargs.items():
            handler = new.get_parameter_handler(handler_name)
            if handler._INFOTYPE is None:
                warnings.warn(f"{handler_name} has no INFOTYPE so I don't know how to add params")
                continue
            for atoms, mean_kwargs in handler_kwargs.items():
                for smirk in self.get_residue_smirks(atoms, handler_name):
                    param = dict(smirks=smirk, **mean_kwargs)
                    handler.add_parameter(param)
        return new
    

    def get_residue_smirks(
            self,
            qualified_atom_indices: List[Tuple[int, int]],
            handler_name: Optional[str] = None,
        ) -> List[str]:
        monomers, atoms = zip(*qualified_atom_indices)
        first = self.monomers[monomers[0]]
        monomers = set(monomers)
        if len(monomers) == 1:
            return [first.get_residue_smirks(qualified_atom_indices)]

        smirks = set()
        for mindex in monomers:
            oligomers = self.monomer_oligomers[mindex]
            for oligomer in oligomers:
                smirk = oligomer.get_residue_smirks(qualified_atoms, handler_name)
                if smirk is not None:
                    smirks.add(smirk)
        return sorted(smirks)
    
    def build_combination_forcefield(
            self,
            averaged_handler_kwargs: Dict[str, dict]
        ) -> ForceField:
        combinations = []
        for monomer in self.monomers:
            built = monomer.build_all_combinations(self.r_group_caps)
            combinations.append(built)

        new = ForceField()
        for handler_name, handler_kwargs in averaged_handler_kwargs.items():
            handler = new.get_parameter_handler(handler_name)
            if handler._INFOTYPE is None:
                warnings.warn(f"{handler_name} has no INFOTYPE so I don't know how to add params")
                continue
            for atoms, mean_kwargs in handler_kwargs.items():
                smirks = set()
                for oligomer in combinations:
                    smirk = oligomer.get_residue_smirks(atoms, handler_name,
                                                        oligomer.indices)
                    if smirk is not None:
                        smirks.add(smirk)
                for smirk in smirks:
                    param = dict(smirks=smirk, **mean_kwargs)
                    handler.add_parameter(param)
        return new
        
        

    def polymetrize(
            self,
            forcefield: ForceField,
            n_neighbor_monomers_in_oligomer: int = 1,
            n_overlapping_atoms: int = 3,
            fragmenter: Optional[Fragmenter] = None,
            residue_based: bool = True,
        ):
        self.create_oligomers(n_neighbor_monomers_in_oligomer,
                              fragmenter=fragmenter)
        averaged_handler_kwargs = self.get_forcefield_parameters(forcefield,
                                                                 n_overlapping_atoms)
        if residue_based:
            ff = self.build_residue_forcefield(averaged_handler_kwargs)
        else:
            ff = self.build_combination_forcefield(averaged_handler_kwargs)
        return ff



    