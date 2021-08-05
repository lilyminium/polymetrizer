from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import warnings

# from openff.fragmenter.fragment import Fragmenter
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField

from .oligomer import Monomer, create_hydrogen_caps, HYDROGEN_CAP
from .smirker import Smirker
from . import offutils

try:
    from .oefuncs import fragment_into_dummy_smiles
except ImportError:
    from .rdfuncs import fragment_into_dummy_smiles


class Polymetrizer:
    """
    Overall class for joining fragments of monomers together,
    coming up with parameters, and creating a force field.

    Parameters
    ----------
    monomers: list of Monomer objects
    r_linkages:
        dictionary of r-group linkages
    """

    @classmethod
    def from_molecule_smiles(
        cls,
        smiles: str,
        bond_smirks_pattern: Optional[str] = offutils.DEFAULT_BOND_PATTERN,
        bond_atom_numbers: Tuple[int, int] = (1, 2),
        unique_r_groups: bool = True,
        return_cleaved_bonds: bool = False,
        **kwargs,
    ):
        """
        Create a Polymetrizer from an OpenFF toolkit Molecule
        by breaking bonds

        Parameters
        ----------
        smiles: str
            SMILES pattern
        bond_smirks_pattern: str (optional)
            The bond smirks pattern to detect bonds for
            cleaving. If not given, it will default to a
            pattern of three single bonds over four non-aromatic atoms,
            where the middle bond is cleaved.
        bond_atom_numbers: tuple of ints
            The atom mapping numbers for the bond to cleave. e.g.
            in the default bond smirks pattern, the atom mapping
            numbers for a linear chain of four atoms are
            (4, 1, 2, 3). Specifying ``bond_atom_numbers=(1, 2)``
            cleaves the bond between atom 1 and atom 2.
        return_cleaved_bonds: bool (optional)
            Whether to return the atom indices of the cleaved bonds

        Returns
        -------
        Polymetrizer (, list of tuple of ints)
        """
        offmol = offutils.mol_from_smiles(smiles)
        return cls.from_offmolecule(offmol,
                                    bond_smirks_pattern,
                                    bond_atom_numbers,
                                    unique_r_groups,
                                    return_cleaved_bonds,
                                    **kwargs)

    @classmethod
    def from_offmolecule(cls, offmol: OFFMolecule,
                         bond_smirks_pattern: Optional[str] = offutils.DEFAULT_BOND_PATTERN,
                         bond_atom_numbers: Tuple[int, int] = (1, 2),
                         unique_r_groups: bool = True,
                         return_cleaved_bonds: bool = False,
                         **kwargs):
        """
        Create a Polymetrizer from an OpenFF toolkit Molecule
        by breaking bonds

        Parameters
        ----------
        offmol: openff.toolkit.topology.Molecule
        bond_smirks_pattern: str (optional)
            The bond smirks pattern to detect bonds for
            cleaving. If not given, it will default to a
            pattern of three single bonds over four non-aromatic atoms,
            where the middle bond is cleaved.
        bond_atom_numbers: tuple of ints
            The atom mapping numbers for the bond to cleave. e.g.
            in the default bond smirks pattern, the atom mapping
            numbers for a linear chain of four atoms are
            (4, 1, 2, 3). Specifying ``bond_atom_numbers=(1, 2)``
            cleaves the bond between atom 1 and atom 2.
        return_cleaved_bonds: bool (optional)
            Whether to return the atom indices of the cleaved bonds

        Returns
        -------
        Polymetrizer (, list of tuple of ints)
        """
        bonds = offutils.get_bonds(offmol,
                                   pattern=bond_smirks_pattern,
                                   ignore_neighbors=True,
                                   bond_atom_numbers=bond_atom_numbers,
                                   get_bonds_only=True)
        new = cls.from_offmolecule_and_bonds(offmol, bonds, unique_r_groups, **kwargs)
        if return_cleaved_bonds:
            return new, bonds
        return new

    @classmethod
    def from_offmolecule_and_bonds(cls, offmol, bonds, unique_r_groups, **kwargs):
        smiles, r_linkages = fragment_into_dummy_smiles(offmol, bonds,
                                                        unique_r_groups=unique_r_groups)
        monomers = [Monomer(smi) for smi in smiles]
        return cls(monomers, r_linkages, **kwargs)

    def __init__(self, monomers: List[Monomer] = [],
                 r_linkages: Dict[int, Set[int]] = {},
                 default_caps=None):
        self.monomers = [Monomer(m) for m in monomers]

        # if no r_linkages, just set self-to-self
        if not r_linkages:
            r_linkages = {}
            for monomer in self.monomers:
                for num in monomer.r_group_indices:
                    r_linkages[num] = {num}

        # symmetrize links
        self.r_linkages = defaultdict(set)
        for num, partners in r_linkages.items():
            self.r_linkages[num] |= set(partners)
            for partner in partners:
                self.r_linkages[partner].add(num)

        self.caps_for_r_groups = defaultdict(list)
        for monomer in self.monomers:
            for num in monomer.r_group_indices:
                for partner in self.r_linkages[num]:
                    self.caps_for_r_groups[partner].append((num, monomer))

        self.monomer_oligomers = defaultdict(list)
        self.oligomers = []

        if default_caps is None:
            default_caps = [HYDROGEN_CAP]

        self.default_caps = defaultdict(list)
        for monomer in default_caps:
            monomer = Monomer(monomer)
            for num in monomer.r_group_indices:
                for partner in self.caps_for_r_groups:
                    self.default_caps[partner].append((num, monomer))

    def create_oligomers(self, n_neighbor_monomers: int = 1,
                         fragmenter=None):
        # TODO: is Hs the best way to go?

        caps = self.default_caps

        while n_neighbor_monomers:
            new_caps = defaultdict(list)
            for monomer in self.monomers:
                substituted = monomer.generate_substituted_caps(caps)
                for root_r, monomer_list in substituted.items():
                    for partner in self.r_linkages[root_r]:
                        new_caps[partner].extend([(root_r, x) for x in monomer_list])
            caps = new_caps
            n_neighbor_monomers -= 1

        for monomer in self.monomers:
            capped_monomers = monomer.generate_substituted(caps)
            self.monomer_oligomers[monomer].extend(capped_monomers)
            if fragmenter is not None:
                capped_monomers = [x.fragment_around_central_atoms(fragmenter)
                                   for x in capped_monomers]
            self.oligomers.extend(capped_monomers)

    def get_forcefield_parameters(self, forcefield: ForceField,
                                  n_overlapping_atoms: int = 3):
        all_handler_kwargs = defaultdict(lambda: defaultdict(list))

        seen_kwargs = {}

        for oligomer in self.oligomers:
            central = oligomer.get_central_forcefield_parameters(forcefield, n_overlapping_atoms, seen=seen_kwargs)
            key = hash(oligomer)
            seen_kwargs[key] = central
            for handler_name, parameters in central.items():
                for monomer_atoms, params in parameters.items():
                    all_handler_kwargs[handler_name][monomer_atoms].extend(params)
        return all_handler_kwargs

    def build_residue_forcefield(self, forcefield: ForceField,
                                 all_handler_kwargs: Dict[str, dict],
                                 compressed: bool = True) -> ForceField:

        for handler_name, handler_kwargs in all_handler_kwargs.items():
            handler = forcefield.get_parameter_handler(handler_name)
            if handler._INFOTYPE is None:
                warnings.warn(f"{handler_name} has no INFOTYPE so I don't know how to add params")
                continue

            # TODO: hierarchical smarts
            smirker = Smirker(handler_kwargs)
            if handler_name in ("LibraryCharges",):
                for monomer in self.monomers:
                    param = smirker.get_combined_smirks_parameter(monomer,
                                                                  compressed=compressed)
                    try:
                        handler.add_parameter(param)
                    except:  # TODO: figure out the error
                        pass
                continue

            unified_smirks = smirker.get_unified_smirks_parameters(context="residue",
                                                                   compressed=compressed)
            for smirks, group_parameter in unified_smirks.items():
                param = dict(smirks=smirks, **group_parameter.mean_parameter.fields)
                handler.add_parameter(param)

        return forcefield

    def build_combination_forcefield(self, forcefield: ForceField,
                                     all_handler_kwargs: Dict[str, dict],
                                     compressed: bool = True) -> ForceField:
        combinations = []
        for monomer in self.monomers:
            built = monomer.build_all_combinations(self.caps_for_r_groups)
            combinations.append(built)

        for handler_name, handler_kwargs in all_handler_kwargs.items():
            handler = forcefield.get_parameter_handler(handler_name)
            if handler._INFOTYPE is None:
                warnings.warn(f"{handler_name} has no INFOTYPE so I don't know how to add params")
                continue

            smirker = Smirker(handler_kwargs)

            if handler_name in ("LibraryCharges",):
                for oligomer in combinations:
                    param = smirker.get_combined_smirks_parameter(oligomer, compressed=compressed)
                    try:
                        handler.add_parameter(param)
                    except:  # TODO: figure out the error
                        pass
                continue
            for oligomer in combinations:
                unified_smirks = smirker.get_smirks_for_oligomer(oligomer, compressed=compressed)
                for smirks, group_parameter in unified_smirks.items():
                    param = dict(smirks=smirks, **group_parameter.mean_parameter.fields)
                    try:
                        handler.add_parameter(param)
                    except:  # TODO: figure out error
                        pass
        return forcefield

    def _build_forcefield(self, forcefield_parameters, residue_based=True):
        forcefield_parameters = dict(**forcefield_parameters)
        new_ff = ForceField()
        if residue_based:
            if residue_based is True:
                residue_based = list(forcefield_parameters.keys())
            specific_kwargs = {}
            for k in residue_based:
                specific_kwargs[k] = forcefield_parameters.pop(k, {})
            self.build_residue_forcefield(new_ff, specific_kwargs)
        if forcefield_parameters:
            self.build_combination_forcefield(new_ff, forcefield_parameters)
        return new_ff

    def build_forcefield(self, forcefield: ForceField,
                         n_overlapping_atoms: int = 3,
                         residue_based: bool = True):
        parameters = self.get_forcefield_parameters(forcefield, n_overlapping_atoms)
        return self._build_forcefield(parameters, residue_based)

    def polymetrize(self, forcefield: ForceField,
                    n_neighbor_monomers: int = 1,
                    n_overlapping_atoms: int = 3,
                    fragmenter=None,
                    residue_based: bool = True) -> ForceField:
        """
        Overall function for adding parameters to a forcefield.

        Parameters
        ----------
        forcefield: openff.toolkit.typing.engines.smirnoff.forcefield.ForceField
            Initial force field to pull parameters from
        n_neighbor_monomers: int
            Number of monomers to attach to each R-group point to build
            an Oligomer. If a monomer has two R-group points, and
            ``n_neighbor_monomers=1``, the resulting Oligomers will
            be built to include three monomers (1 for each attachment to the
            central monomer)
        n_overlapping_atoms: int
            Initial parameters are obtained for each oligomer from the
            ``forcefield``. Oligomers are built by joining Monomers together.
            When parameters are collected for the final forcefield, this
            parameter controls how many bonds out from each residue
            constitutes the "overlap" region where parameters are averaged
            between Oligomers.
        fragmenter: Fragmenter (optional)
            If provided, each Oligomer is fragmented around the central
            residue before parametrizing. This can speed things up for further
            post-processing work, e.g. torsional scans
        residue_based: bool or iterable of strings
            This controls the smirks patterns in the output force field.
            If ``True``, all smirks patterns are given as labeled atoms in
            an individual residue, or joined residues if it is multi-atom and
            spans residues. If ``False``, the Polymetrizer
            uses the given ``r_linkages`` to create combinatorial Oligomers
            that represent all possible configurations for more defined smirks.
            The downside of this approach is that substructure matching consumes
            significantly more memory. If given an iterable of strings, only
            the passed parameter names will be given the residue treatment,
            while all remaining parameters will be given the combinatorial
            treatment. e.g. if passing ``residue_based=("vdW",)`` then
            vdW smirks patterns will be created per-residue, but LibraryCharge
            smirks patterns will be created per-combinatorial Oligomer. Note that
            the combinatorial Oligomers need to terminate. This works best for
            Polymetrizers created with ``from_offmolecule`` or
            ``from_molecule_smiles`` where the original molecule will be returned.

        Returns
        -------
        ForceField
        """
        self.create_oligomers(n_neighbor_monomers,
                              fragmenter=fragmenter)
        return self.build_forcefield(forcefield, n_overlapping_atoms,
                                     residue_based=residue_based)
