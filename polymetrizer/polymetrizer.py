from typing import List, Dict, Set
from typing_extensions import Literal
import itertools
import logging
from collections import defaultdict
import functools

from tqdm import tqdm
import numpy as np
import networkx as nx
from pydantic import PrivateAttr, validator, Field

from . import base, utils
from .monomer import Monomer, Cap, HYDROGEN_CAP
from .oligomer import Oligomer
from .parameters import ForceFieldParameterSets, ParameterSet
from .smirks import SmirkSet

logger = logging.getLogger(__name__)


class Polymetrizer(base.Model):

    monomers: Dict[str, Monomer] = []
    caps: List[Cap] = [HYDROGEN_CAP]
    r_linkages: Dict[int, Set[int]] = {}
    _r_linkage_graph: nx.Graph = PrivateAttr(default_factory=nx.Graph)
    _monomers: List[Monomer] = PrivateAttr(default_factory=list)
    _caps: Dict[str, Cap] = PrivateAttr(default_factory=dict)
    _oligomers: List[Oligomer] = PrivateAttr(default_factory=list)

    def __post_init__(self):
        self._monomers = list(self.monomers.values())
        self._r_linkage_graph = nx.Graph()
        for r, other_rs in self.r_linkages.items():
            for r2 in other_rs:
                self._r_linkage_graph.add_edge(r, r2)
        self._caps = {}
        for i, cap in enumerate(self.caps, 1):
            if not cap.name:
                cap.name = f"cap{i}"
            self._caps[cap.name] = cap

    @validator("caps", pre=True, each_item=True)
    def validate_caps(cls, v):
        v = Cap.from_obj(v)
        assert isinstance(v, Cap)
        return v

    @validator("monomers", pre=True, each_item=False)
    def validate_monomers(cls, v):
        if isinstance(v, dict):
            validated = {}
            for name, monomer in v.items():
                monomer = Monomer.from_obj(monomer, name=name)
                if name != monomer.name:
                    monomer = monomer.copy(deep=True)
                    monomer.name = name
                validated[name] = monomer
            return validated
        return {m.name: m for m in v}

    @property
    def r_linkage_graph(self):
        return self._r_linkage_graph

    @property
    def oligomers(self):
        return self._oligomers

    def add_r_group_linkage(self, r1, r2):
        self._r_linkage_graph.add_edge(r1, r2)

    def remove_r_group_linkage(self, r1, r2):
        self._linkage_graph.remove_edge(r1, r2)

    def enumerate_oligomers(self, n_neighbor_monomers: int = 1,
                            prune_isomorphs: bool = True):
        products = []
        for monomer in tqdm(self._monomers):
            og = monomer.to_oligomer()
            products.extend(
                og.enumerate_substituted_products(substituents=self._monomers,
                                                  caps=self.caps,
                                                  linkage_graph=self._r_linkage_graph,
                                                  n_substitutions=n_neighbor_monomers)
            )
        self._oligomers = products
        if prune_isomorphs:
            self.prune_isomorphic_oligomers()

    def prune_isomorphic_oligomers(self):
        n_original = len(self._oligomers)
        oligomers = sorted(self._oligomers, key=Oligomer._monomer_keyfunc)
        all_unique = []
        for _, group in itertools.groupby(oligomers, key=Oligomer._monomer_keyfunc):
            group = list(group)
            unique = [group.pop(0)]
            for omer in group:
                if not any(omer._is_fully_isomorphic(unq) for unq in unique):
                    unique.append(omer)
            all_unique.extend(unique)
        n_pruned = len(all_unique)
        self._oligomers = all_unique
        n_removed = n_original - n_pruned
        logger.info(f"Removed {n_removed} isomorphic oligomers "
                    f"from {n_original} original oligomers.")

    def generate_openff_parameters(self, forcefield, **kwargs):
        parameters = ForceFieldParameterSets()
        for oligomer in tqdm(self._oligomers):
            parameters += oligomer.to_openff_parameterset(forcefield,
                                                          **kwargs)
        return parameters

    def _get_unit_from_monomer_atom(self, atom):
        try:
            return self.monomers[atom.monomer_name]
        except KeyError as e:
            try:
                return self._caps[atom.monomer_name]
            except KeyError:
                raise e from None

    def get_atom_monomers(self, atoms=[]):
        return [self._get_unit_from_monomer_atom(a) for a in atoms]

    def cap(self, compounds):
        return [x.cap_remaining(self.caps,
                                linkage_graph=self._r_linkage_graph)
                for x in compounds]

    def build_openff_residue_forcefield(self, forcefield,
                                        include_caps: bool = False,
                                        average_same_smarts: bool = True,
                                        split_smarts_into_full: bool = False,
                                        **kwargs
                                        ):
        parameters = self.generate_openff_parameters(forcefield, **kwargs)
        averaged = parameters.average_over_keys()
        smirkset = SmirkSet(average_same_smarts=average_same_smarts,
                            split_smarts_into_full=split_smarts_into_full,
                            include_caps=include_caps,
                            context="residue")
        new = type(forcefield)()

        for parameter_name, parameter_set in averaged.items():
            handler = new.get_parameter_handler(parameter_name)
            if parameter_name in ("LibraryCharges",):
                compounds = self._monomers
                if include_caps:
                    compounds = self.cap(compounds)
                with smirkset.set_compounds(compounds) as smirker:
                    smarts_to_parameter = smirker.generate_combined_smarts(parameter_set)
            else:
                with smirkset.set_compounds(self.oligomers) as smirker:
                    smarts_to_parameter = smirker.generate_unique_smarts(parameter_set)
            for i, (smarts, parameter) in enumerate(smarts_to_parameter.items(), 1):
                parameter = dict(**parameter)
                pid = parameter.pop("id", None)
                if pid is not None:
                    parameter["id"] = f"{pid}_{parameter_name}_{i}"
                handler.add_parameter(dict(smirks=smarts, **parameter))
        return new

    def polymetrize(self, forcefield,
                    n_neighbor_monomers: int = 1,
                    n_overlapping_atoms: int = 3,
                    prune_isomorphs: bool = True,
                    include_caps: bool = False,
                    average_same_smarts: bool = True,
                    split_smarts_into_full: bool = True,
                    partial_charge_method: str = "am1bcc",
                    minimize_geometry: bool = True,
                    optimize_geometry: bool = False,
                    minimize_max_iter: int = 1000,
                    optimize_method: str = "m06-2x/def2-TZVP",
                    ):
        self.enumerate_oligomers(n_neighbor_monomers=n_neighbor_monomers,
                                 prune_isomorphs=prune_isomorphs)
        ff = self.build_openff_residue_forcefield(forcefield,
                                                  n_neighbors=n_overlapping_atoms,
                                                  include_caps=include_caps,
                                                  average_same_smarts=average_same_smarts,
                                                  split_smarts_into_full=split_smarts_into_full,
                                                  partial_charge_method=partial_charge_method,
                                                  minimize_geometry=minimize_geometry,
                                                  optimize_geometry=optimize_geometry,
                                                  minimize_max_iter=minimize_max_iter,
                                                  optimize_method=optimize_method)
        return ff
