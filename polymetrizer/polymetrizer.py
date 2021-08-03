from typing import List, Dict, Set
from typing_extensions import Literal
import itertools

import networkx as nx
from pydantic import PrivateAttr, validator

from . import base, utils
from .monomer import Monomer, Cap, HYDROGEN
from .oligomer import Oligomer
from .parameters import ForceFieldParameterSets, ParameterSet


class Polymetrizer(base.Model):

    monomers: Dict[str, Monomer] = []
    caps: List[Cap] = [HYDROGEN]
    r_linkage_graph: nx.Graph
    _monomers: Dict[str, Monomer] = {}
    _oligomers: List[Oligomer]

    @validator("r_linkage_graph")
    def validate_r_linkages(cls, v):
        if isinstance(v, nx.Graph):
            return v
        graph = nx.Graph()
        for r, other_rs in v.items():
            for r2 in other_rs:
                graph.add_edge((r, r2))
        return graph

    @validator("monomers")
    def validate_monomers(cls, v):
        if isinstance(v, dict):
            validated = {}
            for name, monomer in v.items():
                if name != monomer.name:
                    monomer = monomer.copy(deep=True)
                    monomer.name = name
                validated[name] = monomer
                return validated
        return {m.name: m for m in v}

    def __post_init__(self):
        self._monomers = {}
        for monomer in self.monomers:
            self._monomers[monomer.name] = monomer

    def add_r_group_linkage(self, r1, r2):
        self._r_linkage_graph.add_edge(r1, r2)

    def remove_r_group_linkage(self, r1, r2):
        self._linkage_graph.remove_edge(r1, r2)

    def enumerate_oligomers(self, n_neighbor_monomers: int = 1,
                            prune_isomorphs: bool = True):
        products = []
        for monomer in self.monomers.values():
            og = monomer.to_oligomer()
            products.extend(
                og.enumerate_substituted_products(substituents=self.monomers,
                                                  caps=self.caps,
                                                  linkage_graph=self._r_linkage_graph,
                                                  n_substitutions=n_neighbor_monomers)
            )
        self._oligomers = products
        if prune_isomorphs:
            self.prune_isomorphic_oligomers()

    def prune_isomorphic_oligomers(self):
        oligomers = sorted(self._oligomers, key=Oligomer._monomer_keyfunc)
        all_unique = []
        for _, group in itertools.groupby(oligomers, key=Oligomer._monomer_keyfunc):
            group = list(group)
            unique = [group.pop(0)]
            for omer in group:
                if not any(omer._is_fully_isomorphic(unq) for unq in unique):
                    unique.append(omer)
            all_unique.extend(unique)
        self._oligomers = all_unique

    def generate_openff_parameters(self, forcefield,
                                   n_neighbors: int = 3):
        parameters = ForceFieldParameterSets()
        for oligomer in self._oligomers:
            parameters += oligomer.to_openff_parameterset(forcefield,
                                                          n_neighbors=n_neighbors)
        return parameters

    def get_atom_monomers(self, atoms=[]):
        return [self.monomers[a.monomer_name] for a in atoms]

    def generate_smarts(
            self, atom_graph,
            context: Literal["minimal", "residue", "full"] = "residue",
            include_caps: bool = True,
            enumerate_all: bool = False,
    ):
        monomers = self.get_atom_monomers(atom_graph)
        if len(set(monomers)) == 1 and context != "full":
            # intra-molecular
            monomer = monomers[0]
            smarts = monomer.to_smarts(atom_graph, context=context,
                                       include_caps=include_caps,
                                       enumerate_all=enumerate_all)
            return smarts
        # check for oligomers with matching graphs
        # if minimal or residue, just one will do
        monomer_names = set(m.name for m in monomers)
        all_smarts = []
        all_monomer_names = []
        for omer in self.oligomers:
            # avoid as much graph matching as possible
            if all(m in omer._constituent_monomers for m in monomer_names):
                smarts = omer.monomer_atoms_to_smarts(atom_graph, context=context,
                                                      include_caps=include_caps,
                                                      enumerate_all=enumerate_all,)
                if smarts and context != "full":
                    return smarts
                all_smarts.append(smarts)
        return all_smarts

    def build_openff_residue_forcefield(self, forcefield,
                                        n_neighbors: int = 3,
                                        include_caps: bool = True,
                                        average_same_smarts: bool = True,
                                        split_smarts_into_full: bool = True):
        parameters = self.generate_openff_parameters(forcefield,
                                                     n_neighbors=n_neighbors)
        averaged = parameters.average_over_keys()
        new = type(forcefield)()

        smartsify = functools.partial(self.generate_unique_smarts,
                                      average=average_same_smarts,
                                      split_into_full=split_smarts_into_full)

        for parameter_name, parameter_set in averaged.items():
            handler = forcefield.get_parameter_handler(parameter_name)
            if parameter_name in ("LibraryCharges",):
                # smarts_set = smartsify(parameter_set, context="full")
                smarts_to_parameter = self.generate_combined_residue_smarts(parameter_set,
                                                                            include_caps=include_caps)
            else:
                smarts_to_parameter = smartsify(parameter_set, context="residue")
            for i, (smarts, parameter) in enumerate(smarts_to_parameter.items(), 1):
                pid = parameter.pop("id", None)
                if pid is not None:
                    parameter["id"] = f"{pid}{i}"
                handler.add_parameter(dict(smirks=smarts, **parameter))
        return new

    def generate_combined_residue_smarts(self, parameter_set,
                                         include_caps: bool = True,):
        # currently only valid for single atom parameters
        graphs = list(parameter_set)
        assert all(len(graph) == 1 for graph in graphs)

        monomer_atoms = defaultdict(list)
        monomer_parameters = defaultdict(lambda: defaultdict(list))
        for atom_graph in graphs:
            atom = list(atom_graph)[0]
            monomer = self.monomer[atom.monomer_name]
            monomer_atoms[monomer].append(atom)

        monomer_parameters = {}
        for monomer, atoms in monomer_atoms.items():
            nodes = []
            combined_parameter = defaultdict(list)
            for atom in atoms:
                nodes.append(monomer.get_atom_node(atom))
                parameter = parameter_set[atom]
                for k, v in parameter.items():
                    if utils.is_iterable(v):
                        combined_parameter[k].extend(v)
                    else:
                        combined_parameter[k].append(v)
            smarts = monomer.to_smarts(label_nodes=nodes, context="residue",
                                       include_caps=include_caps)
            combined_parameter["id"] = monomer.name
            monomer_parameters[smarts] = combined_parameter
        return monomer_parameters

    def generate_unique_smarts(self, parameter_set, context="residue",
                               include_caps: bool = True,
                               average=False,
                               split_into_full: bool = True):

        def atom_graph_to_id(graph):
            monomer_names = {x.capitalize()
                             for i, x in graph.nodes("monomer_name")}
            return "".join(sorted(monomer_names))

        smarts_to_parameter = defaultdict(list)
        smarts_to_atomgraph = defaultdict(list)

        for atom_graph, parameter in parameter_set.items():
            smarts = self.generate_smarts(atom_graph, context=context,
                                          include_caps=include_caps)
            smarts_to_parameter[smarts].append(parameter)
            smarts_to_atomgraph[smarts].append(atom_graph)

        smarts_to_unique = {}
        for key, values in smarts_to_parameter.items():
            unique = []
            for v in values:
                if v not in unique:
                    unique.append(v)
            smarts_to_unique[key] = unique

        unified = {}
        for smarts, parameters in smarts_to_unique.items():
            if len(parameters) == 1:
                # is unique, move on
                graph = smarts_to_atomgraph[smarts][0]
                parameter["id"] = atom_graph_to_id(graph)
                unified[smarts] = parameter
            else:
                atom_graphs = smarts_to_atomgraph[smarts]
                full_smarts = defaultdict(list)
                # try getting the full SMARTS pattern first
                for ag in atom_graphs:
                    fulls = self.generate_smarts(ag, context="full",
                                                 include_caps=include_caps)
                    for full in fulls:
                        if full not in full_smarts:
                            break

                    parameter = parameter_set[atom_graph]
                    parameter["id"] = atom_graph_to_id(ag)
                    full_smarts[full].append(parameter)

                # there may be duplicates here too?
                for k, vs in full_smarts.items():
                    if len(vs) == 1 and k not in unified:
                        unified[k] = vs[0]
                    else:
                        options = vs
                        if k in unified:
                            options.append(unified[k])
                        if average:
                            pset = ParameterSet(None)
                            pset.add_parameters({k: options})
                            unified[k] = pset.average_over_keys(drop=["id"])
                            unified[k]["id"] = options[0]["id"]
                        else:
                            err = ("Cannot unify different parameters "
                                   f"for same SMARTS: {k} and "
                                   f"parameters {options}. Try setting "
                                   "average=True")
                            raise ValueError(err)
        return unified

    def polymetrize(self, forcefield,
                    n_neighbor_monomers: int = 1,
                    n_overlapping_atoms: int = 3,
                    prune_isomorphs: bool = True,
                    include_caps: bool = True,
                    average_same_smarts: bool = True,
                    split_smarts_into_full: bool = True,
                    ):
        self.enumerate_oligomers(n_neighbor_monomers=n_neighbors,
                                 prune_isomorphs=prune_isomorphs)
        ff = self.build_openff_residue_forcefield(forcefield,
                                                  n_neighbors=n_overlapping_atoms,
                                                  include_caps=include_caps,
                                                  average_same_smarts=average_same_smarts,
                                                  split_smarts_into_full=split_smarts_into_full)
        return ff
