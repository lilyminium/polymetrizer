import itertools
from collections import defaultdict
from typing import Dict, Union, List, Optional, Set, Tuple
from typing_extensions import Literal

import networkx as nx
import numpy as np
from pydantic import Field, PrivateAttr

from . import base, utils, ommutils
from .graph import BaseMolecule, CapGraph
from .types import ForceFieldParametersByAtomIndex, ForceFieldParametersByNode
from .parameters import ForceFieldParameterSets


class Cap(BaseMolecule):

    name: Optional[str] = None
    graph: CapGraph = Field(default_factory=CapGraph)
    compatible_rs: Optional[List[int]] = []
    r: Optional[int] = None

    def __post_init__(self):
        if self.r is None:
            self.r = self.graph.graph[self.graph._node].get("isotope")

    def get_compatible_rs(self, *oligomers,
                          linkage_graph: Optional[nx.Graph] = None,
                          ):
        if self.compatible_rs:
            return self.compatible_rs
        if linkage_graph is not None and self.r is not None:
            return list(linkage_graph.neighbors(self.r))
        return list({r for o in oligomers for r in o.iter_r_group_numbers()})


HYDROGEN = Cap.from_smiles("[R][H]", name="H")


class Oligomer(BaseMolecule):

    _constituent_monomers: Dict[str, List[int]]
    _monomer_to_atom_nodes: Dict[int, Set[int]]
    _atom_to_monomer_nodes: Dict[int, int]
    _monomer_graph: nx.Graph

    def __post_init__(self):
        self._constituent_monomers = defaultdict(list)
        self._monomer_to_atom_nodes = {}
        self._atom_to_monomer_nodes = {}
        self._monomer_graph = nx.Graph()

    def _monomer_keyfunc(self):
        return (sorted(self._constituent_monomers),
                len(self._monomer_to_atom_nodes),
                len(self.graph.graph.nodes))

    def _is_fully_isomorphic(self, other):
        return self.graph.is_isomorphic(other.graph,
                                        match_isotope=True,
                                        match_formal_charge=True,
                                        match_is_aromatic=True,
                                        match_bond_order=True,
                                        match_monomer_name=True,
                                        match_monomer_atoms=True,
                                        )

    def map_r_substituent_pairs(
            self,
            substituents: List["Monomer"],
            linkage_graph: nx.Graph,
    ) -> Dict[int, List[Tuple[int, "Monomer"]]]:
        """
        Return R-R combinations from a list of available substituent monomers
        and graph of accepted linkages.

        Parameters
        ----------
        substituents: list of Monomers
            Available substituents
        linkage_graph: networkx.Graph
            Linkage graph

        Returns
        -------
        dictionary of R-R combinations
            The keys are the R-groups of self. The values are lists of
            tuples, where each tuple is (R-group of monomer, Monomer).
        """
        r_to_monomer = defaultdict(set)
        for sub in substituents:
            for r in sub.iter_r_group_numbers():
                r_to_monomer[r].add(sub)

        cap_combinations = defaultdict(list)

        for r in self.iter_r_group_numbers():
            neighbors = linkage_graph.neighbors(r)
            for n in neighbors:
                for monomer in r_to_monomer[n]:
                    cap_combinations[r].append((n, monomer))

        return cap_combinations

    def substitute(self, other: "Monomer",
                   r_self: int, r_other: int,
                   inplace: bool = False):
        """Add ``other`` Monomer by joining at ``r_self``, ``r_other``."""
        obj = self
        if not inplace:
            obj = obj.copy(deep=True)
        return obj._substitute(other, r_self, r_other)

    def _substitute(self, other, r_self, r_other):
        # TODO: maybe pin to node instead of r?
        new_nodes, new_edge = self.graph.add_with_r(other.graph, r_self, r_other)
        self._record_monomer(other, new_atom_nodes=new_nodes,
                             new_atom_edge=new_edge,
                             #  r_groups=(r_self, r_other),
                             )
        return self

    def _record_monomer(self, monomer, new_atom_nodes=set(),
                        new_atom_edge: Optional[Tuple[int, int]] = None,
                        # r_groups=Optional[Tuple[int, int]],
                        ):
        new_monomer_node = len(self.monomer_graph)
        self._monomer_to_atom_nodes[new_monomer_node] = new_atom_nodes
        for node in new_atom_nodes:
            self._atom_to_monomer_nodes[node] = new_monomer_node
        self._constituent_monomers[monomer.name].add(new_monomer_node)
        self._monomer_graph.add_node(new_monomer_node, monomer_name=monomer.name)
        if new_atom_edge:
            old_atom_node, new_atom_node = new_atom_edge
            if old_atom_node in new_atom_nodes:
                new_atom_node, old_atom_node = new_atom_edge
            old_monomer_node = self._atom_to_monomer_nodes[old_atom_node]
            edge = (old_monomer_node, new_monomer_node)
            atom_edge = (old_atom_node, new_atom_node)
            names = (self._monomer_graph[old_monomer_node]["monomer_name"],
                     self._monomer_graph[new_monomer_node]["monomer_name"],)
            monomer_atoms = (self.graph[old_atom_node]["monomer_atom"],
                             self.graph[new_atom_node]["monomer_atom"])
            self._monomer_graph.add_edge(edge,
                                         atom_nodes=atom_edge,
                                         monomer_atoms=monomer_atoms,
                                         monomer_nodes=edge,
                                         monomer_names=names)

    def with_substitutions(self, substitutions: List[dict]):
        new = self.copy(deep=True)
        for substitution in substitutions:
            new._substitute(**substitution)
        return new

    def enumerate_substituent_combinations(
            self,
            substituents: List["Monomer"],
            linkage_graph: nx.Graph,
    ) -> List[List[dict]]:
        caps = self.map_r_substituent_pairs(substituents, linkage_graph)
        r_group_numbers = list(self.graph.iter_r_group_numbers())
        sub_choices = [caps.get(r, []) for r in r_group_numbers]
        combinations = [
            [dict(other=other, r_self=r, r_other=r_other)
             for r, (r_other, other) in zip(combination, r_group_numbers)]
            for combination in itertools.product(*sub_choices)
        ]
        return combinations

    def generate_substituted(
            self,
            substituents: List["Monomer"],
            linkage_graph: nx.Graph,
    ):
        groups = self.enumerate_substituent_combinations(substituents,
                                                         linkage_graph)
        return [self.with_substitutions(group) for group in groups]

    def enumerate_substituted_products(
            self,
            substituents: List["Monomer"],
            caps: List["Cap"],
            linkage_graph: nx.Graph,
            n_substitutions: int = 1,
    ):
        products = [self]
        for i in range(n_substitutions):
            products = [x.generate_substituted(substituents, linkage_graph)
                        for x in products]
        cap_rs = [cap.get_compatible_rs(*products, linkage_graph=linkage_graph)
                  for cap in caps]
        for compound in products:
            for cap, r_groups in zip(cap, cap_rs):
                compound._cap_remaining(cap, r_groups)
        return products

    def cap_remaining(
            self,
            cap: Cap = HYDROGEN,
            r_groups: List[int] = [],
            inplace: bool = True,
    ):
        obj = self
        if not inplace:
            obj = obj.copy(deep=True)
        obj._cap_remaining(cap, r_groups)
        return obj

    def _cap_remaining(self, cap, r_groups):
        for r in self.graph.iter_r_group_numbers():
            if r in r_groups:
                self._substitute(cap, r, None)
        return self

    def get_central_indices(
            self,
            n_neighbors: int = 0,
    ) -> List[int]:
        nodes = self.graph.get_central_nodes(n_neighbors=n_neighbors)
        return np.where(np.isin(self.graph.nodes, nodes))[0]

    def get_index_to_monomer_atom_mapping(self):
        return {i: atom
                for i, (node, atom) in enumerate(self.graph.nodes("monomer_atom"))
                if atom is not None}

    def to_openff_parameterset(self, forcefield, n_neighbors: int = -1):
        offmol = self.graph.to_openff()
        system = forcefield.create_openmm_system(offmol.to_topology())
        parameter_set = ommutils.parameter_set_from_openff_system(system)
        if n_neighbors > 0:
            indices = self.get_central_indices(n_neighbors=n_neighbors)
            parameters.filter_keys(keep=indices)
        # atom_mapping = self.get_index_to_monomer_atom_mapping()
        # parameters.remap_keys(mapping=atom_mapping)
        parameters.map_indices_to_graph(self.graph.graph)
        return parameters

    def monomer_atoms_to_smarts(
            self, atom_graph: nx.Graph,
            context: Literal["minimal", "central", "residue", "full"] = "full",
            include_caps: bool = True,
            enumerate_all: bool = True,
            only_central: bool = False,
    ):
        # smarts = []
        matcher = self.graph.create_graphmatcher(atom_graph,
                                                 match_monomer_atoms=True)
        for mapping in matcher.subgraph_isomorphisms_iter():
            # self to other
            subgraph = sorted(mapping.items(), key=lambda x: x[1]["index"])
            nodes = [atom[0] for atom in subgraph]
            if only_central and not any(self.graph.nodes[n].get("central")):
                continue
            sm = self.to_smarts(label_nodes=nodes,
                                context=context,
                                include_caps=include_caps)
            return sm  # I think it's basically equivalent for all smarts?

    def to_smarts(
            self,
            nodes: List[int] = [],
            label_nodes: List[int] = [],
            context: Literal["minimal", "central", "residue", "full"] = "full",
            include_caps: bool = True,
            **kwargs,
    ):
        nodes = set(nodes) + set(label_nodes)
        if context == "central":
            nodes |= self.graph.get_central_nodes(n_neighbors=0)
        elif context == "residue":
            monomer_ints = {self._atom_to_monomer_nodes[i] for i in nodes}
            nodes
        elif context == "full":
            nodes = self.graph.graph.nodes
        if include_caps:
            nodes |= self.graph.get_neighbor_caps(*nodes)
        subgraph = self.graph.subgraph(nodes, cap_broken_bonds=True)
        return subgraph.graph.to_smarts(label_nodes=label_nodes, **kwargs)

    def get_atom_node(self, atom):
        for node, data in self.graph.graph.nodes(data=True):
            if data.get("monomer_atom") == atom:
                return node
