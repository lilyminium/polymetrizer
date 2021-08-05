import itertools
from collections import defaultdict
from typing import Dict, Union, List, Optional, Set, Tuple
from typing_extensions import Literal

import networkx as nx
import numpy as np
from pydantic import Field, PrivateAttr

from . import base, utils, ommutils
from .molecule import BaseMolecule
from .cap import Cap, HYDROGEN_CAP
from .parameters import ForceFieldParameterSets


class Oligomer(BaseMolecule):

    _constituent_monomers: Dict[str, List[int]] = PrivateAttr(default_factory=lambda: defaultdict(list))
    _monomer_to_atom_nodes: Dict[int, Set[int]] = PrivateAttr(default_factory=dict)
    _atom_to_monomer_nodes: Dict[int, int] = PrivateAttr(default_factory=dict)
    _monomer_graph: nx.Graph = PrivateAttr(default_factory=nx.Graph)

    def _monomer_keyfunc(self):
        return (sorted(self._constituent_monomers),
                len(self._monomer_to_atom_nodes),
                len(self.graph_.nodes))

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
                             )
        return self

    def _record_monomer(
            self, monomer,
            new_atom_nodes=set(),
            new_atom_edge: Optional[Tuple[int, int]] = None,
    ):
        """Do some atom/monomer accounting for each new addition.
        This primarily builds a monomer-monomer graph for easy CG-searching
        """
        new_monomer_node = len(self._monomer_graph)
        self._monomer_to_atom_nodes[new_monomer_node] = new_atom_nodes
        for node in new_atom_nodes:
            self._atom_to_monomer_nodes[node] = new_monomer_node
        self._constituent_monomers[monomer.name].append(new_monomer_node)
        # add node
        self._monomer_graph.add_node(new_monomer_node, monomer_name=monomer.name)
        # add edge
        if new_atom_edge:
            old_atom_node, new_atom_node = new_atom_edge
            if old_atom_node in new_atom_nodes:
                new_atom_node, old_atom_node = new_atom_edge

            old_monomer_node = self._atom_to_monomer_nodes[old_atom_node]
            # remove from old node
            to_remove = {i for i in self._monomer_to_atom_nodes[old_monomer_node] if i not in self.graph_.nodes}
            for i in to_remove:
                self._atom_to_monomer_nodes.pop(i)
                self._monomer_to_atom_nodes[old_monomer_node].remove(i)
            edge = (old_monomer_node, new_monomer_node)
            atom_edge = (old_atom_node, new_atom_node)
            names = (self._monomer_graph.nodes[old_monomer_node]["monomer_name"],
                     self._monomer_graph.nodes[new_monomer_node]["monomer_name"],)
            monomer_atoms = (self.graph_.nodes[old_atom_node]["monomer_atom"],
                             self.graph_.nodes[new_atom_node]["monomer_atom"])
            self._monomer_graph.add_edge(*edge,
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
        r_group_numbers = list(self.iter_r_group_numbers())
        sub_choices = [caps.get(r, []) for r in r_group_numbers]
        combinations = [
            [dict(other=other, r_self=r, r_other=r_other)
             for r, (r_other, other) in zip(r_group_numbers, combination)]
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
        substituted = [self.with_substitutions(group) for group in groups]
        return substituted

    def enumerate_substituted_products(
            self,
            substituents: List["Monomer"],
            caps: List["Cap"],
            linkage_graph: nx.Graph,
            n_substitutions: int = 1,
    ):
        products = [self]
        for i in range(n_substitutions):
            products = [p for x in products for p in
                        x.generate_substituted(substituents, linkage_graph)]
        for compound in products:
            compound.cap_remaining(caps=caps, linkage_graph=linkage_graph)
        return products

    def cap_remaining(self, caps: List[Cap] = [HYDROGEN_CAP], linkage_graph=None,
                      inplace: bool = True):
        if not inplace:
            obj = self.copy(deep=True)
            return obj.cap_remaining(caps=caps, linkage_graph=linkage_graph, inplace=True)
        for cap in caps:
            r_groups = cap.get_compatible_rs(self, linkage_graph=linkage_graph)
            self._cap_remaining(cap, r_groups)
        return self

    def _cap_remaining(self, cap, r_groups):
        for r in list(self.iter_r_group_numbers()):
            if r in r_groups:
                self._substitute(cap, r, None)
        return self

    def get_central_indices(
            self,
            n_neighbors: int = 0,
    ) -> List[int]:
        nodes = list(self.graph.get_central_nodes(n_neighbors=n_neighbors))
        return np.where(np.isin(list(self.graph_.nodes), nodes))[0]

    def get_index_to_monomer_atom_mapping(self):
        return {i: atom
                for i, (node, atom) in enumerate(self.graph.nodes("monomer_atom"))
                if atom is not None}

    def to_openff_parameterset(
            self,
            forcefield,
            n_neighbors: int = -1,
            **kwargs,
    ):
        offmol = self.graph.to_openff()
        parameter_set = ForceFieldParameterSets.from_openff_molecule(offmol,
                                                                     forcefield,
                                                                     **kwargs)
        if n_neighbors > 0:
            indices = self.get_central_indices(n_neighbors=n_neighbors)
            parameter_set.filter_keys(keep=indices)
        parameter_set.map_indices_to_graph(self.graph)
        return parameter_set

    def to_smarts(
            self,
            nodes: List[int] = [],
            label_nodes: List[int] = [],
            context: Literal["minimal", "central", "residue", "full"] = "full",
            include_caps: bool = False,
            return_monomer_id=False,
            **kwargs,
    ):
        nodes = set(nodes) | set(label_nodes)
        if context == "central":
            central = self.graph.get_central_nodes(n_neighbors=0)
            # TODO: cheap way to get all nodes in between
            nodes |= central
        elif context == "residue":
            monomer_ints = {self._atom_to_monomer_nodes[i] for i in nodes}
            for monomer in monomer_ints:
                nodes |= self._monomer_to_atom_nodes[monomer]
        elif context == "full":
            nodes = self.graph_.nodes
        nodes = {i for i in nodes if i in self.graph_}
        if include_caps:
            nodes |= self.graph.get_neighbor_caps(*nodes)
        else:
            nodes = {i for i in nodes if not self.graph_.nodes[i].get("cap")}
            nodes |= set(label_nodes)
        subgraph = self.graph.subgraph(nodes, cap_broken_bonds=True)

        sm = subgraph.to_smarts(label_nodes=label_nodes, **kwargs)
        if return_monomer_id:
            return sm, self.nodes_to_monomer_id(nodes)
        return sm

    def get_atom_node(self, atom):
        for node, data in self.graph_.nodes(data=True):
            if data.get("monomer_atom") == atom:
                return node

    def iter_r_group_numbers(self):
        for _, data in self.graph_.nodes(data=True):
            if data["atomic_number"] == 0:
                yield data["atom_map_number"]

    def visualize(self, backend="rdkit"):
        return self.graph.to_openff().visualize(backend=backend)

    def nodes_to_monomer_id(self, nodes):
        names = [self.graph_.nodes[n]["monomer_name"] for n in nodes]
        monomer_nodes = [self._atom_to_monomer_nodes[n] for n in nodes]
        _, index = np.unique(monomer_nodes, return_index=True)
        final_names = sorted([names[i].capitalize() for i in index])
        return "".join(final_names)
