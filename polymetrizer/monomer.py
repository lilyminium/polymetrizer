from collections import defaultdict
from typing import Dict, Optional, Union, List

import networkx as nx
from pydantic import Field, validator

from . import base, utils, rdutils
from .oligomer import Oligomer, Cap, HYDROGEN
from .graph import BaseMolecule


class Atom(base.Model):

    atomic_number: int
    isotope: int
    formal_charge: int
    is_aromatic: bool
    monomer_name: str
    monomer_node: int


class Monomer(Oligomer):

    name: Optional[str] = None

    def __post_init__(self):
        self.graph.set_node_attr(monomer_name=self.name, central=True)
        for node in self.graph.graph:
            hashable = Atom(monomer_node=node, **self.graph.graph.nodes[node])
            self.graph.graph.nodes[node]["monomer_atom"] = hashable

    def substitute(self, other: "Monomer",
                   r_self: int, r_other: int):
        obj = self.to_oligomer()
        return obj._substitute(other, r_self, r_other)

    def cap_remaining(
            self,
            cap: Cap = HYDROGEN,
            r_groups: List[int] = [],
    ):
        obj = self.to_oligomer()
        return obj._cap_remaining(cap, r_groups)

    def to_oligomer(self):
        oligomer = Oligomer(graph=self.graph.copy(deep=True))
        starting_nodes = set(oligomer.graph.graph.nodes)
        oligomer._record_monomer(self, new_atom_nodes=starting_nodes)
        return oligomer

# @classmethod
# def from_smiles(cls, smiles: str, **kwargs):
#     new = cls(**kwargs)
#     new.graph = rdutils.smiles_to_nxgraph(smiles)
#     return new

# def __post_init__(self):
#     self._graph = nx.Graph()
#     self._r_groups = defaultdict(list)
#     self._r_group_neighbors = defaultdict(list)

# def is_isomorphic(self, other):

#     def node_match(a, b):
#         if type(a)

# @property
# def graph(self):
#     return self._graph

# @graph.setter
# def graph(self, value):
#     self._graph = value
#     self._update_from_graph()
#     self.set_node_attr(central=True)

# def set_node_attr(self, **kwargs):
#     for node in self._graph.nodes:
#         for k, v in kwargs.items():
#             self._graph.graph[node][k] = v

# def _update_from_graph(self):
#     for k in list(self._r_groups.keys()):
#         self._r_groups.pop(k)
#     for node in self.graph.nodes():
#         if isinstance(node, RGroup):
#             edges = self.graph.edges(node)
#             n_edges = len(edges)
#             if n_edges != 1:
#                 err = ("RGroup must be bonded to one atom only. "
#                        f"Found: {n_edges} bonds")
#                 raise ValueError(err)
#             # neighbor = utils.get_other_in_pair(node, edges)
#             self._r_groups[node.r].append(node)

# def contains_r(self, r: RGroupLike):
#     if isinstance(r, int):
#         return r in self._r_groups
#     return r in self._r_groups[r.r]

# def get_r_group(self, r: RGroupLike):
#     if isinstance(r, RGroup):
#         if self.contains(r):
#             return r
#         r = r.r

#     try:
#         r_values = self._r_groups[r]
#         assert len(r_values)
#     except (KeyError, AssertionError):
#         raise ValueError(f"RGroup {r} not found in {self}")
#     else:
#         return next(iter(r_values))

# def get_r_group_index(self, r: RGroup):
#     return self._r_groups[r.r].index(r)

# def get_r_group_from_index(self, r: RGroupLike, index: int = 0):
#     if isinstance(r, RGroup):
#         r = r.r
#     return self._r_groups[r][index]

# def get_neighbors(self, node):
#     for edge in self.graph.edges(node):
#         yield utils.get_other_in_pair(node, edge)

# def connect(self, other, r, r_other):
#     self_r = self.get_r_group(r)
#     other_r = other.get_r_group(r_other)

#     self_atom = next(self.get_neighbors(self_r))
#     other_atom = next(other.get_neighbors(other_r))

# def iter_r_groups(self):
#     for node in self._graph.nodes():
#         if isinstance(node, RGroup):
#             yield node

# def iter_r_group_numbers(self):
#     for node in self.iter_r_groups():
#         yield node.r
