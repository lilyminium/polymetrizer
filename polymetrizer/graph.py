from typing import List, Set, Optional, Any
import itertools
import functools
import warnings
from collections import defaultdict

from rdkit import Chem
import networkx as nx
import numpy as np
from pydantic import Field, validator, PrivateAttr
from networkx.algorithms import isomorphism as nxiso

from . import base, utils, rdutils
from .utils import cached_property, uncache_properties
from .smirks import BeSmirker


class HashableGraph(base.Model):

    graph_: nx.Graph = Field(alias="graph")

    def _get_comparison_fields(self):
        # networkx graphs seem to hash to memory, i.e.
        # hash(graph) != hash(graph.copy())
        nodes = tuple(self.graph_)
        if nodes[0] > nodes[-1]:
            nodes = nodes[::-1]
        edgeset = frozenset(frozenset(edge) for edge in self.graph_.edges)
        return (nodes, edgeset)

    def __repr__(self):
        clsname = type(self).__name__
        n_nodes = len(self.graph_)
        n_edges = len(self.graph_.edges)
        return f"<{clsname} with {n_nodes} nodes, {n_edges} edges>"

    def __getitem__(self, key):
        return tuple(self.graph_).__getitem__(key)

    def __len__(self):
        return len(self.graph_)

    @property
    def nodes(self):
        return self.graph_.nodes

    @property
    def edges(self):
        return self.graph_.edges

    def get_max_node(self):
        return max(self.graph_.nodes())

    def set_node_attr(self, **kwargs):
        for node in self.graph_.nodes:
            for k, v in kwargs.items():
                self.graph_.nodes[node][k] = v

    def get_nodes(self, n_neighbors: int = 0, **kwargs):
        nodes = {n for n, data in self.graph_.nodes(data=True)
                 if all(data.get(k) == v for k, v in kwargs.items())}
        layer = nodes.copy()
        for i in range(n_neighbors):
            layer = self.neighbors(*layer, **kwargs)
            nodes |= layer
        return nodes

    def get_node_neighbors(self, nodes, n_neighbors: int = 0):
        nodes = set(nodes)
        layer = nodes.copy()
        for i in range(n_neighbors):
            layer = self.neighbors(*layer)
            nodes |= layer
        return nodes

    def neighbors(self, *nodes, **kwargs):
        return {m for n in nodes for m, data in self.graph_[n].items()
                if all(data.get(k) == v for k, v in kwargs.items())}


class AtomGraph(HashableGraph):

    def _get_comparison_fields(self):
        # networkx graphs seem to hash to memory, i.e.
        # hash(graph) != hash(graph.copy())
        nodes = tuple(self.monomer_atoms)
        nodeset = frozenset([nodes, nodes[::-1]])
        edgeset = set()
        for i, j in self.graph_.edges:
            edgeset.add(frozenset({self.graph_.nodes[i]["monomer_atom"],
                                   self.graph_.nodes[j]["monomer_atom"]}))
        return (nodeset, frozenset(edgeset))

    @classmethod
    def from_node_graph(cls, graph):
        return cls(graph=graph.copy())

    # @cached_property
    @property
    def monomer_names(self):
        return [atom.monomer_name for atom in self.monomer_atoms]

    @property
    def nodes(self):
        nodes = sorted(self.graph_.nodes("index"), key=lambda x: x[1])
        return [x[0] for x in nodes]

    # @cached_property
    @property
    def monomer_atoms(self):
        return [self.graph_.nodes[node]["monomer_atom"] for node in self.nodes]

    @property
    def monomer_nodes(self):
        return [a.monomer_node for a in self.monomer_atoms]

    # @cached_property

    @property
    def atomic_numbers(self):
        return [self.graph_.nodes[node]["atomic_number"] for node in self.nodes]

    @property
    def indices(self):
        return [self.graph_.nodes[node]["index"] for node in self.nodes]

    def contains_cap(self):
        return any([x.cap for x in self.monomer_atoms])


class MolecularGraph(HashableGraph):

    @classmethod
    def from_smiles(cls, smiles: str, remove_hs: bool = False):
        smiles = utils.replace_R_with_dummy(smiles)
        smiles_parser = Chem.rdmolfiles.SmilesParserParams()
        smiles_parser.removeHs = remove_hs
        rdmol = Chem.MolFromSmiles(smiles, smiles_parser)
        return cls.from_rdkit(rdmol)

    @classmethod
    def from_rdkit(cls, rdmol):
        graph = rdutils.rdmol_to_nxgraph(rdmol)
        return cls(graph=graph)

    def create_graphmatcher(
            self, other,
            match_isotope=True,
            match_formal_charge=True,
            match_is_aromatic=True,
            match_bond_order=True,
            match_monomer_name=False,
            match_monomer_atoms=False,
    ):
        def node_match(a, b):
            def compare(x): return a[x] == b[x]
            is_match = compare("atomic_number")
            if match_isotope:
                is_match &= compare("atom_map_number")
            if match_formal_charge:
                is_match &= compare("formal_charge")
            if match_is_aromatic:
                is_match &= compare("is_aromatic")
            if match_monomer_name:
                is_match &= compare("monomer_name")
            if match_monomer_atoms:
                is_match &= compare("monomer_atom")
            return is_match

        if match_is_aromatic or match_bond_order:
            def edge_match(a, b):
                aromatic_match = a["is_aromatic"] == b["is_aromatic"]
                bond_order_match = a["order"] == b["order"]
                if match_is_aromatic:
                    if match_bond_order:
                        return aromatic_match or bond_order_match
                    return aromatic_match
                return bond_order_match
        else:
            edge_match = None

        if isinstance(other, HashableGraph):
            other = other.graph_

        matcher = nxiso.GraphMatcher(self.graph_, other,
                                     node_match=node_match,
                                     edge_match=edge_match)
        return matcher

    def is_isomorphic(self, other, match_isotope=True,
                      match_formal_charge=True,
                      match_is_aromatic=True,
                      match_bond_order=True,
                      match_monomer_name=False,
                      match_monomer_atoms=False,
                      ):
        matcher = self.create_graphmatcher(other,
                                           match_isotope=match_isotope,
                                           match_formal_charge=match_formal_charge,
                                           match_is_aromatic=match_is_aromatic,
                                           match_bond_order=match_bond_order,
                                           match_monomer_name=match_monomer_name,
                                           match_monomer_atoms=match_monomer_atoms)
        return matcher.is_isomorphic()

    def iter_r_groups(self):
        for node, atomic_number in self.graph_.nodes(data="atomic_number"):
            if atomic_number == 0:
                yield node

    def get_r_node(self, r: int) -> int:
        for node, data in sorted(self.graph_.nodes(data=True)):
            if data["atomic_number"] == 0 and data["atom_map_number"] == r:
                return node
        if self.graph_.nodes[r]["atomic_number"] == 0:
            return r
        raise ValueError(f"R group {r} not found in molecule.")

    def get_r_by_smiles(self, smiles):
        rdmol = self.to_rdkit()
        return rdutils.get_r_from_smiles(rdmol, smiles)

    def add_with_r(self, other: "MolecularGraph", r_self: int, r_other: int):
        other_node = other.get_r_node(r_other)
        self_node = self.get_r_node(r_self)
        return self.add(other, self_node, other_node)

    @uncache_properties("index_to_node_mapping",
                        "node_to_index_mapping",
                        "monomer_atoms")
    def add(self, other: "MolecularGraph", node_self: int, node_other: int):
        increment = self.get_max_node()
        other_node = node_other + increment
        other = other.relabel_nodes(increment=increment, copy=True)
        other.set_node_attr(central=False)

        self_neighbors = self.graph_[node_self].items()
        other_neighbors = other.graph_[other_node].items()
        self.graph_.remove_node(node_self)
        other.graph_.remove_node(other_node)

        old_nodes = set(self.graph_)
        self.graph_.add_nodes_from(other.graph_.nodes(data=True))
        self.graph_.add_edges_from(other.graph_.edges(data=True))
        for pair in itertools.product(self_neighbors, other_neighbors):
            (a, a_data), (b, b_data) = pair
            if a_data != b_data:
                warnings.warn(f"Bonds not compatible: {a_data} vs {b_data}")
                # TODO: should this error or warn?
                # raise ValueError(f"Bonds not compatible: {a_data} vs {b_data}")
            self.graph_.add_edge(a, b, **a_data)
        new_nodes = set(self.graph_)
        return new_nodes - old_nodes, (a, b)

    def relabel_nodes(self, increment: int = 0, copy: bool = True):
        if copy:
            obj = self.copy(deep=True)
            return obj.relabel_nodes(increment=increment, copy=False)
        mapping = {node: node + increment for node in self.graph_}
        nx.relabel_nodes(self.graph_, mapping, copy=False)
        return self

    def to_rdkit(self, mapped: bool = True):
        return rdutils.nxgraph_to_rdmol(self.graph_, mapped=mapped)

    def to_mdanalysis(self):
        import MDAnalysis as mda
        rdmol = self.to_rdkit()
        return mda.Universe(rdmol)

    def to_openff(self):
        from openff.toolkit.topology import Molecule
        rdmol = self.to_rdkit(mapped=True)
        return Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

    def to_smiles(self, mapped: bool = True):
        return Chem.MolToSmiles(self.to_rdkit(mapped=mapped))

    def to_smarts(self, label_nodes=[], **kwargs):
        smirker = BeSmirker(**kwargs)
        return smirker(self, label_atom_numbers=label_nodes)

    def subgraph(self, nodes=[], cap_broken_bonds: bool = True):
        sub = self.graph_.subgraph(nodes).copy()
        DUMMY = dict(atomic_number=0, atom_map_number=0)
        if cap_broken_bonds:
            next_node = self.get_max_node() + 1
            for node in list(sub):
                new_neighbors = sub[node]
                for old_neighbor, edge_data in self.graph_[node].items():
                    if old_neighbor not in new_neighbors:
                        kws = dict(**self.graph_.nodes[old_neighbor])
                        kws["atomic_number"] = 0
                        kws["atom_map_number"] = 0
                        sub.add_node(next_node, **kws)
                        sub.add_edge(node, next_node, **edge_data)
                        next_node += 1
        graph = type(self)(graph=sub)
        return graph

    def get_neighbor_caps(self, *nodes):
        caps = self.neighbors(*nodes, cap=True)
        cap_neighbors = self.neighbors(*caps, cap=True)
        # traverse to end. There's probably a better way...
        while len(caps | cap_neighbors) > len(caps):
            caps += cap_neighbors
            cap_neighbors = self.neighbors(*caps, cap=True)
        return caps

    @cached_property
    def index_to_node_mapping(self):
        return {i: node for i, node in enumerate(self.graph_.nodes)}

    @cached_property
    def node_to_index_mapping(self):
        return {node: i for i, node in enumerate(self.graph_.nodes)}

    @cached_property
    def monomer_atoms(self):
        return [self.graph_.nodes[node]["monomer_atom"] for node in self.graph_]

    def atom_subgraph_by_indices(self, indices: List[int]):
        nodes = [self.index_to_node_mapping[i] for i in indices]
        atomgraph = AtomGraph.from_node_graph(self.graph_.subgraph(nodes))
        for i, node in enumerate(nodes):
            # atom = self.graph_.nodes[node]["monomer_atom"]
            atomgraph.graph_.nodes[node]["index"] = i
        return atomgraph

    def iter_subgraph_atom_isomorphisms(self, atom_graph: AtomGraph,
                                        match_is_aromatic=True,
                                        match_bond_order=True):
        def node_match(a, b):
            match = a["monomer_atom"] == b["monomer_atom"]
            return match

        if match_is_aromatic or match_bond_order:
            def edge_match(a, b):
                aromatic_match = a["is_aromatic"] == b["is_aromatic"]
                bond_order_match = a["order"] == b["order"]
                if match_is_aromatic:
                    if match_bond_order:
                        return aromatic_match or bond_order_match
                    return aromatic_match
                return bond_order_match
        else:
            edge_match = None

        if isinstance(atom_graph, HashableGraph):
            atom_graph = atom_graph.graph_

        matcher = nxiso.GraphMatcher(self.graph_, atom_graph,
                                     node_match=node_match,
                                     edge_match=edge_match)
        for mapping in matcher.subgraph_isomorphisms_iter():
            yield mapping

    def iter_isomorphic_atom_nodes(self, atom_graph: AtomGraph,
                                   match_is_aromatic=True,
                                   match_bond_order=True):
        for mapping in self.iter_subgraph_atom_isomorphisms(atom_graph,
                                                            match_is_aromatic,
                                                            match_bond_order):
            self_to_other = list(mapping.items())
            self_to_other.sort(key=lambda x: atom_graph.graph_.nodes[x[1]]["index"])
            nodes = [x[0] for x in self_to_other]
            yield nodes

    def get_central_nodes(
            self,
            n_neighbors: int = 0,
            exclude_dummy_atoms: bool = True,
    ) -> Set[int]:
        # TODO: much of this is redundant with get_nodes
        nodes = self.get_nodes(central=True)
        if exclude_dummy_atoms:
            nodes = {k for k in nodes if self.graph_.nodes[k]["atomic_number"]}
        layer = nodes.copy()
        for i in range(n_neighbors):
            layer = self.neighbors(*layer) - layer
            if exclude_dummy_atoms:
                layer = {k for k in layer if self.graph_.nodes[k]["atomic_number"]}
            nodes |= layer
        return nodes


class CapGraph(MolecularGraph):

    _node: int = PrivateAttr(default=0)
    _r: Optional[int] = None

    def __post_init__(self):
        for node, data in sorted(self.graph_.nodes(data=True)):
            if data["atomic_number"] == 0:
                self._node = node
        self.set_node_attr(cap=True)

    @validator("graph_")
    def validate_graph(cls, v):
        n_r = sum(1 for _, n in v.nodes(data="atomic_number") if n == 0)
        assert n_r == 1
        return v

    def get_r_node(self, *args) -> int:
        return self._node
