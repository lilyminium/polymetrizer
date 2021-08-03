from typing import List, Set, Optional
import itertools

from rdkit import Chem
import networkx as nx
from openff import toolkit as offtk
from pydantic import Field, validator, PrivateAttr
from networkx.algorithms import isomorphism as nxiso

from . import base, utils, rdutils

HYDROGEN_ATOM = dict(
    atomic_number=1,
    isotope=0,
    formal_charge=0,
    is_aromatic=False,
)


class MolecularGraph(base.Model):

    graph: nx.Graph

    @classmethod
    def from_smiles(cls, smiles: str):
        smiles = utils.replace_R_with_dummy(smiles)
        print("smi", smiles)
        rdmol = Chem.MolFromSmiles(smiles)
        print("rdmol", rdmol)
        return cls.from_rdkit(rdmol)

    @classmethod
    def from_rdkit(cls, rdmol):
        graph = rdutils.rdmol_to_nxgraph(rdmol)
        return cls(graph=graph)

    def get_max_node(self):
        return max(self.graph.nodes())

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
                is_match &= compare("isotope")
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

        matcher = nxiso.GraphMatcher(self.graph, other.graph,
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

    def set_node_attr(self, **kwargs):
        for node in self.graph.nodes:
            for k, v in kwargs.items():
                self.graph.nodes[node][k] = v

    def iter_r_groups(self):
        for node, atomic_number in self.graph.nodes(data="atomic_number"):
            if atomic_number == 0:
                yield node

    def iter_r_group_numbers(self):
        for _, data in self.graph.nodes(data=True):
            if data["atomic_number"] == 0:
                yield data["isotope"]

    def get_r_node(self, r: int) -> int:
        for node, data in sorted(self.graph.nodes(data=True)):
            if data["atomic_number"] == 0 and data["isotope"] == r:
                return node
        if self.graph.nodes[r]["atomic_number"] == 0:
            return r
        raise ValueError(f"R group {r} not found in molecule.")

    def add_with_r(self, other: "MolecularGraph", r_self: int, r_other: int):
        other_node = other.get_r_node(other)
        self_node = self.get_r_node(r_self)
        return self.add(other, self_node, other_node)

    def add(self, other: "MolecularGraph", node_self: int, node_other: int):
        increment = self.get_max_node()
        other_node = node_other + increment
        other = other.relabel_nodes(increment=increment, copy=True)
        other.set_node_attr(central=False)

        self_neighbors = list(self.graph.neighbors(node_self))
        other_neighbors = list(other.graph.neighbors(other_node))
        self.graph.remove_node(node_self)
        other.graph.remove_node(other_node)

        old_nodes = set(self.graph)
        self.graph.add_edges_from(other.graph.edges(data=True))
        for a, b in itertools.product(self_neighbors, other_neighbors):
            self.graph.add_edge(a, b)
        new_nodes = set(self.graph)
        return new_nodes - old_nodes, (a, b)

    def relabel_nodes(self, increment: int = 0, copy: bool = True):
        mapping = {node: node + increment for node in self.graph}
        return nx.relabel_nodes(self.graph, mapping, copy=copy)

    def to_rdkit(self, mapped: bool = True):
        return rdutils.nxgraph_to_rdmol(self.graph, mapped=mapped)

    def to_mdanalysis(self):
        import MDAnalysis as mda
        rdmol = self.to_rdkit()
        return mda.Universe(rdmol)

    def to_openff(self):
        rdmol = self.to_rdkit(mapped=True)
        return offtk.topology.Molecule.from_rdkit(rdmol)

    def to_smiles(self, mapped: bool = True):
        return Chem.MolToSmiles(self.to_rdkit(mapped=mapped))

    def to_smarts(self, label_nodes=[], **kwargs):
        return rdutils.nxgraph_to_smarts(self.graph,
                                         label_nodes=label_nodes,
                                         **kwargs)

    def subgraph(self, nodes=[], cap_broken_bonds: bool = False):
        graph = type(self)(graph=self.graph.subgraph(nodes).copy())
        if cap_broken_bonds:
            dummies_to_add = {}
            next_node = self.get_max_node() + 1
            for node in graph:
                old_neighbors = len(self.graph[node])
                new_neighbors = len(graph[node])
                dummies_to_add[node] = old_neighbors - new_neighbors
            for node, n_dummies in dummies_to_add.items():
                for i in range(n_hs):
                    graph.add_edge((node, next_node))
                    next_node += 1
        return graph

    def neighbors(self, *nodes, **kwargs):
        return {m for n in nodes for m, data in self.graph[n].items()
                if all(data.get(k) == v for k, v in kwargs.items())}

    def get_neighbor_caps(self, *nodes):
        caps = self.neighbors(*nodes, cap=True)
        cap_neighbors = self.neighbors(*caps, cap=True)
        # traverse to end. There's probably a better way...
        while len(caps + cap_neighbors) > len(caps):
            caps += cap_neighbors
            cap_neighbors = self.neighbors(*caps, cap=True)
        return caps

    def get_central_nodes(
            self,
            n_neighbors: int = 0,
    ) -> Set[int]:
        nodes = {k for k, v in self.graph.nodes("central") if v}
        layer = nodes
        for i in range(n_neighbors):
            layer = self.neighbors(layer) - layer
            nodes |= layer
        return nodes

    def get_index_to_node_mapping(self):
        return {i: node for i, node in enumerate(self.graph.nodes)}

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges


class CapGraph(MolecularGraph):

    _node: int = PrivateAttr(default=0)
    _r: Optional[int] = None

    def __post_init__(self):
        for node, data in sorted(self.graph.nodes(data=True)):
            if data["atomic_number"] == 0:
                self._node = node
        self.set_node_attr(cap=True)

    @validator("graph")
    def validate_graph(cls, v):
        n_r = sum(1 for _, n in v.nodes(data="atomic_number") if n == 0)
        assert n_r == 1
        return v

    def get_r_node(self, *args) -> int:
        return self._node


class BaseMolecule(base.Model):

    graph: MolecularGraph

    @classmethod
    def from_smiles(cls, smiles, **kwargs):
        graphcls = cls.__fields__["graph"].outer_type_

        graph = graphcls.from_smiles(smiles)
        return cls(graph=graph, **kwargs)

    @classmethod
    def from_rdkit(cls, rdmol, **kwargs):
        graphcls = cls.__fields__["graph"].outer_type_
        graph = graphcls.from_rdkit(rdmol)
        return cls(graph=graph, **kwargs)
