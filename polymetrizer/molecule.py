from typing import Any, Optional, List, Dict, Tuple, Union
from typing_extensions import Literal
from collections import defaultdict

import networkx as nx

from . import base
from .graph import MolecularGraph


class Atom(base.Model):

    atomic_number: int
    atom_map_number: int
    formal_charge: int
    is_aromatic: bool
    monomer_name: str
    monomer_node: int
    cap: bool = False

    # def __repr__(self):
    #     return f"<Atom {hash(self)}>"

    def __eq__(self, other):
        return self.dict() == other


class BaseMolecule(base.Model):

    graph: MolecularGraph

    @classmethod
    def from_obj(cls, obj: Any, **kwargs):
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls.from_smiles(obj, **kwargs)
        elif isinstance(obj, Chem.Mol):
            return cls.from_rdkit(obj, **kwargs)
        raise ValueError(f"Cannot create {cls.__name__} from {obj}")

    @classmethod
    def from_smiles(cls, smiles, remove_hs: bool = False, **kwargs):
        graphcls = cls.__fields__["graph"].outer_type_
        graph = graphcls.from_smiles(smiles, remove_hs=remove_hs)
        return cls(graph=graph, **kwargs)

    @classmethod
    def from_rdkit(cls, rdmol, **kwargs):
        graphcls = cls.__fields__["graph"].outer_type_
        graph = graphcls.from_rdkit(rdmol)
        return cls(graph=graph, **kwargs)

    @property
    def graph_(self):
        return self.graph.graph_

    @property
    def monomer_atoms(self):
        return self.graph.monomer_atoms

    def to_openff(self):
        return self.graph.to_openff()

    def to_rdkit(self):
        return self.graph.to_rdkit()

    def to_smiles(self):
        return self.graph.to_smiles()

    def __len__(self):
        return len(self.graph)

    # def get_index_to_monomer_atom_mapping(self):
    #     return {i: atom
    #             for i, (node, atom) in enumerate(self.graph.nodes("monomer_atom"))
    #             if atom is not None}

    def _is_fully_isomorphic(self, other):
        return self.graph.is_isomorphic(other.graph,
                                        match_isotope=True,
                                        match_formal_charge=True,
                                        match_is_aromatic=True,
                                        match_bond_order=True,
                                        match_monomer_name=True,
                                        match_monomer_atoms=True,
                                        )

    def to_smarts(
            self,
            nodes: List[int] = [],
            label_nodes: List[int] = [],
            return_monomer_id=False,
            context=None,
            **kwargs,
    ):
        nodes = set(nodes) | set(label_nodes)
        subgraph = self.graph.subgraph(nodes, cap_broken_bonds=True)
        sm = subgraph.to_smarts(label_nodes=label_nodes, **kwargs)
        if return_monomer_id:
            return sm, self.nodes_to_monomer_id(nodes)
        return sm

    def nodes_to_monomer_id(self, nodes):
        raise NotImplementedError

    def iter_r_group_numbers(self):
        for _, data in self.graph_.nodes(data=True):
            if data["atomic_number"] == 0:
                yield data["atom_map_number"]

    def get_atom_node(self, atom):
        for node, data in self.graph_.nodes(data=True):
            if data.get("monomer_atom") == atom:
                return node

    def visualize(self, backend="rdkit"):
        return self.graph.to_openff().visualize(backend=backend)

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



class Unit(BaseMolecule):

    name: Optional[str] = ""

    def __post_init__(self):
        self.graph.set_node_attr(monomer_name=self.name)
        self.graph.set_node_attr(central=True)
        for node in self.graph_:
            hashable = Atom(monomer_node=node, **self.graph_.nodes[node])
            self.graph_.nodes[node]["monomer_atom"] = hashable

    def nodes_to_monomer_id(self, nodes):
        return self.name
