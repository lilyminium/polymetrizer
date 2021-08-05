from typing import Any, Optional, List
from typing_extensions import Literal

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


class Unit(BaseMolecule):

    name: Optional[str] = ""

    def __post_init__(self):
        self.graph.set_node_attr(monomer_name=self.name)
        for node in self.graph_:
            hashable = Atom(monomer_node=node, **self.graph_.nodes[node])
            self.graph_.nodes[node]["monomer_atom"] = hashable
