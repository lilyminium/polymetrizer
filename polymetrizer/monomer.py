from collections import defaultdict
from typing import Dict, Optional, Union, List

import networkx as nx
from pydantic import Field, validator

from . import base, utils, rdutils
from .molecule import Unit
from .oligomer import Oligomer
from .cap import Cap, HYDROGEN_CAP


class Monomer(Unit, Oligomer):

    def __post_init__(self):
        if not self.name:
            self.name = "Unk"
        super().__post_init__()
        self._record_monomer(self, new_atom_nodes=self.graph.nodes)

    def substitute(self, other: "Monomer",
                   r_self: int, r_other: int):
        obj = self.to_oligomer()
        obj._substitute(other, r_self, r_other)
        return obj

    def cap_remaining(self, caps: List[Cap] = [HYDROGEN_CAP], linkage_graph=None):
        obj = self.to_oligomer()
        return obj.cap_remaining(caps=caps, linkage_graph=linkage_graph, inplace=True)

    def to_oligomer(self):
        oligomer = Oligomer(graph=self.graph.copy(deep=True))
        starting_nodes = set(oligomer.graph_.nodes)
        oligomer._record_monomer(self, new_atom_nodes=starting_nodes)
        return oligomer

    def to_smarts(
            self,
            nodes: List[int] = [],
            label_nodes: List[int] = [],
            context="full",
            include_caps: bool = False,
            **kwargs,
    ):
        nodes = self.graph_.nodes
        subgraph = self.graph.subgraph(nodes, cap_broken_bonds=True)
        return subgraph.to_smarts(label_nodes=label_nodes, **kwargs)
