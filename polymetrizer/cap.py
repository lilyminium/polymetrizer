from typing import Optional, Tuple, List

import networkx as nx
from pydantic import Field

from .graph import CapGraph
from .molecule import Unit


class Cap(Unit):

    graph: CapGraph = Field(default_factory=CapGraph)
    compatible_rs: Optional[Tuple[int, ...]] = tuple()
    r: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.r is None:
            self.r = self.graph_[self.graph._node].get("atom_map_number")

    def get_compatible_rs(self, *oligomers,
                          linkage_graph: Optional[nx.Graph] = None,
                          ):
        if self.compatible_rs:
            return self.compatible_rs
        if linkage_graph is not None and self.r is not None:
            return list(linkage_graph.neighbors(self.r))
        return list({r for o in oligomers for r in o.iter_r_group_numbers()})


HYDROGEN_CAP = Cap.from_smiles("[R][H]", name="H")
