from typing import Optional, Tuple, List

import networkx as nx
from pydantic import Field

from .graph import CapGraph
from .molecule import Unit


class Cap(Unit):

    graph: CapGraph = Field(default_factory=CapGraph)
    compatible_rs: Optional[Tuple[int, ...]] = tuple()
    compatible_smiles: Optional[Tuple[str, ...]] = tuple()
    r: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.r is None:
            self.r = self.graph_.nodes[self.graph._node].get("atom_map_number")

    def get_compatible_rs(self, *oligomers,
                          linkage_graph: Optional[nx.Graph] = None,
                          ):
        if linkage_graph is not None and self.r is not None:
            rs = list(linkage_graph.neighbors(self.r))
            return rs
        if self.compatible_rs:
            return self.compatible_rs
        if self.compatible_smiles:
            r = set()
            for o in oligomers:
                for smi in self.compatible_smiles:
                    r |= o.graph.get_r_by_smiles(smi)
            return r
        return list({r for o in oligomers for r in o.iter_r_group_numbers()})


HYDROGEN_CAP = Cap.from_smiles("[R][H]", name="H")

ACE_CAP = Cap.from_smiles("CC(=O)-[R]", name="Ace",
                          compatible_smiles=["[R]N([H])C([*])C(=O)-[*]"])
NME_CAP = Cap.from_smiles("[R]NC", name="Nme",
                          compatible_smiles=["[*]N([H])C([*])C(=O)-[R]"])
